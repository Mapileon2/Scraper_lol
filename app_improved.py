"""
Improved Web Scraper Application with enhanced error handling, first-page analysis,
and better UI organization.
"""

# Set environment variables before any other imports to prevent conflicts
import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Import Streamlit initialization first to handle PyTorch compatibility
import streamlit_init  # noqa: F401

# Standard library imports
import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import http.client as http_client
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from urllib.parse import urlparse, urljoin
import re

# Third-party imports
import pandas as pd
import requests
import streamlit as st
from streamlit.components.v1 import html
from bs4 import BeautifulSoup
import pinecone
from sentence_transformers import SentenceTransformer
import hashlib
import google.generativeai as genai

# Initialize Streamlit configuration
st.set_page_config(
    page_title="Web Page Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from pydantic import BaseModel
from requests.exceptions import RequestException
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from webdriver_manager.chrome import ChromeDriverManager

# Constants for chunking and rate limiting
DESIRED_CHUNKS = 5  # Target number of chunks (adjustable)
MIN_CHUNK_SIZE = 50000  # Minimum characters per chunk
MAX_CHUNK_SIZE = 100000  # Maximum characters per chunk (to stay under 1M tokens)
REQUESTS_PER_MINUTE = 2  # Free tier limit
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # 30 seconds
TOTAL_REQUESTS_TODAY = 0
LAST_RESET = time.time()

# Cache for storing analysis results (TTL: 1 hour)
ANALYSIS_CACHE = {}
CACHE_TTL = 3600  # 1 hour in seconds

# Rate limiting settings for Gemini 1.5 Pro free tier
REQUESTS_PER_MINUTE = 2  # Gemini 1.5 Pro free tier limit (2 RPM)
REQUESTS_PER_DAY = 50    # Gemini 1.5 Pro free tier daily limit
MIN_DELAY_BETWEEN_REQUESTS = 60  # Increased minimum seconds between requests (for 1 RPM)
MAX_RETRIES = 5  # Maximum number of retries for rate limit errors
INITIAL_RETRY_DELAY = 60  # Initial delay in seconds
MAX_RETRY_DELAY = 300  # Maximum delay between retries in seconds

# File to persist daily request count
REQUEST_COUNT_FILE = 'daily_requests.json'

# Track request count for rate limiting
REQUEST_COUNT = 0
LAST_REQUEST_TIME = 0

# Initialize daily request tracking
def init_daily_requests():
    """Initialize or load daily request count from file."""
    today = str(date.today())
    try:
        if os.path.exists(REQUEST_COUNT_FILE):
            with open(REQUEST_COUNT_FILE, 'r') as f:
                data = json.load(f)
                if data.get('date') == today:
                    return data.get('count', 0), today
    except Exception as e:
        logger.error(f"Error loading daily requests: {e}")
    return 0, today

# Save daily request count
def save_daily_requests(count: int, request_date: str):
    """Save the daily request count to a file."""
    try:
        with open(REQUEST_COUNT_FILE, 'w') as f:
            json.dump({'date': request_date, 'count': count}, f)
    except Exception as e:
        logger.error(f"Error saving daily requests: {e}")

def split_html_into_chunks(html: str, desired_chunks: int = DESIRED_CHUNKS) -> List[str]:
    """Split HTML into chunks based on total characters / desired number of chunks."""
    total_chars = len(html)
    chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, total_chars // desired_chunks or 1))
    logger.info(f"Splitting HTML ({total_chars} chars) into chunks of ~{chunk_size} chars")
    return [html[i:i + chunk_size] for i in range(0, total_chars, chunk_size)]

# Initialize or load daily request count
DAILY_REQUESTS, LAST_REQUEST_DATE = init_daily_requests()

# Initialize rate limiting variables
LAST_REQUEST_TIME = 0
MIN_DELAY_BETWEEN_REQUESTS = 1.0  # Minimum delay between API requests in seconds

# Check if we've hit the daily limit
def check_daily_limit() -> bool:
    """Check if we've hit the daily request limit."""
    global DAILY_REQUESTS, LAST_REQUEST_DATE
    
    today = str(date.today())
    if today != LAST_REQUEST_DATE:
        # Reset counter for new day
        DAILY_REQUESTS = 0
        LAST_REQUEST_DATE = today
        save_daily_requests(DAILY_REQUESTS, today)
    
    if DAILY_REQUESTS >= REQUESTS_PER_DAY:
        logger.warning(f"Daily request limit of {REQUESTS_PER_DAY} reached")
        return True
    return False

# Update daily request counter
def update_daily_counter():
    """Increment the daily request counter and save it."""
    global DAILY_REQUESTS
    DAILY_REQUESTS += 1
    save_daily_requests(DAILY_REQUESTS, LAST_REQUEST_DATE)


def chunk_html(html_content: str, max_chars: int = 100000) -> List[Dict[str, Any]]:
    """
    Split HTML content into chunks of approximately max_chars size.
    Tries to split at tag boundaries when possible.
    
    Args:
        html_content: The HTML content to chunk
        max_chars: Maximum characters per chunk
        
    Returns:
        List of dictionaries with 'content' and 'metadata'
    """
    if not html_content or len(html_content) <= max_chars:
        return [{'content': html_content, 'metadata': {'chunk_num': 1, 'total_chunks': 1}}]
    
    try:
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements as they're not needed for analysis
        for element in soup(['script', 'style', 'noscript', 'link', 'meta']):
            element.decompose()
            
        # Get all top-level elements
        elements = list(soup.children)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for element in elements:
            element_str = str(element)
            element_size = len(element_str)
            
            # If element is too large, split it further
            if element_size > max_chars:
                # Try to split by paragraphs or divs if possible
                sub_elements = element.find_all(['p', 'div', 'section', 'article', 'main', 'header', 'footer'])
                if not sub_elements:
                    sub_elements = [element]
                    
                for sub_el in sub_elements:
                    sub_el_str = str(sub_el)
                    if current_size + len(sub_el_str) > max_chars and current_chunk:
                        # Save current chunk
                        chunk_html = ''.join(current_chunk)
                        chunks.append({
                            'content': chunk_html,
                            'metadata': {
                                'chunk_num': len(chunks) + 1,
                                'total_chunks': -1  # Will be updated later
                            }
                        })
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sub_el_str)
                    current_size += len(sub_el_str)
            else:
                if current_size + element_size > max_chars and current_chunk:
                    # Save current chunk
                    chunk_html = ''.join(current_chunk)
                    chunks.append({
                        'content': chunk_html,
                        'metadata': {
                            'chunk_num': len(chunks) + 1,
                            'total_chunks': -1  # Will be updated later
                        }
                    })
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(element_str)
                current_size += element_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_html = ''.join(current_chunk)
            chunks.append({
                'content': chunk_html,
                'metadata': {
                    'chunk_num': len(chunks) + 1,
                    'total_chunks': -1  # Will be updated later
                }
            })
        
        # Update total_chunks in each chunk's metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = total_chunks
            
        return chunks
        
    except Exception as e:
        logger.warning(f"Error chunking HTML, falling back to simple chunking: {e}")
        # Fallback: simple character-based chunking
        return [
            {
                'content': html_content[i:i + max_chars],
                'metadata': {
                    'chunk_num': i // max_chars + 1,
                    'total_chunks': (len(html_content) + max_chars - 1) // max_chars
                }
            }
            for i in range(0, len(html_content), max_chars)
        ]

def track_request():
    """
    Track API requests and enforce rate limiting with minimum delay between requests.
    Also checks and updates the daily request counter.
    """
    global LAST_REQUEST_TIME, DAILY_REQUESTS, LAST_REQUEST_DATE
    
    # Check daily limit first
    if check_daily_limit():
        raise ResourceExhausted("Daily request limit reached. Please try again tomorrow.")
    
    # Enforce minimum delay between requests
    current_time = time.time()
    time_since_last = current_time - LAST_REQUEST_TIME
    
    if time_since_last < MIN_DELAY_BETWEEN_REQUESTS:
        sleep_time = MIN_DELAY_BETWEEN_REQUESTS - time_since_last
        logger.info(f"Rate limiting: Waiting {sleep_time:.1f}s before next request")
        time.sleep(max(0, sleep_time))
    
    # Update last request time
    LAST_REQUEST_TIME = time.time()
    
    # Update daily counter
    update_daily_counter()
    
    # Log current usage
    logger.info(f"API Request: {DAILY_REQUESTS}/{REQUESTS_PER_DAY} requests used today")

def get_cached_analysis(url: str) -> Optional[Dict]:
    """Retrieve cached analysis if available and not expired."""
    if url in ANALYSIS_CACHE:
        cached_time, result = ANALYSIS_CACHE[url]
        if time.time() - cached_time < CACHE_TTL:
            logger.info(f"Retrieved cached analysis for {url}")
            return result
    return None

def set_cached_analysis(url: str, result: Dict):
    """Cache the analysis result with a timestamp."""
    ANALYSIS_CACHE[url] = (time.time(), result)
    logger.debug(f"Cached analysis for {url}")

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up file handler with rotation (max 5MB per file, keep 3 backup files)
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

# Configure specific loggers
logger = logging.getLogger(__name__)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('selenium').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Enable HTTP client debugging
http_client.HTTPConnection.debuglevel = 1
requests_log = logging.getLogger("urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

# Import scrapers and optimizations
from src.crawl4ai_scraper import Crawl4AIScraper
from src.scraper_agent_clean import ScraperAgent
from src.batch_processor import BatchRequest, batch_processor
from src.optimizations import (
    init_driver, 
    wait_for_element,
    get_cached_selectors,
    cache_selectors,
    get_suggested_selectors,
    process_batch,
    process_single_url
)

# Import Crawl4AI scraper
try:
    from src.crawl4ai_scraper import Crawl4AIScraper
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    st.warning("Crawl4AI scraper not available. Some features will be disabled.")

# Performance settings
DEFAULT_WAIT_TIME = 1.0  # Reduced default wait time
MAX_PAGES = 10
DEFAULT_MODEL = "gemini-1.5-pro"
MAX_WORKERS = 3  # Number of parallel workers for scraping
CACHE_TTL = 3600  # 1 hour cache TTL

# Cache for storing scraped data
scrape_cache = {}

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_WAIT_TIME = 1.0  # Reduced default wait time
MAX_PAGES = 10
DEFAULT_MODEL = "gemini-1.5-pro"
MAX_WORKERS = 3  # Number of parallel workers for scraping
CACHE_TTL = 3600  # 1 hour cache TTL

# Cache for storing scraped data
scrape_cache = {}

def get_cached_scrape(url: str, selectors: Dict[str, str]) -> Optional[Dict]:
    """Get cached scrape results if available and not expired."""
    cache_key = (url, frozenset(selectors.items()))
    if cache_key in scrape_cache:
        cached_time, result = scrape_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            return result
    return None

def set_cached_scrape(url: str, selectors: Dict[str, str], result: Dict):
    """Cache the scrape results."""
    cache_key = (url, frozenset(selectors.items()))
    scrape_cache[cache_key] = (time.time(), result)

# --- Pinecone Configuration ---
PINECONE_INDEX_NAME = "web-scraper-cache"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize Pinecone if API key is available
def init_pinecone(api_key: str):
    """Initialize Pinecone with the given API key."""
    try:
        pinecone.init(api_key=api_key, environment="gcp-starter")
        
        # Create the index if it doesn't exist
        if PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine"
            )
        
        return pinecone.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        return None

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    """Get the sentence transformer model for embeddings."""
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        return None

def get_cache_key(url: str, prefix: str = "scrape") -> str:
    """Generate a unique cache key for a URL and prefix."""
    return f"{prefix}:{hashlib.md5(url.encode()).hexdigest()}"

def store_in_pinecone(index, url: str, data: Dict, prefix: str = "scrape") -> bool:
    """
    Store data in Pinecone with URL as the key.
    
    Args:
        index: Pinecone index instance
        url: URL being cached
        data: Data to store (must be JSON serializable)
        prefix: Prefix for the cache key
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate embedding for semantic search
        model = get_embedding_model()
        if not model:
            return False
            
        # Create a text representation of the data for embedding
        text_to_embed = f"{url} {json.dumps(data, default=str)[:1000]}"
        embedding = model.encode(text_to_embed).tolist()
        
        # Prepare metadata (limit size to Pinecone's 10KB limit)
        metadata = {
            'url': url,
            'timestamp': time.time(),
            'prefix': prefix,
            'data': json.dumps(data, default=str)[:8000]  # Truncate if needed
        }
        
        # Generate a unique ID for this entry
        entry_id = get_cache_key(url, prefix)
        
        # Upsert the vector
        index.upsert(vectors=[(entry_id, embedding, metadata)])
        logger.info(f"Stored data in Pinecone for {url} with prefix {prefix}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing in Pinecone: {str(e)}")
        return False

def retrieve_from_pinecone(index, url: str, prefix: str = "scrape", min_similarity: float = 0.8) -> Optional[Dict]:
    """
    Retrieve data from Pinecone using URL and optional prefix.
    
    Args:
        index: Pinecone index instance
        url: URL to look up
        prefix: Prefix used when storing
        min_similarity: Minimum similarity score (0-1) to consider a match
        
    Returns:
        Cached data if found and fresh enough, else None
    """
    try:
        model = get_embedding_model()
        if not model:
            return None
            
        # Generate embedding for the query
        query_embedding = model.encode(url).tolist()
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            filter={"prefix": {"$eq": prefix}}
        )
        
        # Check if we got a match
        if not results.matches or results.matches[0].score < min_similarity:
            return None
            
        match = results.matches[0]
        metadata = match.metadata
        
        # Check if the URL matches exactly (for safety)
        if metadata.get('url') != url:
            logger.warning(f"URL mismatch: requested {url}, got {metadata.get('url')}")
            return None
            
        # Check if the data is still fresh (e.g., less than 7 days old)
        if time.time() - float(metadata.get('timestamp', 0)) > 7 * 24 * 60 * 60:  # 7 days
            logger.info(f"Cached data for {url} is too old")
            return None
            
        # Parse and return the data
        return json.loads(metadata['data'])
        
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {str(e)}")
        return None

# --- Pydantic Models ---
class ScrapedData(BaseModel):
    """Model for scraped data."""
    data: Dict[str, Any]
    url: str
    page_num: int

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AnalysisResult:
    """Container for analysis results with performance metrics."""
    url: str = ""
    title: str = ""
    content_type: str = ""
    elements: List[Dict[str, Any]] = field(default_factory=list)
    suggested_selectors: List[Dict[str, str]] = field(default_factory=list)
    pagination_info: Dict[str, Any] = field(default_factory=dict)
    page_structure: Dict[str, Any] = field(default_factory=dict)
    technologies: List[str] = field(default_factory=list)
    accessibility: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    screenshot: Optional[str] = None
    raw_response: Optional[str] = None
    processing_time: float = 0.0
    forms: Optional[List[Dict[str, Any]]] = None
    links: Optional[List[Dict[str, str]]] = None

# --- Utility Functions ---
def is_valid_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def init_session_state():
    """Initialize session state variables."""
    # Initialize default values if they don't exist
    default_values = {
        'selectors': {},
        'scraping_results': [],
        'analysis_results': None,
        'gemini_api_key': '',
        'pinecone_api_key': '',
        'use_proxy': False,
        'proxy_url': '',
        'default_wait_time': DEFAULT_WAIT_TIME,
        'crawl4ai_url': '',
        'crawl4ai_results': None
    }
    
    # Set default values if they don't exist
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Core Functions ---
def preprocess_html(html: str, max_length: int = 10000) -> str:
    """Preprocess HTML to ensure consistent input size for TensorFlow."""
    if not html:
        return ""
    # Truncate or pad HTML to fixed size
    return html[:max_length].ljust(max_length)[:max_length]

def get_selector_fallback() -> Dict[str, List[str]]:
    """Return fallback selectors when AI inference fails."""
    return {
        'title': ['h1', '.title', 'h2'],
        'content': ['article', '.content', 'main', 'div[role="main"]'],
        'links': ['a[href]'],
        'images': ['img[src]']
    }

def scrape_with_selenium(url: str, use_ai_selectors: bool = True, custom_selectors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Enhanced webpage scraper using Selenium with improved dynamic content handling.
    
    Args:
        url: The URL to scrape
        use_ai_selectors: Whether to use AI for selector suggestion
        custom_selectors: Optional dictionary of custom CSS selectors to use
        
    Returns:
        Dictionary containing scraped data and metadata
    """
    driver = None
    start_time = time.time()
    
    def wait_for_element(driver, selector: str, timeout: int = 10):
        """Wait for an element to be present in the DOM and visible."""
        try:
            return WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
        except:
            return None
    
    def wait_for_elements(driver, selector: str, min_count: int = 1, timeout: int = 10):
        """Wait for at least min_count elements to be present."""
        try:
            return WebDriverWait(driver, timeout).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, selector)) >= min_count
            )
        except:
            return None
    
    try:
        # Initialize the WebDriver with optimized settings
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        
        # Set up Chrome options for better performance
        options.add_experimental_option('prefs', {
            'profile.managed_default_content_settings.images': 2,  # Disable images
            'profile.managed_default_content_settings.javascript': 1,  # Keep JS enabled
        })
        
        driver = webdriver.Chrome(options=options)
        
        # Set timeouts
        driver.set_page_load_timeout(30)  # Increased from 15 to 30 seconds
        driver.set_script_timeout(30)
        
        # Navigate to the URL
        logger.info(f"Navigating to URL: {url}")
        driver.get(url)
        
        # Wait for essential content to load
        logger.info("Waiting for page to load...")
        if not wait_for_element(driver, 'body', 15):
            raise TimeoutError("Page failed to load within timeout")
        
        # Scroll to trigger lazy-loaded content
        logger.info("Scrolling to load dynamic content...")
        for i in range(3):  # Scroll 3 times to load lazy content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)  # Short pause between scrolls
        
        # Get page source after dynamic content loads
        html_content = driver.page_source
        
        # Initialize data structure
        data = {}
        selectors_used = {}
        
        # Use custom selectors if provided, otherwise try to get them
        if custom_selectors:
            selectors = custom_selectors
            logger.info(f"Using custom selectors for {url}")
        else:
            # Try to get cached selectors first
            cached_selectors = get_cached_selectors(url)
            
            if cached_selectors:
                selectors = cached_selectors
                logger.info(f"Using cached selectors for {url}")
            elif use_ai_selectors:
                try:
                    # Try to get AI-suggested selectors
                    selectors = get_suggested_selectors(html_content, url)
                    # Cache the selectors for future use
                    cache_selectors(url, selectors)
                    logger.info(f"Generated new selectors for {url}")
                except Exception as e:
                    logger.warning(f"AI selector generation failed, using fallback: {str(e)}")
                    selectors = get_selector_fallback()
            else:
                # Use fallback selectors
                selectors = get_selector_fallback()
        
        # Extract data using selectors with error handling
        for key, selector_list in selectors.items():
            if not isinstance(selector_list, list):
                selector_list = [selector_list]
                
            for selector in selector_list:
                try:
                    # Wait for elements to be present
                    wait_for_element(driver, selector, 10)
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    if elements:
                        extracted = []
                        for el in elements:
                            try:
                                # Scroll element into view
                                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", el)
                                time.sleep(0.2)  # Small delay for any animations
                                
                                # Get text content
                                text = el.text.strip()
                                if text:  # Only add non-empty text
                                    extracted.append(text)
                                    
                                    # If we found at least one match, consider this selector successful
                                    if key not in data:
                                        data[key] = extracted
                                        selectors_used[key] = selector
                                        logger.info(f"Found {len(extracted)} elements for '{key}' with selector: {selector}")
                                        break  # Move to next key after first successful selector
                                        
                            except Exception as e:
                                logger.debug(f"Error extracting text from element: {str(e)}")
                                continue
                        
                        if key in data:  # If we found data with this selector, move to next key
                            break
                            
                except Exception as e:
                    logger.debug(f"Selector '{selector}' failed: {str(e)}")
                    continue
        
        # Take a screenshot for debugging
        screenshot = None
        try:
            screenshot = driver.get_screenshot_as_base64()
        except Exception as e:
            logger.warning(f"Failed to take screenshot: {str(e)}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        result = {
            'url': url,
            'title': driver.title,
            'html': html_content,
            'data': data,
            'screenshot': screenshot,
            'user_agent': driver.execute_script("return navigator.userAgent"),
            'timestamp': time.time(),
            'processing_time': round(processing_time, 2),
            'status': 'success',
            'selectors_used': selectors_used
        }
        
        logger.info(f"Successfully scraped {url} in {processing_time:.2f} seconds")
        return result
        
    except Exception as e:
        error_msg = f"Selenium scraping failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            'url': url,
            'error': error_msg,
            'status': 'error',
            'processing_time': round(time.time() - start_time, 2) if 'start_time' in locals() else 0,
            'traceback': traceback.format_exc()
        }
        
    finally:
        if 'driver' in locals() and driver:
            try:
                driver.quit()
            except Exception as e:
                logger.error(f"Error closing WebDriver: {str(e)}")

@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(multiplier=1, min=4, max=60),  # Exponential backoff: 4s, 8s, up to 60s
    retry=retry_if_exception_type((ResourceExhausted, requests.exceptions.RequestException)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def process_chunk(chunk: str, url: str, api_key: str, chunk_num: int, total_chunks: int) -> Dict[str, Any]:
    """
    Process a single chunk of HTML content using Mistral API.
    
    Args:
        chunk: The HTML chunk to process
        url: Original URL for reference
        api_key: Mistral API key
        chunk_num: Current chunk number
        total_chunks: Total number of chunks
        
    Returns:
        Dictionary containing processed chunk data
    """
    try:
        # Validate API key
        if not api_key or not isinstance(api_key, str) or not api_key.startswith('m'):
            raise ValueError("Invalid or missing Mistral API key. Please check your API key in the sidebar.")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Try with mixtral-8x7b-instruct first, fall back to mistral-7b-instruct if needed
        models_to_try = ["mixtral-8x7b-instruct", "mistral-7b-instruct"]
        last_error = None
        
        for model in models_to_try:
            try:
                logger.info(f"Trying model: {model} for chunk {chunk_num + 1}/{total_chunks}")
                
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant that processes HTML content chunks. "
                                "Extract and structure the main content, keeping the semantic meaning intact. "
                                "Focus on the most important information and structure it clearly."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Process this HTML chunk ({chunk_num + 1}/{total_chunks}) from {url}:\n\n"
                                f"{chunk[:100000]}"  # Limit chunk size to avoid token limits
                            )
                        }
                    ],
                    "temperature": 0.2,  # Lower temperature for more deterministic output
                    "top_p": 0.95,
                    "max_tokens": 2000,  # Limit response size
                    "presence_penalty": 0.1,
                    "frequency_penalty": 0.1
                }
                
                # Make the API request
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # If we get here, the request was successful
                response.raise_for_status()
                response_data = response.json()
                
                # Validate response format
                if not response_data.get('choices') or not response_data['choices'][0].get('message'):
                    raise ValueError("Invalid response format from Mistral API")
                
                # If we got here, the model worked, so break out of the retry loop
                break
                
            except Exception as e:
                last_error = e
                logger.warning(f"Error with model {model}: {str(e)}")
                if model == models_to_try[-1]:  # If this was the last model to try
                    raise last_error
                # Otherwise, continue to the next model
                continue
        
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            response_data = response.json()
            if 'choices' not in response_data or not response_data['choices']:
                raise ValueError("Invalid response format from Mistral API")
                
            return {
                'success': True,
                'chunk_num': chunk_num,
                'content': response_data['choices'][0]['message']['content'],
                'usage': response_data.get('usage', {})
            }
            
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 401:
                error_detail = "Unauthorized - Please check your Mistral API key"
                if http_err.response.text:
                    try:
                        error_json = http_err.response.json()
                        error_detail = error_json.get('error', {}).get('message', str(http_err))
                    except:
                        error_detail = http_err.response.text
                raise ValueError(f"API Error (401): {error_detail}")
            raise
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
        return {
            'success': False,
            'chunk_num': chunk_num,
            'error': str(e)
        }

def chunk_html(html_content: str, max_chars: int = 100000) -> List[Dict[str, Any]]:
    """
    Split HTML into semantic chunks using BeautifulSoup.
    
    Args:
        html_content: The HTML content to split
        max_chars: Maximum size of each chunk in characters
        
    Returns:
        List of dictionaries containing chunk content and metadata
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove non-content elements
    for element in soup(['script', 'style', 'noscript', 'link', 'meta']):
        element.decompose()
    
    # Priority order for splitting
    priority_tags = ['main', 'article', 'section', 'div', 'p']
    chunks = []
    current_chunk = []
    current_size = 0
    
    def add_chunk():
        nonlocal chunks, current_chunk, current_size
        if current_chunk:
            chunks.append({
                'content': ''.join(current_chunk),
                'metadata': {
                    'chunk_num': len(chunks) + 1,
                    'total_chunks': -1  # Will be updated later
                }
            })
            current_chunk = []
            current_size = 0
    
    # Process elements in priority order
    for element in soup.find_all(priority_tags):
        element_str = str(element)
        element_size = len(element_str)
        
        # If adding this element would exceed max size, finalize current chunk
        if current_size + element_size > max_chars and current_chunk:
            add_chunk()
        
        current_chunk.append(element_str)
        current_size += element_size
    
    # Add any remaining content
    if current_chunk:
        add_chunk()
    
    # Update total_chunks in metadata
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk['metadata']['total_chunks'] = total_chunks
    
    return chunks

def split_html_with_overlap(html: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """
    Split HTML into chunks with overlap, respecting semantic boundaries.
    
    Args:
        html: The HTML content to split
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of HTML chunks
    """
    # First, split into semantic chunks
    semantic_chunks = chunk_html(html, max_chars=chunk_size * 2)
    
    # If we got semantic chunks, use them
    if len(semantic_chunks) > 1:
        return [chunk['content'] for chunk in semantic_chunks]
    
    # Fall back to character-based splitting if semantic splitting didn't work
    chunks = []
    start = 0
    
    while start < len(html):
        end = min(start + chunk_size, len(html))
        
        if end >= len(html):
            chunks.append(html[start:])
            break
        
        # Try to find a good breaking point
        overlap_start = max(start, end - overlap)
        overlap_end = end
        
        # Look for HTML tag boundaries first
        if '>' in html[overlap_start:end]:
            overlap_end = html.rfind('>', overlap_start, end) + 1
        # Then try sentence boundaries
        elif '.' in html[overlap_start:end]:
            overlap_end = html.rfind('.', overlap_start, end) + 1
        # Then word boundaries
        elif ' ' in html[overlap_start:end]:
            overlap_end = html.rfind(' ', overlap_start, end) + 1
        
        chunks.append(html[start:overlap_end])
        start = overlap_end - overlap if overlap_end > overlap_start + overlap else end - overlap
        
    return chunks

def make_mistral_request(api_key: str, payload: Dict, max_retries: int = 3, initial_delay: float = 1.0) -> Dict:
    """
    Make a request to Mistral API with exponential backoff retry logic.
{{ ... }}
    
    Args:
        api_key: Mistral API key
        payload: The request payload
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        
    Returns:
        JSON response from the API
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            # If rate limited, extract retry-after header if available
            if response.status_code == 429:
                retry_after = float(response.headers.get('Retry-After', delay))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_after)
                delay = min(delay * 2, 60)  # Exponential backoff with max 60s
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries:
                sleep_time = delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
    
    # If we've exhausted all retries
    raise last_exception or Exception("Failed to complete request after multiple retries")

def process_chunk_with_retry(chunk: str, url: str, api_key: str, chunk_num: int, total_chunks: int, max_retries: int = 3) -> Dict:
    """Process a single chunk with retry logic."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return process_chunk(chunk, url, api_key, chunk_num, total_chunks)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Chunk {chunk_num + 1}/{total_chunks} failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
            
    # If all retries failed, return error
    return {
        'success': False,
        'chunk_num': chunk_num,
        'error': str(last_exception)
    }

def scrape_with_mistral(url: str, api_key: str, max_workers: int = 3, pinecone_index=None) -> Dict[str, Any]:
    """
    Scrape and process webpage content using Mistral API with parallel chunking and Pinecone caching.
    
    Args:
        url: The URL to scrape
        api_key: Mistral API key
        max_workers: Maximum number of parallel API calls
        pinecone_index: Optional Pinecone index for caching
        
    Returns:
        Dictionary containing scraped and processed data
    """
    logger.info(f"Starting scrape_with_mistral for URL: {url}")
    start_time = time.time()
    
    # Check Pinecone cache first if index is provided
    if pinecone_index:
        cached_result = retrieve_from_pinecone(pinecone_index, url, "scrape")
        if cached_result:
            logger.info(f"Returning cached result from Pinecone for {url}")
            return cached_result
    
    # Fall back to local cache if Pinecone is not available or has no result
    cached_result = get_cached_scrape(url, {})
    if cached_result:
        logger.info(f"Returning locally cached result for {url}")
        # Optionally store in Pinecone for future use
        if pinecone_index:
            store_in_pinecone(pinecone_index, url, cached_result, "scrape")
        return cached_result
        
    try:
        # Get the HTML content
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        html_content = response.text
        
        # Parse the HTML to get the title
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string if soup.title else 'No title found'
        
        # First, try semantic chunking
        chunks = chunk_html(html_content)
        
        # If no semantic chunks, fall back to character-based chunking
        if not chunks:
            logger.info("No semantic chunks found, falling back to character-based chunking")
            chunks = split_html_with_overlap(html_content)
        
        if not chunks:
            raise ValueError("Failed to split HTML into chunks")
            
        logger.info(f"Split HTML into {len(chunks)} chunks")
        
        # Process chunks in parallel with rate limiting
        results = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
            # Create a future for each chunk
            future_to_chunk = {
                executor.submit(
                    process_chunk_with_retry,
                    chunk,
                    url,
                    api_key,
                    i,
                    len(chunks)
                ): i for i, chunk in enumerate(chunks)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    if result.get('success'):
                        results.append(result)
                    else:
                        logger.error(f"Error processing chunk: {result.get('error')}")
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
        
        # Sort results by chunk number to maintain original order
        results.sort(key=lambda x: x['chunk_num'])
        
        # Combine results
        combined_content = "\n\n".join([r['content'] for r in results if 'content' in r])
        
        # Prepare the final payload for Mistral API
        final_payload = {
            "model": "mistral-tiny",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that processes and summarizes web content."},
                {"role": "user", "content": f"Please analyze and summarize the following content from {url}:\n\n{combined_content}"}
            ],
            "max_tokens": 4000,
            "temperature": 0.7
        }
        
        # Use the rate-limited request function
        final_response = make_mistral_request(api_key, final_payload)
        
        # Prepare the final result
        data = {
            'title': soup.title.string if soup.title else url,
            'html': final_response.get('choices', [{}])[0].get('message', {}).get('content', ''),
            'url': url,
            'status': 'success',
            'processing_time': time.time() - start_time,
            'chunks_processed': len(results),
            'total_chunks': len(chunks)
        }
        
        return data
        
    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            'url': url,
            'error': error_msg,
            'status': 'error',
            'processing_time': time.time() - start_time
        }

def analyze_page_with_chunking(url: str, api_key: str, use_selenium: bool = False, 
                             max_retries: int = MAX_RETRIES, pinecone_index=None) -> Optional[Dict]:
    """
    Analyze a webpage by splitting its HTML into chunks and processing each chunk separately,
    with Pinecone caching for improved performance.

    Args:
        url: The URL of the webpage
        api_key: Mistral API key
        use_selenium: Whether to use Selenium for scraping
        max_retries: Maximum number of retry attempts for failed chunks
        pinecone_index: Optional Pinecone index for caching results

    Returns:
        A dictionary containing combined analysis results or None if analysis fails
    """
    start_time = time.time()
    logger.info(f"Starting analysis for URL: {url}")
    
    # Check Pinecone cache first if index is provided
    if pinecone_index:
        cached_analysis = retrieve_from_pinecone(pinecone_index, url, "analysis")
        if cached_analysis and cached_analysis.get('success', False):
            logger.info(f"Returning cached analysis from Pinecone for {url}")
            cached_analysis['cached'] = True
            cached_analysis['processing_time'] = time.time() - start_time
            return cached_analysis
    
    # Fall back to local cache if Pinecone is not available or has no result
    cached_analysis = get_cached_analysis(url)
    if cached_analysis and cached_analysis.get('success', False):
        logger.info(f"Returning locally cached analysis for {url}")
        # Optionally store in Pinecone for future use
        if pinecone_index:
            store_in_pinecone(pinecone_index, url, cached_analysis, "analysis")
        cached_analysis['cached'] = True
        cached_analysis['processing_time'] = time.time() - start_time
        return cached_analysis

    # Check daily limit
    try:
        check_daily_limit()
    except Exception as e:
        error_msg = f"Daily limit error: {str(e)}"
        logger.error(error_msg)
        error_result = {
            'url': url,
            'success': False,
            'error': error_msg,
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        }
        # Cache the error to avoid repeated failures
        if pinecone_index:
            store_in_pinecone(pinecone_index, url, error_result, "analysis_error")
        st.error(error_msg)
        return None

    # Scrape the page content using Mistral API with Pinecone caching
    try:
        scraped_data = scrape_with_mistral(url, api_key, pinecone_index=pinecone_index)
        if not scraped_data or 'html' not in scraped_data:
            error_msg = scraped_data.get('error', 'Failed to scrape content')
            logger.error(f"Failed to scrape content for {url}: {error_msg}")
            error_result = {
                'url': url,
                'success': False,
                'error': f"Failed to scrape content: {error_msg}",
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            # Cache the error to avoid repeated failures
            if pinecone_index:
                store_in_pinecone(pinecone_index, url, error_result, "analysis_error")
            st.error(f"Failed to scrape content for {url}: {error_msg}")
            return None
            
    except Exception as e:
        error_msg = f"Error during scraping: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        error_result = {
            'url': url,
            'success': False,
            'error': error_msg,
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        }
        # Cache the error to avoid repeated failures
        if pinecone_index:
            store_in_pinecone(pinecone_index, url, error_result, "analysis_error")
        st.error(error_msg)
        return None

    try:
        html_content = scraped_data.get('html', '')
        title = scraped_data.get('title', '')

        # Prepare the base result with metadata
        analysis_result = {
            'url': url,
            'title': title,
            'success': True,
            'processing_time': time.time() - start_time,
            'timestamp': time.time(),
            'cached': False,
            'chunks_processed': 0,
            'total_chunks': 0,
            'elements': [],
            'forms': [],
            'navigation': [],
            'recommendations': []
        }

        # Check if we should skip analysis (e.g., for large pages)
        if len(html_content) > 500000:  # 500KB
            logger.warning(f"Page too large for full analysis: {len(html_content)} characters")
            analysis_result['warning'] = "Page too large for full analysis, only basic information extracted"
            analysis_result['content_length'] = len(html_content)
            
            # Cache the basic result
            set_cached_analysis(url, analysis_result)
            if pinecone_index:
                store_in_pinecone(pinecone_index, url, analysis_result, "analysis")
            return analysis_result

        # Split HTML into chunks
        chunks = split_html_into_chunks(html_content)
        logger.info(f"Processing {len(chunks)} chunks for {url}")
        analysis_result['total_chunks'] = len(chunks)

        # List to store analysis results for each chunk
        analyses = []

        # Progress bar for Streamlit
        progress_bar = st.progress(0) if st is not None else None
        status_text = st.empty() if st is not None else None

        # Process each chunk with rate limiting
        global LAST_REQUEST_TIME
        for i, chunk in enumerate(chunks):
            # Calculate time since last request
            current_time = time.time()
            time_since_last = current_time - LAST_REQUEST_TIME
            
            # Enforce minimum delay between requests
            if time_since_last < MIN_DELAY_BETWEEN_REQUESTS:
                wait_time = MIN_DELAY_BETWEEN_REQUESTS - time_since_last
                logger.info(f"Rate limiting: Waiting {wait_time:.1f} seconds before next request")
                time.sleep(wait_time)
            
            if status_text is not None:
                status_text.text(f"Analyzing chunk {i + 1}/{len(chunks)}...")
            
            # Update last request time
            LAST_REQUEST_TIME = time.time()
            
            # Define the JSON template as a raw string to avoid string formatting issues
            json_template = '''
            {
              "chunk_id": %d,
              "structure": "Brief overview of the chunk's structure",
              "elements": [{"type": "string", "selector": "string", "description": "string"}],
              "forms": [{"action": "string", "method": "string", "fields": [{"name": "string", "type": "string"}]}],
              "navigation": ["string"],
              "recommendations": ["string"]
            }
            '''.strip() % (i + 1)
            
            prompt = f"""Analyze this part of the webpage (chunk {i + 1}/{len(chunks)}):
    URL: {url}
    Title: {title}
    HTML Content: {chunk}

    Provide in JSON format:
    {json_template}
    """
            
            try:
                # Configure API with rate limiting
                genai.configure(api_key=api_key)
                
                # Make the API call with exponential backoff
                response = make_mistral_request(
                    api_key=api_key,
                    payload={
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "model": "mistral-tiny",
                        "temperature": 0.3,
                        "max_tokens": 2000
                    },
                    max_retries=max_retries
                )
                
                if not response or 'choices' not in response or not response['choices']:
                    error_msg = f"Invalid or empty response from API for chunk {i + 1}"
                    logger.error(error_msg)
                    logger.debug(f"Response: {response}")
                    continue
                
                try:
                    # Extract the response content from the Mistral API response format
                    response_message = response['choices'][0].get('message', {})
                    response_content = response_message.get('content', '').strip()
                    
                    if not response_content:
                        logger.error(f"Empty content in response for chunk {i + 1}")
                        continue
                        
                    # Try to parse the response as JSON
                    try:
                        chunk_analysis = json.loads(response_content)
                        analyses.append(chunk_analysis)
                        analysis_result['chunks_processed'] += 1
                        
                        # Update progress
                        if progress_bar is not None:
                            progress = (i + 1) / len(chunks)
                            progress_bar.progress(min(progress, 1.0))
                            
                    except json.JSONDecodeError as je:
                        logger.error(f"Failed to parse JSON response for chunk {i + 1}: {je}")
                        logger.debug(f"Response content: {response_content}")
                        continue
                        
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse JSON response for chunk {i + 1}: {je}")
                    logger.debug(f"Response content: {response_text}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                logger.debug(f"{traceback.format_exc()}")
                continue
                
        # Combine analyses from all chunks
        if analyses:
            # Combine elements, forms, navigation, and recommendations
            for analysis in analyses:
                if 'elements' in analysis:
                    analysis_result['elements'].extend(analysis['elements'])
                if 'forms' in analysis:
                    analysis_result['forms'].extend(analysis['forms'])
                if 'navigation' in analysis:
                    analysis_result['navigation'].extend(analysis['navigation'])
                if 'recommendations' in analysis:
                    analysis_result['recommendations'].extend(analysis['recommendations'])
            
            # Remove duplicates while preserving order
            def make_hashable(obj):
                if isinstance(obj, dict):
                    return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
                elif isinstance(obj, list):
                    return tuple(make_hashable(x) for x in obj)
                return obj
                
            for key in ['elements', 'forms', 'navigation', 'recommendations']:
                if key in analysis_result and isinstance(analysis_result[key], list):
                    seen = set()
                    unique_items = []
                    for item in analysis_result[key]:
                        # Create a hashable version of the item
                        item_hashable = make_hashable(item)
                        if item_hashable not in seen:
                            seen.add(item_hashable)
                            unique_items.append(item)
                    analysis_result[key] = unique_items
        
        # Update processing time
        analysis_result['processing_time'] = time.time() - start_time
        
        # Cache the successful analysis
        set_cached_analysis(url, analysis_result)
        if pinecone_index:
            store_in_pinecone(pinecone_index, url, analysis_result, "analysis")
        
        logger.info(f"Successfully analyzed {url} in {analysis_result['processing_time']:.2f} seconds")
        return analysis_result
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        error_result = {
            'url': url,
            'success': False,
            'error': error_msg,
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        }
        # Cache the error to avoid repeated failures
        if pinecone_index:
            store_in_pinecone(pinecone_index, url, error_result, "analysis_error")
        if st is not None:
            st.error(error_msg)
        return error_result

    try:
        if status_text is not None:
            status_text.text("Combining analysis results...")

        # Combine analyses from all chunks
        if analyses:
            # Combine elements, forms, navigation, and recommendations
            for analysis in analyses:
                if 'elements' in analysis and isinstance(analysis['elements'], list):
                    analysis_result['elements'].extend(analysis['elements'])
                if 'forms' in analysis and isinstance(analysis['forms'], list):
                    analysis_result['forms'].extend(analysis['forms'])
                if 'navigation' in analysis and isinstance(analysis['navigation'], list):
                    analysis_result['navigation'].extend(analysis['navigation'])
                if 'recommendations' in analysis and isinstance(analysis['recommendations'], list):
                    analysis_result['recommendations'].extend(analysis['recommendations'])
            
            # Remove duplicates while preserving order
            for key in ['elements', 'forms', 'navigation', 'recommendations']:
                if key in analysis_result and isinstance(analysis_result[key], list):
                    seen = set()
                    analysis_result[key] = [x for x in analysis_result[key] 
                                         if not (str(x) in seen or seen.add(str(x)))]
        
        # Update processing time
        analysis_result['processing_time'] = time.time() - start_time
        
        # Cache the successful analysis
        set_cached_analysis(url, analysis_result)
        if pinecone_index:
            store_in_pinecone(pinecone_index, url, analysis_result, "analysis")
        
        logger.info(f"Successfully analyzed {url} in {analysis_result['processing_time']:.2f} seconds")
        
        # Clear progress bar and status if they exist
        if progress_bar is not None:
            progress_bar.empty()
        if status_text is not None:
            status_text.empty()
            
        return analysis_result
        
    except Exception as e:
        error_msg = f"Error combining analysis results: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        error_result = {
            'url': url,
            'success': False,
            'error': error_msg,
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        }
        # Cache the error to avoid repeated failures
        if pinecone_index:
            store_in_pinecone(pinecone_index, url, error_result, "analysis_error")
        if st is not None:
            st.error(error_msg)
        return error_result

def _prepare_chunk_prompt(url: str, scraped_data: Dict[str, Any], chunk: Dict[str, Any]) -> str:
    """
    Prepare a prompt for analyzing a specific chunk of HTML content.
    
    Args:
        url: The URL being analyzed
        scraped_data: Dictionary containing scraped data including 'title'
        chunk: Dictionary containing chunk data and metadata
        
    Returns:
        str: Formatted prompt for the Gemini API
    """
    chunk_content = chunk.get('content', '')
    chunk_meta = chunk.get('metadata', {})
    chunk_num = chunk_meta.get('chunk_num', 1)
    total_chunks = chunk_meta.get('total_chunks', 1)
    title = scraped_data.get('title', 'No title')
    
    prompt = f"""Analyze the following chunk of a webpage (chunk {chunk_num} of {total_chunks}). 
Focus on the specific content of this chunk and provide a detailed technical analysis.

URL: {url}
Title: {title}
Chunk: {chunk_num} of {total_chunks}

HTML Content:
{chunk_content}

For this chunk, please analyze and provide:
1. Content summary
2. Key elements and their selectors
3. Any forms or interactive elements
4. Notable patterns or structures
5. Any potential issues or recommendations specific to this chunk

Be specific and detailed in your analysis. If this chunk appears to be part of a larger element 
(like a navigation bar, footer, or content section), note that in your analysis.

IMPORTANT: Focus ONLY on the content of this specific chunk. Do not try to analyze the entire page."""
    
    return prompt

def _parse_analysis_result(analysis_text: str, url: str) -> AnalysisResult:
    """
    Parse the analysis text into an AnalysisResult object.
    
    Args:
        analysis_text: Raw analysis text from Gemini
        url: URL that was analyzed
        
    Returns:
        AnalysisResult: Parsed analysis result
    """
    try:
        # Default values
        content_type = "text/html"
        elements = []
        suggested_selectors = []
        pagination_info = {}
        
        # If this is a combined analysis from multiple chunks, process it differently
        if "--- Chunk" in analysis_text:
            # Extract chunk analyses
            chunks = []
            current_chunk = []
            current_chunk_num = 1
            
            # Split the text into lines and process them
            for line in analysis_text.split('\n'):
                if line.strip().startswith('--- Chunk'):
                    if current_chunk:
                        chunks.append(('\n'.join(current_chunk), current_chunk_num))
                        current_chunk = []
                    try:
                        current_chunk_num = int(line.split('Chunk')[1].split('---')[0].strip())
                    except (IndexError, ValueError):
                        current_chunk_num += 1
                else:
                    current_chunk.append(line)
            
            # Add the last chunk
            if current_chunk:
                chunks.append(('\n'.join(current_chunk), current_chunk_num))
            
            # Combine all chunk analyses into elements
            for chunk_content, chunk_num in chunks:
                elements.append({
                    'type': 'chunk',
                    'chunk_number': chunk_num,
                    'content': chunk_content.strip()
                })
        
        # Create and return the AnalysisResult
        return AnalysisResult(
            url=url,
            content_type=content_type,
            elements=elements,
            suggested_selectors=suggested_selectors,
            pagination_info=pagination_info
        )
        
    except Exception as e:
        logger.error(f"Error parsing analysis result: {str(e)}")
        # Return a basic result with the error
        return AnalysisResult(
            url=url,
            content_type="text/plain",
            elements=[{'type': 'error', 'content': f"Error parsing analysis: {str(e)}"}],
            suggested_selectors=[],
            pagination_info={}
        )

def _process_async_analysis(url: str, api_key: str, use_selenium: bool, callback: Callable, start_time: float) -> None:
    """
    Process analysis asynchronously using the batch processor.
    
    Args:
        url: URL to analyze
        api_key: Gemini API key
        use_selenium: Whether to use Selenium for JavaScript rendering
        callback: Callback function to process the result
        start_time: Timestamp when the analysis started
    """
    def process_callback(response):
        """Process the analysis result and call the original callback"""
        try:
            if response.get('status') == 'error':
                error_msg = response.get('error', 'Unknown error in batch processing')
                logger.error(f"Error in batch processing: {error_msg}")
                if callable(callback):
                    callback(None)
                return
                    
            result_data = response.get('result', {})
            if not result_data or 'text' not in result_data:
                logger.error("Invalid response format from batch processor")
                if callable(callback):
                    callback(None)
                return
                
            # Process the response text into an AnalysisResult
            analysis_result = _parse_analysis_result(
                url=url,
                response_text=result_data['text'],
                start_time=start_time,
                metadata={
                    'prompt_feedback': result_data.get('prompt_feedback'),
                    'usage_metadata': result_data.get('usage_metadata')
                }
            )
            
            if callable(callback):
                callback(analysis_result)
                
        except Exception as e:
            logger.error(f"Error processing batch callback: {str(e)}")
            logger.error(traceback.format_exc())
            if callable(callback):
                callback(None)
    
    try:
        # First, scrape the page content
        scraped_data = scrape_with_selenium(url) if use_selenium else None
        
        if not scraped_data or 'html' not in scraped_data:
            logger.error("Failed to scrape page content")
            if callable(callback):
                callback(None)
            return
        
        # Prepare the prompt for analysis
        prompt = _prepare_analysis_prompt(url, scraped_data)
        
        # Prepare the batch request
        request = BatchRequest(
            url=url,
            content={
                'url': url,
                'prompt': prompt,
                'api_key': api_key,
                'use_selenium': use_selenium,
                'start_time': start_time
            },
            callback=process_callback,
            priority=1  # Default priority
        )
        
        # Add to batch processor
        batch_processor.add_request(request)
        
    except Exception as e:
        logger.error(f"Error preparing batch request: {str(e)}")
        logger.error(traceback.format_exc())
        if callable(callback):
            callback(None)
    
    return None

def _process_sync_analysis(url: str, api_key: str, use_selenium: bool, start_time: float) -> Optional[AnalysisResult]:
    """
    Process analysis synchronously with rate limiting and caching.
    
    Args:
        url: URL to analyze
        api_key: Gemini API key
        use_selenium: Whether to use Selenium for JavaScript rendering
        start_time: Timestamp when the analysis started
        
    Returns:
        AnalysisResult with page analysis or None if analysis fails
    """
    # Check cache first to avoid unnecessary API calls
    cached_result = get_cached_analysis(url)
    if cached_result:
        logger.info(f"Using cached analysis for {url}")
        return cached_result
        
    # Enforce rate limiting
    track_request()
    
    try:
        # Scrape the page content
        scraped_data = scrape_with_selenium(url) if use_selenium else scrape_with_requests(url)
        if not scraped_data or 'html' not in scraped_data:
            logger.error(f"Failed to scrape content for {url}")
            return None
            
        # Prepare the optimized prompt
        prompt = _prepare_optimized_prompt(url, scraped_data)
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Generate content using Gemini
        response = model.generate_content(prompt)
        
        # Parse the response
        analysis_result = _parse_analysis_result(
            url=url,
            response_text=response.text,
            start_time=start_time
        )
        
        # Cache the result
        if analysis_result:
            set_cached_analysis(url, analysis_result)
            
        return analysis_result
        
    except ResourceExhausted as e:
        logger.warning(f"Gemini API rate limit exceeded for {url}. Retrying with backoff...")
        raise  # Re-raise to trigger retry with exponential backoff
    except Exception as e:
        logger.error(f"Error in sync analysis for {url}: {str(e)}")
        return None

def _prepare_analysis_prompt(url: str, scraped_data: Dict[str, Any]) -> str:
    """
    Prepare the prompt for analysis based on scraped data.
    
    Args:
        url: The URL being analyzed
        scraped_data: Dictionary containing scraped data including 'html' and other metadata
        
    Returns:
        Formatted prompt string for Gemini API
    """
    # Extract relevant data from scraped_data
    html_content = scraped_data.get('html', '')
    title = scraped_data.get('title', '')
    
    # Truncate HTML if too long
    max_html_length = 15000  # Adjust based on your needs
    if len(html_content) > max_html_length:
        html_content = html_content[:max_html_length] + "... [truncated]"
    
    # Create the prompt
    prompt = f"""Analyze the following webpage and provide a detailed analysis:
    
URL: {url}
Title: {title}

HTML Content:
{html_content}

Please provide analysis including:
1. Page structure and main content areas
2. Key elements and their selectors
3. Any forms and their fields
4. Navigation elements
5. Any potential issues or recommendations
"""
    return prompt

def _parse_analysis_result(url: str, response_text: str, start_time: float, 
                         metadata: Optional[Dict] = None) -> Optional[AnalysisResult]:
    """
    Parse the analysis result from Gemini response into an AnalysisResult object.
    
    Args:
        url: The URL that was analyzed
        response_text: The text response from Gemini
        start_time: Timestamp when analysis started
        metadata: Additional metadata from the API response
        
    Returns:
        AnalysisResult object or None if parsing fails
    """
    try:
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Try to parse the response as JSON first
        try:
            result_data = json.loads(response_text)
        except json.JSONDecodeError:
            # If not JSON, use the text as is
            result_data = {"analysis": response_text}
        
        # Create and return AnalysisResult
        return AnalysisResult(
            url=url,
            title=result_data.get('title', ''),
            content_type=result_data.get('content_type', ''),
            elements=result_data.get('elements', []),
            suggested_selectors=result_data.get('suggested_selectors', []),
            pagination_info=result_data.get('pagination_info', {}),
            page_structure=result_data.get('page_structure', {}),
            technologies=result_data.get('technologies', []),
            accessibility=result_data.get('accessibility', {}),
            performance_metrics={
                'processing_time_seconds': processing_time,
                **result_data.get('performance_metrics', {})
            },
            raw_response=response_text,
            processing_time=processing_time,
            forms=result_data.get('forms'),
            links=result_data.get('links')
        )
        
    except Exception as e:
        logger.error(f"Error parsing analysis result: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def _analyze_page_content(url: str, api_key: str, use_selenium: bool, start_time: float) -> Optional[AnalysisResult]:
    """
    Internal function to analyze page content with error handling.
    
    This is kept for backward compatibility but delegates to the batch processor.
    
    Args:
        url: URL to analyze
        api_key: Gemini API key
        use_selenium: Whether to use Selenium for JavaScript rendering
        start_time: Timestamp when analysis started
        
    Returns:
        AnalysisResult if successful, None otherwise
    """
    # For backward compatibility, we'll use the synchronous method
    return _process_sync_analysis(url, api_key, use_selenium, start_time)

def get_default_selectors() -> Dict[str, List[str]]:
    """
    Return a comprehensive set of default selectors for initial scraping.
    
    Returns:
        Dictionary mapping element names to lists of CSS selectors
    """
    return {
        # Main content selectors (in order of preference)
        'title': ['h1', '.title', 'h1.title', 'h1.entry-title', 'h1.page-title'],
        'content': [
            'article', 
            'main', 
            '#content', 
            '.content', 
            '.entry-content',
            '.post-content',
            'div[role="main"]'
        ],
        
        # Navigation and structure
        'navigation': [
            'nav', 
            '.nav', 
            '.navigation', 
            '#nav', 
            '#navigation',
            'ul.menu',
            'div.nav-menu',
            'header nav'
        ],
        'sidebar': [
            'aside', 
            '.sidebar', 
            '#sidebar', 
            '.widget-area',
            'div[role="complementary"]'
        ],
        'footer': [
            'footer', 
            '#footer', 
            '.site-footer',
            'div[role="contentinfo"]'
        ],
        
        # Common content elements
        'headings': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
        'paragraphs': ['p', '.entry-content p', '.post-content p'],
        'images': [
            'img[src]', 
            'picture source[srcset]',
            '.wp-post-image',
            'img.size-full'
        ],
        'links': ['a[href]', 'a[href*="http"]'],
        'buttons': ['button', '.btn', '.button', 'a.button', 'input[type="submit"]'],
        'forms': ['form', '.form', '#searchform', '.search-form'],
        
        # Metadata
        'author': ['.author', '.byline', '.meta-author', '[rel="author"]'],
        'date': [
            'time', 
            '.date', 
            '.entry-date', 
            '.post-date',
            'time.entry-date',
            '[datetime]'
        ],
        'categories': ['.categories', '.post-categories', '.entry-categories', '.meta-cat'],
        'tags': ['.tags', '.post-tags', '.entry-tags', '.meta-tags'],
        
        # E-commerce specific
        'price': ['.price', '.amount', '.product-price', '.woocommerce-Price-amount'],
        'add_to_cart': [
            '.add-to-cart', 
            '.single_add_to_cart_button',
            'button[name="add-to-cart"]'
        ],
        'product_title': ['.product_title', '.product-title', '.product-name'],
        'product_description': ['.product-description', '.woocommerce-product-details__short-description'],
        'reviews': ['#reviews', '#comments', '.woocommerce-Reviews'],
        
        # Blog/News specific
        'read_more': ['.more-link', '.read-more', '.continue-reading'],
        'pagination': ['.pagination', '.page-numbers', '.pager', '.nav-links'],
        'related_posts': ['.related-posts', '.related', '.post-related'],
        'comments': ['#comments', '.comments-area', '#respond'],
        
        # Social media
        'social_links': ['.social-links', '.social-media', '.share-buttons', '[class*="social-"] a'],
        'share_buttons': ['.share-this', '.shariff', '.addtoany_share'],
        
        # Search
        'search': ['#search', '.search-form', 'form[role="search"]', '.search-box'],
        
        # Utility classes
        'breadcrumbs': ['.breadcrumbs', '.breadcrumb', '.yoast-breadcrumbs'],
        'alerts': ['.alert', '.notice', '.message', '.status', '.error'],
        'tooltips': ['[data-tooltip]', '[title]', '[data-toggle="tooltip"]']
    }

def process_single_page(
    scraper: Optional[ScraperAgent],
    url: str,
    selectors: Dict[str, str],
    wait_time: float = 2.0,
    use_proxy: bool = False,
    proxy: Optional[str] = None,
    use_ai_selectors: bool = True
) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """
    Process a single page using optimized scraping.
    
    Args:
        scraper: Optional ScraperAgent instance (kept for backward compatibility)
        url: URL to scrape
        selectors: Dictionary of CSS selectors
        wait_time: Time to wait between requests (not used in this implementation)
        use_proxy: Whether to use a proxy (not implemented in this version)
        proxy: Proxy URL if use_proxy is True (not implemented in this version)
        use_ai_selectors: Whether to use AI for selector suggestion
        
    Returns:
        Tuple of (success, result) where success is a boolean and result contains the scraped data
    """
    start_time = time.time()
    logger.info(f"Processing page: {url}")
    
    try:
        # Use our optimized scraping function
        page_data = scrape_with_selenium(url, use_ai_selectors=use_ai_selectors)
        
        # If no data found with AI selectors, try with fallback
        if not page_data.get('data') and use_ai_selectors:
            logger.info(f"No data with AI selectors, trying fallback for {url}")
            page_data = scrape_with_selenium(url, use_ai_selectors=False)
        
        if 'error' in page_data:
            error_msg = f"Failed to scrape {url}: {page_data['error']}"
            logger.warning(error_msg)
            return False, error_msg
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare the result with all available data
        result = {
            'url': url,
            'title': page_data.get('title', ''),
            'data': page_data.get('data', {}),
            'html': page_data.get('html', ''),
            'screenshot': page_data.get('screenshot'),
            'user_agent': page_data.get('user_agent', ''),
            'timestamp': page_data.get('timestamp', time.time()),
            'processing_time': processing_time,
            'selectors_used': page_data.get('selectors_used', {})
        }
        
        # Log success
        logger.info(f"Successfully processed {url} in {processing_time:.2f} seconds")
        
        return True, result
        
    except Exception as e:
        error_msg = f"Error processing {url}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return False, error_msg
    finally:
        # Ensure any cleanup happens here
        pass

def scrape_website(
    url: str,
    selectors: Optional[Dict[str, str]] = None,
    api_key: str = '',
    max_pages: int = 3,
    wait_time: float = DEFAULT_WAIT_TIME,
    use_proxy: bool = False,
    proxy_url: Optional[str] = None,
    parallel: bool = True,
    analyze_first: bool = True
) -> List[ScrapedData]:
    """
    Scrape website with the given selectors using optimized Selenium-based scraping.
    
    Args:
        url: The starting URL to scrape
        selectors: Dictionary of CSS selectors for data extraction
        api_key: API key for any required services (kept for backward compatibility)
        max_pages: Maximum number of pages to scrape
        wait_time: Time to wait between requests (seconds)
        use_proxy: Whether to use a proxy (not implemented in this version)
        proxy_url: Proxy URL if use_proxy is True (not implemented in this version)
        parallel: Whether to process pages in parallel
        
    Returns:
        List of ScrapedData objects
    """
    start_time = time.time()
    logger.info(f"Starting website scraping: {url}")
    
    try:
        # Use default selectors if none provided
        if not selectors:
            selectors = get_default_selectors()
        
        # Initialize results list
        results = []
        processed_urls = set()
        
        # Function to process a single URL
        def process_url(page_url: str, page_num: int) -> Optional[ScrapedData]:
            if page_url in processed_urls:
                return None
                
            logger.info(f"Processing page {page_num}: {page_url}")
            page_start = time.time()
            
            try:
                # Use our enhanced Selenium scraper with custom selectors
                page_data = scrape_with_selenium(
                    url=page_url,
                    use_ai_selectors=False,  # We're providing our own selectors
                    custom_selectors=selectors
                )
                
                # Check if we got any data
                if not page_data.get('data') and page_data.get('status') != 'success':
                    error_msg = page_data.get('error', 'Unknown error occurred')
                    logger.warning(f"Failed to scrape {page_url}: {error_msg}")
                    return None
                
                # Add processing time to the data
                page_data['processing_time'] = time.time() - page_start
                
                # Mark URL as processed
                processed_urls.add(page_url)
                
                # Create a ScrapedData object
                return ScrapedData(
                    data=page_data,
                    url=page_url,
                    page_num=page_num
                )
                
            except Exception as e:
                error_msg = f"Error processing {page_url}: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return None
        
        # Process the first page
        first_page = process_url(url, 1)
        if not first_page:
            logger.error(f"Failed to process first page: {url}")
            return []
            
        results = [first_page]
        
        # If we only need the first page, return early
        if max_pages <= 1:
            return results
            
        # Analyze first page to find better selectors if needed
        if analyze_first and not selectors:
            logger.info("Analyzing first page to find optimal selectors...")
            try:
                analysis = analyze_page_with_chunking(
                    url=url,
                    api_key=api_key,
                    use_selenium=True
                )
                
                if analysis and hasattr(analysis, 'suggested_selectors') and analysis.suggested_selectors:
                    # Update selectors based on analysis
                    selectors = analysis.suggested_selectors
                    logger.info(f"Updated selectors based on page analysis: {selectors}")
                    
                    # Reprocess first page with new selectors
                    first_page = process_url(url, 1)
                    if first_page:
                        results[0] = first_page
                        logger.info("Successfully reprocessed first page with optimized selectors")
                    else:
                        logger.warning("Failed to reprocess first page with optimized selectors")
                        
            except Exception as e:
                logger.warning(f"First page analysis failed, continuing with current selectors: {str(e)}")
        
        # Try to find pagination links from the first page
        pagination_links = set()  # Use a set to avoid duplicates
        
        try:
            # Parse the HTML from the first page
            soup = BeautifulSoup(first_page.data.get('html', ''), 'html.parser')
            
            # Common pagination selectors
            pagination_selectors = [
                'a[rel="next"]',
                '.pagination a',
                '.page-numbers a',
                '.pager a',
                'li.next a',
                'a.next',
                'a[class*="pagination"]',
                'a[class*="page"]',
                'a[href*="page"]',
                'a[href*="p="]'
            ]
            
            # Find all potential pagination links
            for selector in pagination_selectors:
                for link in soup.select(selector):
                    href = link.get('href', '')
                    if href and href != '#' and href != url:
                        # Make absolute URL if relative
                        if not href.startswith(('http://', 'https://')):
                            href = urljoin(url, href)
                        pagination_links.add(href)
            
            # Also generate common pagination patterns
            base_url = url.split('?')[0]  # Remove query params
            for i in range(2, max_pages + 1):
                patterns = [
                    f"{base_url}?page={i}",
                    f"{base_url}&page={i}",
                    f"{base_url}/page/{i}",
                    f"{base_url}/p/{i}",
                    f"{base_url}/page-{i}",
                    f"{base_url}?p={i}",
                    f"{base_url}&p={i}",
                    f"{base_url}/index{i}.html",
                    f"{base_url}/{i}",
                    f"{base_url}?pg={i}",
                    f"{base_url}&pg={i}",
                ]
                
                # Add some variations for common CMS patterns
                if i <= 5:  # Limit the number of variations
                    patterns.extend([
                        f"{base_url}/page/{i}/",
                        f"{base_url}/p{i}",
                        f"{base_url}/page{i}",
                        f"{base_url}/page-{i}.html",
                    ])
                
                # Add patterns to our set
                for pattern in patterns:
                    if pattern not in pagination_links:
                        pagination_links.add(pattern)
            
            logger.info(f"Found {len(pagination_links)} potential pagination links")
            
        except Exception as e:
            logger.error(f"Error finding pagination links: {str(e)}")
            # Continue with empty pagination_links if we can't find any
        
        # Process pagination links
        for i, page_url in enumerate(sorted(pagination_links), start=2):
            if len(results) >= max_pages:
                break
                
            # Skip if we've already processed this URL
            if page_url in processed_urls:
                continue
                
            page_result = process_url(page_url, i)
            if page_result:
                results.append(page_result)
                
                # Respect the wait time between requests
                if len(results) < max_pages:  # Don't wait after the last page
                    time.sleep(wait_time)
        
        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Completed scraping {len(results)} pages in {total_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        error_msg = f"Error in scrape_website: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        st.error(f"An error occurred while scraping the website: {str(e)}")
        return results  # Return whatever we have so far
    finally:
        # Ensure progress is complete
        st.session_state.scraping_progress = 1.0

def render_sidebar():
    """Render the sidebar with API configuration."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        with st.expander("API Keys", expanded=True):
            # Get current values from session state
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.gemini_api_key,
                key="gemini_api_key_input",
                help="Enter your Google Gemini API key"
            )
            
            # Update session state only if the input has changed
            if gemini_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = gemini_key
                st.success("✓ Gemini API Key saved")
                
            # Add Mistral API Key
            mistral_key = st.text_input(
                "Mistral API Key",
                type="password",
                value=st.session_state.get("mistral_api_key", ""),
                key="mistral_api_key_input",
                help="Enter your Mistral API key (required for Mistral AI features)"
            )
            
            # Update session state only if the input has changed
            if 'mistral_api_key' not in st.session_state or mistral_key != st.session_state.mistral_api_key:
                st.session_state.mistral_api_key = mistral_key
                if mistral_key:  # Only show success if a key was entered
                    st.success("✓ Mistral API Key saved")
                
            # Add Crawl4AI configuration if available
            if CRAWL4AI_AVAILABLE:
                st.markdown("---")
                st.markdown("### Crawl4AI Settings")
                st.info("Crawl4AI is available and ready to use!")
            
            pinecone_key = st.text_input(
                "Pinecone API Key (Optional)",
                type="password",
                value=st.session_state.pinecone_api_key,
                key="pinecone_api_key_input",
                help="Optional: For vector search functionality"
            )
            
            # Update session state only if the input has changed
            if pinecone_key != st.session_state.pinecone_api_key:
                st.session_state.pinecone_api_key = pinecone_key
        
        with st.expander("Scraper Settings", expanded=False):
            # Default Wait Time
            wait_time = st.number_input(
                "Default Wait Time (s)",
                min_value=0.5,
                max_value=30.0,
                value=st.session_state.default_wait_time,
                step=0.5,
                key="default_wait_time_input"
            )
            
            # Update session state if changed
            if wait_time != st.session_state.default_wait_time:
                st.session_state.default_wait_time = wait_time
            
            # Use Proxy Checkbox
            use_proxy = st.checkbox(
                "Use Proxy",
                value=st.session_state.use_proxy,
                key="use_proxy_checkbox",
                help="Enable if you need to use a proxy"
            )
            
            # Update session state if changed
            if use_proxy != st.session_state.use_proxy:
                st.session_state.use_proxy = use_proxy
            
            # Proxy URL Input
            if st.session_state.use_proxy:
                proxy_url = st.text_input(
                    "Proxy URL",
                    value=st.session_state.proxy_url,
                    key="proxy_url_input",
                    placeholder="http://username:password@proxy:port",
                    help="Format: http://user:pass@host:port"
                )
                
                # Update session state if changed
                if proxy_url != st.session_state.proxy_url:
                    st.session_state.proxy_url = proxy_url

def display_selector_suggestion(selector: Dict[str, str], index: int) -> None:
    """Display a single selector suggestion with interactive elements."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.text_input(
                "Field",
                value=selector.get('name', ''),
                key=f"field_{index}_{selector.get('name', '')}",
                disabled=True
            )
        with col2:
            st.text_input(
                "Selector",
                value=selector.get('selector', ''),
                key=f"selector_{index}_{selector.get('selector', '')}",
                disabled=True
            )
        
        # Add columns for action buttons
        btn_col1, btn_col2, _ = st.columns([1, 1, 3])
        
        with btn_col1:
            if st.button("📋 Use", key=f"use_{index}_{selector.get('name', '')}"):
                field_name = selector.get('name', '').lower().replace(' ', '_')
                st.session_state.selectors[field_name] = selector.get('selector', '')
                st.success(f"Added selector for {field_name}")
                st.rerun()
        
        with btn_col2:
            if st.button("🔍 Test", key=f"test_{index}_{selector.get('name', '')}"):
                st.session_state.testing_selector = {
                    'name': selector.get('name', ''),
                    'selector': selector.get('selector', '')
                }
                st.rerun()
        
        # Show additional info if available
        if 'description' in selector or 'purpose' in selector:
            with st.expander("ℹ️ Details", expanded=False):
                if 'description' in selector:
                    st.write(f"**Description:** {selector['description']}")
                if 'purpose' in selector:
                    st.write(f"**Purpose:** {selector['purpose']}")
                if 'confidence' in selector:
                    st.progress(selector.get('confidence', 0) / 5, 
                               text=f"Confidence: {selector.get('confidence', 0)}/5")
        
        st.markdown("---")

def render_analysis_tab():
    """Render the analysis tab with first-page analysis."""
    st.header("🔍 First-Page Analysis")
    st.markdown("""
    Analyze a webpage to automatically detect content, suggest selectors, and understand the page structure.
    This analysis uses AI to identify key elements and provide detailed information about the page.
    """)
    
    # URL Input and Analysis Options
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            analyze_url = st.text_input(
                "Enter URL to analyze",
                value=st.session_state.get('analyze_url', ''),
                key="analyze_url_input",
                placeholder="https://example.com",
                help="Enter the URL of the page you want to analyze"
            )
            
            # Update session state if URL changed
            if analyze_url != st.session_state.get('analyze_url'):
                st.session_state.analyze_url = analyze_url
                # Clear previous results when URL changes
                if 'analysis_results' in st.session_state:
                    del st.session_state.analysis_results
        
        with col2:
            st.markdown("##")
            use_selenium = st.checkbox(
                "Use Selenium",
                value=st.session_state.get('use_selenium', False),
                key="use_selenium_checkbox",
                help="Use Selenium for dynamic content (slower but more accurate)"
            )
            
            analyze_button = st.button("🔍 Analyze Page", type="primary", key="analyze_page_button", use_container_width=True)
            
            if analyze_button:
                if not analyze_url or not is_valid_url(analyze_url):
                    st.error("Please enter a valid URL")
                elif 'mistral_api_key' not in st.session_state or not st.session_state.mistral_api_key:
                    st.error("Please enter your Mistral API key in the sidebar")
                else:
                    with st.spinner("Analyzing page content (this may take a minute)..."):
                        analysis_result = analyze_page_with_chunking(
                            analyze_url,
                            st.session_state.mistral_api_key,
                            use_selenium=use_selenium
                        )
                        if analysis_result:
                            st.session_state.analysis_results = analysis_result
                            st.session_state.last_analysis_time = time.time()
                            st.rerun()
    
    # Display analysis results if available
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        analysis = st.session_state.analysis_results
        
        if 'last_analysis_time' in st.session_state:
            st.caption(f"Last analyzed: {time.ctime(st.session_state.last_analysis_time)}")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Overview",
            "🧩 Elements",
            "📊 Combined Analysis",
            "⚙️ Technical Details"
        ])
        
        with tab1:  # Overview
            st.subheader("Page Overview")
            
            # Safely get values with defaults
            url = analysis.get('url', 'N/A')
            title = analysis.get('title', 'No title available')
            combined = analysis.get('combined_analysis', {})
            
            st.metric("URL", url)
            st.metric("Title", title)
            st.metric("Total Chunks", combined.get("total_chunks", 0))
            
            processing_time = combined.get('processing_time', 0)
            st.metric("Processing Time", f"{processing_time:.2f} seconds")
            
            if TOTAL_REQUESTS_TODAY > 40:
                st.warning("Approaching daily limit of 50 requests!")
        
        with tab2:  # Elements
            st.subheader("Detected Elements")
            
            # Check for elements in both the root and combined_analysis
            elements = []
            combined = analysis.get("combined_analysis", {})
            
            # Check for elements in combined_analysis first
            if 'elements' in combined and isinstance(combined['elements'], list):
                elements = combined['elements']
            # Fall back to root level elements if not found in combined_analysis
            elif 'elements' in analysis and isinstance(analysis['elements'], list):
                elements = analysis['elements']
            
            if elements:
                try:
                    # Convert elements to DataFrame for better display
                    df = pd.DataFrame(elements)
                    
                    # Display elements in a table with expandable details
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "type": "Element Type",
                            "selector": "CSS Selector",
                            "description": "Description"
                        },
                        hide_index=True
                    )
                    
                    # Show raw JSON in expander for debugging
                    with st.expander("View Raw Elements Data"):
                        st.json(elements)
                        
                except Exception as e:
                    st.error(f"Error displaying elements: {str(e)}")
                    st.json(elements)  # Show raw data if dataframe conversion fails
            else:
                st.info("No elements detected in the page analysis.")
                
                # Debug information
                with st.expander("Debug: Analysis Data"):
                    st.json({
                        'has_combined_analysis': 'combined_analysis' in analysis,
                        'has_elements_in_combined': 'elements' in combined if 'combined_analysis' in analysis else False,
                        'has_elements_in_root': 'elements' in analysis
                    })
            
            # Handle forms if they exist
            forms = []
            if 'forms' in combined and isinstance(combined['forms'], list):
                forms = combined['forms']
            elif 'forms' in analysis and isinstance(analysis['forms'], list):
                forms = analysis['forms']
                
            if forms:
                with st.expander("📋 Detected Forms", expanded=False):
                    for i, form in enumerate(forms, 1):
                        action = form.get('action', 'N/A')
                        method = form.get('method', 'get').upper()
                        st.markdown(f"**Form {i}** (Action: `{action}`, Method: `{method}`)")
                        
                        # Display form fields
                        fields = form.get("fields", [])
                        if fields:
                            st.markdown("**Fields:**")
                            for field in fields:
                                field_name = field.get('name', 'unnamed')
                                field_type = field.get('type', 'text')
                                field_id = field.get('id', '')
                                field_placeholder = field.get('placeholder', '')
                                field_required = ' (required)' if field.get('required') else ''
                                
                                field_info = f"- `{field_name}` ({field_type}{field_required})"
                                if field_id:
                                    field_info += f" [ID: {field_id}]"
                                if field_placeholder:
                                    field_info += f" - {field_placeholder}"
                                    
                                st.markdown(field_info)
                        else:
                            st.markdown("*No form fields detected.*")
                            
                        st.markdown("---")
            
            # Debug information section
            with st.expander("🔍 Debug Information", expanded=False):
                st.subheader("Analysis Structure")
                st.json({
                    'keys_in_analysis': list(analysis.keys()) if isinstance(analysis, dict) else 'Not a dictionary',
                    'has_combined_analysis': 'combined_analysis' in analysis,
                    'has_elements': 'elements' in analysis,
                    'has_forms': 'forms' in analysis,
                    'has_navigation': 'navigation' in analysis
                })
                
                st.subheader("Sample Data")
                st.json({
                    'sample_elements': analysis.get('elements', [])[:2] if 'elements' in analysis else 'No elements',
                    'sample_forms': analysis.get('forms', [])[:1] if 'forms' in analysis else 'No forms',
                    'sample_navigation': analysis.get('navigation', [])[:3] if 'navigation' in analysis else 'No navigation'
                })

        with tab4:  # Technical Details
            st.subheader("Technical Details")
            
            # Show raw analysis data
            with st.expander("View Raw Analysis Data", expanded=False):
                st.json(analysis)
            
            # Show processing information
            st.subheader("Processing Information")
            
            # Show chunk information if available
            if 'chunk_analyses' in analysis and analysis['chunk_analyses']:
                st.metric("Chunks Processed", len(analysis['chunk_analyses']))
                
                # Show chunk details in an expander
                with st.expander("View Chunk Details", expanded=False):
                    for i, chunk in enumerate(analysis['chunk_analyses'], 1):
                        with st.expander(f"Chunk {i}", expanded=False):
                            st.json(chunk)
            
            # Show any errors or warnings
            if 'errors' in analysis and analysis['errors']:
                st.error("Processing Errors")
                for error in analysis['errors']:
                    st.error(error)
            
            # Show performance metrics if available
            if 'performance_metrics' in analysis:
                st.subheader("Performance Metrics")
                metrics = analysis['performance_metrics']
                cols = st.columns(3)
                for i, (key, value) in enumerate(metrics.items()):
                    with cols[i % 3]:
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}s" if isinstance(value, (int, float)) else str(value))
            
            # Handle forms if they exist
            forms = combined.get("forms", [])
            if forms:
                with st.expander("📋 Detected Forms", expanded=False):
                    for form in forms:
                        action = form.get('action', 'N/A')
                        st.markdown(f"**Form (Action: {action})**")
                        for field in form.get("fields", []):
                            st.markdown(f"- {field.get('name', 'unnamed')} ({field.get('type', 'text')})")
            
            # Navigation links
            navigation_links = combined.get("navigation", [])
            if navigation_links:
                with st.expander("🔗 Navigation Links", expanded=False):
                    for link in navigation_links:
                        st.markdown(f"- {link}")
        
        with tab3:  # Combined Analysis
            st.subheader("Combined Analysis")
            
            # Safely get combined_analysis with defaults
            combined = analysis.get("combined_analysis", {})
            
            if combined:
                # Show a more user-friendly summary
                st.subheader("Analysis Summary")
                
                # Create columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Elements summary
                    elements = combined.get("elements", [])
                    st.metric("Elements Found", len(elements))
                
                with col2:
                    # Forms summary
                    forms = combined.get("forms", [])
                    st.metric("Forms Found", len(forms))
                
                with col3:
                    # Navigation summary
                    nav_links = combined.get("navigation", [])
                    st.metric("Navigation Links", len(nav_links))
                
                # Processing time
                proc_time = combined.get("processing_time", 0)
                st.metric("Processing Time", f"{proc_time:.2f} seconds")
                
                # Show raw data in expander
                with st.expander("View Raw Combined Analysis", expanded=False):
                    st.json(combined)
            else:
                st.warning("No combined analysis data available. The analysis may have failed or is still in progress.")
                if 'error' in analysis:
                    st.error(f"Error during analysis: {analysis['error']}")
            
            # Show forms if available
            if 'forms' in analysis.get('combined_analysis', {}) and analysis['combined_analysis']['forms']:
                with st.expander("📝 Detected Forms", expanded=False):
                    for i, form in enumerate(analysis['combined_analysis']['forms']):
                        st.markdown(f"**Form {i+1}**")
                        st.code(f"Action: {form.get('action', 'N/A')}\n"
                              f"Method: {form.get('method', 'get').upper()}")
                        
                        if form.get('fields'):
                            st.markdown("**Fields:**")
                            for field in form.get('fields', []):
                                st.markdown(f"- `{field.get('name', 'unnamed')} `"
                                          f"({field.get('type', 'text')})" +
                                          (" 🔴" if field.get('required') else ""))
        
        with tab4:  # Technical Details
            st.subheader("Technical Information")
            
            # Technologies
            if 'technologies' in analysis.get('combined_analysis', {}) and analysis['combined_analysis']['technologies']:
                st.markdown("#### Detected Technologies")
                st.write(", ".join(analysis['combined_analysis']['technologies']))
            
            # Accessibility
            if 'accessibility' in analysis.get('combined_analysis', {}) and analysis['combined_analysis']['accessibility']:
                with st.expander("♿ Accessibility Analysis", expanded=False):
                    accessibility = analysis['combined_analysis']['accessibility']
                    if 'score' in accessibility:
                        score = accessibility['score']
                        st.metric("Accessibility Score", f"{score}/5")
                        st.progress(score/5)
                    
                    if 'issues' in accessibility and accessibility['issues']:
                        st.markdown("**Potential Issues:**")
                        for issue in accessibility['issues']:
                            st.markdown(f"- {issue}")
            
            # Pagination Info
            if 'pagination_info' in analysis.get('combined_analysis', {}) and analysis['combined_analysis']['pagination_info']:
                with st.expander("🔢 Pagination Details", expanded=False):
                    st.json(analysis['combined_analysis']['pagination_info'])
            
            # Raw analysis data
            with st.expander("📄 View Raw Analysis", expanded=False):
                st.json(analysis)

def render_scraper_tab():
    """Render the main scraping tab."""
    st.header("🛠️ Web Scraper")
    
    # URL Input
    url = st.text_input(
        "Enter URL to scrape",
        value=st.session_state.get('scrape_url', ''),
        key="scrape_url_input",
        placeholder="https://example.com",
        help="Enter the URL you want to scrape"
    )
    
    # Update session state if URL changed
    if url != st.session_state.get('scrape_url'):
        st.session_state.scrape_url = url
    
    # Scraper Configuration
    with st.expander("⚙️ Scraper Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Max Pages
            max_pages = st.number_input(
                "Max Pages to Scrape",
                min_value=1,
                max_value=MAX_PAGES,
                value=st.session_state.get('max_pages', 3),
                key="max_pages_input"
            )
            
            # Update session state if changed
            if max_pages != st.session_state.get('max_pages'):
                st.session_state.max_pages = max_pages
            
            # Wait Time
            wait_time = st.number_input(
                "Wait Time (seconds)",
                min_value=0.5,
                max_value=30.0,
                value=st.session_state.default_wait_time,
                step=0.5,
                key="scraper_wait_time_input"
            )
            
            # Update session state if changed
            if wait_time != st.session_state.default_wait_time:
                st.session_state.default_wait_time = wait_time
            
            # Parallel Processing Option
            parallel_processing = st.checkbox(
                "Enable Parallel Processing", 
                value=st.session_state.get('parallel_processing', True),
                key="parallel_processing_checkbox"
            )
            
            # Update session state if changed
            if parallel_processing != st.session_state.get('parallel_processing'):
                st.session_state.parallel_processing = parallel_processing
        
        with col2:
            # Use Proxy Checkbox
            use_proxy = st.checkbox(
                "Use Proxy",
                value=st.session_state.use_proxy,
                key="scraper_use_proxy_checkbox"
            )
            
            # Update session state if changed
            if use_proxy != st.session_state.use_proxy:
                st.session_state.use_proxy = use_proxy
            
            proxy = ""
            if st.session_state.use_proxy:
                proxy = st.text_input(
                    "Proxy URL",
                    value=st.session_state.proxy_url,
                    key="scraper_proxy_url_input",
                    help="Format: http://user:pass@host:port"
                )
                
                # Update session state if changed
                if proxy != st.session_state.proxy_url:
                    st.session_state.proxy_url = proxy
    
    # Selector Management
    st.subheader("🔧 Selector Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get current values from session state
        selector_name = st.text_input(
            "Field Name",
            value=st.session_state.get('new_selector_name', ''),
            key="new_selector_name_input",
            placeholder="e.g., title, price, description",
            help="Descriptive name for the data you want to extract"
        )
        
        # Update session state if changed
        if selector_name != st.session_state.get('new_selector_name'):
            st.session_state.new_selector_name = selector_name
        
        selector_value = st.text_input(
            "CSS Selector",
            value=st.session_state.get('new_selector_value', ''),
            key="new_selector_value_input",
            placeholder="e.g., h1, .product-title, #price",
            help="CSS selector for the element you want to extract"
        )
        
        # Update session state if changed
        if selector_value != st.session_state.get('new_selector_value'):
            st.session_state.new_selector_value = selector_value
        
        add_selector = st.button("Add Selector", key="add_selector_button")
        
        if add_selector:
            if not selector_name or not selector_value:
                st.error("Both field name and selector are required")
            else:
                field_name = selector_name.lower().replace(' ', '_')
                # Create a copy to trigger a rerender
                selectors = st.session_state.selectors.copy()
                selectors[field_name] = selector_value
                st.session_state.selectors = selectors
                st.success(f"Added selector: {field_name}")
                # Clear the input fields
                st.session_state.new_selector_name = ""
                st.session_state.new_selector_value = ""
                # Force a rerun to update the UI
                st.rerun()
    
    with col2:
        st.markdown("### Current Selectors")
        if st.session_state.selectors:
            for name, selector in list(st.session_state.selectors.items()):
                st.code(f"{name}: {selector}")
                remove_button = st.button(f"Remove {name}", key=f"remove_{name}_button")
                if remove_button:
                    # Create a copy to trigger a rerender
                    selectors = st.session_state.selectors.copy()
                    if name in selectors:
                        del selectors[name]
                        st.session_state.selectors = selectors
                        st.success(f"Removed selector: {name}")
                        # Force a rerun to update the UI
                        st.rerun()
        else:
            st.info("No selectors added yet")
    
    # Start Scraping Button
    if st.button("🚀 Start Scraping", type="primary", use_container_width=True):
        if not url or not is_valid_url(url):
            st.error("Please enter a valid URL")
        elif not st.session_state.selectors:
            st.error("Please add at least one selector")
        elif 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
            st.error("Please enter your Gemini API key in the sidebar")
        else:
            try:
                # Initialize the scraper once
                scraper = ScraperAgent(api_key=st.session_state.gemini_api_key)
                
                with st.spinner("Scraping in progress..."):
                    results = []
                    urls_to_scrape = [url]
                    
                    # First, collect all URLs if pagination is detected
                    if max_pages > 1:
                        with st.spinner("Discovering pagination..."):
                            current_url = url
                            for _ in range(max_pages - 1):
                                try:
                                    # Just get the next URL without full scraping
                                    result = scraper.scrape_page(
                                        url=current_url,
                                        selectors={'next_page': 'a[rel=next], .next, a.next, .pagination-next, a.pagination-next'},
                                        wait_time=wait_time / 2,  # Faster for discovery
                                        use_proxy=use_proxy,
                                        proxy=proxy if use_proxy else None,
                                        headless=True  # If your scraper supports headless mode
                                    )
                                    
                                    if result and result.get('next_page'):
                                        next_url = result['next_page']
                                        if next_url and is_valid_url(next_url) and next_url not in urls_to_scrape:
                                            urls_to_scrape.append(next_url)
                                            current_url = next_url
                                        else:
                                            break
                                    else:
                                        break
                                except Exception as e:
                                    logger.warning(f"Error discovering pagination: {str(e)}")
                                    break
                    
                    # Limit to max_pages
                    urls_to_scrape = urls_to_scrape[:max_pages]
                    
                    # Initialize progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Scrape pages in parallel
                        parallel = st.session_state.get('parallel_processing', True) and len(urls_to_scrape) > 1
                        
                        if parallel:
                            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(urls_to_scrape))) as executor:
                                # Submit all scraping tasks
                                future_to_url = {
                                    executor.submit(
                                        process_single_page,
                                        scraper,
                                        url,
                                        st.session_state.selectors,
                                        wait_time,
                                        use_proxy,
                                        proxy if use_proxy else None
                                    ): url for url in urls_to_scrape
                                }
                                
                                # Process completed tasks
                                completed = 0
                                total = len(urls_to_scrape)
                                
                                for future in as_completed(future_to_url):
                                    try:
                                        success, result = future.result()
                                        if success and result:
                                            results.append(result)
                                        completed += 1
                                        # Update progress
                                        progress = completed / total
                                        progress_bar.progress(min(progress, 1.0))
                                        status_text.text(f"Scraped {completed}/{total} pages...")
                                        
                                    except Exception as e:
                                        logger.error(f"Error processing page: {str(e)}")
                                        completed += 1
                                        progress = completed / total
                                        progress_bar.progress(min(progress, 1.0))
                                        status_text.text(f"Error on page {completed}/{total}")
                        else:
                            # Fallback to sequential scraping
                            total = len(urls_to_scrape)
                            for i, url in enumerate(urls_to_scrape, 1):
                                try:
                                    status_text.text(f"Scraping page {i}/{total}...")
                                    success, result = process_single_page(
                                        scraper,
                                        url,
                                        st.session_state.selectors,
                                        wait_time,
                                        use_proxy,
                                        proxy if use_proxy else None
                                    )
                                    if success and result:
                                        results.append(result)
                                    
                                    # Update progress
                                    progress = i / total
                                    progress_bar.progress(min(progress, 1.0))
                                    status_text.text(f"Scraped {i}/{total} pages...")
                                    
                                except Exception as e:
                                    logger.error(f"Error processing page {i}: {str(e)}")
                                    progress = i / total
                                    progress_bar.progress(min(progress, 1.0))
                                    status_text.text(f"Error on page {i}/{total}")
                                    continue
                    except Exception as e:
                        st.error(f"An error occurred during scraping: {str(e)}")
                        logger.error(f"Scraping error: {str(e)}\n{traceback.format_exc()}")
                    finally:
                        # Ensure progress bar is complete
                        progress_bar.progress(1.0)
                        status_text.text(f"Completed: Scraped {len(results)}/{len(urls_to_scrape)} pages")
                    
                # Process results after scraping is complete
                if results:
                    st.session_state.scraping_results = results
                    st.success(f"✅ Successfully scraped {len(results)} items!")
                    
                    # Convert results to DataFrame for display
                    try:
                        df = pd.json_normalize([r for r in results if r])
                        
                        # Display results in a scrollable container
                        with st.expander("📋 View Scraped Data", expanded=True):
                            st.dataframe(df, use_container_width=True)
                            
                        # Download button for results
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv',
                            key='download_csv'
                        )
                    except Exception as e:
                        st.error(f"Error processing results: {str(e)}")
                        logger.error(f"Error processing results: {str(e)}")
                else:
                    st.warning("No data was scraped. Please check your selectors and try again.")
                    
            except Exception as e:
                st.error(f"An error occurred during scraping: {str(e)}")
                logger.error(f"Scraping error: {str(e)}\n{traceback.format_exc()}")

def render_crawl4ai_tab():
    """Render the Crawl4AI tab."""
    st.header("🤖 Crawl4AI Scraper")
    
    if not CRAWL4AI_AVAILABLE:
        st.error("Crawl4AI scraper is not available. Please check the installation.")
        return
    
    st.markdown("""
    Use Crawl4AI to extract structured data from web pages with AI-powered extraction.
    This scraper can handle dynamic content and provides better handling of modern websites.
    """)
    
    # URL input
    url = st.text_input(
        "Enter URL to scrape with Crawl4AI",
        value=st.session_state.get('crawl4ai_url', ''),
        key="crawl4ai_url_input",
        placeholder="https://example.com",
        help="Enter the URL you want to scrape with Crawl4AI"
    )
    
    # Update session state if URL changed
    if url != st.session_state.get('crawl4ai_url'):
        st.session_state.crawl4ai_url = url
    
    # Scraping options
    with st.expander("⚙️ Crawl4AI Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            max_pages = st.number_input(
                "Max Pages to Scrape",
                min_value=1,
                max_value=10,
                value=1,
                key="crawl4ai_max_pages"
            )
            
            extract_structured = st.checkbox(
                "Extract Structured Data",
                value=True,
                key="crawl4ai_extract_structured",
                help="Use AI to extract structured data from the page"
            )
        
        with col2:
            include_links = st.checkbox(
                "Include Links",
                value=True,
                key="crawl4ai_include_links",
                help="Include links in the scraped data"
            )
            
            use_js = st.checkbox(
                "Use JavaScript",
                value=True,
                key="crawl4ai_use_js",
                help="Enable JavaScript rendering (slower but more accurate for dynamic sites)"
            )
    
    # Start scraping button
    if st.button("🚀 Start Crawl4AI Scraping", type="primary", use_container_width=True):
        if not url or not is_valid_url(url):
            st.error("Please enter a valid URL")
        else:
            with st.spinner("Scraping with Crawl4AI..."):
                try:
                    # Initialize the scraper
                    scraper = Crawl4AIScraper()
                    
                    # Run the scraper
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    if extract_structured:
                        # Define a schema for structured data extraction
                        schema = {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "The title of the page"},
                                "summary": {"type": "string", "description": "A brief summary of the main content"},
                                "key_topics": {
                                    "type": "array", 
                                    "items": {"type": "string"}, 
                                    "description": "Main topics covered in the article"
                                },
                                "technologies_mentioned": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Web technologies or libraries mentioned"
                                },
                                "main_content": {
                                    "type": "string",
                                    "description": "The main content of the page"
                                }
                            },
                            "required": ["title", "summary"]
                        }
                        
                        # Extract structured data
                        structured_data = loop.run_until_complete(
                            scraper.extract_structured_data(url=url, schema=schema)
                        )
                        
                        # Store results in session state
                        st.session_state.crawl4ai_results = {
                            'type': 'structured',
                            'data': structured_data,
                            'url': url,
                            'timestamp': pd.Timestamp.now().isoformat()
                        }
                        
                        st.success("✅ Successfully extracted structured data with Crawl4AI!")
                        
                    else:
                        # Regular scraping with selectors
                        selectors = {
                            'title': 'h1',
                            'content': 'main, article, .content, #content',
                            'links': 'a[href]'
                        }
                        
                        results = loop.run_until_complete(
                            scraper.scrape_pages(
                                url=url,
                                selectors=selectors,
                                max_pages=max_pages,
                                show_progress=False
                            )
                        )
                        
                        # Store results in session state
                        if results:
                            st.session_state.crawl4ai_results = {
                                'type': 'scraped',
                                'data': results,
                                'url': url,
                                'timestamp': pd.Timestamp.now().isoformat()
                            }
                            st.success(f"✅ Successfully scraped {len(results)} pages with Crawl4AI!")
                        else:
                            st.warning("No data was scraped. Please check the URL and try again.")
                    
                    # Close the event loop
                    loop.close()
                    
                except Exception as e:
                    st.error(f"Error during Crawl4AI scraping: {str(e)}")
                    logger.error(f"Crawl4AI error: {str(e)}\n{traceback.format_exc()}")
    
    # Display results if available
    if 'crawl4ai_results' in st.session_state and st.session_state.crawl4ai_results:
        results = st.session_state.crawl4ai_results
        
        st.subheader("📊 Crawl4AI Results")
        st.write(f"URL: {results['url']}")
        st.write(f"Scraped at: {pd.to_datetime(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        if results['type'] == 'structured':
            # Display structured data
            st.subheader("Structured Data")
            
            # Display title and summary
            if 'title' in results['data']:
                st.markdown(f"### {results['data']['title']}")
            
            if 'summary' in results['data']:
                st.markdown("#### Summary")
                st.write(results['data']['summary'])
            
            # Display key topics if available
            if 'key_topics' in results['data'] and results['data']['key_topics']:
                st.markdown("#### Key Topics")
                for i, topic in enumerate(results['data']['key_topics'][:10], 1):
                    st.write(f"{i}. {topic}")
            
            # Display main content if available
            if 'main_content' in results['data'] and results['data']['main_content']:
                with st.expander("View Full Content", expanded=False):
                    st.markdown(results['data']['main_content'])
            
            # Display technologies mentioned if available
            if 'technologies_mentioned' in results['data'] and results['data']['technologies_mentioned']:
                st.markdown("#### Technologies Mentioned")
                st.write(", ".join(results['data']['technologies_mentioned']))
            
        else:
            # Display scraped data
            st.subheader("Scraped Data")
            
            # Convert to DataFrame for better display
            df = pd.json_normalize(results['data'])
            
            # Display the first few rows
            st.dataframe(df.head())
            
            # Add download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download as CSV",
                data=csv,
                file_name=f"crawl4ai_scraped_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def render_results_tab():
    """Render the results tab."""
    st.header("📊 Results")
    
    if 'scraping_results' in st.session_state and st.session_state.scraping_results:
        results = st.session_state.scraping_results
        df = pd.json_normalize([r for r in results if r])
        
        # Display results
        st.dataframe(df)
        
        # Add download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download as CSV",
            data=csv,
            file_name="scraped_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No scraping results available. Run a scrape first!")

# --- Main App ---
def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🕸️ Advanced Web Scraper")
    st.markdown("Extract data from any website with AI-powered analysis and custom selectors.")
    
    # Create tabs based on Crawl4AI availability
    tab_titles = (["🔍 Analyze", "🛠️ Scrape", "🤖 Crawl4AI", "📊 Results"] 
                 if CRAWL4AI_AVAILABLE 
                 else ["🔍 Analyze", "🛠️ Scrape", "📊 Results"])
    tabs = st.tabs(tab_titles)
    
    with tabs[0]:  # Analyze tab
        render_analysis_tab()
    
    with tabs[1]:  # Scrape tab
        render_scraper_tab()
    
    # Add Crawl4AI tab if available
    if CRAWL4AI_AVAILABLE:
        with tabs[2]:  # Crawl4AI tab
            render_crawl4ai_tab()
        
        with tabs[3]:  # Results tab
            render_results_tab()
    else:
        with tabs[2]:  # Results tab
            render_results_tab()
    
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2em;">
            <p>Advanced Web Scraper v2.0 | Built with Streamlit and Gemini AI</p>
            <p>© 2023-2024 Web Scraper Pro. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Set environment variables before any imports
    import os
    os.environ['PYTORCH_JIT'] = '0'  # Disable PyTorch JIT
    
    # Configure asyncio for Windows if needed
    import sys
    import asyncio
    
    if sys.platform == 'win32':
        if sys.version_info >= (3, 8):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Import Streamlit
    import streamlit as st
    
    # Run the main function
    main()
