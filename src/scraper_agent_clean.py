from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException,
    InvalidSessionIdException, StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager
from typing import List, Dict, Any, Optional, Callable, TypeVar, Type, Tuple, Union
import time
import json
import re
import random
import hashlib
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import os
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse
from fake_useragent import UserAgent
import google.generativeai as genai
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log')
    ]
)
logger = logging.getLogger(__name__)

# Type variables for type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def retry_on_session_error(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (WebDriverException, TimeoutException, InvalidSessionIdException, StaleElementReferenceException)
) -> Callable[[F], F]:
    """Decorator to retry a function when session errors occur."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
                    
                    # If it's a session error, try to reinitialize the WebDriver
                    if any(err in str(e).lower() for err in ["invalid session id", "session not found"]):
                        logger.info("Session error detected, attempting to reinitialize WebDriver...")
                        if args and hasattr(args[0], '_driver'):
                            try:
                                if args[0]._driver is not None:
                                    args[0]._driver.quit()
                            except Exception as e:
                                logger.warning(f"Error quitting old WebDriver: {str(e)}")
                            args[0]._driver = None
                    
                    if attempt < max_retries:
                        wait_time = delay * (2 ** (attempt - 1))  # Exponential backoff
                        logger.info(f"Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries ({max_retries}) reached. Last error: {str(e)}")
                        raise
            
            if last_exception:
                raise last_exception
            raise Exception("Unknown error in retry decorator")
        return wrapper
    return decorator

class CacheManager:
    """Simple file-based cache manager for storing scraped data."""
    
    def __init__(self, cache_dir: str = ".scraper_cache", ttl: int = 3600):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.ttl = timedelta(seconds=ttl)
        logger.info(f"Cache initialized at {self.cache_dir.absolute()}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key."""
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and hasn't expired."""
        if not self.cache_dir.exists():
            return None
            
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            logger.debug(f"Cache miss for key: {key}")
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                timestamp, data = pickle.load(f)
                
            # Check if cache is expired
            if datetime.now() - timestamp > self.ttl:
                logger.debug(f"Cache expired for key: {key}")
                return None
                
            logger.debug(f"Cache hit for key: {key}")
            return data
            
        except Exception as e:
            logger.warning(f"Error reading cache for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        try:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            cache_file = self._get_cache_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump((datetime.now(), value), f)
            logger.debug(f"Cached data for key: {key}")
        except Exception as e:
            logger.error(f"Error writing to cache for key {key}: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        try:
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Error deleting cache file {cache_file}: {str(e)}")
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

# Define models if not available
try:
    from .models import SelectorSuggestion, ScrapedDataInput
except ImportError:
    @dataclass
    class SelectorSuggestion:
        field: str
        selector: str
        sample_data: Optional[str]
        confidence: float
        
    @dataclass
    class ScrapedDataInput:
        data: Dict[str, Any]
        url: str
        page_num: int
        
        def dict(self):
            return {"data": self.data, "url": self.url, "page_num": self.page_num}

class ScraperAgent:
    """A web scraping agent with caching, parallel processing, and error handling."""
    
    def __init__(
        self, 
        api_key: str = "", 
        model: str = "gemini-1.5-flash", 
        headless: bool = True,
        cache_enabled: bool = True,
        max_workers: int = 3,
        request_timeout: int = 30,
        user_agent: Optional[str] = None,
        cache_ttl: int = 3600
    ):
        """Initialize the ScraperAgent with configuration options.
        
        Args:
            api_key: Google Gemini API key (optional if not using AI features)
            model: Name of the Gemini model to use (default: "gemini-1.5-flash")
            headless: Whether to run the WebDriver in headless mode (default: True)
            cache_enabled: Whether to enable caching of scraped data (default: True)
            max_workers: Maximum number of concurrent workers for parallel processing (default: 3)
            request_timeout: Default timeout for web requests in seconds (default: 30)
            user_agent: Custom user agent string (randomized if not provided)
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.headless = headless
        self.max_workers = max(1, min(max_workers, 10))  # Limit between 1-10 workers
        self.request_timeout = max(10, min(request_timeout, 300))  # 10s to 5min
        self.user_agent = user_agent or self.get_random_user_agent()
        
        # Initialize cache if enabled
        self.cache_enabled = cache_enabled
        self.cache = CacheManager(ttl=cache_ttl) if cache_enabled else None
        
        # Initialize Gemini API if API key is provided
        if api_key:
            self._init_gemini(api_key, model)
        else:
            self.model = None
            logger.info("Gemini API not initialized (no API key provided)")
        
        # Initialize WebDriver components
        self.chrome_service = ChromeService(ChromeDriverManager().install())
        self.By = By  # Make By available as instance attribute
        self._driver = None  # Will hold the WebDriver instance
        self._driver_lock = threading.Lock()  # Thread lock for WebDriver access
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"ScraperAgent initialized with {self.max_workers} workers")
    
    def _init_gemini(self, api_key: str, model: str) -> None:
        """Initialize the Gemini model with error handling."""
        if not api_key or not isinstance(api_key, str):
            raise ValueError("A valid Gemini API key is required")
            
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_output_tokens": 1000
                }
            )
            # Test the connection
            self.model.generate_content("Test connection")
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise ConnectionError(f"Failed to initialize Gemini API: {str(e)}") from e

    @retry_on_session_error(max_retries=3, delay=1)
    def get_random_user_agent(self) -> str:
        """Generate a random user agent to avoid detection.
        
        Returns:
            A random user agent string
            
        Note:
            Uses fake-useragent library with fallback to a list of common user agents.
        """
        try:
            ua = UserAgent()
            user_agent = ua.random
            logger.debug(f"Generated random user agent: {user_agent}")
            return user_agent
        except Exception as e:
            logger.warning(f"Error generating random user agent: {str(e)}. Using fallback.")
            # Fallback user agents (updated to more recent versions)
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
            ]
            return random.choice(user_agents)
    
    @retry_on_session_error(max_retries=3, delay=1)
    def _init_webdriver(self, proxy: Optional[str] = None, timeout: int = 30) -> webdriver.Chrome:
        """Initialize and return a configured WebDriver instance with improved session management.
        
        Args:
            proxy: Optional proxy server in format 'host:port'
            timeout: Page load timeout in seconds
            
        Returns:
            Configured Chrome WebDriver instance
            
        Raises:
            WebDriverException: If WebDriver initialization fails
        """
        with self._driver_lock:
            # Reuse existing driver if available and session is valid
            if self._driver is not None:
                try:
                    # Check if session is still valid
                    self._driver.current_url  # Will raise if session is invalid
                    logger.debug("Reusing existing WebDriver session")
                    return self._driver
                except (WebDriverException, InvalidSessionIdException):
                    logger.info("Existing WebDriver session is invalid, creating a new one")
                    self._driver = None
            
            # Configure Chrome options
            chrome_options = Options()
            
            # Set headless mode
            if self.headless:
                chrome_options.add_argument("--headless=new")  # Use new headless mode
            
            # Add common options
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument(f"--user-agent={self.user_agent}")
            
            # Set proxy if provided
            if proxy:
                chrome_options.add_argument(f"--proxy-server={proxy}")
            
            # Disable images for faster loading
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.javascript": 1,
                "profile.default_content_setting_values.notifications": 2,
                "profile.managed_default_content_settings.stylesheets": 2,
                "profile.managed_default_content_settings.cookies": 2,
                "profile.managed_default_content_settings.plugins": 1,
                "profile.managed_default_content_settings.popups": 2,
                "profile.managed_default_content_settings.geolocation": 2,
                "profile.managed_default_content_settings.media_stream": 2,
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            # Additional options to avoid detection
            chrome_options.add_argument("--disable-blink-features")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            try:
                logger.info("Initializing new WebDriver instance")
                driver = webdriver.Chrome(
                    service=self.chrome_service,
                    options=chrome_options
                )
                
                # Set timeouts
                driver.set_page_load_timeout(timeout)
                driver.set_script_timeout(timeout)
                driver.implicitly_wait(5)  # Shorter implicit wait
                
                # Execute CDP commands to avoid detection
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        });
                        window.navigator.chrome = {
                            runtime: {},
                            // etc.
                        };
                    """
                })
                
                self._driver = driver
                logger.info("WebDriver initialized successfully")
                return driver
                
            except Exception as e:
                error_msg = f"Failed to initialize WebDriver: {str(e)}"
                logger.error(error_msg)
                if '_driver' in locals():
                    try:
                        driver.quit()
                    except:
                        pass
                self._driver = None
                raise WebDriverException(error_msg) from e

    def _extract_review_content(self, element, driver) -> str:
        """Extract review content from an element with fallback methods."""
        try:
            # First try to get the text directly
            text = element.text.strip()
            if text and len(text.split()) > 5:  # At least 5 words
                return text

            # If direct text is too short, try to find text in child elements
            paragraphs = []
            
            try:
                content_elements = element.find_elements(
                    self.By.XPATH,
                    ".//*[self::p or self::div or self::section or self::article or self::blockquote]"
                )
                
                for el in content_elements:
                    try:
                        el_text = el.text.strip()
                        if el_text and len(el_text.split()) > 3:  # At least 3 words
                            paragraphs.append(el_text)
                    except Exception as e:
                        continue
                            
                if paragraphs:
                    return '\n\n'.join(paragraphs)
            except Exception as e:
                pass

            # As a last resort, try to get all text nodes
            try:
                js_code = """
                var nodes = [];
                var walker = document.createTreeWalker(
                    arguments[0],
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode: function(node) {
                            return node.textContent.trim() ?
                                NodeFilter.FILTER_ACCEPT :
                                NodeFilter.FILTER_REJECT;
                        }
                    }
                );
                var node = walker.nextNode();
                while (node) {
                    nodes.push(node.textContent.trim());
                    node = walker.nextNode();
                }
                return nodes.join(' ');
                """
                all_text = driver.execute_script(js_code, element)
                
                if all_text and len(all_text.split()) > 5:
                    return all_text
            except Exception:
                pass

            return text if text else ""

        except Exception as e:
            print(f"Error extracting review content: {str(e)}")
            return ""

    def analyze_first_page(self, url: str, wait_time: int = 5, proxy: Optional[str] = None) -> List[SelectorSuggestion]:
        """Analyze the first page to suggest CSS selectors for data extraction."""
        if url in self._selector_cache:
            return self._selector_cache[url]

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument(f"--user-agent={self.get_random_user_agent()}")
        if proxy:
            chrome_options.add_argument(f"--proxy-server={proxy}")

        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        try:
            driver.get(url)
            time.sleep(wait_time)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            # Get page source and analyze with Gemini
            page_source = driver.page_source
            
            # Simple selector suggestions (you can enhance this with more complex logic)
            suggestions = [
                SelectorSuggestion(
                    field="title",
                    selector="h1",
                    sample_data=driver.find_element(self.By.TAG_NAME, "h1").text if driver.find_elements(self.By.TAG_NAME, "h1") else "",
                    confidence=0.9
                ),
                SelectorSuggestion(
                    field="content",
                    selector=".content, .post, article",
                    sample_data=driver.find_element(self.By.TAG_NAME, "body").text[:200] + "..." if driver.find_elements(self.By.TAG_NAME, "body") else "",
                    confidence=0.7
                )
            ]
            
            self._selector_cache[url] = suggestions
            return suggestions
            
        except Exception as e:
            print(f"Error analyzing page: {str(e)}")
            return self._get_fallback_selectors(driver)
            
        finally:
            driver.quit()
    
    def _get_fallback_selectors(self, driver) -> List[SelectorSuggestion]:
        """Provide fallback selectors when analysis fails."""
        return [
            SelectorSuggestion(
                field="title",
                selector="h1, .title",
                sample_data="",
                confidence=0.5
            ),
            SelectorSuggestion(
                field="content",
                selector="p, .content",
                sample_data="",
                confidence=0.5
            )
        ]
        
    def _init_webdriver(self, proxy: Optional[str] = None, timeout: int = 30):
        """Initialize and return a configured WebDriver instance."""
        if self._driver is not None:
            return self._driver
            
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # Use new headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument(f"--user-agent={self.get_random_user_agent()}")
        
        # Performance optimizations
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-application-cache")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-default-apps")
        chrome_options.add_argument("--mute-audio")
        
        # Add proxy if provided
        if proxy:
            if not re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$', proxy):
                raise ValueError("Invalid proxy format. Use host:port")
            chrome_options.add_argument(f"--proxy-server={proxy}")
        
        # Initialize WebDriver with the pre-configured service
        self._driver = webdriver.Chrome(
            service=self.chrome_service,
            options=chrome_options
        )
        self._driver.set_page_load_timeout(timeout)
        
        # Set timeouts for script and page load
        self._driver.set_script_timeout(15)
        self._driver.implicitly_wait(5)  # Lower implicit wait
        
        return self._driver
        
    def _scroll_page(self, driver, scroll_pause: float = 0.2):
        """Optimized page scrolling to trigger dynamic content."""
        try:
            # Single smooth scroll to bottom
            driver.execute_script("""
                window.scrollTo({
                    top: document.body.scrollHeight,
                    behavior: 'smooth'
                });
            """)
            time.sleep(scroll_pause)
            
            # Check if page height changed (lazy loading)
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
        except Exception as e:
            print(f"Warning during scrolling: {str(e)}")
    
    def _estimate_scraping_time(self, num_pages: int, avg_page_load: float = 3.0, avg_extraction: float = 1.0) -> str:
        """Estimate total scraping time based on number of pages and averages.
        
        Args:
            num_pages: Number of pages to scrape
            avg_page_load: Average page load time in seconds (default: 3.0s)
            avg_extraction: Average extraction time per page in seconds (default: 1.0s)
            
        Returns:
            Formatted string with estimated time (e.g., "2 minutes 30 seconds")
        """
        total_seconds = num_pages * (avg_page_load + avg_extraction)
        
        if total_seconds < 60:
            return f"{int(total_seconds)} seconds"
        elif total_seconds < 3600:
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} {seconds} second{'s' if seconds != 1 else ''}"
        else:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"

    def _print_progress(self, current: int, total: int, start_time: float, page_load: float = 0, extract: float = 0):
        """Print progress information with time estimation."""
        elapsed = time.time() - start_time
        avg_time_per_page = elapsed / current if current > 0 else 0
        remaining_pages = total - current
        eta_seconds = avg_time_per_page * remaining_pages if current > 0 else 0
        
        # Format ETA
        if eta_seconds < 60:
            eta_str = f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
        else:
            eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"
        
        print(f"\r📊 Progress: {current}/{total} pages "
              f"| ⏱️  ETA: {eta_str} "
              f"| 🚀 Load: {page_load:.1f}s "
              f"| 🔍 Extract: {extract:.1f}s"
              f"{' ' * 10}",  # Clear any leftover text
              end="", flush=True)

    def _scrape_pages_parallel(
        self,
        url: str,
        selectors: Dict[str, str],
        max_pages: int,
        proxy: Optional[str] = None,
        wait_time: float = 2.0,
        timeout: int = 30,
        enable_scroll: bool = True,
        max_retries: int = 2,
        min_wait: float = 1.0,
        max_wait: float = 5.0,
        pagination_strategy: str = "auto",
        show_progress: bool = True,
        start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Scrape multiple pages in parallel using ThreadPoolExecutor.
        
        Args:
            url: The starting URL to scrape
            selectors: Dictionary of {field: css_selector} for data extraction
            max_pages: Maximum number of pages to scrape
            proxy: Optional proxy server to use (format: host:port)
            wait_time: Base time to wait for page load in seconds
            timeout: Maximum time to wait for page load in seconds
            enable_scroll: Whether to enable page scrolling
            max_retries: Maximum number of retries for failed requests
            min_wait: Minimum wait time between requests
            max_wait: Maximum wait time between requests
            pagination_strategy: Strategy for pagination
            show_progress: Whether to show progress information
            start_time: Start time of the scraping operation
            
        Returns:
            List of scraped page results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        current_url = url
        pages_to_scrape = []
        
        # First, discover all page URLs if possible
        if pagination_strategy != "url_increment":
            try:
                driver = self._init_webdriver(proxy, timeout)
                try:
                    # Get the first page to discover pagination
                    driver.get(url)
                    
                    # Add first page
                    pages_to_scrape.append((1, url))
                    
                    # Try to find all page URLs
                    for page_num in range(2, max_pages + 1):
                        next_url = self._find_next_page(driver, pagination_strategy)
                        if not next_url or next_url == current_url:
                            break
                        pages_to_scrape.append((page_num, next_url))
                        current_url = next_url
                        
                        # Navigate to next page to continue discovery
                        try:
                            driver.get(next_url)
                            time.sleep(wait_time)  # Wait for page to load
                        except:
                            break
                            
                finally:
                    try:
                        driver.quit()
                    except:
                        pass
            except Exception as e:
                if show_progress:
                    print(f"\n⚠️  Error discovering page URLs: {str(e)}")
                # Fall back to URL increment strategy
                pages_to_scrape = [(i, f"{url}?page={i}" if '?' not in url else f"{url}&page={i}")
                                 for i in range(1, max_pages + 1)]
        else:
            # For URL increment strategy, we can generate all URLs upfront
            pages_to_scrape = [(i, f"{url}?page={i}" if '?' not in url else f"{url}&page={i}")
                             for i in range(1, max_pages + 1)]
        
        if not pages_to_scrape:
            return []
            
        # Scrape pages in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scraping tasks
            future_to_page = {}
            for page_num, page_url in pages_to_scrape:
                future = executor.submit(
                    self._scrape_single_page,
                    url=page_url,
                    selectors=selectors,
                    page_num=page_num,
                    proxy=proxy,
                    enable_scroll=enable_scroll,
                    wait_time=wait_time,
                    min_wait=min_wait,
                    max_wait=max_wait,
                    max_retries=max_retries,
                    pagination_strategy=pagination_strategy,
                    show_progress=show_progress,
                    start_time=start_time,
                    is_last_page=page_num == len(pages_to_scrape)
                )
                future_to_page[future] = page_num
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if show_progress:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed if completed > 0 else 0
                        remaining = max(0, (len(pages_to_scrape) - completed) * avg_time)
                        
                        print(f"\r📊 Progress: {completed}/{len(pages_to_scrape)} pages "
                              f"({completed/len(pages_to_scrape):.0%}) | "
                              f"⏱️  ETA: {remaining//60:.0f}m {remaining%60:.0f}s",
                              end="", flush=True)
                    
                except Exception as e:
                    if show_progress:
                        print(f"\n⚠️  Error scraping page {page_num}: {str(e)}")
        
        # Sort results by page number
        results.sort(key=lambda x: x.get('page_num', 0))
        return results
    
    def _scrape_single_page(
        self,
        url: str,
        selectors: Dict[str, str],
        page_num: int,
        proxy: Optional[str] = None,
        enable_scroll: bool = True,
        wait_time: float = 2.0,
        min_wait: float = 1.0,
        max_wait: float = 5.0,
        max_retries: int = 2,
        pagination_strategy: str = "auto",
        show_progress: bool = True,
        start_time: Optional[float] = None,
        is_last_page: bool = False
    ) -> Dict[str, Any]:
        """Scrape a single page with retry logic."""
        retry_count = 0
        page_start_time = time.time()
        
        if start_time is None:
            start_time = page_start_time
        
        while retry_count <= max_retries:
            try:
                # Show progress
                if show_progress:
                    self._print_progress(
                        current=page_num - 1,
                        total=page_num + 1 if is_last_page else page_num + 2,
                        start_time=start_time,
                        page_load=time.time() - page_start_time if retry_count > 0 else 0,
                        extract=0
                    )
                
                # Initialize WebDriver for this thread
                driver = self._init_webdriver(proxy, timeout=30)
                
                try:
                    # Navigate to URL with timeout
                    driver.get(url)
                    
                    # Wait for dynamic content (adaptive wait time)
                    current_wait = min(max(min_wait, wait_time / (retry_count + 1)), max_wait)
                    time.sleep(current_wait)
                    
                    # Scroll to trigger dynamic content if enabled
                    if enable_scroll:
                        self._scroll_page(driver)
                    
                    # Extract data from current page
                    extract_start = time.time()
                    page_data = self._extract_page_data(driver, selectors, page_num, url)
                    extract_time = time.time() - extract_start
                    
                    if page_data and any(page_data.values()):
                        # Format the result with metadata and timing info
                        result = {
                            "data": page_data,
                            "url": url,
                            "page_num": page_num,
                            "status": "success",
                            "retries": retry_count,
                            "timing": {
                                "page_load": time.time() - page_start_time,
                                "extraction": extract_time
                            }
                        }
                        
                        if show_progress:
                            print(f"\n✅ Page {page_num} scraped in "
                                  f"{time.time() - page_start_time:.1f}s "
                                  f"(load: {time.time() - page_start_time - extract_time:.1f}s, "
                                  f"extract: {extract_time:.1f}s)")
                        
                        return result
                    else:
                        raise ValueError("No data extracted from the page")
                        
                finally:
                    # Always quit the driver to free resources
                    try:
                        driver.quit()
                    except:
                        pass
                        
            except TimeoutException as e:
                retry_count += 1
                if retry_count > max_retries:
                    if show_progress:
                        print(f"\n❌ Page {page_num} timed out after {max_retries} retries")
                    return {
                        "data": {},
                        "url": url,
                        "page_num": page_num,
                        "status": "error",
                        "error": f"Timeout after {max_retries} retries: {str(e)}",
                        "retries": retry_count
                    }
                if show_progress:
                    print(f"\n⚠️  Retry {retry_count}/{max_retries} for page {page_num}...")
                time.sleep(1)  # Small delay before retry
                
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    if show_progress:
                        print(f"\n❌ Page {page_num} failed after {max_retries} retries: {str(e)}")
                    return {
                        "data": {},
                        "url": url,
                        "page_num": page_num,
                        "status": "error",
                        "error": str(e),
                        "retries": retry_count
                    }
                if show_progress:
                    print(f"\n⚠️  Retry {retry_count}/{max_retries} for page {page_num}...")
                time.sleep(1)  # Small delay before retry
        
        return {
            "data": {},
            "url": url,
            "page_num": page_num,
            "status": "error",
            "error": "Max retries reached",
            "retries": retry_count
        }
    
    @retry_on_session_error(max_retries=3, delay=1)
    def scrape_pages(
        self,
        url: str,
        selectors: Dict[str, str],
        max_pages: int = 1,
        wait_time: float = 2.0,
        proxy: Optional[str] = None,
        pagination_strategy: str = "auto",
        max_retries: int = 2,
        timeout: int = 20,
        enable_scroll: bool = True,
        show_progress: bool = True,
        min_wait: float = 1.0,
        max_wait: float = 5.0,
        use_cache: bool = True,
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Scrape multiple pages using the provided selectors with enhanced error handling and retries.
        
        Args:
            url: The starting URL to scrape
            selectors: Dictionary of {field: css_selector} for data extraction
            max_pages: Maximum number of pages to scrape (default: 1)
            wait_time: Base time to wait for page load in seconds (min: 0.5, max: 60, default: 2.0)
            proxy: Optional proxy server to use (format: host:port)
            pagination_strategy: Strategy for pagination ("auto", "url_increment", "next_button")
            max_retries: Maximum number of retries for failed requests (default: 2)
            timeout: Maximum time in seconds to wait for page load (default: 20)
            enable_scroll: Whether to enable page scrolling (default: True)
            show_progress: Whether to show progress information (default: True)
            min_wait: Minimum wait time between requests (default: 1.0s)
            max_wait: Maximum wait time between requests (default: 5.0s)
            use_cache: Whether to use caching for scraped pages (default: True)
            parallel: Whether to scrape pages in parallel (default: True)
            
        Returns:
            List of dictionaries in the format: [{"data": {field: value}, "url": str, "page_num": int, "status": str}]
            
        Raises:
            ValueError: If URL is invalid or selectors are empty
            ConnectionError: If unable to establish connection after max_retries
            TimeoutError: If page loading times out
        """
        # Input validation
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")
        if not selectors or not isinstance(selectors, dict):
            raise ValueError("Selectors must be a non-empty dictionary")
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        
        # Clamp wait times
        wait_time = max(0.5, min(float(wait_time), 60.0))
        min_wait = max(0.1, float(min_wait))
        max_wait = max(min_wait, float(max_wait))
        
        # Initialize variables
        results = []
        current_page = 1
        consecutive_errors = 0
        max_consecutive_errors = 3
        start_time = time.time()
        
        # Generate cache key for this request
        cache_key = None
        if use_cache and self.cache_enabled:
            cache_key = hashlib.md5(
                (url + str(selectors) + str(max_pages)).encode('utf-8')
            ).hexdigest()
            
            # Try to get cached results
            cached_results = self.cache.get(cache_key)
            if cached_results:
                if show_progress:
                    print("\nℹ️  Using cached results")
                return cached_results
        
        # Print initial information
        if show_progress:
            print(f"\n🚀 Starting scraping of up to {max_pages} pages")
            print(f"📊 Estimated time: {self._estimate_scraping_time(max_pages)}")
            print("-" * 60)
        
        try:
            # Scrape pages sequentially or in parallel
            if parallel and max_pages > 1:
                results = self._scrape_pages_parallel(
                    url=url,
                    selectors=selectors,
                    max_pages=max_pages,
                    proxy=proxy,
                    wait_time=wait_time,
                    timeout=timeout,
                    enable_scroll=enable_scroll,
                    max_retries=max_retries,
                    min_wait=min_wait,
                    max_wait=max_wait,
                    pagination_strategy=pagination_strategy,
                    show_progress=show_progress,
                    start_time=start_time
                )
            else:
                # Scrape pages sequentially
                while current_page <= max_pages and consecutive_errors < max_consecutive_errors:
                    # Check if we should use cache for this page
                    page_cache_key = f"{cache_key}_page_{current_page}" if cache_key else None
                    page_data = None
                    
                    if use_cache and self.cache_enabled and page_cache_key:
                        page_data = self.cache.get(page_cache_key)
                    
                    if page_data:
                        if show_progress:
                            print(f"\nℹ️  Using cached data for page {current_page}")
                        results.append(page_data)
                        current_page += 1
                        continue
                    
                    # Scrape the page
                    result = self._scrape_single_page(
                        url=url,
                        selectors=selectors,
                        page_num=current_page,
                        proxy=proxy,
                        enable_scroll=enable_scroll,
                        wait_time=wait_time,
                        min_wait=min_wait,
                        max_wait=max_wait,
                        max_retries=max_retries,
                        pagination_strategy=pagination_strategy,
                        show_progress=show_progress,
                        start_time=start_time,
                        is_last_page=current_page == max_pages
                    )
                    
                    if result["status"] == "success":
                        results.append(result)
                        
                        # Cache the result for this page
                        if use_cache and self.cache_enabled and page_cache_key:
                            self.cache.set(page_cache_key, result)
                        
                        # Get next page URL if available
                        if current_page < max_pages:
                            try:
                                driver = self._init_webdriver(proxy, timeout)
                                try:
                                    driver.get(url)
                                    next_url = self._find_next_page(driver, pagination_strategy)
                                    if next_url and next_url != url:
                                        url = next_url
                                    else:
                                        if show_progress:
                                            print("\nℹ️  No more pages found or pagination limit reached")
                                        break
                                finally:
                                    try:
                                        driver.quit()
                                    except:
                                        pass
                            except Exception as e:
                                if show_progress:
                                    print(f"\n⚠️  Error finding next page: {str(e)}")
                                consecutive_errors += 1
                        
                        current_page += 1
                        consecutive_errors = 0  # Reset error counter on success
                    else:
                        consecutive_errors += 1
                        if show_progress:
                            print(f"\n⚠️  Failed to scrape page {current_page}: {result.get('error', 'Unknown error')}")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            if show_progress:
                                print(f"\n❌ Too many consecutive errors ({consecutive_errors}), stopping...")
                            break
                        
                        # Add increasing delay between retries
                        delay = min(2 ** (consecutive_errors - 1), 30)  # Exponential backoff, max 30s
                        if show_progress:
                            print(f"⏳ Waiting {delay}s before next attempt...")
                        time.sleep(delay)
            
            # Cache the complete results if successful
            if results and use_cache and self.cache_enabled and cache_key:
                self.cache.set(cache_key, results)
            
            # Print final statistics
            if show_progress:
                total_time = time.time() - start_time
                success_count = sum(1 for r in results if r.get("status") == "success")
                error_count = len(results) - success_count
                
                print("\n" + "=" * 60)
                print(f"✅ Scraping completed in {total_time:.1f} seconds")
                print(f"📊 Pages: {len(results)} (✅ {success_count} | ❌ {error_count})")
                
                if results:
                    avg_time = total_time / len(results)
                    print(f"⏱️  Average time per page: {avg_time:.1f}s")
                
                if error_count > 0:
                    print("\n⚠️  Some pages had errors. Check the logs for details.")
                
                print("=" * 60)
            
            return results
            
        except Exception as e:
            if show_progress:
                print(f"\n❌ An unexpected error occurred: {str(e)}")
            raise
        finally:
            # Ensure any remaining WebDriver instances are closed
            try:
                if hasattr(self, '_driver') and self._driver:
                    self._driver.quit()
            except:
                pass
        """Scrape multiple pages using the provided selectors with enhanced error handling and retries.
        
        Args:
            url: The starting URL to scrape
            selectors: Dictionary of {field: css_selector} for data extraction
            max_pages: Maximum number of pages to scrape (default: 1)
            wait_time: Base time to wait for page load in seconds (min: 0.5, max: 60, default: 2.0)
            proxy: Optional proxy server to use (format: host:port)
            pagination_strategy: Strategy for pagination ("auto", "url_increment", "next_button")
            max_retries: Maximum number of retries for failed requests (default: 2)
            timeout: Maximum time in seconds to wait for page load (default: 20)
            enable_scroll: Whether to enable page scrolling (default: True)
            show_progress: Whether to show progress information (default: True)
            min_wait: Minimum wait time between requests (default: 1.0s)
            max_wait: Maximum wait time between requests (default: 5.0s)
            
        Returns:
            List of dictionaries in the format: [{"data": {field: value}, "url": str, "page_num": int, "status": str}]
            
        Raises:
            ValueError: If URL is invalid or selectors are empty
            ConnectionError: If unable to establish connection after max_retries
            TimeoutError: If page loading times out
        """
        # Input validation
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")
        if not selectors or not isinstance(selectors, dict):
            raise ValueError("Selectors must be a non-empty dictionary")
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        
        # Clamp wait times
        wait_time = max(0.5, min(float(wait_time), 60.0))
        min_wait = max(0.1, float(min_wait))
        max_wait = max(min_wait, float(max_wait))
        
        # Initialize variables
        results = []
        current_page = 1
        consecutive_errors = 0
        max_consecutive_errors = 2
        
        # Print initial information
        if show_progress:
            print(f"🚀 Starting scraping of up to {max_pages} pages")
            print(f"📊 Estimated time: {self._estimate_scraping_time(max_pages)}")
            print("-" * 60)
        
        try:
            # Initialize WebDriver
            driver = self._init_webdriver(proxy, timeout)
            start_time = time.time()
            
            while current_page <= max_pages and consecutive_errors < max_consecutive_errors:
                retry_count = 0
                page_success = False
                page_start_time = time.time()
                
                while retry_count <= max_retries and not page_success:
                    try:
                        # Show progress
                        if show_progress:
                            self._print_progress(
                                current=len(results),
                                total=max_pages,
                                start_time=start_time,
                                page_load=time.time() - page_start_time if retry_count > 0 else 0,
                                extract=0
                            )
                        
                        # Navigate to URL with timeout
                        driver.get(url)
                        
                        # Wait for dynamic content (adaptive wait time)
                        current_wait = min(max(min_wait, wait_time / (retry_count + 1)), max_wait)
                        time.sleep(current_wait)
                        
                        # Scroll to trigger dynamic content if enabled
                        if enable_scroll:
                            self._scroll_page(driver)
                        
                        # Extract data from current page
                        extract_start = time.time()
                        page_data = self._extract_page_data(driver, selectors, current_page, url)
                        extract_time = time.time() - extract_start
                        
                        if page_data and any(page_data.values()):
                            # Format the result with metadata and timing info
                            result = {
                                "data": page_data,
                                "url": url,
                                "page_num": current_page,
                                "status": "success",
                                "retries": retry_count,
                                "timing": {
                                    "page_load": time.time() - page_start_time,
                                    "extraction": extract_time
                                }
                            }
                            results.append(result)
                            
                            if show_progress:
                                print(f"\n✅ Page {current_page}/{max_pages} scraped in "
                                      f"{time.time() - page_start_time:.1f}s "
                                      f"(load: {time.time() - page_start_time - extract_time:.1f}s, "
                                      f"extract: {extract_time:.1f}s)")
                            
                            page_success = True
                            consecutive_errors = 0  # Reset error counter on success
                        else:
                            raise ValueError("No data extracted from the page")
                            
                    except TimeoutException as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            if show_progress:
                                print(f"\n❌ Page {current_page} timed out after {max_retries} retries")
                            consecutive_errors += 1
                            break
                        if show_progress:
                            print(f"\n⚠️  Retry {retry_count}/{max_retries} for page {current_page}...")
                        time.sleep(1)  # Small delay before retry
                    
                    except Exception as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            if show_progress:
                                print(f"\nPage {current_page} timed out after {max_retries} retries")
                            consecutive_errors += 1
                            break
                        if show_progress:
                            print(f"\nRetry {retry_count}/{max_retries} for page {current_page}...")
                        time.sleep(1)  # Small delay before retry
                
                # Move to next page if successful
                if page_success:
                    current_page += 1
                    
                    # Find next page URL if not the last page
                    if current_page <= max_pages:
                        next_url = self._find_next_page(driver, pagination_strategy)
                        if next_url and next_url != url:
                            url = next_url
                        else:
                            if show_progress:
                                print("\nℹ️  No more pages found or pagination limit reached")
                            break
                        
                        # Add small random delay between pages to avoid detection
                        time.sleep(random.uniform(0.5, 1.5))
            
            # Print final statistics
            if show_progress:
                total_time = time.time() - start_time
                print("\n" + "=" * 60)
                print(f"Scraping completed in {total_time:.1f} seconds")
                if results:
                    avg_time = total_time / len(results)
                    print(f"Average time per page: {avg_time:.1f}s")
                print(f"Successfully scraped: {len(results)}/{max_pages} pages")
                print("=" * 60)
            
            # Clean up WebDriver
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    if show_progress:
                        print(f"⚠️ Error closing WebDriver: {str(e)}")
            
            return results
            
        except Exception as e:
            if show_progress:
                print(f"❌ Critical error in scrape_pages: {str(e)}")
            raise
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
            
        # Replace multiple whitespace characters with a single space
        text = ' '.join(text.split())
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # Normalize quotes and dashes
        text = text.replace('"', "'")
        text = text.replace('–', '-')
        return text.strip()

    def _extract_structured_data(self, element) -> Dict:
        """Extract structured data from an element."""
        data = {}
        try:
            # Try to extract table data if element is a table
            if element.tag_name.lower() == 'table':
                rows = element.find_elements(self.By.TAG_NAME, 'tr')
                if rows:
                    headers = []
                    table_data = []
                    
                    # Get headers from first row if it's a thead or first row has th elements
                    header_row = rows[0]
                    header_cells = header_row.find_elements(self.By.TAG_NAME, 'th')
                    if not header_cells:
                        header_cells = header_row.find_elements(self.By.TAG_NAME, 'td')
                    
                    if header_cells:
                        headers = [self._clean_text(cell.text) for cell in header_cells]
                        rows = rows[1:]  # Skip header row
                    
                    # Process data rows
                    for row in rows:
                        cells = row.find_elements(self.By.TAG_NAME, 'td')
                        if not cells:
                            continue
                            
                        row_data = [self._clean_text(cell.text) for cell in cells]
                        if headers and len(headers) == len(row_data):
                            table_data.append(dict(zip(headers, row_data)))
                        else:
                            table_data.append(row_data)
                    
                    if table_data:
                        data['table_data'] = table_data
                        return data
            
            # For non-table elements, try to extract key-value pairs
            key_value_pairs = {}
            
            # Look for definition lists
            dl_elements = element.find_elements(self.By.TAG_NAME, 'dl')
            for dl in dl_elements:
                try:
                    terms = dl.find_elements(self.By.TAG_NAME, 'dt')
                    descs = dl.find_elements(self.By.TAG_NAME, 'dd')
                    for term, desc in zip(terms, descs):
                        key = self._clean_text(term.text)
                        value = self._clean_text(desc.text)
                        if key and value:
                            key_value_pairs[key] = value
                except:
                    continue
            
            # Look for heading + next sibling patterns
            for h_level in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                try:
                    headers = element.find_elements(self.By.TAG_NAME, h_level)
                    for header in headers:
                        try:
                            key = self._clean_text(header.text)
                            if not key:
                                continue
                                
                            # Get next sibling element's text
                            next_sibling = header.find_element(self.By.XPATH, './following-sibling::*[1]')
                            if next_sibling:
                                value = self._clean_text(next_sibling.text)
                                if value:
                                    key_value_pairs[key] = value
                        except:
                            continue
                except:
                    continue
            
            if key_value_pairs:
                data.update(key_value_pairs)
            
        except Exception as e:
            print(f"⚠️ Error extracting structured data: {str(e)}")
        
        return data

    def _extract_page_data(self, driver, selectors: Dict[str, str], page_num: int, url: str) -> Dict:
        """
        Extract data from the current page using provided selectors with enhanced error handling
        and data alignment.
        
        Args:
            driver: Selenium WebDriver instance
            selectors: Dictionary of {field: css_selector} for data extraction
            page_num: Current page number (for logging)
            url: Current page URL (for logging)
            
        Returns:
            Dictionary containing the extracted data with field names as keys.
            Returns empty dict if no data could be extracted.
            
        Note:
            - Handles both single and multiple elements per selector
            - Preserves data types (str for single, list for multiple)
            - Includes structured data extraction for tables and key-value pairs
            - Performs text cleaning and normalization
        """
        if not selectors or not isinstance(selectors, dict):
            print("⚠️ No selectors provided for data extraction")
            return {}
            
        page_data = {}
        errors = []
        
        for field, selector in selectors.items():
            if not selector or not isinstance(selector, str):
                errors.append(f"Invalid selector for field '{field}': {selector}")
                page_data[field] = ""
                continue
                
            try:
                # Wait for at least one element to be present
                try:
                    elements = WebDriverWait(driver, 5).until(
                        EC.presence_of_all_elements_located((self.By.CSS_SELECTOR, selector))
                    )
                except TimeoutException:
                    # No elements found within timeout
                    elements = []
                
                if not elements:
                    errors.append(f"No elements found for '{field}' with selector: {selector}")
                    page_data[field] = ""
                    continue
                
                # Process elements based on count
                if len(elements) == 1:
                    # Single element - get text and attributes
                    try:
                        element = elements[0]
                        
                        # First try to extract structured data
                        structured_data = self._extract_structured_data(element)
                        if structured_data:
                            page_data[field] = structured_data
                            continue
                        
                        # Fall back to text extraction
                        text = element.text.strip()
                        
                        # If no text, try to get value attribute
                        if not text and element.get_attribute("value"):
                            text = element.get_attribute("value").strip()
                            
                        # If still no text, try to get innerHTML
                        if not text and element.get_attribute("innerHTML"):
                            text = element.get_attribute("innerHTML").strip()
                        
                        # Clean and normalize the text
                        text = self._clean_text(text)
                        page_data[field] = text if text else ""
                        
                    except Exception as e:
                        errors.append(f"Error processing single element for '{field}': {str(e)}")
                        page_data[field] = ""
                        
                else:
                    # Multiple elements - collect all texts with context
                    try:
                        items = []
                        for i, el in enumerate(elements, 1):
                            try:
                                # Try to get structured data first
                                structured_data = self._extract_structured_data(el)
                                if structured_data:
                                    items.append(structured_data)
                                    continue
                                
                                # Fall back to text extraction
                                text = el.text.strip()
                                if not text and el.get_attribute("value"):
                                    text = el.get_attribute("value").strip()
                                
                                if text:  # Only add non-empty texts
                                    items.append(self._clean_text(text))
                                    
                            except Exception as e:
                                errors.append(f"Error processing element {i} for '{field}': {str(e)}")
                        
                        page_data[field] = items if items else [""]
                        
                    except Exception as e:
                        errors.append(f"Error processing multiple elements for '{field}': {str(e)}")
                        page_data[field] = [""]
                        
            except Exception as e:
                error_msg = f"Unexpected error extracting '{field}': {str(e)}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)
                page_data[field] = ""
        
        # Add metadata
        page_data['_metadata'] = {
            'url': url,
            'page_num': page_num,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Log any errors that occurred during extraction
        if errors:
            print(f"⚠️ Encountered {len(errors)} errors while extracting data from page {page_num}:")
            for i, error in enumerate(errors[:5], 1):  # Show first 5 errors to avoid log spam
                print(f"  {i}. {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        return page_data
    
    def _find_next_page(self, driver, pagination_strategy: str) -> Optional[str]:
        """Find the URL of the next page based on the pagination strategy."""
        if pagination_strategy == "url_increment":
            # Try to increment page number in URL
            current_url = driver.current_url
            if "?" in current_url:
                base_url, params = current_url.split("?", 1)
                if "page=" in params:
                    # Increment page number
                    import re
                    new_params = re.sub(r'(?<=page=)(\d+)', 
                                     lambda m: str(int(m.group(1)) + 1), 
                                     params)
                    return f"{base_url}?{new_params}"
            return None
            
        elif pagination_strategy in ["next_button", "auto"]:
            # Try to find and click next button
            next_selectors = [
                "a[rel='next']",
                ".pagination .next a",
                "a.next",
                "button.next",
                "a:contains('Next')",
                "a:contains('>')",
                "a:contains('»')"
            ]
            
            for selector in next_selectors:
                try:
                    next_btn = driver.find_element(self.By.CSS_SELECTOR, selector)
                    if next_btn.is_displayed() and next_btn.is_enabled():
                        return next_btn.get_attribute("href")
                except:
                    continue
                    
        return None
