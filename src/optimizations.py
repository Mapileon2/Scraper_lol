"""
Performance optimizations for the web scraper.
"""
import os
import pickle
import time
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

# Configure logging
logging.basicConfig(level=logging.INFO, filename="scraper.log")
logger = logging.getLogger(__name__)

# Global cache for selectors
SELECTOR_CACHE_FILE = "selector_cache.pkl"

def init_driver(headless: bool = True) -> WebDriver:
    """Initialize and return a configured WebDriver instance."""
    options = Options()
    if headless:
        options.add_argument("--headless")
    
    # Configure options for performance and stealth
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-agent={UserAgent().random}")
    options.add_argument("--window-size=1920,1080")
    
    # Disable images and unnecessary resources
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.javascript": 1,
    }
    options.add_experimental_option("prefs", prefs)
    
    from selenium import webdriver
    driver = webdriver.Chrome(options=options)
    
    # Remove webdriver property to avoid detection
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    
    return driver

def wait_for_element(driver: WebDriver, selector: str, timeout: int = 10) -> bool:
    """Wait for an element to be present on the page."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return True
    except Exception as e:
        logger.warning(f"Element {selector} not found: {str(e)}")
        return False

def get_cached_selectors(url: str) -> Optional[Dict]:
    """Get cached selectors for a URL if they exist and are not expired."""
    if not os.path.exists(SELECTOR_CACHE_FILE):
        return None
        
    try:
        with open(SELECTOR_CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
            if url in cache:
                # Check if cache is expired (1 day TTL)
                if time.time() - cache[url]['timestamp'] < 86400:
                    return cache[url]['selectors']
    except Exception as e:
        logger.error(f"Error reading selector cache: {str(e)}")
    return None

def cache_selectors(url: str, selectors: Dict) -> None:
    """Cache selectors for a URL."""
    cache = {}
    if os.path.exists(SELECTOR_CACHE_FILE):
        try:
            with open(SELECTOR_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading selector cache: {str(e)}")
    
    cache[url] = {
        'selectors': selectors,
        'timestamp': time.time()
    }
    
    try:
        with open(SELECTOR_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.error(f"Error writing to selector cache: {str(e)}")

@lru_cache(maxsize=100)
def get_suggested_selectors(html_content: str, url: str) -> Dict[str, List[str]]:
    """Get suggested selectors for a page using AI or fallback to heuristics."""
    # Try to get from cache first
    cached = get_cached_selectors(url)
    if cached:
        return cached
    
    # Fallback to simple heuristics if AI is not available
    selectors = {
        'title': ['h1', 'h2', '.title', 'header h1'],
        'content': ['article', '.content', 'main', '#main'],
        'price': ['.price', '[itemprop="price"]', '.amount'],
        'description': ['[itemprop="description"]', '.description', 'p'],
        'images': ['img[src]', '[itemprop="image"]', '.gallery img']
    }
    
    # Cache the results
    cache_selectors(url, selectors)
    return selectors

def process_batch(urls: List[str], max_workers: int = 3) -> List[Dict]:
    """Process a batch of URLs in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(process_single_url, url): url
            for url in urls
        }
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                results.append({'url': url, 'error': str(e)})
    
    return results

async def process_single_url(url: str) -> Dict:
    """Process a single URL with retry logic."""
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            driver = init_driver()
            try:
                # Set a reasonable page load timeout
                driver.set_page_load_timeout(15)
                
                # Navigate to the URL
                driver.get(url)
                
                # Wait for content to load
                if not wait_for_element(driver, 'body', 10):
                    raise TimeoutError("Page failed to load")
                
                # Get page content
                html = driver.page_source
                
                # Get suggested selectors
                selectors = get_suggested_selectors(html, url)
                
                # Extract data using selectors
                data = {}
                for key, selector_list in selectors.items():
                    for selector in selector_list:
                        try:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements:
                                data[key] = [el.text for el in elements if el.text.strip()]
                                if data[key]:  # Stop at first matching selector
                                    break
                        except Exception as e:
                            logger.debug(f"Selector {selector} failed: {str(e)}")
                
                return {
                    'url': url,
                    'data': data,
                    'status': 'success',
                    'selectors_used': {k: v[0] for k, v in selectors.items() if v}
                }
                
            finally:
                driver.quit()
                
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {last_error}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    return {
        'url': url,
        'error': f"Failed after {max_retries + 1} attempts: {last_error}",
        'status': 'error'
    }
