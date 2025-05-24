"""
Modern web scraper implementation with support for JavaScript rendering, AI-assisted selector inference,
and robust error handling.
"""

import asyncio
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from pydantic import BaseModel, HttpUrl, Field, field_validator
from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .config import SCRAPER_CONFIG, API_CONFIG, MODEL_CONFIG
from .models import ScrapedDataInput, SelectorSuggestion
from .utils.logging import log_error, logger, log_execution_time
from .utils.validation import is_valid_url, validate_css_selector, ValidationError

# Type aliases
Url = str
Selector = str

@dataclass
class ScrapeResult:
    """Container for scraped page results."""
    url: str
    content: str
    status_code: int
    selectors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if the scrape was successful."""
        return self.error is None and 200 <= self.status_code < 400

class WebScraper:
    """
    Modern web scraper with JavaScript support, AI-assisted selector inference,
    and robust error handling.
    """
    
    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        proxy: Optional[str] = None,
        **kwargs
    ):
        """Initialize the web scraper.
        
        Args:
            headless: Whether to run browser in headless mode
            timeout: Default timeout for page loads and element waits (seconds)
            max_retries: Maximum number of retry attempts for failed requests
            user_agent: Custom user agent string
            proxy: Proxy server URL (e.g., 'http://user:pass@host:port')
            **kwargs: Additional browser options
        """
        self.headless = headless
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or UserAgent().random
        self.proxy = proxy
        self.browser = None
        self._init_browser(**kwargs)
    
    def _init_browser(self, **options) -> None:
        """Initialize the Selenium WebDriver with the given options."""
        try:
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.options import Options
            
            chrome_options = Options()
            
            # Set headless mode
            if self.headless:
                chrome_options.add_argument('--headless=new')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
            
            # Set user agent
            chrome_options.add_argument(f'user-agent={self.user_agent}')
            
            # Set proxy if provided
            if self.proxy:
                chrome_options.add_argument(f'--proxy-server={self.proxy}')
            
            # Additional options
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize the WebDriver with webdriver-manager
            self.browser = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            # Set page load timeout
            self.browser.set_page_load_timeout(self.timeout)
            
            logger.info("WebDriver initialized successfully")
            
            # Add additional options
            for arg in options.get('arguments', []):
                chrome_options.add_argument(arg)
            
            # Initialize the WebDriver
            service = Service(options=chrome_options)
            self.browser = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set page load and script timeouts
            self.browser.set_page_load_timeout(self.timeout)
            self.browser.set_script_timeout(self.timeout)
            
            logger.info("WebDriver initialized successfully")
            
        except Exception as e:
            log_error(e, "Failed to initialize WebDriver")
            raise
    
    def close(self) -> None:
        """Close the browser and clean up resources."""
        if self.browser:
            try:
                self.browser.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                log_error(e, "Error while closing WebDriver")
            finally:
                self.browser = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @log_execution_time(logger)
    def scrape(
        self,
        url: str,
        selectors: Optional[Dict[str, str]] = None,
        wait_for: Optional[Union[str, List[str]]] = None,
        wait_time: Optional[float] = None,
        js: bool = True,
        screenshot: bool = False,
        **kwargs
    ) -> ScrapeResult:
        """Scrape a single page with the given selectors.
        
        Args:
            url: URL to scrape
            selectors: Dictionary of {field_name: css_selector} for data extraction
            wait_for: CSS selector or list of selectors to wait for before scraping
            wait_time: Time to wait for page to load (seconds)
            js: Whether to execute JavaScript
            screenshot: Whether to take a screenshot
            **kwargs: Additional options
            
        Returns:
            ScrapeResult containing the scraped data and metadata
        """
        logger.info(f"Starting scrape for URL: {url}")
        if not self.browser:
            logger.debug("Initializing browser...")
            self._init_browser()
        
        result = ScrapeResult(url=url, content='', status_code=0)
        
        try:
            # Navigate to the URL
            logger.info(f"Navigating to URL: {url}")
            self.browser.get(url)
            
            # Wait for page to load
            wait_time = wait_time or self.timeout
            logger.debug(f"Waiting up to {wait_time} seconds for page to load")
            wait = WebDriverWait(self.browser, wait_time)
            
            # Wait for specific elements if specified
            if wait_for:
                selector_list = [wait_for] if isinstance(wait_for, str) else wait_for
                logger.debug(f"Waiting for selectors: {', '.join(selector_list)}")
                
                for selector in selector_list:
                    try:
                        logger.debug(f"Waiting for selector: {selector}")
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        logger.debug(f"Found selector: {selector}")
                    except TimeoutException as te:
                        logger.warning(f"Timeout waiting for selector: {selector} - {str(te)}")
                        result.error = f"Timeout waiting for selector: {selector}"
                    except Exception as e:
                        logger.error(f"Error waiting for selector {selector}: {str(e)}")
                        result.error = f"Error waiting for selector {selector}: {str(e)}"
            
            # Get page source
            logger.debug("Getting page source")
            page_source = self.browser.page_source
            
            if not page_source or len(page_source) < 100:  # Arbitrary minimum length
                error_msg = "Page source is empty or too small"
                logger.error(error_msg)
                result.error = error_msg
                return result
                
            result.content = page_source
            result.status_code = 200
            logger.info(f"Successfully loaded page: {url}")
            
            # Extract data using selectors if provided
            if selectors:
                logger.debug(f"Extracting data with selectors: {', '.join(selectors.keys())}")
                try:
                    extracted_data = self._extract_with_selectors(selectors)
                    if extracted_data:
                        result.selectors = extracted_data
                        logger.debug(f"Successfully extracted data: {extracted_data.keys()}")
                    else:
                        logger.warning("No data extracted with the provided selectors")
                        result.error = "No data extracted with the provided selectors"
                except Exception as e:
                    logger.error(f"Error extracting data with selectors: {str(e)}")
                    result.error = f"Error extracting data: {str(e)}"
            
            # Take screenshot if requested
            if screenshot:
                screenshot_path = Path("screenshots") / f"{urlparse(url).netloc}_{int(time.time())}.png"
                screenshot_path.parent.mkdir(exist_ok=True)
                self.browser.save_screenshot(str(screenshot_path))
                result.metadata['screenshot'] = str(screenshot_path)
            
            logger.info(f"Successfully scraped {url}")
            
        except WebDriverException as e:
            error_msg = f"Error scraping {url}: {str(e)}"
            logger.error(error_msg)
            result.error = error_msg
            result.status_code = getattr(e, 'status_code', 500)
        
        return result
    
    def _extract_with_selectors(self, selectors: Dict[str, str]) -> Dict[str, Any]:
        """Extract data from the current page using the given CSS selectors.
        
        Args:
            selectors: Dictionary of {field_name: css_selector}
            
        Returns:
            Dictionary of extracted data with field names as keys and extracted content as values.
            Returns None for fields that couldn't be extracted.
        """
        if not selectors:
            logger.warning("No selectors provided for extraction")
            return {}
            
        data = {}
        
        for field, selector in selectors.items():
            if not selector or not isinstance(selector, str):
                logger.warning(f"Invalid selector for field '{field}': {selector}")
                data[field] = None
                continue
                
            try:
                logger.debug(f"Finding elements with selector: {selector}")
                elements = self.browser.find_elements(By.CSS_SELECTOR, selector)
                
                if not elements:
                    logger.debug(f"No elements found for selector: {selector}")
                    data[field] = None
                    continue
                    
                # If only one element found, return its text
                if len(elements) == 1:
                    try:
                        element_text = elements[0].text.strip()
                        data[field] = element_text if element_text else None
                        logger.debug(f"Extracted text for '{field}': {element_text[:50]}..." if element_text else "No text content")
                    except Exception as e:
                        logger.error(f"Error extracting text from element: {str(e)}")
                        data[field] = None
                # If multiple elements found, return list of texts
                else:
                    try:
                        extracted_texts = [el.text.strip() for el in elements if el and el.text.strip()]
                        data[field] = extracted_texts if extracted_texts else None
                        logger.debug(f"Extracted {len(extracted_texts)} items for '{field}'")
                    except Exception as e:
                        logger.error(f"Error extracting multiple elements: {str(e)}")
                        data[field] = None
                        
            except Exception as e:
                logger.error(f"Failed to extract '{field}' with selector '{selector}': {str(e)}")
                data[field] = None
        
        # Log summary of extraction
        success_count = sum(1 for v in data.values() if v is not None)
        logger.info(f"Extracted {success_count}/{len(selectors)} fields successfully")
        
        return data
    
    @log_execution_time(logger)
    def scrape_multiple(
        self,
        urls: List[str],
        selectors: Optional[Dict[str, str]] = None,
        max_workers: int = 3,
        **kwargs
    ) -> Dict[str, ScrapeResult]:
        """Scrape multiple URLs in parallel.
        
        Args:
            urls: List of URLs to scrape
            selectors: Dictionary of {field_name: css_selector} for data extraction
            max_workers: Maximum number of concurrent scrapers
            **kwargs: Additional options to pass to the scrape method
            
        Returns:
            Dictionary mapping URLs to their respective ScrapeResults
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {
                executor.submit(self.scrape, url, selectors, **kwargs): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                except Exception as e:
                    log_error(e, f"Error scraping {url}")
                    results[url] = ScrapeResult(
                        url=url,
                        content='',
                        status_code=500,
                        error=str(e)
                    )
        
        return results
    
    def infer_selectors(
        self,
        url: str,
        fields: Optional[List[str]] = None,
        sample_size: int = 3,
        **kwargs
    ) -> List[SelectorSuggestion]:
        """Infer CSS selectors for the given fields using AI.
        
        Args:
            url: URL to analyze
            fields: List of field names to infer selectors for
            sample_size: Number of sample elements to analyze per field
            **kwargs: Additional options
            
        Returns:
            List of SelectorSuggestion objects
        """
        # This is a placeholder for AI-based selector inference
        # In a real implementation, this would use an AI model to analyze the page
        # and suggest the best selectors for the given fields
        
        # For now, return some basic selectors based on common patterns
        common_selectors = {
            'title': ['h1', '.title', '#title', 'header h1'],
            'price': ['.price', '.amount', '.value', '[itemprop=price]'],
            'description': ['[itemprop=description]', '.description', '#description'],
            'image': ['img[src]', '[itemprop=image]', '.product-image img'],
        }
        
        suggestions = []
        
        for field in (fields or common_selectors.keys()):
            for selector in common_selectors.get(field, [f'[data-{field}]', f'.{field}']):
                suggestions.append(SelectorSuggestion(
                    field=field,
                    selector=selector,
                    confidence=0.8,
                    sample_data=f"Sample {field} data"
                ))
        
        return suggestions

def quick_scrape(url: str, selectors: Dict[str, str], **kwargs) -> Dict[str, Any]:
    """Quickly scrape a single page with the given selectors.
    
    Args:
        url: URL to scrape
        selectors: Dictionary of {field_name: css_selector}
        **kwargs: Additional options to pass to WebScraper
        
    Returns:
        Dictionary of scraped data
    """
    with WebScraper(**kwargs) as scraper:
        result = scraper.scrape(url, selectors=selectors, **kwargs)
        return result.selectors if result.success else {}

def quick_scrape_multiple(urls: List[str], selectors: Dict[str, str], **kwargs) -> Dict[str, Dict[str, Any]]:
    """Quickly scrape multiple pages with the given selectors.
    
    Args:
        urls: List of URLs to scrape
        selectors: Dictionary of {field_name: css_selector}
        **kwargs: Additional options to pass to WebScraper
        
    Returns:
        Dictionary mapping URLs to their scraped data
    """
    with WebScraper(**kwargs) as scraper:
        results = scraper.scrape_multiple(urls, selectors=selectors, **kwargs)
        return {url: result.selectors for url, result in results.items() if result.success}

# Example usage
if __name__ == "__main__":
    # Example: Scrape a single page
    data = quick_scrape(
        url="https://example.com",
        selectors={
            "title": "h1",
            "content": "article p"
        },
        headless=True
    )
    print("Scraped data:", json.dumps(data, indent=2))
    
    # Example: Scrape multiple pages
    urls = ["https://example.com/page1", "https://example.com/page2"]
    all_data = quick_scrape_multiple(
        urls=urls,
        selectors={
            "title": "h1",
            "content": "article p"
        },
        headless=True,
        max_workers=2
    )
    print("\nAll scraped data:", json.dumps(all_data, indent=2))
