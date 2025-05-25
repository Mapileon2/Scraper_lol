import asyncio
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from models import ScrapeRequest, ScrapeField, ScrapedData
from llm_interface import LLMInterface, GeminiLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIScraperAgent:
    """AI-powered web scraper agent with Pydantic validation and LLM integration."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize the scraper agent with caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.llm_cache = {}
    
    async def scrape(self, request: ScrapeRequest) -> List[ScrapedData]:
        """
        Scrape data from the provided URL using the specified fields and configuration.
        
        Args:
            request: ScrapeRequest containing configuration and fields to scrape
            
        Returns:
            List of ScrapedData objects with the results
        """
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        # Set up Chrome options
        chrome_options = webdriver.ChromeOptions()
        if request.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Initialize the WebDriver
        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(request.timeout)
            
            # Load the page
            logger.info(f"Loading URL: {request.url}")
            driver.get(str(request.url))
            
            # Get page HTML for selector inference
            html = driver.page_source
            
            # Generate or validate selectors
            fields_with_selectors = await self._prepare_selectors(
                html=html,
                fields=request.fields,
                llm_provider=request.llm_provider,
                llm_api_key=request.llm_api_key
            )
            
            # Scrape the data
            results = await self._scrape_page(
                driver=driver,
                url=str(request.url),
                fields=fields_with_selectors
            )
            
            return [ScrapedData(url=str(request.url), data=results)]
            
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            return [ScrapedData.error(url=str(request.url), error=str(e))]
            
        finally:
            if driver:
                driver.quit()
    
    async def _prepare_selectors(
        self,
        html: str,
        fields: List[ScrapeField],
        llm_provider: str,
        llm_api_key: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Generate or validate CSS selectors for the given fields."""
        fields_with_selectors = []
        fields_need_inference = []
        
        # Separate fields with and without selectors
        for field in fields:
            if field.selector:
                fields_with_selectors.append({
                    "name": field.name,
                    "selector": field.selector,
                    "description": field.description
                })
            else:
                fields_need_inference.append({
                    "name": field.name,
                    "description": field.description
                })
        
        # If no fields need inference, return early
        if not fields_need_inference:
            return fields_with_selectors
        
        # If no API key is provided, we can't infer selectors
        if not llm_api_key:
            logger.warning("No LLM API key provided. Some fields may not be scraped.")
            return fields_with_selectors
        
        # Initialize LLM for selector inference
        try:
            if llm_provider.lower() == "gemini":
                llm = GeminiLLM(api_key=llm_api_key)
                inferred_selectors = await llm.generate_selectors(html, fields_need_inference)
                
                # Add inferred selectors to results
                for suggestion in inferred_selectors:
                    fields_with_selectors.append({
                        "name": suggestion.field,
                        "selector": suggestion.selector,
                        "description": next(
                            (f["description"] for f in fields_need_inference 
                             if f["name"] == suggestion.field), ""
                        )
                    })
                    logger.info(f"Inferred selector for {suggestion.field}: {suggestion.selector} (confidence: {suggestion.confidence:.2f})")
            
            return fields_with_selectors
            
        except Exception as e:
            logger.error(f"Error during selector inference: {str(e)}")
            return fields_with_selectors
    
    async def _scrape_page(
        self,
        driver: WebDriver,
        url: str,
        fields: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Scrape data from the current page using the provided selectors."""
        results = {}
        
        for field in fields:
            try:
                elements = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, field["selector"]))
                )
                
                if not elements:
                    results[field["name"]] = None
                    continue
                
                # Get text from all matching elements
                values = [el.text.strip() for el in elements if el.text.strip()]
                
                # Store single value or list based on number of matches
                if len(values) == 1:
                    results[field["name"]] = values[0]
                else:
                    results[field["name"]] = values
                    
            except Exception as e:
                logger.warning(f"Failed to extract field '{field['name']}': {str(e)}")
                results[field["name"]] = None
        
        return results

    async def scrape_multiple(self, requests: List[ScrapeRequest], max_workers: int = 3) -> List[ScrapedData]:
        """Scrape multiple URLs in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, lambda r=req: asyncio.create_task(self.scrape(r)))
                for req in requests
            ]
            
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    results.extend(result)
                except Exception as e:
                    logger.error(f"Error in parallel scraping: {str(e)}")
        
        return results
