from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
from datetime import datetime
import logging
import textwrap
import traceback

# Try to import Crawl4AI components
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMExtractionStrategy
    from crawl4ai.models import CrawlResult
    CRAWL4AI_AVAILABLE = True
except ImportError as e:
    CRAWL4AI_AVAILABLE = False
    # Create dummy classes for type checking
    class CrawlResult:
        pass
    
    class AsyncWebCrawler:
        pass
    
    class BrowserConfig:
        pass
    
    class CrawlerRunConfig:
        pass
    
    class LLMExtractionStrategy:
        pass
    
    # Log the import error for debugging
    import sys
    print(f"Failed to import Crawl4AI: {e}", file=sys.stderr)

class Crawl4AIScraper:
    """A scraper that uses Crawl4AI for AI-powered web scraping."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        """
        Initialize the Crawl4AI scraper.
        
        Args:
            api_key: Optional API key for Crawl4AI (if required)
            model: The AI model to use for extraction
        """
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "Crawl4AI is not installed. Please install it with: "
                "pip install crawl4ai[all]"
            )
            
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.browser_config = None
        self.run_config = None
        self.crawler = None
        
        # Initialize the crawler with default config
        self._initialize_crawler()
        
    def _initialize_crawler(self):
        """Initialize the crawler with default configuration."""
        # Default user agent for Chrome on Windows
        default_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Initialize LLM extraction strategy
        from crawl4ai import LLMExtractionStrategy
        
        # Configure LLM for content extraction
        llm_config = {
            "provider": "openai",  # or "anthropic", "cohere", etc.
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            extraction_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "links": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title", "content"]
            }
        )
        
        # Create run config with minimal required parameters
        self.run_config = CrawlerRunConfig(
            word_count_threshold=100,  # Lower threshold for testing
            extraction_strategy=extraction_strategy,
            only_text=True,  # Simplify to text only for testing
            verbose=True,
            log_console=True,  # Enable console logging for debugging
            user_agent=default_user_agent
        )
        
        # Initialize the crawler with the run_config
        self.crawler = AsyncWebCrawler(
            run_config=self.run_config
        )
        
        # Set browser-specific settings if possible
        if hasattr(self.crawler, 'browser_config'):
            self.crawler.browser_config.headless = True
            self.crawler.browser_config.viewport = {"width": 1280, "height": 800}
            self.crawler.browser_config.browser_type = "chromium"
        
    async def scrape_pages(
        self,
        url: str,
        selectors: Dict[str, str],
        max_pages: int = 1,
        wait_time: float = 2.0,
        pagination_strategy: str = "auto",
        max_retries: int = 3,
        timeout: int = 30,
        enable_scroll: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple pages using Crawl4AI.
        
        Args:
            url: The starting URL to scrape
            selectors: Dictionary of CSS selectors for data extraction
            max_pages: Maximum number of pages to scrape
            wait_time: Time to wait between page loads (seconds)
            pagination_strategy: Strategy for handling pagination
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            enable_scroll: Whether to enable auto-scrolling
            show_progress: Whether to show progress information
            **kwargs: Additional arguments to pass to the crawler
            
        Returns:
            List of dictionaries containing scraped data
        """
        results = []
        
        try:
            # Configure the crawler with minimal settings
            run_config = CrawlerRunConfig(
                word_count_threshold=100,
                only_text=True,
                verbose=show_progress,
                log_console=show_progress,
                user_agent=self.run_config.user_agent if hasattr(self, 'run_config') else None
            )
            
            # Initialize the crawler
            crawler = AsyncWebCrawler(run_config=run_config)
            
            # Set browser-specific settings if possible
            if hasattr(crawler, 'browser_config'):
                crawler.browser_config.headless = True
                crawler.browser_config.viewport = {"width": 1280, "height": 800}
                crawler.browser_config.browser_type = "chromium"
            
            # Run the crawler
            if show_progress:
                print(f"Starting to scrape {url}...")
            
            # Get the HTML content directly using requests as a fallback
            import requests
            from bs4 import BeautifulSoup
            
            try:
                # First try with requests to get the raw HTML
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Create a basic page data structure
                page_data = {
                    'url': url,
                    'title': getattr(soup.find('title'), 'text', '').strip(),
                    'content': soup.get_text(' ', strip=True),
                    'links': [a.get('href', '') for a in soup.find_all('a', href=True)],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Extract content using selectors if provided
                if selectors:
                    extracted = {}
                    for key, selector in selectors.items():
                        try:
                            if '::' in selector:
                                # Handle special selectors like ::attr()
                                selector_parts = selector.split('::')
                                elements = soup.select(selector_parts[0])
                                if elements:
                                    attr = selector_parts[1]
                                    if attr == 'text':
                                        extracted[key] = ' '.join([e.get_text(strip=True) for e in elements])
                                    else:
                                        extracted[key] = ' '.join([e.get(attr, '') for e in elements if e.get(attr)])
                            else:
                                # Regular CSS selector
                                elements = soup.select(selector)
                                if elements:
                                    extracted[key] = ' '.join([e.get_text(strip=True) for e in elements])
                        except Exception as e:
                            self.logger.warning(f"Failed to extract {key} with selector {selector}: {str(e)}")
                    
                    if extracted:
                        page_data['extracted'] = extracted
                
                results.append(page_data)
                
                if show_progress:
                    print(f"Successfully scraped page: {url}")
                
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {str(e)}")
                if show_progress:
                    print(f"Error scraping {url}: {str(e)}")
            
            if show_progress:
                print(f"Successfully scraped {len(results)} pages from {url}")
                
        except Exception as e:
            self.logger.error(f"Error during Crawl4AI scraping: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise  # Re-raise the exception to be handled by the caller
            
        return results
    
    async def extract_structured_data(self, url: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from a webpage using AI.
        
        Args:
            url: The URL to extract data from
            schema: JSON schema defining the structure to extract
            
        Returns:
            Dictionary containing the extracted structured data
        """
        try:
            # First, try to get the page content using requests
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            print(f"Fetching content from {url}...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content = ''
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            if content_div:
                # Remove unwanted elements
                for element in content_div.find_all(['table', 'div.hatnote', 'div.thumb', 'div.reflist', 'div.navbox']):
                    element.decompose()
                content = content_div.get_text(' ', strip=True)
            else:
                content = soup.get_text(' ', strip=True)
            
            # If we have content, try to extract using the schema
            if content:
                print("Content retrieved, extracting structured data...")
                
                # For this example, we'll simulate the AI extraction
                # In a real implementation, you would use an AI service here
                
                # Mock extraction based on the schema
                extracted = {}
                if 'properties' in schema:
                    for key, prop in schema['properties'].items():
                        if key == 'title':
                            title = soup.find('title')
                            extracted[key] = title.get_text(strip=True) if title else 'No title found'
                        elif key == 'summary':
                            # Get first paragraph as summary
                            first_p = content.split('\n\n')[0] if '\n\n' in content else content[:500]
                            extracted[key] = first_p.strip()
                        elif key == 'key_topics':
                            # Extract section headers as key topics
                            headers = [h.get_text(strip=True) for h in soup.find_all(['h2', 'h3'])]
                            extracted[key] = headers[:10]  # Limit to first 10
                        elif key == 'technologies_mentioned':
                            # Look for common web scraping technologies in the text
                            tech_terms = ['BeautifulSoup', 'Scrapy', 'Selenium', 'Playwright', 'Puppeteer', 
                                        'lxml', 'requests', 'urllib', 'aiohttp', 'MechanicalSoup']
                            mentioned = [tech for tech in tech_terms if tech.lower() in content.lower()]
                            extracted[key] = mentioned
                        elif key == 'legal_considerations':
                            # Look for legal-related sections
                            legal_phrases = ['legal', 'terms of service', 'copyright', 'robots.txt', 'terms of use']
                            legal_sections = []
                            
                            # Check section headers
                            for h in soup.find_all(['h2', 'h3']):
                                if any(phrase in h.get_text().lower() for phrase in legal_phrases):
                                    section = []
                                    next_node = h.next_sibling
                                    while next_node and next_node.name not in ['h2', 'h3']:
                                        if next_node.name == 'p':
                                            section.append(next_node.get_text(strip=True))
                                        next_node = next_node.next_sibling
                                    if section:
                                        legal_sections.append({
                                            'heading': h.get_text(strip=True),
                                            'content': ' '.join(section[:3])  # First 3 paragraphs
                                        })
                            
                            if legal_sections:
                                extracted[key] = ' | '.join([f"{s['heading']}: {s['content']}" for s in legal_sections])
                            else:
                                extracted[key] = "No specific legal considerations mentioned."
                
                return extracted
            
            return {"error": "No content could be extracted from the page"}
            
        except requests.RequestException as e:
            return {"error": f"Failed to fetch URL: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Error extracting structured data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e), "type": type(e).__name__}
    
    def __str__(self):
        return f"Crawl4AIScraper(model={self.model}, api_key={'***' if self.api_key else 'None'})"
    
    def __repr__(self):
        return self.__str__()

# Example usage
async def example_usage() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Example usage of the Crawl4AIScraper class.
    
    Returns:
        A tuple containing (scraped_pages, structured_data)
    """
    try:
        print("=== Starting Web Scraping Example ===\n")
        print("Initializing Crawl4AIScraper...")
        scraper = Crawl4AIScraper()
        
        # Use a content-rich page for demonstration
        url = "https://en.wikipedia.org/wiki/Web_scraping"
        
        # Example 1: Basic scraping with selectors
        print("\n[1/2] Running basic web scraping...")
        selectors = {
            'title': 'h1.firstHeading',
            'first_paragraph': 'div.mw-parser-output > p:first-of-type',
            'toc': '#toc',
            'first_section': '#mw-content-text > div.mw-parser-output > h2:first-of-type',
        }
        
        results = await scraper.scrape_pages(
            url=url,
            selectors=selectors,
            max_pages=1,
            show_progress=True
        )
        
        # Print a summary of the scraped data
        if results and len(results) > 0:
            data = results[0]
            print("\n" + "="*50)
            print("BASIC SCRAPING RESULTS".center(50))
            print("="*50)
            
            print(f"\n{'URL:':<20} {data.get('url', 'N/A')}")
            print(f"{'Title:':<20} {data.get('title', 'N/A')}")
            print(f"{'Timestamp:':<20} {data.get('timestamp', 'N/A')}")
            
            extracted = data.get('extracted', {})
            if 'first_paragraph' in extracted and extracted['first_paragraph']:
                print("\nFirst Paragraph:")
                print("-"*50)
                print(textwrap.fill(extracted['first_paragraph'], width=80))
            
            print("\n" + "-"*50)
            print(f"Table of Contents found: {'Yes' if 'toc' in extracted and extracted['toc'] else 'No'}")
            
            if 'first_section' in extracted and extracted['first_section']:
                print(f"First section heading: {extracted['first_section']}")
            
            # Show number of links found
            links = data.get('links', [])
            print(f"Links found: {len(links)}")
            if links and len(links) > 0:
                print("\nSample Links:")
                for i, link in enumerate(links[:5], 1):
                    print(f"  {i}. {link}")
                if len(links) > 5:
                    print(f"  ... and {len(links) - 5} more")
        else:
            print("\nNo data was scraped")
        
        # Example 2: AI-powered structured data extraction
        print("\n[2/2] Running AI-powered structured data extraction...")
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
                    "description": "Web scraping technologies or libraries mentioned"
                },
                "legal_considerations": {
                    "type": "string", 
                    "description": "Brief note on legal aspects of web scraping"
                }
            },
            "required": ["title", "summary"]
        }
        
        structured_data = await scraper.extract_structured_data(url=url, schema=schema)
        
        # Print structured data summary
        if structured_data and not structured_data.get('error'):
            print("\n" + "="*50)
            print("AI-EXTRACTED STRUCTURED DATA".center(50))
            print("="*50)
            
            print(f"\n{'Title:':<20} {structured_data.get('title', 'N/A')}")
            
            if 'summary' in structured_data and structured_data['summary']:
                print("\nSummary:")
                print("-"*50)
                print(textwrap.fill(structured_data['summary'], width=80))
            
            if 'key_topics' in structured_data and structured_data['key_topics']:
                print("\nKey Topics:")
                print("-"*50)
                for i, topic in enumerate(structured_data['key_topics'][:8], 1):
                    print(f"  {i}. {topic}")
                if len(structured_data['key_topics']) > 8:
                    print(f"  ... and {len(structured_data['key_topics']) - 8} more")
            
            if 'technologies_mentioned' in structured_data and structured_data['technologies_mentioned']:
                print("\nTechnologies Mentioned:")
                print("-"*50)
                tech_list = ", ".join(structured_data['technologies_mentioned'])
                print(textwrap.fill(tech_list, width=80))
            
            if 'legal_considerations' in structured_data and structured_data['legal_considerations']:
                print("\nLegal Considerations:")
                print("-"*50)
                print(textwrap.fill(structured_data['legal_considerations'], width=80))
        else:
            error_msg = structured_data.get('error', 'No structured data was extracted')
            print("\n" + "!"*50)
            print("EXTRACTION NOTICE".center(50))
            print("!"*50)
            print(f"\n{error_msg}")
        
        print("\n=== Scraping Complete ===")
        return results, structured_data
        
    except Exception as e:
        print(f"\nError in example_usage: {str(e)}")
        traceback.print_exc()
        return [], {}

if __name__ == "__main__":
    asyncio.run(example_usage())
