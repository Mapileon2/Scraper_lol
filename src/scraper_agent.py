import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from typing import List, Dict, Any, Optional, Tuple
import time
import json
import re
import random
import streamlit as st
from dataclasses import dataclass
from collections import namedtuple
from fake_useragent import UserAgent

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
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """Initialize the Gemini model with proper error handling.
        Defaults to gemini-1.5-flash which has higher free-tier quotas than gemini-1.5-pro.
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 1000
            }
        )
        # Keep a cache of analyzed URLs
        self._selector_cache = {}

    def get_random_user_agent(self) -> str:
        """Generate a random user agent to avoid detection"""
        try:
            ua = UserAgent()
            return ua.random
        except Exception:
            # Fallback user agents if fake-useragent fails
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ]
            return random.choice(user_agents)
            
    def _is_review_element(self, element) -> bool:
        """Check if an element is likely to contain review text."""
        try:
            # Check element's class, id, and other attributes for review indicators
            attrs = [
                element.get_attribute('class', '').lower(),
                element.get_attribute('id', '').lower(),
                element.get_attribute('itemprop', '').lower(),
                element.get_attribute('data-testid', '').lower()
            ]
            
            review_indicators = [
                'review', 'comment', 'feedback', 'testimonial',
                'user-content', 'message', 'post', 'content',
                'description', 'text', 'body', 'article'
            ]
            
            # Check if any attribute contains a review indicator
            for attr in attrs:
                if any(indicator in attr for indicator in review_indicators):
                    return True
                    
            # Check element's tag name
            if element.tag_name.lower() in ['article', 'section', 'div', 'p', 'blockquote', 'li']:
                # Check if the element contains enough text to be a review
                text = element.text.strip() if element.text else ''
                word_count = len(text.split())
                return 10 <= word_count <= 2000  # Reasonable review length
                
            return False
        except Exception as e:
            print(f"Error checking review element: {str(e)}")
            return False

    def analyze_first_page(self, url: str, wait_time: int = 5, proxy: str = None) -> List[SelectorSuggestion]:
        """Analyzes a webpage and suggests CSS selectors for data extraction with improved reasoning.
        
        Args:
            url: The URL of the webpage to analyze
            wait_time: Time to wait for page to load (increased default)
            proxy: Optional proxy to use
            
        Returns:
            List of SelectorSuggestion objects
        """
        # Check cache first
        if url in self._selector_cache:
            return self._selector_cache[url]
            
        # Setup headless Chrome with enhanced options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--user-agent={self.get_random_user_agent()}")
        
        # Add proxy if provided
        if proxy:
            chrome_options.add_argument(f'--proxy-server={proxy}')
            
        try:
            # Use ChromeService instead of Service for better compatibility with Python 3.12
            driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            # Expanded wait strategy
            driver.get(url)
            # Initial static wait
            time.sleep(wait_time)
            
            # Wait for document ready state
            driver.execute_script("return document.readyState") == "complete"
            
            # Scroll down to load dynamic content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Get page HTML
            page_html = driver.page_source
            
            # Try executing a test to grab content directly to validate it worked
            test_elements = []
            try:
                # Try common container selectors
                for test_selector in [".review", ".product", ".item", ".card", "article", ".post", "ul li"]:
                    elements = driver.find_elements(By.CSS_SELECTOR, test_selector)
                    if elements and len(elements) > 1:  # If we find multiple elements
                        test_elements.append((test_selector, elements[0].text[:100] if elements[0].text else ""))
                        break
            except Exception:
                pass
                
            # Optimize HTML to reduce token usage - keep only a specific section if possible
            html_sample = self._extract_relevant_html_sample(page_html)
            
            # Enhanced reasoning prompt for better selector inference
            prompt = f"""
            Analyze this HTML from {url} and suggest precise CSS selectors for extracting structured data.
            
            First, determine what type of page this is: product listing, review page, article, blog, etc.  
            Then suggest specific CSS selectors for extracting meaningful data fields.
            
            HTML to analyze (partial):
            ```html
            {html_sample}
            ```
            
            IMPORTANT GUIDELINES:
            1. Focus on content-rich fields that would be valuable to extract (titles, descriptions, prices, ratings, etc.)
            2. For review sites like MouthShut, extract review titles, text content, ratings, and author information
            3. For e-commerce sites, prioritize product names, prices, ratings, and descriptions
            4. For article sites, focus on headlines, article text, and publication dates
            5. Use precise selectors that uniquely identify elements
            6. Suggest a container selector if this page has a list of similar items
            7. Suggest between 3-6 fields total, prioritizing the most important data
            
            For each field, provide:
            1. field: A descriptive name (e.g., review_title, product_name, price)
            2. selector: The exact CSS selector to extract this data
            3. sample_data: An example of the data that would be extracted
            4. confidence: Your confidence in this selector (0.0-1.0)
            
            Return a JSON array containing these fields:
            [
                {{"field": "review_title", "selector": ".review-title h3", "sample_data": "Great Service", "confidence": 0.9}},
                {{"field": "review_text", "selector": ".review-content p", "sample_data": "I had a wonderful experience...", "confidence": 0.85}}
            ]
            """
            
            # Add any test elements we found for context
            if test_elements:
                test_context = "\n\nI found these potential content elements:\n"
                for selector, text in test_elements:
                    test_context += f"Selector '{selector}' contains: {text}\n"
                prompt += test_context
            
            # Get suggestions from Gemini with retries for quota errors
            suggestions = []
            for attempt in range(3):  # Try up to 3 times
                try:
                    response = self.model.generate_content(prompt)
                    result_text = response.text
                    
                    # Extract JSON from the response using more robust pattern matching
                    json_match = re.search(r'\[\s*\{.*?\}.*?\]', result_text, re.DOTALL)
                    if json_match:
                        try:
                            suggestions_data = json.loads(json_match.group(0))
                            suggestions = []
                            for item in suggestions_data:
                                suggestions.append(SelectorSuggestion(
                                    field=item.get("field", ""),
                                    selector=item.get("selector", ""),
                                    sample_data=item.get("sample_data", ""),
                                    confidence=item.get("confidence", 0.0)
                                ))
                            # Verify selectors by testing them
                            suggestions = self._verify_selectors(driver, suggestions)
                            break  # Exit retry loop if successful
                        except Exception as e:
                            print(f"Error parsing suggestions: {str(e)}")
                            if attempt == 2:  # Final attempt - use fallback selectors
                                suggestions = self._get_fallback_selectors(driver)
                    else:
                        # Try to manually parse JSON-like content
                        print("No valid JSON found in response, trying manual extraction")
                        if attempt == 2:  # Final attempt - use fallback selectors
                            suggestions = self._get_fallback_selectors(driver)
                except Exception as e:
                    error_message = str(e)
                    print(f"Error on attempt {attempt+1}: {error_message}")
                    if "429" in error_message or "quota" in error_message.lower():
                        # API quota error - use fallback selectors
                        if attempt == 2:  # Final attempt
                            suggestions = self._get_fallback_selectors(driver)
                    time.sleep(2)  # Small delay before retry
            
            # Cache the results for this URL
            self._selector_cache[url] = suggestions
            return suggestions
                
        except Exception as e:
            print(f"Error analyzing page: {str(e)}")
            return self._get_fallback_selectors(None)  # Return fallback selectors
        finally:
            driver.quit()
            
    def _extract_relevant_html_sample(self, html: str) -> str:
        """Extract a relevant sample from the HTML to reduce token usage."""
        # Try to find common content containers and extract just that section
        content_patterns = [
            # Review and content specific patterns
            r'<div[^>]*class=[\"\'][^\"\']*?(?:review|comment|feedback|testimonial|content|message|post|text|body|article)[^\"\']*?[\"\'][^>]*>.*?</div>',
            r'<section[^>]*class=[\"\'][^\"\']*?(?:review|comment|feedback|testimonial|content|message|post|text|body|article)[^\"\']*?[\"\'][^>]*>.*?</section>',
            r'<article[^>]*>.*?</article>',
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:main|content|container|wrapper)[^\"\']*?[\"\'][^>]*>.*?</div>',
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:reviews?|comments?|feedback|testimonials?)[^\"\']*?[\"\'][^>]*>.*?</div>',
            
            # Common content structures
            r'<ul[^>]*class=[\"\'][^\"\']*?(?:list|items|products|reviews|comments)[^\"\']*?[\"\'][^>]*>.*?</ul>',
            r'<div[^>]*class=[\"\'][^\"\']*?item[^\"\']*?[\"\'][^>]*>.*?</div>',
            r'<div[^>]*class=[\"\'][^\"\']*?card[^\"\']*?[\"\'][^>]*>.*?</div>',
            
            # Fallback to main content areas
            r'<main[^>]*>.*?</main>',
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:main|primary|content|page)[^\"\']*?[\"\'][^>]*>.*?</div>',
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:main-content|primary-content|page-content)[^\"\']*?[\"\'][^>]*>.*?</div>',
            
            # Common e-commerce and review site patterns
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:product|item|listing)[^\"\']*?[\"\'][^>]*>.*?</div>',
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:review|comment)-container[^\"\']*?[\"\'][^>]*>.*?</div>',
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?user-content[^\"\']*?[\"\'][^>]*>.*?</div>',
            
            # Social media and forum patterns
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:post|message|comment|reply)[^\"\']*?[\"\'][^>]*>.*?</div>',
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:thread|discussion|conversation)[^\"\']*?[\"\'][^>]*>.*?</div>',
            
            # Blog and article patterns
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:entry|blog|article|post)[^\"\']*?[\"\'][^>]*>.*?</div>',
            r'<div[^>]*(?:id|class)=[\"\'][^\"\']*?(?:entry-content|post-content|article-content)[^\"\']*?[\"\'][^>]*>.*?</div>'
        ]
        
        # Try each pattern and return the first match
        for pattern in content_patterns:
            try:
                match = re.search(pattern, html, re.DOTALL)
                if match:
                    # Extract and clean the matched content
                    content = match.group(0)
                    # Remove script and style tags
                    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
                    # Limit length to reduce token usage
                    content = content[:10000]  # Limit to 10,000 characters
                    return content
            except Exception as e:
                print(f"Pattern {pattern} failed: {e}")
                continue
        
        # Fallback to full body if no specific patterns match
        try:
            body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL)
            if body_match:
                return body_match.group(1)[:10000]
        except Exception as e:
            print(f"Body extraction failed: {e}")
        
        # Last resort - return empty string or a portion of the HTML if it's very large
        total_len = len(html)
        if total_len > 10000:  # For very large documents
            start_pos = max(0, total_len // 2 - 2000)  # Start from middle minus 2000 chars
            return html[start_pos:start_pos + 4000]
        else:
            return html[:4000]  # Just return the first 4000 chars
    
    def _verify_selectors(self, driver, suggestions: List[SelectorSuggestion]) -> List[SelectorSuggestion]:
        """Test selectors on the page and update confidence based on results."""
        if not driver:
            return suggestions
            
        verified_suggestions = []
        for suggestion in suggestions:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, suggestion.selector)
                if elements:
                    # Selector works and found elements
                    sample_data = elements[0].text.strip() if elements[0].text else ""
                    if sample_data:  # We found real content
                        # Update with real sample data and high confidence
                        verified_suggestions.append(SelectorSuggestion(
                            field=suggestion.field,
                            selector=suggestion.selector,
                            sample_data=sample_data[:100],  # Limit length
                            confidence=min(1.0, suggestion.confidence + 0.1)  # Increase confidence
                        ))
                    else:
                        # Element exists but no text content
                        verified_suggestions.append(SelectorSuggestion(
                            field=suggestion.field,
                            selector=suggestion.selector,
                            sample_data="[Empty content]",
                            confidence=suggestion.confidence - 0.1  # Reduce confidence slightly
                        ))
                else:
                    # Selector doesn't work
                    verified_suggestions.append(SelectorSuggestion(
                        field=suggestion.field,
                        selector=suggestion.selector,
                        sample_data="[No matching elements]",
                        confidence=max(0.1, suggestion.confidence - 0.2)  # Reduce confidence more
                    ))
            except Exception:
                # Selector is invalid
                verified_suggestions.append(SelectorSuggestion(
                    field=suggestion.field,
                    selector=suggestion.selector,
                    sample_data="[Invalid selector]",
                    confidence=max(0.1, suggestion.confidence - 0.3)  # Reduce confidence significantly
                ))
        
        return verified_suggestions
    
    def _get_fallback_selectors(self, driver) -> List[SelectorSuggestion]:
        """Generate fallback selectors when AI inference fails."""
        fallback_selectors = [
            SelectorSuggestion(field="title", selector="h1, .title, .heading", sample_data="", confidence=0.5),
            SelectorSuggestion(field="content", selector="p, .content, .description", sample_data="", confidence=0.5),
            SelectorSuggestion(field="container", selector=".item, article, .product, .review, li", sample_data="", confidence=0.4)
        ]
        
        # If we have a driver, try to validate these selectors
        if driver:
            return self._verify_selectors(driver, fallback_selectors)
            
        return fallback_selectors
            
    def scrape_pages(self, url: str, selectors: Dict[str, str], max_pages: int = 3, wait_time: int = 5,
                    pagination_strategy: str = "auto", proxy: str = None, retry_limit: int = 2) -> List[Dict]:
        """
        Enhanced scraper that extracts detailed data from pages using the provided selectors.
        
        Args:
            url: The starting URL
        
        driver = None
        results = []
        current_page = 1
        
        try:
            print(f"🚀 Starting scraping: {url}")
            print(f"Pagination strategy: {pagination_strategy}")
            print(f"Selectors: {selectors}")
            
            # Initialize WebDriver
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            while current_page <= max_pages:
                print(f"\n📄 Processing page {current_page}/{max_pages}...")
                
                # Navigate to the page
                if current_page == 1:
                    print(f"🌐 Loading initial URL: {url}")
                    driver.get(url)
                else:
                    if pagination_strategy == "url_increment":
                        # Handle URL-based pagination
                        if "?" in url:
                            page_url = f"{url}&page={current_page}"
                    time.sleep(1)
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    
                    # Extract data using selectors
                    page_results = self._extract_page_data(driver, selectors, current_page, current_url)
                    
                    if page_results:
                        results.extend(page_results)
                        retry_count = 0  # Reset retry count on success
                    else:
                        # If no results on this page, increment retry count
                        retry_count += 1
                        print(f"No results found on page {current_page}, retry {retry_count}")
                        if retry_count > retry_limit:
                            break
                        
                    # Handle pagination based on strategy
                    if pagination_strategy == "url_increment" and url_pattern:
                        current_page += 1
                        current_url = url_pattern.format(page_num=current_page)
                    else:
                        # Find the next page using buttons/links
                        next_url = self._find_next_page(driver, pagination_strategy)
                        if next_url:
                            current_url = next_url
                            current_page += 1
                        else:
                            break  # No more pages found
                    
                    # Add a small random delay between pages to avoid detection
                    delay = random.uniform(1.0, 3.0)
                    time.sleep(delay)
                    
                except Exception as e:
                    print(f"Error on page {current_page}: {str(e)}")
                    retry_count += 1
                    if retry_count > retry_limit:
                        print(f"Exceeded retry limit after {retry_limit} attempts")
                        break
                    time.sleep(2)  # Wait before retry
            
            # Format results using ScrapedDataInput structure
            formatted_results = []
            for item in results:
                # Separate metadata from data fields
                data = {k: v for k, v in item.items() if k not in ["page_num", "url"]}
                input_data = ScrapedDataInput(
                    data=data,
                    url=item.get("url", url),
                    page_num=item.get("page_num", 1)
                )
                formatted_results.append(input_data.dict())
                
            return formatted_results
            
        except Exception as e:
            print(f"Error scraping pages: {str(e)}")
            return results
        finally:
            if driver:
                driver.quit()
    
    def _extract_review_content(self, element, driver) -> str:
        """Extract review content from an element with fallback methods."""
        try:
            # First try to get the text directly
            text = element.text.strip()
            if text and len(text.split()) > 5:  # At least 5 words
                return text

            # If direct text is too short, try to find text in child elements
            paragraphs = []
            
            # Look for common content containers
            try:
                content_elements = element.find_elements(self.By.XPATH, 
                    ".//*[self::p or self::div or self::section or self::article or self::blockquote]")
                
                for el in content_elements:
                    try:
                        el_text = el.text.strip()
                        if el_text and len(el_text.split()) > 3:  # At least 3 words
                            paragraphs.append(el_text)
                    except Exception as e:
                        print(f"Warning: Error extracting text from element: {str(e)}")
                        continue
                        
                if paragraphs:
                    return '\n\n'.join(paragraphs)
            except Exception as e:
                print(f"Warning: Error finding content elements: {str(e)}")

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
            except Exception as e:
                print(f"Warning: Error executing JavaScript for text extraction: {str(e)}")

            return text if text else ""

        except Exception as e:
            print(f"Error extracting review content: {str(e)}")
            return ""
    
    def _extract_page_data(self, driver, selectors: Dict[str, str], page_num: int, url: str) -> List[Dict]:
        """Extract data from the current page using provided selectors with improved error handling."""
        # Import required modules
        import time
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, WebDriverException
        
        results = []
        
        # Log the extraction start with page number and selectors
        print(f"🔍 Extracting data from page {page_num} with selectors: {selectors}")
        
        try:
            # Wait for dynamic content to load
            time.sleep(2)  # Initial wait
            
            # Try to find containers that might hold the data
            container_candidates = [
                "div[data-testid='review']",  # Common in review sites
                "div.review",                 # Common class name
                "div[itemprop='review']",     # Schema.org markup
                "div[class*='review']",       # Partial class match
                "div[class*='item']",         # More generic
                "div[class*='card']",         # Card-based layouts
                "article",                    # HTML5 article tag
                "div[role='article']",        # ARIA role
            ]
            
            containers = []
            for selector in container_candidates:
                try:
                    elements = WebDriverWait(driver, 5).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    if elements:
                        print(f"Found {len(elements)} containers with selector: {selector}")
                        containers.extend(elements)
                        break  # Use the first selector that finds containers
                except (TimeoutException, Exception) as e:
                    print(f"Selector '{selector}' didn't match any elements")
                    continue
            
            # If no containers found, try to use the whole page
            if not containers:
                print("No containers found, using body as container")
                containers = [driver.find_element(By.TAG_NAME, "body")]
            
            # Process each container
            for i, container in enumerate(containers[:20]):  # Limit to first 20 containers
                try:
                    if not container.is_displayed():
                        print(f"Container {i+1} is not visible, skipping...")
                        continue
                        
                    item_data = {"page_num": page_num, "url": url, "container_index": i}
                    has_data = False
                    
                    # Scroll to the element to ensure it's in view
                    try:
                        driver.execute_script("arguments[0].scrollIntoView(true);", container)
                        time.sleep(0.5)  # Small delay for any lazy loading
                    except:
                        pass
                    
                    # Extract data using provided selectors
                    for field, selector in selectors.items():
                        try:
                            # First try relative to container
                            elements = container.find_elements(By.CSS_SELECTOR, selector)
                            
                            # If no elements found, try absolute selector
                            if not elements:
                                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            
                            if elements:
                                # Get text from all matching elements
                                texts = [el.text.strip() for el in elements if el.text.strip()]
                                if texts:
                                    # Join multiple matches with newlines
                                    item_data[field] = "\n".join(texts)
                                    has_data = True
                                else:
                                    # Try to get attribute if text is empty
                                    for attr in ['data-content', 'content', 'value', 'title']:
                                        attr_texts = [el.get_attribute(attr) for el in elements 
                                                  if el.get_attribute(attr)]
                                        if attr_texts:
                                            item_data[f"{field}_{attr}"] = "\n".join(attr_texts)
                                            has_data = True
                                            break
                        except Exception as e:
                            print(f"Error extracting {field}: {str(e)}")
                            continue
                    
                    # If we got any data, add it to results
                    if has_data:
                        # Clean up the data
                        cleaned_data = {}
                        for k, v in item_data.items():
                            if isinstance(v, str):
                                # Remove extra whitespace and newlines
                                v = ' '.join(v.split())
                                # Remove common unwanted text
                                v = re.sub(r'\s*Read (more|less)\s*', ' ', v, flags=re.IGNORECASE)
                                v = v.strip()
                            cleaned_data[k] = v
                        
                        print(f"✅ Extracted data: {list(cleaned_data.keys())}")
                        results.append(cleaned_data)
                    else:
                        print(f"⚠️ No data extracted from container {i+1}")
                
                except Exception as e:
                    print(f"Error processing container {i+1}: {str(e)}")
                    continue
            
            # If no data found with selectors, try to extract any text content
            if not results:
                print("⚠️ No data found with provided selectors, trying fallback extraction...")
                try:
                    # Try to get main content areas
                    content_selectors = [
                        "main", 
                        "article", 
                        "div[role='main']",
                        "div.container", 
                        "div#content",
                        "div.main-content",
                        "body"
                    ]
                    
                    for selector in content_selectors:
                        try:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            for i, el in enumerate(elements[:3]):  # Limit to first 3 matches
                                text = el.text.strip()
                                if len(text.split()) > 10:  # Only include if there's substantial text
                                    results.append({
                                        "page_num": page_num,
                                        "url": url,
                                        "content": text[:1000] + "..." if len(text) > 1000 else text
                                    })
                                    break
                            if results:
                                break
                        except:
                            continue
                except Exception as e:
                    print(f"Fallback extraction failed: {str(e)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in _extract_page_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
        def get_element_text(element, selector):
            """Helper function to safely get text from an element."""
            try:
                # First try direct text extraction
                text = element.text.strip()
                if text:
                    return text
                    
                # If no text, try to get value attribute for input elements
                text = element.get_attribute("value")
                if text and text.strip():
                    return text.strip()
                    
                # Try to get text from child elements
                child_text = " ".join([e.text.strip() for e in element.find_elements(By.XPATH, ".//*[text()]")
                                     if e.text.strip()])
                if child_text:
                    return child_text
                    
                # Try to get text using JavaScript (as a last resort)
                try:
                    js_text = driver.execute_script("return arguments[0].innerText || arguments[0].textContent;", element)
                    if js_text and js_text.strip():
                        return js_text.strip()
                except:
                    pass
                    
                return ""
            except Exception as e:
                print(f"Error getting text from element: {str(e)}")
                return ""
        
        try:
            if container_selector:
                # If there's a container selector, find all containers and extract data
                print(f"Looking for containers with selector: {container_selector}")
                containers = []
                
                # Try the provided container selector first
                containers = driver.find_elements(By.CSS_SELECTOR, container_selector)
                print(f"Found {len(containers)} containers with provided selector")
                
                # If no containers found, try alternative common container selectors
                if not containers or len(containers) < 2:  # Require at least 2 containers to be sure
                    print("Trying alternative container selectors...")
                    alternative_selectors = [
                        ".review", ".product", "article", 
                        ".item", ".card", "ul li", 
                        ".list-item", ".result", ".search-result",
                        ".post", ".entry", ".product-item"
                    ]
                    
                    for alt_selector in alternative_selectors:
                        if alt_selector == container_selector:
                            continue  # Skip if same as provided selector
                            
                        try:
                            alt_containers = driver.find_elements(By.CSS_SELECTOR, alt_selector)
                            if alt_containers and len(alt_containers) > 1:  # At least 2 to ensure it's really a container
                                print(f"Found {len(alt_containers)} containers with alternative selector: {alt_selector}")
                                containers = alt_containers
                                container_selector = alt_selector  # Update the container selector for this page
                                break
                        except Exception as e:
                            print(f"Error trying alternative selector {alt_selector}: {str(e)}")
                
                # Process each container
                print(f"Processing {len(containers)} containers...")
                items_processed = 0
                max_items_to_process = 50  # Limit to prevent timeout on pages with many items
                
                for i, container in enumerate(containers):
                    try:
                        if items_processed >= max_items_to_process:
                            print(f"Reached maximum items limit ({max_items_to_process}), stopping processing")
                            break
                            
                        if not container.is_displayed():
                            print(f"Container {i+1} is not visible, skipping...")
                            continue
                            
                        # For review scraping, check if this looks like a review container
                        if is_review_scraping and not self._is_review_element(container):
                            print(f"Container {i+1} doesn't appear to be a review, skipping...")
                            continue
                            
                        # Create data item with metadata
                        item_data = {"page_num": page_num, "url": url}
                        has_data = False
                        
                        # Extract data fields from this container
                        for field, selector in expanded_selectors.items():
                            if field == "container" or field.endswith("_alt"):
                                continue
                                
                            try:
                                # Special handling for review content
                                if field.lower() in ['review', 'comment', 'content', 'text', 'description']:
                                    try:
                                        # First try the specific selector
                                        elements = container.find_elements(By.CSS_SELECTOR, selector)
                                        if elements:
                                            review_text = self._extract_review_content(elements[0], driver)
                                            if review_text:
                                                item_data[field] = review_text
                                                has_data = True
                                                continue
                                        
                                        # Try alternative selectors for review content
                                        alt_selectors = [
                                            f"[itemprop='{field}']",
                                            f"[data-{field}]",
                                            f"[class*='{field}']",
                                            f"[id*='{field}']",
                                            "p, div, span, section, article"
                                        ]
                                        
                                        for alt_selector in alt_selectors:
                                            try:
                                                elements = container.find_elements(By.CSS_SELECTOR, alt_selector)
                                                if elements:
                                                    for el in elements[:3]:  # Check first few elements
                                                        review_text = self._extract_review_content(el, driver)
                                                        if review_text and len(review_text.split()) > 5:
                                                            item_data[field] = review_text
                                                            has_data = True
                                                            raise StopIteration  # Break out of both loops
                                            except StopIteration:
                                                break
                                            except Exception:
                                                continue
                                                
                                    except Exception as e:
                                        print(f"Error extracting review content: {str(e)}")
                                
                                # Standard field extraction
                                try:
                                    # Try relative selector first (within container)
                                    elements = container.find_elements(By.CSS_SELECTOR, selector)
                                    if not elements and ' ' in selector:
                                        # Try with direct child selector if the selector has spaces
                                        elements = container.find_elements(By.CSS_SELECTOR, 
                                                                         ' > '.join(selector.split()))
                                    
                                    if elements:
                                        text = get_element_text(elements[0], selector)
                                        if text:
                                            item_data[field] = text
                                            has_data = True
                                            continue  # Skip to next field if found
                                except Exception as e:
                                    print(f"Relative selector failed for {field}: {str(e)}")
                                
                                # If relative selector fails, try absolute selector
                                try:
                                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                                    if elements:
                                        text = get_element_text(elements[0], selector)
                                        if text:
                                            item_data[field] = text
                                            has_data = True
                                except Exception as e:
                                    print(f"Absolute selector failed for {field}: {str(e)}")
                                
                                # If still no data, try to find any element with similar class/ID
                                if not item_data.get(field):
                                    try:
                                        # Try to find any element with field name in class or id
                                        field_selector = f"[class*='{field}'], [id*='{field}']"
                                        elements = container.find_elements(By.CSS_SELECTOR, field_selector)
                                        if not elements:
                                            elements = driver.find_elements(By.CSS_SELECTOR, field_selector)
                                        
                                        if elements:
                                            text = get_element_text(elements[0], field_selector)
                                            if text:
                                                item_data[field] = text
                                                has_data = True
                                    except Exception:
                                        pass
                                        
                            except Exception as e:
                                print(f"Error extracting field '{field}': {str(e)}")
                                item_data[field] = ""
                        
                        # Only add if we have actual data beyond metadata
                        if has_data:
                            print(f"Adding item {i+1}")
                            if is_review_scraping:
                                # For review content, ensure we have enough text
                                review_fields = [f for f in item_data.keys() if f.lower() in ['review', 'comment', 'content']]
                                if review_fields and any(len(str(item_data.get(f, '')).split()) < 5 for f in review_fields):
                                    print(f"Skipping item {i+1} - review text too short")
                                    continue
                                    
                                # Clean up the review text
                                for field in review_fields:
                                    text = item_data[field]
                                    # Remove excessive whitespace and normalize
                                    text = ' '.join(str(text).split())
                                    # Remove common boilerplate
                                    text = re.sub(r'\b(?:read more|read less|show more|show less)\b', '', text, flags=re.IGNORECASE)
                                    item_data[field] = text.strip()
                            
                            print(f"Adding item {i+1} with data: {item_data}")
                            page_results.append(item_data)
                            items_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing container {i+1}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                print(f"Extracted {len(page_results)} items from containers")
                
            else:
                # If there's no container, extract data directly from the page (single item page)
                print("No container selector, extracting data directly from page")
                item_data = {"page_num": page_num, "url": url}
                has_data = False
                
                for field, selector in expanded_selectors.items():
                    if field == "container" or field.endswith("_alt"):
                        continue
                        
                    try:
                        # Special handling for review content
                        if field.lower() in ['review', 'comment', 'content', 'text', 'description']:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements:
                                review_text = self._extract_review_content(elements[0], driver)
                                if review_text:
                                    item_data[field] = review_text
                                    has_data = True
                                    continue
                            
                            # Try alternative selectors for review content
                            alt_selectors = [
                                f"[itemprop='{field}']",
                                f"[data-{field}]",
                                f"[class*='{field}']",
                                f"[id*='{field}']",
                                "p, div, section, article"
                            ]
                            
                            for alt_selector in alt_selectors:
                                try:
                                    elements = driver.find_elements(By.CSS_SELECTOR, alt_selector)
                                    if elements:
                                        for el in elements[:5]:  # Check first few elements
                                            review_text = self._extract_review_content(el, driver)
                                            if review_text and len(review_text.split()) > 5:
                                                item_data[field] = review_text
                                                has_data = True
                                                raise StopIteration  # Break out of both loops
                                except StopIteration:
                                    break
                                except Exception:
                                    continue
                                    
                        # Standard field extraction
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            text = get_element_text(elements[0], selector)
                            if text:
                                item_data[field] = text
                                has_data = True
                                continue
                        
                        # If no elements found, try case-insensitive attribute matching
                        if '=' in selector:
                            # Try to find elements with attribute containing the field name
                            attr_selector = f"[class*='{field}'], [id*='{field}'], [itemprop*='{field}'], [data-{field}]"
                            elements = driver.find_elements(By.CSS_SELECTOR, attr_selector)
                            if elements:
                                text = get_element_text(elements[0], attr_selector)
                                if text:
                                    item_data[field] = text
                                    has_data = True
                                    
                    except Exception as e:
                        print(f"Error extracting field '{field}': {str(e)}")
                        item_data[field] = ""
                
                # Add the item if it has any non-empty fields
                if has_data:
                    print(f"Adding single item with data: {item_data}")
                    page_results.append(item_data)
                
        except Exception as e:
            print(f"Error in _extract_page_data: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"Total items extracted: {len(page_results)}")
        return page_results
    
    def _find_next_page(self, driver, pagination_strategy: str) -> Optional[str]:
        """Find the URL for the next page based on the pagination strategy."""
        from typing import Optional
        
        if pagination_strategy == "url_increment":
            return None  # Handled by the main loop
            
        # Define selectors based on pagination strategy
        if pagination_strategy in ["next_button", "auto"]:
            next_page_selectors = [
                "a[rel='next']", 
                ".pagination .next a", 
                "a.next", 
                ".next a",
                ".pagination-next",
                "li.next a",
                "a:contains('Next')",
                "a[aria-label='Next']",
                "a.page-link[aria-label='Next']",
                ".paginate-next"
            ]
            
            # Try each selector
            for selector in next_page_selectors:
                try:
                    # Replace :contains which is jQuery selector with more generic approach
                    if ":contains" in selector:
                        # Try to find elements with text 'Next'
                        elements = driver.find_elements(By.TAG_NAME, "a")
                        for element in elements:
                            if element.text and "next" in element.text.lower():
                                if element.is_displayed() and element.get_attribute("href"):
                                    return element.get_attribute("href")
                    else:
                        # Use standard CSS selector
                        element = driver.find_element(By.CSS_SELECTOR, selector)
                        if element.is_displayed() and element.get_attribute("href"):
                            return element.get_attribute("href")
                except Exception as e:
                    print(f"Warning: Failed to find next page with selector {selector}: {str(e)}")
                    continue
        
        # If we're using 'auto' strategy, also check for numbered pagination
        if pagination_strategy == "auto":
            try:
                # Find current page number in pagination
                current_elements = driver.find_elements(
                    By.CSS_SELECTOR,
                    ".pagination .active, .pagination .current, .page-item.active, .selected"
                )
                
                if current_elements:
                    current_page_element = current_elements[0]
                    # Get the current page number
                    try:
                        current_text = current_page_element.text.strip()
                        match = re.search(r'\d+', current_text)
                        if match:
                            current_num = int(match.group())
                            next_num = current_num + 1
                            
                            # Look for an element with the next number
                            page_links = driver.find_elements(
                                By.CSS_SELECTOR, 
                                ".pagination a, .page-item a"
                            )
                            for link in page_links:
                                if link.text and str(next_num) == link.text.strip():
                                    return link.get_attribute("href")
                    except Exception as e:
                        print(f"Warning: Error processing pagination numbers: {str(e)}")
            except Exception as e:
                print(f"Warning: Error in auto pagination detection: {str(e)}")
        
        return None