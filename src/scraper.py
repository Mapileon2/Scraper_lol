import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import streamlit as st
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import json
import re
from fake_useragent import UserAgent
from urllib.parse import urlparse

# Dynamic pagination handler
async def detect_pagination(page, pagination_strategy="auto"):
    """
    Detects and handles different pagination patterns.
    
    Args:
        page: Playwright page object
        pagination_strategy: Can be "auto", "next_button", "infinite_scroll", or a custom CSS selector
    
    Returns:
        Next page URL or None if no next page found
    """
    if pagination_strategy == "auto":
        # Try multiple pagination patterns
        # Check for next button
        next_button = await page.query_selector('a[rel="next"], button.next, a.next, .pagination-next, li.next a')
        if next_button:
            return await next_button.get_attribute('href')
        
        # Check for numbered pagination
        current_page = await page.query_selector('.pagination .active, .pagination .current')
        if current_page:
            next_page_num = None
            try:
                # Get the current page number
                current_text = await current_page.text_content()
                current_num = int(current_text.strip())
                next_page_num = current_num + 1
            except (ValueError, TypeError):
                pass
                
            if next_page_num:
                # Look for the next page number in pagination
                next_page = await page.query_selector(f'.pagination a:text("{next_page_num}")')
                if next_page:
                    return await next_page.get_attribute('href')
        
        # Handle infinite scroll by scrolling to bottom
        initial_height = await page.evaluate('document.body.scrollHeight')
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        await page.wait_for_timeout(2000)  # Wait for 2 seconds for content to load
        new_height = await page.evaluate('document.body.scrollHeight')
        
        # If page height increased, content was loaded dynamically
        if new_height > initial_height:
            return "infinite_scroll"
        
        return None
    elif pagination_strategy == "infinite_scroll":
        # Just scroll and return the same "page"
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        await page.wait_for_timeout(2000)
        return "infinite_scroll"
    elif pagination_strategy == "next_button":
        # Try standard next buttons
        for selector in ['a[rel="next"]', 'button.next', 'a.next', '.pagination-next', 'li.next a', 'a:text("Next")']:
            next_button = await page.query_selector(selector)
            if next_button:
                return await next_button.get_attribute('href')
        return None
    else:
        # Custom selector provided
        next_button = await page.query_selector(pagination_strategy)
        if next_button:
            return await next_button.get_attribute('href')
        return None

def scrape_and_index(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless Chrome
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # Try to extract title and description
    try:
        title = driver.find_element(By.TAG_NAME, 'h1').text
    except Exception:
        title = ''
    try:
        description = driver.find_element(By.TAG_NAME, 'p').text
    except Exception:
        description = ''

    driver.quit()

    st.info(f"Extracted data: {{'title': title, 'description': description}}")
    if not title or not description:
        st.error(f"Failed to extract title or description from the webpage.")
        return None

    text_data = f"{title} {description}"
    embedding = generate_gemini_embedding(text_data)
    st.info(f"Embedding: {embedding}")
    if embedding:
        return {
            "title": title,
            "description": description,
            "embedding": embedding
        }
    else:
        st.error("Failed to generate embedding.")
        return None

def generate_gemini_embedding(text):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

async def handle_authentication(page, auth_config):
    """
    Handle website authentication using different strategies.
    
    Args:
        page: Playwright page object
        auth_config: Dict containing authentication config:
            - type: "basic", "form", "cookie", "oauth"
            - username: Username for form/basic auth
            - password: Password for form/basic auth
            - selectors: Dict of selectors for form elements
            - cookies: List of cookie objects to set
            - token: OAuth token
    
    Returns:
        Boolean indicating if authentication was successful
    """
    auth_type = auth_config.get("type", "").lower()
    
    if auth_type == "form":
        # Handle form-based authentication
        username = auth_config.get("username")
        password = auth_config.get("password")
        selectors = auth_config.get("selectors", {})
        
        username_selector = selectors.get("username", 'input[type="email"], input[name="email"], input[name="username"]')
        password_selector = selectors.get("password", 'input[type="password"]')
        submit_selector = selectors.get("submit", 'button[type="submit"], input[type="submit"]')
        
        try:
            # Navigate to login page if provided, otherwise assume we're already there
            if "login_url" in auth_config:
                await page.goto(auth_config["login_url"], wait_until="networkidle")
            
            # Fill in the login form
            await page.fill(username_selector, username)
            await page.fill(password_selector, password)
            await page.click(submit_selector)
            
            # Wait for navigation to complete
            await page.wait_for_load_state("networkidle")
            
            # Check for authentication failure indicators
            auth_failure_texts = ["incorrect password", "login failed", "authentication failed"]
            page_content = await page.content()
            if any(text in page_content.lower() for text in auth_failure_texts):
                return False
                
            return True
        except Exception as e:
            st.error(f"Form authentication error: {str(e)}")
            return False
            
    elif auth_type == "cookie":
        # Handle cookie-based authentication
        cookies = auth_config.get("cookies", [])
        
        try:
            if isinstance(cookies, str):
                # Parse cookies from string
                cookies = json.loads(cookies)
            
            # Set cookies
            for cookie in cookies:
                await page.context.add_cookies([cookie])
            
            return True
        except Exception as e:
            st.error(f"Cookie authentication error: {str(e)}")
            return False
            
    elif auth_type == "basic":
        # Handle HTTP Basic authentication
        username = auth_config.get("username")
        password = auth_config.get("password")
        
        try:
            # Set HTTP authentication
            await page.context.set_http_credentials({
                "username": username,
                "password": password
            })
            return True
        except Exception as e:
            st.error(f"Basic authentication error: {str(e)}")
            return False
            
    elif auth_type == "oauth":
        # Handle OAuth token authentication
        token = auth_config.get("token")
        token_type = auth_config.get("token_type", "Bearer")
        
        try:
            # Set authorization header for all requests
            await page.set_extra_http_headers({
                "Authorization": f"{token_type} {token}"
            })
            return True
        except Exception as e:
            st.error(f"OAuth authentication error: {str(e)}")
            return False
    
    return False

async def infer_selectors_with_gemini(page_content, api_key, business_data=True):
    """
    Use Gemini to infer the CSS selectors for common business data elements 
    in the provided HTML content.
    
    Args:
        page_content: HTML content of the page
        api_key: Gemini API key
        business_data: Whether to focus on business data selectors (True) or generic content (False)
    
    Returns:
        Dictionary of inferred selectors
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Truncate HTML to avoid exceeding context length
    max_length = 30000
    truncated_html = page_content[:max_length]
    
    if business_data:
        prompt = f"""
        Analyze this HTML and identify the CSS selectors for business data.
        Return ONLY a JSON object with these keys:
        - business_name: CSS selector for business/company name
        - address: CSS selector for business address
        - phone: CSS selector for phone number
        - website: CSS selector for website link
        - email: CSS selector for email address
        - description: CSS selector for business description
        
        Focus on the most specific selectors that will uniquely identify each element.
        Include classes and IDs where available, or use parent-child relationships.
        
        HTML to analyze:
        {truncated_html}
        """
    else:
        prompt = f"""
        Analyze this HTML and identify the main content selectors.
        Return ONLY a JSON object with these keys:
        - title: CSS selector for the main title
        - content: CSS selector for the main content block
        - image: CSS selector for main images
        - link: CSS selector for important links
        - date: CSS selector for date information if present
        
        Focus on the most specific selectors that will uniquely identify each element.
        Include classes and IDs where available, or use parent-child relationships.
        
        HTML to analyze:
        {truncated_html}
        """
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text
        
        # Extract JSON using regex (in case the model includes extra text)
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            selectors = json.loads(json_str)
            return selectors
        else:
            # Fallback to default selectors
            if business_data:
                return {
                    "business_name": ".business-name, h1, h2.title",
                    "address": ".address, .location, address",
                    "phone": ".phone, .tel, [href^='tel:']",
                    "website": ".website a, .website, a.website",
                    "email": ".email, [href^='mailto:']",
                    "description": ".description, .about, p"
                }
            else:
                return {
                    "title": "h1, .title, .headline",
                    "content": "article, .content, main",
                    "image": "img.featured, .main-image, article img",
                    "link": "a.link, .links a, .nav a",
                    "date": ".date, time, .published"
                }
    except Exception as e:
        st.error(f"Error inferring selectors: {str(e)}")
        # Return fallback selectors
        if business_data:
            return {
                "business_name": ".business-name, h1, h2.title",
                "address": ".address, .location, address",
                "phone": ".phone, .tel, [href^='tel:']",
                "website": ".website a, .website, a.website",
                "email": ".email, [href^='mailto:']",
                "description": ".description, .about, p"
            }
        else:
            return {
                "title": "h1, .title, .headline",
                "content": "article, .content, main",
                "image": "img.featured, .main-image, article img",
                "link": "a.link, .links a, .nav a",
                "date": ".date, time, .published"
            }

async def test_selectors(page, selectors):
    """
    Test selectors on the page and refine them if needed.
    
    Args:
        page: Playwright page
        selectors: Dict of selectors to test
        
    Returns:
        Refined selectors
    """
    refined_selectors = {}
    
    for field, selector in selectors.items():
        # Try each selector
        if "," in selector:
            # Multiple alternative selectors
            selector_alternatives = [s.strip() for s in selector.split(",")]
            for alt_selector in selector_alternatives:
                element = await page.query_selector(alt_selector)
                if element:
                    refined_selectors[field] = alt_selector
                    break
            # If none worked, keep the original
            if field not in refined_selectors:
                refined_selectors[field] = selector
        else:
            # Single selector
            element = await page.query_selector(selector)
            if element:
                refined_selectors[field] = selector
            else:
                # If selector doesn't work, try a more generic one
                if field == "business_name":
                    refined_selectors[field] = "h1, h2, .title"
                elif field == "address":
                    refined_selectors[field] = "address, .address, p:contains('Street')"
                elif field == "phone":
                    refined_selectors[field] = "[href^='tel:'], .phone, p:contains('Phone')"
                else:
                    refined_selectors[field] = selector
    
    return refined_selectors

def get_random_user_agent():
    """
    Generate a random user agent string to avoid detection.
    """
    try:
        ua = UserAgent()
        return ua.random
    except Exception:
        # Fallback user agents if fake_useragent fails
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36 Edg/91.0.864.41"
        ]
        return random.choice(user_agents)

async def detect_and_handle_blocks(page, url):
    """
    Detect common blocking or anti-bot mechanisms and attempt to bypass them.
    
    Args:
        page: Playwright page object
        url: URL being scraped
        
    Returns:
        Boolean indicating if the page loaded successfully
    """
    # Check for CAPTCHA presence
    captcha_selectors = [
        "iframe[src*='captcha']", 
        "iframe[src*='recaptcha']",
        ".captcha", 
        "#captcha",
        "div[class*='captcha']",
        "div[id*='captcha']"
    ]
    
    for selector in captcha_selectors:
        captcha = await page.query_selector(selector)
        if captcha:
            st.warning(f"CAPTCHA detected on {url}. Cannot proceed automatically.")
            return False
    
    # Check for common bot detection screens
    bot_detection_texts = [
        "verify you are a human",
        "bot detected",
        "automated access",
        "access denied",
        "forbidden",
        "unusual traffic",
        "security check"
    ]
    
    page_content = await page.content()
    page_text = await page.evaluate("document.body.innerText")
    
    for text in bot_detection_texts:
        if text.lower() in page_text.lower():
            st.warning(f"Bot detection triggered on {url}: '{text}'")
            return False
    
    # Check HTTP status code
    response = await page.evaluate("""() => {
        return {
            status: window.performance.getEntriesByType('navigation')[0].responseStatus,
        }
    }""")
    
    if response.get("status") in [403, 429, 503]:
        st.warning(f"Received blocking status code {response.get('status')} from {url}")
        return False
    
    # If we passed all checks, the page loaded successfully
    return True

async def apply_rate_limiting(domain, rate_limits):
    """
    Apply rate limiting for a specific domain.
    
    Args:
        domain: Website domain
        rate_limits: Dictionary of domain-specific rate limits
        
    Returns:
        None
    """
    # Get domain-specific rate limit or use default
    if domain in rate_limits:
        min_delay = rate_limits[domain].get("min_delay", 2)
        max_delay = rate_limits[domain].get("max_delay", 5)
    else:
        # Default rate limits
        min_delay = 2
        max_delay = 5
    
    # Random delay
    delay = random.uniform(min_delay, max_delay)
    await asyncio.sleep(delay)

# Modified to use Selenium instead of Playwright to avoid asyncio issues in Python 3.12
def selenium_scrape(url, max_pages=3, pagination_strategy="auto", selectors=None, wait_time=2, 
                    retry_limit=3):
    """
    Simplified version of advanced_scrape using Selenium instead of Playwright.
    
    Args:
        url: Starting URL to scrape
        max_pages: Maximum number of pages to scrape
        pagination_strategy: Strategy for pagination
        selectors: Dictionary of CSS selectors for data extraction
        wait_time: Time to wait after page loads in seconds
        retry_limit: Number of retries if scraping fails
        
    Returns:
        List of extracted data items
    """
    results = []
    
    if selectors is None:
        selectors = {
            "title": "h1",
            "description": "p",
            "business_name": ".business-name, .company-name, h3",
            "address": ".address, .location",
            "phone": ".phone, .tel",
            "website": "a.website, .website a",
        }
    
    # Initialize attempt counter
    attempt = 0
    
    while attempt < retry_limit:
        try:
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Add random user agent
            user_agent = UserAgent().random
            chrome_options.add_argument(f"--user-agent={user_agent}")
            
            # Initialize driver
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)  # 30 seconds timeout
            
            current_url = url
            page_count = 0
            
            while current_url and page_count < max_pages:
                # Navigate to URL
                driver.get(current_url)
                
                # Wait for page to load
                time.sleep(wait_time)
                
                # Scroll down to load any lazy-loaded content
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                # Extract data based on selectors
                page_data = {}
                
                for field, selector in selectors.items():
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            page_data[field] = [el.text for el in elements]
                    except Exception as e:
                        st.warning(f"Error extracting {field}: {str(e)}")
                
                # Add metadata
                page_data["url"] = current_url
                page_data["page_num"] = page_count + 1
                
                # Add to results
                results.append(page_data)
                
                # Handle pagination
                page_count += 1
                if page_count >= max_pages:
                    break
                
                # Find next page link
                current_url = None
                if pagination_strategy == "auto" or pagination_strategy == "next_button":
                    next_selectors = ['a[rel="next"]', 'button.next', 'a.next', '.pagination-next', 'li.next a', 'a:contains("Next")']
                    for selector in next_selectors:
                        try:
                            next_element = driver.find_element(By.CSS_SELECTOR, selector)
                            if next_element and next_element.is_displayed():
                                current_url = next_element.get_attribute("href")
                                break
                        except:
                            continue
                elif pagination_strategy == "infinite_scroll":
                    # For infinite scroll, stay on same page and scroll again
                    height_before = driver.execute_script("return document.body.scrollHeight")
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    height_after = driver.execute_script("return document.body.scrollHeight")
                    
                    if height_after > height_before:
                        # Content was loaded, stay on same URL
                        current_url = driver.current_url
                    else:
                        # No more content
                        current_url = None
                else:
                    # Try custom selector
                    try:
                        next_element = driver.find_element(By.CSS_SELECTOR, pagination_strategy)
                        if next_element and next_element.is_displayed():
                            current_url = next_element.get_attribute("href")
                    except:
                        current_url = None
            
            # Close the driver
            driver.quit()
            
            # Return results
            return results
            
        except Exception as e:
            st.error(f"Error during scraping (attempt {attempt+1}): {str(e)}")
            attempt += 1
            time.sleep(2)  # Wait before retrying
    
    return {"error": f"Failed after {retry_limit} attempts"}

# Keep the async version with a note about compatibility
async def advanced_scrape(url, max_pages=3, pagination_strategy="auto", selectors=None, wait_time=2, 
                         auth_config=None, infer_selectors=False, api_key=None, proxy=None, 
                         retry_limit=3, rate_limits=None):
    """
    This function has compatibility issues with Python 3.12.
    Please use selenium_scrape instead for better compatibility.
    """
    st.warning("The advanced_scrape function has compatibility issues with Python 3.12. Using selenium_scrape instead.")
    return selenium_scrape(url, max_pages, pagination_strategy, selectors, wait_time, retry_limit)

# A new convenience function to wrap the selenium_scrape for multi-threading
def multi_url_scrape(urls, max_pages=3, pagination_strategy="auto", selectors=None, max_workers=3):
    """
    Scrape multiple URLs in parallel using Selenium.
    
    Args:
        urls: List of URLs to scrape
        max_pages: Maximum number of pages to scrape per URL
        pagination_strategy: Strategy for pagination
        selectors: Dictionary of CSS selectors for data extraction
        max_workers: Maximum number of concurrent scrapers
        
    Returns:
        Dictionary mapping each URL to its scraped results
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(selenium_scrape, url, max_pages, pagination_strategy, selectors): url for url in urls}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as e:
                results[url] = {"error": str(e)}
    
    return results

# Update multi_thread_scrape to use our new compatible scraper
def multi_thread_scrape(urls, max_pages=3, pagination_strategy="auto", selectors=None, max_workers=5, 
                       auth_config=None, infer_selectors=False, api_key=None, proxy=None, 
                       retry_limit=3, rate_limits=None):
    """
    Simplified version for multi-thread scraping using Selenium.
    """
    return multi_url_scrape(urls, max_pages, pagination_strategy, selectors, max_workers)
