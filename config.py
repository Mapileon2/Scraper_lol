import os

# Default settings
DEFAULT_MAX_PAGES = 3
DEFAULT_INSTRUCTIONS = "Extract all business information: 'name', 'address', 'website', 'phone number' and a one-sentence 'description' from the following content."
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"

LLM_MODEL = "gemini/gemini-2.0-flash"
API_TOKEN = os.getenv("GEMINI_API_KEY")
CSS_SELECTOR = ".resultbox_text"
MAX_PAGES = 3
SCRAPER_INSTRUCTIONS = (
    "Extract all business information: 'name', 'address', 'phone number' from the following content."
)
