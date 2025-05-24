"""
Web Scraper Pro - A powerful web scraping and data extraction tool.

This package provides tools for web scraping, data extraction, and analysis
with support for various AI models and data storage backends.
"""

__version__ = "1.0.0"
__author__ = "Your Name <your.email@example.com>"
__license__ = "MIT"

# Import key components for easier access
from .config import (
    get_config,
    update_config,
    API_CONFIG,
    SCRAPER_CONFIG,
    MODEL_CONFIG,
    LOGGING_CONFIG
)

# Initialize logging
from .utils.logging import setup_logger, log_error, log_execution_time, logger

# Set up default logger
logger = setup_logger()

# Import core components
from .models import (
    ScrapedDataInput,
    DataAnalysisOutput,
    SelectorSuggestion
)

# Import main scraper
from .web_scraper import WebScraper, quick_scrape, quick_scrape_multiple

# Clean up namespace
__all__ = [
    # Core components
    'WebScraper',
    'ScrapedDataInput',
    'DataAnalysisOutput',
    'SelectorSuggestion',
    
    # Configuration
    'get_config',
    'update_config',
    'API_CONFIG',
    'SCRAPER_CONFIG',
    'MODEL_CONFIG',
    'LOGGING_CONFIG',
    
    # Logging
    'logger',
    'setup_logger',
    'log_error',
    'log_execution_time',
    
    # Version
    '__version__',
    '__author__',
    '__license__'
]
