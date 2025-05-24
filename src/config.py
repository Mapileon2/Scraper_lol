"""Centralized configuration for the Web Scraper application."""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file if it exists
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".cache"

# Create necessary directories
for directory in [DATA_DIR, LOG_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# API Configuration
class APIConfig:
    """Configuration for external APIs."""
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "web-scraper-cache")

# Scraping Configuration
class ScraperConfig:
    """Configuration for web scraping."""
    DEFAULT_WAIT_TIME: int = 2  # seconds
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30  # seconds
    MAX_WORKERS: int = 5
    CHUNK_SIZE: int = 4000
    CHUNK_OVERLAP: int = 200

# Model Configuration
class ModelConfig:
    """Configuration for ML models."""
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMERS_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")

# Logging Configuration
class LoggingConfig:
    """Configuration for logging."""
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: Path = LOG_DIR / "web_scraper.log"
    MAX_LOG_SIZE: int = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT: int = 5

# Initialize configurations
API_CONFIG = APIConfig()
SCRAPER_CONFIG = ScraperConfig()
MODEL_CONFIG = ModelConfig()
LOGGING_CONFIG = LoggingConfig()

# Export all configurations for easy access
CONFIG: Dict[str, Any] = {
    "api": {k: v for k, v in vars(API_CONFIG).items() if not k.startswith("_")},
    "scraper": {k: v for k, v in vars(SCRAPER_CONFIG).items() if not k.startswith("_")},
    "model": {k: v for k, v in vars(MODEL_CONFIG).items() if not k.startswith("_")},
    "logging": {k: v for k, v in vars(LOGGING_CONFIG).items() if not k.startswith("_")},
    "paths": {
        "base": str(BASE_DIR),
        "data": str(DATA_DIR),
        "logs": str(LOG_DIR),
        "cache": str(CACHE_DIR)
    }
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration as a dictionary."""
    return CONFIG

def update_config(updates: Dict[str, Dict[str, Any]]) -> None:
    """Update configuration values.
    
    Args:
        updates: Dictionary with section names as keys and config updates as values.
    """
    for section, values in updates.items():
        if section in CONFIG and isinstance(CONFIG[section], dict):
            CONFIG[section].update(values)
