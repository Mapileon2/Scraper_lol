from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from enum import Enum

class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROK = "grok"

class ScrapeField(BaseModel):
    """Model for defining fields to scrape."""
    name: str = Field(..., description="Name of the field (must be a valid Python identifier)")
    description: str = Field(..., description="Description of what to scrape")
    selector: Optional[str] = Field(None, description="CSS selector (optional, can be auto-inferred)")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.isidentifier():
            raise ValueError("Field name must be a valid Python identifier")
        return v

class ScrapeRequest(BaseModel):
    """Model for scrape request configuration."""
    url: HttpUrl = Field(..., description="URL to scrape")
    fields: List[ScrapeField] = Field(..., min_length=1, description="List of fields to scrape")
    llm_provider: LLMProvider = Field(LLMProvider.GEMINI, description="LLM provider to use")
    llm_api_key: Optional[str] = Field(None, description="API key for the LLM provider")
    max_pages: int = Field(1, ge=1, description="Maximum number of pages to scrape")
    headless: bool = Field(True, description="Run browser in headless mode")
    timeout: int = Field(30, ge=1, description="Timeout in seconds")
    
    class Config:
        use_enum_values = True

class ScrapedData(BaseModel):
    """Model for scraped data output."""
    url: str
    data: Dict[str, Any]
    status: str = "success"
    error: Optional[str] = None
    
    @classmethod
    def error(cls, url: str, error: str):
        return cls(url=url, data={}, status="error", error=error)
