from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from datetime import datetime

class AnalysisType(str, Enum):
    SENTIMENT = "sentiment"
    SUMMARY = "summary"
    KEY_POINTS = "key_points"
    CUSTOM = "custom"

class ScrapedDataInput(BaseModel):
    """Input model for scraped data with validation."""
    data: Dict[str, Any] = Field(
        ...,
        description="Dictionary of scraped fields (e.g., title, text, url)",
        example={"title": "Example Title", "content": "Sample content"}
    )
    url: HttpUrl = Field(
        ...,
        description="Source URL of the scraped data"
    )
    page_num: int = Field(
        ...,
        ge=1,
        description="Page number of the scraped data (1-based index)"
    )

    @field_validator('data')
    @classmethod
    def validate_data_not_empty(cls, v):
        if not v or not any(v.values()):
            raise ValueError("Scraped data cannot be empty")
        return v

class DataAnalysisOutput(BaseModel):
    """Output model for processed analysis results."""
    analysis_type: AnalysisType = Field(
        ...,
        description="Type of analysis performed"
    )
    result: Dict[str, Any] = Field(
        ...,
        description="Analysis results with structure depending on analysis_type"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of when the analysis was performed"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "analysis_type": "sentiment",
                "result": {
                    "score": 0.85,
                    "label": "positive"
                },
                "metadata": {
                    "model": "gemini-1.5-flash",
                    "version": "1.0.0"
                },
                "timestamp": "2023-01-01T12:00:00Z"
            }
        }
    }

class SelectorSuggestion(BaseModel):
    """Model for CSS selector suggestions from AI analysis."""
    field: str = Field(..., description="Name of the field this selector extracts")
    selector: str = Field(..., description="CSS selector for the field")
    sample_data: Optional[str] = Field(
        None,
        description="Example of extracted data using this selector"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) of this selector's accuracy"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "field": "title",
                "selector": "h1.product-title",
                "sample_data": "Example Product Name",
                "confidence": 0.92
            }
        }