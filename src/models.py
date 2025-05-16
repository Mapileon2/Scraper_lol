from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

# Generic input model for scraped data
class ScrapedDataInput(BaseModel):
    data: Dict[str, Any] = Field(description="Dictionary of scraped fields (e.g., title, text, url)")
    url: str
    page_num: int

# Generic output model for processed data
class DataAnalysisOutput(BaseModel):
    analysis_type: str = Field(description="Type of analysis, e.g., sentiment, summary")
    result: Dict[str, Any] = Field(description="Analysis results, e.g., sentiment score, summary text")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata, e.g., key issues") 