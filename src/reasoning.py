import json
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, TypeVar, Generic
from datetime import datetime, timedelta
import hashlib
import pickle

import google.generativeai as genai
from pydantic import BaseModel, ValidationError

from .models import ScrapedDataInput, DataAnalysisOutput, AnalysisType

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic caching
T = TypeVar('T')

class CacheManager(Generic[T]):
    """Generic cache manager with TTL and persistence."""
    
    def __init__(self, cache_dir: str = '.cache', ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.memory_cache: Dict[str, T] = {}
    
    def _get_cache_path(self, key: str) -> Path:
        """Generate a filesystem path for a cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[T]:
        """Get a value from cache, checking both memory and disk."""
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - mtime < self.ttl:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        self.memory_cache[key] = data  # Populate memory cache
                        return data
                else:
                    cache_file.unlink()  # Remove expired cache
            except (pickle.PickleError, EOFError) as e:
                logger.warning(f"Cache read error for {key}: {e}")
                cache_file.unlink()
        
        return None
    
    def set(self, key: str, value: T) -> None:
        """Store a value in both memory and disk caches."""
        self.memory_cache[key] = value
        try:
            cache_file = self._get_cache_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except (IOError, pickle.PickleError) as e:
            logger.warning(f"Failed to write to cache: {e}")

class RateLimiter:
    """Simple rate limiter to prevent API quota issues."""
    
    def __init__(self, max_requests: int, per_seconds: int):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        # Remove old requests outside the time window
        self.requests = [t for t in self.requests if now - t < self.per_seconds]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = self.per_seconds - (now - self.requests[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.requests.append(time.time())

class ReasoningAgent:
    """Agent for performing AI-powered analysis on scraped data."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gemini-1.5-flash",
        max_retries: int = 3,
        cache_ttl_hours: int = 24,
        rate_limit: Optional[Dict[str, int]] = None
    ):
        """Initialize the reasoning agent.
        
        Args:
            api_key: Google Gemini API key
            model: Model name to use for analysis
            max_retries: Maximum number of retries for API calls
            cache_ttl_hours: Hours to keep cached results
            rate_limit: Optional dict with 'max_requests' and 'per_seconds'
        """
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.4,  # Lower temperature for more consistent results
                "top_p": 0.95,
                "max_output_tokens": 1024,
                "top_k": 40
            },
            safety_settings={
                "HARASSMENT": "block_none",
                "HATE_SPEECH": "block_none",
                "SEXUALLY_EXPLICIT": "block_none",
                "DANGEROUS_CONTENT": "block_none"
            }
        )
        self.max_retries = max_retries
        self.cache = CacheManager[DataAnalysisOutput](ttl_hours=cache_ttl_hours)
        
        # Set up rate limiting
        rate_limit = rate_limit or {"max_requests": 10, "per_seconds": 60}
        self.rate_limiter = RateLimiter(**rate_limit)
        
        logger.info(f"Initialized ReasoningAgent with model: {model}")
    
    def _generate_cache_key(self, data: ScrapedDataInput, analysis_type: str) -> str:
        """Generate a unique cache key for the given data and analysis type."""
        data_str = json.dumps({
            "url": str(data.url),
            "page_num": data.page_num,
            "data_keys": sorted(data.data.keys()),
            "analysis_type": analysis_type,
            "model": self.model_name
        }, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _truncate_text(self, text: str, max_length: int = 200) -> str:
        """Truncate text to reduce token usage while preserving words."""
        if not isinstance(text, str) or len(text) <= max_length:
            return text
            
        # Try to truncate at word boundary
        truncated = text[:max_length]
        if len(text) > max_length:
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]
        return truncated + "..."
    
    def _prepare_prompt(
        self, 
        data: ScrapedDataInput, 
        analysis_type: str
    ) -> str:
        """Prepare the prompt for the AI model."""
        # Format data for the prompt
        data_items = []
        for key, value in data.data.items():
            if isinstance(value, str):
                value = self._truncate_text(value, max_length=500)
            elif isinstance(value, list):
                value = [
                    self._truncate_text(item, max_length=200) 
                    if isinstance(item, str) else str(item) 
                    for item in value[:10]  # Limit number of items
                ]
            data_items.append(f"{key}: {value}")
        
        data_str = "\n".join(data_items)
        
        # System message with instruction
        system_msg = (
            "You are a data analysis assistant. Analyze the following data "
            f"and provide a {analysis_type} analysis. Be concise and factual."
        )
        
        # Format the prompt
        prompt = f"""{system_msg}

Data from {data.url} (page {data.page_num}):
{data_str}

Analysis ({analysis_type}):"""
        
        return prompt
    
    def analyze_data(self, data: ScrapedDataInput, analysis_type: str) -> DataAnalysisOutput:
        """
        Analyze scraped data based on the specified analysis type.
        With token optimization and quota error handling.
        """
        cache_key = self._generate_cache_key(data, analysis_type)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Truncate data to reduce token usage
        data_items = []
        for key, value in data.data.items():
            # Truncate large text fields
            if isinstance(value, str):
                value = self._truncate_text(value)
            elif isinstance(value, list):
                value = [self._truncate_text(item) if isinstance(item, str) else item for item in value[:5]]
            data_items.append(f"{key}: {value}")
        
        data_str = "\n".join(data_items)
        
        prompt = f"""{self._prepare_prompt(data, analysis_type)}
        
        Return JSON: {{"analysis_type": "{analysis_type}", "result": {{...}}, "metadata": {{}}}}
        """
        
        for attempt in range(3):
            try:
                # Add progressive delay between retries
                if attempt > 0:
                    delay = 5 * (2 ** attempt)  # 10s, then 20s
                    time.sleep(delay)
                
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Parse the response as JSON
                try:
                    result_dict = json.loads(result_text)
                except json.JSONDecodeError:
                    # Try to extract JSON using regex if direct parsing fails
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        try:
                            result_dict = json.loads(json_match.group(0))
                        except:
                            # Fallback to a simple structure if JSON extraction fails
                            result_dict = {
                                "analysis_type": analysis_type,
                                "result": {"text": result_text},
                                "metadata": {}
                            }
                    else:
                        result_dict = {
                            "analysis_type": analysis_type,
                            "result": {"text": result_text},
                            "metadata": {}
                        }
                
                # Create output object
                output = DataAnalysisOutput(
                    analysis_type=result_dict.get("analysis_type", analysis_type),
                    result=result_dict.get("result", {"text": result_text}),
                    metadata=result_dict.get("metadata", {})
                )
                
                # Cache the result
                self._cache[cache_key] = output
                return output
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "exceeded your current quota" in error_str:
                    if attempt == 2:
                        return DataAnalysisOutput(
                            analysis_type=analysis_type,
                            result={"error": "API quota exceeded"},
                            metadata={
                                "error": "Gemini API quota exceeded. Try gemini-1.5-flash or wait until quota resets.",
                                "details": error_str[:200]
                            }
                        )
                else:
                    if attempt == 2:
                        return DataAnalysisOutput(
                            analysis_type=analysis_type,
                            result={"error": "Analysis failed"},
                            metadata={"error": error_str[:200]}
                        )
        
        # This should never happen, but just in case
        return DataAnalysisOutput(
            analysis_type=analysis_type,
            result={"error": "Analysis failed after multiple attempts"},
            metadata={}
        )

    def answer_query(self, query: str, processed_data: List[DataAnalysisOutput], history: List[Dict] = None) -> str:
        """
        Answer a query based on processed data.
        Optimized for lower token usage.
        """
        history = history or []
        
        # Limit history to most recent 3 exchanges to reduce tokens
        recent_history = history[-3:] if len(history) > 3 else history
        history_prompt = "\n".join([f"Q: {h['question'][:100]} A: {h['answer'][:200]}" for h in recent_history])
        
        # Limit to fewer items to avoid token limits
        data_summary = "\n".join([
            f"Analysis: {d.analysis_type}, Result: {str(d.result)[:150]}" 
            for d in processed_data[:5]  # Reduced from 10 to 5
        ])
        
        prompt = f"""
        Based on this data:
        {data_summary}

        Previous conversation:
        {history_prompt}

        Answer: {query}
        """
        
        for attempt in range(3):
            try:
                # Add delay between retries
                if attempt > 0:
                    delay = 5 * (2 ** attempt)
                    time.sleep(delay)
                
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "exceeded your current quota" in error_str:
                    if attempt == 2:
                        return "Unable to process query due to API quota limits. Please try again later or switch to gemini-1.5-flash model."
                elif attempt == 2:
                    return f"Unable to process query. Error: {str(e)[:100]}"
                
        return "Unable to process query after multiple attempts."

    def process_data(self, data_list: List[ScrapedDataInput], analysis_type: str) -> List[DataAnalysisOutput]:
        """
        Process a list of scraped data items with rate limiting.
        """
        results = []
        for i, data in enumerate(data_list):
            # Add delay every 2 items to avoid hitting rate limits
            if i > 0 and i % 2 == 0:
                time.sleep(3)  # 3 second pause every 2 items
            results.append(self.analyze_data(data, analysis_type))
        return results
    
    def process_data_batched(self, data_list: List[ScrapedDataInput], analysis_type: str, batch_size: int = 5, 
                           retry_on_quota_error: bool = True, resume_file: str = None) -> List[DataAnalysisOutput]:
        """
        Process a list of scraped data items in smaller batches to manage API usage with enhanced robustness.
        
        Args:
            data_list: List of data items to process
            analysis_type: Type of analysis to perform
            batch_size: Number of items to process in each batch
            retry_on_quota_error: Whether to retry with exponential backoff when quota is exceeded
            resume_file: Optional file path to save progress for resuming later
            
        Returns:
            List of analysis results
        """
        import os
        import json
        from datetime import datetime
        
        logger.info(f"Starting batched processing of {len(data_list)} items with batch size {batch_size}")
        results = []
        processed_indices = set()
        
        # Check for existing resume file
        if resume_file and os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                    if resume_data.get('analysis_type') == analysis_type:
                        # Convert string indices back to integers
                        processed_indices = set(int(idx) for idx in resume_data.get('processed_indices', []))
                        results = resume_data.get('results', [])
                        logger.info(f"Resuming from checkpoint with {len(processed_indices)} items already processed")
            except Exception as e:
                logger.error(f"Error loading resume file: {e}")
        
        # Process in batches
        for i in range(0, len(data_list), batch_size):
            if all(j in processed_indices for j in range(i, min(i+batch_size, len(data_list)))):
                logger.info(f"Skipping batch {i//batch_size + 1} as all items are already processed")
                continue
                
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(data_list) + batch_size - 1)//batch_size}")
            
            # Filter out already processed items
            batch_indices = [j for j in range(i, min(i+batch_size, len(data_list))) if j not in processed_indices]
            batch = [data_list[j] for j in batch_indices]
            
            if not batch:  # Skip if all items in this batch were already processed
                continue
                
            try:
                batch_results = self.process_data(batch, analysis_type)
                
                # Update results and processed indices
                for idx, result in zip(batch_indices, batch_results):
                    # Convert DataAnalysisOutput to dict for JSON serialization
                    if hasattr(result, 'dict'):
                        results.append(result.dict())
                    else:
                        results.append(result)
                    processed_indices.add(idx)
                
                # Save progress checkpoint
                if resume_file:
                    with open(resume_file, 'w') as f:
                        json.dump({
                            'timestamp': datetime.now().isoformat(),
                            'analysis_type': analysis_type,
                            'processed_indices': list(processed_indices),
                            'results': results
                        }, f)
                    logger.info(f"Saved progress checkpoint with {len(processed_indices)}/{len(data_list)} items processed")
                
                # Add adaptive delay between batches based on batch size
                if i + batch_size < len(data_list):
                    delay = min(5 + batch_size, 15)  # Scale delay with batch size, max 15 seconds
                    logger.info(f"Pausing for {delay}s before next batch")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                if "quota" in str(e).lower() and retry_on_quota_error:
                    # Exponential backoff for quota errors
                    backoff = 30 * (2 ** (i // batch_size % 4))  # 30s, 60s, 120s, 240s
                    logger.warning(f"API quota error, backing off for {backoff}s before retry")
                    time.sleep(backoff)
                    # Reduce batch size to avoid hitting quota again
                    new_batch_size = max(1, batch_size // 2)
                    if new_batch_size != batch_size:
                        logger.info(f"Reducing batch size from {batch_size} to {new_batch_size}")
                        # Recursively call with smaller batch size
                        remaining_results = self.process_data_batched(
                            data_list[i:], analysis_type, new_batch_size, 
                            retry_on_quota_error, resume_file
                        )
                        return results + remaining_results
        
        logger.info(f"Completed processing {len(processed_indices)}/{len(data_list)} items")
        return results