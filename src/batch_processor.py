"""
Batch processing for Gemini API requests to handle rate limits and improve efficiency.
"""
import time
import threading
import logging
import json
import traceback
from queue import Queue, Empty
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Import Gemini
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger(__name__)

# Default model to use
DEFAULT_MODEL = "gemini-1.5-pro"

@dataclass
class BatchRequest:
    """Class to hold batch request data"""
    url: str
    content: Dict[str, Any]
    callback: Callable
    retries: int = 0
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # Higher number = higher priority

class GeminiBatchProcessor:
    """Batch processor for Gemini API requests"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, batch_size: int = 5, batch_delay: float = 2.0, max_retries: int = 3):
        if self._initialized:
            return
            
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.max_retries = max_retries
        
        self.queue = Queue()
        self.processing = False
        self.last_batch_time = 0
        self._initialized = True
        
        # Start processing thread
        self.processing_thread = None
        self.start_processing()
    
    def add_request(self, request: BatchRequest):
        """Add a request to the processing queue"""
        self.queue.put(request)
    
    def start_processing(self):
        """Start the background processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing = True
            self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
            self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the background processing thread"""
        self.processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
    
    def _process_batches(self):
        """Process batches from the queue"""
        current_batch = []
        
        while self.processing or not self.queue.empty():
            try:
                # Wait for batch size or timeout
                start_time = time.time()
                while len(current_batch) < self.batch_size and (time.time() - start_time) < 1.0:
                    try:
                        item = self.queue.get(timeout=0.1)
                        current_batch.append(item)
                    except Empty:
                        pass
                
                if not current_batch:
                    time.sleep(0.1)
                    continue
                
                # Process batch
                self._process_batch(current_batch)
                current_batch = []
                
                # Respect rate limits
                time_since_last_batch = time.time() - self.last_batch_time
                if time_since_last_batch < self.batch_delay:
                    time.sleep(self.batch_delay - time_since_last_batch)
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                time.sleep(1)  # Prevent tight loop on errors
    
    def _process_batch(self, batch: List[BatchRequest]):
        """
        Process a single batch of requests.
        
        Args:
            batch: List of BatchRequest objects to process
        """
        if not batch:
            return
            
        # Sort batch by priority (highest first) and timestamp (oldest first)
        batch.sort(key=lambda x: (-x.priority, x.timestamp))
        
        # Process each item in batch
        for request in batch:
            try:
                logger.info(f"Processing request for {request.url}")
                
                # Make the API call
                response = self._call_gemini_api(request.content)
                
                # Call the callback with the result
                if callable(request.callback):
                    request.callback(response)
                
            except Exception as e:
                logger.error(f"Error processing {request.url}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Retry logic with exponential backoff
                if request.retries < self.max_retries:
                    request.retries += 1
                    backoff = min(60 * (2 ** (request.retries - 1)), 300)  # Max 5 minutes
                    logger.info(f"Retrying {request.url} (attempt {request.retries}/{self.max_retries}) in {backoff}s")
                    request.timestamp = time.time() + backoff
                    self.queue.put(request)
                else:
                    # Max retries reached, call callback with error
                    if callable(request.callback):
                        request.callback({
                            "status": "error",
                            "error": f"Max retries ({self.max_retries}) exceeded: {str(e)}"
                        })
    
    def _call_gemini_api(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the actual API call to Gemini with proper error handling and rate limiting.
        
        Args:
            content: Dictionary containing request data including 'prompt' and other parameters
            
        Returns:
            Dictionary with 'status' and either 'result' or 'error' key
        """
        try:
            # Configure Gemini with the API key from the content
            genai.configure(api_key=content.get('api_key', ''))
            model = genai.GenerativeModel(DEFAULT_MODEL)
            
            # Extract the prompt and other parameters
            prompt = content.get('prompt', '')
            if not prompt:
                return {"status": "error", "error": "No prompt provided"}
                
            # Make the API call
            response = model.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': 2048,
                    'temperature': 0.3,
                },
                stream=False
            )
            
            # Process the response
            if not hasattr(response, 'text'):
                return {"status": "error", "error": "Invalid response from Gemini API"}
                
            # Return the result with metadata
            result = {
                "status": "success",
                "result": {
                    "text": response.text,
                    "prompt_feedback": getattr(response, 'prompt_feedback', None),
                    "usage_metadata": getattr(response, 'usage_metadata', None)
                }
            }
            
            return result
            
        except ResourceExhausted as e:
            # Handle rate limiting
            retry_after = 60  # Default 60 seconds
            logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds before retry...")
            time.sleep(retry_after)
            # Retry the request
            return self._call_gemini_api(content)
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            logger.error(traceback.format_exc())
            return {"status": "error", "error": str(e)}

# Global batch processor instance
batch_processor = GeminiBatchProcessor()
