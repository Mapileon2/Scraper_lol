import google.generativeai as genai
import json
from typing import List, Dict, Optional, Any
import time
import re
from .models import ScrapedDataInput, DataAnalysisOutput

class ReasoningAgent:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """Initialize the Gemini model with proper error handling.
        Defaults to gemini-1.5-flash which has higher free-tier quotas than gemini-1.5-pro.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 800  # Reduced to conserve tokens
            }
        )
        # Cache to avoid repeated analysis
        self._cache = {}

    def _truncate_text(self, text: str, max_length: int = 200) -> str:
        """Truncate text to reduce token usage"""
        if isinstance(text, str) and len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def analyze_data(self, data: ScrapedDataInput, analysis_type: str) -> DataAnalysisOutput:
        """
        Analyze scraped data based on the specified analysis type.
        With token optimization and quota error handling.
        """
        # Generate a cache key based on data and analysis type
        cache_key = f"{data.url}_{data.page_num}_{analysis_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
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
        
        # More concise prompt to reduce tokens
        prompt = f"""
        Data from {data.url} (page {data.page_num}):
        {data_str}

        {analysis_type}

        Return JSON: {{analysis_type, result, metadata}}
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
    
    def process_data_batched(self, data_list: List[ScrapedDataInput], analysis_type: str, batch_size: int = 5) -> List[DataAnalysisOutput]:
        """
        Process a list of scraped data items in smaller batches to manage API usage.
        Reduced default batch size from 10 to 5.
        """
        results = []
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            batch_results = self.process_data(batch, analysis_type)
            results.extend(batch_results)
            # Add longer delay between batches
            if i + batch_size < len(data_list):
                time.sleep(5)  # 5 second pause between batches
        return results 