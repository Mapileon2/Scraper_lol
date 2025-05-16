import google.generativeai as genai
import json
from typing import List, Dict, Optional, Any
import time
from .models import ScrapedDataInput, DataAnalysisOutput

class ReasoningAgent:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 1000
            }
        )

    def analyze_data(self, data: ScrapedDataInput, analysis_type: str) -> DataAnalysisOutput:
        """
        Analyze scraped data based on the specified analysis type.
        """
        data_str = "\n".join([f"{key}: {value}" for key, value in data.data.items()])
        prompt = f"""
        Analyze the following scraped data:
        URL: {data.url}
        Page: {data.page_num}
        Data:
        {data_str}

        Perform the following analysis: {analysis_type}
        Examples of analysis types:
        - 'sentiment': Determine sentiment (Positive, Negative, Neutral) and score (-1 to 1).
        - 'summary': Summarize the content in 1-2 sentences.
        - 'categorize': Categorize the data (e.g., product feature, complaint).
        - 'extract_issues': List key issues or problems mentioned.

        Return a JSON object with:
        - analysis_type: The requested analysis type
        - result: The analysis output (e.g., {{'sentiment': 'Negative', 'score': -0.8}})
        - metadata: Optional additional info (e.g., {{'issues': ['poor cooling']}})

        Ensure the response is concise and accurate.
        """
        
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Try to parse as JSON
                try:
                    result_dict = json.loads(result_text)
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract JSON using text processing
                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        result_dict = json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not extract valid JSON from response")
                
                return DataAnalysisOutput(**result_dict)
            except Exception as e:
                if attempt == 2:
                    print(f"Error analyzing data after 3 attempts: {e}")
                    return DataAnalysisOutput(
                        analysis_type=analysis_type,
                        result={"error": "Analysis failed"},
                        metadata={"error": str(e)}
                    )
                time.sleep(2)  # Wait before retry

    def answer_query(self, query: str, processed_data: List[DataAnalysisOutput], history: List[Dict] = None) -> str:
        """
        Answer a query based on processed data.
        """
        history = history or []
        history_prompt = "\n".join([f"Q: {h['question']} A: {h['answer']}" for h in history])
        # Limit to 10 items to avoid token limits
        data_summary = "\n".join([
            f"Analysis: {d.analysis_type}, Result: {d.result}, Metadata: {d.metadata or {}}" 
            for d in processed_data[:10]
        ])
        
        prompt = f"""
        Based on the following processed data:
        {data_summary}

        Previous conversation:
        {history_prompt}

        Answer the query: {query}
        Provide a concise, reasoned response.
        """
        
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                if attempt == 2:
                    print(f"Error answering query after 3 attempts: {e}")
                    return f"Unable to process query. Error: {str(e)}"
                time.sleep(2)  # Wait before retry

    def process_data(self, data_list: List[ScrapedDataInput], analysis_type: str) -> List[DataAnalysisOutput]:
        """
        Process a list of scraped data items.
        """
        results = []
        for data in data_list:
            results.append(self.analyze_data(data, analysis_type))
        return results
    
    def process_data_batched(self, data_list: List[ScrapedDataInput], analysis_type: str, batch_size: int = 10) -> List[DataAnalysisOutput]:
        """
        Process a list of scraped data items in batches to manage API usage.
        """
        results = []
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            batch_results = self.process_data(batch, analysis_type)
            results.extend(batch_results)
            if i + batch_size < len(data_list):
                time.sleep(2)  # Add delay between batches
        return results 