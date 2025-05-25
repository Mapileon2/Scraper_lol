from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel
import json

class SelectorSuggestion(BaseModel):
    field: str
    selector: str
    confidence: float

class LLMInterface(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_selectors(self, html: str, fields: List[dict]) -> List[SelectorSuggestion]:
        """Generate CSS selectors for the given fields based on the HTML content."""
        pass

class GeminiLLM(LLMInterface):
    """LLM implementation using Google's Gemini API with support for both 1.5 Pro and 2.0 models."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini LLM with the specified model.
        
        Args:
            api_key: Google AI API key
            model: Model to use ('gemini-1.5-pro' or 'gemini-2.0')
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Validate model name
            valid_models = ["gemini-1.5-pro", "gemini-2.0"]
            if model not in valid_models:
                raise ValueError(f"Invalid model. Must be one of: {', '.join(valid_models)}")
                
            self.model_name = model
            self.client = genai
            self.model = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more deterministic output
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
            )
            
        except ImportError as e:
            raise ImportError(
                "Please install the Google Generative AI package: pip install google-generativeai"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {str(e)}") from e
    
    async def generate_selectors(self, html: str, fields: List[dict]) -> List[SelectorSuggestion]:
        """Generate CSS selectors for the given fields based on HTML content."""
        try:
            prompt = self._build_prompt(html, fields)
            
            # Generate content with error handling
            response = await self.model.generate_content_async(prompt)
            
            # Handle different response formats between models
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                response_text = response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Unexpected response format from Gemini API")
                
            return self._parse_response(response_text, fields)
            
        except Exception as e:
            error_msg = f"Error generating selectors with Gemini {self.model_name}: {str(e)}"
            if "safety" in str(e).lower():
                error_msg += "\nNote: The response was blocked for safety reasons. Try adjusting your prompt."
            raise Exception(error_msg) from e
    
    def _build_prompt(self, html: str, fields: List[dict]) -> str:
        """
        Build the prompt for the Gemini model.
        
        Args:
            html: HTML content to analyze
            fields: List of field definitions
            
        Returns:
            Formatted prompt string
        """
        field_descriptions = "\n".join([
            f"- {f['name']}: {f['description']}" 
            for f in fields
        ])
        
        # Truncate HTML to avoid token limits
        max_html_length = 30000 if self.model_name == "gemini-1.5-pro" else 10000
        truncated_html = html[:max_html_length]
        
        return f"""You are an expert web scraping assistant. Analyze the HTML and suggest the best CSS selectors 
for the requested fields. Consider both uniqueness and reliability of the selectors.

Fields to find (name and description):
{field_descriptions}

HTML to analyze (truncated to {len(truncated_html)} chars):
{truncated_html}

Instructions:
1. Analyze the HTML structure and find the best selectors for each field
2. For each field, provide a CSS selector that uniquely identifies the element
3. Rate your confidence in each selector from 0.0 to 1.0
4. Return only valid JSON with no additional text

Return format (JSON array of objects):
[
    {{
        "field": "field_name",
        "selector": "css.selector",
        "confidence": 0.95,
        "explanation": "Brief explanation of why this selector was chosen"
    }}
]

Important:
- Only return the JSON array, no additional text or markdown formatting
- Selectors should work even if the page structure changes slightly
- Prefer data attributes, IDs, or semantic classes when available
- For text content, include the :contains() pseudo-class if needed
- For multiple matches, use :first-child or :nth-child() to be specific"""
    
    def _parse_response(self, response: str, fields: List[dict]) -> List[SelectorSuggestion]:
        """
        Parse the LLM response into SelectorSuggestion objects.
        
        Args:
            response: Raw response from the LLM
            fields: List of field definitions for validation
            
        Returns:
            List of SelectorSuggestion objects
            
        Raises:
            ValueError: If the response cannot be parsed or is invalid
        """
        try:
            # Clean the response
            response = response.strip()
            
            # Extract JSON from markdown code blocks if present
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response and '```' != response[:3]:
                # Handle cases where response is wrapped in markdown code blocks
                response = response.split('```')[1].strip()
            
            # Handle cases where the response might be wrapped in markdown code blocks
            if response.startswith('```') and response.endswith('```'):
                response = response[3:-3].strip()
            
            # Parse the JSON response
            try:
                suggestions = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON if it's embedded in text
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    suggestions = json.loads(json_match.group(0))
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Convert single suggestion to list if needed
            if isinstance(suggestions, dict):
                suggestions = [suggestions]
            
            # Validate and convert to SelectorSuggestion objects
            results = []
            field_names = {f['name'] for f in fields}
            
            for s in suggestions:
                if not isinstance(s, dict):
                    continue
                    
                try:
                    field = s.get('field', '')
                    if field not in field_names:
                        continue
                        
                    selector = s.get('selector', '').strip()
                    if not selector:
                        continue
                        
                    # Ensure confidence is a float between 0 and 1
                    confidence = float(s.get('confidence', 0.5))
                    confidence = max(0.0, min(1.0, confidence))
                    
                    results.append(SelectorSuggestion(
                        field=field,
                        selector=selector,
                        confidence=confidence
                    ))
                except (ValueError, TypeError):
                    continue
            
            if not results:
                raise ValueError("No valid selectors found in the response")
                
            return results
            
        except Exception as e:
            error_msg = f"Failed to parse LLM response: {str(e)}"
            if len(response) > 500:
                error_msg += f"\nResponse (truncated): {response[:500]}..."
            else:
                error_msg += f"\nResponse: {response}"
            raise ValueError(error_msg) from e

# Example usage:
# llm = GeminiLLM(api_key="your-api-key")
# selectors = await llm.generate_selectors(html, fields)
