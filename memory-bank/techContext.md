# Tech Context: AI-Powered Web Scraper

**Technologies Used:**

*   Frontend: Streamlit (streamlit)
*   Web Scraping: Crawl4AI (crawl4ai), Playwright (playwright)
*   Data Processing: Pydantic-AI (pydantic-ai), Gemini LLM (google-generativeai), Pydantic (pydantic)
*   File Handling: Pandas (pandas), Openpyxl (openpyxl), Tabulate (tabulate)
*   Environment: Python-dotenv (python-dotenv)
*   Language: Python 3.11+

**Development Setup:**

1.  Install Python 3.11+
2.  Install dependencies: `pip install -r requirements.txt`
3.  Install Playwright browsers: `playwright install`
4.  Set up the .env file with the Gemini API key.

**Technical Constraints:**

*   Gemini API rate limits and costs.
*   Website HTML structure variations.
*   Crawl4AI pagination limitations.

**Dependencies:**

```
crawl4ai
pydantic-ai
google-generativeai
pandas
tabulate
openpyxl
pydantic
python-dotenv
playwright
streamlit
```

**Tool Usage Patterns:**

*   Use Streamlit components for the UI.
*   Use Crawl4AI's CSSSelectorStrategy for scraping.
*   Use Pydantic models for data validation.
*   Use Gemini LLM for data extraction and parsing.
*   Use Pandas DataFrames for data structuring.
*   Use Openpyxl for Excel output.
*   Use Tabulate for Markdown output.
