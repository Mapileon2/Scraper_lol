# Product Context: AI-Powered Web Scraper

**Problem:** Data analysts, marketers, and business owners need to efficiently extract business data from websites for lead generation, competitor analysis, and market research. Manually scraping data is time-consuming and prone to errors.

**Solution:** An AI-powered web scraper that automates the process of extracting, structuring, and formatting business data from websites.

**How it should work:**

1.  The user provides a Gemini API key and a target website URL.
2.  The scraper uses Crawl4AI to extract raw content from the website.
3.  Pydantic-AI with Gemini LLM processes the raw content into structured data (name, address, phone).
4.  The data is saved in Excel (XLS) and Markdown (MD) formats.
5.  The user can download the results.

**User Experience Goals:**

*   User-friendly Streamlit interface.
*   Clear instructions and feedback.
*   Reliable data extraction and formatting.
*   Efficient scraping process.
