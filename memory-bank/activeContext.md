# Active Context: AI-Powered Web Scraper

**Current Work Focus:** Setting up the project structure and implementing the Streamlit UI.

**Recent Changes:**

*   Created the memory-bank directory and core files.
*   Populated projectbrief.md and productContext.md with information from the PRD.

**Next Steps:**

*   Create the remaining core files (systemPatterns.md, techContext.md, progress.md) and populate them with information from the PRD.
*   Set up the basic project structure (app.py, config.py, src/, models/, requirements.txt, .env).
*   Implement the Streamlit UI with input fields for API key, URL, CSS selector, max pages, instructions, and a scrape button.

**Active Decisions and Considerations:**

*   How to structure the project for modularity and extensibility.
*   Which Streamlit components to use for the UI.
*   How to handle user input validation and error handling.

**Important Patterns and Preferences:**

*   Use Crawl4AI for web scraping.
*   Use Pydantic-AI with Gemini LLM for data processing.
*   Output data in Excel (XLS) and Markdown (MD) formats.
*   Follow the filename format: scrape\_\<website\>+\<date\>+\<time\>.\<extension\>.

**Learnings and Project Insights:**

*   The PRD provides a comprehensive overview of the project requirements and specifications.
*   The project should be designed with future enhancements in mind, such as vector server integration.
