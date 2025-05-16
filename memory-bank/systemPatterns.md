# System Patterns: AI-Powered Web Scraper

**System Architecture:**

The system consists of the following components:

*   **Streamlit UI:** Provides the user interface for configuring and running the scraper.
*   **Crawl4AI:** Handles the web scraping logic.
*   **Pydantic-AI with Gemini LLM:** Processes the scraped data and structures it into a Pydantic model.
*   **File Generation:** Creates Excel (XLS) and Markdown (MD) files.

**Key Technical Decisions:**

*   Using Streamlit for the UI due to its ease of use and rapid development capabilities.
*   Using Crawl4AI for web scraping due to its robustness and ability to handle complex websites.
*   Using Pydantic-AI with Gemini LLM for data processing due to its AI-powered parsing capabilities.
*   Using Pandas and Openpyxl for Excel output and Tabulate for Markdown output.

**Design Patterns:**

*   **Modular Design:** The system is designed with a modular architecture to allow for easy extension and modification.
*   **Configuration-Driven:** The scraper is configured through user input and configuration files, allowing for flexibility and customization.

**Component Relationships:**

1.  The user interacts with the Streamlit UI to configure the scraper.
2.  The Streamlit UI passes the configuration to the Crawl4AI component.
3.  The Crawl4AI component scrapes the data from the target website.
4.  The scraped data is passed to the Pydantic-AI with Gemini LLM component.
5.  The Pydantic-AI with Gemini LLM component processes the data and structures it into a Pydantic model.
6.  The structured data is passed to the File Generation component.
7.  The File Generation component creates Excel (XLS) and Markdown (MD) files.
8.  The Streamlit UI provides download buttons for the generated files.

**Critical Implementation Paths:**

*   The scraping process must be robust and handle various website structures and pagination patterns.
*   The data processing must be accurate and efficient, leveraging the AI capabilities of Pydantic-AI with Gemini LLM.
*   The file generation must create valid and well-formatted Excel and Markdown files.
