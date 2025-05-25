# Web Scraper Pro

A powerful web scraping tool with AI-powered data extraction capabilities, supporting multiple scraping backends including Selenium and Crawl4AI.

## Features

- **Multiple Scraping Backends**: Choose between Standard, Clean, or AI-Powered (Crawl4AI) scrapers
- **AI-Powered Data Extraction**: Uses Gemini AI for intelligent data extraction
- **Multi-URL Scraping**: Scrape multiple URLs in parallel
- **Advanced Analysis**: Built-in data analysis and visualization
- **Session Management**: Save and load scraping sessions
- **Export Data**: Export scraped data to CSV, JSON, or SQLite

## Installation
- **Parallel and Asynchronous Scraping**: Process multiple URLs concurrently to maximize efficiency.
- **Authentication Support**: Handle form logins, basic auth, cookies, and OAuth for scraped sites requiring authentication.
- **Smart Selector Detection**: Uses Gemini AI to automatically detect and suggest CSS selectors for target elements.
- **Anti-Blocking Protection**: Implements rate limiting, user-agent rotation, and proxy support to avoid detection.
- **Multi-Source Data Aggregation**: Scrape multiple websites in one session with unified output.

### Enhanced Data Processing

- **Structured Data Extraction**: Extract business names, addresses, phone numbers, and other fields automatically.
- **Data Cleaning Tools**: Remove duplicates, standardize formats, and filter unwanted data.
- **Flexible Output Formats**: Export to Excel, CSV, JSON, or Markdown formats.
- **Data Visualization**: Quickly analyze scraping results with integrated visualizations.
- **Session Management**: Save and resume scraping sessions for later use.

## Getting Started

### Prerequisites

- Python 3.10+
- Gemini API key (for AI-powered features)
- Pinecone API key (for vector storage)

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Playwright browsers:
   ```bash
   playwright install
   ```

4. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

### Running the Application

Launch the Streamlit application:
```bash
streamlit run app_improved.py
```

## Resource Requirements

This application can be resource-intensive, especially when using features like:

*   **Selenium/Playwright-based scraping**: Running headless browsers consumes significant CPU and RAM. Parallel scraping will multiply this effect.
*   **AI Models**: Embedding models (like Sentence Transformers) are loaded into memory.
*   **Data Processing**: Handling large datasets with Pandas can also require substantial memory.

Ensure your deployment environment has adequate resources (e.g., at least 2GB RAM, 1 CPU core per parallel worker is a rough guideline, but actual needs may vary based on usage intensity and website complexity). Insufficient resources can lead to slow performance or crashes.

## Usage Guide

### Single URL Scraping

1. Navigate to the "Advanced Scrape" tab
2. Enter the URL to scrape
3. Configure optional settings in the "Advanced Options" panel:
   - Pagination strategy
   - Authentication details
   - Custom selectors
   - Proxy settings
4. Click "Scrape Website"
5. View, analyze, and export your results

### Multi-URL Scraping

1. Navigate to the "Multi-URL Scrape" tab
2. Enter multiple URLs (one per line)
3. Set global options like max pages and pagination strategy
4. Click "Scrape Multiple URLs"
5. View results organized by URL and export combined data

### Data Analysis

1. Navigate to the "Data Analysis" tab
2. Choose data from current session or previous sessions
3. Clean and transform the data using the provided tools
4. Generate visualizations to identify patterns
5. Export processed data in your preferred format

## Advanced Configuration

### Configuring Selectors

When not using AI-powered selector detection, you can manually specify CSS selectors:

- Business Name: `.business-name, h1, h2.title`
- Address: `.address, .location, address`
- Phone: `.phone, .tel, [href^='tel:']`
- Website: `.website a, a.website`

### Authentication

The scraper supports multiple authentication methods:

- **Form Login**: Provide username, password, and optional login URL
- **HTTP Basic**: Provide username and password for HTTP authentication
- **Cookie-based**: Paste cookies in JSON format
- **OAuth**: Provide token and token type

### Rate Limiting

To avoid being blocked, configure rate limiting for specific domains:

```python
rate_limits = {
    "example.com": {"min_delay": 3, "max_delay": 7},
    "default": {"min_delay": 2, "max_delay": 5}
}
```

### Proxy Configuration

For websites with strict anti-scraping measures, configure proxy support:

```python
proxy = {
    "server": "http://proxy.example.com:8080", 
    "username": "user", 
    "password": "pass"
}
```

Or provide a list of proxies for rotation:

```python
proxies = [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080"
]
```

### Deployment Considerations

#### Daily Request Limit Tracking

The application uses a local file (`daily_requests.json`) to track daily API request counts for rate limiting purposes. This method is suitable for single-instance deployments where the application has write access to the local filesystem.

**Important Considerations for Scaled Deployments:**

*   In multi-instance environments (e.g., when deployed on platforms like Streamlit Cloud with multiple workers, Kubernetes, or other forms of horizontal scaling), each instance will maintain its own `daily_requests.json`. This will result in an inaccurate global request count, and the rate limiting may not function as intended across all instances.
*   In read-only filesystem environments, the application will fail to write to `daily_requests.json`, potentially causing errors or disabling the rate limiting feature.

For robust and accurate daily request tracking in such environments, it is recommended to implement a centralized solution, such as:
*   A database (e.g., Redis, PostgreSQL).
*   A distributed caching service.

#### API Key Provisioning

The application requires API keys for services like Google Gemini, Mistral AI, and optionally Pinecone.

*   **Local Development**: You can place your API keys in a `.env` file in the root directory of the project. The application will load these keys at startup. Example `.env` file content:

    ```
    GEMINI_API_KEY=your_gemini_api_key
    MISTRAL_API_KEY=your_mistral_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    ```

*   **Deployed Environments**: When deploying the application (e.g., to Streamlit Cloud, Heroku, AWS, GCP, Azure), it is recommended to use the platform's system for managing environment variables or secrets. Do not commit your `.env` file or hardcode API keys directly into the application. The application will read these environment variables if set. The sidebar configuration for API keys can also be used, but for persistent settings in a deployed app, environment variables are preferred.

## API Reference

The scraper exposes several key functions that can be imported and used in other projects:

- `advanced_scrape(url, max_pages, pagination_strategy, selectors, auth_config, proxy)`: Advanced single-URL scraper
- `multi_thread_scrape(urls, max_pages, pagination_strategy, selectors, max_workers)`: Multi-threaded scraper for multiple URLs
- `infer_selectors_with_gemini(page_content, api_key)`: Uses Gemini AI to detect page selectors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 