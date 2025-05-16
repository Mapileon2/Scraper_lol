# Enhanced Web Scraper & Data Processor

A powerful, AI-assisted web scraping tool designed to extract structured business data from various websites with advanced scraping capabilities and data processing features.

## Features

### Advanced Scraping Capabilities

- **Dynamic Pagination Handling**: Automatically detects and handles various pagination patterns including infinite scroll, numbered pages, and next buttons.
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
streamlit run app.py
```

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

## API Reference

The scraper exposes several key functions that can be imported and used in other projects:

- `advanced_scrape(url, max_pages, pagination_strategy, selectors, auth_config, proxy)`: Advanced single-URL scraper
- `multi_thread_scrape(urls, max_pages, pagination_strategy, selectors, max_workers)`: Multi-threaded scraper for multiple URLs
- `infer_selectors_with_gemini(page_content, api_key)`: Uses Gemini AI to detect page selectors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 