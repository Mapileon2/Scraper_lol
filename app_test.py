import streamlit as st
import pandas as pd
import os
from src.scraper_agent import ScraperAgent

# App title
st.title("ScraperAgent Test")
st.write("Testing the ScraperAgent class")

# API key input
api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")

# URL input
url = st.text_input("Enter Website URL", "https://example.com")
max_pages = st.number_input("Max Pages to Scrape", min_value=1, max_value=10, value=3)
wait_time = st.number_input("Wait Time (seconds)", min_value=1, max_value=10, value=2)

# Initialize session state
if "selector_suggestions" not in st.session_state:
    st.session_state.selector_suggestions = []
if "confirmed_selectors" not in st.session_state:
    st.session_state.confirmed_selectors = {}
if "scraped_preview" not in st.session_state:
    st.session_state.scraped_preview = []

# Step 1: Analyze First Page
if st.button("Analyze First Page"):
    if api_key:
        with st.spinner("Analyzing first page..."):
            agent = ScraperAgent(api_key, model="gemini-1.5-flash")
            suggestions = agent.analyze_first_page(url, wait_time)
            st.session_state.selector_suggestions = suggestions
            preview_selectors = {s.field: s.selector for s in suggestions}
            preview_data = agent.scrape_pages(url, preview_selectors, max_pages=1, wait_time=wait_time)
            st.session_state.scraped_preview = preview_data
            st.success("First page analyzed! Review suggested selectors and preview below.")
    else:
        st.error("Please provide a Gemini API key.")

# Step 2: Display Preview and Confirm Selectors
if st.session_state.scraped_preview:
    st.subheader("Preview of First Page Data")
    st.dataframe(pd.DataFrame(st.session_state.scraped_preview))

if st.session_state.selector_suggestions:
    st.subheader("Suggested Selectors")
    confirmed_selectors = {}
    for suggestion in st.session_state.selector_suggestions:
        st.write(f"**Field**: {suggestion.field}")
        st.write(f"**Sample Data**: {suggestion.sample_data or 'N/A'}")
        st.write(f"**Confidence**: {suggestion.confidence:.2f}")
        selector = st.text_input(f"Selector for {suggestion.field}", value=suggestion.selector, key=f"selector_{suggestion.field}")
        confirmed_selectors[suggestion.field] = selector
    if st.button("Confirm Selectors"):
        st.session_state.confirmed_selectors = confirmed_selectors
        st.success("Selectors confirmed! Proceed to scrape all pages.")

# Step 3: Scrape All Pages
if st.session_state.confirmed_selectors:
    if st.button("Scrape All Pages"):
        with st.spinner("Scraping website..."):
            agent = ScraperAgent(api_key, model="gemini-1.5-flash")
            results = agent.scrape_pages(
                url=url,
                selectors=st.session_state.confirmed_selectors,
                max_pages=max_pages,
                wait_time=wait_time
            )
            if results:
                st.success(f"Scraped {len(results)} items")
                st.dataframe(pd.DataFrame(results))
            else:
                st.error("No data scraped. Check URL or selectors.") 