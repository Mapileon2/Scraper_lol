import streamlit as st
import os

st.set_page_config(
    page_title="Web Scraper Pro",
    page_icon="🕸️",
    layout="wide"
)

st.title("Web Scraper Pro")
st.write("Please use the full version of the application for all features.")
st.write("This is a lightweight version for Vercel deployment.")

# Add a simple form to demonstrate functionality
with st.form("demo_form"):
    url = st.text_input("Enter a URL to scrape")
    submitted = st.form_submit_button("Scrape")
    if submitted and url:
        st.info(f"This is a demo. In the full version, we would scrape: {url}")
        st.success("Scraping complete! (Demo mode)")

st.warning("Note: Some features are disabled in this demo version.")
