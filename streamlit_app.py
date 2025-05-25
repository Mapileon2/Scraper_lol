import streamlit as st
import os
import sys

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main app
from app_improved import main

# Set page config
st.set_page_config(
    page_title="Web Scraper Pro",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run the main app
if __name__ == "__main__":
    main()
