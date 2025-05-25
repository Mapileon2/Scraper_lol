"""
Streamlit initialization and configuration module.
This module handles the initialization of Streamlit settings and configurations.
"""

import streamlit as st

# Apply custom CSS for better styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)
