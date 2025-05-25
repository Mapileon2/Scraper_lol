import streamlit as st
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from models import ScrapeRequest, ScrapeField, LLMProvider
from ai_scraper_agent import AIScraperAgent

# Custom CSS
st.markdown("""
    <style>
    .stTextArea [data-baseweb="textarea"] {
        min-height: 100px;
    }
    .field-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'fields' not in st.session_state:
    st.session_state.fields = [{"name": "", "description": "", "selector": ""}]
if 'scraping' not in st.session_state:
    st.session_state.scraping = False

# Helper functions
def add_field():
    st.session_state.fields.append({"name": "", "description": "", "selector": ""})
    # Force a rerun to update the UI
    st.rerun()

def remove_field(index):
    if len(st.session_state.fields) > 1:
        st.session_state.fields.pop(index)
        # Force a rerun to update the UI
        st.rerun()

def update_field(index, key, value):
    if 0 <= index < len(st.session_state.fields):
        st.session_state.fields[index][key] = value

# Main app
def main():
    """Main entry point for the Streamlit app."""
    # Only set page config if running as a standalone app
    if __name__ == "__main__":
        st.set_page_config(
            page_title="AI-Powered Web Scraper",
            page_icon="🕸️",
            layout="wide"
        )
    
    st.title("🕸️ AI-Powered Web Scraper")
    st.markdown("Extract structured data from any website using AI-powered selector inference.")
    
    with st.expander("🛠️ Scraper Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            url = st.text_input("🔗 URL to Scrape", placeholder="https://example.com")
            llm_provider = st.selectbox(
                "🤖 LLM Provider",
                options=[provider.value for provider in LLMProvider],
                format_func=lambda x: x.capitalize()
            )
            llm_api_key = st.text_input(
                "🔑 LLM API Key",
                type="password",
                help="Required for selector inference"
            )
        
        with col2:
            max_pages = st.number_input("📄 Max Pages", min_value=1, value=1, 
                                     help="Maximum number of pages to scrape")
            headless = st.checkbox("Headless Browser", value=True,
                                help="Run browser in headless mode")
            timeout = st.number_input("⏱️ Timeout (seconds)", min_value=10, value=30)
    
    # Fields to scrape
    st.subheader("📝 Fields to Scrape")
    st.markdown("Add fields you want to extract. The AI will try to find the best selectors.")
    
    for i, field in enumerate(st.session_state.fields):
        with st.container():
            cols = st.columns([1, 3, 5, 1])
            with cols[0]:
                name = st.text_input(
                    "Name", 
                    value=field["name"],
                    key=f"name_{i}",
                    on_change=lambda i=i: update_field(i, "name", st.session_state[f"name_{i}"]),
                    args=()
                )
                if name != field["name"]:
                    update_field(i, "name", name)
                    
            with cols[1]:
                desc = st.text_input(
                    "Description", 
                    value=field["description"],
                    key=f"desc_{i}",
                    on_change=lambda i=i: update_field(i, "description", st.session_state[f"desc_{i}"]),
                    args=()
                )
                if desc != field["description"]:
                    update_field(i, "description", desc)
                    
            with cols[2]:
                selector = st.text_input(
                    "Selector (optional)", 
                    value=field["selector"],
                    key=f"selector_{i}",
                    on_change=lambda i=i: update_field(i, "selector", st.session_state[f"selector_{i}"]),
                    args=()
                )
                if selector != field["selector"]:
                    update_field(i, "selector", selector)
                    
            with cols[3]:
                st.markdown("##")
                st.button("❌", 
                         key=f"remove_{i}",
                         on_click=remove_field, 
                         args=(i,))
    
    # Add field button
    st.button("➕ Add Field", 
             key="add_field_button",
             on_click=add_field)
    
    # Scrape button
    if st.button("🚀 Start Scraping", 
                type="primary", 
                use_container_width=True,
                key="start_scraping_button"):
        if not url:
            st.error("Please enter a URL to scrape")
            return
            
        # Validate fields
        fields = []
        for field in st.session_state.fields:
            if not field["name"] or not field["description"]:
                st.error("All fields must have a name and description")
                return
            fields.append(ScrapeField(**field))
        
        # Create scrape request
        try:
            request = ScrapeRequest(
                url=url,
                fields=fields,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key or None,
                max_pages=max_pages,
                headless=headless,
                timeout=timeout
            )
            
            # Start scraping
            st.session_state.scraping = True
            with st.spinner("🔄 Scraping in progress..."):
                asyncio.run(run_scraper(request))
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.scraping = False
    
    # Display results if available
    if 'scraper_results' in st.session_state:
        st.subheader("📊 Results")
        results = st.session_state.scraper_results
        
        # Show raw results
        with st.expander("View Raw Results"):
            # Convert results to a serializable format
            serializable_results = [
                {
                    "url": r.get("url", ""),
                    "data": r.get("data", {}),
                    "status": r.get("status", ""),
                    "error": r.get("error", "")
                }
                for r in results
            ]
            st.json(serializable_results)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(serializable_results, indent=2, default=str),
                file_name="scraped_data.json",
                mime="application/json"
            )
        with col2:
            # Convert to CSV if results are tabular
            try:
                import pandas as pd
                # Flatten the data for CSV export
                flat_data = []
                for item in results:
                    flat_item = {"url": item.get("url", "")}
                    if "data" in item and isinstance(item["data"], dict):
                        flat_item.update({
                            f"data_{k}": v for k, v in item["data"].items()
                        })
                    flat_data.append(flat_item)
                
                df = pd.DataFrame(flat_data)
                if not df.empty:
                    st.download_button(
                        label="📊 Download CSV",
                        data=df.to_csv(index=False, encoding='utf-8'),
                        file_name="scraped_data.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.warning(f"Could not convert to CSV: {str(e)}")

async def run_scraper(request: ScrapeRequest):
    """Run the scraper and update the UI with results."""
    try:
        scraper = AIScraperAgent()
        results = await scraper.scrape(request)
        
        # Store results in session state
        st.session_state.scraper_results = [
            {"url": r.url, "data": r.data, "status": r.status, "error": r.error}
            for r in results
        ]
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
        st.session_state.scraping = False
        st.rerun()

if __name__ == "__main__":
    main()
