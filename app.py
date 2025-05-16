import sys
import asyncio
import streamlit as st
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from src.scraper import scrape_and_index, advanced_scrape, multi_thread_scrape
import pandas as pd
import io
import json
import plotly.express as px
import csv
from uuid import uuid4
import sqlite3
from datetime import datetime

# Add Reasoning imports
from src.reasoning import ReasoningAgent
try:
    from src.models import ScrapedDataInput, DataAnalysisOutput
except ImportError:
    # Define a minimal version
    from dataclasses import dataclass
    from typing import Dict, Any, Optional
    
    @dataclass
    class ScrapedDataInput:
        data: Dict[str, Any]
        url: str
        page_num: int
        
        def dict(self):
            return {"data": self.data, "url": self.url, "page_num": self.page_num}
    
    @dataclass
    class DataAnalysisOutput:
        analysis_type: str
        result: Dict[str, Any]
        metadata: Optional[Dict[str, Any]] = None
        
        def dict(self):
            return {
                "analysis_type": self.analysis_type,
                "result": self.result,
                "metadata": self.metadata
            }

# Load environment variables (if needed)
from dotenv import load_dotenv
load_dotenv()

def generate_gemini_embedding(text):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Database setup
def setup_database():
    """Set up SQLite database for storing scraping results"""
    conn = sqlite3.connect('scraping_results.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scraping_sessions (
        id TEXT PRIMARY KEY,
        date TEXT,
        url TEXT,
        num_results INTEGER,
        selectors TEXT,
        pagination_strategy TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scraped_data (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        page_num INTEGER,
        url TEXT,
        data TEXT,
        FOREIGN KEY (session_id) REFERENCES scraping_sessions(id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Call setup on app start
setup_database()

# Save scraped results to database
def save_to_database(url, results, selectors, pagination_strategy):
    """Save scraping results to SQLite database"""
    session_id = str(uuid4())
    conn = sqlite3.connect('scraping_results.db')
    cursor = conn.cursor()
    
    # Save session info
    cursor.execute(
        "INSERT INTO scraping_sessions VALUES (?, ?, ?, ?, ?, ?)",
        (
            session_id,
            datetime.now().isoformat(),
            url,
            len(results),
            json.dumps(selectors),
            pagination_strategy
        )
    )
    
    # Save individual results
    for i, result in enumerate(results):
        result_id = str(uuid4())
        cursor.execute(
            "INSERT INTO scraped_data VALUES (?, ?, ?, ?, ?)",
            (
                result_id,
                session_id,
                result.get('page_num', i+1),
                result.get('url', url),
                json.dumps(result)
            )
        )
    
    conn.commit()
    conn.close()
    return session_id

# Get previous scraping sessions
def get_sessions():
    """Get list of previous scraping sessions"""
    conn = sqlite3.connect('scraping_results.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, date, url, num_results FROM scraping_sessions ORDER BY date DESC")
    sessions = cursor.fetchall()
    conn.close()
    return sessions

# Load results for a specific session
def load_session(session_id):
    """Load scraping results for a specific session"""
    conn = sqlite3.connect('scraping_results.db')
    cursor = conn.cursor()
    
    # Get session info
    cursor.execute("SELECT * FROM scraping_sessions WHERE id = ?", (session_id,))
    session = cursor.fetchone()
    
    # Get session data
    cursor.execute("SELECT data FROM scraped_data WHERE session_id = ? ORDER BY page_num", (session_id,))
    data_rows = cursor.fetchall()
    
    conn.close()
    
    if not session:
        return None, []
    
    # Parse results
    results = [json.loads(row[0]) for row in data_rows]
    session_info = {
        'id': session[0],
        'date': session[1],
        'url': session[2],
        'num_results': session[3],
        'selectors': json.loads(session[4]),
        'pagination_strategy': session[5]
    }
    
    return session_info, results

# Set up API keys from environment variables or sidebar
with st.sidebar:
    st.header("API Keys")
    gemini_api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password", key="gemini_api_key")
    pinecone_api_key = st.text_input("Pinecone API Key", value=os.getenv("PINECONE_API_KEY", ""), type="password", key="pinecone_api_key")
    if st.button("Save Keys", key="save_keys_btn"):
        if gemini_api_key and pinecone_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            os.environ["PINECONE_API_KEY"] = pinecone_api_key
            st.success("API keys saved!")
        else:
            st.error("Please provide both API keys")
    
    # Option to load previous sessions
    st.header("Previous Sessions")
    sessions = get_sessions()
    if sessions:
        session_options = [f"{session[1][:10]} - {session[2]} ({session[3]} results)" for session in sessions]
        selected_session = st.selectbox("Load previous session", [""] + session_options)
        if selected_session:
            # Get the session ID from the selected option
            session_idx = session_options.index(selected_session)
            session_id = sessions[session_idx][0]
            st.session_state['selected_session_id'] = session_id

# Use API keys as before
if not gemini_api_key or not pinecone_api_key:
    st.error("API keys for Gemini and Pinecone must be set.")
    st.stop()

# Import Gemini after API key check
import google.generativeai as genai
genai.configure(api_key=gemini_api_key)

# Initialize Pinecone with the new class-based approach
pc = Pinecone(api_key=pinecone_api_key)

# Ensure index exists
index_name = "tekken"
if index_name not in pc.list_indexes().names():
    # Import ServerlessSpec here since we need it only for creation
    from pinecone import ServerlessSpec
    pc.create_index(
        name=index_name,
        dimension=768,  # Gemini embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Use Streamlit tabs for organization
tab1, tab2, tab3, tab4 = st.tabs(["Search Vector DB", "Advanced Scrape", "Multi-URL Scrape", "Data Analysis"])

with tab1:
    st.header("Search Vector DB")
    query = st.text_input("Search Query", "", key="search_query")
    sentiment_filter = st.selectbox("Sentiment Filter", ["", "positive", "negative", "neutral"], key="sentiment_filter")
    if st.button("Search", key="search_btn"):
        if query:
            with st.spinner("Searching..."):
                query_embedding = generate_gemini_embedding(query)
                if query_embedding is None:
                    st.error("Failed to generate embedding for search query.")
                else:
                    filter = {"sentiment": {"$eq": sentiment_filter}} if sentiment_filter else {}
                    results = index.query(vector=query_embedding, top_k=5, include_metadata=True, filter=filter)
                    if results and "matches" in results:
                        for match in results["matches"]:
                            meta = match["metadata"]
                            st.markdown(f"**Summary:** {meta['summary']}\n\n**Sentiment:** {meta['sentiment']}\n**Keywords:** {', '.join(meta['keywords'])}\n\n{meta['text']}")
                    else:
                        st.info("No results found.")
        else:
            st.warning("Please enter a search query.")

with tab2:
    st.header("Advanced Web Scraper")
    
    # URL Input
    url = st.text_input("Enter URL to scrape", "", key="advanced_scrape_url")
    
    # Advanced Options Expander
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Scraping Options
            max_pages = st.number_input("Max pages to scrape", min_value=1, max_value=50, value=5, key="max_pages")
            
            pagination_options = ["auto", "next_button", "infinite_scroll", "custom"]
            pagination_strategy = st.selectbox("Pagination Strategy", pagination_options, key="pagination_strategy")
            
            if pagination_strategy == "custom":
                custom_pagination = st.text_input("Custom CSS Selector for pagination", ".pagination a.next", key="custom_pagination")
                pagination_strategy = custom_pagination
            
            wait_time = st.slider("Wait time between actions (seconds)", min_value=1, max_value=10, value=2, key="wait_time")
            
            retry_limit = st.number_input("Retry limit for failures", min_value=1, max_value=10, value=3, key="retry_limit")
        
        with col2:
            # Selector Options
            use_ai_selectors = st.checkbox("Use AI to detect selectors", value=True, key="use_ai_selectors")
            
            if not use_ai_selectors:
                st.text("Enter CSS Selectors:")
                business_name_selector = st.text_input("Business Name selector", ".business-name, h1, h2.title", key="business_name")
                address_selector = st.text_input("Address selector", ".address, .location, address", key="address")
                phone_selector = st.text_input("Phone selector", ".phone, .tel, [href^='tel:']", key="phone")
                website_selector = st.text_input("Website selector", ".website a, a.website", key="website")
                custom_field = st.text_input("Custom field name", "", key="custom_field_name")
                if custom_field:
                    custom_field_selector = st.text_input(f"{custom_field} selector", "", key="custom_field_selector")
            
            # Authentication Options
            use_auth = st.checkbox("Use Authentication", value=False, key="use_auth")
            if use_auth:
                auth_type = st.selectbox("Authentication Type", ["form", "basic", "cookie", "oauth"], key="auth_type")
                if auth_type in ["form", "basic"]:
                    username = st.text_input("Username", "", key="auth_username")
                    password = st.text_input("Password", "", type="password", key="auth_password")
                    if auth_type == "form":
                        login_url = st.text_input("Login URL (if different from main URL)", "", key="login_url")
                elif auth_type == "cookie":
                    cookie_input = st.text_area("Cookies (JSON format)", "{\"name\": \"session\", \"value\": \"your-value\", \"domain\": \"example.com\"}", key="auth_cookies")
                elif auth_type == "oauth":
                    token = st.text_input("OAuth Token", "", type="password", key="auth_token")
                    token_type = st.selectbox("Token Type", ["Bearer", "Basic", "Custom"], key="auth_token_type")
            
            # Proxy Options
            use_proxy = st.checkbox("Use Proxy", value=False, key="use_proxy")
            if use_proxy:
                proxy_server = st.text_input("Proxy Server (e.g., http://proxy.example.com:8080)", "", key="proxy_server")
                proxy_auth = st.checkbox("Proxy requires authentication", value=False, key="proxy_auth")
                if proxy_auth:
                    proxy_username = st.text_input("Proxy Username", "", key="proxy_username")
                    proxy_password = st.text_input("Proxy Password", "", type="password", key="proxy_password")
    
    # Prepare selectors
    if url and st.button("Scrape Website", key="advanced_scrape_btn"):
        with st.spinner("Scraping website..."):
            try:
                # Prepare selectors
                if use_ai_selectors:
                    selectors = None  # Will be inferred by AI
                else:
                    selectors = {
                        "business_name": business_name_selector,
                        "address": address_selector,
                        "phone": phone_selector,
                        "website": website_selector
                    }
                    if custom_field and custom_field_selector:
                        selectors[custom_field] = custom_field_selector
                
                # Prepare authentication
                auth_config = None
                if use_auth:
                    auth_config = {"type": auth_type}
                    if auth_type in ["form", "basic"]:
                        auth_config["username"] = username
                        auth_config["password"] = password
                        if auth_type == "form" and login_url:
                            auth_config["login_url"] = login_url
                    elif auth_type == "cookie":
                        try:
                            auth_config["cookies"] = json.loads(cookie_input)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format for cookies")
                            auth_config["cookies"] = []
                    elif auth_type == "oauth":
                        auth_config["token"] = token
                        auth_config["token_type"] = token_type
                
                # Prepare proxy
                proxy = None
                if use_proxy:
                    if proxy_auth:
                        proxy = {
                            "server": proxy_server,
                            "username": proxy_username,
                            "password": proxy_password
                        }
                    else:
                        proxy = proxy_server
                
                # Run the scraper
                results = asyncio.run(advanced_scrape(
                    url=url,
                    max_pages=max_pages,
                    pagination_strategy=pagination_strategy,
                    selectors=selectors,
                    wait_time=wait_time,
                    auth_config=auth_config,
                    infer_selectors=use_ai_selectors,
                    api_key=gemini_api_key,
                    proxy=proxy,
                    retry_limit=retry_limit
                ))
                
                # Save results to session state
                if isinstance(results, list):
                    st.session_state['scraped_results'] = results
                    
                    # Also save to database
                    if selectors is None:
                        selectors = {
                            "inferred_by_ai": True
                        }
                    session_id = save_to_database(url, results, selectors, pagination_strategy)
                    
                    # Display results
                    st.success(f"Successfully scraped {len(results)} results!")
                    
                    # Convert to DataFrame for display
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    
                    # Download options
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Excel download
                    xls_buffer = io.BytesIO()
                    with pd.ExcelWriter(xls_buffer, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False)
                    xls_data = xls_buffer.getvalue()
                    col1.download_button("Download Excel", data=xls_data, file_name="scraped_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    
                    # CSV download
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    col2.download_button("Download CSV", data=csv_data, file_name="scraped_data.csv", mime="text/csv")
                    
                    # JSON download
                    json_data = df.to_json(orient="records")
                    col3.download_button("Download JSON", data=json_data, file_name="scraped_data.json", mime="application/json")
                    
                    # Markdown download
                    md_content = "# Scraped Data\n\n"
                    for i, row in df.iterrows():
                        md_content += f"## Result {i+1}\n\n"
                        for col in df.columns:
                            md_content += f"**{col}**: {row[col]}\n\n"
                    col4.download_button("Download Markdown", data=md_content, file_name="scraped_data.md", mime="text/markdown")
                    
                    # Display some basic stats
                    st.subheader("Quick Stats")
                    st.write(f"Total pages scraped: {max_pages}")
                    st.write(f"Total items found: {len(results)}")
                    
                    # Generate a simple chart if we have numeric data
                    st.subheader("Data Visualization")
                    try:
                        # Try to create a bar chart of some meaningful data
                        if "page_num" in df.columns:
                            counts = df["page_num"].value_counts().sort_index()
                            fig = px.bar(counts, x=counts.index, y=counts.values, title="Items per Page", labels={"x": "Page Number", "y": "Number of Items"})
                            st.plotly_chart(fig)
                    except:
                        pass
                else:
                    # Handle error
                    if isinstance(results, dict) and "error" in results:
                        st.error(f"Scraping failed: {results['error']}")
                    else:
                        st.error("Scraping failed with unknown error")
            
            except Exception as e:
                st.error(f"Error during scraping: {str(e)}")

with tab3:
    st.header("Multi-URL Scraper")
    
    urls_input = st.text_area("Enter URLs (one per line)", "", key="multi_urls")
    
    with st.expander("Advanced Options"):
        # Similar options as in Advanced Scrape tab
        max_pages_multi = st.number_input("Max pages per URL", min_value=1, max_value=20, value=3, key="max_pages_multi")
        max_workers = st.number_input("Max concurrent workers", min_value=1, max_value=10, value=3, key="max_workers")
        pagination_strategy_multi = st.selectbox("Pagination Strategy", ["auto", "next_button", "infinite_scroll"], index=0, key="pagination_strategy_multi")
        use_ai_selectors_multi = st.checkbox("Use AI to detect selectors", value=True, key="use_ai_selectors_multi")
    
    if urls_input and st.button("Scrape Multiple URLs", key="multi_scrape_btn"):
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
        
        if not urls:
            st.error("Please enter at least one valid URL")
        else:
            with st.spinner(f"Scraping {len(urls)} URLs in parallel..."):
                try:
                    # Prepare options similar to advanced scrape
                    results = multi_thread_scrape(
                        urls=urls,
                        max_pages=max_pages_multi,
                        pagination_strategy=pagination_strategy_multi,
                        selectors=None,  # Use AI inference or defaults
                        max_workers=max_workers,
                        infer_selectors=use_ai_selectors_multi,
                        api_key=gemini_api_key
                    )
                    
                    st.session_state['multi_scrape_results'] = results
                    
                    # Display results
                    st.success(f"Successfully scraped {len(results)} URLs!")
                    
                    # Display tabs for each URL
                    url_tabs = st.tabs([url.split("//")[-1][:20] + "..." for url in results.keys()])
                    
                    for i, (url, result) in enumerate(results.items()):
                        with url_tabs[i]:
                            if isinstance(result, list):
                                # Convert to DataFrame
                                df = pd.DataFrame(result)
                                st.dataframe(df)
                                
                                # Download buttons
                                col1, col2 = st.columns(2)
                                
                                # Excel download
                                xls_buffer = io.BytesIO()
                                with pd.ExcelWriter(xls_buffer, engine="openpyxl") as writer:
                                    df.to_excel(writer, index=False)
                                xls_data = xls_buffer.getvalue()
                                col1.download_button(f"Download Excel", data=xls_data, file_name=f"scraped_{i}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                                
                                # CSV download
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                col2.download_button(f"Download CSV", data=csv_data, file_name=f"scraped_{i}.csv", mime="text/csv")
                            else:
                                # Handle error
                                st.error(f"Scraping failed: {result.get('error', 'Unknown error')}")
                    
                    # Option to download all results as one file
                    st.subheader("Download All Results")
                    
                    # Combine all successful results
                    all_data = []
                    for url, result in results.items():
                        if isinstance(result, list):
                            for item in result:
                                item['source_url'] = url
                                all_data.append(item)
                    
                    if all_data:
                        all_df = pd.DataFrame(all_data)
                        
                        # Excel download for all
                        all_xls_buffer = io.BytesIO()
                        with pd.ExcelWriter(all_xls_buffer, engine="openpyxl") as writer:
                            all_df.to_excel(writer, index=False)
                        all_xls_data = all_xls_buffer.getvalue()
                        st.download_button("Download All Results (Excel)", data=all_xls_data, file_name="all_scraped_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                except Exception as e:
                    st.error(f"Error during multi-URL scraping: {str(e)}")

with tab4:
    st.header("Data Analysis")
    
    # Load data from session state or previous sessions
    data_source = st.radio("Data Source", ["Current Session", "Previous Session"])
    
    df = None
    original_df = None
    
    if data_source == "Current Session":
        if 'scraped_results' in st.session_state:
            original_df = pd.DataFrame(st.session_state['scraped_results'])
            df = original_df.copy()
            st.success(f"Loaded {len(df)} results from current session")
        elif 'multi_scrape_results' in st.session_state:
            # Flatten multi-URL results
            all_data = []
            for url, result in st.session_state['multi_scrape_results'].items():
                if isinstance(result, list):
                    for item in result:
                        item['source_url'] = url
                        all_data.append(item)
            original_df = pd.DataFrame(all_data)
            df = original_df.copy()
            st.success(f"Loaded {len(df)} results from multi-URL scrape")
        else:
            st.info("No data available in current session. Run a scrape first or select a previous session.")
    elif data_source == "Previous Session":
        if 'selected_session_id' in st.session_state:
            session_info, results = load_session(st.session_state['selected_session_id'])
            if results:
                original_df = pd.DataFrame(results)
                df = original_df.copy()
                st.success(f"Loaded {len(df)} results from session on {session_info['date'][:10]}")
                st.write(f"URL: {session_info['url']}")
                
                # Load previously processed data
                processed_df = load_processed_data(st.session_state['selected_session_id'])
                if not processed_df.empty:
                    st.session_state.processed_data = processed_df
                    st.success(f"Loaded {len(processed_df)} previously processed results")
            else:
                st.error("Failed to load data from selected session")
        else:
            st.info("Select a previous session from the sidebar")
    
    if df is not None and not df.empty:
        tabs = st.tabs(["Data Preview & Cleaning", "AI Analysis", "Chat Interface"])
        
        with tabs[0]:
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Data cleaning options
            st.subheader("Data Cleaning")
            
            with st.expander("Clean and Transform Data"):
                # Remove duplicates
                if st.checkbox("Remove Duplicates"):
                    try:
                        orig_len = len(df)
                        # Convert list columns to strings to avoid unhashable type error
                        for col in df.columns:
                            if df[col].apply(lambda x: isinstance(x, list)).any():
                                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                        df = df.drop_duplicates()
                        st.write(f"Removed {orig_len - len(df)} duplicate rows")
                    except Exception as e:
                        st.error(f"Error removing duplicates: {e}")
                
                # Fill empty values
                if st.checkbox("Fill Empty Values"):
                    fill_value = st.text_input("Fill Value", "N/A")
                    df = df.fillna(fill_value)
                    st.write("Filled empty values")
                
                # Clean text columns
                if st.checkbox("Clean Text Columns (strip whitespace)"):
                    for col in df.select_dtypes(include=['object']).columns:
                        df[col] = df[col].astype(str).str.strip()
                    st.write("Cleaned text columns")
                
                # Column filtering
                if st.checkbox("Select Columns to Keep"):
                    columns_to_keep = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist())
                    if columns_to_keep:
                        df = df[columns_to_keep]
                        st.write(f"Kept {len(columns_to_keep)} columns")
            
            # Data visualization
            st.subheader("Data Visualization")
            
            # Auto-detect numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if len(categorical_cols) > 0 and len(df) > 0:
                # Bar chart for categorical data
                try:
                    selected_cat_col = st.selectbox("Select categorical column", categorical_cols)
                    if selected_cat_col:
                        # Count values and take top 10
                        value_counts = df[selected_cat_col].value_counts().reset_index().head(10)
                        value_counts.columns = ['value', 'count']
                        
                        fig = px.bar(value_counts, x='value', y='count', title=f"Top 10 {selected_cat_col} values")
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
            
            # Data export
            st.subheader("Export Processed Data")
            col1, col2, col3 = st.columns(3)
            
            # Excel export
            try:
                xls_buffer = io.BytesIO()
                with pd.ExcelWriter(xls_buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False)
                xls_data = xls_buffer.getvalue()
                col1.download_button("Download Excel", data=xls_data, file_name="processed_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                # CSV export
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                col2.download_button("Download CSV", data=csv_data, file_name="processed_data.csv", mime="text/csv")
                
                # JSON export
                json_data = df.to_json(orient="records")
                col3.download_button("Download JSON", data=json_data, file_name="processed_data.json", mime="application/json")
            except Exception as e:
                st.error(f"Error exporting data: {e}")
        
        with tabs[1]:
            st.subheader("Analyze Scraped Data with Gemini API")
            
            # Get API key
            api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
            if not api_key:
                st.warning("Please provide a Gemini API key to use AI analysis features")
                st.stop()
            
            # Model selection
            model = st.selectbox("Select Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
            
            # Analysis options
            analysis_type = st.selectbox(
                "Analysis Type",
                ["sentiment", "summary", "categorize", "extract_issues", "custom"],
                help="Choose how to analyze the data (e.g., sentiment, summary)"
            )
            
            # Custom analysis prompt
            custom_analysis = ""
            if analysis_type == "custom":
                custom_analysis = st.text_area(
                    "Custom Analysis Prompt", 
                    "Analyze the data and provide key insights",
                    help="Describe what kind of analysis you want Gemini to perform"
                )
                
                # Save custom prompts for reuse
                if "custom_prompts" not in st.session_state:
                    st.session_state.custom_prompts = []
                
                # Saved prompts
                saved_prompts = st.session_state.custom_prompts
                if saved_prompts:
                    selected_prompt = st.selectbox(
                        "Or select from saved prompts", 
                        [""] + saved_prompts
                    )
                    if selected_prompt:
                        custom_analysis = selected_prompt
                
                # Save current prompt
                if custom_analysis and st.button("Save this prompt"):
                    if custom_analysis not in st.session_state.custom_prompts:
                        st.session_state.custom_prompts.append(custom_analysis)
                        st.success("Prompt saved!")
            
            # Batch size
            batch_size = st.slider("Batch Size (items per API call)", 1, 20, 5, 
                                 help="Smaller batches are more reliable but take longer")
            
            # Process button
            if st.button("Process Data with Gemini"):
                if not api_key:
                    st.error("Please provide a Gemini API key")
                elif analysis_type == "custom" and not custom_analysis:
                    st.error("Please provide a custom analysis prompt")
                else:
                    with st.spinner("Processing data with Gemini..."):
                        try:
                            # Check if original_df has the expected structure and convert if needed
                            data_list = []
                            for index, row in original_df.iterrows():
                                # Check if data is already in the right format
                                if "data" in row and isinstance(row["data"], dict):
                                    data = row
                                else:
                                    # Convert to new format with data in a nested dict
                                    data = {
                                        "data": {},
                                        "url": row.get("url", row.get("source_url", "")),
                                        "page_num": row.get("page_num", index + 1)
                                    }
                                    
                                    # Move all fields into data dict except url and page_num
                                    for key, value in row.items():
                                        if key not in ["url", "page_num", "source_url"]:
                                            data["data"][key] = value
                                
                                # Create ScrapedDataInput
                                try:
                                    validated_data = ScrapedDataInput(**data)
                                    data_list.append(validated_data)
                                except Exception as e:
                                    st.warning(f"Skipping invalid row {index}: {e}")
                            
                            # Initialize analysis
                            agent = ReasoningAgent(api_key, model)
                            analysis = custom_analysis if analysis_type == "custom" else analysis_type
                            
                            # Process in batches
                            processed_data = agent.process_data_batched(
                                data_list, 
                                analysis, 
                                batch_size=batch_size
                            )
                            
                            # Convert to DataFrame
                            processed_df = pd.DataFrame([p.dict() for p in processed_data])
                            
                            # Store in session state
                            st.session_state.processed_data = processed_df
                            
                            # Display results
                            st.success(f"Processed {len(processed_data)} items!")
                            st.dataframe(processed_df)
                            
                            # Save to database if session ID exists
                            if 'selected_session_id' in st.session_state:
                                save_processed_to_database(
                                    processed_df, 
                                    st.session_state.get("selected_session_id", "default")
                                )
                                st.success("Results saved to database")
                            
                            # Offer download
                            st.download_button(
                                "Download Analysis Results (JSON)",
                                data=processed_df.to_json(orient="records"),
                                file_name="gemini_analysis.json",
                                mime="application/json"
                            )
                                
                        except Exception as e:
                            st.error(f"Error processing data: {e}")
                            import traceback
                            st.error(traceback.format_exc())
        
        with tabs[2]:
            st.subheader("Chat with Processed Data")
            
            if "processed_data" in st.session_state:
                processed_df = st.session_state.processed_data
                st.write("Analysis data loaded - ask questions about it in the chat below")
                
                # Initialize chat history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                # Display chat history
                for chat in st.session_state.chat_history:
                    with st.chat_message("user"):
                        st.write(chat["question"])
                    with st.chat_message("assistant"):
                        st.write(chat["answer"])
                
                # Chat input
                chat_input = st.chat_input("Ask a question about the processed data (e.g., 'What are common themes?')")
                
                if chat_input:
                    with st.chat_message("user"):
                        st.write(chat_input)
                    
                    with st.spinner("Generating response..."):
                        try:
                            # Get API key
                            api_key = st.session_state.get("gemini_api_key", 
                                                         os.getenv("GEMINI_API_KEY", ""))
                            if not api_key:
                                st.error("Please provide a Gemini API key")
                                st.stop()
                            
                            # Initialize the agent
                            agent = ReasoningAgent(api_key, "gemini-1.5-pro")
                            
                            # Convert DataFrame rows to DataAnalysisOutput objects
                            processed_data = []
                            for _, row in processed_df.iterrows():
                                try:
                                    output = DataAnalysisOutput(
                                        analysis_type=row["analysis_type"],
                                        result=row["result"],
                                        metadata=row.get("metadata", {})
                                    )
                                    processed_data.append(output)
                                except Exception as e:
                                    st.warning(f"Skipping invalid analysis row: {e}")
                            
                            # Generate response
                            response = agent.answer_query(
                                chat_input, 
                                processed_data, 
                                st.session_state.chat_history
                            )
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "question": chat_input, 
                                "answer": response
                            })
                            
                            # Display response
                            with st.chat_message("assistant"):
                                st.write(response)
                                
                        except Exception as e:
                            st.error(f"Error in chat response: {e}")
            else:
                st.info("Process data first using the 'AI Analysis' tab to enable chat functionality.")
    else:
        st.info("No data available. Please scrape data first or select a previous session.")

# Section to search for similar content
st.header("Search")
query = st.text_input("Enter search query", "", key="search_query_2")
if st.button("Search", key="search_btn_2"):
    if query:
        with st.spinner("Searching..."):
            query_embedding = generate_gemini_embedding(query)
            if query_embedding is None:
                st.error("Failed to generate embedding for search query.")
            else:
                results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
                if results and "matches" in results:
                    for match in results["matches"]:
                        meta = match["metadata"]
                        st.markdown(f"**Title:** {meta.get('title', 'No title')}\n\n**Description:** {meta.get('description', 'No description')}\n\n**URL:** {meta.get('url', 'No URL')}")
                else:
                    st.info("No results found.")
    else:
        st.warning("Please enter a search query.")

def search_query(query, index):
    try:
        # Generate embedding for the query
        query_embedding = generate_gemini_embedding(query)
        
        if not query_embedding:
            return []
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        return results.matches
    except Exception as e:
        st.error(f"Error querying Pinecone: {str(e)}")
        return []

# Function to save processed data to database
def save_processed_to_database(processed_df, session_id):
    """Save AI-processed analysis results to database"""
    conn = sqlite3.connect('scraping_results.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_data (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            analysis_type TEXT,
            result TEXT,
            metadata TEXT,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES scraping_sessions(id)
        )
    """)
    
    for _, row in processed_df.iterrows():
        result_id = str(uuid4())
        cursor.execute(
            "INSERT INTO processed_data VALUES (?, ?, ?, ?, ?, ?)",
            (
                result_id,
                session_id,
                row["analysis_type"],
                json.dumps(row["result"]),
                json.dumps(row["metadata"] if "metadata" in row and row["metadata"] else {}),
                datetime.now().isoformat()
            )
        )
    
    conn.commit()
    conn.close()
    return True

# Function to load processed data from database
def load_processed_data(session_id):
    """Load processed analysis results from database"""
    conn = sqlite3.connect('scraping_results.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, analysis_type, result, metadata, created_at
        FROM processed_data
        WHERE session_id = ?
        ORDER BY created_at DESC
    """, (session_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    # Create DataFrame with processed results
    data = []
    for row in rows:
        data.append({
            "id": row[0],
            "analysis_type": row[1],
            "result": json.loads(row[2]),
            "metadata": json.loads(row[3]),
            "created_at": row[4]
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    pass
