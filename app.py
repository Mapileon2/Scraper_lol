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
    
    if data_source == "Current Session":
        if 'scraped_results' in st.session_state:
            df = pd.DataFrame(st.session_state['scraped_results'])
            st.success(f"Loaded {len(df)} results from current session")
        elif 'multi_scrape_results' in st.session_state:
            # Flatten multi-URL results
            all_data = []
            for url, result in st.session_state['multi_scrape_results'].items():
                if isinstance(result, list):
                    for item in result:
                        item['source_url'] = url
                        all_data.append(item)
            df = pd.DataFrame(all_data)
            st.success(f"Loaded {len(df)} results from multi-URL scrape")
        else:
            st.info("No data available in current session. Run a scrape first or select a previous session.")
    elif data_source == "Previous Session":
        if 'selected_session_id' in st.session_state:
            session_info, results = load_session(st.session_state['selected_session_id'])
            if results:
                df = pd.DataFrame(results)
                st.success(f"Loaded {len(df)} results from session on {session_info['date'][:10]}")
                st.write(f"URL: {session_info['url']}")
            else:
                st.error("Failed to load data from selected session")
        else:
            st.info("Select a previous session from the sidebar")
    
    if df is not None and not df.empty:
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Data cleaning options
        st.subheader("Data Cleaning")
        
        with st.expander("Clean and Transform Data"):
            # Remove duplicates
            if st.checkbox("Remove Duplicates"):
                orig_len = len(df)
                df = df.drop_duplicates()
                st.write(f"Removed {orig_len - len(df)} duplicate rows")
            
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
            selected_cat_col = st.selectbox("Select categorical column", categorical_cols)
            if selected_cat_col:
                # Count values and take top 10
                value_counts = df[selected_cat_col].value_counts().reset_index().head(10)
                value_counts.columns = ['value', 'count']
                
                fig = px.bar(value_counts, x='value', y='count', title=f"Top 10 {selected_cat_col} values")
                st.plotly_chart(fig)
        
        # Data export
        st.subheader("Export Processed Data")
        col1, col2, col3 = st.columns(3)
        
        # Excel export
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

if __name__ == "__main__":
    pass
