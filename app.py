import sys
import asyncio
import streamlit as st
import os
import json
import sqlite3
from uuid import uuid4
import time
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
import sys
import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from src.scraper_agent_clean import ScraperAgent, SelectorSuggestion, ScrapedDataInput
import time
import json
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure enhanced logging for production use
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting Web Scraper Pro application")

# Import pinecone
try:
    from pinecone import Pinecone
    PINE_IMPORTED = True
except ImportError as e:
    logger.error(f"Error importing Pinecone: {e}")
    PINE_IMPORTED = False

# Global variables for Pinecone
pc = None
pc_index = None

def init_pinecone():
    """Initialize Pinecone with the API key from session state."""
    global pc, pc_index
    
    if not PINE_IMPORTED:
        st.error("Pinecone is not installed. Please install it with 'pip install pinecone'")
        return False
        
    if 'pinecone_api_key' not in st.session_state or not st.session_state.pinecone_api_key:
        st.warning("⚠️ Pinecone API key is not set. Please enter your Pinecone API key in the sidebar.")
        return False
        
    try:
        logger.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=st.session_state.pinecone_api_key)
        logger.info("Pinecone client initialized successfully")
        
        # Ensure index exists
        index_name = "tekken"
        
        # List indexes
        existing_indexes = pc.list_indexes().names()
        
        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {index_name}")
            try:
                # Create the index with the new API
                pc.create_index(
                    name=index_name,
                    dimension=768,  # Gemini embeddings dimension
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                st.toast(f"Created new Pinecone index: {index_name}")
            except Exception as e:
                logger.error(f"Error creating Pinecone index: {e}")
                st.error(f"Failed to create Pinecone index: {e}")
                return False
        
        # Connect to the index
        pc_index = pc.Index(index_name)
        st.session_state.pinecone_index = pc_index
        logger.info("Pinecone index connected successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        st.error(f"Failed to initialize Pinecone: {e}")
        return False

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

# Import local modules with error handling
try:
    logger.info("Attempting to import local modules...")
    
    # Import base utilities first
    from src.models import ScrapedDataInput, DataAnalysisOutput, AnalysisType, SelectorSuggestion
    logger.info("Imported src.models")
    
    from src.database import SessionLocal, engine, init_db, Base, ScrapingSession, ScrapedPage, ExtractedData
    logger.info("Imported src.database")
    
    # Import functional modules
    from src.scraper import scrape_and_index, advanced_scrape, multi_thread_scrape
    logger.info("Imported src.scraper")
    
    from src.reasoning import ReasoningAgent
    logger.info("Imported src.reasoning")
    
    from src.async_utils import run_async
    logger.info("Imported src.async_utils")
    
    # Import scraper agents
    from src.scraper_agent_clean import ScraperAgent as CleanScraperAgent
    logger.info("Imported src.scraper_agent_clean")
    
    try:
        from src.crawl4ai_scraper import Crawl4AIScraper
        logger.info("Imported src.crawl4ai_scraper")
        CRAWL4AI_AVAILABLE = True
    except ImportError:
        logger.warning("Crawl4AI module not available - this feature will be disabled")
        CRAWL4AI_AVAILABLE = False
        
        # Define a stub for the Crawl4AI class
        class Crawl4AIScraper:
            def __init__(self, *args, **kwargs):
                raise ImportError("Crawl4AI is not available. Please install it with 'pip install crawl4ai[all]'")
    
    # Use CleanScraperAgent as the main ScraperAgent
    ScraperAgent = CleanScraperAgent
    logger.info("Aliased CleanScraperAgent to ScraperAgent")
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}", exc_info=True)
    st.error(f"⚠️ Critical error: Failed to load required modules. Application may not function correctly.\n\nError details: {str(e)}")
    CRAWL4AI_AVAILABLE = False
    
    # Define stubs for IDE autocompletion
    class Crawl4AIScraper:
        def __init__(self, *args, **kwargs):
            raise ImportError("Crawl4AI is not available. Please install it with 'pip install crawl4ai[all]'")

# Set page config at the very top - this must be the first Streamlit command
st.set_page_config(
    page_title="Web Scraper Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize database and run automatic migration if needed
try:
    # Initialize the database using the imported init_db function
    init_db()
    logger.info("Database initialization successful")
    
    # Run automatic migration from legacy database if needed
    from src.migrate_db import auto_migrate_on_startup
    integrity_ok, migration_count = auto_migrate_on_startup(silent=True)
    
    if migration_count > 0:
        logger.info(f"Automatic migration completed: {migration_count} sessions migrated")
        st.toast(f"✅ Migrated {migration_count} sessions from legacy database")
    
    if not integrity_ok:
        logger.warning("Database integrity issues detected during startup")
        # We'll show a warning in the sidebar but still allow the app to run
    
    init_db_complete = True
except Exception as e:
    logger.error(f"Database initialization failed: {e}", exc_info=True)
    init_db_complete = False
    # We'll display an error to the user in the sidebar instead of immediately
    # so the app can still load even with database issues

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
        .stSelectbox>div>div>div {
            border-radius: 10px;
        }
        .stProgress>div>div>div>div {
            background-color: #4CAF50;
        }
        .success-msg {
            color: #4CAF50;
            font-weight: bold;
        }
        .error-msg {
            color: #f44336;
            font-weight: bold;
        }
        .warning-msg {
            color: #ff9800;
            font-weight: bold;
        }
        .info-msg {
            color: #2196F3;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.scraped_data = None
    st.session_state.processed_data = None
    st.session_state.analysis_results = {}
    st.session_state.chat_history = []
    st.session_state.show_advanced = False

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

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

def generate_gemini_embedding(text):
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
    """Set up SQLite database for storing scraping results (legacy and new format)"""
    try:
        # Use our new database setup for the main database
        init_db()
        
        # Also ensure legacy database is set up for backward compatibility
        legacy_conn = sqlite3.connect('scraping_results.db')
        cursor = legacy_conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraping_sessions (
            id TEXT PRIMARY KEY,
            url TEXT,
            timestamp TEXT,
            selectors TEXT,
            num_pages INTEGER,
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
        
        legacy_conn.commit()
        legacy_conn.close()
        
        logger.info("Database setup complete for both current and legacy formats")
        return True
    except Exception as e:
        logger.error(f"Error setting up databases: {e}", exc_info=True)
        st.error(f"⚠️ Database setup error: {str(e)}. Some features may not work correctly.")
        return False

# Only call setup_database() if our initial database initialization succeeded
# to ensure the legacy database is also set up
if init_db_complete:
    try:
        legacy_db_setup = setup_database()
        if legacy_db_setup:
            logger.info("Legacy database setup completed successfully")
        else:
            logger.warning("Legacy database setup completed with warnings")
    except Exception as e:
        logger.error(f"Failed to set up legacy database: {e}", exc_info=True)
        # We already display database errors in the sidebar, so no need for another alert

# Save scraped results to database
def save_to_database(url, results, selectors, pagination_strategy):
    """Save scraping results to both SQLite databases (new and legacy format)"""
    try:
        # Generate a unique session ID
        session_id = str(uuid4())
        timestamp = datetime.now().isoformat()
        
        # First save to the new primary database
        try:
            db = SessionLocal()
            # Create session record
            db_session = ScrapingSession(
                url=url,
                timestamp=datetime.now(),
                status="completed",
                num_pages=len(results),
                config=json.dumps({
                    "selectors": selectors,
                    "pagination_strategy": pagination_strategy
                })
            )
            db.add(db_session)
            db.flush()
            db.refresh(db_session)
            
            # Add all results
            for i, result in enumerate(results):
                db_result = ScrapedPage(
                    session_id=db_session.id,
                    url=result.get('url', url),
                    page_num=result.get('page_num', i+1),
                    data=json.dumps(result)
                )
                db.add(db_result)
                
            db.commit()
            logger.info(f"Saved session {db_session.id} to primary database")
        except Exception as e:
            logger.error(f"Error saving to primary database: {e}")
            if 'db' in locals():
                db.rollback()
        finally:
            if 'db' in locals():
                db.close()
        
        # Also save to legacy database for backward compatibility
        try:
            conn = sqlite3.connect('scraping_results.db')
            cursor = conn.cursor()
            
            # Save session info
            cursor.execute(
                "INSERT INTO scraping_sessions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    timestamp,
                    url,
                    json.dumps(selectors),
                    len(results),
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
            logger.info(f"Saved session {session_id} to legacy database")
        except Exception as e:
            logger.error(f"Error saving to legacy database: {e}")
        
        return session_id
    except Exception as e:
        logger.error(f"Unexpected error in save_to_database: {e}", exc_info=True)
        return None

# Get previous scraping sessions
def get_sessions():
    """Get list of previous scraping sessions from both databases"""
    all_sessions = []
    
    # First try the new database
    try:
        db = SessionLocal()
        new_sessions = db.query(ScrapingSession).order_by(ScrapingSession.timestamp.desc()).all()
        
        for session in new_sessions:
            all_sessions.append([
                session.id,
                session.timestamp.isoformat() if hasattr(session.timestamp, 'isoformat') else str(session.timestamp),
                session.url,
                session.num_pages
            ])
    except Exception as e:
        logger.error(f"Error getting sessions from primary database: {e}")
    finally:
        if 'db' in locals():
            db.close()
    
    # Then try the legacy database
    try:
        conn = sqlite3.connect('scraping_results.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, url, num_pages FROM scraping_sessions ORDER BY timestamp DESC")
        legacy_sessions = cursor.fetchall()
        conn.close()
        
        # Add legacy sessions that aren't already in our list
        legacy_ids = [s[0] for s in legacy_sessions]
        existing_ids = [s[0] for s in all_sessions]
        
        for session in legacy_sessions:
            if session[0] not in existing_ids:
                all_sessions.append(session)
    except Exception as e:
        logger.error(f"Error getting sessions from legacy database: {e}")
    
    return all_sessions

# Load results for a specific session
def load_session(session_id):
    """Load scraping results for a specific session from either database"""
    # First try the new database
    try:
        db = SessionLocal()
        # Check if session_id is a string (UUID) or integer
        if isinstance(session_id, str) and not session_id.isdigit():
            # Try to convert to int if it's a digit string
            query = db.query(ScrapingSession).filter(ScrapingSession.id == session_id)
        else:
            # Otherwise treat as integer ID
            query = db.query(ScrapingSession).filter(ScrapingSession.id == int(session_id))
            
        session = query.first()
        
        if session:
            # Get the results for this session
            results = db.query(ScrapedPage).filter(ScrapedPage.session_id == session.id).all()
            
            # Format session info
            session_info = {
                'id': session.id,
                'url': session.url,
                'timestamp': session.timestamp.isoformat() if hasattr(session.timestamp, 'isoformat') else str(session.timestamp),
                'status': session.status,
                'num_pages': session.num_pages,
                'config': json.loads(session.config) if session.config else {}
            }
            
            # Format results
            results_data = []
            for result in results:
                try:
                    data = json.loads(result.data)
                    data['page_num'] = result.page_num
                    data['url'] = result.url
                    results_data.append(data)
                except:
                    # If we can't parse JSON, create a basic result
                    results_data.append({
                        'page_num': result.page_num,
                        'url': result.url,
                        'data': result.data
                    })
            
            db.close()
            return session_info, results_data
    except Exception as e:
        logger.error(f"Error loading session from primary database: {e}", exc_info=True)
    finally:
        if 'db' in locals():
            db.close()
    
    # If we're here, try the legacy database
    try:
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
        results = []
        for row in data_rows:
            try:
                results.append(json.loads(row[0]))
            except json.JSONDecodeError:
                # Handle corrupted data
                logger.warning(f"Corrupted JSON data in session {session_id}")
                results.append({"error": "Data corrupted", "raw": str(row[0])[:100]})
                
        session_info = {
            'id': session[0],
            'date': session[1],
            'url': session[2],
            'num_results': session[3] if len(session) > 3 else len(results),
            'selectors': json.loads(session[4]) if len(session) > 4 and session[4] else {},
            'pagination_strategy': session[5] if len(session) > 5 else "auto"
        }
        
        return session_info, results
    except Exception as e:
        logger.error(f"Error loading session from legacy database: {e}", exc_info=True)
        return None, []

# Sidebar with API keys and app info
with st.sidebar:
    st.title("🔑 API Configuration")
    
    # Display database status
    if not init_db_complete:
        st.error("⚠️ **Database Error**: Application may not function properly. Check logs for details.")
        st.info("You can still use some features but data will not be saved properly.")
    
    with st.expander("API Keys", expanded=True):
        # Initialize API keys in session state if not already present
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if 'pinecone_api_key' not in st.session_state:
            st.session_state.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
            
        # Update session state when input changes
        new_gemini_key = st.text_input(
            "Gemini API Key", 
            value=st.session_state.gemini_api_key, 
            type="password",
            help="Get your API key from https://ai.google.dev/",
            key="gemini_api_key_input"
        )
        
        new_pinecone_key = st.text_input(
            "Pinecone API Key", 
            value=st.session_state.pinecone_api_key,
            type="password",
            help="Get your API key from https://www.pinecone.io/",
            key="pinecone_api_key_input"
        )
        
        # Update session state when input changes
        st.session_state.gemini_api_key = new_gemini_key
        st.session_state.pinecone_api_key = new_pinecone_key
        
        if st.button("💾 Save Keys", key="save_keys_btn", use_container_width=True):
            if st.session_state.gemini_api_key and st.session_state.pinecone_api_key:
                os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
                os.environ["PINECONE_API_KEY"] = st.session_state.pinecone_api_key
                st.success("✅ API keys saved successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please provide both API keys")
    
    # App info and status
    st.divider()
    st.markdown("### App Status")
    
    # API status indicators
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gemini API", "✅ Ready" if st.session_state.get("gemini_api_key") else "❌ Missing")
    with col2:
        st.metric("Pinecone API", "✅ Ready" if st.session_state.get("pinecone_api_key") else "❌ Missing")
    
    # Session info
    st.divider()
    st.markdown("### Session Info")
    st.code(f"Session ID: {st.session_state.session_id}")
    
    if st.button("🔄 New Session", key="new_session_btn", use_container_width=True):
        st.session_state.session_id = str(uuid4())
        st.session_state.scraped_data = None
        st.session_state.processed_data = None
        st.session_state.analysis_results = {}
        st.session_state.chat_history = []
        st.rerun()
    
    # App info
    st.divider()
    st.markdown("### About")
    st.markdown("""
        **Web Scraper Pro**  
        Version 1.0.0  
        [GitHub Repo](https://github.com/yourusername/web-scraper-pro)  
        
        Built with ❤️ using Streamlit
    """)
    
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

# Use API keys from session state
if not st.session_state.get("gemini_api_key") or not st.session_state.get("pinecone_api_key"):
    st.error("API keys for Gemini and Pinecone must be set.")
    st.stop()

# Import Gemini after API key check
import google.generativeai as genai
genai.configure(api_key=st.session_state.get("gemini_api_key", ""))

# Initialize Pinecone if API key is available
if 'pinecone_api_key' in st.session_state and st.session_state.pinecone_api_key:
    if not init_pinecone():
        st.warning("Pinecone is not properly initialized. Some features may not work.")
else:
    st.warning("Please enter your Pinecone API key in the sidebar to enable vector search.")

# Create tabs with icons
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
    url = st.text_input("Enter Website URL")
    max_pages = st.number_input("Max Pages to Scrape", min_value=1, max_value=10, value=5)
    wait_time = st.number_input("Wait Time (seconds)", min_value=1.0, max_value=15.0, value=10.0)
    api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.get("gemini_api_key", ""))

    # ... rest of tab2 code, properly indented ...

with tab3:
    st.header("Multi-URL Scraper")
    urls_input = st.text_area("Enter URLs (one per line)", "", key="multi_urls")
    with st.expander("Advanced Options"):
        # ... advanced options code ...
        pass

    if urls_input and st.button("Scrape Multiple URLs", key="multi_scrape_btn"):
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
        if not urls:
            st.error("Please enter at least one valid URL")
        else:
            with st.spinner(f"Scraping {len(urls)} URLs in parallel..."):
                try:
                    results = multi_thread_scrape(
                        urls=urls,
                        max_pages=max_pages_multi,
                        pagination_strategy=pagination_strategy_multi,
                        selectors=None,
                        max_workers=max_workers,
                        infer_selectors=use_ai_selectors_multi,
                        api_key=gemini_api_key
                    )
                    st.session_state['multi_scrape_results'] = results
                    st.success(f"Successfully scraped {len(results)} URLs!")
                    url_tabs = st.tabs([url.split("//")[-1][:20] + "..." for url in results.keys()])
                    for i, (url, result) in enumerate(results.items()):
                        with url_tabs[i]:
                            if isinstance(result, list):
                                df = pd.DataFrame(result)
                                st.dataframe(df)
                                col1, col2 = st.columns(2)
                                # Excel download
                                xls_buffer = io.BytesIO()
                                with pd.ExcelWriter(xls_buffer, engine="openpyxl") as writer:
                                    df.to_excel(writer, index=False)
                                xls_data = xls_buffer.getvalue()
                                col1.download_button(
                                    f"Download Excel",
                                    data=xls_data,
                                    file_name=f"scraped_{i}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                                # CSV download
                                csv_buffer = io.StringIO()
                                df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                col2.download_button(
                                    f"Download CSV",
                                    data=csv_data,
                                    file_name=f"scraped_{i}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error(f"Scraping failed: {result.get('error', 'Unknown error')}")
                    # ... rest of tab3 code ...
                except Exception as e:
                    st.error(f"Error during multi-URL scraping: {str(e)}")

with tab4:
    st.header("Data Analysis")
    # ... rest of tab4 code, properly indented ...

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
    """Save AI-processed analysis results to database with error handling"""
    try:
        conn = sqlite3.connect('scraping_results.db')
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_data (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                analysis_type TEXT,
                result TEXT,
                metadata_json TEXT,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES scraping_sessions(id)
            )
        """)
        
        successful_saves = 0
        errors = 0
        
        for _, row in processed_df.iterrows():
            try:
                result_id = str(uuid4())
                
                # Safely convert result to JSON, handling complex objects
                try:
                    result_json = json.dumps(row["result"])
                except (TypeError, OverflowError):
                    # If complex object that can't be JSON serialized, convert to string representation
                    result_json = json.dumps(str(row["result"]))
                
                # Safely convert metadata to JSON
                if "metadata" in row and row["metadata"]:
                    try:
                        metadata_json = json.dumps(row["metadata"])
                    except (TypeError, OverflowError):
                        metadata_json = json.dumps(str(row["metadata"]))
                else:
                    metadata_json = json.dumps({})
                
                cursor.execute(
                    "INSERT INTO processed_data VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        result_id,
                        session_id,
                        row["analysis_type"],
                        result_json,
                        metadata_json,
                        datetime.now().isoformat()
                    )
                )
                successful_saves += 1
            except Exception as e:
                errors += 1
                print(f"Error saving row to database: {e}")
        
        conn.commit()
        conn.close()
        
        if errors > 0:
            print(f"Warning: Failed to save {errors} out of {len(processed_df)} results")
            
        return successful_saves
    except Exception as e:
        print(f"Database error: {e}")
        return 0

# Function to load processed data from database
def load_processed_data(session_id):
    """Load processed analysis results from database with error handling"""
    try:
        conn = sqlite3.connect('scraping_results.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, analysis_type, result, metadata_json, created_at
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
            try:
                # Safe JSON parsing
                try:
                    result = json.loads(row[2])
                except json.JSONDecodeError:
                    result = {"error": "Invalid JSON", "text": row[2][:100]}
                    
                try:
                    metadata = json.loads(row[3])
                except json.JSONDecodeError:
                    metadata = {"error": "Invalid JSON"}
                
                data.append({
                    "id": row[0],
                    "analysis_type": row[1],
                    "result": result,
                    "metadata": metadata,
                    "created_at": row[4]
                })
            except Exception as e:
                print(f"Error processing database row: {e}")
                # Add a minimal entry so we don't lose data completely
                data.append({
                    "id": row[0] if row[0] else "unknown",
                    "analysis_type": row[1] if row[1] else "unknown",
                    "result": {"error": "Failed to load data"},
                    "metadata": {"error_details": str(e)},
                    "created_at": row[4] if row[4] else datetime.now().isoformat()
                })
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return pd.DataFrame()

class ScrapedData(BaseModel):
    data: Dict[str, Any]
    url: str
    page_num: int

class SelectorSuggestion(BaseModel):
    field: str
    selector: str
    sample_data: Optional[str]
    confidence: float

if __name__ == "__main__":
    # This block is just for testing and will only run when the script is executed directly
    # It's not part of the Streamlit app's functionality
    try:
        url = "https://example.com"
        response = requests.get(url, timeout=5)  # Add a timeout to prevent hanging
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, "html.parser")
        titles = [h1.text for h1 in soup.find_all("h1")]
        print(f"Found {len(titles)} headings on {url}")
        
        df = pd.DataFrame({"url": [url]})
        print(df)
    except requests.RequestException as e:
        # Silently handle any request errors - this is just example code
        pass

        # Advanced Web Scraping Section
    st.markdown("---")
    st.header("🔍 Advanced Web Scraper")
    
    with st.expander("⚙️ Configure Scraper", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            url = st.text_input("Enter URL to scrape", "https://example.com")
            max_pages = st.number_input("Max pages to scrape", min_value=1, max_value=100, value=3)
            wait_time = st.number_input("Wait time between pages (seconds)", min_value=0.5, max_value=10.0, value=2.0)
            
        with col2:
            enable_js = st.checkbox("Enable JavaScript", value=True)
            use_proxy = st.checkbox("Use proxy", value=False)
            proxy = st.text_input("Proxy (optional)", "") if use_proxy else ""
    
    # Selector configuration
    st.subheader("🔧 Configure Selectors")
    selectors = {}
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            selector_name = st.text_input("Field Name (e.g., 'title', 'price')")
            selector_value = st.text_input("CSS Selector (e.g., 'h1.product-title')")
            
            if st.button("Add Selector") and selector_name and selector_value:
                if 'selectors' not in st.session_state:
                    st.session_state.selectors = {}
                st.session_state.selectors[selector_name] = selector_value
                st.success(f"Added selector: {selector_name}")
                
        with col2:
            st.markdown("### Current Selectors")
            if 'selectors' in st.session_state and st.session_state.selectors:
                for name, selector in st.session_state.selectors.items():
                    st.code(f"{name}: {selector}")
                    if st.button(f"Remove {name}", key=f"remove_{name}"):
                        del st.session_state.selectors[name]
                        st.rerun()
    
    # Start scraping button
    if st.button("🚀 Start Scraping", type="primary", use_container_width=True):
        if 'selectors' not in st.session_state or not st.session_state.selectors:
            st.error("Please add at least one selector before starting.")
        else:
            with st.spinner("🚀 Starting web scraper..."):
                try:
                    # Initialize the scraper
                    scraper = ScraperAgent(api_key=st.session_state.get('gemini_api_key', ''))
                    
                    # Start scraping
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(max_pages):
                        try:
                            status_text.text(f"Scraping page {i+1} of {max_pages}...")
                            
                            # Get the URL for the current page
                            current_url = url if i == 0 else f"{url}?page={i+1}"  # Simple pagination
                            
                            # Scrape the page
                            page_results = scraper.scrape_pages(
                                url=current_url,
                                selectors=st.session_state.selectors,
                                max_pages=1,  # We're handling pagination manually
                                wait_time=wait_time,
                                proxy=proxy if use_proxy and proxy else None,
                                show_progress=False
                            )
                            
                            if page_results:
                                results.extend(page_results)
                                
                            # Update progress
                            progress = (i + 1) / max_pages
                            progress_bar.progress(min(progress, 1.0))
                            
                            # Add a small delay between pages
                            time.sleep(wait_time)
                            
                        except Exception as e:
                            st.error(f"Error scraping page {i+1}: {str(e)}")
                            continue
                    
                    # Display results
                    if results:
                        st.success(f"✅ Successfully scraped {len(results)} items!")
                        
                        # Show results in a table
                        st.subheader("📊 Scraping Results")
                        
                        # Convert results to DataFrame for better display
                        import pandas as pd
                        df = pd.json_normalize([r for r in results if r])
                        st.dataframe(df)
                        
                        # Add download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download as CSV",
                            data=csv,
                            file_name="scraped_data.csv",
                            mime="text/csv"
                        )
                        
                        # Save results to session state
                        st.session_state.scraping_results = results
                        
                except Exception as e:
                    st.error(f"An error occurred during scraping: {str(e)}")
                    st.error(traceback.format_exc())
    
    # Add a footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2em;">
            <p>Web Scraper Pro v2.0 | Built with Streamlit</p>
            <p>© 2023-2024 Web Scraper Pro. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add some custom JavaScript for better UX
    st.components.v1.html("""
        <script>
            // Auto-scroll to top on page load
            window.onload = function() {
                window.scrollTo(0, 0);
            };
            
            // Add smooth scrolling
            document.addEventListener('DOMContentLoaded', function() {
                const links = document.querySelectorAll('a[href^="#"]');
                links.forEach(anchor => {
                    anchor.addEventListener('click', function (e) {
                        e.preventDefault();
                        const target = document.querySelector(this.getAttribute('href'));
                        if (target) {
                            target.scrollIntoView({
                                behavior: 'smooth',
                                block: 'start'
                            });
                        }
                    });
                });
            });
        </script>
    """)
