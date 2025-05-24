import sqlite3
import logging
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Define SQLAlchemy components
SQLALCHEMY_DATABASE_URL = "sqlite:///./scraper.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define models to match the database tables
class ScrapingSession(Base):
    __tablename__ = "scrape_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="completed")
    num_pages = Column(Integer, default=1)
    config = Column(Text)
    
    # Relationships
    results = relationship("ScrapedPage", back_populates="session")
    
class ScrapedPage(Base):
    __tablename__ = "scrape_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("scrape_sessions.id"))
    url = Column(String, nullable=False)
    page_num = Column(Integer)
    data = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ScrapingSession", back_populates="results")
    
class ExtractedData(Base):
    __tablename__ = "processed_data"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("scrape_sessions.id"))
    analysis_type = Column(String)
    result_data = Column(Text)
    metadata_json = Column(Text)  # Renamed from 'metadata' as it's a reserved keyword
    timestamp = Column(DateTime, default=datetime.utcnow)

def init_db():
    """Initialize the database and create tables."""
    try:
        # Create SQLAlchemy tables
        Base.metadata.create_all(bind=engine)
        
        # Also initialize SQLite tables for backward compatibility
        conn = sqlite3.connect('scraper.db')
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scrape_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'completed',
            num_pages INTEGER DEFAULT 1,
            config TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scrape_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            url TEXT NOT NULL,
            page_num INTEGER,
            data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES scrape_sessions(id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            analysis_type TEXT,
            result_data TEXT,
            metadata_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES scrape_sessions(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully with SQLAlchemy and SQLite")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

# Function to get a new database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
