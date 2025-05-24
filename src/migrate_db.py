"""
Database migration helper to ensure smooth upgrades between database versions.

This module provides utilities to:
1. Migrate data from legacy to new database format
2. Check database integrity
3. Perform basic database maintenance
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from .database import SessionLocal, ScrapingSession, ScrapedPage, ExtractedData, Base, engine

# Configure logging
logger = logging.getLogger(__name__)

def check_database_integrity():
    """Check the integrity of both databases and report any issues."""
    issues = []
    
    # Check primary database
    try:
        db = SessionLocal()
        # Test simple query
        sessions = db.query(ScrapingSession).count()
        logger.info(f"Primary database has {sessions} sessions")
        db.close()
    except Exception as e:
        issues.append(f"Primary database integrity issue: {str(e)}")
        logger.error(f"Primary database check failed: {e}", exc_info=True)
    
    # Check legacy database
    try:
        if Path("scraping_results.db").exists():
            conn = sqlite3.connect('scraping_results.db')
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            if result[0] != "ok":
                issues.append(f"Legacy database integrity issue: {result[0]}")
            
            # Count sessions
            cursor.execute("SELECT COUNT(*) FROM scraping_sessions")
            count = cursor.fetchone()[0]
            logger.info(f"Legacy database has {count} sessions")
            conn.close()
        else:
            logger.info("Legacy database file not found - no integrity check needed")
    except Exception as e:
        issues.append(f"Legacy database integrity issue: {str(e)}")
        logger.error(f"Legacy database check failed: {e}", exc_info=True)
    
    return issues

def migrate_legacy_to_new(auto_backup=True):
    """Migrate data from legacy database to new format with automatic backup.
    
    Args:
        auto_backup: If True, automatically backup databases before migration
        
    Returns:
        Number of successfully migrated sessions
    """
    if not Path("scraping_results.db").exists():
        logger.info("No legacy database found, nothing to migrate")
        return 0
        
    # Create backup before migration if requested
    if auto_backup:
        logger.info("Creating automatic backup before migration...")
        backup_time = backup_database()
        logger.info(f"Backup completed with timestamp {backup_time}")
    
    try:
        # Open connections to both databases
        legacy_conn = sqlite3.connect('scraping_results.db')
        legacy_cursor = legacy_conn.cursor()
        
        db = SessionLocal()
        
        # Get all sessions from legacy database
        legacy_cursor.execute("SELECT * FROM scraping_sessions")
        legacy_sessions = legacy_cursor.fetchall()
        
        migrated_count = 0
        for session in legacy_sessions:
            try:
                # Check if this session already exists in new DB
                session_id = session[0]
                existing = db.query(ScrapingSession).filter_by(url=session[2]).first()
                
                if existing:
                    logger.info(f"Session {session_id} already exists in new database, skipping")
                    continue
                
                # Create new session record
                timestamp = datetime.fromisoformat(session[1]) if session[1] else datetime.now()
                
                # Extract config data
                try:
                    selectors = json.loads(session[4]) if len(session) > 4 and session[4] else {}
                    pagination_strategy = session[5] if len(session) > 5 else "auto"
                    config = json.dumps({
                        "selectors": selectors,
                        "pagination_strategy": pagination_strategy
                    })
                except:
                    config = "{}"
                
                # Create session in new database
                db_session = ScrapingSession(
                    url=session[2],
                    timestamp=timestamp,
                    status="completed",
                    num_pages=session[3] if len(session) > 3 else 0,
                    config=config
                )
                db.add(db_session)
                db.flush()
                db.refresh(db_session)
                
                # Get all results for this session
                legacy_cursor.execute("SELECT * FROM scraped_data WHERE session_id = ?", (session_id,))
                legacy_results = legacy_cursor.fetchall()
                
                # Migrate each result
                for result in legacy_results:
                    try:
                        # Create new result record
                        page_num = result[2] if len(result) > 2 else 1
                        url = result[3] if len(result) > 3 else session[2]
                        data = result[4] if len(result) > 4 else "{}"
                        
                        db_result = ScrapedPage(
                            session_id=db_session.id,
                            url=url,
                            page_num=page_num,
                            data=data
                        )
                        db.add(db_result)
                    except Exception as e:
                        logger.warning(f"Error migrating result {result[0]}: {e}")
                
                db.commit()
                migrated_count += 1
                logger.info(f"Successfully migrated session {session_id} to new database")
                
            except Exception as e:
                db.rollback()
                logger.error(f"Error migrating session {session[0]}: {e}", exc_info=True)
        
        db.close()
        legacy_conn.close()
        
        logger.info(f"Migration complete. Migrated {migrated_count} of {len(legacy_sessions)} sessions")
        return migrated_count
    
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 0

def backup_database():
    """Create a backup of both databases."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Backup primary database
    try:
        if Path("scraper.db").exists():
            backup_path = f"scraper_{timestamp}.db.bak"
            with open("scraper.db", "rb") as src, open(backup_path, "wb") as dst:
                dst.write(src.read())
            logger.info(f"Primary database backed up to {backup_path}")
    except Exception as e:
        logger.error(f"Failed to backup primary database: {e}")
    
    # Backup legacy database
    try:
        if Path("scraping_results.db").exists():
            backup_path = f"scraping_results_{timestamp}.db.bak"
            with open("scraping_results.db", "rb") as src, open(backup_path, "wb") as dst:
                dst.write(src.read())
            logger.info(f"Legacy database backed up to {backup_path}")
    except Exception as e:
        logger.error(f"Failed to backup legacy database: {e}")
    
    return timestamp

def auto_migrate_on_startup(silent=False):
    """Automatically run database checks and migration on application startup.
    
    Args:
        silent: If True, suppress non-critical log messages
        
    Returns:
        Tuple of (integrity_ok, migration_count)
    """
    try:
        # Check integrity first
        if not silent:
            logger.info("Checking database integrity...")
            
        issues = check_database_integrity()
        integrity_ok = len(issues) == 0
        
        if issues:
            for issue in issues:
                logger.error(issue)
            # Don't proceed with migration if there are integrity issues
            return False, 0
        elif not silent:
            logger.info("Database integrity check passed")
        
        # Check if migration is needed
        if Path("scraping_results.db").exists():
            if not silent:
                logger.info("Legacy database found, starting automatic migration...")
            
            # Run migration with automatic backup
            count = migrate_legacy_to_new(auto_backup=True)
            
            if not silent:
                logger.info(f"Migration completed. Migrated {count} sessions.")
            return integrity_ok, count
        else:
            if not silent:
                logger.info("No legacy database found, no migration needed")
            return integrity_ok, 0
    except Exception as e:
        logger.error(f"Error during automatic migration: {e}", exc_info=True)
        return False, 0

if __name__ == "__main__":
    # Configure logging when run as script
    logging.basicConfig(level=logging.INFO)
    
    # Manual mode
    print("Database Migration Utility")
    print("=========================")
    
    # Create backup
    print("\n1. Creating database backup...")
    backup_time = backup_database()
    
    # Check integrity
    print("\n2. Checking database integrity...")
    issues = check_database_integrity()
    if issues:
        print("\nWARNING: Database integrity issues found:")
        for issue in issues:
            print(f" - {issue}")
    else:
        print("\nDatabase integrity check passed")
    
    # Migrate if needed
    if Path("scraping_results.db").exists():
        choice = input("\n3. Legacy database found. Migrate data to new format? (y/n): ").lower()
        if choice == 'y':
            print("\nStarting migration...")
            count = migrate_legacy_to_new()
            print(f"\nMigration completed. Migrated {count} sessions.")
    else:
        print("\n3. No legacy database found, no migration needed")
