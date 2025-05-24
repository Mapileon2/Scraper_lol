"""Logging configuration and utilities for the Web Scraper application."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import traceback
from datetime import datetime

from ..config import LOGGING_CONFIG, BASE_DIR

class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'location': f"{record.pathname}:{record.lineno}",
            'function': record.funcName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add stack trace for errors
        if record.levelno >= logging.ERROR:
            log_entry['stack_trace'] = traceback.format_stack()
            
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text',
                         'filename', 'funcName', 'id', 'levelname', 'levelno',
                         'lineno', 'module', 'msecs', 'message', 'msg',
                         'name', 'pathname', 'process', 'processName',
                         'relativeCreated', 'stack_info', 'thread', 'threadName'):
                if key not in log_entry:  # Don't override existing fields
                    log_entry[key] = value
                    
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logger(
    name: str = 'web_scraper',
    log_level: Optional[Union[str, int]] = None,
    log_file: Optional[Union[str, Path]] = None,
    max_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True
) -> logging.Logger:
    """Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses LOGGING_CONFIG.LOG_FILE
        max_size: Maximum log file size in bytes
        backup_count: Number of backup logs to keep
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    if log_level is None:
        log_level = LOGGING_CONFIG.LOG_LEVEL
    if log_file is None:
        log_file = LOGGING_CONFIG.LOG_FILE
    
    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times in case of module reload
    if logger.handlers:
        return logger
    
    # Create formatter
    json_formatter = JSONFormatter()
    
    # Create file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    
    # Create console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Use a simpler format for console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Configure root logger to prevent double logging
    logging.basicConfig(handlers=[], level=logging.WARNING)
    
    return logger

def log_execution_time(logger: logging.Logger):
    """Decorator to log the execution time of a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            
            logger.debug(
                f"Function {func.__name__} executed in {end_time - start_time}",
                extra={
                    'function': func.__name__,
                    'execution_time_ms': (end_time - start_time).total_seconds() * 1000,
                    'module': func.__module__
                }
            )
            return result
        return wrapper
    return decorator

# Create default logger instance
logger = setup_logger()

def log_error(
    error: Exception,
    message: str = "An error occurred",
    extra: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log an error with exception information.
    
    Args:
        error: The exception that was raised
        message: Custom error message
        extra: Additional context to include in the log
        logger: Logger instance to use. If None, uses the default logger.
    """
    if logger is None:
        logger = globals().get('logger', logging.getLogger('web_scraper'))
    
    exc_info = (type(error), error, error.__traceback__)
    extra = extra or {}
    extra.update({
        'exception_type': type(error).__name__,
        'exception_args': list(error.args) if error.args else None
    })
    
    logger.error(
        f"{message}: {str(error)}",
        exc_info=exc_info,
        extra=extra,
        stack_info=True
    )

# Configure root logger to use our handler
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)
root_logger.handlers = []  # Remove any existing handlers
root_logger.addHandler(logging.NullHandler())  # Prevent "No handlers could be found" warnings
