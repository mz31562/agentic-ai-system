# core/logging_config.py
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    console_level: str = "WARNING",
    file_level: str = "INFO",
    log_dir: str = "logs"
):
    """
    Configure logging with clean console output and detailed file logging
    
    Args:
        console_level: Level for console output (WARNING = quiet, INFO = verbose)
        file_level: Level for file output (always detailed)
        log_dir: Directory for log files
    """
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create formatters
    # Simple format for console (only message)
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    
    # Detailed format for file
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (clean output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(console_formatter)
    
    # File handler (detailed output)
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"system_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Silence noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return log_file


def setup_user_logging():
    """
    Setup logging optimized for end users (minimal console noise)
    """
    return setup_logging(
        console_level="ERROR",  # Only show errors in console
        file_level="INFO"       # Everything else goes to file
    )


def setup_debug_logging():
    """
    Setup logging for debugging (verbose console output)
    """
    return setup_logging(
        console_level="DEBUG",  # Show everything in console
        file_level="DEBUG"      # Show everything in file
    )