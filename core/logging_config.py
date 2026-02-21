"""
core/logging_config.py

Single source of truth for logging configuration.
cli_app.py and main.py should both call setup_logging() from here
instead of configuring logging themselves.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(debug: bool = False, log_dir: str = "logs") -> Path:
    """
    Configure application logging.

    Args:
        debug:   True  → DEBUG level on console (verbose)
                 False → WARNING level on console (quiet, good for end users)
        log_dir: Directory for log files (created if it doesn't exist)

    Returns:
        Path to the current log file
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    log_file = log_path / f"system_{datetime.now().strftime('%Y%m%d')}.log"

    console_level = logging.DEBUG if debug else logging.WARNING
    console_format = (
        '%(levelname)-8s | %(name)-25s | %(message)s'
        if debug else
        '%(message)s'
    )
    file_format = '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s'

    # File handler — always DEBUG so nothing is lost
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S'))

    # Console handler — level depends on debug flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(console_format))

    # Root logger — capture everything, let handlers filter
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Silence noisy third-party libraries
    for noisy in ('httpx', 'httpcore', 'urllib3', 'requests'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_file


def toggle_debug(enabled: bool, log_file: Path):
    """
    Hot-toggle debug mode without restarting.
    Called by the /debug CLI command.

    Args:
        enabled:  True → switch console to DEBUG, False → back to WARNING
        log_file: Existing log file path (keep the same file handler)
    """
    root = logging.getLogger()
    root.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    console_level = logging.DEBUG if enabled else logging.WARNING
    console_format = (
        '%(levelname)-8s | %(name)-25s | %(message)s'
        if enabled else
        '%(message)s'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(console_format))

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    for noisy in ('httpx', 'httpcore', 'urllib3', 'requests'):
        logging.getLogger(noisy).setLevel(logging.WARNING)