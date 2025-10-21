"""
Logging utilities for the scanner.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL)

def setup_logging(log_dir: str = "logs") -> None:
    """
    Configure logging to both file and console with rotation.
    
    Args:
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(message)s'  # Simpler format for console
    )
    
    # File handler with current date
    file_handler = logging.FileHandler(
        f"{log_dir}/scanner_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
