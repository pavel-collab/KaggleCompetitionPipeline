"""
Logging configuration for Kaggle Pipeline.

Two handlers:
- Console: for Docker logs (stdout)
- File: for persistent logs (mounted volume)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Log directory - will be mounted from Docker
LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")
LOG_FILE = os.path.join(LOG_DIR, "kaggle_pipeline.log")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


def setup_logging() -> logging.Logger:
    """
    Setup logging with console and file handlers.

    Returns:
        Root logger configured for the application.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create logger
    logger = logging.getLogger("kaggle_pipeline")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Format: timestamp - level - module - message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (for Docker logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating, max 10MB, keep 5 backups)
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger with the specified name.

    Args:
        name: Logger name (usually module name like 'telegram', 'worker')

    Returns:
        Logger instance for the specified module.
    """
    # Ensure root logger is configured
    setup_logging()
    return logging.getLogger(f"kaggle_pipeline.{name}")
