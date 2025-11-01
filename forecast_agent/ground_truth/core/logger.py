"""Centralized logging configuration for Forecast Agent.

Provides consistent logging across all modules with configurable levels and formatting.

Usage:
    from ground_truth.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Starting forecast generation")
    logger.warning("Missing data detected")
    logger.error("Forecast failed", exc_info=True)
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Get or create a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               Defaults to INFO, or reads from LOG_LEVEL env var
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance

    Example:
        >>> from ground_truth.core.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Forecast completed successfully")
        2025-10-31 14:23:45 - ground_truth.models.arima - INFO - Forecast completed successfully

    Environment Variables:
        LOG_LEVEL: Set default log level (DEBUG, INFO, WARNING, ERROR)
        LOG_FORMAT: Set custom log format
    """
    # Get logger
    logger = logging.getLogger(name)

    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        # Determine log level
        if level is None:
            import os
            level = os.environ.get('LOG_LEVEL', 'INFO').upper()

        logger.setLevel(getattr(logging, level))

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level))

        # Create formatter
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger


def set_log_level(level: str):
    """Set log level for all forecast_agent loggers.

    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

    Example:
        >>> from ground_truth.core.logger import set_log_level
        >>> set_log_level('DEBUG')  # Enable debug logging for all modules
    """
    level_value = getattr(logging, level.upper())

    # Update all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('ground_truth'):
            logger = logging.getLogger(logger_name)
            logger.setLevel(level_value)
            for handler in logger.handlers:
                handler.setLevel(level_value)


# Convenience functions for quick logging without creating logger
_default_logger = get_logger('forecast_agent')

def debug(msg: str, *args, **kwargs):
    """Log debug message."""
    _default_logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    """Log info message."""
    _default_logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    """Log warning message."""
    _default_logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    """Log error message."""
    _default_logger.error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    """Log critical message."""
    _default_logger.critical(msg, *args, **kwargs)
