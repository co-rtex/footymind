"""
Logging utilities for FootyMind.

Provides a simple, consistent logger configuration so that every module can log
to stdout with a formatted timestamp and log level.
"""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger with the given name.

    If the root logger has no handlers configured yet, this function also
    configures a basic StreamHandler.

    Parameters
    ----------
    name : str | None
        Logger name. If None, the root logger is returned.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger_name = name if name is not None else "footymind"
    logger = logging.getLogger(logger_name)

    if not logging.getLogger().handlers:
        # Configure root logger once
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

    return logger
