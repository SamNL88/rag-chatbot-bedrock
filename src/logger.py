"""
Central logging configuration for the application.

This module defines a setup function that configures Loguru with
a consistent format, log level, and output destination.
"""


import sys
from loguru import logger


def setup_logger():
    """
    Configure and return a Loguru logger instance.

    Removes the default Loguru handler and adds a clean stdout logger
    with timestamp, log level, and colored output.
    """
    # remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    return logger
