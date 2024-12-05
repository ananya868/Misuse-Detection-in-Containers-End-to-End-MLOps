# logging_config.py
from loguru import logger

# Centralized logger configuration
logger.add("ml_workflow.log", rotation="2 MB", retention="10 days", level="INFO")

# Export the logger instance
__all__ = ["logger"]
