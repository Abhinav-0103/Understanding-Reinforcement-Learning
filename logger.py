import logging
import os
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger that writes to a timestamped log file inside the logs directory.
    
    Args:
        name (str): Name of the logger (usually __name__).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create main logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create a subdirectory with current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_log_dir = os.path.join(log_dir, timestamp)
    os.makedirs(session_log_dir, exist_ok=True)

    # Define log file path
    log_file = os.path.join(session_log_dir, f"{name}.log")

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if logger already configured
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger