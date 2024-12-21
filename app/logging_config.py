import logging
import datetime


def setup_logging():
    logger = logging.getLogger("usaspending")
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f"logs/bdt_usas_{timestamp}.log"

    # Create file handler
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
