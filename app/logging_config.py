import logging


def setup_logging():
    logger = logging.getLogger("usaspending")
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler("bdt_usaspending_app.log")
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
