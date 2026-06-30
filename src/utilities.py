import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

_logging_configured = False


def setup_logging(level=logging.INFO):
    
    global _logging_configured
    if _logging_configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name):
    
    setup_logging()
    return logging.getLogger(name)
    
