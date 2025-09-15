import logging
import sys
try:
    from pythonjsonlogger import jsonlogger
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = jsonlogger.JsonFormatter()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
except ImportError:
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name) 