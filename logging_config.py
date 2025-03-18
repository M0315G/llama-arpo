import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {  # More detailed logs for file
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "simple": {  # Less detailed logs for console
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {  # Console logs only higher-level messages
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO",  # Only show INFO and above in console
        },
        "file": {  # File logs all details
            "class": "logging.FileHandler",
            "filename": "app.log",
            "mode": "a",
            "formatter": "detailed",
            "level": "DEBUG",  # Log everything in the file
        }
    },
    "root": {
        "handlers": ["console", "file"],  # Both handlers are attached
        "level": "DEBUG",  # Root logger collects all messages
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
