import logging
import sys

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application-wide logging.
    Sets root logger level and silences verbose third-party libraries.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    # Silence chatty third-party libraries
    silenced_loggers = [
        "urllib3",
        "botocore",
        "boto3",
        "httpcore",
        "httpx",
        "asyncio",
        "redis",
        "minio",
        "ppocr",
        "paddle",
        "pdfminer",
        "matplotlib",
        "PIL",
        "uvicorn.access",
    ]
    for logger_name in silenced_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
