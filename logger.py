import logging
import sys

def setup_logger(name: str = "CancerCombo", log_file: str = "cancercombo.log") -> logging.Logger:
    """Setup console and file logging configurations.

    Args:
        name: Logger module namespace name.
        log_file: Target log filename.

    Returns:
        logging.Logger: Instantiated logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
