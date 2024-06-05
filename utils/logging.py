import logging

def configuracionDeLoggin(log_file):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    
    # Check if the logger has handlers to avoid duplicate logs
    if not logger.handlers:
        # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger.setLevel(logging.DEBUG)
        
        # Create handlers
        c_handler = logging.StreamHandler()  # Console handler
        f_handler = logging.FileHandler(log_file)  # File handler
        
        # Set the log level for handlers
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)
        
        # Create formatters and add them to handlers
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger