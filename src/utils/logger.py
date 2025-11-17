#!/usr/bin/env python3
"""
Logger Utility - Centralized logging configuration
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys

class LoggerSetup:
    """Configure centralized logging"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerSetup, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def get_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
        """
        Get or create logger
        
        Args:
            name: Logger name (usually __name__)
            log_file: Optional log file path
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
        Returns:
            Configured logger instance
        """
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path("logs") / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        LoggerSetup._loggers[name] = logger
        return logger


# Convenience function
def get_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Get logger instance"""
    return LoggerSetup.get_logger(name, log_file, level)