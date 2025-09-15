#!/usr/bin/env python3
"""
Logging Configuration for XRP Trading Bot
"""

import logging
import sys
from typing import Optional
from datetime import datetime
import os

class DedupFilter(logging.Filter):
    """Filter to prevent duplicate log messages"""
    
    def __init__(self):
        super().__init__()
        self.seen_messages = set()
        self.max_duplicates = 5
        self.duplicate_counts = {}
    
    def filter(self, record):
        # Create a key for the message
        message_key = f"{record.levelname}:{record.getMessage()}"
        
        if message_key in self.seen_messages:
            # Increment duplicate count
            self.duplicate_counts[message_key] = self.duplicate_counts.get(message_key, 0) + 1
            
            # Only log duplicates up to max_duplicates
            if self.duplicate_counts[message_key] <= self.max_duplicates:
                record.msg = f"[DUPLICATE {self.duplicate_counts[message_key]}] {record.msg}"
                return True
            else:
                return False
        else:
            self.seen_messages.add(message_key)
            return True

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_colors: bool = True,
    enable_dedup: bool = True
) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_colors: Enable colored output
        enable_dedup: Enable duplicate message filtering
    
    Returns:
        Configured logger
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("XRP_BOT")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if enable_colors:
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(console_formatter)
    
    if enable_dedup:
        console_handler.addFilter(DedupFilter())
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str = "XRP_BOT") -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

def log_trade(logger: logging.Logger, trade_type: str, details: dict) -> None:
    """Log trade information"""
    logger.info(f"💰 {trade_type.upper()} TRADE: {details}")

def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """Log error with context"""
    logger.error(f"❌ {context}: {error}", exc_info=True)

def log_performance(logger: logging.Logger, metrics: dict) -> None:
    """Log performance metrics"""
    logger.info(f"📊 PERFORMANCE: {metrics}")

def log_signal(logger: logging.Logger, signal_type: str, confidence: float, price: float) -> None:
    """Log trading signal"""
    logger.info(f"📈 {signal_type.upper()} SIGNAL: Confidence={confidence:.2f}, Price=${price:.4f}")

def log_position(logger: logging.Logger, position: dict) -> None:
    """Log position information"""
    size = position.get('size', 0)
    entry_price = position.get('entry_price', 0)
    pnl = position.get('unrealized_pnl', 0)
    logger.info(f"📊 POSITION: Size={size} XRP, Entry=${entry_price:.4f}, PnL=${pnl:.2f}")

def log_api_call(logger: logging.Logger, endpoint: str, status: str, duration: float) -> None:
    """Log API call information"""
    logger.debug(f"🌐 API: {endpoint} - {status} ({duration:.3f}s)")

def log_risk_check(logger: logging.Logger, check_type: str, result: bool, details: str = "") -> None:
    """Log risk check results"""
    status = "✅ PASSED" if result else "❌ FAILED"
    logger.info(f"🛡️ RISK CHECK ({check_type}): {status} {details}")

# Legacy compatibility
def setup_bot_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Legacy function for backward compatibility"""
    return setup_logging(level, log_file) 