"""
Diagnostics helper module for XRP Trading Bot
Provides structured logging, correlation IDs, timing, and monitoring capabilities
"""

import asyncio
import functools
import logging
import os
import psutil
import time
import traceback
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass

# Global run ID and counters
RUN_ID = os.getenv("RUN_ID", str(uuid.uuid4())[:8])
TRADE_COUNTER = 0
SIGNAL_COUNTER = 0

# Structured logging setup
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

# Base logger
logger = structlog.get_logger() if STRUCTLOG_AVAILABLE else logging.getLogger(__name__)

def setup_logging(verbose: bool = False, json_fmt: bool = False) -> None:
    """Setup logging with optional JSON format"""
    level = logging.DEBUG if verbose else logging.INFO
    
    if not logging.root.handlers:
        if json_fmt and STRUCTLOG_AVAILABLE:
            # JSON format already configured in structlog
            logging.basicConfig(level=level)
        else:
            # Standard format
            format_str = "%(asctime)s %(levelname)s:%(name)s: %(message)s"
            logging.basicConfig(level=level, format=format_str)
    
    # Reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

def get_correlation_logger(correlation_id: str = None, **kwargs) -> Any:
    """Get logger with correlation ID binding"""
    if STRUCTLOG_AVAILABLE:
        log = structlog.get_logger()
        if correlation_id:
            log = log.bind(correlation_id=correlation_id, run_id=RUN_ID, **kwargs)
        else:
            log = log.bind(run_id=RUN_ID, **kwargs)
        return log
    else:
        log = logging.getLogger(__name__)
        if correlation_id:
            extra = {'correlation_id': correlation_id, 'run_id': RUN_ID, **kwargs}
            log = logging.LoggerAdapter(log, extra)
        return log

def new_trade_logger(signal_id: str = None) -> Any:
    """Create logger for trade correlation"""
    global TRADE_COUNTER
    TRADE_COUNTER += 1
    trade_id = f"trade_{TRADE_COUNTER:06d}"
    return get_correlation_logger(trade_id, signal_id=signal_id)

def new_signal_logger() -> Any:
    """Create logger for signal correlation"""
    global SIGNAL_COUNTER
    SIGNAL_COUNTER += 1
    signal_id = f"sig_{SIGNAL_COUNTER:06d}"
    return get_correlation_logger(signal_id)

def log_timing(func: Callable) -> Callable:
    """Decorator to log function timing"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug("timing", 
                        fn=func.__name__, 
                        ms=round(elapsed_ms, 2),
                        status="success")
            return result
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error("timing", 
                        fn=func.__name__, 
                        ms=round(elapsed_ms, 2),
                        status="error",
                        error=str(e))
            raise
    return wrapper

def debug_signal(name: str, **metrics) -> None:
    """Log signal metrics for debugging"""
    logger.debug("signal.metrics", 
                signal_name=name,
                **metrics)

def log_risk_check(check_name: str, ok: bool, **details) -> None:
    """Log risk check results"""
    logger.info("risk.check", 
                check=check_name,
                ok=ok,
                **details)

def log_tpsl(action: str, leg_id: str, **details) -> None:
    """Log TP/SL lifecycle events"""
    logger.info("tpsl", 
                action=action,
                leg_id=leg_id,
                **details)

@contextmanager
def log_operation(operation: str, **context):
    """Context manager for logging operations with timing"""
    start_time = time.time()
    logger.info("operation.start", operation=operation, **context)
    try:
        yield
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("operation.complete", 
                    operation=operation,
                    duration_ms=round(elapsed_ms, 2),
                    **context)
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error("operation.error", 
                    operation=operation,
                    duration_ms=round(elapsed_ms, 2),
                    error=str(e),
                    **context)
        raise

async def monitor_resources() -> None:
    """Monitor system resources and loop performance"""
    process = psutil.Process()
    
    while True:
        try:
            # CPU and memory
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Loop lag measurement
            loop = asyncio.get_event_loop()
            start_time = time.time()
            await asyncio.sleep(0.001)  # Minimal sleep
            loop_lag_ms = (time.time() - start_time) * 1000
            
            logger.info("sys.resource", 
                        cpu=cpu_percent,
                        mem_mb=round(memory_mb, 1),
                        loop_lag_ms=round(loop_lag_ms, 1))
            
            await asyncio.sleep(15)  # Log every 15 seconds
            
        except Exception as e:
            logger.error("sys.resource.error", error=str(e))
            await asyncio.sleep(15)

def install_exception_handlers() -> None:
    """Install global exception handlers"""
    
    def handle_exception(loop, context):
        exception = context.get('exception')
        if exception:
            logger.error("asyncio.unhandled",
                        msg=str(exception),
                        type=type(exception).__name__,
                        traceback=traceback.format_exc())
        else:
            logger.error("asyncio.error",
                        msg=context.get('message', 'Unknown error'))
    
    def handle_thread_exception(args):
        logger.error("thread.unhandled",
                    exc_type=args.exc_type.__name__,
                    exc_value=str(args.exc_value),
                    exc_traceback=args.exc_traceback)
    
    # Install asyncio exception handler
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    
    # Install thread exception handler
    import threading
    threading.excepthook = handle_thread_exception

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    signal_generation_ms: float = 0.0
    order_execution_ms: float = 0.0
    risk_check_ms: float = 0.0
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    
    def log_summary(self) -> None:
        """Log performance summary"""
        win_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        logger.info("performance.summary",
                    total_trades=self.total_trades,
                    successful_trades=self.successful_trades,
                    failed_trades=self.failed_trades,
                    win_rate=round(win_rate, 2),
                    avg_signal_ms=round(self.signal_generation_ms, 2),
                    avg_order_ms=round(self.order_execution_ms, 2),
                    avg_risk_ms=round(self.risk_check_ms, 2))

# Global performance tracker
performance = PerformanceMetrics()

def log_performance_metric(metric: str, value: float) -> None:
    """Log a performance metric"""
    logger.debug("performance.metric",
                metric=metric,
                value=value)

def log_config_dump(config: Dict[str, Any]) -> None:
    """Log configuration with sensitive data redacted"""
    # Redact sensitive fields
    safe_config = config.copy()
    sensitive_keys = ['api_key', 'secret', 'password', 'token']
    
    for key in sensitive_keys:
        if key in safe_config:
            safe_config[key] = '***REDACTED***'
    
    logger.info("config.dump",
                run_id=RUN_ID,
                config=safe_config)

def log_market_data_update(symbol: str, price: float, volume: float, timestamp: float) -> None:
    """Log market data updates"""
    logger.debug("market.data",
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=timestamp,
                latency_ms=round((time.time() - timestamp) * 1000, 2))

def log_order_event(event: str, order_id: str, **details) -> None:
    """Log order lifecycle events"""
    logger.info("order.event",
                event=event,
                order_id=order_id,
                **details)

def log_position_update(symbol: str, size: float, side: str, **details) -> None:
    """Log position updates"""
    logger.info("position.update",
                symbol=symbol,
                size=size,
                side=side,
                **details)

def log_pnl_update(realized_pnl: float, unrealized_pnl: float, total_pnl: float) -> None:
    """Log P&L updates"""
    logger.info("pnl.update",
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl)

# Initialize logging on import
setup_logging() 