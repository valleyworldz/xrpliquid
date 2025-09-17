"""
Advanced Monitoring System
Structured logging + diagnostics with advanced rate limiting.
"""

import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import numpy as np
from contextlib import contextmanager
import psutil
import sys

logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    component: str
    message: str
    data: Dict[str, Any]
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class MetricEntry:
    """Metric entry for monitoring."""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]
    component: str

@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: str
    threshold: float
    severity: str
    cooldown_seconds: int
    enabled: bool = True

class AdvancedRateLimiter:
    """Advanced rate limiter with multiple algorithms."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.buckets = defaultdict(lambda: {
            'tokens': 0,
            'last_refill': time.time(),
            'requests': deque(),
            'sliding_window': deque()
        })
        self.lock = threading.Lock()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default rate limiting configuration."""
        return {
            'algorithms': ['token_bucket', 'sliding_window', 'fixed_window'],
            'default_limits': {
                'api_calls': {'rate': 100, 'window': 60},  # 100 calls per minute
                'trades': {'rate': 10, 'window': 60},      # 10 trades per minute
                'logs': {'rate': 1000, 'window': 60},      # 1000 logs per minute
                'metrics': {'rate': 500, 'window': 60}     # 500 metrics per minute
            },
            'burst_allowance': 1.5,  # Allow 1.5x burst
            'refill_rate': 0.1       # Refill rate per second
        }
    
    def is_allowed(self, key: str, limit_type: str = 'default') -> bool:
        """Check if request is allowed."""
        
        with self.lock:
            bucket = self.buckets[key]
            current_time = time.time()
            
            # Get limits for this type
            limits = self.config['default_limits'].get(limit_type, 
                                                      self.config['default_limits']['api_calls'])
            
            # Apply all rate limiting algorithms
            token_bucket_allowed = self._check_token_bucket(bucket, limits, current_time)
            sliding_window_allowed = self._check_sliding_window(bucket, limits, current_time)
            fixed_window_allowed = self._check_fixed_window(bucket, limits, current_time)
            
            # All algorithms must allow the request
            return token_bucket_allowed and sliding_window_allowed and fixed_window_allowed
    
    def _check_token_bucket(self, bucket: Dict, limits: Dict, current_time: float) -> bool:
        """Check token bucket algorithm."""
        
        # Refill tokens
        time_passed = current_time - bucket['last_refill']
        tokens_to_add = time_passed * limits['rate'] / limits['window']
        bucket['tokens'] = min(limits['rate'] * self.config['burst_allowance'], 
                              bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = current_time
        
        # Check if request is allowed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        
        return False
    
    def _check_sliding_window(self, bucket: Dict, limits: Dict, current_time: float) -> bool:
        """Check sliding window algorithm."""
        
        window_start = current_time - limits['window']
        
        # Remove old requests
        while bucket['sliding_window'] and bucket['sliding_window'][0] < window_start:
            bucket['sliding_window'].popleft()
        
        # Check if request is allowed
        if len(bucket['sliding_window']) < limits['rate']:
            bucket['sliding_window'].append(current_time)
            return True
        
        return False
    
    def _check_fixed_window(self, bucket: Dict, limits: Dict, current_time: float) -> bool:
        """Check fixed window algorithm."""
        
        window_start = current_time - (current_time % limits['window'])
        
        # Reset window if needed
        if bucket.get('window_start', 0) != window_start:
            bucket['window_start'] = window_start
            bucket['requests'] = deque()
        
        # Check if request is allowed
        if len(bucket['requests']) < limits['rate']:
            bucket['requests'].append(current_time)
            return True
        
        return False
    
    def get_rate_limit_status(self, key: str) -> Dict[str, Any]:
        """Get current rate limit status."""
        
        with self.lock:
            bucket = self.buckets[key]
            current_time = time.time()
            
            status = {
                'token_bucket_tokens': bucket['tokens'],
                'sliding_window_requests': len(bucket['sliding_window']),
                'fixed_window_requests': len(bucket['requests']),
                'last_refill': bucket['last_refill']
            }
            
            return status

class StructuredLogger:
    """Structured logger with advanced features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.rate_limiter = AdvancedRateLimiter()
        self.log_buffer = deque(maxlen=self.config['buffer_size'])
        self.metrics_buffer = deque(maxlen=self.config['metrics_buffer_size'])
        self.alert_rules = {}
        self.alert_history = defaultdict(list)
        self.lock = threading.Lock()
        
        # Setup logging
        self._setup_logging()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            'log_level': 'INFO',
            'log_format': 'json',
            'log_file': 'logs/structured.log',
            'metrics_file': 'logs/metrics.log',
            'buffer_size': 1000,
            'metrics_buffer_size': 500,
            'flush_interval': 5,  # seconds
            'enable_console': True,
            'enable_file': True,
            'enable_metrics': True,
            'compression': True,
            'retention_days': 30
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        
        # Create logs directory
        Path(self.config['log_file']).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['log_level']))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.config['enable_console']:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config['enable_file']:
            file_handler = logging.FileHandler(self.config['log_file'])
            file_handler.setLevel(getattr(logging, self.config['log_level']))
            
            if self.config['log_format'] == 'json':
                file_formatter = logging.Formatter('%(message)s')
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
    
    def log(self, level: str, component: str, message: str, 
           data: Dict[str, Any] = None, trace_id: str = None,
           user_id: str = None, session_id: str = None):
        """Log structured message."""
        
        # Rate limiting
        if not self.rate_limiter.is_allowed(f"{component}:{level}", 'logs'):
            return
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            data=data or {},
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id
        )
        
        # Add to buffer
        with self.lock:
            self.log_buffer.append(log_entry)
        
        # Log to standard logger
        log_message = self._format_log_message(log_entry)
        getattr(logger, level.lower(), logger.info)(log_message)
        
        # Check alert rules
        self._check_alert_rules(log_entry)
    
    def _format_log_message(self, log_entry: LogEntry) -> str:
        """Format log message."""
        
        if self.config['log_format'] == 'json':
            return json.dumps(asdict(log_entry), default=str)
        else:
            return f"[{log_entry.component}] {log_entry.message} | Data: {log_entry.data}"
    
    def metric(self, metric_name: str, value: float, tags: Dict[str, str] = None,
              component: str = 'system'):
        """Log metric."""
        
        # Rate limiting
        if not self.rate_limiter.is_allowed(f"{component}:metrics", 'metrics'):
            return
        
        # Create metric entry
        metric_entry = MetricEntry(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            tags=tags or {},
            component=component
        )
        
        # Add to buffer
        with self.lock:
            self.metrics_buffer.append(metric_entry)
        
        # Log metric
        if self.config['enable_metrics']:
            metric_message = json.dumps(asdict(metric_entry), default=str)
            logger.info(f"METRIC: {metric_message}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.alert_rules[rule.name] = rule
    
    def _check_alert_rules(self, log_entry: LogEntry):
        """Check alert rules against log entry."""
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_alert = self.alert_history[rule_name]
            if last_alert and (datetime.now() - last_alert[-1]).total_seconds() < rule.cooldown_seconds:
                continue
            
            # Check condition (simplified)
            if self._evaluate_condition(rule.condition, log_entry):
                self._trigger_alert(rule, log_entry)
    
    def _evaluate_condition(self, condition: str, log_entry: LogEntry) -> bool:
        """Evaluate alert condition."""
        
        # Simplified condition evaluation
        try:
            # Replace placeholders with actual values
            condition = condition.replace('{level}', log_entry.level)
            condition = condition.replace('{component}', log_entry.component)
            condition = condition.replace('{message}', log_entry.message)
            
            # Evaluate condition
            return eval(condition)
        except:
            return False
    
    def _trigger_alert(self, rule: AlertRule, log_entry: LogEntry):
        """Trigger alert."""
        
        alert_message = f"ALERT [{rule.severity}]: {rule.name} - {log_entry.message}"
        logger.warning(alert_message)
        
        # Store alert history
        self.alert_history[rule.name].append(datetime.now())
        
        # Keep only recent alerts
        if len(self.alert_history[rule.name]) > 100:
            self.alert_history[rule.name] = self.alert_history[rule.name][-100:]
    
    def flush_buffers(self):
        """Flush log and metric buffers."""
        
        with self.lock:
            # Flush logs
            if self.log_buffer:
                log_file = Path(self.config['log_file'])
                with open(log_file, 'a') as f:
                    for log_entry in self.log_buffer:
                        f.write(json.dumps(asdict(log_entry), default=str) + '\n')
                self.log_buffer.clear()
            
            # Flush metrics
            if self.metrics_buffer:
                metrics_file = Path(self.config['metrics_file'])
                with open(metrics_file, 'a') as f:
                    for metric_entry in self.metrics_buffer:
                        f.write(json.dumps(asdict(metric_entry), default=str) + '\n')
                self.metrics_buffer.clear()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        
        with self.lock:
            return {
                'log_buffer_size': len(self.log_buffer),
                'metrics_buffer_size': len(self.metrics_buffer),
                'alert_rules_count': len(self.alert_rules),
                'rate_limit_status': self.rate_limiter.get_rate_limit_status('system:logs')
            }

class SystemDiagnostics:
    """System diagnostics and health monitoring."""
    
    def __init__(self, structured_logger: StructuredLogger):
        self.logger = structured_logger
        self.start_time = datetime.now()
        self.metrics_history = defaultdict(list)
        
    def collect_system_metrics(self):
        """Collect system metrics."""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free = disk.free / (1024**3)  # GB
        
        # Network metrics
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024**2)  # MB
        process_cpu = process.cpu_percent()
        
        # Log metrics
        self.logger.metric('system.cpu.percent', cpu_percent, {'component': 'system'})
        self.logger.metric('system.memory.percent', memory_percent, {'component': 'system'})
        self.logger.metric('system.memory.available_gb', memory_available, {'component': 'system'})
        self.logger.metric('system.disk.percent', disk_percent, {'component': 'system'})
        self.logger.metric('system.disk.free_gb', disk_free, {'component': 'system'})
        self.logger.metric('system.network.bytes_sent', network_bytes_sent, {'component': 'system'})
        self.logger.metric('system.network.bytes_recv', network_bytes_recv, {'component': 'system'})
        self.logger.metric('system.process.memory_mb', process_memory, {'component': 'system'})
        self.logger.metric('system.process.cpu_percent', process_cpu, {'component': 'system'})
        
        # Store metrics for trend analysis
        current_time = datetime.now()
        self.metrics_history['cpu_percent'].append((current_time, cpu_percent))
        self.metrics_history['memory_percent'].append((current_time, memory_percent))
        self.metrics_history['disk_percent'].append((current_time, disk_percent))
        
        # Keep only recent metrics
        cutoff_time = current_time - timedelta(hours=1)
        for metric_name in self.metrics_history:
            self.metrics_history[metric_name] = [
                (t, v) for t, v in self.metrics_history[metric_name] 
                if t > cutoff_time
            ]
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system health."""
        
        health_status = {
            'overall_health': 'healthy',
            'issues': [],
            'warnings': [],
            'timestamp': datetime.now()
        }
        
        # Get current metrics
        current_metrics = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                current_metrics[metric_name] = history[-1][1]
        
        # Check CPU
        if current_metrics.get('cpu_percent', 0) > 90:
            health_status['issues'].append('High CPU usage')
            health_status['overall_health'] = 'critical'
        elif current_metrics.get('cpu_percent', 0) > 80:
            health_status['warnings'].append('Elevated CPU usage')
            if health_status['overall_health'] == 'healthy':
                health_status['overall_health'] = 'warning'
        
        # Check Memory
        if current_metrics.get('memory_percent', 0) > 95:
            health_status['issues'].append('Critical memory usage')
            health_status['overall_health'] = 'critical'
        elif current_metrics.get('memory_percent', 0) > 85:
            health_status['warnings'].append('High memory usage')
            if health_status['overall_health'] == 'healthy':
                health_status['overall_health'] = 'warning'
        
        # Check Disk
        if current_metrics.get('disk_percent', 0) > 95:
            health_status['issues'].append('Critical disk usage')
            health_status['overall_health'] = 'critical'
        elif current_metrics.get('disk_percent', 0) > 85:
            health_status['warnings'].append('High disk usage')
            if health_status['overall_health'] == 'healthy':
                health_status['overall_health'] = 'warning'
        
        # Log health status
        self.logger.log('INFO', 'system', f"System health: {health_status['overall_health']}", 
                       {'health_status': health_status})
        
        return health_status
    
    def get_uptime(self) -> timedelta:
        """Get system uptime."""
        return datetime.now() - self.start_time

class AdvancedMonitoringSystem:
    """Main advanced monitoring system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.structured_logger = StructuredLogger(self.config.get('logging', {}))
        self.diagnostics = SystemDiagnostics(self.structured_logger)
        self.monitoring_thread = None
        self.running = False
        
        # Setup alert rules
        self._setup_alert_rules()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'logging': {
                'log_level': 'INFO',
                'log_format': 'json',
                'log_file': 'logs/structured.log',
                'metrics_file': 'logs/metrics.log',
                'buffer_size': 1000,
                'metrics_buffer_size': 500,
                'flush_interval': 5,
                'enable_console': True,
                'enable_file': True,
                'enable_metrics': True
            },
            'monitoring': {
                'collect_interval': 10,  # seconds
                'health_check_interval': 30,  # seconds
                'enable_system_metrics': True,
                'enable_health_checks': True
            }
        }
    
    def _setup_alert_rules(self):
        """Setup default alert rules."""
        
        # High error rate alert
        error_alert = AlertRule(
            name='high_error_rate',
            condition="'{level}' == 'ERROR'",
            threshold=5,
            severity='high',
            cooldown_seconds=300
        )
        self.structured_logger.add_alert_rule(error_alert)
        
        # Critical system alert
        critical_alert = AlertRule(
            name='critical_system_issue',
            condition="'{level}' == 'CRITICAL'",
            threshold=1,
            severity='critical',
            cooldown_seconds=60
        )
        self.structured_logger.add_alert_rule(critical_alert)
    
    def start_monitoring(self):
        """Start monitoring system."""
        
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.structured_logger.log('INFO', 'monitoring', 'Advanced monitoring system started')
    
    def stop_monitoring(self):
        """Stop monitoring system."""
        
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.structured_logger.log('INFO', 'monitoring', 'Advanced monitoring system stopped')
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        
        last_health_check = datetime.now()
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Collect system metrics
                if self.config['monitoring']['enable_system_metrics']:
                    self.diagnostics.collect_system_metrics()
                
                # Health check
                if (self.config['monitoring']['enable_health_checks'] and 
                    (current_time - last_health_check).total_seconds() >= 
                    self.config['monitoring']['health_check_interval']):
                    
                    health_status = self.diagnostics.check_system_health()
                    last_health_check = current_time
                
                # Flush buffers
                self.structured_logger.flush_buffers()
                
                # Sleep
                time.sleep(self.config['monitoring']['collect_interval'])
                
            except Exception as e:
                self.structured_logger.log('ERROR', 'monitoring', f'Monitoring loop error: {e}')
                time.sleep(5)
    
    @contextmanager
    def trace(self, component: str, operation: str, trace_id: str = None):
        """Context manager for tracing operations."""
        
        if not trace_id:
            trace_id = f"{component}_{operation}_{int(time.time())}"
        
        start_time = time.time()
        
        try:
            self.structured_logger.log('INFO', component, f"Starting {operation}", 
                                     {'trace_id': trace_id})
            yield trace_id
        except Exception as e:
            self.structured_logger.log('ERROR', component, f"Error in {operation}: {e}", 
                                     {'trace_id': trace_id, 'error': str(e)})
            raise
        finally:
            duration = time.time() - start_time
            self.structured_logger.log('INFO', component, f"Completed {operation}", 
                                     {'trace_id': trace_id, 'duration_seconds': duration})
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        
        return {
            'running': self.running,
            'uptime': str(self.diagnostics.get_uptime()),
            'log_stats': self.structured_logger.get_log_stats(),
            'system_health': self.diagnostics.check_system_health()
        }

def main():
    """Demonstrate advanced monitoring system."""
    
    # Initialize monitoring system
    monitoring = AdvancedMonitoringSystem()
    
    print("🧪 Testing Advanced Monitoring System")
    print("=" * 50)
    
    # Start monitoring
    monitoring.start_monitoring()
    
    # Test structured logging
    monitoring.structured_logger.log('INFO', 'test', 'Test info message', 
                                   {'test_data': 'value1', 'number': 42})
    monitoring.structured_logger.log('WARNING', 'test', 'Test warning message', 
                                   {'warning_data': 'value2'})
    monitoring.structured_logger.log('ERROR', 'test', 'Test error message', 
                                   {'error_data': 'value3'})
    
    # Test metrics
    monitoring.structured_logger.metric('test.metric1', 123.45, {'tag1': 'value1'})
    monitoring.structured_logger.metric('test.metric2', 67.89, {'tag2': 'value2'})
    
    # Test tracing
    with monitoring.trace('test', 'test_operation') as trace_id:
        time.sleep(0.1)
        monitoring.structured_logger.log('INFO', 'test', 'Inside trace', {'trace_id': trace_id})
    
    # Wait for monitoring to collect some data
    time.sleep(2)
    
    # Get status
    status = monitoring.get_monitoring_status()
    
    print(f"Monitoring Status:")
    print(f"  Running: {status['running']}")
    print(f"  Uptime: {status['uptime']}")
    print(f"  System Health: {status['system_health']['overall_health']}")
    
    if status['system_health']['warnings']:
        print(f"  Warnings: {status['system_health']['warnings']}")
    
    if status['system_health']['issues']:
        print(f"  Issues: {status['system_health']['issues']}")
    
    print(f"  Log Stats: {status['log_stats']}")
    
    # Stop monitoring
    monitoring.stop_monitoring()
    
    print("\n✅ Advanced monitoring system test completed")
    
    return 0

if __name__ == "__main__":
    exit(main())
