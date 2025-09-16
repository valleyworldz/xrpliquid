"""
⚡ LATENCY PROFILER
===================
Production-grade latency profiling and observability system.

Features:
- Comprehensive latency tracking
- Performance metrics collection
- Real-time monitoring
- Prometheus-style metrics
- Performance optimization insights
"""

import asyncio
import time
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger

@dataclass
class LatencyMetrics:
    """Latency metrics data structure"""
    
    # Timing metrics
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    p99_9_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    
    # Count metrics
    total_measurements: int = 0
    successful_measurements: int = 0
    failed_measurements: int = 0
    
    # Time window
    window_start: float = 0.0
    window_end: float = 0.0
    window_duration_seconds: float = 0.0

@dataclass
class PerformanceEvent:
    """Performance event data structure"""
    
    timestamp: float
    event_type: str  # 'order_placement', 'order_fill', 'data_update', 'strategy_signal'
    duration_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class LatencyProfiler:
    """
    ⚡ LATENCY PROFILER
    
    Production-grade latency profiling and observability system
    with comprehensive performance metrics collection.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or Logger()
        
        # Profiling state
        self.measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.events: List[PerformanceEvent] = []
        self.start_time = time.time()
        
        # Performance tracking
        self.total_measurements = 0
        self.successful_measurements = 0
        self.failed_measurements = 0
        
        # Real-time metrics
        self.current_metrics: Dict[str, LatencyMetrics] = {}
        self.last_update_time = time.time()
        
        # Configuration
        self.profiling_config = {
            'window_size_seconds': 60,      # 1-minute windows
            'max_measurements_per_window': 1000,
            'enable_real_time_monitoring': True,
            'enable_prometheus_metrics': True,
            'log_threshold_ms': 100,        # Log events > 100ms
            'alert_threshold_ms': 500,      # Alert on events > 500ms
        }
        
        self.logger.info("⚡ [LATENCY_PROFILER] Latency Profiler initialized")
        self.logger.info(f"⚡ [LATENCY_PROFILER] Window size: {self.profiling_config['window_size_seconds']} seconds")
        self.logger.info(f"⚡ [LATENCY_PROFILER] Log threshold: {self.profiling_config['log_threshold_ms']} ms")
    
    async def measure_latency(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any:
        """
        Measure latency of an operation
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result of the operation
        """
        try:
            start_time = time.time()
            
            # Execute operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            # Calculate latency
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Record measurement
            await self._record_measurement(operation_name, latency_ms, True)
            
            # Log if above threshold
            if latency_ms > self.profiling_config['log_threshold_ms']:
                self.logger.warning(f"⚡ [LATENCY] {operation_name}: {latency_ms:.2f} ms (above threshold)")
            
            # Alert if above alert threshold
            if latency_ms > self.profiling_config['alert_threshold_ms']:
                self.logger.critical(f"🚨 [LATENCY_ALERT] {operation_name}: {latency_ms:.2f} ms (CRITICAL)")
            
            return result
            
        except Exception as e:
            # Record failed measurement
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            await self._record_measurement(operation_name, latency_ms, False)
            
            self.logger.error(f"❌ [LATENCY_ERROR] {operation_name}: {latency_ms:.2f} ms - {str(e)}")
            
            raise e
    
    async def _record_measurement(self, operation_name: str, latency_ms: float, success: bool):
        """Record a latency measurement"""
        try:
            # Add to measurements
            self.measurements[operation_name].append({
                'timestamp': time.time(),
                'latency_ms': latency_ms,
                'success': success
            })
            
            # Update counters
            self.total_measurements += 1
            if success:
                self.successful_measurements += 1
            else:
                self.failed_measurements += 1
            
            # Create performance event
            event = PerformanceEvent(
                timestamp=time.time(),
                event_type=operation_name,
                duration_ms=latency_ms,
                success=success,
                metadata={'operation': operation_name}
            )
            
            self.events.append(event)
            
            # Update real-time metrics if enabled
            if self.profiling_config['enable_real_time_monitoring']:
                await self._update_real_time_metrics()
            
        except Exception as e:
            self.logger.error(f"❌ [RECORD_MEASUREMENT] Error recording measurement: {e}")
    
    async def _update_real_time_metrics(self):
        """Update real-time metrics"""
        try:
            current_time = time.time()
            
            # Update metrics for each operation
            for operation_name, measurements in self.measurements.items():
                if not measurements:
                    continue
                
                # Get recent measurements (last window)
                window_start = current_time - self.profiling_config['window_size_seconds']
                recent_measurements = [
                    m for m in measurements 
                    if m['timestamp'] >= window_start
                ]
                
                if not recent_measurements:
                    continue
                
                # Calculate metrics
                latencies = [m['latency_ms'] for m in recent_measurements]
                successful = [m for m in recent_measurements if m['success']]
                
                metrics = LatencyMetrics(
                    p50_ms=statistics.median(latencies),
                    p95_ms=self._percentile(latencies, 95),
                    p99_ms=self._percentile(latencies, 99),
                    p99_9_ms=self._percentile(latencies, 99.9),
                    max_ms=max(latencies),
                    min_ms=min(latencies),
                    mean_ms=statistics.mean(latencies),
                    std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                    total_measurements=len(recent_measurements),
                    successful_measurements=len(successful),
                    failed_measurements=len(recent_measurements) - len(successful),
                    window_start=window_start,
                    window_end=current_time,
                    window_duration_seconds=self.profiling_config['window_size_seconds']
                )
                
                self.current_metrics[operation_name] = metrics
            
            self.last_update_time = current_time
            
        except Exception as e:
            self.logger.error(f"❌ [UPDATE_METRICS] Error updating real-time metrics: {e}")
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        try:
            if not data:
                return 0.0
            
            sorted_data = sorted(data)
            index = (percentile / 100) * (len(sorted_data) - 1)
            
            if index.is_integer():
                return sorted_data[int(index)]
            else:
                lower = sorted_data[int(index)]
                upper = sorted_data[int(index) + 1]
                return lower + (upper - lower) * (index - int(index))
                
        except Exception as e:
            self.logger.error(f"❌ [PERCENTILE] Error calculating percentile: {e}")
            return 0.0
    
    def get_latency_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get latency metrics for operation(s)"""
        try:
            if operation_name:
                if operation_name in self.current_metrics:
                    return {operation_name: self.current_metrics[operation_name].__dict__}
                else:
                    return {}
            else:
                return {name: metrics.__dict__ for name, metrics in self.current_metrics.items()}
                
        except Exception as e:
            self.logger.error(f"❌ [GET_METRICS] Error getting latency metrics: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            uptime_seconds = time.time() - self.start_time
            
            return {
                'uptime_seconds': uptime_seconds,
                'uptime_hours': uptime_seconds / 3600,
                'total_measurements': self.total_measurements,
                'successful_measurements': self.successful_measurements,
                'failed_measurements': self.failed_measurements,
                'success_rate': self.successful_measurements / self.total_measurements if self.total_measurements > 0 else 0.0,
                'measurements_per_second': self.total_measurements / uptime_seconds if uptime_seconds > 0 else 0.0,
                'operations_tracked': len(self.measurements),
                'current_metrics': {name: metrics.__dict__ for name, metrics in self.current_metrics.items()},
                'config': self.profiling_config,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_SUMMARY] Error getting performance summary: {e}")
            return {}
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            current_time = time.time()
            
            # Get all measurements
            all_measurements = []
            for operation_name, measurements in self.measurements.items():
                for measurement in measurements:
                    all_measurements.append({
                        'operation': operation_name,
                        'timestamp': measurement['timestamp'],
                        'latency_ms': measurement['latency_ms'],
                        'success': measurement['success']
                    })
            
            # Sort by timestamp
            all_measurements.sort(key=lambda x: x['timestamp'])
            
            # Calculate overall metrics
            if all_measurements:
                latencies = [m['latency_ms'] for m in all_measurements]
                successful = [m for m in all_measurements if m['success']]
                
                overall_metrics = {
                    'p50_ms': statistics.median(latencies),
                    'p95_ms': self._percentile(latencies, 95),
                    'p99_ms': self._percentile(latencies, 99),
                    'p99_9_ms': self._percentile(latencies, 99.9),
                    'max_ms': max(latencies),
                    'min_ms': min(latencies),
                    'mean_ms': statistics.mean(latencies),
                    'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                    'total_measurements': len(all_measurements),
                    'successful_measurements': len(successful),
                    'failed_measurements': len(all_measurements) - len(successful),
                    'success_rate': len(successful) / len(all_measurements) if all_measurements else 0.0,
                }
            else:
                overall_metrics = {}
            
            # Generate report
            report = {
                'report_timestamp': current_time,
                'report_date': datetime.fromtimestamp(current_time).isoformat(),
                'overall_metrics': overall_metrics,
                'operation_metrics': {name: metrics.__dict__ for name, metrics in self.current_metrics.items()},
                'performance_summary': self.get_performance_summary(),
                'config': self.profiling_config,
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_REPORT] Error generating performance report: {e}")
            return {}
    
    async def export_metrics_to_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            prometheus_metrics = []
            
            # Add overall metrics
            prometheus_metrics.append(f"# HELP trading_system_uptime_seconds Total uptime in seconds")
            prometheus_metrics.append(f"# TYPE trading_system_uptime_seconds counter")
            prometheus_metrics.append(f"trading_system_uptime_seconds {time.time() - self.start_time}")
            
            prometheus_metrics.append(f"# HELP trading_system_total_measurements Total number of measurements")
            prometheus_metrics.append(f"# TYPE trading_system_total_measurements counter")
            prometheus_metrics.append(f"trading_system_total_measurements {self.total_measurements}")
            
            prometheus_metrics.append(f"# HELP trading_system_success_rate Success rate of operations")
            prometheus_metrics.append(f"# TYPE trading_system_success_rate gauge")
            success_rate = self.successful_measurements / self.total_measurements if self.total_measurements > 0 else 0.0
            prometheus_metrics.append(f"trading_system_success_rate {success_rate}")
            
            # Add operation-specific metrics
            for operation_name, metrics in self.current_metrics.items():
                prometheus_metrics.append(f"# HELP trading_system_latency_ms Latency in milliseconds")
                prometheus_metrics.append(f"# TYPE trading_system_latency_ms histogram")
                prometheus_metrics.append(f'trading_system_latency_ms{{operation="{operation_name}",quantile="0.5"}} {metrics.p50_ms}')
                prometheus_metrics.append(f'trading_system_latency_ms{{operation="{operation_name}",quantile="0.95"}} {metrics.p95_ms}')
                prometheus_metrics.append(f'trading_system_latency_ms{{operation="{operation_name}",quantile="0.99"}} {metrics.p99_ms}')
                prometheus_metrics.append(f'trading_system_latency_ms{{operation="{operation_name}",quantile="0.999"}} {metrics.p99_9_ms}')
                prometheus_metrics.append(f'trading_system_latency_ms{{operation="{operation_name}",quantile="1.0"}} {metrics.max_ms}')
                
                prometheus_metrics.append(f"# HELP trading_system_operation_count Total number of operations")
                prometheus_metrics.append(f"# TYPE trading_system_operation_count counter")
                prometheus_metrics.append(f'trading_system_operation_count{{operation="{operation_name}"}} {metrics.total_measurements}')
            
            return '\n'.join(prometheus_metrics)
            
        except Exception as e:
            self.logger.error(f"❌ [PROMETHEUS_EXPORT] Error exporting Prometheus metrics: {e}")
            return ""
    
    def get_slowest_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest operations"""
        try:
            slowest = []
            
            for operation_name, metrics in self.current_metrics.items():
                slowest.append({
                    'operation': operation_name,
                    'p95_ms': metrics.p95_ms,
                    'p99_ms': metrics.p99_ms,
                    'max_ms': metrics.max_ms,
                    'mean_ms': metrics.mean_ms,
                    'total_measurements': metrics.total_measurements
                })
            
            # Sort by p95 latency
            slowest.sort(key=lambda x: x['p95_ms'], reverse=True)
            
            return slowest[:limit]
            
        except Exception as e:
            self.logger.error(f"❌ [SLOWEST_OPERATIONS] Error getting slowest operations: {e}")
            return []
    
    def get_performance_insights(self) -> List[str]:
        """Get performance optimization insights"""
        try:
            insights = []
            
            # Check for high latency operations
            for operation_name, metrics in self.current_metrics.items():
                if metrics.p95_ms > 100:
                    insights.append(f"High latency detected in {operation_name}: p95={metrics.p95_ms:.2f}ms")
                
                if metrics.p99_ms > 500:
                    insights.append(f"Critical latency in {operation_name}: p99={metrics.p99_ms:.2f}ms")
                
                if metrics.failed_measurements > 0:
                    failure_rate = metrics.failed_measurements / metrics.total_measurements
                    if failure_rate > 0.01:  # 1% failure rate
                        insights.append(f"High failure rate in {operation_name}: {failure_rate:.2%}")
            
            # Check overall performance
            if self.total_measurements > 0:
                success_rate = self.successful_measurements / self.total_measurements
                if success_rate < 0.95:  # 95% success rate
                    insights.append(f"Overall success rate below target: {success_rate:.2%}")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_INSIGHTS] Error getting performance insights: {e}")
            return []
