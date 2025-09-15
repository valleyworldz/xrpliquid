#!/usr/bin/env python3
"""
Metrics and Monitoring for XRP Trading Bot
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
import threading

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠️ Prometheus client not available. Metrics will be disabled.")

class MetricsManager:
    """Centralized metrics management"""
    
    def __init__(self, enable_prometheus: bool = True, port: int = 8000):
        """Initialize metrics manager"""
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.port = port
        self.metrics_started = False
        self.lock = threading.Lock()
        
        # Initialize metrics
        self._init_metrics()
        
        # Start metrics server if enabled
        if self.enable_prometheus:
            self._start_metrics_server()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        if not self.enable_prometheus:
            return
        
        # Trading metrics
        self.trades_total = Counter(
            'xrp_bot_trades_total',
            'Total number of trades',
            ['side', 'result']
        )
        
        self.trade_value = Histogram(
            'xrp_bot_trade_value_usd',
            'Trade value in USD',
            buckets=[10, 50, 100, 500, 1000, 5000, 10000]
        )
        
        self.position_size = Gauge(
            'xrp_bot_position_size_xrp',
            'Current position size in XRP'
        )
        
        self.position_value = Gauge(
            'xrp_bot_position_value_usd',
            'Current position value in USD'
        )
        
        self.unrealized_pnl = Gauge(
            'xrp_bot_unrealized_pnl_usd',
            'Unrealized PnL in USD'
        )
        
        # Performance metrics
        self.win_rate = Gauge(
            'xrp_bot_win_rate',
            'Current win rate (0-1)'
        )
        
        self.daily_pnl = Gauge(
            'xrp_bot_daily_pnl_usd',
            'Daily PnL in USD'
        )
        
        self.consecutive_losses = Gauge(
            'xrp_bot_consecutive_losses',
            'Number of consecutive losses'
        )
        
        # API metrics
        self.api_calls_total = Counter(
            'xrp_bot_api_calls_total',
            'Total API calls',
            ['endpoint', 'status']
        )
        
        self.api_latency = Histogram(
            'xrp_bot_api_latency_seconds',
            'API call latency in seconds',
            ['endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Signal metrics
        self.signals_generated = Counter(
            'xrp_bot_signals_generated_total',
            'Total signals generated',
            ['type', 'confidence_level']
        )
        
        # Risk metrics
        self.risk_checks_failed = Counter(
            'xrp_bot_risk_checks_failed_total',
            'Total risk check failures',
            ['check_type']
        )
        
        self.margin_ratio = Gauge(
            'xrp_bot_margin_ratio',
            'Current margin ratio'
        )
        
        # System metrics
        self.bot_uptime = Gauge(
            'xrp_bot_uptime_seconds',
            'Bot uptime in seconds'
        )
        
        self.cycle_duration = Histogram(
            'xrp_bot_cycle_duration_seconds',
            'Trading cycle duration in seconds',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        if not self.enable_prometheus or self.metrics_started:
            return
        
        try:
            start_http_server(self.port)
            self.metrics_started = True
            print(f"✅ Prometheus metrics server started on port {self.port}")
        except Exception as e:
            print(f"❌ Failed to start metrics server: {e}")
            self.enable_prometheus = False
    
    def record_trade(self, side: str, result: str, value_usd: float, size_xrp: float):
        """Record a trade"""
        if not self.enable_prometheus:
            return
        
        with self.lock:
            self.trades_total.labels(side=side, result=result).inc()
            self.trade_value.observe(value_usd)
    
    def update_position(self, size_xrp: float, value_usd: float, pnl_usd: float):
        """Update position metrics"""
        if not self.enable_prometheus:
            return
        
        with self.lock:
            self.position_size.set(size_xrp)
            self.position_value.set(value_usd)
            self.unrealized_pnl.set(pnl_usd)
    
    def update_performance(self, win_rate: float, daily_pnl: float, consecutive_losses: int):
        """Update performance metrics"""
        if not self.enable_prometheus:
            return
        
        with self.lock:
            self.win_rate.set(win_rate)
            self.daily_pnl.set(daily_pnl)
            self.consecutive_losses.set(consecutive_losses)
    
    def record_api_call(self, endpoint: str, status: str, duration: float):
        """Record API call metrics"""
        if not self.enable_prometheus:
            return
        
        with self.lock:
            self.api_calls_total.labels(endpoint=endpoint, status=status).inc()
            self.api_latency.labels(endpoint=endpoint).observe(duration)
    
    def record_signal(self, signal_type: str, confidence: float):
        """Record signal generation"""
        if not self.enable_prometheus:
            return
        
        # Categorize confidence level
        if confidence >= 0.9:
            confidence_level = "high"
        elif confidence >= 0.7:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        with self.lock:
            self.signals_generated.labels(type=signal_type, confidence_level=confidence_level).inc()
    
    def record_risk_check_failure(self, check_type: str):
        """Record risk check failure"""
        if not self.enable_prometheus:
            return
        
        with self.lock:
            self.risk_checks_failed.labels(check_type=check_type).inc()
    
    def update_margin_ratio(self, ratio: float):
        """Update margin ratio"""
        if not self.enable_prometheus:
            return
        
        with self.lock:
            self.margin_ratio.set(ratio)
    
    def update_uptime(self, uptime_seconds: float):
        """Update bot uptime"""
        if not self.enable_prometheus:
            return
        
        with self.lock:
            self.bot_uptime.set(uptime_seconds)
    
    def record_cycle_duration(self, duration: float):
        """Record trading cycle duration"""
        if not self.enable_prometheus:
            return
        
        with self.lock:
            self.cycle_duration.observe(duration)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        if not self.enable_prometheus:
            return {"status": "metrics_disabled"}
        
        # This would return actual metric values
        # For now, return a placeholder
        return {
            "status": "metrics_enabled",
            "port": self.port,
            "uptime": "available",
            "trades": "tracked",
            "performance": "monitored"
        }

# Global metrics instance
metrics = MetricsManager()

# Convenience functions for backward compatibility
def record_trade(side: str, result: str, value_usd: float, size_xrp: float):
    """Record a trade (convenience function)"""
    metrics.record_trade(side, result, value_usd, size_xrp)

def update_position(size_xrp: float, value_usd: float, pnl_usd: float):
    """Update position (convenience function)"""
    metrics.update_position(size_xrp, value_usd, pnl_usd)

def record_api_call(endpoint: str, status: str, duration: float):
    """Record API call (convenience function)"""
    metrics.record_api_call(endpoint, status, duration)

def record_signal(signal_type: str, confidence: float):
    """Record signal (convenience function)"""
    metrics.record_signal(signal_type, confidence)

def record_risk_check_failure(check_type: str):
    """Record risk check failure (convenience function)"""
    metrics.record_risk_check_failure(check_type) 