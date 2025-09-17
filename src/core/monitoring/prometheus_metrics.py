"""
Prometheus Metrics Server - Expose required metrics for monitoring
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
from datetime import datetime, timedelta

class PrometheusMetrics:
    """
    Prometheus metrics for XRP trading system
    """
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Bot cycle metrics
        self.bot_cycle_seconds = Histogram(
            'bot_cycle_seconds',
            'Time taken for each bot cycle',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Order submission metrics
        self.order_submit_seconds = Histogram(
            'order_submit_seconds',
            'Time taken to submit orders',
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # Feasibility metrics
        self.feasibility_pass_total = Counter(
            'feasibility_pass_total',
            'Total number of feasibility checks passed',
            registry=self.registry
        )
        
        self.feasibility_fail_total = Counter(
            'feasibility_fail_total',
            'Total number of feasibility checks failed',
            ['reason'],
            registry=self.registry
        )
        
        # Guardian metrics
        self.guardian_triggers_total = Counter(
            'guardian_triggers_total',
            'Total number of guardian triggers',
            ['trigger_type'],
            registry=self.registry
        )
        
        # PnL metrics
        self.funding_pnl_total = Counter(
            'funding_pnl_total',
            'Total funding PnL',
            registry=self.registry
        )
        
        self.fee_rebate_total = Counter(
            'fee_rebate_total',
            'Total fee rebates',
            registry=self.registry
        )
        
        # System health metrics
        self.system_health = Gauge(
            'system_health',
            'System health status (1=healthy, 0=unhealthy)',
            registry=self.registry
        )
        
        self.active_orders = Gauge(
            'active_orders',
            'Number of active orders',
            registry=self.registry
        )
        
        self.account_balance = Gauge(
            'account_balance',
            'Account balance',
            ['currency'],
            registry=self.registry
        )
        
        # Risk metrics
        self.var_95 = Gauge(
            'var_95',
            'Value at Risk 95%',
            registry=self.registry
        )
        
        self.expected_shortfall_95 = Gauge(
            'expected_shortfall_95',
            'Expected Shortfall 95%',
            registry=self.registry
        )
        
        # Performance metrics
        self.sharpe_ratio = Gauge(
            'sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            'max_drawdown',
            'Maximum drawdown',
            registry=self.registry
        )
        
        # Start metrics server
        self._start_server()
    
    def _start_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port, registry=self.registry)
            self.logger.info(f"📊 Prometheus metrics server started on port {self.port}")
            self.logger.info(f"📊 Metrics available at http://localhost:{self.port}/metrics")
        except Exception as e:
            self.logger.error(f"❌ Failed to start metrics server: {e}")
    
    def record_bot_cycle(self, duration: float):
        """Record bot cycle duration"""
        self.bot_cycle_seconds.observe(duration)
    
    def record_order_submit(self, duration: float):
        """Record order submission duration"""
        self.order_submit_seconds.observe(duration)
    
    def record_feasibility_pass(self):
        """Record feasibility check pass"""
        self.feasibility_pass_total.inc()
    
    def record_feasibility_fail(self, reason: str):
        """Record feasibility check fail"""
        self.feasibility_fail_total.labels(reason=reason).inc()
    
    def record_guardian_trigger(self, trigger_type: str):
        """Record guardian trigger"""
        self.guardian_triggers_total.labels(trigger_type=trigger_type).inc()
    
    def record_funding_pnl(self, amount: float):
        """Record funding PnL"""
        self.funding_pnl_total.inc(amount)
    
    def record_fee_rebate(self, amount: float):
        """Record fee rebate"""
        self.fee_rebate_total.inc(amount)
    
    def update_system_health(self, healthy: bool):
        """Update system health status"""
        self.system_health.set(1 if healthy else 0)
    
    def update_active_orders(self, count: int):
        """Update active orders count"""
        self.active_orders.set(count)
    
    def update_account_balance(self, currency: str, balance: float):
        """Update account balance"""
        self.account_balance.labels(currency=currency).set(balance)
    
    def update_risk_metrics(self, var_95: float, es_95: float):
        """Update risk metrics"""
        self.var_95.set(var_95)
        self.expected_shortfall_95.set(es_95)
    
    def update_performance_metrics(self, sharpe: float, max_dd: float):
        """Update performance metrics"""
        self.sharpe_ratio.set(sharpe)
        self.max_drawdown.set(max_dd)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for dashboard"""
        try:
            # This would typically query the metrics registry
            # For now, return a summary structure
            return {
                'bot_cycle_avg': 0.0,  # Would be calculated from histogram
                'feasibility_pass_rate': 0.0,  # Would be calculated from counters
                'guardian_triggers': 0,  # Would be calculated from counters
                'system_health': 1,  # Current health status
                'active_orders': 0,  # Current active orders
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting metrics summary: {e}")
            return {}

# Global metrics instance
_metrics_instance: Optional[PrometheusMetrics] = None

def get_metrics() -> PrometheusMetrics:
    """Get global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        port = int(os.getenv('METRICS_PORT', '8000'))
        _metrics_instance = PrometheusMetrics(port)
    return _metrics_instance

def start_metrics_server(port: int = 8000) -> PrometheusMetrics:
    """Start metrics server and return instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PrometheusMetrics(port)
    return _metrics_instance

# Demo function
def demo_metrics():
    """Demo the metrics system"""
    print("📊 Prometheus Metrics Demo")
    print("=" * 50)
    
    # Start metrics server
    metrics = start_metrics_server(8001)  # Use different port for demo
    
    # Simulate some metrics
    print("📊 Recording sample metrics...")
    
    # Bot cycle
    metrics.record_bot_cycle(0.5)
    metrics.record_bot_cycle(1.2)
    metrics.record_bot_cycle(0.8)
    
    # Order submission
    metrics.record_order_submit(0.05)
    metrics.record_order_submit(0.12)
    
    # Feasibility
    metrics.record_feasibility_pass()
    metrics.record_feasibility_pass()
    metrics.record_feasibility_fail('insufficient_depth')
    
    # Guardian
    metrics.record_guardian_trigger('tp_sl_activation_failed')
    
    # PnL
    metrics.record_funding_pnl(10.5)
    metrics.record_fee_rebate(2.3)
    
    # System health
    metrics.update_system_health(True)
    metrics.update_active_orders(3)
    metrics.update_account_balance('USDC', 1000.0)
    
    # Risk metrics
    metrics.update_risk_metrics(-3.05, -4.22)
    
    # Performance metrics
    metrics.update_performance_metrics(1.80, -0.15)
    
    print("✅ Metrics recorded successfully")
    print(f"📊 Metrics available at: http://localhost:8001/metrics")
    
    # Get summary
    summary = metrics.get_metrics_summary()
    print(f"📊 Metrics Summary: {summary}")

if __name__ == "__main__":
    demo_metrics()