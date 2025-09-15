"""
📊 PROMETHEUS METRICS COLLECTOR
Comprehensive metrics collection for trading system monitoring
"""

import time
import threading
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    start_http_server, CollectorRegistry, generate_latest
)
import json
import os

from src.core.utils.logger import Logger

class TradingMetricsCollector:
    """
    📊 Comprehensive Prometheus metrics collector for trading system
    """
    
    def __init__(self, port: int = 8000, logger: Optional[Logger] = None):
        self.port = port
        self.logger = logger or Logger()
        self.registry = CollectorRegistry()
        
        # Initialize all metrics
        self._initialize_metrics()
        
        # Start HTTP server for metrics endpoint
        self._start_metrics_server()
        
        self.logger.info(f"📊 [METRICS] Prometheus metrics collector initialized on port {port}")
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics"""
        
        # === TRADING METRICS ===
        
        # Trade counters
        self.trades_total = Counter(
            'trading_trades_total',
            'Total number of trades executed',
            ['strategy', 'side', 'symbol', 'order_type', 'is_live'],
            registry=self.registry
        )
        
        self.trades_successful = Counter(
            'trading_trades_successful_total',
            'Total number of successful trades',
            ['strategy', 'side', 'symbol', 'order_type', 'is_live'],
            registry=self.registry
        )
        
        self.trades_failed = Counter(
            'trading_trades_failed_total',
            'Total number of failed trades',
            ['strategy', 'side', 'symbol', 'order_type', 'is_live', 'error_type'],
            registry=self.registry
        )
        
        # PnL metrics
        self.pnl_total = Gauge(
            'trading_pnl_total_usd',
            'Total profit and loss in USD',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.pnl_realized = Gauge(
            'trading_pnl_realized_usd',
            'Realized profit and loss in USD',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.pnl_unrealized = Gauge(
            'trading_pnl_unrealized_usd',
            'Unrealized profit and loss in USD',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.pnl_percentage = Gauge(
            'trading_pnl_percentage',
            'Total PnL as percentage of initial capital',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        # Position metrics
        self.position_size = Gauge(
            'trading_position_size',
            'Current position size',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.position_value = Gauge(
            'trading_position_value_usd',
            'Current position value in USD',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.avg_entry_price = Gauge(
            'trading_avg_entry_price',
            'Average entry price',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        # === EXECUTION METRICS ===
        
        # Latency metrics
        self.order_latency = Histogram(
            'trading_order_latency_seconds',
            'Order execution latency in seconds',
            ['strategy', 'order_type', 'network_condition'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.api_latency = Histogram(
            'trading_api_latency_seconds',
            'API call latency in seconds',
            ['api_endpoint', 'method'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Slippage metrics
        self.slippage_bps = Histogram(
            'trading_slippage_basis_points',
            'Slippage in basis points',
            ['strategy', 'side', 'symbol', 'order_type'],
            buckets=[-100, -50, -25, -10, -5, -1, 0, 1, 5, 10, 25, 50, 100, 200, 500],
            registry=self.registry
        )
        
        self.slippage_cost = Counter(
            'trading_slippage_cost_usd_total',
            'Total slippage cost in USD',
            ['strategy', 'side', 'symbol'],
            registry=self.registry
        )
        
        self.market_impact = Histogram(
            'trading_market_impact_percentage',
            'Market impact as percentage',
            ['strategy', 'side', 'symbol', 'order_size_category'],
            buckets=[0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # === FEES AND COSTS ===
        
        self.fees_paid = Counter(
            'trading_fees_paid_usd_total',
            'Total fees paid in USD',
            ['strategy', 'symbol', 'fee_type'],
            registry=self.registry
        )
        
        self.funding_payments = Counter(
            'trading_funding_payments_usd_total',
            'Total funding payments in USD',
            ['strategy', 'symbol', 'payment_type'],
            registry=self.registry
        )
        
        self.funding_rate = Gauge(
            'trading_funding_rate',
            'Current funding rate',
            ['symbol'],
            registry=self.registry
        )
        
        # === MARKET DATA METRICS ===
        
        self.price_current = Gauge(
            'trading_price_current',
            'Current market price',
            ['symbol'],
            registry=self.registry
        )
        
        self.price_change_24h = Gauge(
            'trading_price_change_24h_percentage',
            '24-hour price change percentage',
            ['symbol'],
            registry=self.registry
        )
        
        self.volume_24h = Gauge(
            'trading_volume_24h_usd',
            '24-hour trading volume in USD',
            ['symbol'],
            registry=self.registry
        )
        
        self.spread_bps = Gauge(
            'trading_spread_basis_points',
            'Current bid-ask spread in basis points',
            ['symbol'],
            registry=self.registry
        )
        
        # === SYSTEM PERFORMANCE ===
        
        self.system_uptime = Gauge(
            'trading_system_uptime_seconds',
            'System uptime in seconds',
            ['strategy'],
            registry=self.registry
        )
        
        self.cycle_count = Gauge(
            'trading_cycle_count',
            'Number of trading cycles completed',
            ['strategy'],
            registry=self.registry
        )
        
        self.emergency_mode = Gauge(
            'trading_emergency_mode',
            'Emergency mode status (1=enabled, 0=disabled)',
            ['strategy'],
            registry=self.registry
        )
        
        self.margin_usage_percentage = Gauge(
            'trading_margin_usage_percentage',
            'Margin usage as percentage of available margin',
            ['strategy'],
            registry=self.registry
        )
        
        # === RISK METRICS ===
        
        self.drawdown_percentage = Gauge(
            'trading_drawdown_percentage',
            'Current drawdown as percentage',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.max_drawdown_percentage = Gauge(
            'trading_max_drawdown_percentage',
            'Maximum drawdown as percentage',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.volatility = Gauge(
            'trading_volatility',
            'Current volatility',
            ['symbol', 'timeframe'],
            registry=self.registry
        )
        
        # === STRATEGY PERFORMANCE ===
        
        self.win_rate = Gauge(
            'trading_win_rate_percentage',
            'Win rate as percentage',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.profit_factor = Gauge(
            'trading_profit_factor',
            'Profit factor (gross profit / gross loss)',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'trading_sharpe_ratio',
            'Sharpe ratio',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        # === SYSTEM INFO ===
        
        self.system_info = Info(
            'trading_system_info',
            'System information',
            registry=self.registry
        )
        
        # Set system info
        self.system_info.info({
            'version': '2.0.0',
            'strategy': 'Ultra-Efficient XRP Trading System',
            'exchange': 'Hyperliquid',
            'symbol': 'XRP'
        })
        
        self.logger.info("📊 [METRICS] All Prometheus metrics initialized")
    
    def _start_metrics_server(self):
        """Start HTTP server for metrics endpoint"""
        try:
            start_http_server(self.port, registry=self.registry)
            self.logger.info(f"📊 [METRICS] Metrics server started on port {self.port}")
        except Exception as e:
            self.logger.error(f"❌ [METRICS] Failed to start metrics server: {e}")
    
    # === TRADING METRICS METHODS ===
    
    def record_trade(self, 
                    strategy: str, 
                    side: str, 
                    symbol: str, 
                    order_type: str, 
                    is_live: bool,
                    success: bool = True,
                    error_type: Optional[str] = None):
        """Record trade execution"""
        labels = {
            'strategy': strategy,
            'side': side,
            'symbol': symbol,
            'order_type': order_type,
            'is_live': str(is_live)
        }
        
        self.trades_total.labels(**labels).inc()
        
        if success:
            self.trades_successful.labels(**labels).inc()
        else:
            error_labels = labels.copy()
            error_labels['error_type'] = error_type or 'unknown'
            self.trades_failed.labels(**error_labels).inc()
    
    def update_pnl(self, 
                  strategy: str, 
                  symbol: str, 
                  total_pnl: float, 
                  realized_pnl: float, 
                  unrealized_pnl: float,
                  pnl_percentage: float):
        """Update PnL metrics"""
        labels = {'strategy': strategy, 'symbol': symbol}
        
        self.pnl_total.labels(**labels).set(total_pnl)
        self.pnl_realized.labels(**labels).set(realized_pnl)
        self.pnl_unrealized.labels(**labels).set(unrealized_pnl)
        self.pnl_percentage.labels(**labels).set(pnl_percentage)
    
    def update_position(self, 
                       strategy: str, 
                       symbol: str, 
                       position_size: float, 
                       position_value: float, 
                       avg_entry_price: float):
        """Update position metrics"""
        labels = {'strategy': strategy, 'symbol': symbol}
        
        self.position_size.labels(**labels).set(position_size)
        self.position_value.labels(**labels).set(position_value)
        self.avg_entry_price.labels(**labels).set(avg_entry_price)
    
    # === EXECUTION METRICS METHODS ===
    
    def record_order_latency(self, 
                           strategy: str, 
                           order_type: str, 
                           network_condition: str, 
                           latency_seconds: float):
        """Record order execution latency"""
        labels = {
            'strategy': strategy,
            'order_type': order_type,
            'network_condition': network_condition
        }
        self.order_latency.labels(**labels).observe(latency_seconds)
    
    def record_api_latency(self, 
                          api_endpoint: str, 
                          method: str, 
                          latency_seconds: float):
        """Record API call latency"""
        labels = {'api_endpoint': api_endpoint, 'method': method}
        self.api_latency.labels(**labels).observe(latency_seconds)
    
    def record_slippage(self, 
                       strategy: str, 
                       side: str, 
                       symbol: str, 
                       order_type: str, 
                       slippage_bps: float, 
                       slippage_cost: float):
        """Record slippage metrics"""
        labels = {
            'strategy': strategy,
            'side': side,
            'symbol': symbol,
            'order_type': order_type
        }
        
        self.slippage_bps.labels(**labels).observe(slippage_bps)
        
        # Record slippage cost
        cost_labels = {'strategy': strategy, 'side': side, 'symbol': symbol}
        self.slippage_cost.labels(**cost_labels).inc(slippage_cost)
    
    def record_market_impact(self, 
                           strategy: str, 
                           side: str, 
                           symbol: str, 
                           order_size_category: str, 
                           impact_percentage: float):
        """Record market impact"""
        labels = {
            'strategy': strategy,
            'side': side,
            'symbol': symbol,
            'order_size_category': order_size_category
        }
        self.market_impact.labels(**labels).observe(impact_percentage)
    
    # === FEES AND COSTS METHODS ===
    
    def record_fees(self, 
                   strategy: str, 
                   symbol: str, 
                   fee_type: str, 
                   fee_amount: float):
        """Record fees paid"""
        labels = {'strategy': strategy, 'symbol': symbol, 'fee_type': fee_type}
        self.fees_paid.labels(**labels).inc(fee_amount)
    
    def record_funding_payment(self, 
                              strategy: str, 
                              symbol: str, 
                              payment_type: str, 
                              payment_amount: float):
        """Record funding payment"""
        labels = {'strategy': strategy, 'symbol': symbol, 'payment_type': payment_type}
        self.funding_payments.labels(**labels).inc(payment_amount)
    
    def update_funding_rate(self, symbol: str, funding_rate: float):
        """Update funding rate"""
        self.funding_rate.labels(symbol=symbol).set(funding_rate)
    
    # === MARKET DATA METHODS ===
    
    def update_market_data(self, 
                          symbol: str, 
                          price: float, 
                          price_change_24h: float, 
                          volume_24h: float, 
                          spread_bps: float):
        """Update market data metrics"""
        self.price_current.labels(symbol=symbol).set(price)
        self.price_change_24h.labels(symbol=symbol).set(price_change_24h)
        self.volume_24h.labels(symbol=symbol).set(volume_24h)
        self.spread_bps.labels(symbol=symbol).set(spread_bps)
    
    # === SYSTEM PERFORMANCE METHODS ===
    
    def update_system_performance(self, 
                                 strategy: str, 
                                 uptime_seconds: float, 
                                 cycle_count: int, 
                                 emergency_mode: bool, 
                                 margin_usage_percentage: float):
        """Update system performance metrics"""
        self.system_uptime.labels(strategy=strategy).set(uptime_seconds)
        self.cycle_count.labels(strategy=strategy).set(cycle_count)
        self.emergency_mode.labels(strategy=strategy).set(1 if emergency_mode else 0)
        self.margin_usage_percentage.labels(strategy=strategy).set(margin_usage_percentage)
    
    # === RISK METRICS METHODS ===
    
    def update_risk_metrics(self, 
                           strategy: str, 
                           symbol: str, 
                           drawdown_percentage: float, 
                           max_drawdown_percentage: float, 
                           volatility: float, 
                           timeframe: str = '1h'):
        """Update risk metrics"""
        self.drawdown_percentage.labels(strategy=strategy, symbol=symbol).set(drawdown_percentage)
        self.max_drawdown_percentage.labels(strategy=strategy, symbol=symbol).set(max_drawdown_percentage)
        self.volatility.labels(symbol=symbol, timeframe=timeframe).set(volatility)
    
    # === STRATEGY PERFORMANCE METHODS ===
    
    def update_strategy_performance(self, 
                                   strategy: str, 
                                   symbol: str, 
                                   win_rate: float, 
                                   profit_factor: float, 
                                   sharpe_ratio: float):
        """Update strategy performance metrics"""
        labels = {'strategy': strategy, 'symbol': symbol}
        
        self.win_rate.labels(**labels).set(win_rate)
        self.profit_factor.labels(**labels).set(profit_factor)
        self.sharpe_ratio.labels(**labels).set(sharpe_ratio)
    
    # === UTILITY METHODS ===
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        try:
            metrics_data = generate_latest(self.registry).decode('utf-8')
            return {
                'status': 'active',
                'port': self.port,
                'metrics_count': len([line for line in metrics_data.split('\n') if line and not line.startswith('#')]),
                'endpoint': f'http://localhost:{self.port}/metrics'
            }
        except Exception as e:
            self.logger.error(f"❌ [METRICS] Error getting metrics summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            self.logger.error(f"❌ [METRICS] Error exporting metrics: {e}")
            return f"# Error exporting metrics: {e}"
    
    def save_metrics_snapshot(self, filepath: str):
        """Save current metrics to file"""
        try:
            metrics_data = self.export_metrics()
            with open(filepath, 'w') as f:
                f.write(metrics_data)
            self.logger.info(f"📊 [METRICS] Metrics snapshot saved to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ [METRICS] Error saving metrics snapshot: {e}")

# Global metrics collector instance
_metrics_collector: Optional[TradingMetricsCollector] = None

def get_metrics_collector(port: int = 8000, logger: Optional[Logger] = None) -> TradingMetricsCollector:
    """Get or create global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = TradingMetricsCollector(port=port, logger=logger)
    return _metrics_collector

def record_trade_metrics(trade_data: Dict[str, Any], metrics_collector: Optional[TradingMetricsCollector] = None):
    """Convenience function to record trade metrics"""
    if metrics_collector is None:
        metrics_collector = get_metrics_collector()
    
    # Record trade
    metrics_collector.record_trade(
        strategy=trade_data.get('strategy', 'unknown'),
        side=trade_data.get('side', 'unknown'),
        symbol=trade_data.get('symbol', 'XRP'),
        order_type=trade_data.get('order_type', 'MARKET'),
        is_live=trade_data.get('is_live_trade', False),
        success=trade_data.get('success', True),
        error_type=trade_data.get('error_type')
    )
    
    # Record slippage if available
    if 'slippage_bps' in trade_data and 'slippage_cost' in trade_data:
        metrics_collector.record_slippage(
            strategy=trade_data.get('strategy', 'unknown'),
            side=trade_data.get('side', 'unknown'),
            symbol=trade_data.get('symbol', 'XRP'),
            order_type=trade_data.get('order_type', 'MARKET'),
            slippage_bps=trade_data['slippage_bps'],
            slippage_cost=trade_data['slippage_cost']
        )
    
    # Record fees if available
    if 'fees_paid' in trade_data:
        metrics_collector.record_fees(
            strategy=trade_data.get('strategy', 'unknown'),
            symbol=trade_data.get('symbol', 'XRP'),
            fee_type='trading',
            fee_amount=trade_data['fees_paid']
        )
    
    # Record latency if available
    if 'latency_ms' in trade_data:
        metrics_collector.record_order_latency(
            strategy=trade_data.get('strategy', 'unknown'),
            order_type=trade_data.get('order_type', 'MARKET'),
            network_condition=trade_data.get('network_condition', 'unknown'),
            latency_seconds=trade_data['latency_ms'] / 1000.0
        )
