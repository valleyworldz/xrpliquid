"""
Prometheus Metrics Exporter for Hat Manifesto Trading System
Exports real-time metrics for Grafana dashboards.
"""

import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from prometheus_client import start_http_server, Counter, Histogram, Gauge, CollectorRegistry


class HatManifestoPrometheusExporter:
    """Prometheus metrics exporter for the Hat Manifesto trading system."""
    
    def __init__(self, port: int = 8000, reports_dir: str = "reports"):
        self.port = port
        self.reports_dir = Path(reports_dir)
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self._init_metrics()
        
        # Start metrics server
        self.server = start_http_server(port, registry=self.registry)
        print(f"📊 Prometheus metrics server started on port {port}")
        
        # Start background metrics update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
        self.update_thread.start()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        
        # Portfolio metrics
        self.portfolio_value = Gauge(
            'hat_manifesto_portfolio_value',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.daily_pnl = Gauge(
            'hat_manifesto_daily_pnl',
            'Daily P&L in USD',
            registry=self.registry
        )
        
        self.total_pnl = Gauge(
            'hat_manifesto_total_pnl',
            'Total P&L in USD',
            registry=self.registry
        )
        
        # Position metrics
        self.active_positions = Gauge(
            'hat_manifesto_active_positions',
            'Number of active positions',
            registry=self.registry
        )
        
        self.position_value = Gauge(
            'hat_manifesto_position_value',
            'Total position value in USD',
            registry=self.registry
        )
        
        # Risk metrics
        self.risk_level = Gauge(
            'hat_manifesto_risk_level',
            'Current risk level (0-1)',
            registry=self.registry
        )
        
        self.drawdown = Gauge(
            'hat_manifesto_drawdown',
            'Current drawdown percentage',
            registry=self.registry
        )
        
        self.margin_ratio = Gauge(
            'hat_manifesto_margin_ratio',
            'Current margin ratio',
            registry=self.registry
        )
        
        # Execution metrics
        self.execution_latency = Histogram(
            'hat_manifesto_execution_latency_seconds',
            'Execution latency in seconds',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.trades_total = Counter(
            'hat_manifesto_trades_total',
            'Total number of trades',
            ['strategy', 'side', 'maker'],
            registry=self.registry
        )
        
        self.maker_trades_total = Counter(
            'hat_manifesto_maker_trades_total',
            'Total number of maker trades',
            registry=self.registry
        )
        
        self.slippage_bps = Histogram(
            'hat_manifesto_slippage_bps',
            'Slippage in basis points',
            buckets=[0, 1, 2, 5, 10, 20, 50, 100],
            registry=self.registry
        )
        
        # Risk event metrics
        self.risk_events_total = Counter(
            'hat_manifesto_risk_events_total',
            'Total number of risk events',
            ['severity', 'type'],
            registry=self.registry
        )
        
        # Strategy metrics
        self.strategy_pnl = Gauge(
            'hat_manifesto_strategy_pnl',
            'P&L by strategy',
            ['strategy'],
            registry=self.registry
        )
        
        self.strategy_trades = Counter(
            'hat_manifesto_strategy_trades_total',
            'Total trades by strategy',
            ['strategy'],
            registry=self.registry
        )
        
        # Funding arbitrage metrics
        self.funding_rate = Gauge(
            'hat_manifesto_funding_rate',
            'Current funding rate',
            registry=self.registry
        )
        
        self.funding_arbitrage_opportunities = Gauge(
            'hat_manifesto_funding_arbitrage_opportunities',
            'Number of funding arbitrage opportunities',
            registry=self.registry
        )
        
        # Performance metrics
        self.win_rate = Gauge(
            'hat_manifesto_win_rate',
            'Current win rate percentage',
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'hat_manifesto_sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )
        
        # System health metrics
        self.system_health = Gauge(
            'hat_manifesto_system_health',
            'System health status (1=healthy, 0=unhealthy)',
            registry=self.registry
        )
        
        self.last_update = Gauge(
            'hat_manifesto_last_update_timestamp',
            'Timestamp of last metrics update',
            registry=self.registry
        )
    
    def _load_trade_data(self) -> Optional[pd.DataFrame]:
        """Load latest trade data."""
        try:
            # Try to load from parquet first, then CSV
            ledger_path = self.reports_dir / "ledgers" / "trades.parquet"
            if ledger_path.exists():
                return pd.read_parquet(ledger_path)
            
            ledger_path = self.reports_dir / "ledgers" / "trades.csv"
            if ledger_path.exists():
                return pd.read_csv(ledger_path)
                
        except Exception as e:
            print(f"Warning: Could not load trade data: {e}")
        
        return None
    
    def _load_risk_events(self) -> list:
        """Load latest risk events."""
        try:
            risk_path = self.reports_dir / "risk_events" / "risk_events.json"
            if risk_path.exists():
                with open(risk_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load risk events: {e}")
        
        return []
    
    def _load_latency_data(self) -> dict:
        """Load latest latency data."""
        try:
            latency_path = self.reports_dir / "latency" / "latency_analysis.json"
            if latency_path.exists():
                with open(latency_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load latency data: {e}")
        
        return {}
    
    def _update_metrics_loop(self):
        """Background loop to update metrics."""
        while self.running:
            try:
                self._update_all_metrics()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"Error updating metrics: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _update_all_metrics(self):
        """Update all Prometheus metrics."""
        # Load data
        trades_df = self._load_trade_data()
        risk_events = self._load_risk_events()
        latency_data = self._load_latency_data()
        
        if trades_df is not None and not trades_df.empty:
            # Portfolio metrics
            total_pnl = trades_df['pnl_realized'].sum()
            self.total_pnl.set(total_pnl)
            self.portfolio_value.set(10000 + total_pnl)  # Starting with $10K
            
            # Daily P&L (last 24 hours)
            now = datetime.now()
            daily_trades = trades_df[trades_df['ts'] >= (now.timestamp() - 86400)]
            daily_pnl = daily_trades['pnl_realized'].sum()
            self.daily_pnl.set(daily_pnl)
            
            # Position metrics
            active_positions = len(trades_df[trades_df['pnl_unrealized'] != 0])
            self.active_positions.set(active_positions)
            
            position_value = trades_df['pnl_unrealized'].sum()
            self.position_value.set(position_value)
            
            # Risk metrics
            if len(trades_df) > 0:
                returns = trades_df['pnl_realized']
                if returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * (252**0.5)
                    self.sharpe_ratio.set(sharpe)
                
                win_rate = (returns > 0).mean() * 100
                self.win_rate.set(win_rate)
                
                # Calculate drawdown
                cumulative = returns.cumsum()
                peak = cumulative.expanding().max()
                drawdown = ((cumulative - peak) / peak * 100).min()
                self.drawdown.set(abs(drawdown))
                
                # Risk level based on drawdown
                risk_level = min(abs(drawdown) / 10.0, 1.0)  # 10% max drawdown = 100% risk
                self.risk_level.set(risk_level)
            
            # Strategy metrics
            strategy_pnl = trades_df.groupby('strategy_name')['pnl_realized'].sum()
            for strategy, pnl in strategy_pnl.items():
                self.strategy_pnl.labels(strategy=strategy).set(pnl)
            
            # Trade counts
            for _, trade in trades_df.iterrows():
                strategy = trade.get('strategy_name', 'unknown')
                side = trade.get('side', 'unknown')
                maker = 'true' if trade.get('maker_flag', False) else 'false'
                
                self.trades_total.labels(
                    strategy=strategy,
                    side=side,
                    maker=maker
                ).inc()
                
                if trade.get('maker_flag', False):
                    self.maker_trades_total.inc()
                
                # Slippage
                slippage = trade.get('slippage_bps', 0)
                if slippage > 0:
                    self.slippage_bps.observe(slippage)
        
        # Risk events
        for event in risk_events:
            severity = event.get('severity', 'unknown')
            event_type = event.get('type', 'unknown')
            self.risk_events_total.labels(
                severity=severity,
                type=event_type
            ).inc()
        
        # Latency metrics
        if 'latency_samples' in latency_data:
            for sample in latency_data['latency_samples']:
                self.execution_latency.observe(sample / 1000.0)  # Convert ms to seconds
        
        # System health (simple check)
        self.system_health.set(1 if trades_df is not None else 0)
        
        # Last update timestamp
        self.last_update.set(time.time())
    
    def stop(self):
        """Stop the metrics exporter."""
        self.running = False
        if hasattr(self, 'server'):
            self.server.shutdown()
        print("📊 Prometheus metrics server stopped")


def main():
    """Main function to run the Prometheus exporter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hat Manifesto Prometheus Metrics Exporter')
    parser.add_argument('--port', type=int, default=8000, help='Port for metrics server')
    parser.add_argument('--reports-dir', default='reports', help='Reports directory path')
    
    args = parser.parse_args()
    
    # Start exporter
    exporter = HatManifestoPrometheusExporter(port=args.port, reports_dir=args.reports_dir)
    
    try:
        print("📊 Prometheus exporter running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Prometheus exporter...")
        exporter.stop()


if __name__ == "__main__":
    main()
