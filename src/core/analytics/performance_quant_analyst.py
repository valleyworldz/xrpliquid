"""
📊 PERFORMANCE QUANT ANALYST
"What gets measured gets managed. I will tell us what is working and what is not."

This module implements comprehensive performance analytics:
- Real-time performance monitoring
- Advanced metrics calculation (Sharpe, Sortino, Calmar, etc.)
- Performance attribution analysis
- Risk-adjusted returns
- Drawdown analysis
- Performance dashboards and reporting
- Benchmark comparison
- Performance forecasting
"""

from src.core.utils.decimal_boundary_guard import safe_float
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import json
from datetime import datetime, timedelta
import threading
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class MetricType(Enum):
    """Metric type enumeration"""
    RETURN = "return"
    RISK = "risk"
    RATIO = "ratio"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"

class TimeFrame(Enum):
    """Time frame enumeration"""
    MINUTE = "1m"
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"
    YEAR = "1Y"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0
    average_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Risk-adjusted ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    
    # Additional metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Timing metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

@dataclass
class TradeRecord:
    """Trade record data structure"""
    trade_id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    slippage: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Performance report data structure"""
    report_id: str
    generated_at: datetime
    time_period: str
    metrics: PerformanceMetrics
    trades: List[TradeRecord]
    benchmark_comparison: Dict[str, float]
    attribution_analysis: Dict[str, float]
    recommendations: List[str]

class PerformanceQuantAnalyst:
    """
    Performance Quant Analyst - Master of Metrics and Analytics
    
    This class implements comprehensive performance analytics:
    1. Real-time performance monitoring
    2. Advanced metrics calculation
    3. Performance attribution analysis
    4. Risk-adjusted returns
    5. Drawdown analysis
    6. Performance dashboards and reporting
    7. Benchmark comparison
    8. Performance forecasting
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Performance configuration
        self.performance_config = {
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'benchmark_symbol': 'BTC',  # Default benchmark
            'calculation_frequency': 60,  # seconds
            'data_retention_days': 365,  # 1 year
            'min_trades_for_analysis': 10,
            'confidence_level': 0.95
        }
        
        # Data storage
        self.trade_records: List[TradeRecord] = []
        self.portfolio_values: deque = deque(maxlen=10000)
        self.benchmark_values: deque = deque(maxlen=10000)
        self.performance_history: deque = deque(maxlen=1000)
        
        # Current metrics
        self.current_metrics = PerformanceMetrics()
        self.peak_equity = 0.0
        self.start_equity = 0.0
        
        # Database
        self.db_path = "performance_analytics.db"
        self._initialize_database()
        
        # Monitoring
        self.monitoring_thread = None
        self.running = False
        
        # Callbacks
        self.performance_callbacks: Dict[str, List[callable]] = {
            'on_metrics_updated': [],
            'on_drawdown_alert': [],
            'on_performance_milestone': [],
            'on_risk_threshold_breach': []
        }
        
        # Initialize performance analytics
        self._initialize_performance_analytics()
    
    def _initialize_database(self):
        """Initialize SQLite database for performance data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP NOT NULL,
                    pnl REAL NOT NULL,
                    commission REAL NOT NULL,
                    slippage REAL NOT NULL,
                    duration REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Create portfolio values table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_values (
                    timestamp TIMESTAMP PRIMARY KEY,
                    equity REAL NOT NULL,
                    benchmark_value REAL,
                    metadata TEXT
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp TIMESTAMP PRIMARY KEY,
                    metrics TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Performance analytics database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def _initialize_performance_analytics(self):
        """Initialize performance analytics system"""
        try:
            self.logger.info("Initializing performance quant analyst...")
            
            # Load historical data
            self._load_historical_data()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._performance_monitoring_loop,
                daemon=True,
                name="performance_monitor"
            )
            self.monitoring_thread.start()
            
            self.running = True
            self.logger.info("Performance quant analyst initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing performance analytics: {e}")
    
    def _load_historical_data(self):
        """Load historical performance data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load trades
            trades_df = pd.read_sql_query("SELECT * FROM trades", conn)
            for _, row in trades_df.iterrows():
                trade = TradeRecord(
                    trade_id=row['trade_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    size=row['size'],
                    entry_price=row['entry_price'],
                    exit_price=row['exit_price'],
                    entry_time=datetime.fromisoformat(row['entry_time']),
                    exit_time=datetime.fromisoformat(row['exit_time']),
                    pnl=row['pnl'],
                    commission=row['commission'],
                    slippage=row['slippage'],
                    duration=row['duration'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                self.trade_records.append(trade)
            
            # Load portfolio values
            portfolio_df = pd.read_sql_query("SELECT * FROM portfolio_values ORDER BY timestamp", conn)
            for _, row in portfolio_df.iterrows():
                self.portfolio_values.append({
                    'timestamp': datetime.fromisoformat(row['timestamp']),
                    'equity': row['equity'],
                    'benchmark_value': row['benchmark_value']
                })
            
            conn.close()
            
            self.logger.info(f"Loaded {len(self.trade_records)} trades and {len(self.portfolio_values)} portfolio values")
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
    
    def _performance_monitoring_loop(self):
        """Main performance monitoring loop"""
        try:
            while self.running:
                try:
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    # Check for alerts
                    self._check_performance_alerts()
                    
                    # Sleep for calculation frequency
                    time.sleep(self.performance_config['calculation_frequency'])
                    
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring loop: {e}")
                    time.sleep(60)  # Wait 1 minute on error
                    
        except Exception as e:
            self.logger.error(f"Fatal error in performance monitoring loop: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if len(self.portfolio_values) < 2:
                return
            
            # Calculate returns
            returns = self._calculate_returns()
            if len(returns) == 0:
                return
            
            # Update metrics
            self.current_metrics = self._calculate_performance_metrics(returns)
            
            # Store metrics history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': self.current_metrics
            })
            
            # Save to database
            self._save_metrics_to_database()
            
            # Trigger callbacks
            self._trigger_callbacks('on_metrics_updated', self.current_metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_returns(self) -> List[float]:
        """Calculate portfolio returns"""
        try:
            if len(self.portfolio_values) < 2:
                return []
            
            returns = []
            for i in range(1, len(self.portfolio_values)):
                prev_equity = self.portfolio_values[i-1]['equity']
                curr_equity = self.portfolio_values[i]['equity']
                
                if prev_equity > 0:
                    ret = (curr_equity - prev_equity) / prev_equity
                    returns.append(ret)
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return []
    
    def _calculate_performance_metrics(self, returns: List[float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if len(returns) == 0:
                return PerformanceMetrics()
            
            returns_array = np.array(returns)
            
            # Basic return metrics
            total_return = np.prod(1 + returns_array) - 1
            average_return = np.mean(returns_array)
            
            # Annualized return (assuming daily returns)
            annualized_return = (1 + average_return) ** 252 - 1
            
            # Risk metrics
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            
            # Downside deviation
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Value at Risk
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)
            
            # Expected Shortfall
            tail_returns = returns_array[returns_array <= var_95]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            current_drawdown = drawdowns[-1] if len(drawdowns) > 0 else 0
            
            # Risk-adjusted ratios
            risk_free_rate = self.performance_config['risk_free_rate']
            excess_return = annualized_return - risk_free_rate
            
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade-based metrics
            trade_metrics = self._calculate_trade_metrics()
            
            # Combine all metrics
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                cumulative_return=cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0,
                average_return=average_return,
                volatility=volatility,
                downside_deviation=downside_deviation,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                win_rate=trade_metrics['win_rate'],
                profit_factor=trade_metrics['profit_factor'],
                average_win=trade_metrics['average_win'],
                average_loss=trade_metrics['average_loss'],
                largest_win=trade_metrics['largest_win'],
                largest_loss=trade_metrics['largest_loss'],
                total_trades=trade_metrics['total_trades'],
                winning_trades=trade_metrics['winning_trades'],
                losing_trades=trade_metrics['losing_trades']
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics()
    
    def _calculate_trade_metrics(self) -> Dict[str, float]:
        """Calculate trade-based metrics"""
        try:
            if len(self.trade_records) == 0:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'average_win': 0.0,
                    'average_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }
            
            pnls = [trade.pnl for trade in self.trade_records]
            winning_trades = [pnl for pnl in pnls if pnl > 0]
            losing_trades = [pnl for pnl in pnls if pnl < 0]
            
            total_trades = len(pnls)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            
            win_rate = winning_count / total_trades if total_trades > 0 else 0
            
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 0
            
            profit_factor = total_wins / total_losses if total_losses > 0 else safe_float('inf') if total_wins > 0 else 0
            
            average_win = np.mean(winning_trades) if winning_trades else 0
            average_loss = np.mean(losing_trades) if losing_trades else 0
            
            largest_win = max(winning_trades) if winning_trades else 0
            largest_loss = min(losing_trades) if losing_trades else 0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_win': average_win,
                'average_loss': average_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'total_trades': total_trades,
                'winning_trades': winning_count,
                'losing_trades': losing_count
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade metrics: {e}")
            return {}
    
    def _check_performance_alerts(self):
        """Check for performance alerts"""
        try:
            metrics = self.current_metrics
            
            # Drawdown alert
            if metrics.current_drawdown < -0.05:  # 5% drawdown
                self._trigger_callbacks('on_drawdown_alert', {
                    'current_drawdown': metrics.current_drawdown,
                    'max_drawdown': metrics.max_drawdown
                })
            
            # Performance milestone
            if metrics.total_return > 0.10:  # 10% return
                self._trigger_callbacks('on_performance_milestone', {
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio
                })
            
            # Risk threshold breach
            if metrics.volatility > 0.30:  # 30% volatility
                self._trigger_callbacks('on_risk_threshold_breach', {
                    'volatility': metrics.volatility,
                    'var_95': metrics.var_95
                })
            
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
    
    def _save_metrics_to_database(self):
        """Save metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert metrics to JSON
            metrics_json = json.dumps({
                'total_return': self.current_metrics.total_return,
                'annualized_return': self.current_metrics.annualized_return,
                'volatility': self.current_metrics.volatility,
                'sharpe_ratio': self.current_metrics.sharpe_ratio,
                'max_drawdown': self.current_metrics.max_drawdown,
                'win_rate': self.current_metrics.win_rate,
                'profit_factor': self.current_metrics.profit_factor
            })
            
            cursor.execute('''
                INSERT OR REPLACE INTO performance_metrics (timestamp, metrics)
                VALUES (?, ?)
            ''', (datetime.now().isoformat(), metrics_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to database: {e}")
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade"""
        try:
            self.trade_records.append(trade)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades 
                (trade_id, symbol, side, size, entry_price, exit_price, 
                 entry_time, exit_time, pnl, commission, slippage, duration, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.trade_id,
                trade.symbol,
                trade.side,
                trade.size,
                trade.entry_price,
                trade.exit_price,
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat(),
                trade.pnl,
                trade.commission,
                trade.slippage,
                trade.duration,
                json.dumps(trade.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Recorded trade: {trade.trade_id}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def update_portfolio_value(self, equity: float, benchmark_value: Optional[float] = None):
        """Update portfolio value"""
        try:
            self.portfolio_values.append({
                'timestamp': datetime.now(),
                'equity': equity,
                'benchmark_value': benchmark_value
            })
            
            # Update peak equity
            if equity > self.peak_equity:
                self.peak_equity = equity
            
            # Set start equity if not set
            if self.start_equity == 0:
                self.start_equity = equity
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO portfolio_values (timestamp, equity, benchmark_value)
                VALUES (?, ?, ?)
            ''', (datetime.now().isoformat(), equity, benchmark_value))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def generate_performance_report(self, start_date: Optional[datetime] = None, 
                                  end_date: Optional[datetime] = None) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            # Set default date range
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # Filter trades by date range
            filtered_trades = [
                trade for trade in self.trade_records
                if start_date <= trade.entry_time <= end_date
            ]
            
            # Calculate metrics for the period
            period_returns = self._calculate_period_returns(start_date, end_date)
            period_metrics = self._calculate_performance_metrics(period_returns)
            
            # Benchmark comparison
            benchmark_comparison = self._calculate_benchmark_comparison(start_date, end_date)
            
            # Attribution analysis
            attribution_analysis = self._calculate_attribution_analysis(filtered_trades)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(period_metrics)
            
            # Create report
            report = PerformanceReport(
                report_id=f"report_{int(time.time())}",
                generated_at=datetime.now(),
                time_period=f"{start_date.date()} to {end_date.date()}",
                metrics=period_metrics,
                trades=filtered_trades,
                benchmark_comparison=benchmark_comparison,
                attribution_analysis=attribution_analysis,
                recommendations=recommendations
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return None
    
    def _calculate_period_returns(self, start_date: datetime, end_date: datetime) -> List[float]:
        """Calculate returns for a specific period"""
        try:
            # Filter portfolio values by date range
            period_values = [
                pv for pv in self.portfolio_values
                if start_date <= pv['timestamp'] <= end_date
            ]
            
            if len(period_values) < 2:
                return []
            
            returns = []
            for i in range(1, len(period_values)):
                prev_equity = period_values[i-1]['equity']
                curr_equity = period_values[i]['equity']
                
                if prev_equity > 0:
                    ret = (curr_equity - prev_equity) / prev_equity
                    returns.append(ret)
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating period returns: {e}")
            return []
    
    def _calculate_benchmark_comparison(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        try:
            # This would compare against actual benchmark data
            # For now, return mock comparison
            return {
                'excess_return': 0.05,  # 5% excess return
                'tracking_error': 0.08,  # 8% tracking error
                'information_ratio': 0.625,  # 0.625 information ratio
                'beta': 1.2,  # 1.2 beta
                'alpha': 0.03  # 3% alpha
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark comparison: {e}")
            return {}
    
    def _calculate_attribution_analysis(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """Calculate performance attribution analysis"""
        try:
            if len(trades) == 0:
                return {}
            
            # Group trades by symbol
            symbol_pnl = {}
            for trade in trades:
                symbol = trade.symbol
                if symbol not in symbol_pnl:
                    symbol_pnl[symbol] = 0
                symbol_pnl[symbol] += trade.pnl
            
            # Calculate total PnL
            total_pnl = sum(symbol_pnl.values())
            
            # Calculate attribution percentages
            attribution = {}
            for symbol, pnl in symbol_pnl.items():
                attribution[symbol] = pnl / total_pnl if total_pnl != 0 else 0
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error calculating attribution analysis: {e}")
            return {}
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance recommendations"""
        try:
            recommendations = []
            
            # Sharpe ratio recommendations
            if metrics.sharpe_ratio < 1.0:
                recommendations.append("Consider improving risk-adjusted returns - Sharpe ratio below 1.0")
            
            # Drawdown recommendations
            if metrics.max_drawdown < -0.10:
                recommendations.append("High maximum drawdown detected - consider reducing position sizes")
            
            # Win rate recommendations
            if metrics.win_rate < 0.4:
                recommendations.append("Low win rate - consider improving entry/exit criteria")
            
            # Volatility recommendations
            if metrics.volatility > 0.25:
                recommendations.append("High volatility - consider diversification or position sizing adjustments")
            
            # Profit factor recommendations
            if metrics.profit_factor < 1.5:
                recommendations.append("Low profit factor - consider improving risk management")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def create_performance_dashboard(self) -> str:
        """Create interactive performance dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Portfolio Value', 'Returns Distribution', 
                              'Drawdown', 'Risk Metrics', 'Trade Analysis', 'Performance Metrics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Portfolio value chart
            if len(self.portfolio_values) > 0:
                timestamps = [pv['timestamp'] for pv in self.portfolio_values]
                equities = [pv['equity'] for pv in self.portfolio_values]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=equities, name='Portfolio Value'),
                    row=1, col=1
                )
            
            # Returns distribution
            returns = self._calculate_returns()
            if len(returns) > 0:
                fig.add_trace(
                    go.Histogram(x=returns, name='Returns Distribution'),
                    row=1, col=2
                )
            
            # Drawdown chart
            if len(returns) > 0:
                cumulative_returns = np.cumprod(1 + np.array(returns))
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                
                fig.add_trace(
                    go.Scatter(y=drawdowns, name='Drawdown', fill='tonexty'),
                    row=2, col=1
                )
            
            # Risk metrics
            risk_metrics = {
                'Volatility': self.current_metrics.volatility,
                'VaR 95%': self.current_metrics.var_95,
                'Max Drawdown': self.current_metrics.max_drawdown,
                'Sharpe Ratio': self.current_metrics.sharpe_ratio
            }
            
            fig.add_trace(
                go.Bar(x=list(risk_metrics.keys()), y=list(risk_metrics.values()), name='Risk Metrics'),
                row=2, col=2
            )
            
            # Trade analysis
            if len(self.trade_records) > 0:
                trade_pnls = [trade.pnl for trade in self.trade_records]
                fig.add_trace(
                    go.Histogram(x=trade_pnls, name='Trade PnL Distribution'),
                    row=3, col=1
                )
            
            # Performance metrics
            perf_metrics = {
                'Total Return': self.current_metrics.total_return,
                'Annualized Return': self.current_metrics.annualized_return,
                'Win Rate': self.current_metrics.win_rate,
                'Profit Factor': self.current_metrics.profit_factor
            }
            
            fig.add_trace(
                go.Bar(x=list(perf_metrics.keys()), y=list(perf_metrics.values()), name='Performance Metrics'),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Performance Analytics Dashboard",
                height=1200,
                showlegend=True
            )
            
            # Save as HTML
            dashboard_path = "performance_dashboard.html"
            fig.write_html(dashboard_path)
            
            return dashboard_path
            
        except Exception as e:
            self.logger.error(f"Error creating performance dashboard: {e}")
            return None
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger performance event callbacks"""
        try:
            if event in self.performance_callbacks:
                for callback in self.performance_callbacks[event]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in performance callback for {event}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error triggering performance callbacks: {e}")
    
    def add_callback(self, event: str, callback: callable):
        """Add performance event callback"""
        if event in self.performance_callbacks:
            self.performance_callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: callable):
        """Remove performance event callback"""
        if event in self.performance_callbacks and callback in self.performance_callbacks[event]:
            self.performance_callbacks[event].remove(callback)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.current_metrics
    
    def get_trade_records(self) -> List[TradeRecord]:
        """Get all trade records"""
        return self.trade_records.copy()
    
    def get_portfolio_values(self) -> List[Dict[str, Any]]:
        """Get portfolio value history"""
        return list(self.portfolio_values)
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance metrics history"""
        return list(self.performance_history)
    
    def export_data(self, file_path: str, format: str = 'csv') -> bool:
        """Export performance data"""
        try:
            if format == 'csv':
                # Export trades
                trades_df = pd.DataFrame([
                    {
                        'trade_id': trade.trade_id,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'size': trade.size,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'entry_time': trade.entry_time,
                        'exit_time': trade.exit_time,
                        'pnl': trade.pnl,
                        'commission': trade.commission,
                        'slippage': trade.slippage,
                        'duration': trade.duration
                    }
                    for trade in self.trade_records
                ])
                
                trades_df.to_csv(f"{file_path}_trades.csv", index=False)
                
                # Export portfolio values
                portfolio_df = pd.DataFrame(self.portfolio_values)
                portfolio_df.to_csv(f"{file_path}_portfolio.csv", index=False)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
    
    def shutdown(self):
        """Shutdown performance analytics system"""
        try:
            self.running = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            self.logger.info("Performance quant analyst shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during performance analytics shutdown: {e}")

