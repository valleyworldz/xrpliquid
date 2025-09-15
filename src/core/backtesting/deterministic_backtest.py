"""
🎯 DETERMINISTIC BACKTEST ENGINE
Comprehensive backtesting system with deterministic results and detailed tear sheets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.analytics.trade_ledger import TradeLedgerManager, TradeRecord
from core.utils.logger import Logger

class DeterministicBacktestEngine:
    """
    🎯 DETERMINISTIC BACKTEST ENGINE
    Comprehensive backtesting system with reproducible results
    """
    
    def __init__(self, initial_capital: float = 1000.0, logger=None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = logger or Logger()
        
        # Portfolio tracking
        self.portfolio_value = []
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        self.dates = []
        
        # Performance metrics
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.calmar_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        
        # Trade ledger for backtest
        self.trade_ledger = TradeLedgerManager(data_dir="data/backtest", logger=self.logger)
        
        self.logger.info("🎯 [BACKTEST] Deterministic Backtest Engine initialized")
    
    def generate_historical_data(self, 
                                start_date: str = "2024-01-01", 
                                end_date: str = "2024-12-31",
                                symbol: str = "XRP") -> pd.DataFrame:
        """
        Generate deterministic historical data for backtesting
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            symbol: Trading symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Create date range
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            
            # Generate deterministic price data using sine wave with trend
            np.random.seed(42)  # For deterministic results
            n_periods = len(dates)
            
            # Base price trend (upward trend with some volatility)
            base_price = 0.50
            trend = np.linspace(0, 0.3, n_periods)  # 30% upward trend over period
            cycle = 0.1 * np.sin(np.linspace(0, 4*np.pi, n_periods))  # Cyclical component
            noise = 0.02 * np.random.randn(n_periods)  # Random noise
            
            # Generate prices
            prices = base_price + trend + cycle + noise
            prices = np.maximum(prices, 0.01)  # Ensure positive prices
            
            # Generate OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Generate realistic OHLC from base price
                volatility = 0.005  # 0.5% intraday volatility
                high = price * (1 + volatility * abs(np.random.randn()))
                low = price * (1 - volatility * abs(np.random.randn()))
                open_price = price * (1 + 0.001 * np.random.randn())
                close_price = price
                volume = 1000000 + 500000 * abs(np.random.randn())
                
                # Ensure OHLC consistency
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                data.append({
                    'datetime': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume,
                    'symbol': symbol
                })
            
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)
            
            self.logger.info(f"📊 [BACKTEST] Generated {len(df)} historical data points for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ [BACKTEST] Error generating historical data: {e}")
            return pd.DataFrame()
    
    def simulate_trading_strategy(self, 
                                 data: pd.DataFrame,
                                 strategy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simulate trading strategy on historical data
        
        Args:
            data: Historical OHLCV data
            strategy_config: Strategy configuration
            
        Returns:
            List of simulated trades
        """
        try:
            trades = []
            position = 0.0
            entry_price = 0.0
            trade_id = 0
            
            # Strategy parameters
            lookback_period = strategy_config.get('lookback_period', 20)
            entry_threshold = strategy_config.get('entry_threshold', 0.02)
            exit_threshold = strategy_config.get('exit_threshold', 0.05)
            max_position_size = strategy_config.get('max_position_size', 0.1)
            
            # Calculate technical indicators
            data['sma_short'] = data['close'].rolling(window=10).mean()
            data['sma_long'] = data['close'].rolling(window=20).mean()
            data['rsi'] = self._calculate_rsi(data['close'], 14)
            data['bollinger_upper'] = data['close'].rolling(window=20).mean() + 2 * data['close'].rolling(window=20).std()
            data['bollinger_lower'] = data['close'].rolling(window=20).mean() - 2 * data['close'].rolling(window=20).std()
            
            for i in range(lookback_period, len(data)):
                current_price = data['close'].iloc[i]
                current_date = data.index[i]
                
                # Entry conditions - More permissive for testing
                if position == 0.0:  # No position
                    # Buy signal: Simple moving average crossover or RSI oversold
                    sma_cross = data['sma_short'].iloc[i] > data['sma_long'].iloc[i]
                    rsi_oversold = data['rsi'].iloc[i] < 40  # More permissive RSI
                    price_below_bb = current_price < data['bollinger_lower'].iloc[i]
                    
                    # Any of these conditions can trigger a buy
                    if sma_cross or rsi_oversold or price_below_bb:
                        
                        # Calculate position size
                        position_size = min(max_position_size, self.current_capital * 0.1 / current_price)
                        if position_size > 0:
                            position = position_size
                            entry_price = current_price
                            trade_id += 1
                            
                            # Record trade
                            trade = {
                                'trade_id': f"BACKTEST_{trade_id:06d}",
                                'timestamp': current_date.timestamp(),
                                'datetime_utc': current_date.isoformat(),
                                'trade_type': 'BUY',
                                'strategy': 'Deterministic Backtest Strategy',
                                'hat_role': 'Chief Quantitative Strategist',
                                'symbol': 'XRP',
                                'side': 'BUY',
                                'quantity': position_size,
                                'price': current_price,
                                'mark_price': current_price,
                                'order_type': 'MARKET',
                                'order_id': f"BACKTEST_ORDER_{trade_id}",
                                'execution_time': current_date.timestamp(),
                                'slippage': 0.001,
                                'fees_paid': position_size * current_price * 0.001,
                                'position_size_before': 0.0,
                                'position_size_after': position_size,
                                'avg_entry_price': current_price,
                                'unrealized_pnl': 0.0,
                                'realized_pnl': 0.0,
                                'margin_used': position_size * current_price,
                                'margin_ratio': 0.1,
                                'risk_score': 0.5,
                                'stop_loss_price': current_price * 0.95,
                                'take_profit_price': current_price * 1.05,
                                'profit_loss': 0.0,
                                'profit_loss_percent': 0.0,
                                'win_loss': 'BREAKEVEN',
                                'trade_duration': 0.0,
                                'funding_rate': 0.0001,
                                'volatility': data['close'].rolling(window=20).std().iloc[i] / data['close'].iloc[i],
                                'volume_24h': data['volume'].rolling(window=24).sum().iloc[i],
                                'market_regime': 'NORMAL',
                                'system_score': 10.0,
                                'confidence_score': 0.8,
                                'emergency_mode': False,
                                'cycle_count': i,
                                'data_source': 'backtest',
                                'is_live_trade': False,
                                'notes': 'Deterministic Backtest Buy Signal',
                                'tags': ['xrp', 'buy', 'backtest', 'deterministic'],
                                'metadata': {
                                    'rsi': data['rsi'].iloc[i],
                                    'bollinger_position': (current_price - data['bollinger_lower'].iloc[i]) / (data['bollinger_upper'].iloc[i] - data['bollinger_lower'].iloc[i]),
                                    'sma_cross': data['sma_short'].iloc[i] - data['sma_long'].iloc[i]
                                }
                            }
                            
                            trades.append(trade)
                            self.trade_ledger.record_trade(trade)
                
                # Exit conditions - More permissive for testing
                elif position > 0.0:  # Long position
                    # Sell signal: RSI overbought, price above upper Bollinger Band, or profit/loss targets
                    rsi_overbought = data['rsi'].iloc[i] > 60  # More permissive RSI
                    price_above_bb = current_price > data['bollinger_upper'].iloc[i]
                    profit_target = current_price >= entry_price * (1 + exit_threshold)
                    loss_target = current_price <= entry_price * (1 - exit_threshold)
                    
                    # Any of these conditions can trigger a sell
                    if rsi_overbought or price_above_bb or profit_target or loss_target:
                        
                        # Calculate PnL
                        pnl = (current_price - entry_price) * position
                        pnl_percent = (current_price - entry_price) / entry_price
                        win_loss = 'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'BREAKEVEN'
                        
                        # Record exit trade
                        trade_id += 1
                        exit_trade = {
                            'trade_id': f"BACKTEST_{trade_id:06d}",
                            'timestamp': current_date.timestamp(),
                            'datetime_utc': current_date.isoformat(),
                            'trade_type': 'SELL',
                            'strategy': 'Deterministic Backtest Strategy',
                            'hat_role': 'Chief Quantitative Strategist',
                            'symbol': 'XRP',
                            'side': 'SELL',
                            'quantity': position,
                            'price': current_price,
                            'mark_price': current_price,
                            'order_type': 'MARKET',
                            'order_id': f"BACKTEST_ORDER_{trade_id}",
                            'execution_time': current_date.timestamp(),
                            'slippage': 0.001,
                            'fees_paid': position * current_price * 0.001,
                            'position_size_before': position,
                            'position_size_after': 0.0,
                            'avg_entry_price': entry_price,
                            'unrealized_pnl': 0.0,
                            'realized_pnl': pnl,
                            'margin_used': 0.0,
                            'margin_ratio': 0.0,
                            'risk_score': 0.5,
                            'stop_loss_price': None,
                            'take_profit_price': None,
                            'profit_loss': pnl,
                            'profit_loss_percent': pnl_percent * 100,
                            'win_loss': win_loss,
                            'trade_duration': 0.0,  # Will be calculated
                            'funding_rate': 0.0001,
                            'volatility': data['close'].rolling(window=20).std().iloc[i] / data['close'].iloc[i],
                            'volume_24h': data['volume'].rolling(window=24).sum().iloc[i],
                            'market_regime': 'NORMAL',
                            'system_score': 10.0,
                            'confidence_score': 0.8,
                            'emergency_mode': False,
                            'cycle_count': i,
                            'data_source': 'backtest',
                            'is_live_trade': False,
                            'notes': 'Deterministic Backtest Sell Signal',
                            'tags': ['xrp', 'sell', 'backtest', 'deterministic'],
                            'metadata': {
                                'rsi': data['rsi'].iloc[i],
                                'bollinger_position': (current_price - data['bollinger_lower'].iloc[i]) / (data['bollinger_upper'].iloc[i] - data['bollinger_lower'].iloc[i]),
                                'entry_price': entry_price,
                                'hold_duration': 0  # Will be calculated
                            }
                        }
                        
                        trades.append(exit_trade)
                        self.trade_ledger.record_trade(exit_trade)
                        
                        # Update capital
                        self.current_capital += pnl
                        
                        # Reset position
                        position = 0.0
                        entry_price = 0.0
                
                # Track portfolio value
                portfolio_value = self.current_capital + (position * current_price if position > 0 else 0)
                self.portfolio_value.append(portfolio_value)
                self.dates.append(current_date)
                
                # Calculate daily returns
                if len(self.portfolio_value) > 1:
                    daily_return = (portfolio_value - self.portfolio_value[-2]) / self.portfolio_value[-2]
                    self.daily_returns.append(daily_return)
            
            self.trades = trades
            self.logger.info(f"📊 [BACKTEST] Generated {len(trades)} trades from backtest")
            return trades
            
        except Exception as e:
            self.logger.error(f"❌ [BACKTEST] Error simulating trading strategy: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.portfolio_value or len(self.portfolio_value) < 2:
                return {}
            
            # Convert to numpy arrays
            portfolio_values = np.array(self.portfolio_value)
            returns = np.array(self.daily_returns) if self.daily_returns else np.array([])
            
            # Basic metrics
            self.total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            
            # Annualized return (assuming daily data)
            if len(portfolio_values) > 1:
                days = len(portfolio_values)
                self.annualized_return = (1 + self.total_return) ** (365 / days) - 1
            
            # Volatility
            if len(returns) > 1:
                self.volatility = np.std(returns) * np.sqrt(365)  # Annualized
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            if self.volatility > 0:
                self.sharpe_ratio = (self.annualized_return - risk_free_rate) / self.volatility
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            self.max_drawdown = np.min(drawdown)
            
            # Calmar ratio
            if abs(self.max_drawdown) > 0:
                self.calmar_ratio = self.annualized_return / abs(self.max_drawdown)
            
            # Win rate and profit factor from trades
            if self.trades:
                winning_trades = [t for t in self.trades if t.get('win_loss') == 'WIN']
                losing_trades = [t for t in self.trades if t.get('win_loss') == 'LOSS']
                
                self.win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
                
                total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
                total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
                
                self.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            metrics = {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'volatility': self.volatility,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'calmar_ratio': self.calmar_ratio,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'total_trades': len(self.trades),
                'winning_trades': len([t for t in self.trades if t.get('win_loss') == 'WIN']),
                'losing_trades': len([t for t in self.trades if t.get('win_loss') == 'LOSS']),
                'initial_capital': self.initial_capital,
                'final_capital': portfolio_values[-1] if len(portfolio_values) > 0 else self.initial_capital,
                'total_pnl': portfolio_values[-1] - self.initial_capital if len(portfolio_values) > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ [BACKTEST] Error calculating performance metrics: {e}")
            return {}
    
    def generate_tear_sheet(self, 
                           output_dir: str = "reports",
                           filename: str = None) -> str:
        """
        Generate comprehensive tear sheet with charts and analysis
        
        Args:
            output_dir: Output directory for tear sheet
            filename: Custom filename (optional)
            
        Returns:
            Path to generated tear sheet
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_tear_sheet_{timestamp}.html"
            
            filepath = output_path / filename
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics()
            
            # Generate HTML tear sheet
            html_content = self._generate_html_tear_sheet(metrics)
            
            # Save tear sheet
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save metrics as JSON
            metrics_file = output_path / f"backtest_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            self.logger.info(f"📊 [BACKTEST] Tear sheet generated: {filepath}")
            self.logger.info(f"📊 [BACKTEST] Metrics saved: {metrics_file}")
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"❌ [BACKTEST] Error generating tear sheet: {e}")
            return ""
    
    def _generate_html_tear_sheet(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML content for tear sheet"""
        
        # Create portfolio value chart
        portfolio_chart = self._create_portfolio_chart()
        
        # Create returns distribution chart
        returns_chart = self._create_returns_chart()
        
        # Create drawdown chart
        drawdown_chart = self._create_drawdown_chart()
        
        # Create trade analysis chart
        trade_chart = self._create_trade_analysis_chart()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultra-Efficient XRP Trading System - Backtest Tear Sheet</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-section {{
            padding: 30px;
            border-top: 1px solid #eee;
        }}
        .chart-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .summary-table th,
        .summary-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .summary-table th {{
            background-color: #667eea;
            color: white;
        }}
        .summary-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .footer {{
            background: #333;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .positive {{
            color: #28a745;
        }}
        .negative {{
            color: #dc3545;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Ultra-Efficient XRP Trading System</h1>
            <p>Deterministic Backtest Tear Sheet</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.get('total_return', 0) >= 0 else 'negative'}">
                    {metrics.get('total_return', 0):.2%}
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.get('annualized_return', 0) >= 0 else 'negative'}">
                    {metrics.get('annualized_return', 0):.2%}
                </div>
                <div class="metric-label">Annualized Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {metrics.get('sharpe_ratio', 0):.2f}
                </div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">
                    {metrics.get('max_drawdown', 0):.2%}
                </div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {metrics.get('win_rate', 0):.1f}%
                </div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {metrics.get('profit_factor', 0):.2f}
                </div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {metrics.get('total_trades', 0)}
                </div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {metrics.get('volatility', 0):.2%}
                </div>
                <div class="metric-label">Volatility</div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">📈 Portfolio Performance</h2>
            <div class="chart-container">
                {portfolio_chart}
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">📊 Returns Distribution</h2>
            <div class="chart-container">
                {returns_chart}
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">📉 Drawdown Analysis</h2>
            <div class="chart-container">
                {drawdown_chart}
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">🎯 Trade Analysis</h2>
            <div class="chart-container">
                {trade_chart}
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">📋 Performance Summary</h2>
            <table class="summary-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Initial Capital</td>
                    <td>${metrics.get('initial_capital', 0):,.2f}</td>
                    <td>Starting portfolio value</td>
                </tr>
                <tr>
                    <td>Final Capital</td>
                    <td>${metrics.get('final_capital', 0):,.2f}</td>
                    <td>Ending portfolio value</td>
                </tr>
                <tr>
                    <td>Total PnL</td>
                    <td class="{'positive' if metrics.get('total_pnl', 0) >= 0 else 'negative'}">
                        ${metrics.get('total_pnl', 0):,.2f}
                    </td>
                    <td>Total profit/loss</td>
                </tr>
                <tr>
                    <td>Winning Trades</td>
                    <td>{metrics.get('winning_trades', 0)}</td>
                    <td>Number of profitable trades</td>
                </tr>
                <tr>
                    <td>Losing Trades</td>
                    <td>{metrics.get('losing_trades', 0)}</td>
                    <td>Number of losing trades</td>
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{metrics.get('calmar_ratio', 0):.2f}</td>
                    <td>Annual return / Max drawdown</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>🎯 Ultra-Efficient XRP Trading System - Deterministic Backtest Engine</p>
            <p>Generated with comprehensive trade ledger and analytics</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _create_portfolio_chart(self) -> str:
        """Create portfolio value chart"""
        try:
            if not self.portfolio_value or not self.dates:
                return "<p>No portfolio data available</p>"
            
            # Create simple ASCII chart representation
            values = self.portfolio_value
            dates = self.dates
            
            # Find min/max for scaling
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val
            
            if range_val == 0:
                return "<p>No portfolio variation</p>"
            
            # Create simple chart
            chart_lines = []
            chart_lines.append("Portfolio Value Over Time:")
            chart_lines.append("=" * 50)
            
            # Sample points for display
            step = max(1, len(values) // 20)
            for i in range(0, len(values), step):
                date_str = dates[i].strftime('%Y-%m-%d') if hasattr(dates[i], 'strftime') else str(dates[i])
                value = values[i]
                bar_length = int((value - min_val) / range_val * 30)
                bar = "█" * bar_length
                chart_lines.append(f"{date_str}: ${value:,.2f} {bar}")
            
            return f"<pre>{chr(10).join(chart_lines)}</pre>"
            
        except Exception as e:
            return f"<p>Error creating portfolio chart: {e}</p>"
    
    def _create_returns_chart(self) -> str:
        """Create returns distribution chart"""
        try:
            if not self.daily_returns:
                return "<p>No returns data available</p>"
            
            returns = np.array(self.daily_returns)
            
            # Create histogram
            hist, bins = np.histogram(returns, bins=20)
            max_hist = max(hist) if len(hist) > 0 else 1
            
            chart_lines = []
            chart_lines.append("Returns Distribution:")
            chart_lines.append("=" * 40)
            
            for i, (count, bin_edge) in enumerate(zip(hist, bins[:-1])):
                bar_length = int(count / max_hist * 20)
                bar = "█" * bar_length
                chart_lines.append(f"{bin_edge:.3f}: {count:3d} {bar}")
            
            return f"<pre>{chr(10).join(chart_lines)}</pre>"
            
        except Exception as e:
            return f"<p>Error creating returns chart: {e}</p>"
    
    def _create_drawdown_chart(self) -> str:
        """Create drawdown chart"""
        try:
            if not self.portfolio_value:
                return "<p>No portfolio data available</p>"
            
            values = np.array(self.portfolio_value)
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            
            chart_lines = []
            chart_lines.append("Drawdown Analysis:")
            chart_lines.append("=" * 30)
            chart_lines.append(f"Maximum Drawdown: {np.min(drawdown):.2%}")
            chart_lines.append(f"Current Drawdown: {drawdown[-1]:.2%}")
            
            # Find worst drawdown period
            worst_idx = np.argmin(drawdown)
            chart_lines.append(f"Worst Drawdown Date: {self.dates[worst_idx] if worst_idx < len(self.dates) else 'N/A'}")
            
            return f"<pre>{chr(10).join(chart_lines)}</pre>"
            
        except Exception as e:
            return f"<p>Error creating drawdown chart: {e}</p>"
    
    def _create_trade_analysis_chart(self) -> str:
        """Create trade analysis chart"""
        try:
            if not self.trades:
                return "<p>No trades available</p>"
            
            # Analyze trades
            winning_trades = [t for t in self.trades if t.get('win_loss') == 'WIN']
            losing_trades = [t for t in self.trades if t.get('win_loss') == 'LOSS']
            
            avg_win = np.mean([t.get('profit_loss', 0) for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.get('profit_loss', 0) for t in losing_trades]) if losing_trades else 0
            
            chart_lines = []
            chart_lines.append("Trade Analysis:")
            chart_lines.append("=" * 20)
            chart_lines.append(f"Total Trades: {len(self.trades)}")
            chart_lines.append(f"Winning Trades: {len(winning_trades)}")
            chart_lines.append(f"Losing Trades: {len(losing_trades)}")
            chart_lines.append(f"Average Win: ${avg_win:.2f}")
            chart_lines.append(f"Average Loss: ${avg_loss:.2f}")
            chart_lines.append(f"Win Rate: {len(winning_trades)/len(self.trades)*100:.1f}%")
            
            return f"<pre>{chr(10).join(chart_lines)}</pre>"
            
        except Exception as e:
            return f"<p>Error creating trade analysis chart: {e}</p>"
    
    def run_backtest(self, 
                    start_date: str = "2024-01-01",
                    end_date: str = "2024-12-31",
                    strategy_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run complete backtest and generate tear sheet
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_config: Strategy configuration
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info("🚀 [BACKTEST] Starting deterministic backtest")
            
            # Default strategy config
            if not strategy_config:
                strategy_config = {
                    'lookback_period': 20,
                    'entry_threshold': 0.02,
                    'exit_threshold': 0.05,
                    'max_position_size': 0.1
                }
            
            # Generate historical data
            self.logger.info("📊 [BACKTEST] Generating historical data")
            data = self.generate_historical_data(start_date, end_date)
            
            if data.empty:
                self.logger.error("❌ [BACKTEST] Failed to generate historical data")
                return {}
            
            # Simulate trading strategy
            self.logger.info("🎯 [BACKTEST] Simulating trading strategy")
            trades = self.simulate_trading_strategy(data, strategy_config)
            
            # Calculate performance metrics
            self.logger.info("📈 [BACKTEST] Calculating performance metrics")
            metrics = self.calculate_performance_metrics()
            
            # Generate tear sheet
            self.logger.info("📊 [BACKTEST] Generating tear sheet")
            tear_sheet_path = self.generate_tear_sheet()
            
            # Save trade ledger
            self.trade_ledger.save_to_parquet()
            self.trade_ledger.save_to_csv()
            
            results = {
                'success': True,
                'tear_sheet_path': tear_sheet_path,
                'metrics': metrics,
                'trades': trades,
                'data_points': len(data),
                'backtest_period': f"{start_date} to {end_date}",
                'strategy_config': strategy_config
            }
            
            self.logger.info("✅ [BACKTEST] Backtest completed successfully")
            self.logger.info(f"📊 [BACKTEST] Total Return: {metrics.get('total_return', 0):.2%}")
            self.logger.info(f"📊 [BACKTEST] Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"📊 [BACKTEST] Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            self.logger.info(f"📊 [BACKTEST] Win Rate: {metrics.get('win_rate', 0):.1f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ [BACKTEST] Error running backtest: {e}")
            return {'success': False, 'error': str(e)}
