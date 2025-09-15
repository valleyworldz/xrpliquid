"""
🎯 COMPREHENSIVE DETERMINISTIC BACKTEST ENGINE
=============================================
12-36 month XRP-perp backtests with realistic market conditions
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class BacktestConfig:
    """Configuration for comprehensive backtesting"""
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 10000.0
    commission_rate: float = 0.0005  # 0.05% per trade
    slippage_bps: float = 2.0  # 2 basis points
    spread_bps: float = 1.0  # 1 basis point spread
    funding_frequency_hours: int = 8  # Hyperliquid funding every 8 hours
    volatility_regime_threshold: float = 0.02  # 2% daily vol threshold
    min_position_size_usd: float = 25.0
    max_position_size_usd: float = 1000.0
    risk_unit_size: float = 0.01  # 1% of capital per risk unit

@dataclass
class TradeRecord:
    """Canonical trade record schema"""
    timestamp: datetime
    strategy: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    expected_price: float
    fill_price: float
    fee: float
    funding: float
    slippage_bps: float
    pnl_realized: float
    pnl_unrealized: float
    reason_code: str
    maker_flag: bool
    queue_jump: bool = False
    market_regime: str = "normal"
    volatility_percent: float = 0.0

class ComprehensiveBacktestEngine:
    """Comprehensive backtesting engine with realistic market simulation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.funding_payments: List[float] = []
        self.current_capital = config.initial_capital
        self.current_position = 0.0
        self.current_price = 0.0
        self.peak_capital = config.initial_capital
        self.max_drawdown = 0.0
        self.regime_stats = {"bull": [], "bear": [], "chop": []}
        self.vol_terciles = {"low": [], "medium": [], "high": []}
        
    def generate_realistic_market_data(self) -> pd.DataFrame:
        """Generate realistic XRP market data with regime changes"""
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        dates = pd.date_range(start, end, freq='1H')
        
        # Generate price data with realistic characteristics
        np.random.seed(42)  # For reproducibility
        
        # Base price trend with regime changes
        n_periods = len(dates)
        base_price = 0.52
        
        # Create regime periods
        regime_length = n_periods // 6  # 6 regime changes
        prices = []
        volatilities = []
        regimes = []
        
        for i in range(0, n_periods, regime_length):
            regime_end = min(i + regime_length, n_periods)
            regime_type = np.random.choice(['bull', 'bear', 'chop'])
            
            if regime_type == 'bull':
                trend = np.linspace(0, 0.3, regime_end - i)  # 30% uptrend
                vol = 0.02  # 2% volatility
            elif regime_type == 'bear':
                trend = np.linspace(0, -0.2, regime_end - i)  # 20% downtrend
                vol = 0.025  # 2.5% volatility
            else:  # chop
                trend = np.linspace(0, 0.05, regime_end - i)  # 5% uptrend
                vol = 0.015  # 1.5% volatility
            
            # Generate price movements
            regime_prices = []
            regime_vols = []
            regime_labels = []
            
            for j in range(regime_end - i):
                if j == 0:
                    price = base_price
                else:
                    # Add trend and random walk
                    price_change = trend[j] / (regime_end - i) + np.random.normal(0, vol)
                    price = regime_prices[-1] * (1 + price_change)
                
                regime_prices.append(price)
                regime_vols.append(vol)
                regime_labels.append(regime_type)
            
            prices.extend(regime_prices)
            volatilities.extend(regime_vols)
            regimes.extend(regime_labels)
            base_price = regime_prices[-1]
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates[:len(prices)],
            'price': prices,
            'volatility': volatilities,
            'regime': regimes
        })
        
        # Add funding rates (realistic for XRP)
        df['funding_rate'] = np.random.normal(0.0001, 0.0002, len(df))
        df['funding_rate'] = np.clip(df['funding_rate'], -0.01, 0.01)  # Cap at 1%
        
        # Add spread
        df['spread'] = df['price'] * self.config.spread_bps / 10000
        
        return df
    
    def calculate_volatility_terciles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility terciles for regime analysis"""
        # Add small random noise to avoid duplicate edges
        df['volatility_noise'] = df['volatility'] + np.random.normal(0, 0.0001, len(df))
        df['vol_tercile'] = pd.qcut(df['volatility_noise'], 3, labels=['low', 'medium', 'high'], duplicates='drop')
        df.drop('volatility_noise', axis=1, inplace=True)
        return df
    
    def simulate_funding_arbitrage_strategy(self, df: pd.DataFrame) -> None:
        """Simulate optimized funding arbitrage strategy"""
        self.current_price = df['price'].iloc[0]
        
        for i, row in df.iterrows():
            self.current_price = row['price']
            current_funding = row['funding_rate']
            
            # Strategy logic: Enter when funding rate is high
            funding_threshold = 0.0008  # 0.08% threshold
            
            if abs(current_funding) > funding_threshold:
                # Determine trade direction
                if current_funding > 0:
                    # Positive funding: short XRP (receive funding)
                    side = 'sell'
                    expected_return = current_funding
                else:
                    # Negative funding: long XRP (pay funding)
                    side = 'buy'
                    expected_return = -current_funding
                
                # Calculate position size using risk unit sizing
                position_size_usd = self._calculate_risk_unit_size(row['volatility'])
                
                if position_size_usd >= self.config.min_position_size_usd:
                    # Execute trade
                    self._execute_trade(
                        timestamp=row['timestamp'],
                        strategy="funding_arbitrage",
                        side=side,
                        quantity=position_size_usd / row['price'],
                        expected_price=row['price'],
                        funding=current_funding,
                        market_regime=row['regime'],
                        volatility_percent=row['volatility'] * 100
                    )
            
            # Update unrealized PnL
            self._update_unrealized_pnl()
            
            # Record equity
            self.equity_curve.append(self.current_capital)
            
            # Calculate daily returns
            if len(self.equity_curve) > 24:  # Daily
                daily_return = (self.equity_curve[-1] - self.equity_curve[-25]) / self.equity_curve[-25]
                self.daily_returns.append(daily_return)
    
    def simulate_optimized_funding_arbitrage_strategy(self, df: pd.DataFrame) -> None:
        """Simulate OPTIMIZED funding arbitrage strategy with better filtering"""
        self.current_price = df['price'].iloc[0]
        
        for i, row in df.iterrows():
            self.current_price = row['price']
            current_funding = row['funding_rate']
            
            # OPTIMIZED Strategy logic with better filtering
            funding_threshold = 0.0012  # Higher threshold: 0.12%
            
            # Additional filters for profitability
            volatility_filter = row['volatility'] < 0.03  # Avoid high volatility periods
            regime_filter = row['regime'] in ['bull', 'chop']  # Avoid bear markets
            spread_filter = row['spread'] < row['price'] * 0.001  # Avoid wide spreads
            
            if (abs(current_funding) > funding_threshold and 
                volatility_filter and regime_filter and spread_filter):
                
                # Determine trade direction
                if current_funding > 0:
                    # Positive funding: short XRP (receive funding)
                    side = 'sell'
                    expected_return = current_funding
                else:
                    # Negative funding: long XRP (pay funding)
                    side = 'buy'
                    expected_return = -current_funding
                
                # Calculate position size using risk unit sizing
                position_size_usd = self._calculate_risk_unit_size(row['volatility'])
                
                # Additional position size filters
                if (position_size_usd >= self.config.min_position_size_usd and
                    position_size_usd <= self.config.max_position_size_usd):
                    
                    # Execute trade
                    self._execute_trade(
                        timestamp=row['timestamp'],
                        strategy="optimized_funding_arbitrage",
                        side=side,
                        quantity=position_size_usd / row['price'],
                        expected_price=row['price'],
                        funding=current_funding,
                        market_regime=row['regime'],
                        volatility_percent=row['volatility'] * 100
                    )
            
            # Update unrealized PnL
            self._update_unrealized_pnl()
            
            # Record equity
            self.equity_curve.append(self.current_capital)
            
            # Calculate daily returns
            if len(self.equity_curve) > 24:  # Daily
                daily_return = (self.equity_curve[-1] - self.equity_curve[-25]) / self.equity_curve[-25]
                self.daily_returns.append(daily_return)
    
    def _calculate_risk_unit_size(self, volatility: float) -> float:
        """Calculate position size using risk unit sizing"""
        # Risk unit sizing based on volatility
        risk_per_unit = self.current_capital * self.config.risk_unit_size
        vol_adjusted_size = risk_per_unit / (volatility * 2)  # 2x volatility buffer
        
        return min(vol_adjusted_size, self.config.max_position_size_usd)
    
    def _execute_trade(self, timestamp: datetime, strategy: str, side: str, 
                      quantity: float, expected_price: float, funding: float,
                      market_regime: str, volatility_percent: float) -> None:
        """Execute a trade with realistic execution simulation"""
        
        # Calculate slippage based on volatility and market regime
        base_slippage = self.config.slippage_bps
        vol_multiplier = 1 + (volatility_percent / 100) * 2  # Higher vol = more slippage
        regime_multiplier = 1.5 if market_regime == 'bear' else 1.0  # More slippage in bear markets
        
        slippage_bps = base_slippage * vol_multiplier * regime_multiplier
        
        # Calculate fill price
        if side == 'buy':
            fill_price = expected_price * (1 + slippage_bps / 10000)
        else:
            fill_price = expected_price * (1 - slippage_bps / 10000)
        
        # Calculate fees
        notional = quantity * fill_price
        fee = notional * self.config.commission_rate
        
        # Determine if maker (simplified)
        maker_flag = np.random.random() > 0.3  # 70% maker rate
        
        # Calculate PnL
        if side == 'buy':
            self.current_position += quantity
            pnl_realized = 0.0  # Will be realized on exit
        else:
            self.current_position -= quantity
            pnl_realized = quantity * (fill_price - self.current_price)
        
        # Create trade record
        trade = TradeRecord(
            timestamp=timestamp,
            strategy=strategy,
            side=side,
            quantity=quantity,
            price=expected_price,
            expected_price=expected_price,
            fill_price=fill_price,
            fee=fee,
            funding=funding,
            slippage_bps=slippage_bps,
            pnl_realized=pnl_realized,
            pnl_unrealized=0.0,
            reason_code="funding_arbitrage",
            maker_flag=maker_flag,
            market_regime=market_regime,
            volatility_percent=volatility_percent
        )
        
        self.trades.append(trade)
        
        # Update capital
        self.current_capital -= fee
        self.current_capital += pnl_realized
        
        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized PnL for current position"""
        if self.current_position != 0:
            unrealized_pnl = self.current_position * (self.current_price - self.current_price)
            # Update last trade's unrealized PnL
            if self.trades:
                self.trades[-1].pnl_unrealized = unrealized_pnl
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        # Calculate key metrics
        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital
        annualized_return = (1 + total_return) ** (365 / len(self.equity_curve) * 24) - 1
        
        # Sharpe ratio
        if len(self.daily_returns) > 1:
            sharpe_ratio = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(365)
        else:
            sharpe_ratio = 0.0
        
        # MAR ratio
        mar_ratio = annualized_return / self.max_drawdown if self.max_drawdown > 0 else 0.0
        
        # Win rate
        winning_trades = [t for t in self.trades if t.pnl_realized > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl_realized for t in self.trades if t.pnl_realized > 0)
        gross_loss = abs(sum(t.pnl_realized for t in self.trades if t.pnl_realized < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Regime analysis
        regime_performance = self._analyze_regime_performance()
        
        # Volatility tercile analysis
        vol_tercile_performance = self._analyze_vol_tercile_performance()
        
        # Execution quality analysis
        execution_quality = self._analyze_execution_quality()
        
        return {
            "summary": {
                "initial_capital": self.config.initial_capital,
                "final_capital": self.current_capital,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "max_drawdown": self.max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "mar_ratio": mar_ratio,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": len(self.trades),
                "winning_trades": len(winning_trades)
            },
            "regime_analysis": regime_performance,
            "volatility_analysis": vol_tercile_performance,
            "execution_quality": execution_quality,
            "risk_metrics": {
                "max_drawdown": self.max_drawdown,
                "time_under_water": self._calculate_time_under_water(),
                "var_95": self._calculate_var_95(),
                "expected_shortfall": self._calculate_expected_shortfall()
            }
        }
    
    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        regime_stats = {"bull": [], "bear": [], "chop": []}
        
        for trade in self.trades:
            regime_stats[trade.market_regime].append(trade.pnl_realized)
        
        regime_performance = {}
        for regime, pnls in regime_stats.items():
            if pnls:
                regime_performance[regime] = {
                    "trades": len(pnls),
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls)
                }
            else:
                regime_performance[regime] = {
                    "trades": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "win_rate": 0.0
                }
        
        return regime_performance
    
    def _analyze_vol_tercile_performance(self) -> Dict[str, Any]:
        """Analyze performance by volatility tercile"""
        vol_stats = {"low": [], "medium": [], "high": []}
        
        for trade in self.trades:
            if trade.volatility_percent < 1.5:
                vol_stats["low"].append(trade.pnl_realized)
            elif trade.volatility_percent < 2.5:
                vol_stats["medium"].append(trade.pnl_realized)
            else:
                vol_stats["high"].append(trade.pnl_realized)
        
        vol_performance = {}
        for vol_level, pnls in vol_stats.items():
            if pnls:
                vol_performance[vol_level] = {
                    "trades": len(pnls),
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls)
                }
            else:
                vol_performance[vol_level] = {
                    "trades": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "win_rate": 0.0
                }
        
        return vol_performance
    
    def _analyze_execution_quality(self) -> Dict[str, Any]:
        """Analyze execution quality metrics"""
        if not self.trades:
            return {}
        
        slippages = [t.slippage_bps for t in self.trades]
        maker_ratio = len([t for t in self.trades if t.maker_flag]) / len(self.trades)
        
        return {
            "avg_slippage_bps": np.mean(slippages),
            "max_slippage_bps": np.max(slippages),
            "maker_ratio": maker_ratio,
            "total_fees": sum(t.fee for t in self.trades),
            "avg_fee_per_trade": np.mean([t.fee for t in self.trades])
        }
    
    def _calculate_time_under_water(self) -> float:
        """Calculate time under water (drawdown duration)"""
        if not self.equity_curve:
            return 0.0
        
        peak = self.equity_curve[0]
        underwater_periods = 0
        total_periods = len(self.equity_curve)
        
        for equity in self.equity_curve:
            if equity < peak:
                underwater_periods += 1
            else:
                peak = equity
        
        return underwater_periods / total_periods
    
    def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk"""
        if len(self.daily_returns) < 20:
            return 0.0
        return np.percentile(self.daily_returns, 5)
    
    def _calculate_expected_shortfall(self) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(self.daily_returns) < 20:
            return 0.0
        var_95 = self._calculate_var_95()
        return np.mean([r for r in self.daily_returns if r <= var_95])
    
    def save_trade_ledger(self, filename: str) -> None:
        """Save trade ledger to CSV/Parquet"""
        if not self.trades:
            return
        
        # Convert to DataFrame
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'timestamp': trade.timestamp,
                'strategy': trade.strategy,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'expected_price': trade.expected_price,
                'fill_price': trade.fill_price,
                'fee': trade.fee,
                'funding': trade.funding,
                'slippage_bps': trade.slippage_bps,
                'pnl_realized': trade.pnl_realized,
                'pnl_unrealized': trade.pnl_unrealized,
                'reason_code': trade.reason_code,
                'maker_flag': trade.maker_flag,
                'queue_jump': trade.queue_jump,
                'market_regime': trade.market_regime,
                'volatility_percent': trade.volatility_percent
            })
        
        df = pd.DataFrame(trade_data)
        
        # Save as CSV
        csv_path = f"reports/{filename}.csv"
        os.makedirs("reports", exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        # Save as Parquet
        parquet_path = f"reports/{filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        print(f"✅ Trade ledger saved to {csv_path} and {parquet_path}")
    
    def generate_equity_curve_plot(self, filename: str) -> None:
        """Generate equity curve visualization"""
        if not self.equity_curve:
            return
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve - XRP Funding Arbitrage Strategy')
        plt.xlabel('Time (Hours)')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        
        # Add drawdown visualization
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        plt.title('Drawdown (%)')
        plt.xlabel('Time (Hours)')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Save plots
        os.makedirs("reports", exist_ok=True)
        plt.savefig(f"reports/{filename}_equity_curve.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"reports/{filename}_drawdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Equity curve plots saved to reports/{filename}_*.png")
    
    def save_performance_report(self, filename: str) -> None:
        """Save comprehensive performance report"""
        report = self.generate_performance_report()
        
        # Save as JSON
        json_path = f"reports/{filename}_performance.json"
        os.makedirs("reports", exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save as HTML tear sheet
        html_path = f"reports/{filename}_tear_sheet.html"
        self._generate_html_tear_sheet(report, html_path)
        
        print(f"✅ Performance report saved to {json_path}")
        print(f"✅ HTML tear sheet saved to {html_path}")
    
    def _generate_html_tear_sheet(self, report: Dict[str, Any], filename: str) -> None:
        """Generate HTML tear sheet"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>XRP Funding Arbitrage - Performance Tear Sheet</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 XRP Funding Arbitrage Strategy - Performance Tear Sheet</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📊 Key Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Initial Capital</td><td>${report['summary']['initial_capital']:,.2f}</td></tr>
                <tr><td>Final Capital</td><td>${report['summary']['final_capital']:,.2f}</td></tr>
                <tr><td>Total Return</td><td class="{'positive' if report['summary']['total_return'] > 0 else 'negative'}">{report['summary']['total_return']:.2%}</td></tr>
                <tr><td>Annualized Return</td><td class="{'positive' if report['summary']['annualized_return'] > 0 else 'negative'}">{report['summary']['annualized_return']:.2%}</td></tr>
                <tr><td>Max Drawdown</td><td class="negative">{report['summary']['max_drawdown']:.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{report['summary']['sharpe_ratio']:.2f}</td></tr>
                <tr><td>MAR Ratio</td><td>{report['summary']['mar_ratio']:.2f}</td></tr>
                <tr><td>Win Rate</td><td>{report['summary']['win_rate']:.2%}</td></tr>
                <tr><td>Profit Factor</td><td>{report['summary']['profit_factor']:.2f}</td></tr>
                <tr><td>Total Trades</td><td>{report['summary']['total_trades']}</td></tr>
            </table>
            
            <h2>🎯 Regime Analysis</h2>
            <table>
                <tr><th>Regime</th><th>Trades</th><th>Total PnL</th><th>Avg PnL</th><th>Win Rate</th></tr>
                <tr><td>Bull</td><td>{report['regime_analysis']['bull']['trades']}</td><td>${report['regime_analysis']['bull']['total_pnl']:.2f}</td><td>${report['regime_analysis']['bull']['avg_pnl']:.2f}</td><td>{report['regime_analysis']['bull']['win_rate']:.2%}</td></tr>
                <tr><td>Bear</td><td>{report['regime_analysis']['bear']['trades']}</td><td>${report['regime_analysis']['bear']['total_pnl']:.2f}</td><td>${report['regime_analysis']['bear']['avg_pnl']:.2f}</td><td>{report['regime_analysis']['bear']['win_rate']:.2%}</td></tr>
                <tr><td>Chop</td><td>{report['regime_analysis']['chop']['trades']}</td><td>${report['regime_analysis']['chop']['total_pnl']:.2f}</td><td>${report['regime_analysis']['chop']['avg_pnl']:.2f}</td><td>{report['regime_analysis']['chop']['win_rate']:.2%}</td></tr>
            </table>
            
            <h2>⚡ Execution Quality</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Avg Slippage</td><td>{report['execution_quality']['avg_slippage_bps']:.2f} bps</td></tr>
                <tr><td>Max Slippage</td><td>{report['execution_quality']['max_slippage_bps']:.2f} bps</td></tr>
                <tr><td>Maker Ratio</td><td>{report['execution_quality']['maker_ratio']:.2%}</td></tr>
                <tr><td>Total Fees</td><td>${report['execution_quality']['total_fees']:.2f}</td></tr>
            </table>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
