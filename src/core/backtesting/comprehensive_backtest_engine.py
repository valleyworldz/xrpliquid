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
    funding_frequency_hours: int = 1  # Hyperliquid funding every 1 hour
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
    order_state: str = "filled"
    cloid: str = ""

class ComprehensiveBacktestEngine:
    """Comprehensive backtest engine with realistic market conditions"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[TradeRecord] = []
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.position = 0.0
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self.slippage_cost = 0.0
        
        # Performance tracking
        self.equity_curve = []
        self.drawdown_curve = []
        self.peak_equity = config.initial_capital
        
        logger.info(f"🎯 [BACKTEST] Engine initialized with ${config.initial_capital:,.2f} capital")
    
    def generate_market_data(self) -> pd.DataFrame:
        """Generate realistic market data with regimes"""
        date_range = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='1H')
        
        # Generate price data with realistic volatility
        np.random.seed(42)
        base_price = 0.5
        returns = np.random.normal(0, 0.02, len(date_range))  # 2% hourly volatility
        
        # Add regime changes
        regime_changes = np.random.choice([0, 1, 2], len(date_range), p=[0.7, 0.2, 0.1])
        regime_multipliers = np.where(regime_changes == 1, 1.5, np.where(regime_changes == 2, 0.5, 1.0))
        returns *= regime_multipliers
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate volume data
        volume = np.random.lognormal(10, 0.5, len(date_range))
        
        # Generate spread data
        spread = np.random.uniform(0.5, 2.0, len(date_range))  # 0.5-2.0 bps spread
        
        # Generate funding rates
        funding_rates = np.random.normal(0.0001, 0.0002, len(date_range))  # 0.01% ± 0.02%
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'price': prices,
            'volume': volume,
            'spread_bps': spread,
            'funding_rate': funding_rates,
            'regime': regime_changes
        })
        
        logger.info(f"🎯 [BACKTEST] Generated {len(df)} hours of market data")
        return df
    
    def calculate_slippage(self, quantity: float, price: float, side: str, 
                          market_data: pd.Series) -> Tuple[float, float]:
        """Calculate realistic slippage based on market depth"""
        # Base slippage from config
        base_slippage = self.config.slippage_bps / 10000
        
        # Volume impact
        volume_impact = min(quantity / market_data['volume'], 0.1)  # Max 10% impact
        
        # Spread cost
        spread_cost = market_data['spread_bps'] / 10000 / 2  # Half spread
        
        # Total slippage
        total_slippage = base_slippage + volume_impact + spread_cost
        
        # Apply to fill price
        if side == 'buy':
            fill_price = price * (1 + total_slippage)
        else:
            fill_price = price * (1 - total_slippage)
        
        return fill_price, total_slippage * 10000  # Return in bps
    
    def calculate_funding(self, position: float, price: float, funding_rate: float) -> float:
        """Calculate funding payment"""
        if position == 0:
            return 0.0
        
        notional = abs(position) * price
        funding_payment = notional * funding_rate
        
        # Long positions pay funding when rate is positive
        if position > 0 and funding_rate > 0:
            return -funding_payment  # Pay funding
        elif position < 0 and funding_rate < 0:
            return -funding_payment  # Pay funding
        else:
            return funding_payment  # Receive funding
    
    def execute_trade(self, strategy: str, side: str, quantity: float, 
                     price: float, market_data: pd.Series, reason_code: str = "strategy_signal") -> bool:
        """Execute a trade with realistic market conditions"""
        
        # Calculate slippage and fill price
        fill_price, slippage_bps = self.calculate_slippage(quantity, price, side, market_data)
        
        # Calculate fees
        notional = quantity * fill_price
        fee = notional * self.config.commission_rate
        
        # Check if we have enough capital
        if side == 'buy':
            required_capital = notional + fee
            if required_capital > self.cash:
                logger.warning(f"🎯 [BACKTEST] Insufficient capital for buy: ${required_capital:.2f} > ${self.cash:.2f}")
                return False
        else:
            if quantity > self.position:
                logger.warning(f"🎯 [BACKTEST] Insufficient position for sell: {quantity} > {self.position}")
                return False
        
        # Execute trade
        if side == 'buy':
            self.cash -= required_capital
            self.position += quantity
        else:
            self.cash += notional - fee
            self.position -= quantity
        
        # Update fees and slippage
        self.fees_paid += fee
        self.slippage_cost += abs(fill_price - price) * quantity
        
        # Create trade record
        trade = TradeRecord(
            timestamp=market_data['timestamp'],
            strategy=strategy,
            side=side,
            quantity=quantity,
            price=price,
            expected_price=price,
            fill_price=fill_price,
            fee=fee,
            funding=0.0,  # Will be calculated separately
            slippage_bps=slippage_bps,
            pnl_realized=0.0,  # Will be calculated on exit
            pnl_unrealized=0.0,
            reason_code=reason_code,
            maker_flag=np.random.random() > 0.3,  # 70% maker
            market_regime="normal" if market_data['regime'] == 0 else "high_vol",
            order_state="filled",
            cloid=f"{strategy}_{int(market_data['timestamp'].timestamp())}"
        )
        
        self.trades.append(trade)
        
        # Update portfolio value
        self.portfolio_value = self.cash + self.position * fill_price
        
        logger.info(f"🎯 [BACKTEST] {side.upper()} {quantity} {strategy} @ {fill_price:.4f} (slippage: {slippage_bps:.1f}bps)")
        return True
    
    def process_funding(self, market_data: pd.Series):
        """Process funding payments"""
        if self.position == 0:
            return
        
        funding_payment = self.calculate_funding(self.position, market_data['price'], market_data['funding_rate'])
        self.funding_paid += funding_payment
        self.cash += funding_payment
        
        # Update portfolio value
        self.portfolio_value = self.cash + self.position * market_data['price']
        
        if abs(funding_payment) > 0.01:  # Only log significant funding
            logger.info(f"🎯 [BACKTEST] Funding payment: ${funding_payment:.4f} (rate: {market_data['funding_rate']:.4f})")
    
    def run_backtest(self, strategies: List[str]) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        logger.info(f"🎯 [BACKTEST] Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Generate market data
        market_data = self.generate_market_data()
        
        # Run backtest
        for i, (_, row) in enumerate(market_data.iterrows()):
            # Process funding every hour
            if i % self.config.funding_frequency_hours == 0:
                self.process_funding(row)
            
            # Generate trading signals
            for strategy in strategies:
                if strategy == "BUY":
                    # Simple buy and hold strategy
                    if i == 0:  # Buy at start
                        self.execute_trade("BUY", "buy", 1000, row['price'], row, "initial_buy")
                elif strategy == "SCALP":
                    # Scalping strategy
                    if i % 24 == 0:  # Trade once per day
                        side = "buy" if np.random.random() > 0.5 else "sell"
                        quantity = np.random.uniform(50, 200)
                        self.execute_trade("SCALP", side, quantity, row['price'], row, "scalp_signal")
                elif strategy == "FUNDING_ARBITRAGE":
                    # Funding arbitrage strategy
                    if abs(row['funding_rate']) > 0.0005:  # 0.05% threshold
                        side = "buy" if row['funding_rate'] > 0 else "sell"
                        quantity = np.random.uniform(100, 500)
                        self.execute_trade("FUNDING_ARBITRAGE", side, quantity, row['price'], row, "funding_arb")
            
            # Update equity curve
            self.equity_curve.append(self.portfolio_value)
            
            # Update drawdown
            if self.portfolio_value > self.peak_equity:
                self.peak_equity = self.portfolio_value
            
            drawdown = (self.peak_equity - self.portfolio_value) / self.peak_equity
            self.drawdown_curve.append(drawdown)
        
        # Calculate final metrics
        metrics = self.calculate_metrics()
        
        logger.info(f"🎯 [BACKTEST] Backtest completed: {len(self.trades)} trades, ${self.portfolio_value:.2f} final value")
        return metrics
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl_realized > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns for Sharpe ratio
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)
        
        returns = np.array(returns)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0
        
        # Strategy performance
        strategy_perf = {}
        for strategy in set(t.strategy for t in self.trades):
            strategy_trades = [t for t in self.trades if t.strategy == strategy]
            strategy_perf[strategy] = {
                'trades': len(strategy_trades),
                'total_fees': sum(t.fee for t in strategy_trades),
                'total_slippage': sum(t.slippage_bps for t in strategy_trades),
                'maker_ratio': len([t for t in strategy_trades if t.maker_flag]) / len(strategy_trades)
            }
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_fees': self.fees_paid,
            'total_funding': self.funding_paid,
            'total_slippage': self.slippage_cost,
            'final_portfolio_value': self.portfolio_value,
            'strategy_performance': strategy_perf,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve
        }
    
    def save_trade_ledger(self, filepath: str):
        """Save trade ledger to CSV and Parquet"""
        if not self.trades:
            logger.warning("🎯 [BACKTEST] No trades to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'ts': t.timestamp.timestamp(),
                'strategy_name': t.strategy,
                'side': t.side,
                'qty': t.quantity,
                'price': t.price,
                'fee': t.fee,
                'fee_bps': (t.fee / (t.quantity * t.price)) * 10000 if t.quantity * t.price > 0 else 0,
                'funding': t.funding,
                'slippage_bps': t.slippage_bps,
                'pnl_realized': t.pnl_realized,
                'pnl_unrealized': t.pnl_unrealized,
                'reason_code': t.reason_code,
                'maker_flag': t.maker_flag,
                'order_state': t.order_state,
                'regime_label': t.market_regime,
                'cloid': t.cloid
            }
            for t in self.trades
        ])
        
        # Save CSV
        csv_path = filepath.replace('.parquet', '.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"🎯 [BACKTEST] Saved trade ledger: {csv_path}")
        
        # Save Parquet
        df.to_parquet(filepath, index=False)
        logger.info(f"🎯 [BACKTEST] Saved trade ledger: {filepath}")
    
    def generate_tearsheet(self, metrics: Dict[str, Any], filepath: str):
        """Generate HTML tearsheet"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>📊 Comprehensive Backtest Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; min-width: 150px; text-align: center; }}
                .performance {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }}
                .risk {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Comprehensive Backtest Results</h1>
                <p>Hat Manifesto Ultimate Trading System</p>
                <p>📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>📊 Period: {self.config.start_date} to {self.config.end_date}</p>
            </div>
            
            <div class="section">
                <h2>📊 Performance Summary</h2>
                <div class="metric performance">
                    <h3>Total Return</h3>
                    <h2>{metrics.get('total_return', 0):.2%}</h2>
                </div>
                <div class="metric performance">
                    <h3>Sharpe Ratio</h3>
                    <h2>{metrics.get('sharpe_ratio', 0):.2f}</h2>
                </div>
                <div class="metric risk">
                    <h3>Max Drawdown</h3>
                    <h2>{metrics.get('max_drawdown', 0):.2%}</h2>
                </div>
                <div class="metric performance">
                    <h3>Win Rate</h3>
                    <h2>{metrics.get('win_rate', 0):.2%}</h2>
                </div>
                <div class="metric performance">
                    <h3>Total Trades</h3>
                    <h2>{metrics.get('total_trades', 0)}</h2>
                </div>
            </div>
            
            <div class="section">
                <h2>💰 Cost Analysis</h2>
                <p>Total Fees: ${metrics.get('total_fees', 0):.2f}</p>
                <p>Total Funding: ${metrics.get('total_funding', 0):.2f}</p>
                <p>Total Slippage: ${metrics.get('total_slippage', 0):.2f}</p>
            </div>
            
            <div class="section">
                <h2>🎯 Strategy Performance</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 10px; border: 1px solid #ddd;">Strategy</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Trades</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Total Fees</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Maker Ratio</th>
                    </tr>
        """
        
        for strategy, perf in metrics.get('strategy_performance', {}).items():
            html += f"""
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">{strategy}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{perf['trades']}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">${perf['total_fees']:.2f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{perf['maker_ratio']:.1%}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"🎯 [BACKTEST] Generated tearsheet: {filepath}")

# Global logger
import logging
logger = logging.getLogger(__name__)

def run_comprehensive_backtest(start_date: str = "2023-01-01", end_date: str = "2024-12-31",
                              strategies: List[str] = ["BUY", "SCALP", "FUNDING_ARBITRAGE"],
                              initial_capital: float = 10000.0) -> Dict[str, Any]:
    """Run comprehensive backtest and return results"""
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    engine = ComprehensiveBacktestEngine(config)
    metrics = engine.run_backtest(strategies)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save trade ledger
    engine.save_trade_ledger(f'reports/ledgers/comprehensive_backtest_{timestamp}.parquet')
    
    # Generate tearsheet
    engine.generate_tearsheet(metrics, f'reports/tearsheets/comprehensive_backtest_{timestamp}.html')
    
    return metrics

if __name__ == "__main__":
    # Run example backtest
    results = run_comprehensive_backtest()
    print(f"Backtest completed: {results.get('total_return', 0):.2%} return")