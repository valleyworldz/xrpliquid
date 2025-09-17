"""
🎯 DETERMINISTIC BACKTESTING ENGINE
==================================
Comprehensive backtesting system for funding arbitrage strategy with
realistic market conditions, fees, slippage, and funding payments.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.strategies.funding_arbitrage import (
    FundingArbitrageStrategy, 
    FundingArbitrageConfig, 
    FundingArbitrageOpportunity
)
from src.core.risk.risk_unit_sizing import RiskUnitSizing, RiskUnitConfig
from src.core.utils.logger import Logger

@dataclass
class BacktestConfig:
    """Configuration for deterministic backtesting"""
    
    # Data parameters
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    symbol: str = "XRP"
    timeframe: str = "1h"  # 1-hour bars
    
    # Market conditions
    initial_capital: float = 10000.0
    commission_rate: float = 0.0001  # 0.01% commission
    slippage_bps: float = 2.0  # 2 basis points slippage
    spread_bps: float = 5.0  # 5 basis points spread
    
    # Funding parameters
    funding_frequency_hours: int = 8  # Every 8 hours
    funding_times: List[str] = field(default_factory=lambda: ["00:00", "08:00", "16:00"])
    
    # Risk parameters
    max_position_size_percent: float = 10.0  # Max 10% of capital per position
    max_total_exposure_percent: float = 50.0  # Max 50% total exposure
    stop_loss_percent: float = 5.0  # 5% stop loss
    take_profit_percent: float = 2.0  # 2% take profit
    
    # Backtest parameters
    warmup_period_days: int = 30  # 30 days warmup for indicators
    min_trades_for_analysis: int = 10  # Minimum trades for statistical significance

@dataclass
class MarketData:
    """Market data structure for backtesting"""
    
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    funding_rate: float
    spread_bps: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'funding_rate': self.funding_rate,
            'spread_bps': self.spread_bps
        }

@dataclass
class BacktestTrade:
    """Individual trade record for backtesting"""
    
    # Trade identification
    trade_id: str
    timestamp: pd.Timestamp
    
    # Market data
    symbol: str
    entry_price: float
    exit_price: float
    funding_rate: float
    
    # Position details
    side: str  # 'long' or 'short'
    quantity: float
    position_size_usd: float
    
    # Costs and fees
    commission: float
    slippage: float
    funding_payments: float
    total_costs: float
    
    # Performance
    gross_pnl: float
    net_pnl: float
    pnl_percent: float
    holding_period_hours: float
    
    # Risk metrics
    max_drawdown_during_trade: float
    volatility_during_trade: float
    
    # Strategy metrics
    confidence_score: float
    expected_value: float
    risk_score: float

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Risk metrics
    volatility: float
    var_95: float
    cvar_95: float
    time_under_water: float
    
    # Funding arbitrage specific
    total_funding_payments: float
    avg_funding_rate: float
    funding_efficiency: float
    
    # Regime analysis
    bull_market_performance: Dict[str, float]
    bear_market_performance: Dict[str, float]
    chop_market_performance: Dict[str, float]
    
    # Volatility analysis
    low_vol_performance: Dict[str, float]
    medium_vol_performance: Dict[str, float]
    high_vol_performance: Dict[str, float]

class DeterministicBacktestEngine:
    """
    🎯 DETERMINISTIC BACKTESTING ENGINE
    Comprehensive backtesting system for funding arbitrage strategy
    """
    
    def __init__(self, 
                 config: BacktestConfig,
                 strategy_config: FundingArbitrageConfig,
                 logger: Optional[Logger] = None):
        self.config = config
        self.strategy_config = strategy_config
        self.logger = logger or Logger()
        
        # Initialize strategy
        self.strategy = FundingArbitrageStrategy(strategy_config, None, self.logger)
        
        # Initialize risk unit sizing
        risk_config = RiskUnitConfig(
            target_volatility_percent=2.0,
            max_equity_at_risk_percent=1.0,
            base_equity_at_risk_percent=0.5,
            kelly_multiplier=0.25,
            min_position_size_usd=25.0,
            max_position_size_usd=config.initial_capital * 0.1
        )
        self.risk_sizing = RiskUnitSizing(risk_config, self.logger)
        
        # Backtest state
        self.current_capital = config.initial_capital
        self.positions: Dict[str, BacktestTrade] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.drawdown_curve: List[Tuple[pd.Timestamp, float]] = []
        
        # Performance tracking
        self.peak_capital = config.initial_capital
        self.max_drawdown = 0.0
        self.total_funding_payments = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Market data
        self.market_data: List[MarketData] = []
        self.current_timestamp: Optional[pd.Timestamp] = None
        
        self.logger.info("🎯 [BACKTEST] Deterministic Backtest Engine initialized")
        self.logger.info(f"📊 [BACKTEST] Period: {config.start_date} to {config.end_date}")
        self.logger.info(f"📊 [BACKTEST] Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"📊 [BACKTEST] Symbol: {config.symbol}")
    
    def generate_market_data(self) -> List[MarketData]:
        """Generate realistic market data for backtesting"""
        
        self.logger.info("📊 [BACKTEST] Generating market data...")
        
        # Create date range
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Generate realistic XRP price data
        np.random.seed(42)  # For reproducible results
        
        # Base price parameters
        base_price = 0.5  # Starting XRP price
        annual_volatility = 0.8  # 80% annual volatility
        daily_volatility = annual_volatility / np.sqrt(365)
        hourly_volatility = daily_volatility / np.sqrt(24)
        
        # Generate price series with realistic characteristics
        prices = [base_price]
        returns = []
        
        for i in range(1, len(date_range)):
            # Add some mean reversion and trend
            trend = 0.0001 * np.sin(i / 1000)  # Long-term cyclical trend
            mean_reversion = -0.01 * (prices[-1] - base_price) / base_price
            
            # Generate return
            random_return = np.random.normal(0, hourly_volatility)
            total_return = trend + mean_reversion + random_return
            
            # Apply some fat tails (occasional large moves)
            if np.random.random() < 0.01:  # 1% chance of large move
                total_return *= np.random.choice([-3, 3])  # 3x larger move
            
            returns.append(total_return)
            new_price = prices[-1] * (1 + total_return)
            prices.append(max(new_price, 0.01))  # Floor at $0.01
        
        # Generate OHLC data
        market_data = []
        for i, timestamp in enumerate(date_range):
            if i >= len(prices):
                break
                
            close = prices[i]
            
            # Generate realistic OHLC
            volatility_factor = abs(returns[i-1]) if i > 0 else 0.01
            high = close * (1 + volatility_factor * 0.5)
            low = close * (1 - volatility_factor * 0.5)
            open_price = prices[i-1] if i > 0 else close
            
            # Generate volume (higher during volatile periods)
            base_volume = 1000000
            volume_multiplier = 1 + volatility_factor * 5
            volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
            
            # Generate funding rate (mean-reverting around 0)
            if i == 0:
                funding_rate = 0.0
            else:
                # Funding rate mean reversion
                prev_funding = market_data[-1].funding_rate
                mean_reversion = -0.1 * prev_funding
                random_component = np.random.normal(0, 0.001)
                funding_rate = mean_reversion + random_component
                funding_rate = np.clip(funding_rate, -0.01, 0.01)  # Clip to ±1%
            
            # Generate spread (varies with volatility)
            base_spread = self.config.spread_bps
            spread_multiplier = 1 + volatility_factor * 2
            spread_bps = base_spread * spread_multiplier
            
            market_data.append(MarketData(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                funding_rate=funding_rate,
                spread_bps=spread_bps
            ))
        
        self.market_data = market_data
        self.logger.info(f"📊 [BACKTEST] Generated {len(market_data)} data points")
        
        return market_data
    
    def calculate_funding_payment(self, 
                                position_size: float,
                                funding_rate: float,
                                holding_period_hours: float) -> float:
        """Calculate funding payment for a position"""
        
        funding_payments = holding_period_hours / self.config.funding_frequency_hours
        funding_payment = position_size * abs(funding_rate) * funding_payments
        
        return funding_payment
    
    def calculate_trade_costs(self, 
                            position_size: float,
                            entry_price: float,
                            exit_price: float,
                            funding_rate: float,
                            holding_period_hours: float) -> Tuple[float, float, float, float]:
        """Calculate all trade costs"""
        
        # Commission (round trip)
        commission = position_size * self.config.commission_rate * 2
        
        # Slippage
        slippage = position_size * (self.config.slippage_bps / 10000) * 2
        
        # Funding payments
        funding_payments = self.calculate_funding_payment(
            position_size, funding_rate, holding_period_hours
        )
        
        # Total costs
        total_costs = commission + slippage + funding_payments
        
        return commission, slippage, funding_payments, total_costs
    
    def execute_trade(self, 
                     opportunity: FundingArbitrageOpportunity,
                     current_data: MarketData) -> Optional[BacktestTrade]:
        """Execute a trade in the backtest"""
        
        # Calculate position size using risk unit sizing
        position_size, risk_metrics = self.strategy.calculate_optimal_position_size(
            opportunity.symbol,
            self.current_capital,
            opportunity.current_funding_rate,
            current_data.close,
            confidence_score=opportunity.confidence_score
        )
        
        # Check if we have enough capital
        if position_size > self.current_capital * 0.95:  # Leave 5% buffer
            return None
        
        # Determine trade direction
        if opportunity.current_funding_rate > 0:
            side = 'short'  # Short to receive funding
            entry_price = current_data.close * (1 - self.config.spread_bps / 10000 / 2)
            exit_price = current_data.close * (1 + self.config.spread_bps / 10000 / 2)
        else:
            side = 'long'  # Long to pay funding (expect rate to increase)
            entry_price = current_data.close * (1 + self.config.spread_bps / 10000 / 2)
            exit_price = current_data.close * (1 - self.config.spread_bps / 10000 / 2)
        
        # Calculate quantity
        quantity = position_size / entry_price
        
        # Calculate holding period (simplified - assume 8 hours for funding arbitrage)
        holding_period_hours = 8.0
        
        # Calculate costs
        commission, slippage, funding_payments, total_costs = self.calculate_trade_costs(
            position_size, entry_price, exit_price, 
            opportunity.current_funding_rate, holding_period_hours
        )
        
        # Calculate PnL
        if side == 'long':
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity
        
        net_pnl = gross_pnl - total_costs
        pnl_percent = (net_pnl / position_size) * 100
        
        # Create trade record
        trade = BacktestTrade(
            trade_id=f"trade_{len(self.trades) + 1}",
            timestamp=current_data.timestamp,
            symbol=opportunity.symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            funding_rate=opportunity.current_funding_rate,
            side=side,
            quantity=quantity,
            position_size_usd=position_size,
            commission=commission,
            slippage=slippage,
            funding_payments=funding_payments,
            total_costs=total_costs,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_percent=pnl_percent,
            holding_period_hours=holding_period_hours,
            max_drawdown_during_trade=0.0,  # Will be calculated later
            volatility_during_trade=0.0,  # Will be calculated later
            confidence_score=opportunity.confidence_score,
            expected_value=opportunity.expected_value,
            risk_score=opportunity.risk_score
        )
        
        # Update capital
        self.current_capital += net_pnl
        
        # Update tracking
        self.total_funding_payments += funding_payments
        self.total_commission += commission
        self.total_slippage += slippage
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Record trade
        self.trades.append(trade)
        
        return trade
    
    def run_backtest(self) -> BacktestResults:
        """Run the complete backtest"""
        
        self.logger.info("🚀 [BACKTEST] Starting deterministic backtest...")
        
        # Generate market data
        market_data = self.generate_market_data()
        
        # Run backtest
        for i, data in enumerate(market_data):
            self.current_timestamp = data.timestamp
            
            # Skip warmup period
            if i < self.config.warmup_period_days * 24:
                continue
            
            # Update strategy with current data
            self.strategy.price_history[self.config.symbol] = [d.close for d in market_data[max(0, i-720):i+1]]  # Last 30 days
            
            # Check for funding arbitrage opportunities
            opportunity = self.strategy.assess_opportunity(
                self.config.symbol,
                data.funding_rate,
                data.close,
                self.current_capital
            )
            
            if opportunity:
                # Execute trade
                trade = self.execute_trade(opportunity, data)
                if trade:
                    self.logger.debug(f"📊 [BACKTEST] Trade executed: {trade.side} {trade.quantity:.2f} @ ${trade.entry_price:.4f}")
            
            # Record equity curve
            self.equity_curve.append((data.timestamp, self.current_capital))
            
            # Record drawdown
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.drawdown_curve.append((data.timestamp, current_drawdown))
        
        # Calculate results
        results = self.calculate_results()
        
        self.logger.info("✅ [BACKTEST] Backtest completed")
        self.logger.info(f"📊 [BACKTEST] Total trades: {len(self.trades)}")
        self.logger.info(f"📊 [BACKTEST] Final capital: ${self.current_capital:,.2f}")
        self.logger.info(f"📊 [BACKTEST] Total return: {results.total_return:.2%}")
        
        return results
    
    def calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        if not self.trades:
            self.logger.warning("⚠️ [BACKTEST] No trades executed")
            return self._create_empty_results()
        
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame([trade.__dict__ for trade in self.trades])
        
        # Basic performance metrics
        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate annualized return
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        years = (end_date - start_date).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate Sharpe ratio
        returns = trades_df['net_pnl'] / trades_df['position_size_usd']
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        
        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252 * 24) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / self.max_drawdown if self.max_drawdown > 0 else 0
        
        # Trade statistics
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
        win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else safe_float('inf')
        
        # Average win/loss
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 24)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Time under water (simplified)
        time_under_water = len(self.drawdown_curve) / len(self.equity_curve) if self.equity_curve else 0
        
        # Funding arbitrage specific metrics
        total_funding_payments = trades_df['funding_payments'].sum()
        avg_funding_rate = trades_df['funding_rate'].mean()
        funding_efficiency = total_funding_payments / trades_df['net_pnl'].sum() if trades_df['net_pnl'].sum() != 0 else 0
        
        # Regime analysis (simplified)
        bull_market_performance = self._analyze_regime_performance(trades_df, 'bull')
        bear_market_performance = self._analyze_regime_performance(trades_df, 'bear')
        chop_market_performance = self._analyze_regime_performance(trades_df, 'chop')
        
        # Volatility analysis
        low_vol_performance = self._analyze_volatility_performance(trades_df, 'low')
        medium_vol_performance = self._analyze_volatility_performance(trades_df, 'medium')
        high_vol_performance = self._analyze_volatility_performance(trades_df, 'high')
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=self.max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=len(trades_df),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            time_under_water=time_under_water,
            total_funding_payments=total_funding_payments,
            avg_funding_rate=avg_funding_rate,
            funding_efficiency=funding_efficiency,
            bull_market_performance=bull_market_performance,
            bear_market_performance=bear_market_performance,
            chop_market_performance=chop_market_performance,
            low_vol_performance=low_vol_performance,
            medium_vol_performance=medium_vol_performance,
            high_vol_performance=high_vol_performance
        )
    
    def _create_empty_results(self) -> BacktestResults:
        """Create empty results when no trades are executed"""
        return BacktestResults(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            volatility=0.0,
            var_95=0.0,
            cvar_95=0.0,
            time_under_water=0.0,
            total_funding_payments=0.0,
            avg_funding_rate=0.0,
            funding_efficiency=0.0,
            bull_market_performance={},
            bear_market_performance={},
            chop_market_performance={},
            low_vol_performance={},
            medium_vol_performance={},
            high_vol_performance={}
        )
    
    def _analyze_regime_performance(self, trades_df: pd.DataFrame, regime: str) -> Dict[str, float]:
        """Analyze performance by market regime"""
        # Simplified regime analysis - in practice, you'd use actual regime detection
        if regime == 'bull':
            # Assume first third of trades are in bull market
            regime_trades = trades_df.iloc[:len(trades_df)//3]
        elif regime == 'bear':
            # Assume middle third are in bear market
            regime_trades = trades_df.iloc[len(trades_df)//3:2*len(trades_df)//3]
        else:  # chop
            # Assume last third are in choppy market
            regime_trades = trades_df.iloc[2*len(trades_df)//3:]
        
        if len(regime_trades) == 0:
            return {}
        
        return {
            'total_return': regime_trades['net_pnl'].sum() / self.config.initial_capital,
            'win_rate': len(regime_trades[regime_trades['net_pnl'] > 0]) / len(regime_trades),
            'avg_trade': regime_trades['net_pnl'].mean(),
            'total_trades': len(regime_trades)
        }
    
    def _analyze_volatility_performance(self, trades_df: pd.DataFrame, vol_level: str) -> Dict[str, float]:
        """Analyze performance by volatility level"""
        # Simplified volatility analysis based on funding rate magnitude
        if vol_level == 'low':
            vol_trades = trades_df[abs(trades_df['funding_rate']) < 0.001]
        elif vol_level == 'medium':
            vol_trades = trades_df[(abs(trades_df['funding_rate']) >= 0.001) & (abs(trades_df['funding_rate']) < 0.005)]
        else:  # high
            vol_trades = trades_df[abs(trades_df['funding_rate']) >= 0.005]
        
        if len(vol_trades) == 0:
            return {}
        
        return {
            'total_return': vol_trades['net_pnl'].sum() / self.config.initial_capital,
            'win_rate': len(vol_trades[vol_trades['net_pnl'] > 0]) / len(vol_trades),
            'avg_trade': vol_trades['net_pnl'].mean(),
            'total_trades': len(vol_trades)
        }
    
    def save_results(self, results: BacktestResults, output_dir: str = "reports"):
        """Save backtest results to files"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save results as JSON
        results_dict = {
            'performance_metrics': {
                'total_return': results.total_return,
                'annualized_return': results.annualized_return,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'max_drawdown': results.max_drawdown,
                'calmar_ratio': results.calmar_ratio
            },
            'trade_statistics': {
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss
            },
            'risk_metrics': {
                'volatility': results.volatility,
                'var_95': results.var_95,
                'cvar_95': results.cvar_95,
                'time_under_water': results.time_under_water
            },
            'funding_arbitrage_metrics': {
                'total_funding_payments': results.total_funding_payments,
                'avg_funding_rate': results.avg_funding_rate,
                'funding_efficiency': results.funding_efficiency
            },
            'regime_analysis': {
                'bull_market': results.bull_market_performance,
                'bear_market': results.bear_market_performance,
                'chop_market': results.chop_market_performance
            },
            'volatility_analysis': {
                'low_volatility': results.low_vol_performance,
                'medium_volatility': results.medium_vol_performance,
                'high_volatility': results.high_vol_performance
            }
        }
        
        # Save JSON results
        with open(f"{output_dir}/backtest_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save trades CSV
        if self.trades:
            trades_df = pd.DataFrame([trade.__dict__ for trade in self.trades])
            trades_df.to_csv(f"{output_dir}/trades.csv", index=False)
        
        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
            equity_df.to_csv(f"{output_dir}/equity_curve.csv", index=False)
        
        # Save drawdown curve
        if self.drawdown_curve:
            drawdown_df = pd.DataFrame(self.drawdown_curve, columns=['timestamp', 'drawdown'])
            drawdown_df.to_csv(f"{output_dir}/drawdown_curve.csv", index=False)
        
        self.logger.info(f"📊 [BACKTEST] Results saved to {output_dir}/")
    
    def generate_tear_sheet(self, results: BacktestResults, output_dir: str = "reports"):
        """Generate comprehensive tear sheet"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create tear sheet HTML
        tear_sheet_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Funding Arbitrage Backtest Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .metric-label {{ font-weight: bold; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .neutral {{ color: blue; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Funding Arbitrage Strategy Backtest Results</h1>
                <p><strong>Period:</strong> {self.config.start_date} to {self.config.end_date}</p>
                <p><strong>Symbol:</strong> {self.config.symbol}</p>
                <p><strong>Initial Capital:</strong> ${self.config.initial_capital:,.2f}</p>
                <p><strong>Final Capital:</strong> ${self.current_capital:,.2f}</p>
            </div>
            
            <h2>📊 Performance Metrics</h2>
            <table>
                <tr><td>Total Return</td><td class="{'positive' if results.total_return > 0 else 'negative'}">{results.total_return:.2%}</td></tr>
                <tr><td>Annualized Return</td><td class="{'positive' if results.annualized_return > 0 else 'negative'}">{results.annualized_return:.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td class="{'positive' if results.sharpe_ratio > 1 else 'neutral'}">{results.sharpe_ratio:.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td class="{'positive' if results.sortino_ratio > 1 else 'neutral'}">{results.sortino_ratio:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td class="negative">{results.max_drawdown:.2%}</td></tr>
                <tr><td>Calmar Ratio</td><td class="{'positive' if results.calmar_ratio > 1 else 'neutral'}">{results.calmar_ratio:.2f}</td></tr>
            </table>
            
            <h2>📈 Trade Statistics</h2>
            <table>
                <tr><td>Total Trades</td><td>{results.total_trades}</td></tr>
                <tr><td>Winning Trades</td><td class="positive">{results.winning_trades}</td></tr>
                <tr><td>Losing Trades</td><td class="negative">{results.losing_trades}</td></tr>
                <tr><td>Win Rate</td><td class="{'positive' if results.win_rate > 0.5 else 'neutral'}">{results.win_rate:.2%}</td></tr>
                <tr><td>Profit Factor</td><td class="{'positive' if results.profit_factor > 1 else 'negative'}">{results.profit_factor:.2f}</td></tr>
                <tr><td>Average Win</td><td class="positive">${results.avg_win:.2f}</td></tr>
                <tr><td>Average Loss</td><td class="negative">${results.avg_loss:.2f}</td></tr>
            </table>
            
            <h2>⚠️ Risk Metrics</h2>
            <table>
                <tr><td>Volatility</td><td>{results.volatility:.2%}</td></tr>
                <tr><td>VaR (95%)</td><td class="negative">{results.var_95:.2%}</td></tr>
                <tr><td>CVaR (95%)</td><td class="negative">{results.cvar_95:.2%}</td></tr>
                <tr><td>Time Under Water</td><td>{results.time_under_water:.2%}</td></tr>
            </table>
            
            <h2>💰 Funding Arbitrage Metrics</h2>
            <table>
                <tr><td>Total Funding Payments</td><td>${results.total_funding_payments:.2f}</td></tr>
                <tr><td>Average Funding Rate</td><td>{results.avg_funding_rate:.4f}</td></tr>
                <tr><td>Funding Efficiency</td><td>{results.funding_efficiency:.2f}</td></tr>
            </table>
            
            <h2>📊 Regime Analysis</h2>
            <h3>Bull Market Performance</h3>
            <table>
                <tr><td>Total Return</td><td class="{'positive' if results.bull_market_performance.get('total_return', 0) > 0 else 'negative'}">{results.bull_market_performance.get('total_return', 0):.2%}</td></tr>
                <tr><td>Win Rate</td><td>{results.bull_market_performance.get('win_rate', 0):.2%}</td></tr>
                <tr><td>Total Trades</td><td>{results.bull_market_performance.get('total_trades', 0)}</td></tr>
            </table>
            
            <h3>Bear Market Performance</h3>
            <table>
                <tr><td>Total Return</td><td class="{'positive' if results.bear_market_performance.get('total_return', 0) > 0 else 'negative'}">{results.bear_market_performance.get('total_return', 0):.2%}</td></tr>
                <tr><td>Win Rate</td><td>{results.bear_market_performance.get('win_rate', 0):.2%}</td></tr>
                <tr><td>Total Trades</td><td>{results.bear_market_performance.get('total_trades', 0)}</td></tr>
            </table>
            
            <h3>Chop Market Performance</h3>
            <table>
                <tr><td>Total Return</td><td class="{'positive' if results.chop_market_performance.get('total_return', 0) > 0 else 'negative'}">{results.chop_market_performance.get('total_return', 0):.2%}</td></tr>
                <tr><td>Win Rate</td><td>{results.chop_market_performance.get('win_rate', 0):.2%}</td></tr>
                <tr><td>Total Trades</td><td>{results.chop_market_performance.get('total_trades', 0)}</td></tr>
            </table>
            
            <h2>📈 Volatility Analysis</h2>
            <h3>Low Volatility Performance</h3>
            <table>
                <tr><td>Total Return</td><td class="{'positive' if results.low_vol_performance.get('total_return', 0) > 0 else 'negative'}">{results.low_vol_performance.get('total_return', 0):.2%}</td></tr>
                <tr><td>Win Rate</td><td>{results.low_vol_performance.get('win_rate', 0):.2%}</td></tr>
                <tr><td>Total Trades</td><td>{results.low_vol_performance.get('total_trades', 0)}</td></tr>
            </table>
            
            <h3>Medium Volatility Performance</h3>
            <table>
                <tr><td>Total Return</td><td class="{'positive' if results.medium_vol_performance.get('total_return', 0) > 0 else 'negative'}">{results.medium_vol_performance.get('total_return', 0):.2%}</td></tr>
                <tr><td>Win Rate</td><td>{results.medium_vol_performance.get('win_rate', 0):.2%}</td></tr>
                <tr><td>Total Trades</td><td>{results.medium_vol_performance.get('total_trades', 0)}</td></tr>
            </table>
            
            <h3>High Volatility Performance</h3>
            <table>
                <tr><td>Total Return</td><td class="{'positive' if results.high_vol_performance.get('total_return', 0) > 0 else 'negative'}">{results.high_vol_performance.get('total_return', 0):.2%}</td></tr>
                <tr><td>Win Rate</td><td>{results.high_vol_performance.get('win_rate', 0):.2%}</td></tr>
                <tr><td>Total Trades</td><td>{results.high_vol_performance.get('total_trades', 0)}</td></tr>
            </table>
            
            <h2>📋 Summary</h2>
            <p>The funding arbitrage strategy {'performed well' if results.total_return > 0 else 'underperformed'} during the backtest period, 
            achieving a total return of {results.total_return:.2%} with a Sharpe ratio of {results.sharpe_ratio:.2f}. 
            The strategy executed {results.total_trades} trades with a win rate of {results.win_rate:.2%}.</p>
            
            <p><strong>Key Insights:</strong></p>
            <ul>
                <li>Maximum drawdown was {results.max_drawdown:.2%}</li>
                <li>Total funding payments collected: ${results.total_funding_payments:.2f}</li>
                <li>Average funding rate: {results.avg_funding_rate:.4f}</li>
                <li>Funding efficiency: {results.funding_efficiency:.2f}</li>
            </ul>
        </body>
        </html>
        """
        
        # Save tear sheet
        with open(f"{output_dir}/tear_sheet.html", 'w', encoding='utf-8') as f:
            f.write(tear_sheet_html)
        
        self.logger.info(f"📊 [BACKTEST] Tear sheet saved to {output_dir}/tear_sheet.html")
