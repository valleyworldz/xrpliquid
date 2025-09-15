#!/usr/bin/env python3
"""
🎯 OPTIMIZED BACKTEST RUNNER
===========================
Run an optimized backtest for funding arbitrage strategy with better parameters
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.backtesting.deterministic_backtest_engine import (
    DeterministicBacktestEngine,
    BacktestConfig
)
from src.core.strategies.funding_arbitrage import FundingArbitrageConfig
from src.core.utils.logger import Logger

async def run_optimized_backtest():
    """Run an optimized backtest with better parameters"""
    
    print("🎯 OPTIMIZED BACKTEST FOR FUNDING ARBITRAGE STRATEGY")
    print("=" * 60)
    print("Running optimized backtest with improved parameters")
    print("=" * 60)
    
    # Initialize logger
    logger = Logger()
    
    # Configure backtest with optimized parameters
    backtest_config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-06-30",  # 6 months of data
        symbol="XRP",
        timeframe="1h",
        initial_capital=10000.0,
        commission_rate=0.0001,  # 0.01% commission
        slippage_bps=1.0,  # Reduced slippage to 1 bps
        spread_bps=3.0,  # Reduced spread to 3 bps
        funding_frequency_hours=8,
        funding_times=["00:00", "08:00", "16:00"],
        max_position_size_percent=10.0,
        max_total_exposure_percent=50.0,
        stop_loss_percent=5.0,
        take_profit_percent=2.0,
        warmup_period_days=7,  # Shorter warmup
        min_trades_for_analysis=5
    )
    
    # Configure strategy with optimized parameters
    strategy_config = FundingArbitrageConfig(
        min_funding_rate_threshold=0.0005,  # Increased to 0.05% minimum
        max_funding_rate_threshold=0.01,    # 1% maximum
        optimal_funding_rate=0.005,         # 0.5% optimal
        max_position_size_usd=500.0,  # Increased position size
        position_size_multiplier=0.05,  # 5% of capital
        min_position_size_usd=100.0,  # Increased minimum to $100
        max_drawdown_percent=5.0,
        stop_loss_funding_rate=0.02,
        take_profit_funding_rate=0.001,
        funding_rate_check_interval=300,
        execution_delay_seconds=30,
        max_execution_time_seconds=60,
        expected_holding_period_hours=8.0,
        funding_payment_frequency_hours=8.0,
        transaction_cost_bps=1.0,  # Reduced transaction cost
        slippage_cost_bps=0.5,  # Reduced slippage cost
        min_volume_24h_usd=1000000.0,
        max_spread_bps=10.0,
        min_liquidity_usd=50000.0
    )
    
    print("📊 Optimized Backtest Configuration:")
    print(f"   Period: {backtest_config.start_date} to {backtest_config.end_date}")
    print(f"   Initial Capital: ${backtest_config.initial_capital:,.2f}")
    print(f"   Commission Rate: {backtest_config.commission_rate:.4f}")
    print(f"   Slippage: {backtest_config.slippage_bps} bps")
    print(f"   Spread: {backtest_config.spread_bps} bps")
    print(f"   Funding Frequency: Every {backtest_config.funding_frequency_hours} hours")
    
    print("\n📊 Optimized Strategy Configuration:")
    print(f"   Min Funding Threshold: {strategy_config.min_funding_rate_threshold:.4f}")
    print(f"   Max Funding Threshold: {strategy_config.max_funding_rate_threshold:.4f}")
    print(f"   Optimal Funding Rate: {strategy_config.optimal_funding_rate:.4f}")
    print(f"   Max Position Size: ${strategy_config.max_position_size_usd:,.2f}")
    print(f"   Min Position Size: ${strategy_config.min_position_size_usd:,.2f}")
    print(f"   Transaction Cost: {strategy_config.transaction_cost_bps} bps")
    print(f"   Slippage Cost: {strategy_config.slippage_cost_bps} bps")
    
    # Initialize backtest engine
    backtest_engine = DeterministicBacktestEngine(
        backtest_config,
        strategy_config,
        logger
    )
    
    print("\n🚀 Starting optimized backtest...")
    start_time = time.time()
    
    # Run backtest
    results = backtest_engine.run_backtest()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️ Optimized backtest completed in {execution_time:.2f} seconds")
    
    # Display results
    print("\n📊 OPTIMIZED BACKTEST RESULTS")
    print("=" * 50)
    
    print("📈 Performance Metrics:")
    print(f"   Total Return: {results.total_return:.2%}")
    print(f"   Annualized Return: {results.annualized_return:.2%}")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {results.sortino_ratio:.2f}")
    print(f"   Max Drawdown: {results.max_drawdown:.2%}")
    print(f"   Calmar Ratio: {results.calmar_ratio:.2f}")
    
    print("\n📊 Trade Statistics:")
    print(f"   Total Trades: {results.total_trades}")
    print(f"   Winning Trades: {results.winning_trades}")
    print(f"   Losing Trades: {results.losing_trades}")
    print(f"   Win Rate: {results.win_rate:.2%}")
    print(f"   Profit Factor: {results.profit_factor:.2f}")
    print(f"   Average Win: ${results.avg_win:.2f}")
    print(f"   Average Loss: ${results.avg_loss:.2f}")
    
    print("\n⚠️ Risk Metrics:")
    print(f"   Volatility: {results.volatility:.2%}")
    print(f"   VaR (95%): {results.var_95:.2%}")
    print(f"   CVaR (95%): {results.cvar_95:.2%}")
    print(f"   Time Under Water: {results.time_under_water:.2%}")
    
    print("\n💰 Funding Arbitrage Metrics:")
    print(f"   Total Funding Payments: ${results.total_funding_payments:.2f}")
    print(f"   Average Funding Rate: {results.avg_funding_rate:.4f}")
    print(f"   Funding Efficiency: {results.funding_efficiency:.2f}")
    
    # Create reports directory
    reports_dir = "reports"
    Path(reports_dir).mkdir(exist_ok=True)
    
    print(f"\n💾 Saving results to {reports_dir}/...")
    
    # Save results
    backtest_engine.save_results(results, reports_dir)
    backtest_engine.generate_tear_sheet(results, reports_dir)
    
    print(f"✅ Results saved to {reports_dir}/")
    print(f"   - backtest_results.json")
    print(f"   - trades.csv")
    print(f"   - equity_curve.csv")
    print(f"   - drawdown_curve.csv")
    print(f"   - tear_sheet.html")
    
    # Profitability assessment
    print("\n🎯 PROFITABILITY ASSESSMENT")
    print("=" * 50)
    
    if results.total_return > 0:
        print("✅ STRATEGY IS PROFITABLE")
        print(f"   Total Return: {results.total_return:.2%}")
        print(f"   Annualized Return: {results.annualized_return:.2%}")
        
        if results.sharpe_ratio > 1.0:
            print("✅ GOOD RISK-ADJUSTED RETURNS")
            print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        else:
            print("⚠️ MODERATE RISK-ADJUSTED RETURNS")
            print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        
        if results.max_drawdown < 0.1:
            print("✅ LOW MAXIMUM DRAWDOWN")
            print(f"   Max Drawdown: {results.max_drawdown:.2%}")
        else:
            print("⚠️ HIGH MAXIMUM DRAWDOWN")
            print(f"   Max Drawdown: {results.max_drawdown:.2%}")
        
        if results.win_rate > 0.5:
            print("✅ GOOD WIN RATE")
            print(f"   Win Rate: {results.win_rate:.2%}")
        else:
            print("⚠️ MODERATE WIN RATE")
            print(f"   Win Rate: {results.win_rate:.2%}")
        
        if results.profit_factor > 1.5:
            print("✅ GOOD PROFIT FACTOR")
            print(f"   Profit Factor: {results.profit_factor:.2f}")
        else:
            print("⚠️ MODERATE PROFIT FACTOR")
            print(f"   Profit Factor: {results.profit_factor:.2f}")
        
    else:
        print("❌ STRATEGY IS NOT PROFITABLE")
        print(f"   Total Return: {results.total_return:.2%}")
        print("   Strategy needs further optimization")
    
    print("\n📋 SUMMARY")
    print("=" * 50)
    print(f"The optimized funding arbitrage strategy {'achieved profitability' if results.total_return > 0 else 'failed to achieve profitability'} during the {backtest_config.start_date} to {backtest_config.end_date} backtest period.")
    print(f"Key metrics:")
    print(f"  - Total Return: {results.total_return:.2%}")
    print(f"  - Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"  - Max Drawdown: {results.max_drawdown:.2%}")
    print(f"  - Win Rate: {results.win_rate:.2%}")
    print(f"  - Total Trades: {results.total_trades}")
    print(f"  - Funding Payments: ${results.total_funding_payments:.2f}")
    
    if results.total_return > 0:
        print("\n🎉 The optimized strategy demonstrates profitability and is ready for live trading!")
    else:
        print("\n⚠️ The strategy needs further optimization before live trading.")
    
    return results

async def main():
    """Main function"""
    try:
        results = await run_optimized_backtest()
        return results
    except Exception as e:
        print(f"❌ Optimized backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main())
