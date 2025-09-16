#!/usr/bin/env python3
"""
📊 COMPREHENSIVE BACKTEST RUNNER
================================
Orchestrates comprehensive backtesting with trade ledger generation and tearsheets.

Usage:
    python scripts/run_comprehensive_backtest.py --start 2022-01-01 --end 2025-09-15 --include_scalp --include_funding_arb
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.backtesting.comprehensive_backtest_engine import ComprehensiveBacktestEngine, BacktestConfig
from src.core.utils.logger import Logger

async def main():
    """Main backtest runner function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run comprehensive backtest')
    parser.add_argument('--start', type=str, default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-09-15', help='End date (YYYY-MM-DD)')
    parser.add_argument('--include_scalp', action='store_true', help='Include SCALP strategy')
    parser.add_argument('--include_funding_arb', action='store_true', help='Include FUNDING_ARBITRAGE strategy')
    parser.add_argument('--initial_capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--symbol', type=str, default='XRP', help='Trading symbol')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger()
    
    logger.info("📊 [BACKTEST_RUNNER] Starting comprehensive backtest...")
    logger.info(f"📊 [BACKTEST_RUNNER] Period: {args.start} to {args.end}")
    logger.info(f"📊 [BACKTEST_RUNNER] Initial Capital: ${args.initial_capital:,.2f}")
    
    # Configure strategies
    strategies = ['BUY']  # Always include BUY strategy
    if args.include_scalp:
        strategies.append('SCALP')
    if args.include_funding_arb:
        strategies.append('FUNDING_ARBITRAGE')
    
    logger.info(f"📊 [BACKTEST_RUNNER] Strategies: {', '.join(strategies)}")
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        strategies=strategies,
        max_position_size=0.1,  # 10% of capital
        max_drawdown=0.05,      # 5% max drawdown
        stop_loss=0.02,         # 2% stop loss
        maker_fee=0.0001,       # 0.01% maker fee
        taker_fee=0.0005,       # 0.05% taker fee
        maker_rebate=0.00005,   # 0.005% maker rebate
        base_slippage=0.0002,   # 0.02% base slippage
        funding_interval_hours=1,  # 1-hour funding cycles
        base_funding_rate=0.0001,  # 0.01% base funding rate
    )
    
    # Create and run backtest engine
    engine = ComprehensiveBacktestEngine(config, logger)
    
    try:
        # Run backtest
        result = await engine.run_backtest()
        
        # Print results
        logger.info("📊 [BACKTEST_RUNNER] Backtest completed successfully!")
        logger.info(f"📊 [BACKTEST_RUNNER] Total Return: {result.total_return:.2%}")
        logger.info(f"📊 [BACKTEST_RUNNER] Annualized Return: {result.annualized_return:.2%}")
        logger.info(f"📊 [BACKTEST_RUNNER] Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"📊 [BACKTEST_RUNNER] Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"📊 [BACKTEST_RUNNER] Win Rate: {result.win_rate:.2%}")
        logger.info(f"📊 [BACKTEST_RUNNER] Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"📊 [BACKTEST_RUNNER] Total Trades: {result.total_trades}")
        
        # Print strategy performance
        logger.info("📊 [BACKTEST_RUNNER] Strategy Performance:")
        for strategy, metrics in result.strategy_performance.items():
            logger.info(f"  {strategy}: {metrics['trades']} trades, ${metrics['total_pnl']:.2f} P&L, {metrics['win_rate']:.2%} win rate")
        
        # Print component attribution
        logger.info("📊 [BACKTEST_RUNNER] Component Attribution:")
        logger.info(f"  Directional P&L: ${result.directional_pnl:.2f}")
        logger.info(f"  Fee P&L: ${result.fee_pnl:.2f}")
        logger.info(f"  Funding P&L: ${result.funding_pnl:.2f}")
        logger.info(f"  Slippage P&L: ${result.slippage_pnl:.2f}")
        
        # Print regime performance
        logger.info("📊 [BACKTEST_RUNNER] Regime Performance:")
        for regime, metrics in result.regime_performance.items():
            logger.info(f"  {regime}: {metrics['trades']} trades, ${metrics['total_pnl']:.2f} P&L, {metrics['win_rate']:.2%} win rate")
        
        logger.info("📊 [BACKTEST_RUNNER] Reports generated in reports/tearsheets/")
        logger.info("📊 [BACKTEST_RUNNER] Trade records saved in reports/ledgers/")
        
    except Exception as e:
        logger.error(f"❌ [BACKTEST_RUNNER] Error running backtest: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)