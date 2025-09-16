#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE BACKTEST RUNNER
===============================
Orchestrates comprehensive backtesting with trade ledger and tearsheet generation
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from core.backtesting.comprehensive_backtest_engine import run_comprehensive_backtest

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/backtest.log')
        ]
    )

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run comprehensive backtest')
    parser.add_argument('--start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategies', nargs='+', 
                       default=['BUY', 'SCALP', 'FUNDING_ARBITRAGE'],
                       help='Strategies to test')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital')
    parser.add_argument('--include_all_strategies', action='store_true',
                       help='Include all available strategies')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create reports directories
    Path('reports/ledgers').mkdir(parents=True, exist_ok=True)
    Path('reports/tearsheets').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Determine strategies
    if args.include_all_strategies:
        strategies = ['BUY', 'SCALP', 'FUNDING_ARBITRAGE', 'MEAN_REVERSION', 'MOMENTUM']
    else:
        strategies = args.strategies
    
    logger.info(f"🎯 [COMPREHENSIVE_BACKTEST] Starting backtest from {args.start} to {args.end}")
    logger.info(f"🎯 [COMPREHENSIVE_BACKTEST] Strategies: {strategies}")
    logger.info(f"🎯 [COMPREHENSIVE_BACKTEST] Initial capital: ${args.capital:,.2f}")
    
    try:
        # Run backtest
        results = run_comprehensive_backtest(
            start_date=args.start,
            end_date=args.end,
            strategies=strategies,
            initial_capital=args.capital
        )
        
        # Print results
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE BACKTEST RESULTS")
        print("="*60)
        print(f"📊 Total Return: {results.get('total_return', 0):.2%}")
        print(f"📊 Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"📊 Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"📊 Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"📊 Total Trades: {results.get('total_trades', 0)}")
        print(f"📊 Total Fees: ${results.get('total_fees', 0):.2f}")
        print(f"📊 Total Funding: ${results.get('total_funding', 0):.2f}")
        print(f"📊 Final Portfolio Value: ${results.get('final_portfolio_value', 0):,.2f}")
        
        print("\n🎯 Strategy Performance:")
        for strategy, perf in results.get('strategy_performance', {}).items():
            print(f"  {strategy}: {perf['trades']} trades, {perf['maker_ratio']:.1%} maker ratio")
        
        print("\n✅ Backtest completed successfully!")
        print("📁 Check reports/ledgers/ and reports/tearsheets/ for detailed results")
        
    except Exception as e:
        logger.error(f"❌ [COMPREHENSIVE_BACKTEST] Error running backtest: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()