#!/usr/bin/env python3
"""
🎯 BACKTEST RUNNER
Script to run deterministic backtests and generate tear sheets
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.backtesting.deterministic_backtest import DeterministicBacktestEngine
from core.utils.logger import Logger

def run_backtest_test():
    """Run a test backtest to verify the system works"""
    print("🎯 DETERMINISTIC BACKTEST SYSTEM TEST")
    print("=" * 60)
    
    # Initialize logger
    logger = Logger()
    
    # Initialize backtest engine
    print("🚀 Initializing backtest engine...")
    backtest_engine = DeterministicBacktestEngine(
        initial_capital=1000.0,
        logger=logger
    )
    
    # Define strategy configuration
    strategy_config = {
        'lookback_period': 20,
        'entry_threshold': 0.02,  # 2% entry threshold
        'exit_threshold': 0.05,   # 5% exit threshold
        'max_position_size': 0.1  # 10% of capital per trade
    }
    
    print("📊 Strategy Configuration:")
    for key, value in strategy_config.items():
        print(f"   {key}: {value}")
    print()
    
    # Run backtest
    print("🚀 Running backtest...")
    results = backtest_engine.run_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        strategy_config=strategy_config
    )
    
    if results.get('success'):
        print("✅ Backtest completed successfully!")
        print()
        
        # Display results
        metrics = results.get('metrics', {})
        print("📈 PERFORMANCE METRICS:")
        print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"   Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   Volatility: {metrics.get('volatility', 0):.2%}")
        print()
        
        # Display trade summary
        trades = results.get('trades', [])
        print("📊 TRADE SUMMARY:")
        print(f"   Total Trades: {len(trades)}")
        print(f"   Winning Trades: {metrics.get('winning_trades', 0)}")
        print(f"   Losing Trades: {metrics.get('losing_trades', 0)}")
        print()
        
        # Display file outputs
        print("📁 OUTPUT FILES:")
        print(f"   Tear Sheet: {results.get('tear_sheet_path', 'N/A')}")
        print(f"   Data Points: {results.get('data_points', 0)}")
        print(f"   Backtest Period: {results.get('backtest_period', 'N/A')}")
        print()
        
        # Check if files exist
        tear_sheet_path = results.get('tear_sheet_path', '')
        if tear_sheet_path and Path(tear_sheet_path).exists():
            file_size = Path(tear_sheet_path).stat().st_size / 1024
            print(f"✅ Tear sheet file created: {file_size:.1f} KB")
        else:
            print("❌ Tear sheet file not found")
        
        # Check trade ledger files
        trade_ledger_csv = Path("data/backtest/trade_ledger.csv")
        trade_ledger_parquet = Path("data/backtest/trade_ledger.parquet")
        
        if trade_ledger_csv.exists():
            csv_size = trade_ledger_csv.stat().st_size / 1024
            print(f"✅ Trade ledger CSV created: {csv_size:.1f} KB")
        else:
            print("❌ Trade ledger CSV not found")
            
        if trade_ledger_parquet.exists():
            parquet_size = trade_ledger_parquet.stat().st_size / 1024
            print(f"✅ Trade ledger Parquet created: {parquet_size:.1f} KB")
        else:
            print("❌ Trade ledger Parquet not found")
        
        print()
        print("🎉 BACKTEST TEST COMPLETED SUCCESSFULLY!")
        
        return True
        
    else:
        print("❌ Backtest failed!")
        error = results.get('error', 'Unknown error')
        print(f"   Error: {error}")
        return False

def run_multiple_backtests():
    """Run multiple backtests with different configurations"""
    print("🎯 MULTIPLE BACKTEST CONFIGURATIONS")
    print("=" * 60)
    
    # Different strategy configurations to test
    configs = [
        {
            'name': 'Conservative Strategy',
            'config': {
                'lookback_period': 30,
                'entry_threshold': 0.01,
                'exit_threshold': 0.03,
                'max_position_size': 0.05
            }
        },
        {
            'name': 'Aggressive Strategy',
            'config': {
                'lookback_period': 10,
                'entry_threshold': 0.03,
                'exit_threshold': 0.08,
                'max_position_size': 0.2
            }
        },
        {
            'name': 'Balanced Strategy',
            'config': {
                'lookback_period': 20,
                'entry_threshold': 0.02,
                'exit_threshold': 0.05,
                'max_position_size': 0.1
            }
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"🚀 Running Backtest {i}: {config['name']}")
        print("-" * 40)
        
        # Initialize new backtest engine for each test
        logger = Logger()
        backtest_engine = DeterministicBacktestEngine(
            initial_capital=1000.0,
            logger=logger
        )
        
        # Run backtest
        result = backtest_engine.run_backtest(
            start_date="2024-01-01",
            end_date="2024-12-31",
            strategy_config=config['config']
        )
        
        if result.get('success'):
            metrics = result.get('metrics', {})
            print(f"✅ {config['name']} Results:")
            print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"   Total Trades: {metrics.get('total_trades', 0)}")
            
            results.append({
                'name': config['name'],
                'metrics': metrics,
                'success': True
            })
        else:
            print(f"❌ {config['name']} failed: {result.get('error', 'Unknown error')}")
            results.append({
                'name': config['name'],
                'success': False,
                'error': result.get('error', 'Unknown error')
            })
        
        print()
    
    # Summary
    print("📊 BACKTEST COMPARISON SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r.get('success')]
    
    if successful_results:
        print("Strategy Comparison:")
        print(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<10} {'Trades':<8}")
        print("-" * 70)
        
        for result in successful_results:
            metrics = result['metrics']
            print(f"{result['name']:<20} "
                  f"{metrics.get('total_return', 0):>8.2%} "
                  f"{metrics.get('sharpe_ratio', 0):>7.2f} "
                  f"{metrics.get('max_drawdown', 0):>9.2%} "
                  f"{metrics.get('win_rate', 0):>9.1f}% "
                  f"{metrics.get('total_trades', 0):>7}")
        
        # Find best performing strategy
        best_return = max(successful_results, key=lambda x: x['metrics'].get('total_return', 0))
        best_sharpe = max(successful_results, key=lambda x: x['metrics'].get('sharpe_ratio', 0))
        
        print()
        print(f"🏆 Best Return: {best_return['name']} ({best_return['metrics'].get('total_return', 0):.2%})")
        print(f"🏆 Best Sharpe: {best_sharpe['name']} ({best_sharpe['metrics'].get('sharpe_ratio', 0):.2f})")
    
    print()
    print("🎉 MULTIPLE BACKTEST COMPARISON COMPLETED!")
    
    return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Deterministic Backtests')
    parser.add_argument('--test', action='store_true', help='Run single backtest test')
    parser.add_argument('--multiple', action='store_true', help='Run multiple backtest configurations')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.test or args.all:
        print("Running single backtest test...")
        success = run_backtest_test()
        if not success:
            sys.exit(1)
    
    if args.multiple or args.all:
        print("\nRunning multiple backtest configurations...")
        results = run_multiple_backtests()
    
    if not any([args.test, args.multiple, args.all]):
        # Default: run single test
        print("Running default single backtest test...")
        success = run_backtest_test()
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
