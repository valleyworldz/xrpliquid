#!/usr/bin/env python3
"""
🎯 BACKTEST TEST WITH TRADES
Test the backtest system with more aggressive parameters to generate trades
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.backtesting.deterministic_backtest import DeterministicBacktestEngine
from core.utils.logger import Logger

def test_backtest_with_trades():
    """Test backtest with more aggressive parameters to generate trades"""
    print("🎯 BACKTEST TEST WITH TRADE GENERATION")
    print("=" * 60)
    
    # Initialize logger
    logger = Logger()
    
    # Initialize backtest engine
    print("🚀 Initializing backtest engine...")
    backtest_engine = DeterministicBacktestEngine(
        initial_capital=1000.0,
        logger=logger
    )
    
    # More aggressive strategy configuration to generate trades
    strategy_config = {
        'lookback_period': 10,      # Shorter lookback
        'entry_threshold': 0.005,   # Lower entry threshold (0.5%)
        'exit_threshold': 0.01,     # Lower exit threshold (1%)
        'max_position_size': 0.2    # Larger position size (20%)
    }
    
    print("📊 Aggressive Strategy Configuration:")
    for key, value in strategy_config.items():
        print(f"   {key}: {value}")
    print()
    
    # Run backtest
    print("🚀 Running backtest with aggressive parameters...")
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
        
        if trades:
            print("\n📋 SAMPLE TRADES:")
            for i, trade in enumerate(trades[:5]):  # Show first 5 trades
                print(f"   Trade {i+1}: {trade.get('trade_type', 'N/A')} "
                      f"{trade.get('quantity', 0):.3f} XRP @ ${trade.get('price', 0):.4f} "
                      f"PnL: ${trade.get('profit_loss', 0):.4f} "
                      f"({trade.get('win_loss', 'N/A')})")
        
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
        
        # Test trade ledger functionality
        print("\n🔍 TESTING TRADE LEDGER FUNCTIONALITY:")
        try:
            # Get trade analytics
            analytics = backtest_engine.trade_ledger.get_trade_analytics()
            if 'summary' in analytics:
                summary = analytics['summary']
                print(f"   Trade Ledger Total Trades: {summary.get('total_trades', 0)}")
                print(f"   Trade Ledger Live Trades: {summary.get('live_trades', 0)}")
                print(f"   Trade Ledger Simulated Trades: {summary.get('simulated_trades', 0)}")
                print(f"   Trade Ledger Total PnL: ${summary.get('total_pnl', 0):.4f}")
                print(f"   Trade Ledger Win Rate: {summary.get('win_rate', 0):.1f}%")
            
            # Get recent trades
            recent_trades = backtest_engine.trade_ledger.get_recent_trades(3)
            print(f"   Recent Trades Retrieved: {len(recent_trades)}")
            
            # Export functionality test
            export_files = backtest_engine.trade_ledger.export_trades("csv")
            if 'csv' in export_files:
                print(f"   Export Test: ✅ CSV export successful")
            else:
                print(f"   Export Test: ❌ CSV export failed")
                
        except Exception as e:
            print(f"   Trade Ledger Test Error: {e}")
        
        print()
        print("🎉 BACKTEST WITH TRADES TEST COMPLETED SUCCESSFULLY!")
        
        return True
        
    else:
        print("❌ Backtest failed!")
        error = results.get('error', 'Unknown error')
        print(f"   Error: {error}")
        return False

if __name__ == "__main__":
    test_backtest_with_trades()
