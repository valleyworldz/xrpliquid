#!/usr/bin/env python3
"""
📊 TRADE LEDGER VIEWER
Simple script to view and analyze trade ledger data
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.analytics.trade_ledger import TradeLedgerManager

def view_trade_ledger():
    """View trade ledger data"""
    print("📊 TRADE LEDGER VIEWER")
    print("=" * 50)
    
    # Initialize trade ledger
    trade_ledger = TradeLedgerManager(data_dir="data/trades")
    
    # Get ledger summary
    summary = trade_ledger.get_ledger_summary()
    
    print("📋 LEDGER INFO:")
    ledger_info = summary.get('ledger_info', {})
    print(f"   Total Trades: {ledger_info.get('total_trades', 0)}")
    print(f"   Data Directory: {ledger_info.get('data_dir', 'N/A')}")
    print(f"   CSV Path: {ledger_info.get('csv_path', 'N/A')}")
    print(f"   Parquet Path: {ledger_info.get('parquet_path', 'N/A')}")
    print(f"   Last Updated: {ledger_info.get('last_updated', 'N/A')}")
    print()
    
    # Get analytics
    analytics = summary.get('analytics', {})
    if 'summary' in analytics:
        print("📈 TRADE SUMMARY:")
        trade_summary = analytics['summary']
        print(f"   Total Trades: {trade_summary.get('total_trades', 0)}")
        print(f"   Live Trades: {trade_summary.get('live_trades', 0)}")
        print(f"   Simulated Trades: {trade_summary.get('simulated_trades', 0)}")
        print(f"   Total PnL: ${trade_summary.get('total_pnl', 0):.4f}")
        print(f"   Win Rate: {trade_summary.get('win_rate', 0):.1f}%")
        print(f"   Max Drawdown: ${trade_summary.get('max_drawdown', 0):.4f}")
        print()
    
    # Show recent trades
    recent_trades = summary.get('recent_trades', [])
    if recent_trades:
        print("🔄 RECENT TRADES:")
        for trade in recent_trades[:5]:  # Show last 5
            print(f"   {trade.get('trade_id', 'N/A')} - {trade.get('trade_type', 'N/A')} - {trade.get('side', 'N/A')} - ${trade.get('profit_loss', 0):.4f}")
        print()
    
    # Check if data files exist
    csv_path = Path(ledger_info.get('csv_path', ''))
    parquet_path = Path(ledger_info.get('parquet_path', ''))
    
    print("📁 DATA FILES:")
    print(f"   CSV exists: {'Yes' if csv_path.exists() else 'No'}")
    print(f"   Parquet exists: {'Yes' if parquet_path.exists() else 'No'}")
    
    if csv_path.exists():
        print(f"   CSV size: {csv_path.stat().st_size / 1024:.1f} KB")
    if parquet_path.exists():
        print(f"   Parquet size: {parquet_path.stat().st_size / 1024:.1f} KB")
    
    print()
    print("✅ Trade ledger viewer complete!")

if __name__ == "__main__":
    view_trade_ledger()
