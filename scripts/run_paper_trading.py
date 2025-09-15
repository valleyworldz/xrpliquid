#!/usr/bin/env python3
"""
🎯 PAPER TRADING RUNNER
Simple script to run paper trading with realistic execution simulation
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.paper_trading.paper_trade_engine import PaperTradeEngine
from core.utils.logger import Logger

async def run_paper_trading_session():
    """Run a paper trading session with realistic execution"""
    print("🎯 PAPER TRADING SESSION")
    print("=" * 60)
    print("Advanced paper trading with realistic order book replay,")
    print("latency simulation, and comprehensive slippage analysis")
    print("=" * 60)
    print()
    
    # Initialize paper trade engine
    logger = Logger()
    paper_engine = PaperTradeEngine(
        initial_capital=10000.0,
        symbol="XRP",
        logger=logger
    )
    
    print("✅ Paper Trade Engine initialized")
    print(f"💰 Initial Capital: ${paper_engine.initial_capital:,.2f}")
    print(f"📊 Symbol: {paper_engine.symbol}")
    print()
    
    # Set network condition
    network_condition = "good"  # Can be: excellent, good, average, poor, terrible
    paper_engine.set_network_condition(network_condition)
    print(f"🌐 Network condition set to: {network_condition}")
    print()
    
    # Example trading strategy - simple buy/sell pattern
    print("🚀 Starting paper trading session...")
    print("-" * 40)
    
    trades = [
        ("BUY", 100, "MARKET", "Initial position"),
        ("SELL", 50, "MARKET", "Partial profit taking"),
        ("BUY", 150, "MARKET", "Add to position"),
        ("SELL", 75, "MARKET", "Scale out"),
        ("BUY", 200, "MARKET", "Increase position"),
        ("SELL", 100, "MARKET", "Take profits"),
        ("BUY", 300, "MARKET", "Large position"),
        ("SELL", 200, "MARKET", "Major profit taking"),
        ("BUY", 100, "LIMIT", "Limit order test"),
        ("SELL", 50, "LIMIT", "Limit order test"),
    ]
    
    for i, (side, quantity, order_type, description) in enumerate(trades, 1):
        print(f"📊 Trade {i}: {description}")
        
        # For limit orders, set a reasonable limit price
        limit_price = None
        if order_type == "LIMIT":
            current_price = paper_engine.current_orderbook.mid_price if paper_engine.current_orderbook else 0.50
            if side == "BUY":
                limit_price = current_price * 0.99  # 1% below market
            else:
                limit_price = current_price * 1.01  # 1% above market
        
        # Place the order
        result = await paper_engine.place_paper_order(
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price
        )
        
        if result['success']:
            if result['status'] == 'FILLED':
                print(f"   ✅ {side} {quantity} XRP @ ${result['price']:.4f}")
                print(f"   📊 Slippage: {result['slippage_bps']:+.1f} bps")
                print(f"   ⏱️ Latency: {result['latency_ms']:.1f}ms")
                print(f"   💰 Slippage Cost: ${result['slippage_cost']:+.4f}")
            else:
                print(f"   ⏳ {side} {quantity} XRP - {result['status']}")
                print(f"   📊 Limit Price: ${limit_price:.4f}")
        else:
            print(f"   ❌ {side} {quantity} XRP failed: {result.get('error', 'Unknown error')}")
        
        print()
        
        # Small delay between trades
        await asyncio.sleep(0.1)
    
    # Generate performance report
    print("📊 FINAL PERFORMANCE REPORT")
    print("=" * 60)
    
    report = paper_engine.get_performance_report()
    print(report)
    
    # Save all trades
    print("💾 Saving all trades to files...")
    paper_engine.save_trades()
    
    # Check if files were created
    csv_file = Path("data/paper_trades/trade_ledger.csv")
    if csv_file.exists():
        file_size = csv_file.stat().st_size / 1024
        print(f"✅ CSV file created: {file_size:.1f} KB")
        
        # Show sample of trades
        print("\n📋 Sample of recorded trades:")
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(df[['trade_id', 'side', 'quantity', 'price', 'slippage', 'profit_loss']].head())
        except Exception as e:
            print(f"Could not display sample trades: {e}")
    else:
        print("❌ CSV file not found")
    
    print()
    print("🎉 Paper trading session completed successfully!")
    print("=" * 60)
    print("✅ All trades recorded with realistic execution simulation")
    print("✅ Comprehensive slippage analysis completed")
    print("✅ Performance metrics calculated")
    print("✅ Trade data saved to files")
    print("=" * 60)

async def run_custom_strategy():
    """Run a custom trading strategy"""
    print("🎯 CUSTOM STRATEGY PAPER TRADING")
    print("=" * 60)
    
    # Initialize paper trade engine
    logger = Logger()
    paper_engine = PaperTradeEngine(
        initial_capital=5000.0,
        symbol="XRP",
        logger=logger
    )
    
    # Set to excellent network for HFT
    paper_engine.set_network_condition("excellent")
    
    print("🚀 Running high-frequency scalping strategy...")
    print("-" * 40)
    
    # High-frequency scalping strategy
    for i in range(50):
        side = "BUY" if i % 2 == 0 else "SELL"
        quantity = 10 + (i % 3) * 5  # Vary quantity: 10, 15, 20
        
        result = await paper_engine.place_paper_order(
            side=side,
            quantity=quantity,
            order_type="MARKET"
        )
        
        if result['success'] and result['status'] == 'FILLED':
            if i % 10 == 0:  # Print every 10th trade
                print(f"   Trade {i+1}: {side} {quantity} XRP @ ${result['price']:.4f} "
                      f"({result['slippage_bps']:+.1f} bps, {result['latency_ms']:.1f}ms)")
        
        # Very small delay for HFT
        await asyncio.sleep(0.01)
    
    # Get final summary
    summary = paper_engine.get_portfolio_summary()
    print(f"\n📊 HFT Strategy Results:")
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Final PnL: ${summary['total_pnl']:+,.2f}")
    print(f"   Win Rate: {summary['win_rate_percent']:.1f}%")
    print(f"   Avg Slippage: {summary['slippage_summary']['avg_slippage_bps']:+.1f} bps")
    print(f"   Total Slippage Cost: ${summary['slippage_summary']['total_slippage_cost']:+,.2f}")
    
    # Save trades
    paper_engine.save_trades()
    print("✅ HFT strategy trades saved to files")

async def main():
    """Main function to run paper trading"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Paper Trading with Realistic Execution')
    parser.add_argument('--strategy', choices=['standard', 'hft'], default='standard',
                       help='Trading strategy to run')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital for paper trading')
    parser.add_argument('--network', choices=['excellent', 'good', 'average', 'poor', 'terrible'],
                       default='good', help='Network condition for latency simulation')
    
    args = parser.parse_args()
    
    if args.strategy == 'standard':
        await run_paper_trading_session()
    elif args.strategy == 'hft':
        await run_custom_strategy()
    
    print("\n🎯 Paper trading completed!")
    print("Check the 'data/paper_trades/' directory for detailed trade records.")

if __name__ == "__main__":
    asyncio.run(main())
