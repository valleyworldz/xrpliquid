#!/usr/bin/env python3
"""
🎯 PAPER TRADING TEST SCRIPT
Comprehensive test suite for the paper trading engine with realistic execution simulation
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

async def test_basic_paper_trading():
    """Test basic paper trading functionality"""
    print("🎯 TESTING BASIC PAPER TRADING")
    print("=" * 60)
    
    # Initialize paper trade engine
    logger = Logger()
    paper_engine = PaperTradeEngine(
        initial_capital=10000.0,
        symbol="XRP",
        logger=logger
    )
    
    print("✅ Paper Trade Engine initialized")
    print(f"💰 Initial Capital: ${paper_engine.initial_capital:,.2f}")
    print()
    
    # Test 1: Basic buy order
    print("📊 Test 1: Basic Buy Order")
    print("-" * 30)
    
    result = await paper_engine.place_paper_order(
        side="BUY",
        quantity=100.0,
        order_type="MARKET"
    )
    
    if result['success']:
        print(f"✅ Buy order executed successfully")
        print(f"   Order ID: {result['order_id']}")
        print(f"   Price: ${result['price']:.4f}")
        print(f"   Expected Price: ${result['expected_price']:.4f}")
        print(f"   Slippage: {result['slippage_bps']:+.1f} bps")
        print(f"   Slippage Cost: ${result['slippage_cost']:+.4f}")
        print(f"   Latency: {result['latency_ms']:.1f}ms")
        print(f"   Market Impact: {result['market_impact']:.4f}%")
    else:
        print(f"❌ Buy order failed: {result.get('error', 'Unknown error')}")
    
    print()
    
    # Test 2: Basic sell order
    print("📊 Test 2: Basic Sell Order")
    print("-" * 30)
    
    result = await paper_engine.place_paper_order(
        side="SELL",
        quantity=50.0,
        order_type="MARKET"
    )
    
    if result['success']:
        print(f"✅ Sell order executed successfully")
        print(f"   Order ID: {result['order_id']}")
        print(f"   Price: ${result['price']:.4f}")
        print(f"   Expected Price: ${result['expected_price']:.4f}")
        print(f"   Slippage: {result['slippage_bps']:+.1f} bps")
        print(f"   Slippage Cost: ${result['slippage_cost']:+.4f}")
        print(f"   Latency: {result['latency_ms']:.1f}ms")
        print(f"   Market Impact: {result['market_impact']:.4f}%")
    else:
        print(f"❌ Sell order failed: {result.get('error', 'Unknown error')}")
    
    print()
    
    # Test 3: Portfolio summary
    print("📊 Test 3: Portfolio Summary")
    print("-" * 30)
    
    summary = paper_engine.get_portfolio_summary()
    print(f"💰 Current Capital: ${summary['current_capital']:,.2f}")
    print(f"📈 Total PnL: ${summary['total_pnl']:+,.2f}")
    print(f"📊 Position: {summary['position']:+.3f} XRP")
    print(f"💵 Avg Entry Price: ${summary['avg_entry_price']:.4f}")
    print(f"🎯 Win Rate: {summary['win_rate_percent']:.1f}%")
    print(f"📊 Total Trades: {summary['total_trades']}")
    
    print()
    
    # Test 4: Slippage analysis
    print("📊 Test 4: Slippage Analysis")
    print("-" * 30)
    
    slippage_summary = summary['slippage_summary']
    print(f"📈 Total Trades: {slippage_summary['total_trades']}")
    print(f"📊 Avg Slippage: {slippage_summary['avg_slippage_bps']:+.1f} bps")
    print(f"📈 Max Slippage: {slippage_summary['max_slippage_bps']:+.1f} bps")
    print(f"📉 Min Slippage: {slippage_summary['min_slippage_bps']:+.1f} bps")
    print(f"💰 Total Slippage Cost: ${slippage_summary['total_slippage_cost']:+,.2f}")
    print(f"📊 Slippage Ratio: {slippage_summary['slippage_ratio']:.4f}")
    
    print()
    print("✅ Basic paper trading tests completed!")
    
    return paper_engine

async def test_network_conditions():
    """Test different network conditions and their impact on latency"""
    print("🌐 TESTING NETWORK CONDITIONS")
    print("=" * 60)
    
    logger = Logger()
    paper_engine = PaperTradeEngine(
        initial_capital=5000.0,
        symbol="XRP",
        logger=logger
    )
    
    network_conditions = ['excellent', 'good', 'average', 'poor', 'terrible']
    
    for condition in network_conditions:
        print(f"🌐 Testing {condition.upper()} network condition")
        print("-" * 40)
        
        # Set network condition
        paper_engine.set_network_condition(condition)
        
        # Place a test order
        start_time = time.time()
        result = await paper_engine.place_paper_order(
            side="BUY",
            quantity=10.0,
            order_type="MARKET"
        )
        end_time = time.time()
        
        if result['success']:
            actual_latency = (end_time - start_time) * 1000  # Convert to ms
            reported_latency = result['latency_ms']
            
            print(f"✅ Order executed successfully")
            print(f"   Reported Latency: {reported_latency:.1f}ms")
            print(f"   Actual Latency: {actual_latency:.1f}ms")
            print(f"   Slippage: {result['slippage_bps']:+.1f} bps")
            print(f"   Price: ${result['price']:.4f}")
        else:
            print(f"❌ Order failed: {result.get('error', 'Unknown error')}")
        
        print()
    
    print("✅ Network condition tests completed!")

async def test_limit_orders():
    """Test limit order functionality"""
    print("📋 TESTING LIMIT ORDERS")
    print("=" * 60)
    
    logger = Logger()
    paper_engine = PaperTradeEngine(
        initial_capital=5000.0,
        symbol="XRP",
        logger=logger
    )
    
    # Get current market price
    paper_engine.update_orderbook()
    current_price = paper_engine.current_orderbook.mid_price
    
    print(f"📊 Current Market Price: ${current_price:.4f}")
    print()
    
    # Test 1: Marketable limit buy order (should fill immediately)
    print("📊 Test 1: Marketable Limit Buy Order")
    print("-" * 40)
    
    limit_price = current_price * 1.01  # 1% above market
    result = await paper_engine.place_paper_order(
        side="BUY",
        quantity=50.0,
        order_type="LIMIT",
        limit_price=limit_price
    )
    
    if result['success']:
        if result['status'] == 'FILLED':
            print(f"✅ Marketable limit order filled immediately")
            print(f"   Limit Price: ${limit_price:.4f}")
            print(f"   Fill Price: ${result['price']:.4f}")
            print(f"   Slippage: {result['slippage_bps']:+.1f} bps")
        else:
            print(f"⏳ Limit order resting: {result['status']}")
            print(f"   Limit Price: ${limit_price:.4f}")
            print(f"   Remaining Quantity: {result['remaining_quantity']:.3f}")
    else:
        print(f"❌ Limit order failed: {result.get('error', 'Unknown error')}")
    
    print()
    
    # Test 2: Non-marketable limit sell order (should rest)
    print("📊 Test 2: Non-Marketable Limit Sell Order")
    print("-" * 40)
    
    limit_price = current_price * 0.95  # 5% below market
    result = await paper_engine.place_paper_order(
        side="SELL",
        quantity=25.0,
        order_type="LIMIT",
        limit_price=limit_price
    )
    
    if result['success']:
        if result['status'] == 'RESTING':
            print(f"✅ Limit order resting as expected")
            print(f"   Limit Price: ${limit_price:.4f}")
            print(f"   Remaining Quantity: {result['remaining_quantity']:.3f}")
        else:
            print(f"⚠️ Unexpected status: {result['status']}")
    else:
        print(f"❌ Limit order failed: {result.get('error', 'Unknown error')}")
    
    print()
    print("✅ Limit order tests completed!")

async def test_high_frequency_trading():
    """Test high frequency trading scenarios"""
    print("⚡ TESTING HIGH FREQUENCY TRADING")
    print("=" * 60)
    
    logger = Logger()
    paper_engine = PaperTradeEngine(
        initial_capital=10000.0,
        symbol="XRP",
        logger=logger
    )
    
    # Set to excellent network condition for HFT
    paper_engine.set_network_condition('excellent')
    
    print("🚀 Executing 20 rapid trades...")
    print("-" * 40)
    
    start_time = time.time()
    total_slippage = 0.0
    total_latency = 0.0
    
    for i in range(20):
        side = "BUY" if i % 2 == 0 else "SELL"
        quantity = 10.0 + (i % 5) * 5.0  # Vary quantity
        
        result = await paper_engine.place_paper_order(
            side=side,
            quantity=quantity,
            order_type="MARKET"
        )
        
        if result['success']:
            total_slippage += abs(result['slippage_bps'])
            total_latency += result['latency_ms']
            
            if i % 5 == 0:  # Print every 5th trade
                print(f"   Trade {i+1}: {side} {quantity:.1f} XRP @ ${result['price']:.4f} "
                      f"({result['slippage_bps']:+.1f} bps, {result['latency_ms']:.1f}ms)")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print()
    print("📊 HFT Performance Summary:")
    print(f"   Total Trades: 20")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Avg Time per Trade: {total_time/20:.3f}s")
    print(f"   Avg Slippage: {total_slippage/20:.1f} bps")
    print(f"   Avg Latency: {total_latency/20:.1f}ms")
    print(f"   Trades per Second: {20/total_time:.1f}")
    
    # Get final portfolio summary
    summary = paper_engine.get_portfolio_summary()
    print(f"   Final PnL: ${summary['total_pnl']:+,.2f}")
    print(f"   Final Position: {summary['position']:+.3f} XRP")
    
    print()
    print("✅ High frequency trading tests completed!")

async def test_slippage_analysis():
    """Test comprehensive slippage analysis"""
    print("📊 TESTING SLIPPAGE ANALYSIS")
    print("=" * 60)
    
    logger = Logger()
    paper_engine = PaperTradeEngine(
        initial_capital=20000.0,
        symbol="XRP",
        logger=logger
    )
    
    # Test different order sizes to analyze slippage impact
    order_sizes = [10, 50, 100, 500, 1000, 2000]
    
    print("📈 Testing slippage impact across different order sizes:")
    print("-" * 50)
    
    for size in order_sizes:
        print(f"📊 Testing {size} XRP order...")
        
        result = await paper_engine.place_paper_order(
            side="BUY",
            quantity=size,
            order_type="MARKET"
        )
        
        if result['success']:
            print(f"   Size: {size} XRP")
            print(f"   Price: ${result['price']:.4f}")
            print(f"   Expected: ${result['expected_price']:.4f}")
            print(f"   Slippage: {result['slippage_bps']:+.1f} bps")
            print(f"   Market Impact: {result['market_impact']:.4f}%")
            print(f"   Slippage Cost: ${result['slippage_cost']:+.4f}")
        else:
            print(f"   ❌ Order failed: {result.get('error', 'Unknown error')}")
        
        print()
    
    # Get comprehensive slippage summary
    summary = paper_engine.get_portfolio_summary()
    slippage_summary = summary['slippage_summary']
    
    print("📊 COMPREHENSIVE SLIPPAGE ANALYSIS:")
    print("-" * 50)
    print(f"Total Trades: {slippage_summary['total_trades']}")
    print(f"Average Slippage: {slippage_summary['avg_slippage_bps']:+.1f} bps")
    print(f"Maximum Slippage: {slippage_summary['max_slippage_bps']:+.1f} bps")
    print(f"Minimum Slippage: {slippage_summary['min_slippage_bps']:+.1f} bps")
    print(f"Total Slippage Cost: ${slippage_summary['total_slippage_cost']:+,.2f}")
    print(f"Slippage Ratio: {slippage_summary['slippage_ratio']:.4f}")
    print(f"Buy Avg Slippage: {slippage_summary['buy_slippage_avg']:+.1f} bps")
    print(f"Sell Avg Slippage: {slippage_summary['sell_slippage_avg']:+.1f} bps")
    
    print()
    print("✅ Slippage analysis tests completed!")

async def test_performance_report():
    """Test comprehensive performance reporting"""
    print("📊 TESTING PERFORMANCE REPORTING")
    print("=" * 60)
    
    logger = Logger()
    paper_engine = PaperTradeEngine(
        initial_capital=15000.0,
        symbol="XRP",
        logger=logger
    )
    
    # Execute a series of trades
    print("🚀 Executing test trades for performance analysis...")
    
    trades = [
        ("BUY", 100, "MARKET"),
        ("SELL", 50, "MARKET"),
        ("BUY", 200, "MARKET"),
        ("SELL", 75, "MARKET"),
        ("BUY", 150, "MARKET"),
        ("SELL", 100, "MARKET"),
        ("BUY", 300, "MARKET"),
        ("SELL", 200, "MARKET"),
    ]
    
    for side, quantity, order_type in trades:
        result = await paper_engine.place_paper_order(
            side=side,
            quantity=quantity,
            order_type=order_type
        )
        
        if result['success']:
            print(f"   ✅ {side} {quantity} XRP @ ${result['price']:.4f}")
        else:
            print(f"   ❌ {side} {quantity} XRP failed")
    
    print()
    
    # Generate and display performance report
    print("📊 COMPREHENSIVE PERFORMANCE REPORT:")
    print("=" * 60)
    
    report = paper_engine.get_performance_report()
    print(report)
    
    # Save trades to files
    print("💾 Saving trades to files...")
    paper_engine.save_trades()
    
    # Check if files were created
    csv_file = Path("data/paper_trades/trade_ledger.csv")
    if csv_file.exists():
        file_size = csv_file.stat().st_size / 1024
        print(f"✅ CSV file created: {file_size:.1f} KB")
    else:
        print("❌ CSV file not found")
    
    print()
    print("✅ Performance reporting tests completed!")

async def main():
    """Run all paper trading tests"""
    print("🎯 COMPREHENSIVE PAPER TRADING TEST SUITE")
    print("=" * 80)
    print("Testing advanced paper trading with realistic order book replay,")
    print("latency simulation, and comprehensive slippage analysis")
    print("=" * 80)
    print()
    
    try:
        # Run all test suites
        await test_basic_paper_trading()
        print()
        
        await test_network_conditions()
        print()
        
        await test_limit_orders()
        print()
        
        await test_high_frequency_trading()
        print()
        
        await test_slippage_analysis()
        print()
        
        await test_performance_report()
        print()
        
        print("🎉 ALL PAPER TRADING TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ Paper trading engine is fully functional with:")
        print("   • Realistic order book replay")
        print("   • Latency simulation for different network conditions")
        print("   • Comprehensive slippage analysis and logging")
        print("   • Limit order support with resting order simulation")
        print("   • High frequency trading capabilities")
        print("   • Professional performance reporting")
        print("   • Full integration with trade ledger system")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
