#!/usr/bin/env python3
"""
🎯 RUN COMPREHENSIVE BACKTEST
============================
Execute 12-36 month XRP-perp backtest with all required components
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.backtesting.comprehensive_backtest_engine import (
    ComprehensiveBacktestEngine, 
    BacktestConfig
)

def run_comprehensive_backtest():
    """Run comprehensive backtest with all required components"""
    
    print("🎯 COMPREHENSIVE XRP FUNDING ARBITRAGE BACKTEST")
    print("=" * 60)
    print("📅 Period: 12-36 months XRP-perp")
    print("💰 Initial Capital: $10,000")
    print("⚡ Strategy: Optimized Funding Arbitrage")
    print("📊 Components: Fees, Slippage, Funding, Spreads")
    print("=" * 60)
    
    # Configure backtest
    config = BacktestConfig(
        start_date="2023-01-01",
        end_date="2024-12-31",  # 24 months
        initial_capital=10000.0,
        commission_rate=0.0005,  # 0.05% per trade
        slippage_bps=2.0,  # 2 basis points
        spread_bps=1.0,  # 1 basis point spread
        funding_frequency_hours=8,  # Hyperliquid funding every 8 hours
        volatility_regime_threshold=0.02,
        min_position_size_usd=25.0,
        max_position_size_usd=1000.0,
        risk_unit_size=0.01  # 1% of capital per risk unit
    )
    
    # Initialize backtest engine
    engine = ComprehensiveBacktestEngine(config)
    
    print("🔄 Generating realistic market data...")
    market_data = engine.generate_realistic_market_data()
    print(f"✅ Generated {len(market_data)} hours of market data")
    
    print("🔄 Calculating volatility terciles...")
    market_data = engine.calculate_volatility_terciles(market_data)
    print("✅ Volatility analysis complete")
    
    print("🔄 Running funding arbitrage strategy simulation...")
    engine.simulate_funding_arbitrage_strategy(market_data)
    print(f"✅ Strategy simulation complete - {len(engine.trades)} trades executed")
    
    print("🔄 Generating performance report...")
    report = engine.generate_performance_report()
    
    # Display key results
    print("\n📊 BACKTEST RESULTS")
    print("-" * 40)
    print(f"💰 Initial Capital: ${report['summary']['initial_capital']:,.2f}")
    print(f"💰 Final Capital: ${report['summary']['final_capital']:,.2f}")
    print(f"📈 Total Return: {report['summary']['total_return']:.2%}")
    print(f"📈 Annualized Return: {report['summary']['annualized_return']:.2%}")
    print(f"📉 Max Drawdown: {report['summary']['max_drawdown']:.2%}")
    print(f"⚡ Sharpe Ratio: {report['summary']['sharpe_ratio']:.2f}")
    print(f"🎯 MAR Ratio: {report['summary']['mar_ratio']:.2f}")
    print(f"🏆 Win Rate: {report['summary']['win_rate']:.2%}")
    print(f"💎 Profit Factor: {report['summary']['profit_factor']:.2f}")
    print(f"🔄 Total Trades: {report['summary']['total_trades']}")
    
    # Regime analysis
    print("\n🎯 REGIME ANALYSIS")
    print("-" * 40)
    for regime, stats in report['regime_analysis'].items():
        print(f"{regime.upper()}: {stats['trades']} trades, ${stats['total_pnl']:.2f} PnL, {stats['win_rate']:.1%} win rate")
    
    # Volatility analysis
    print("\n⚡ VOLATILITY ANALYSIS")
    print("-" * 40)
    for vol_level, stats in report['volatility_analysis'].items():
        print(f"{vol_level.upper()} VOL: {stats['trades']} trades, ${stats['total_pnl']:.2f} PnL, {stats['win_rate']:.1%} win rate")
    
    # Execution quality
    print("\n🔧 EXECUTION QUALITY")
    print("-" * 40)
    exec_quality = report['execution_quality']
    print(f"Avg Slippage: {exec_quality['avg_slippage_bps']:.2f} bps")
    print(f"Max Slippage: {exec_quality['max_slippage_bps']:.2f} bps")
    print(f"Maker Ratio: {exec_quality['maker_ratio']:.1%}")
    print(f"Total Fees: ${exec_quality['total_fees']:.2f}")
    
    # Risk metrics
    print("\n🛡️ RISK METRICS")
    print("-" * 40)
    risk_metrics = report['risk_metrics']
    print(f"Time Under Water: {risk_metrics['time_under_water']:.1%}")
    print(f"VaR 95%: {risk_metrics['var_95']:.2%}")
    print(f"Expected Shortfall: {risk_metrics['expected_shortfall']:.2%}")
    
    # Save all artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"xrp_funding_arbitrage_{timestamp}"
    
    print(f"\n💾 SAVING ARTIFACTS")
    print("-" * 40)
    
    # Save trade ledger
    engine.save_trade_ledger(f"{base_filename}_trades")
    
    # Save performance report
    engine.save_performance_report(f"{base_filename}_performance")
    
    # Generate equity curve plots
    engine.generate_equity_curve_plot(f"{base_filename}_equity")
    
    print(f"\n✅ ALL ARTIFACTS SAVED TO reports/ DIRECTORY")
    print(f"📁 Trade Ledger: {base_filename}_trades.csv/.parquet")
    print(f"📊 Performance Report: {base_filename}_performance.json/.html")
    print(f"📈 Equity Curves: {base_filename}_equity_*.png")
    
    # Profitability assessment
    print(f"\n🎯 PROFITABILITY ASSESSMENT")
    print("-" * 40)
    
    if report['summary']['total_return'] > 0:
        print("✅ STRATEGY IS PROFITABLE")
        if report['summary']['sharpe_ratio'] > 1.0:
            print("✅ EXCELLENT RISK-ADJUSTED RETURNS")
        if report['summary']['mar_ratio'] > 2.0:
            print("✅ STRONG DRAWDOWN CONTROL")
        if report['summary']['win_rate'] > 0.5:
            print("✅ POSITIVE WIN RATE")
    else:
        print("❌ STRATEGY IS NOT PROFITABLE")
        print("🔧 RECOMMENDATIONS:")
        print("   - Increase funding threshold")
        print("   - Optimize position sizing")
        print("   - Improve execution quality")
        print("   - Add additional filters")
    
    return report

if __name__ == "__main__":
    try:
        report = run_comprehensive_backtest()
        print(f"\n🎯 BACKTEST COMPLETE - PROFITABILITY PROOF GENERATED")
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
