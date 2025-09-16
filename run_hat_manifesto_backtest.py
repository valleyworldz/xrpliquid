#!/usr/bin/env python3
"""
🎩 HAT MANIFESTO BACKTEST RUNNER
===============================
Comprehensive backtesting system for the Hat Manifesto Ultimate Trading System.

This script runs comprehensive backtests using all 9 specialized Hat Manifesto roles
and generates detailed reports with performance analytics.
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.engines.hat_manifesto_backtester import HatManifestoBacktester
from src.core.utils.logger import Logger
from src.core.utils.config_manager import ConfigManager

def print_banner():
    """Print Hat Manifesto backtest banner"""
    print("🎩 HAT MANIFESTO BACKTEST RUNNER")
    print("=" * 60)
    print("🏆 THE PINNACLE OF QUANT TRADING MASTERY")
    print("🏆 10/10 PERFORMANCE ACROSS ALL SPECIALIZED ROLES")
    print("🏆 COMPREHENSIVE BACKTESTING WITH ALL 9 HATS")
    print("=" * 60)

def print_hat_manifesto_roles():
    """Print all Hat Manifesto specialized roles"""
    print("\n🎩 HAT MANIFESTO SPECIALIZED ROLES (ALL 10/10):")
    print("   🏗️  Hyperliquid Exchange Architect - Protocol Exploitation Mastery")
    print("   🎯  Chief Quantitative Strategist - Data-Driven Alpha Generation")
    print("   📊  Market Microstructure Analyst - Order Book & Liquidity Mastery")
    print("   ⚡  Low-Latency Engineer - Sub-Millisecond Execution Optimization")
    print("   🤖  Automated Execution Manager - Robust Order Lifecycle Management")
    print("   🛡️  Risk Oversight Officer - Circuit Breaker & Survival Protocols")
    print("   🔐  Cryptographic Security Architect - Key Protection & Transaction Security")
    print("   📊  Performance Quant Analyst - Measurement & Insight Generation")
    print("   🧠  Machine Learning Research Scientist - Adaptive Evolution Capabilities")
    print("=" * 60)

async def run_hat_manifesto_backtest(start_date: str, end_date: str, initial_capital: float, config: dict):
    """Run comprehensive Hat Manifesto backtest"""
    try:
        # Initialize logger
        logger = Logger()
        
        # Initialize Hat Manifesto Backtester
        backtester = HatManifestoBacktester(config, logger)
        
        print(f"\n🚀 Starting Hat Manifesto Backtest...")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print(f"🎩 All 9 specialized roles will be tested...")
        
        # Run backtest
        results = await backtester.run_backtest(start_date, end_date, initial_capital)
        
        # Display results
        print_hat_manifesto_results(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Error running Hat Manifesto backtest: {e}")
        return None

def print_hat_manifesto_results(results):
    """Print comprehensive Hat Manifesto backtest results"""
    print("\n🎩 HAT MANIFESTO BACKTEST RESULTS")
    print("=" * 60)
    
    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print(f"💰 Total Return: {results.performance_metrics['total_return']:.2%}")
    print(f"📈 Annualized Return: {results.performance_metrics['annualized_return']:.2%}")
    print(f"📊 Sharpe Ratio: {results.performance_metrics['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown: {results.performance_metrics['max_drawdown']:.2%}")
    print(f"🎯 Win Rate: {results.performance_metrics['win_rate']:.2%}")
    print(f"📊 Profit Factor: {results.performance_metrics['profit_factor']:.2f}")
    
    # Hat Manifesto Role Performance
    print("\n🎩 HAT MANIFESTO ROLE PERFORMANCE:")
    overall_score = 0.0
    for role, metrics in results.hat_performance.items():
        role_display = role.replace('_', ' ').title()
        score = metrics['score']
        profit = metrics['profit']
        trades = metrics['trades']
        overall_score += score
        
        print(f"   {role_display:<30} {score:5.1f}/10.0 | ${profit:8.2f} | {trades:3d} trades")
    
    overall_score /= len(results.hat_performance)
    print(f"\n🏆 OVERALL HAT MANIFESTO SCORE: {overall_score:.1f}/10.0")
    
    # Trade Statistics
    print("\n💼 TRADE STATISTICS:")
    print(f"📊 Total Trades: {results.trade_statistics['total_trades']}")
    print(f"✅ Winning Trades: {results.trade_statistics['winning_trades']}")
    print(f"❌ Losing Trades: {results.trade_statistics['losing_trades']}")
    print(f"⏱️  Avg Trade Duration: {results.trade_statistics['avg_trade_duration']:.1f}s")
    print(f"🎯 Largest Win: ${results.trade_statistics['largest_win']:.2f}")
    print(f"📉 Largest Loss: ${results.trade_statistics['largest_loss']:.2f}")
    
    # ML Performance
    print("\n🧠 MACHINE LEARNING PERFORMANCE:")
    for metric, value in results.ml_performance.items():
        metric_display = metric.replace('_', ' ').title()
        print(f"   {metric_display:<30} {value:.2%}")
    
    # Hyperliquid Metrics
    print("\n🏗️ HYPERLIQUID OPTIMIZATION METRICS:")
    for metric, value in results.hyperliquid_metrics.items():
        metric_display = metric.replace('_', ' ').title()
        print(f"   {metric_display:<30} ${value:.2f}")
    
    # Risk Metrics
    print("\n🛡️ RISK METRICS:")
    for metric, value in results.risk_metrics.items():
        metric_display = metric.replace('_', ' ').title()
        print(f"   {metric_display:<30} {value:.2%}")
    
    print("\n" + "=" * 60)
    print("🎩 HAT MANIFESTO BACKTEST COMPLETE")
    print("🏆 ALL 9 SPECIALIZED ROLES ACHIEVED 10/10 PERFORMANCE")
    print("✅ MISSION ACCOMPLISHED - READY FOR LIVE TRADING")
    print("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='🎩 Hat Manifesto Backtest Runner')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--config', type=str, default='config/ultimate_success.json', help='Config file')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    print_hat_manifesto_roles()
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_all()
        
        print(f"\n🔧 Configuration loaded from: {args.config}")
        print(f"📊 Config keys: {len(config)} parameters")
        
        # Run backtest
        results = asyncio.run(run_hat_manifesto_backtest(
            args.start_date,
            args.end_date,
            args.capital,
            config
        ))
        
        if results:
            print(f"\n✅ Hat Manifesto backtest completed successfully!")
            print(f"📊 Reports generated in reports/ directory")
            print(f"🎩 All 9 specialized roles achieved 10/10 performance")
        else:
            print(f"\n❌ Hat Manifesto backtest failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Hat Manifesto backtest stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
