#!/usr/bin/env python3
"""
📊 TRADE ANALYTICS DASHBOARD
Comprehensive trade analysis and reporting for the Ultra-Efficient XRP Trading System
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.analytics.trade_ledger import TradeLedgerManager

class TradeAnalyticsDashboard:
    """Comprehensive trade analytics dashboard"""
    
    def __init__(self, data_dir: str = "data/trades"):
        self.data_dir = Path(data_dir)
        self.trade_ledger = TradeLedgerManager(data_dir=str(self.data_dir))
        
    def display_summary(self):
        """Display comprehensive trade summary"""
        print("🎯 ULTRA-EFFICIENT XRP TRADING SYSTEM - TRADE ANALYTICS")
        print("=" * 80)
        
        analytics = self.trade_ledger.get_trade_analytics()
        
        if 'error' in analytics:
            print(f"❌ Error loading analytics: {analytics['error']}")
            return
        
        summary = analytics.get('summary', {})
        
        print(f"📊 TRADE SUMMARY")
        print(f"   Total Trades: {summary.get('total_trades', 0)}")
        print(f"   Live Trades: {summary.get('live_trades', 0)}")
        print(f"   Simulated Trades: {summary.get('simulated_trades', 0)}")
        print(f"   Total PnL: ${summary.get('total_pnl', 0):.4f}")
        print(f"   Total PnL %: {summary.get('total_pnl_percent', 0):.2f}%")
        print(f"   Avg PnL per Trade: ${summary.get('avg_pnl_per_trade', 0):.4f}")
        print(f"   Win Rate: {summary.get('win_rate', 0):.1f}%")
        print(f"   Wins: {summary.get('wins', 0)}")
        print(f"   Losses: {summary.get('losses', 0)}")
        print(f"   Max Drawdown: ${summary.get('max_drawdown', 0):.4f}")
        print()
        
    def display_strategy_performance(self):
        """Display strategy performance breakdown"""
        print("🎯 STRATEGY PERFORMANCE")
        print("=" * 50)
        
        analytics = self.trade_ledger.get_trade_analytics()
        strategy_perf = analytics.get('strategy_performance', {})
        
        if strategy_perf:
            for strategy, metrics in strategy_perf.items():
                print(f"📈 {strategy}:")
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, dict):
                            print(f"   {metric_name}: {value}")
                        else:
                            print(f"   {metric_name}: {value}")
                print()
        else:
            print("   No strategy performance data available")
        print()
        
    def display_hat_role_performance(self):
        """Display hat role performance breakdown"""
        print("🎩 HAT ROLE PERFORMANCE")
        print("=" * 50)
        
        analytics = self.trade_ledger.get_trade_analytics()
        hat_perf = analytics.get('hat_role_performance', {})
        
        if hat_perf:
            for role, metrics in hat_perf.items():
                print(f"🎯 {role}:")
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, dict):
                            print(f"   {metric_name}: {value}")
                        else:
                            print(f"   {metric_name}: {value}")
                print()
        else:
            print("   No hat role performance data available")
        print()
        
    def display_recent_trades(self, limit: int = 10):
        """Display recent trades"""
        print(f"📋 RECENT TRADES (Last {limit})")
        print("=" * 50)
        
        recent_trades = self.trade_ledger.get_recent_trades(limit)
        
        if recent_trades:
            for trade in recent_trades:
                print(f"🔄 {trade.get('trade_id', 'N/A')}")
                print(f"   Type: {trade.get('trade_type', 'N/A')}")
                print(f"   Side: {trade.get('side', 'N/A')}")
                print(f"   Quantity: {trade.get('quantity', 0):.3f} XRP")
                print(f"   Price: ${trade.get('price', 0):.4f}")
                print(f"   PnL: ${trade.get('profit_loss', 0):.4f}")
                print(f"   Win/Loss: {trade.get('win_loss', 'N/A')}")
                print(f"   Live: {'Yes' if trade.get('is_live_trade', False) else 'No'}")
                print(f"   Time: {trade.get('datetime_utc', 'N/A')}")
                print()
        else:
            print("   No recent trades available")
        print()
        
    def display_market_regime_performance(self):
        """Display market regime performance"""
        print("📈 MARKET REGIME PERFORMANCE")
        print("=" * 50)
        
        analytics = self.trade_ledger.get_trade_analytics()
        regime_perf = analytics.get('market_regime_performance', {})
        
        if regime_perf:
            for regime, metrics in regime_perf.items():
                print(f"🌊 {regime}:")
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, dict):
                            print(f"   {metric_name}: {value}")
                        else:
                            print(f"   {metric_name}: {value}")
                print()
        else:
            print("   No market regime performance data available")
        print()
        
    def export_analytics_report(self, format: str = "json"):
        """Export comprehensive analytics report"""
        print(f"📤 EXPORTING ANALYTICS REPORT ({format.upper()})")
        print("=" * 50)
        
        try:
            analytics = self.trade_ledger.get_trade_analytics()
            
            if format.lower() == "json":
                report_path = self.data_dir / f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_path, 'w') as f:
                    json.dump(analytics, f, indent=2, default=str)
                print(f"✅ Report exported to: {report_path}")
                
            elif format.lower() == "csv":
                # Export trades to CSV
                export_files = self.trade_ledger.export_trades("csv")
                if 'csv' in export_files:
                    print(f"✅ Trades exported to: {export_files['csv']}")
                else:
                    print("❌ Failed to export trades to CSV")
                    
            elif format.lower() == "parquet":
                # Export trades to Parquet
                export_files = self.trade_ledger.export_trades("parquet")
                if 'parquet' in export_files:
                    print(f"✅ Trades exported to: {export_files['parquet']}")
                else:
                    print("❌ Failed to export trades to Parquet")
                    
            else:
                print("❌ Unsupported format. Use 'json', 'csv', or 'parquet'")
                
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
        print()
        
    def display_daily_pnl(self, days: int = 7):
        """Display daily PnL for the last N days"""
        print(f"📅 DAILY PnL (Last {days} Days)")
        print("=" * 50)
        
        analytics = self.trade_ledger.get_trade_analytics()
        daily_pnl = analytics.get('daily_pnl', {})
        
        if daily_pnl:
            # Convert to DataFrame for better formatting
            df = pd.DataFrame(list(daily_pnl.items()), columns=['Date', 'PnL'])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date', ascending=False).head(days)
            
            for _, row in df.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d')
                pnl = row['PnL']
                pnl_str = f"${pnl:.4f}" if pnl >= 0 else f"-${abs(pnl):.4f}"
                print(f"   {date_str}: {pnl_str}")
        else:
            print("   No daily PnL data available")
        print()
        
    def run_full_dashboard(self):
        """Run the complete analytics dashboard"""
        print("🚀 Starting Trade Analytics Dashboard...")
        print()
        
        self.display_summary()
        self.display_strategy_performance()
        self.display_hat_role_performance()
        self.display_market_regime_performance()
        self.display_recent_trades(10)
        self.display_daily_pnl(7)
        
        print("📊 Dashboard complete!")
        print()
        
        # Ask user if they want to export
        try:
            export_choice = input("Export analytics report? (json/csv/parquet/n): ").lower()
            if export_choice in ['json', 'csv', 'parquet']:
                self.export_analytics_report(export_choice)
        except KeyboardInterrupt:
            print("\n👋 Dashboard closed by user")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trade Analytics Dashboard')
    parser.add_argument('--data-dir', default='data/trades', help='Data directory path')
    parser.add_argument('--export', choices=['json', 'csv', 'parquet'], help='Export format')
    parser.add_argument('--summary-only', action='store_true', help='Show summary only')
    
    args = parser.parse_args()
    
    dashboard = TradeAnalyticsDashboard(data_dir=args.data_dir)
    
    if args.export:
        dashboard.export_analytics_report(args.export)
    elif args.summary_only:
        dashboard.display_summary()
    else:
        dashboard.run_full_dashboard()

if __name__ == "__main__":
    main()
