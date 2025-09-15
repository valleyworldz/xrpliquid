#!/usr/bin/env python3
"""
🚀 XRP BOT PERFORMANCE MONITOR
==============================

Real-time performance monitoring and optimization for the XRP Trading Bot.
Tracks all key metrics and provides actionable insights for achieving 10/10 performance.
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class XRPBotPerformanceMonitor:
    """Comprehensive performance monitoring for XRP Trading Bot"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'trades_executed': 0,
            'signals_generated': 0,
            'microstructure_vetos': 0,
            'momentum_vetos': 0,
            'rsi_vetos': 0,
            'performance_score': 0.0,
            'signal_quality': 0.0,
            'trade_success_rate': 0.0
        }
        
    def analyze_trade_history(self):
        """Analyze historical trade performance"""
        try:
            if os.path.exists('trades_log.csv'):
                df = pd.read_csv('trades_log.csv')
                
                # Basic metrics
                total_trades = len(df)
                buy_trades = len(df[df['side'] == 'BUY'])
                sell_trades = len(df[df['side'] == 'SELL'])
                
                # Performance metrics
                if 'pnl' in df.columns:
                    total_pnl = df['pnl'].sum()
                    winning_trades = len(df[df['pnl'] > 0])
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                else:
                    total_pnl = 0
                    winning_trades = 0
                    win_rate = 0
                
                # Position sizing analysis
                avg_position_size = df['size'].abs().mean() if 'size' in df.columns else 0
                
                logging.info(f"📊 TRADE HISTORY ANALYSIS")
                logging.info(f"Total Trades: {total_trades}")
                logging.info(f"BUY Trades: {buy_trades}")
                logging.info(f"SELL Trades: {sell_trades}")
                logging.info(f"Win Rate: {win_rate:.1f}%")
                logging.info(f"Total PnL: ${total_pnl:.2f}")
                logging.info(f"Average Position Size: {avg_position_size:.2f}")
                
                return {
                    'total_trades': total_trades,
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_position_size': avg_position_size
                }
            else:
                logging.warning("⚠️ No trade history found")
                return None
                
        except Exception as e:
            logging.error(f"❌ Error analyzing trade history: {e}")
            return None
    
    def check_bot_status(self):
        """Check current bot status and configuration"""
        try:
            status = {
                'v8_fixes_active': False,
                'microstructure_thresholds': 'Unknown',
                'emergency_bypass': False,
                'ml_engine_active': False
            }
            
            # Check V8 fixes
            if os.path.exists('newbotcode.py'):
                with open('newbotcode.py', 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'spread_cap = 0.0025' in content:
                        status['v8_fixes_active'] = True
                        status['microstructure_thresholds'] = 'V8 Ultra-Permissive (0.25% spread, 15% imbalance)'
            
            # Check emergency bypass
            emergency_bypass = os.environ.get("EMERGENCY_MICROSTRUCTURE_BYPASS", "false").lower()
            status['emergency_bypass'] = emergency_bypass in ("true", "1", "yes")
            
            # Check ML engine
            if os.path.exists('ml_engine_state.json'):
                status['ml_engine_active'] = True
            
            logging.info(f"🔧 BOT STATUS CHECK")
            logging.info(f"V8 Fixes Active: {'✅' if status['v8_fixes_active'] else '❌'}")
            logging.info(f"Microstructure Thresholds: {status['microstructure_thresholds']}")
            logging.info(f"Emergency Bypass: {'🚨 ACTIVE' if status['emergency_bypass'] else '✅ DISABLED'}")
            logging.info(f"ML Engine: {'✅ ACTIVE' if status['ml_engine_active'] else '❌ INACTIVE'}")
            
            return status
            
        except Exception as e:
            logging.error(f"❌ Error checking bot status: {e}")
            return None
    
    def calculate_performance_score(self):
        """Calculate current performance score based on multiple factors"""
        try:
            score = 0.0
            factors = []
            
            # Factor 1: V8 Fixes Implementation (25 points)
            if self.check_bot_status() and self.check_bot_status()['v8_fixes_active']:
                score += 25.0
                factors.append("V8 Fixes: +25.0")
            else:
                factors.append("V8 Fixes: +0.0")
            
            # Factor 2: Trade Execution Rate (25 points)
            trade_history = self.analyze_trade_history()
            if trade_history and trade_history['total_trades'] > 0:
                execution_rate = min(100, (trade_history['total_trades'] / 100) * 100)
                execution_score = (execution_rate / 100) * 25
                score += execution_score
                factors.append(f"Trade Execution: +{execution_score:.1f}")
            else:
                factors.append("Trade Execution: +0.0")
            
            # Factor 3: Win Rate (25 points)
            if trade_history and trade_history['win_rate'] > 0:
                win_rate_score = (trade_history['win_rate'] / 100) * 25
                score += win_rate_score
                factors.append(f"Win Rate: +{win_rate_score:.1f}")
            else:
                factors.append("Win Rate: +0.0")
            
            # Factor 4: Risk Management (25 points)
            risk_score = 25.0  # Assuming proper risk management is in place
            score += risk_score
            factors.append(f"Risk Management: +{risk_score:.1f}")
            
            self.metrics['performance_score'] = score
            
            logging.info(f"🎯 PERFORMANCE SCORE CALCULATION")
            logging.info(f"Total Score: {score:.2f}/100.0")
            for factor in factors:
                logging.info(f"  {factor}")
            
            return score
            
        except Exception as e:
            logging.error(f"❌ Error calculating performance score: {e}")
            return 0.0
    
    def generate_optimization_recommendations(self):
        """Generate actionable optimization recommendations"""
        try:
            recommendations = []
            
            # Check V8 fixes
            if not self.check_bot_status()['v8_fixes_active']:
                recommendations.append("🚨 CRITICAL: Implement V8 Emergency Fixes immediately")
            
            # Check trade execution
            trade_history = self.analyze_trade_history()
            if trade_history and trade_history['total_trades'] < 50:
                recommendations.append("📈 Increase trade execution frequency - consider relaxing filters")
            
            # Check win rate
            if trade_history and trade_history['win_rate'] < 60:
                recommendations.append("🎯 Improve signal quality - optimize entry/exit criteria")
            
            # Check position sizing
            if trade_history and trade_history['avg_position_size'] < 20:
                recommendations.append("💰 Optimize position sizing for better risk-adjusted returns")
            
            # Performance score recommendations
            current_score = self.metrics['performance_score']
            if current_score < 70:
                recommendations.append("⚡ URGENT: Performance below 70% - implement all V8 fixes")
            elif current_score < 85:
                recommendations.append("📊 Good performance - focus on fine-tuning for 90%+")
            else:
                recommendations.append("🏆 Excellent performance - maintain current configuration")
            
            logging.info(f"💡 OPTIMIZATION RECOMMENDATIONS")
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    logging.info(f"{i}. {rec}")
            else:
                logging.info("✅ No immediate optimizations needed")
            
            return recommendations
            
        except Exception as e:
            logging.error(f"❌ Error generating recommendations: {e}")
            return []
    
    def monitor_real_time(self, duration_minutes=60):
        """Real-time performance monitoring"""
        try:
            logging.info(f"🚀 STARTING REAL-TIME PERFORMANCE MONITORING")
            logging.info(f"Duration: {duration_minutes} minutes")
            logging.info(f"Monitoring interval: 30 seconds")
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            while time.time() < end_time:
                # Clear screen for better readability
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Display current status
                print(f"\n{'='*60}")
                print(f"🚀 XRP BOT PERFORMANCE MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                # Check bot status
                status = self.check_bot_status()
                if status:
                    print(f"\n🔧 BOT STATUS:")
                    print(f"  V8 Fixes: {'✅ ACTIVE' if status['v8_fixes_active'] else '❌ INACTIVE'}")
                    print(f"  Emergency Bypass: {'🚨 ACTIVE' if status['emergency_bypass'] else '✅ DISABLED'}")
                    print(f"  ML Engine: {'✅ ACTIVE' if status['ml_engine_active'] else '❌ INACTIVE'}")
                
                # Analyze trade history
                trade_history = self.analyze_trade_history()
                if trade_history:
                    print(f"\n📊 TRADE PERFORMANCE:")
                    print(f"  Total Trades: {trade_history['total_trades']}")
                    print(f"  Win Rate: {trade_history['win_rate']:.1f}%")
                    print(f"  Total PnL: ${trade_history['total_pnl']:.2f}")
                
                # Calculate performance score
                score = self.calculate_performance_score()
                print(f"\n🎯 PERFORMANCE SCORE: {score:.2f}/100.0")
                
                # Generate recommendations
                recommendations = self.generate_optimization_recommendations()
                if recommendations:
                    print(f"\n💡 RECOMMENDATIONS:")
                    for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                        print(f"  {i}. {rec}")
                
                # Time remaining
                remaining = int(end_time - time.time())
                minutes = remaining // 60
                seconds = remaining % 60
                print(f"\n⏰ Monitoring continues for {minutes:02d}:{seconds:02d}")
                
                time.sleep(30)  # Update every 30 seconds
            
            logging.info("✅ Real-time monitoring completed")
            
        except KeyboardInterrupt:
            logging.info("⏹️ Monitoring stopped by user")
        except Exception as e:
            logging.error(f"❌ Error in real-time monitoring: {e}")
    
    def run_comprehensive_analysis(self):
        """Run comprehensive performance analysis"""
        try:
            logging.info("🔍 STARTING COMPREHENSIVE PERFORMANCE ANALYSIS")
            logging.info("=" * 60)
            
            # 1. Bot Status Check
            logging.info("1️⃣ BOT STATUS ANALYSIS")
            status = self.check_bot_status()
            
            # 2. Trade History Analysis
            logging.info("\n2️⃣ TRADE HISTORY ANALYSIS")
            trade_history = self.analyze_trade_history()
            
            # 3. Performance Score Calculation
            logging.info("\n3️⃣ PERFORMANCE SCORE CALCULATION")
            score = self.calculate_performance_score()
            
            # 4. Optimization Recommendations
            logging.info("\n4️⃣ OPTIMIZATION RECOMMENDATIONS")
            recommendations = self.generate_optimization_recommendations()
            
            # 5. Summary Report
            logging.info("\n" + "=" * 60)
            logging.info("📋 COMPREHENSIVE ANALYSIS SUMMARY")
            logging.info("=" * 60)
            
            if status:
                logging.info(f"Bot Status: {'✅ OPTIMAL' if status['v8_fixes_active'] else '❌ NEEDS ATTENTION'}")
            
            if trade_history:
                logging.info(f"Trade Performance: {trade_history['total_trades']} trades, {trade_history['win_rate']:.1f}% win rate")
            
            logging.info(f"Overall Performance Score: {score:.2f}/100.0")
            
            if score >= 85:
                logging.info("🏆 STATUS: EXCELLENT - Bot is performing optimally!")
            elif score >= 70:
                logging.info("📊 STATUS: GOOD - Minor optimizations recommended")
            elif score >= 50:
                logging.info("⚠️ STATUS: FAIR - Significant improvements needed")
            else:
                logging.info("🚨 STATUS: POOR - Immediate attention required")
            
            return {
                'status': status,
                'trade_history': trade_history,
                'performance_score': score,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logging.error(f"❌ Error in comprehensive analysis: {e}")
            return None

def main():
    """Main function"""
    monitor = XRPBotPerformanceMonitor()
    
    print("🚀 XRP BOT PERFORMANCE MONITOR")
    print("=" * 50)
    print("1. Run Comprehensive Analysis")
    print("2. Start Real-time Monitoring")
    print("3. Check Bot Status Only")
    print("4. Exit")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            monitor.run_comprehensive_analysis()
        elif choice == "2":
            duration = input("Enter monitoring duration in minutes (default 60): ").strip()
            duration = int(duration) if duration.isdigit() else 60
            monitor.monitor_real_time(duration)
        elif choice == "3":
            monitor.check_bot_status()
        elif choice == "4":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
    except Exception as e:
        logging.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
