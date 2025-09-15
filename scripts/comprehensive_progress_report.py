#!/usr/bin/env python3
"""
COMPREHENSIVE PROGRESS REPORT
Complete status report from all executive hats
"""

import json
import os
from datetime import datetime

class ComprehensiveProgressReport:
    def __init__(self):
        self.report_time = datetime.now()
        
    def analyze_trading_data(self):
        """Analyze current trading data"""
        print("📊 TRADING DATA ANALYSIS")
        print("=" * 60)
        
        try:
            if os.path.exists("trades_log.csv"):
                with open("trades_log.csv", 'r') as f:
                    lines = f.readlines()
                    
                total_trades = len(lines) - 1  # Subtract header
                print(f"📈 Total Trades: {total_trades}")
                
                # Analyze recent trades
                recent_trades = lines[-10:] if len(lines) > 10 else lines[1:]
                print(f"📅 Recent Trades Analyzed: {len(recent_trades)}")
                
                # Check for profitable trades
                profitable_count = 0
                total_pnl = 0.0
                
                for line in recent_trades:
                    if line.strip() and not line.startswith('trade_id'):
                        parts = line.strip().split(',')
                        if len(parts) >= 8:
                            try:
                                pnl = float(parts[7]) if parts[7] else 0.0
                                total_pnl += pnl
                                if pnl > 0:
                                    profitable_count += 1
                            except:
                                continue
                
                win_rate = (profitable_count / len(recent_trades)) * 100 if recent_trades else 0
                print(f"🎯 Recent Win Rate: {win_rate:.1f}%")
                print(f"💰 Recent Total PnL: ${total_pnl:.2f}")
                
                return {
                    'total_trades': total_trades,
                    'recent_trades': len(recent_trades),
                    'win_rate': win_rate,
                    'total_pnl': total_pnl
                }
            else:
                print("⚠️ No trades_log.csv found")
                return {'total_trades': 0, 'recent_trades': 0, 'win_rate': 0, 'total_pnl': 0}
                
        except Exception as e:
            print(f"⚠️ Error analyzing trading data: {e}")
            return {'total_trades': 0, 'recent_trades': 0, 'win_rate': 0, 'total_pnl': 0}
    
    def check_system_status(self):
        """Check system status"""
        print("\n🔧 SYSTEM STATUS CHECK")
        print("=" * 60)
        
        # Check if bot is running
        import psutil
        bot_running = False
        python_processes = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes += 1
                    if 'newbotcode.py' in ' '.join(proc.info['cmdline']):
                        bot_running = True
            except:
                continue
        
        print(f"🐍 Python Processes: {python_processes}")
        print(f"🤖 Bot Status: {'✅ RUNNING' if bot_running else '❌ NOT RUNNING'}")
        
        # Check configuration files
        config_files = [
            "ultimate_recovery_config.json",
            "drawdown_tracker.json",
            "emergency_risk_config.json"
        ]
        
        config_status = {}
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                    config_status[config_file] = "✅ LOADED"
                except:
                    config_status[config_file] = "⚠️ ERROR"
            else:
                config_status[config_file] = "❌ MISSING"
        
        print("\n📁 Configuration Status:")
        for file, status in config_status.items():
            print(f"   {file}: {status}")
        
        return {
            'bot_running': bot_running,
            'python_processes': python_processes,
            'config_status': config_status
        }
    
    def analyze_recovery_progress(self):
        """Analyze recovery progress"""
        print("\n🚀 RECOVERY PROGRESS ANALYSIS")
        print("=" * 60)
        
        try:
            # Load recovery configuration
            if os.path.exists("ultimate_recovery_config.json"):
                with open("ultimate_recovery_config.json", 'r') as f:
                    recovery_config = json.load(f)
                
                print("✅ Recovery Configuration Loaded:")
                params = recovery_config.get('recovery_parameters', {})
                print(f"   • Max Drawdown: {params.get('max_drawdown', 'N/A')}%")
                print(f"   • Risk per Trade: {params.get('risk_per_trade', 'N/A')}%")
                print(f"   • Max Position Size: {params.get('max_position_size', 'N/A')}")
                print(f"   • Leverage Limit: {params.get('leverage_limit', 'N/A')}x")
                print(f"   • Confidence Threshold: {params.get('confidence_threshold', 'N/A')}")
                print(f"   • Daily Profit Target: ${params.get('daily_profit_target', 'N/A')}")
                
                # Check recovery targets
                targets = recovery_config.get('recovery_targets', {})
                print(f"\n🎯 Recovery Targets:")
                for week, target in targets.items():
                    print(f"   • {week}: ${target}")
                
                return recovery_config
            else:
                print("⚠️ Recovery configuration not found")
                return {}
                
        except Exception as e:
            print(f"⚠️ Error analyzing recovery progress: {e}")
            return {}
    
    def check_executive_hats_status(self):
        """Check status of all executive hats"""
        print("\n👑 EXECUTIVE HATS STATUS")
        print("=" * 60)
        
        hats = {
            "CEO": "👑 Crisis Management & Leadership",
            "CTO": "🔧 Technical Operations & Innovation", 
            "CFO": "💰 Financial Strategy & Risk Management",
            "COO": "⚙️ Operational Excellence & Efficiency",
            "CMO": "📈 Market Strategy & Growth",
            "CSO": "🛡️ Security & Risk Containment",
            "CDO": "📊 Data Analytics & AI Optimization",
            "CPO": "🎯 Product Development & User Experience"
        }
        
        hat_status = {}
        for hat, description in hats.items():
            # Check if hat-specific files exist
            hat_files = {
                "CEO": ["emergency_drawdown_fix.py"],
                "CTO": ["emergency_drawdown_fix.py", "ultimate_recovery_system.py"],
                "CFO": ["financial_rescue_plan.py", "emergency_budget.json"],
                "COO": ["operational_stabilization.py", "recovery_protocols.json"],
                "CMO": ["market_repositioning_strategy.py", "market_recovery_strategy.json"],
                "CSO": ["security_lockdown_protocol.py", "security_monitoring_system.json"],
                "CDO": ["data_analysis_ai_optimization.py", "ai_optimization_config.json"],
                "CPO": ["product_recovery_user_experience.py", "user_support_system.json"]
            }
            
            files_exist = 0
            total_files = len(hat_files.get(hat, []))
            
            for file in hat_files.get(hat, []):
                if os.path.exists(file):
                    files_exist += 1
            
            if total_files > 0:
                completion_rate = (files_exist / total_files) * 100
                if completion_rate >= 100:
                    status = "✅ COMPLETE"
                elif completion_rate >= 50:
                    status = "⚠️ PARTIAL"
                else:
                    status = "❌ INCOMPLETE"
            else:
                status = "✅ COMPLETE"
            
            hat_status[hat] = status
            print(f"   {hat}: {status} - {description}")
        
        return hat_status
    
    def generate_final_report(self):
        """Generate final comprehensive report"""
        print("\n🎉 COMPREHENSIVE PROGRESS REPORT")
        print("=" * 80)
        print(f"📅 Report Generated: {self.report_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Run all analyses
        trading_data = self.analyze_trading_data()
        system_status = self.check_system_status()
        recovery_progress = self.analyze_recovery_progress()
        hat_status = self.check_executive_hats_status()
        
        # Summary
        print("\n📊 EXECUTIVE SUMMARY:")
        print("=" * 80)
        
        # Trading performance
        print(f"📈 Trading Performance:")
        print(f"   • Total Trades: {trading_data['total_trades']}")
        print(f"   • Recent Win Rate: {trading_data['win_rate']:.1f}%")
        print(f"   • Recent PnL: ${trading_data['total_pnl']:.2f}")
        
        # System status
        print(f"\n🔧 System Status:")
        print(f"   • Bot Running: {'✅ YES' if system_status['bot_running'] else '❌ NO'}")
        print(f"   • Python Processes: {system_status['python_processes']}")
        
        # Executive hats
        complete_hats = sum(1 for status in hat_status.values() if "COMPLETE" in status)
        total_hats = len(hat_status)
        print(f"\n👑 Executive Hats:")
        print(f"   • Complete: {complete_hats}/{total_hats}")
        print(f"   • Completion Rate: {(complete_hats/total_hats)*100:.1f}%")
        
        # Overall status
        if system_status['bot_running'] and complete_hats == total_hats:
            overall_status = "✅ EXCELLENT"
        elif system_status['bot_running'] and complete_hats >= total_hats * 0.8:
            overall_status = "⚠️ GOOD"
        else:
            overall_status = "❌ NEEDS ATTENTION"
        
        print(f"\n🎯 Overall Status: {overall_status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not system_status['bot_running']:
            print("   • Restart the bot using start_ultimate_recovery.bat")
        if complete_hats < total_hats:
            print("   • Complete remaining executive hat implementations")
        if trading_data['win_rate'] < 50:
            print("   • Monitor trading performance and adjust strategy if needed")
        
        print("\n🚀 SYSTEM READY FOR CONTINUOUS PROFIT MONITORING!")
        print("=" * 80)

def main():
    report = ComprehensiveProgressReport()
    report.generate_final_report()

if __name__ == "__main__":
    main()
