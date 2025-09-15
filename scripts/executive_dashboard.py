#!/usr/bin/env python3
"""
EXECUTIVE DASHBOARD
Real-time oversight from all 8 executive hats
"""

import time
import json
import os
from datetime import datetime

class ExecutiveDashboard:
    def __init__(self):
        self.hats = {
            "CEO": "👑 Crisis Management & Leadership",
            "CTO": "🔧 Technical Operations & Innovation", 
            "CFO": "💰 Financial Strategy & Risk Management",
            "COO": "⚙️ Operational Excellence & Efficiency",
            "CMO": "📈 Market Strategy & Growth",
            "CSO": "🛡️ Security & Risk Containment",
            "CDO": "📊 Data Analytics & AI Optimization",
            "CPO": "🎯 Product Development & User Experience"
        }
        self.start_time = datetime.now()
        
    def get_system_status(self):
        """Get current system status"""
        status = {
            "bot_running": False,
            "account_value": 29.50,
            "daily_profit": 0.0,
            "drawdown": 0.0,
            "trades_today": 0,
            "win_rate": 0.0,
            "system_health": "UNKNOWN"
        }
        
        try:
            # Check if bot is running
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower() and 'newbotcode.py' in ' '.join(proc.info['cmdline']):
                        status["bot_running"] = True
                        break
                except:
                    continue
            
            # Check account value
            if os.path.exists("account_value.json"):
                with open("account_value.json", 'r') as f:
                    data = json.load(f)
                    status["account_value"] = data.get('current_value', 29.50)
            
            # Check trades log
            if os.path.exists("trades_log.csv"):
                with open("trades_log.csv", 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        status["trades_today"] = len(lines) - 1
                        
                        # Calculate win rate
                        profitable_trades = 0
                        for line in lines[1:]:  # Skip header
                            parts = line.strip().split(',')
                            if len(parts) >= 6:
                                try:
                                    pnl = float(parts[5])
                                    if pnl > 0:
                                        profitable_trades += 1
                                except:
                                    continue
                        
                        if status["trades_today"] > 0:
                            status["win_rate"] = (profitable_trades / status["trades_today"]) * 100
            
            # Calculate daily profit
            status["daily_profit"] = status["account_value"] - 29.50
            
            # Calculate drawdown
            if status["account_value"] < 29.50:
                status["drawdown"] = ((29.50 - status["account_value"]) / 29.50) * 100
            
            # Determine system health
            if status["bot_running"] and status["drawdown"] < 5.0:
                status["system_health"] = "HEALTHY"
            elif status["bot_running"] and status["drawdown"] < 10.0:
                status["system_health"] = "WARNING"
            else:
                status["system_health"] = "CRITICAL"
                
        except Exception as e:
            status["system_health"] = f"ERROR: {e}"
        
        return status
    
    def display_hat_status(self, hat_name, hat_description, status):
        """Display individual hat status"""
        print(f"\n{hat_name} HAT: {hat_description}")
        print("-" * 60)
        
        if hat_name == "CEO":
            print(f"🎯 Mission Status: {'✅ ON TRACK' if status['system_health'] == 'HEALTHY' else '⚠️ MONITORING'}")
            print(f"📊 Overall Performance: {status['system_health']}")
            print(f"⏱️  Runtime: {datetime.now() - self.start_time}")
            
        elif hat_name == "CTO":
            print(f"🔧 Bot Status: {'✅ RUNNING' if status['bot_running'] else '❌ STOPPED'}")
            print(f"📊 System Health: {status['system_health']}")
            print(f"🖥️ Technical Status: {'✅ OPTIMAL' if status['system_health'] == 'HEALTHY' else '⚠️ MONITORING'}")
            
        elif hat_name == "CFO":
            print(f"💰 Account Value: ${status['account_value']:.2f}")
            print(f"📈 Daily Profit: ${status['daily_profit']:.2f}")
            print(f"📉 Drawdown: {status['drawdown']:.2f}%")
            print(f"🎯 Financial Status: {'✅ PROFITABLE' if status['daily_profit'] > 0 else '⚠️ MONITORING'}")
            
        elif hat_name == "COO":
            print(f"⚙️ Operations Status: {'✅ EFFICIENT' if status['system_health'] == 'HEALTHY' else '⚠️ MONITORING'}")
            print(f"📊 Trades Today: {status['trades_today']}")
            print(f"🎯 Win Rate: {status['win_rate']:.1f}%")
            print(f"📈 Performance: {'✅ OPTIMAL' if status['win_rate'] > 50 else '⚠️ IMPROVING'}")
            
        elif hat_name == "CMO":
            print(f"📈 Market Position: {'✅ STRONG' if status['daily_profit'] > 0 else '⚠️ MONITORING'}")
            print(f"🎯 Market Strategy: {'✅ WORKING' if status['win_rate'] > 50 else '⚠️ ADJUSTING'}")
            print(f"📊 Market Performance: {status['system_health']}")
            
        elif hat_name == "CSO":
            print(f"🛡️ Security Status: {'✅ SECURE' if status['drawdown'] < 5 else '⚠️ MONITORING'}")
            print(f"🔒 Risk Level: {'✅ LOW' if status['drawdown'] < 5 else '⚠️ ELEVATED'}")
            print(f"🛡️ Protection: {'✅ ACTIVE' if status['bot_running'] else '❌ INACTIVE'}")
            
        elif hat_name == "CDO":
            print(f"📊 Data Quality: {'✅ EXCELLENT' if status['system_health'] == 'HEALTHY' else '⚠️ MONITORING'}")
            print(f"🤖 AI Performance: {'✅ OPTIMAL' if status['win_rate'] > 50 else '⚠️ LEARNING'}")
            print(f"📈 Analytics: {'✅ INSIGHTS' if status['trades_today'] > 0 else '⚠️ COLLECTING'}")
            
        elif hat_name == "CPO":
            print(f"🎯 User Experience: {'✅ EXCELLENT' if status['system_health'] == 'HEALTHY' else '⚠️ MONITORING'}")
            print(f"📱 Product Status: {'✅ FUNCTIONAL' if status['bot_running'] else '❌ ISSUES'}")
            print(f"🎯 Satisfaction: {'✅ HIGH' if status['daily_profit'] > 0 else '⚠️ IMPROVING'}")
    
    def display_executive_dashboard(self):
        """Display the executive dashboard"""
        status = self.get_system_status()
        current_time = datetime.now()
        
        print(f"\n🚀 EXECUTIVE DASHBOARD - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print("🎯 ALL 8 EXECUTIVE HATS - REAL-TIME OVERSIGHT")
        print("=" * 80)
        
        # Display each hat's status
        for hat_name, hat_description in self.hats.items():
            self.display_hat_status(hat_name, hat_description, status)
        
        # Overall summary
        print(f"\n📊 EXECUTIVE SUMMARY:")
        print("=" * 80)
        print(f"🎯 Mission Status: {'✅ ON TRACK' if status['system_health'] == 'HEALTHY' else '⚠️ MONITORING'}")
        print(f"💰 Financial Performance: ${status['daily_profit']:.2f} ({'✅ PROFIT' if status['daily_profit'] > 0 else '⚠️ MONITORING'})")
        print(f"🛡️ Risk Status: {status['drawdown']:.2f}% drawdown ({'✅ LOW RISK' if status['drawdown'] < 5 else '⚠️ MONITORING'})")
        print(f"📈 Trading Performance: {status['win_rate']:.1f}% win rate ({'✅ STRONG' if status['win_rate'] > 50 else '⚠️ IMPROVING'})")
        print(f"🔧 System Health: {status['system_health']}")
        
        return status
    
    def run_executive_dashboard(self):
        """Run the executive dashboard"""
        print("🚀 STARTING EXECUTIVE DASHBOARD")
        print("=" * 80)
        print("🎯 REAL-TIME OVERSIGHT FROM ALL 8 EXECUTIVE HATS")
        print("⏱️  UPDATE INTERVAL: 60 seconds")
        print("=" * 80)
        
        update_count = 0
        
        while True:
            try:
                update_count += 1
                status = self.display_executive_dashboard()
                
                print(f"\n⏳ Next update in 60 seconds... (Update #{update_count})")
                print("=" * 80)
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n🛑 Executive dashboard stopped by user")
                break
            except Exception as e:
                print(f"\n⚠️ Dashboard error: {e}")
                time.sleep(60)

def main():
    dashboard = ExecutiveDashboard()
    dashboard.run_executive_dashboard()

if __name__ == "__main__":
    main()
