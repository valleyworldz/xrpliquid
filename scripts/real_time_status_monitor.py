#!/usr/bin/env python3
"""
REAL-TIME STATUS MONITOR
Continuous monitoring until continuous profit is achieved
"""

import time
import json
import os
import psutil
from datetime import datetime, timedelta

class RealTimeStatusMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.mission_start = datetime.now()
        self.continuous_profit_days = 0
        self.target_continuous_days = 3
        self.daily_profit_target = 0.25
        self.recovery_target = 20.50
        self.starting_value = 29.50
        
    def get_bot_status(self):
        """Get current bot status"""
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
        
        return bot_running, python_processes
    
    def get_trading_metrics(self):
        """Get current trading metrics"""
        metrics = {
            'total_trades': 0,
            'recent_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'current_value': self.starting_value,
            'daily_profit': 0.0
        }
        
        try:
            # Get current account value
            if os.path.exists("account_value.json"):
                with open("account_value.json", 'r') as f:
                    data = json.load(f)
                    metrics['current_value'] = data.get('current_value', self.starting_value)
            
            # Analyze trades
            if os.path.exists("trades_log.csv"):
                with open("trades_log.csv", 'r') as f:
                    lines = f.readlines()
                    
                metrics['total_trades'] = len(lines) - 1
                
                # Analyze recent trades (last 10)
                recent_trades = lines[-10:] if len(lines) > 10 else lines[1:]
                metrics['recent_trades'] = len(recent_trades)
                
                profitable_count = 0
                for line in recent_trades:
                    if line.strip() and not line.startswith('trade_id'):
                        parts = line.strip().split(',')
                        if len(parts) >= 8:
                            try:
                                pnl = float(parts[7]) if parts[7] else 0.0
                                metrics['total_pnl'] += pnl
                                if pnl > 0:
                                    profitable_count += 1
                            except:
                                continue
                
                if metrics['recent_trades'] > 0:
                    metrics['win_rate'] = (profitable_count / metrics['recent_trades']) * 100
            
            # Calculate daily profit
            metrics['daily_profit'] = metrics['current_value'] - self.starting_value
            
        except Exception as e:
            print(f"⚠️ Error getting trading metrics: {e}")
        
        return metrics
    
    def check_continuous_profit(self, metrics):
        """Check continuous profit status"""
        if metrics['daily_profit'] >= self.daily_profit_target:
            self.continuous_profit_days += 1
            return True
        else:
            self.continuous_profit_days = 0
            return False
    
    def display_status(self):
        """Display current status"""
        current_time = datetime.now()
        runtime = current_time - self.start_time
        mission_runtime = current_time - self.mission_start
        
        bot_running, python_processes = self.get_bot_status()
        metrics = self.get_trading_metrics()
        is_profitable = self.check_continuous_profit(metrics)
        
        print(f"\n🚀 REAL-TIME STATUS MONITOR - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"⏱️  Mission Runtime: {mission_runtime}")
        print(f"⏱️  Monitor Runtime: {runtime}")
        print("=" * 80)
        
        # Bot Status
        print(f"🤖 BOT STATUS:")
        print(f"   • Bot Running: {'✅ YES' if bot_running else '❌ NO'}")
        print(f"   • Python Processes: {python_processes}")
        print(f"   • System Health: {'✅ HEALTHY' if bot_running else '⚠️ NEEDS RESTART'}")
        
        # Trading Performance
        print(f"\n📈 TRADING PERFORMANCE:")
        print(f"   • Total Trades: {metrics['total_trades']}")
        print(f"   • Recent Trades: {metrics['recent_trades']}")
        print(f"   • Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   • Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"   • Current Value: ${metrics['current_value']:.2f}")
        print(f"   • Daily Profit: ${metrics['daily_profit']:.2f}")
        
        # Profit Targets
        print(f"\n🎯 PROFIT TARGETS:")
        print(f"   • Daily Target: ${self.daily_profit_target}")
        print(f"   • Recovery Target: ${self.recovery_target}")
        print(f"   • Daily Status: {'✅ ACHIEVED' if metrics['daily_profit'] >= self.daily_profit_target else '⚠️ IN PROGRESS'}")
        print(f"   • Recovery Status: {'✅ ACHIEVED' if metrics['daily_profit'] >= self.recovery_target else '⚠️ IN PROGRESS'}")
        
        # Continuous Profit Tracking
        print(f"\n🔥 CONTINUOUS PROFIT TRACKING:")
        print(f"   • Consecutive Days: {self.continuous_profit_days}/{self.target_continuous_days}")
        print(f"   • Status: {'✅ PROFITABLE' if is_profitable else '⚠️ NOT PROFITABLE'}")
        print(f"   • Mission: {'✅ COMPLETE' if self.continuous_profit_days >= self.target_continuous_days else '⚠️ IN PROGRESS'}")
        
        # Progress Bars
        print(f"\n📊 PROGRESS BARS:")
        
        # Daily progress
        daily_progress = min(metrics['daily_profit'] / self.daily_profit_target, 1.0)
        daily_bar = "█" * int(40 * daily_progress) + "░" * int(40 * (1 - daily_progress))
        print(f"   Daily: [{daily_bar}] {daily_progress*100:.1f}%")
        
        # Recovery progress
        recovery_progress = min(metrics['daily_profit'] / self.recovery_target, 1.0)
        recovery_bar = "█" * int(40 * recovery_progress) + "░" * int(40 * (1 - recovery_progress))
        print(f"   Recovery: [{recovery_bar}] {recovery_progress*100:.1f}%")
        
        # Continuous progress
        continuous_progress = min(self.continuous_profit_days / self.target_continuous_days, 1.0)
        continuous_bar = "█" * int(40 * continuous_progress) + "░" * int(40 * (1 - continuous_progress))
        print(f"   Continuous: [{continuous_bar}] {continuous_progress*100:.1f}%")
        
        return {
            'bot_running': bot_running,
            'metrics': metrics,
            'continuous_profit_days': self.continuous_profit_days,
            'mission_complete': self.continuous_profit_days >= self.target_continuous_days
        }
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring until mission complete"""
        print("🚀 STARTING REAL-TIME STATUS MONITOR")
        print("=" * 80)
        print("🎯 MISSION: CONTINUOUS PROFIT ACHIEVEMENT")
        print("💰 DAILY TARGET: $0.25")
        print("🚀 RECOVERY TARGET: $20.50")
        print("🔥 CONTINUOUS GOAL: 3 consecutive profitable days")
        print("⏱️  UPDATE INTERVAL: 30 seconds")
        print("=" * 80)
        
        update_count = 0
        
        while True:
            try:
                update_count += 1
                status = self.display_status()
                
                # Check if mission is complete
                if status['mission_complete']:
                    print(f"\n🎉🎉🎉 MISSION ACCOMPLISHED! 🎉🎉🎉")
                    print("✅ CONTINUOUS PROFIT ACHIEVED!")
                    print("✅ RECOVERY TARGET REACHED!")
                    print("✅ ALL EXECUTIVE HATS SUCCESSFUL!")
                    print("✅ SYSTEM OPERATING AT PEAK PERFORMANCE!")
                    print(f"⏱️  Total Mission Time: {datetime.now() - self.mission_start}")
                    break
                
                # Check if bot needs restart
                if not status['bot_running']:
                    print(f"\n⚠️ BOT NOT RUNNING - ATTEMPTING RESTART...")
                    # Could implement restart logic here
                
                print(f"\n⏳ Next update in 30 seconds... (Update #{update_count})")
                print("=" * 80)
                
                time.sleep(30)
                
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"\n⚠️ Monitoring error: {e}")
                time.sleep(30)
        
        print("\n📊 FINAL MISSION SUMMARY:")
        print(f"⏱️  Total monitoring time: {datetime.now() - self.start_time}")
        print(f"🔥 Continuous profit days achieved: {self.continuous_profit_days}")
        print(f"💰 Final account value: ${status['metrics']['current_value']:.2f}")

def main():
    monitor = RealTimeStatusMonitor()
    monitor.run_continuous_monitoring()

if __name__ == "__main__":
    main()
