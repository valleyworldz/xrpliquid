#!/usr/bin/env python3
"""
PROFIT TARGET ACHIEVEMENT TRACKER
Track progress towards continuous profit goals
"""

import time
import json
import os
from datetime import datetime, timedelta

class ProfitTargetTracker:
    def __init__(self):
        self.starting_value = 29.50
        self.daily_profit_target = 0.25
        self.weekly_profit_target = 1.25
        self.monthly_profit_target = 5.00
        self.recovery_target = 20.50  # $50.00 total target
        self.continuous_profit_days_required = 3
        self.continuous_profit_days = 0
        self.start_date = datetime.now().date()
        
    def get_current_metrics(self):
        """Get current profit metrics"""
        metrics = {
            "current_value": self.starting_value,
            "daily_profit": 0.0,
            "weekly_profit": 0.0,
            "monthly_profit": 0.0,
            "total_profit": 0.0,
            "recovery_percentage": 0.0,
            "trades_today": 0,
            "win_rate": 0.0,
            "continuous_profit_days": 0
        }
        
        try:
            # Get current account value
            if os.path.exists("account_value.json"):
                with open("account_value.json", 'r') as f:
                    data = json.load(f)
                    metrics["current_value"] = data.get('current_value', self.starting_value)
            
            # Calculate profits
            metrics["total_profit"] = metrics["current_value"] - self.starting_value
            metrics["daily_profit"] = metrics["total_profit"]  # Simplified for now
            metrics["weekly_profit"] = metrics["total_profit"]
            metrics["monthly_profit"] = metrics["total_profit"]
            
            # Calculate recovery percentage
            if self.recovery_target > 0:
                metrics["recovery_percentage"] = (metrics["total_profit"] / self.recovery_target) * 100
            
            # Get trading metrics
            if os.path.exists("trades_log.csv"):
                with open("trades_log.csv", 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        metrics["trades_today"] = len(lines) - 1
                        
                        # Calculate win rate
                        profitable_trades = 0
                        for line in lines[1:]:
                            parts = line.strip().split(',')
                            if len(parts) >= 6:
                                try:
                                    pnl = float(parts[5])
                                    if pnl > 0:
                                        profitable_trades += 1
                                except:
                                    continue
                        
                        if metrics["trades_today"] > 0:
                            metrics["win_rate"] = (profitable_trades / metrics["trades_today"]) * 100
            
            # Check continuous profit days
            if metrics["daily_profit"] >= self.daily_profit_target:
                metrics["continuous_profit_days"] = self.continuous_profit_days + 1
            else:
                metrics["continuous_profit_days"] = 0
                
        except Exception as e:
            print(f"⚠️ Error getting metrics: {e}")
        
        return metrics
    
    def display_profit_targets(self):
        """Display profit target progress"""
        metrics = self.get_current_metrics()
        current_time = datetime.now()
        
        print(f"\n🎯 PROFIT TARGET ACHIEVEMENT TRACKER - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Daily targets
        print("📅 DAILY TARGETS:")
        print(f"   💰 Daily Profit Target: ${self.daily_profit_target}")
        print(f"   📈 Current Daily Profit: ${metrics['daily_profit']:.2f}")
        daily_status = "✅ ACHIEVED" if metrics['daily_profit'] >= self.daily_profit_target else "⚠️ IN PROGRESS"
        print(f"   🎯 Daily Status: {daily_status}")
        
        # Weekly targets
        print(f"\n📅 WEEKLY TARGETS:")
        print(f"   💰 Weekly Profit Target: ${self.weekly_profit_target}")
        print(f"   📈 Current Weekly Profit: ${metrics['weekly_profit']:.2f}")
        weekly_status = "✅ ACHIEVED" if metrics['weekly_profit'] >= self.weekly_profit_target else "⚠️ IN PROGRESS"
        print(f"   🎯 Weekly Status: {weekly_status}")
        
        # Monthly targets
        print(f"\n📅 MONTHLY TARGETS:")
        print(f"   💰 Monthly Profit Target: ${self.monthly_profit_target}")
        print(f"   📈 Current Monthly Profit: ${metrics['monthly_profit']:.2f}")
        monthly_status = "✅ ACHIEVED" if metrics['monthly_profit'] >= self.monthly_profit_target else "⚠️ IN PROGRESS"
        print(f"   🎯 Monthly Status: {monthly_status}")
        
        # Recovery targets
        print(f"\n🚀 RECOVERY TARGETS:")
        print(f"   💰 Total Recovery Target: ${self.recovery_target}")
        print(f"   📈 Current Total Profit: ${metrics['total_profit']:.2f}")
        print(f"   📊 Recovery Progress: {metrics['recovery_percentage']:.1f}%")
        recovery_status = "✅ ACHIEVED" if metrics['total_profit'] >= self.recovery_target else "⚠️ IN PROGRESS"
        print(f"   🎯 Recovery Status: {recovery_status}")
        
        # Continuous profit tracking
        print(f"\n🔥 CONTINUOUS PROFIT TRACKING:")
        print(f"   📅 Required Consecutive Days: {self.continuous_profit_days_required}")
        print(f"   📈 Current Consecutive Days: {metrics['continuous_profit_days']}")
        continuous_status = "✅ ACHIEVED" if metrics['continuous_profit_days'] >= self.continuous_profit_days_required else "⚠️ IN PROGRESS"
        print(f"   🎯 Continuous Status: {continuous_status}")
        
        # Trading performance
        print(f"\n📊 TRADING PERFORMANCE:")
        print(f"   📈 Trades Today: {metrics['trades_today']}")
        print(f"   🎯 Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   💰 Current Account Value: ${metrics['current_value']:.2f}")
        
        # Progress bars
        print(f"\n📊 PROGRESS BARS:")
        
        # Daily progress
        daily_progress = min(metrics['daily_profit'] / self.daily_profit_target, 1.0)
        daily_bar = "█" * int(40 * daily_progress) + "░" * int(40 * (1 - daily_progress))
        print(f"   Daily: [{daily_bar}] {daily_progress*100:.1f}%")
        
        # Recovery progress
        recovery_progress = min(metrics['recovery_percentage'] / 100, 1.0)
        recovery_bar = "█" * int(40 * recovery_progress) + "░" * int(40 * (1 - recovery_progress))
        print(f"   Recovery: [{recovery_bar}] {recovery_progress*100:.1f}%")
        
        # Continuous profit progress
        continuous_progress = min(metrics['continuous_profit_days'] / self.continuous_profit_days_required, 1.0)
        continuous_bar = "█" * int(40 * continuous_progress) + "░" * int(40 * (1 - continuous_progress))
        print(f"   Continuous: [{continuous_bar}] {continuous_progress*100:.1f}%")
        
        return metrics
    
    def check_mission_complete(self, metrics):
        """Check if mission is complete"""
        mission_complete = (
            metrics['continuous_profit_days'] >= self.continuous_profit_days_required and
            metrics['total_profit'] >= self.recovery_target
        )
        
        if mission_complete:
            print(f"\n🎉🎉🎉 MISSION ACCOMPLISHED! 🎉🎉🎉")
            print("✅ CONTINUOUS PROFIT ACHIEVED!")
            print("✅ RECOVERY TARGET REACHED!")
            print("✅ ALL EXECUTIVE HATS SUCCESSFUL!")
            print("✅ SYSTEM OPERATING AT PEAK PERFORMANCE!")
            return True
        
        return False
    
    def run_profit_tracking(self):
        """Run profit target tracking"""
        print("🎯 STARTING PROFIT TARGET ACHIEVEMENT TRACKER")
        print("=" * 80)
        print("💰 DAILY TARGET: $0.25")
        print("📅 WEEKLY TARGET: $1.25")
        print("📅 MONTHLY TARGET: $5.00")
        print("🚀 RECOVERY TARGET: $20.50")
        print("🔥 CONTINUOUS PROFIT: 3 consecutive days")
        print("⏱️  UPDATE INTERVAL: 45 seconds")
        print("=" * 80)
        
        update_count = 0
        
        while True:
            try:
                update_count += 1
                metrics = self.display_profit_targets()
                
                # Check if mission is complete
                if self.check_mission_complete(metrics):
                    break
                
                print(f"\n⏳ Next update in 45 seconds... (Update #{update_count})")
                print("=" * 80)
                
                time.sleep(45)
                
            except KeyboardInterrupt:
                print("\n🛑 Profit tracking stopped by user")
                break
            except Exception as e:
                print(f"\n⚠️ Tracking error: {e}")
                time.sleep(45)

def main():
    tracker = ProfitTargetTracker()
    tracker.run_profit_tracking()

if __name__ == "__main__":
    main()
