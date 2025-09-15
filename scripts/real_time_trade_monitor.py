#!/usr/bin/env python3
"""
Real-Time Trade Monitor for Force Trade Execution
Monitors bot performance in real-time and ensures maximum trade execution
"""

import os
import time
import csv
import psutil
from datetime import datetime, timedelta

def monitor_real_time_trades():
    """Monitor trades in real-time"""
    print("🚨 REAL-TIME TRADE MONITOR - FORCE TRADE EXECUTION")
    print("=" * 65)
    
    # Check if bot is running
    bot_running = False
    bot_pid = None
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and 'newbotcode.py' in ' '.join(proc.info['cmdline'] or []):
                bot_running = True
                bot_pid = proc.info['pid']
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not bot_running:
        print("❌ Bot is NOT running!")
        return False
    
    print(f"✅ Bot is running (PID: {bot_pid})")
    
    # Monitor trades in real-time
    trades_file = "trades_log.csv"
    last_check = time.time()
    initial_trade_count = 0
    
    if os.path.exists(trades_file):
        with open(trades_file, 'r') as f:
            initial_trade_count = len(f.readlines()) - 1  # Subtract header
    
    print(f"📊 Initial trade count: {initial_trade_count}")
    print(f"🎯 Monitoring for new trades...")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 80)
    
    while True:
        try:
            current_time = time.time()
            
            # Check bot status
            try:
                process = psutil.Process(bot_pid)
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                status = "🟢 ACTIVE" if cpu_percent > 0 else "🟡 IDLE"
                
            except psutil.NoSuchProcess:
                print("❌ Bot process lost!")
                return False
            
            # Check for new trades
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    current_trade_count = len(f.readlines()) - 1
                
                if current_trade_count > initial_trade_count:
                    # New trades detected
                    with open(trades_file, 'r') as f:
                        reader = csv.DictReader(f)
                        trades = list(reader)
                        new_trades = trades[initial_trade_count:]
                    
                    for trade in new_trades:
                        timestamp = trade.get('timestamp', 'N/A')
                        side = trade.get('side', 'N/A')
                        size = trade.get('size', 'N/A')
                        price = trade.get('price', 'N/A')
                        confidence = trade.get('confidence', 'N/A')
                        
                        # Check if confidence meets our threshold
                        conf_status = "✅ EXECUTED" if float(confidence) >= 0.0001 else "❌ BLOCKED"
                        
                        print(f"🆕 NEW TRADE: {timestamp} | {side} | Size: {size} | Price: ${price} | Conf: {confidence} | {conf_status}")
                    
                    initial_trade_count = current_trade_count
                
                # Display current status
                print(f"\r⏰ {datetime.now().strftime('%H:%M:%S')} | {status} | CPU: {cpu_percent:.1f}% | Memory: {memory_mb:.1f}MB | Trades: {current_trade_count}", end="")
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error in monitoring: {e}")
            time.sleep(10)
    
    return True

if __name__ == "__main__":
    print("🚨 REAL-TIME TRADE MONITOR ACTIVATED")
    print("=" * 50)
    monitor_real_time_trades()
