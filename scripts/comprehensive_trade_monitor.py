#!/usr/bin/env python3
"""
Comprehensive Trade Monitor for Force Trade Execution
Monitors bot performance and ensures maximum trade execution
"""

import os
import time
import csv
import subprocess
import psutil
from datetime import datetime, timedelta

def monitor_bot_performance():
    """Monitor bot performance and trade execution"""
    print("🚨 COMPREHENSIVE TRADE MONITOR - FORCE TRADE EXECUTION")
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
        print("❌ Bot is NOT running - starting emergency restart...")
        return False
    
    print(f"✅ Bot is running (PID: {bot_pid})")
    
    # Monitor trades
    trades_file = "trades_log.csv"
    if os.path.exists(trades_file):
        with open(trades_file, 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        
        if trades:
            recent_trades = trades[-10:]  # Last 10 trades
            print(f"\n📊 Recent Trades (Last 10):")
            print("-" * 80)
            
            for trade in recent_trades:
                timestamp = trade.get('timestamp', 'N/A')
                side = trade.get('side', 'N/A')
                size = trade.get('size', 'N/A')
                price = trade.get('price', 'N/A')
                confidence = trade.get('confidence', 'N/A')
                
                print(f"🕒 {timestamp} | {side} | Size: {size} | Price: ${price} | Conf: {confidence}")
            
            # Analyze confidence thresholds
            confidences = []
            for trade in trades:
                try:
                    conf = float(trade.get('confidence', 0))
                    if conf > 0:
                        confidences.append(conf)
                except:
                    continue
            
            if confidences:
                min_conf = min(confidences)
                max_conf = max(confidences)
                avg_conf = sum(confidences) / len(confidences)
                
                print(f"\n🎯 Confidence Analysis:")
                print(f"   Min: {min_conf:.6f}")
                print(f"   Max: {max_conf:.6f}")
                print(f"   Avg: {avg_conf:.6f}")
                print(f"   Total Trades: {len(confidences)}")
                
                # Check if our 0.0001 threshold is working
                if min_conf <= 0.0001:
                    print("✅ SUCCESS: 0.0001 confidence threshold is working!")
                else:
                    print("⚠️ WARNING: Minimum confidence above 0.0001 - threshold may not be working")
        else:
            print("⚠️ No trades found in log")
    else:
        print("⚠️ Trades log file not found")
    
    # Monitor bot resources
    try:
        process = psutil.Process(bot_pid)
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        print(f"\n💻 Bot Resource Usage:")
        print(f"   CPU: {cpu_percent:.2f}%")
        print(f"   Memory: {memory_mb:.1f} MB")
        print(f"   Status: {'Active' if cpu_percent > 0 else 'Idle'}")
        
    except psutil.NoSuchProcess:
        print("❌ Bot process not found")
    
    # Check for any error logs
    error_files = []
    for file in os.listdir('.'):
        if file.endswith('.log') or file.endswith('.txt'):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if 'error' in content.lower() or 'exception' in content.lower():
                        error_files.append(file)
            except:
                continue
    
    if error_files:
        print(f"\n⚠️ Potential Error Files: {', '.join(error_files)}")
    else:
        print(f"\n✅ No error files detected")
    
    print(f"\n🎯 FORCE TRADE EXECUTION STATUS:")
    print(f"   Bot Running: {'✅ YES' if bot_running else '❌ NO'}")
    print(f"   Confidence Threshold: 0.0001")
    print(f"   Trade Execution: {'✅ ACTIVE' if bot_running else '❌ STOPPED'}")
    
    return True

def emergency_restart():
    """Emergency restart of the bot"""
    print("\n🚨 EMERGENCY RESTART INITIATED")
    print("=" * 40)
    
    # Kill any existing python processes
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == 'python.exe':
                proc.terminate()
                print(f"✅ Terminated Python process {proc.info['pid']}")
        except:
            continue
    
    time.sleep(2)
    
    # Start the bot
    try:
        subprocess.Popen(['start_simple_data_fix_bot.bat'], shell=True)
        print("✅ Bot restart initiated")
        return True
    except Exception as e:
        print(f"❌ Failed to restart bot: {e}")
        return False

if __name__ == "__main__":
    print("🚨 COMPREHENSIVE TRADE MONITOR ACTIVATED")
    print("=" * 50)
    
    while True:
        try:
            if not monitor_bot_performance():
                print("\n🚨 Bot not running - attempting emergency restart...")
                emergency_restart()
            
            print(f"\n⏳ Next check in 30 seconds... (Press Ctrl+C to stop)")
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error in monitoring: {e}")
            time.sleep(10)
