#!/usr/bin/env python3
"""
MONITOR BOT STATUS
Real-time monitoring of bot status and performance
"""

import psutil
import time
import os
import subprocess
import sys

def find_bot_processes():
    """Find all running bot processes"""
    bot_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'newbotcode.py' in cmdline:
                    bot_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'status': proc.status()
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return bot_processes

def monitor_bot():
    """Monitor bot status continuously"""
    print("🔍 BOT STATUS MONITOR")
    print("=" * 40)
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🔍 BOT STATUS MONITOR")
        print("=" * 40)
        print(f"⏰ Time: {time.strftime('%H:%M:%S')}")
        print()
        
        # Find bot processes
        bot_processes = find_bot_processes()
        
        if bot_processes:
            print(f"✅ BOT IS RUNNING - {len(bot_processes)} process(es) active")
            print()
            
            for i, proc in enumerate(bot_processes, 1):
                print(f"🔄 Process {i}:")
                print(f"   PID: {proc['pid']}")
                print(f"   Status: {proc['status']}")
                print(f"   Command: {proc['cmdline'][:80]}...")
                print()
        else:
            print("❌ BOT IS NOT RUNNING")
            print("🚀 Launching automated bot...")
            print()
            
            # Auto-launch if not running
            try:
                subprocess.Popen([sys.executable, "auto_launch_bot.py"])
                print("✅ Auto-launch initiated")
            except:
                print("❌ Auto-launch failed")
        
        print("=" * 40)
        print("🔄 Refreshing in 5 seconds... (Ctrl+C to stop)")
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        monitor_bot()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

