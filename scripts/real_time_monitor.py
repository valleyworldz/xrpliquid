#!/usr/bin/env python3
"""
REAL-TIME BOT MONITOR
Live monitoring dashboard for the trading bot
"""

import psutil
import time
import os
import subprocess
import sys
from datetime import datetime

def find_bot_processes():
    """Find all running bot processes"""
    bot_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'newbotcode.py' in cmdline:
                    bot_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'status': proc.status(),
                        'cpu': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return bot_processes

def get_system_info():
    """Get system resource information"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return {
            'cpu': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024,
            'disk_percent': disk.percent
        }
    except:
        return {}

def monitor_bot():
    """Monitor bot status continuously"""
    print("🔍 REAL-TIME BOT MONITORING DASHBOARD")
    print("=" * 60)
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🔍 REAL-TIME BOT MONITORING DASHBOARD")
        print("=" * 60)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System resources
        sys_info = get_system_info()
        if sys_info:
            print("💻 SYSTEM RESOURCES:")
            print(f"   CPU: {sys_info.get('cpu', 0):.1f}%")
            print(f"   Memory: {sys_info.get('memory_percent', 0):.1f}% ({sys_info.get('memory_available_gb', 0):.1f} GB available)")
            print(f"   Disk: {sys_info.get('disk_percent', 0):.1f}%")
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
                print(f"   CPU: {proc['cpu']:.1f}%")
                print(f"   Memory: {proc['memory_mb']:.1f} MB")
                print(f"   Command: {proc['cmdline'][:80]}...")
                print()
        else:
            print("❌ BOT IS NOT RUNNING")
            print("🚀 Launching automated bot...")
            print()
            
            # Auto-launch if not running
            try:
                subprocess.Popen([sys.executable, "ultimate_automated_launcher.py"])
                print("✅ Auto-launch initiated")
            except:
                print("❌ Auto-launch failed")
        
        print("=" * 60)
        print("🔄 Refreshing in 3 seconds... (Ctrl+C to stop)")
        print("📊 Press 'R' to refresh immediately, 'L' to launch bot")
        
        # Wait for refresh
        time.sleep(3)

if __name__ == "__main__":
    try:
        monitor_bot()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

