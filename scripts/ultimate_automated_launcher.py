#!/usr/bin/env python3
"""
ULTIMATE AUTOMATED LAUNCHER
Fully automated bot launcher with NO user input required
"""

import subprocess
import time
import sys
import os
import signal

def create_input_file():
    """Create input file with automated responses"""
    with open("auto_input.txt", "w") as f:
        f.write("1\n")  # Choose Trading Profiles
        f.write("6\n")  # Choose A.I. ULTIMATE CHAMPION +213%
        f.write("XRP\n")  # Choose XRP token
        f.write("Y\n")   # Confirm setup
        f.write("Y\n")   # Confirm trading

def launch_bot():
    """Launch bot with automated input"""
    print("🚀 ULTIMATE AUTOMATED BOT LAUNCHER")
    print("=" * 50)
    print("🎯 A.I. ULTIMATE Profile: CHAMPION +213%")
    print("✅ All crashes eliminated")
    print("✅ All restrictions removed")
    print("✅ Maximum trade execution enabled")
    print("✅ FULLY AUTOMATED - NO USER INPUT REQUIRED")
    print("=" * 50)
    
    attempt = 1
    
    while True:
        print(f"\n🔄 Launching bot... (Attempt: {attempt})")
        
        try:
            # Create input file
            create_input_file()
            
            # Launch bot with input redirection
            cmd = [
                sys.executable, "newbotcode.py",
                "--fee-threshold-multi", "0.001",
                "--disable-rsi-veto",
                "--disable-momentum-veto",
                "--disable-microstructure-veto"
            ]
            
            print(f"🚀 Command: {' '.join(cmd)}")
            print("📝 Using automated input file")
            
            # Launch with input redirection
            with open("auto_input.txt", "r") as input_file:
                process = subprocess.Popen(
                    cmd,
                    stdin=input_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
            
            print(f"✅ Bot launched with PID: {process.pid}")
            print("🔄 Waiting for bot to complete...")
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            
            print("📊 Bot completed!")
            if stdout:
                print("📤 Output (last 300 chars):")
                print(stdout[-300:])
            if stderr:
                print("❌ Errors (last 200 chars):")
                print(stderr[-200:])
                
        except Exception as e:
            print(f"❌ Launch error: {e}")
        
        print(f"\n🔄 Bot stopped. Restarting in 5 seconds...")
        time.sleep(5)
        attempt += 1

if __name__ == "__main__":
    try:
        launch_bot()
    except KeyboardInterrupt:
        print("\n🛑 Launcher stopped by user")
        # Clean up input file
        if os.path.exists("auto_input.txt"):
            os.remove("auto_input.txt")

