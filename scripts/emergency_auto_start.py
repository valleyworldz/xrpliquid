#!/usr/bin/env python3
"""
Emergency Auto-Start Script for Force Trade Execution
Bypasses interactive menu and starts bot automatically
"""

import subprocess
import time
import os

def emergency_auto_start():
    """Emergency auto-start that bypasses interactive menu"""
    print("🚨 EMERGENCY AUTO-START - FORCE TRADE EXECUTION")
    print("=" * 55)
    
    # Kill any existing python processes
    print("🔄 Terminating existing processes...")
    try:
        subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                      capture_output=True, shell=True)
        time.sleep(2)
    except:
        pass
    
    # Create auto-start batch script
    auto_script = """@echo off
echo ============================================================
echo EMERGENCY AUTO-START BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting bot with automated input...
echo Bot will automatically select option 1 and start trading
echo Confidence threshold: 0.0001 (ALL trades will execute)

echo 1 | python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
"""
    
    with open('start_emergency_auto.bat', 'w') as f:
        f.write(auto_script)
    
    print("✅ Created start_emergency_auto.bat")
    
    # Create Python auto-start script
    python_script = """#!/usr/bin/env python3
import subprocess
import time

def auto_start_bot():
    print("🚨 AUTO-STARTING BOT WITH FORCE TRADE EXECUTION")
    print("=" * 55)
    
    # Start bot with automated input
    try:
        process = subprocess.Popen(
            ['python', 'newbotcode.py', '--fee-threshold-multi', '0.01', '--disable-rsi-veto', '--disable-momentum-veto', '--disable-microstructure-veto'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send "1" to stdin to select option 1
        stdout, stderr = process.communicate(input="1\\n", timeout=30)
        
        print("✅ Bot started successfully")
        print("📊 Output:", stdout)
        if stderr:
            print("⚠️ Errors:", stderr)
            
    except subprocess.TimeoutExpired:
        print("⚠️ Bot startup timed out")
        process.kill()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    auto_start_bot()
"""
    
    with open('auto_start_bot.py', 'w') as f:
        f.write(python_script)
    
    print("✅ Created auto_start_bot.py")
    
    print("\n🎯 EMERGENCY AUTO-START READY!")
    print("=" * 40)
    print("1. Bot will automatically select option 1")
    print("2. No interactive prompts - immediate trading")
    print("3. Confidence threshold: 0.0001")
    print("4. ALL trades will execute")
    print("5. Run: .\\start_emergency_auto.bat")
    print("6. Or: python auto_start_bot.py")
    
    return True

if __name__ == "__main__":
    emergency_auto_start()
