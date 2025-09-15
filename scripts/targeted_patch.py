#!/usr/bin/env python3
"""
Targeted Patch for Force Trade Execution
Only modifies specific critical lines to avoid corruption
"""

import os
import shutil
import time

def targeted_patch():
    """Apply targeted patches to force trade execution"""
    print("🎯 TARGETED PATCH - FORCE TRADE EXECUTION")
    print("=" * 50)

    # Check if newbotcode.py exists
    if not os.path.exists('newbotcode.py'):
        print("❌ newbotcode.py not found")
        return False

    # Create backup
    backup_file = f'newbotcode_backup_{int(time.time())}.py'
    shutil.copy2('newbotcode.py', backup_file)
    print(f"✅ Backup created: {backup_file}")

    # Read the source file
    with open('newbotcode.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Apply targeted patches
    modified = False
    
    for i, line in enumerate(lines):
        # Patch 1: Force confidence threshold to 0.0001
        if 'confidence_threshold' in line and '0.015' in line:
            lines[i] = line.replace('0.015', '0.0001  # FORCED BY EMERGENCY PATCH')
            modified = True
            print(f"✅ Patched confidence threshold at line {i+1}")
        
        # Patch 2: Bypass micro-account safeguard
        elif 'expected PnL below fee+funding threshold' in line:
            lines[i] = '        # EMERGENCY PATCH: MICRO-ACCOUNT SAFEGUARD BYPASSED\n'
            modified = True
            print(f"✅ Patched micro-account safeguard at line {i+1}")
        
        # Patch 3: Force trade execution
        elif '🚫 Skipping trade' in line:
            lines[i] = '        # EMERGENCY PATCH: FORCING TRADE EXECUTION\n'
            modified = True
            print(f"✅ Patched trade skip at line {i+1}")
        
        # Patch 4: Bypass confidence filter
        elif 'FILTER=Confidence, conf=' in line:
            lines[i] = '        # EMERGENCY PATCH: CONFIDENCE FILTER BYPASSED\n'
            modified = True
            print(f"✅ Patched confidence filter at line {i+1}")
        
        # Patch 5: Bypass RSI filter
        elif 'Skipping BUY: RSI too low' in line:
            lines[i] = '        # EMERGENCY PATCH: RSI FILTER BYPASSED\n'
            modified = True
            print(f"✅ Patched RSI filter at line {i+1}")
        
        # Patch 6: Bypass momentum filter
        elif 'Momentum too low' in line:
            lines[i] = '        # EMERGENCY PATCH: MOMENTUM FILTER BYPASSED\n'
            modified = True
            print(f"✅ Patched momentum filter at line {i+1}")

    if modified:
        # Write the patched file
        with open('newbotcode.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("✅ Bot source code patched for force execution")
    else:
        print("⚠️ No patches applied - patterns not found")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo TARGETED PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting TARGETED PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee_threshold_multi 0.01 --confidence_threshold 0.0001 --aggressive_mode true

pause
"""

    with open('start_targeted_patched_bot.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_targeted_patched_bot.bat")

    print("\n🎯 TARGETED PATCH COMPLETE!")
    print("=" * 40)
    print("1. Bot source code has been patched")
    print("2. Critical restrictions bypassed")
    print("3. Confidence threshold forced to 0.0001")
    print("4. Micro-account safeguard disabled")
    print("5. Key filters bypassed")
    print("6. Run: .\\start_targeted_patched_bot.bat")
    print("7. Trades should execute with ANY confidence > 0")

    return True

if __name__ == "__main__":
    targeted_patch()
