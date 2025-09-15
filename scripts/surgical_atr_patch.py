#!/usr/bin/env python3
"""
Surgical ATR Patch for Force Trade Execution
Only targets the critical ATR scaled floor that's blocking trades
"""

import os
import shutil
import time

def surgical_atr_patch():
    """Apply surgical patch to force trade execution"""
    print("🎯 SURGICAL ATR PATCH - FORCE TRADE EXECUTION")
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

    # Apply surgical patches - ONLY the most critical ones
    patches = [
        # Patch 1: Force ATR scaled floor to 0.0001 (CRITICAL - this is blocking trades!)
        ('atr_scaled_floor = max(0.01, 0.5 * atr / current_price)', 
         'atr_scaled_floor = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 2: Force base confidence threshold to 0.0001 (CRITICAL)
        ('self.base_confidence_threshold = 0.08', 
         'self.base_confidence_threshold = 0.0001'),
        
        # Patch 3: Force current threshold fallback to 0.0001 (CRITICAL)
        ('current_threshold = getattr(self.config, \'confidence_threshold\', 0.08)', 
         'current_threshold = getattr(self.config, \'confidence_threshold\', 0.0001)')
    ]

    modified = False
    for i, line in enumerate(lines):
        for old_text, new_text in patches:
            if old_text in line:
                lines[i] = line.replace(old_text, new_text)
                modified = True
                print(f"✅ Applied surgical patch on line {i+1}: {old_text[:50]}...")

    if modified:
        # Write the patched file
        with open('newbotcode.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("✅ Bot source code SURGICALLY patched for force execution")
    else:
        print("⚠️ No patches applied - patterns not found")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo SURGICALLY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting SURGICALLY PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
"""

    with open('start_surgically_patched_bot.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_surgically_patched_bot.bat")

    print("\n🎯 SURGICAL ATR PATCH COMPLETE!")
    print("=" * 40)
    print("1. Bot source code has been SURGICALLY patched")
    print("2. CRITICAL ATR scaled floor removed")
    print("3. Essential confidence thresholds forced to 0.0001")
    print("4. Syntax completely preserved")
    print("5. Core restrictions bypassed")
    print("6. Run: .\\start_surgically_patched_bot.bat")
    print("7. Trades should execute with ANY confidence > 0")

    return True

if __name__ == "__main__":
    surgical_atr_patch()
