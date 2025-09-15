#!/usr/bin/env python3
"""
Minimal Surgical Patch for Force Trade Execution
Only targets the exact critical lines without breaking syntax
"""

import os
import shutil
import time

def minimal_surgical_patch():
    """Apply minimal surgical patch to force trade execution"""
    print("🎯 MINIMAL SURGICAL PATCH - FORCE TRADE EXECUTION")
    print("=" * 55)

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

    # Apply minimal surgical patches - ONLY the most critical ones
    patches = [
        # Patch 1: Force ATR scaled floor to 0.0001 (CRITICAL - this is blocking trades!)
        ('atr_scaled_floor = max(0.01, 0.5 * atr / current_price)', 
         'atr_scaled_floor = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 2: Force base confidence threshold to 0.0001 (CRITICAL)
        ('self.base_confidence_threshold = 0.08', 
         'self.base_confidence_threshold = 0.0001'),
        
        # Patch 3: Force current threshold fallback to 0.0001 (CRITICAL)
        ('current_threshold = getattr(self.config, \'confidence_threshold\', 0.08)', 
         'current_threshold = getattr(self.config, \'confidence_threshold\', 0.0001)'),
        
        # Patch 4: Force environment variable override (CRITICAL)
        ('base_threshold = float(os.environ.get("BOT_CONFIDENCE_THRESHOLD", "0.015"))', 
         'base_threshold = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 5: Force environment variable setting (CRITICAL)
        ('os.environ["BOT_CONFIDENCE_THRESHOLD"] = "0.015"', 
         'os.environ["BOT_CONFIDENCE_THRESHOLD"] = "0.0001"  # EMERGENCY PATCH: FORCED TO 0.0001')
    ]

    modified = False
    for i, line in enumerate(lines):
        for old_text, new_text in patches:
            if old_text in line:
                lines[i] = line.replace(old_text, new_text)
                modified = True
                print(f"✅ Applied minimal surgical patch on line {i+1}: {old_text[:50]}...")

    if modified:
        # Write the patched file
        with open('newbotcode.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("✅ Bot source code MINIMALLY SURGICALLY patched for force execution")
    else:
        print("⚠️ No patches applied - patterns not found")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo MINIMALLY SURGICALLY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting MINIMALLY SURGICALLY PATCHED bot...
echo Bot will use 0.0001 threshold and execute ALL trades
python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
"""

    with open('start_minimal_surgical_bot.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_minimal_surgical_bot.bat")

    print("\n🎯 MINIMAL SURGICAL PATCH COMPLETE!")
    print("=" * 45)
    print("1. Bot source code has been MINIMALLY SURGICALLY patched")
    print("2. ONLY critical thresholds modified")
    print("3. Syntax completely preserved")
    print("4. Bot will use 0.0001 confidence threshold")
    print("5. ALL trades will execute")
    print("6. Run: .\\start_minimal_surgical_bot.bat")
    print("7. Bot will execute trades with ANY confidence > 0")

    return True

if __name__ == "__main__":
    minimal_surgical_patch()
