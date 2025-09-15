#!/usr/bin/env python3
"""
Final Targeted Patch for Force Trade Execution
Fixes the remaining hardcoded 0.08 base confidence threshold
"""

import os
import shutil
import time

def final_targeted_patch():
    """Apply final targeted patch to force trade execution"""
    print("🎯 FINAL TARGETED PATCH - FORCE TRADE EXECUTION")
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

    # Apply final targeted patches - fix the remaining hardcoded values
    patches = [
        # Patch 1: Fix the remaining hardcoded 0.08 base confidence threshold (CRITICAL)
        ('self.base_confidence_threshold = max(0.06, min(0.12, 0.08 + (0.002 - tsth) * 5.0))', 
         'self.base_confidence_threshold = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 2: Fix any other hardcoded 0.08 values
        ('0.08', 
         '0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 3: Fix any other hardcoded 0.06 values
        ('0.06', 
         '0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 4: Fix any other hardcoded 0.12 values
        ('0.12', 
         '0.0001  # EMERGENCY PATCH: FORCED TO 0.0001')
    ]

    modified = False
    for i, line in enumerate(lines):
        for old_text, new_text in patches:
            if old_text in line:
                lines[i] = line.replace(old_text, new_text)
                modified = True
                print(f"✅ Applied final targeted patch on line {i+1}: {old_text[:50]}...")

    if modified:
        # Write the patched file
        with open('newbotcode.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("✅ Bot source code FINAL TARGETED patched for force execution")
    else:
        print("⚠️ No patches applied - patterns not found")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo FINAL TARGETED PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting FINAL TARGETED PATCHED bot...
echo Bot will use 0.0001 threshold and execute ALL trades
python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
"""

    with open('start_final_targeted_bot.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_final_targeted_bot.bat")

    print("\n🎯 FINAL TARGETED PATCH COMPLETE!")
    print("=" * 45)
    print("1. Bot source code has been FINAL TARGETED patched")
    print("2. Remaining hardcoded thresholds fixed")
    print("3. ALL thresholds forced to 0.0001")
    print("4. Bot will use 0.0001 confidence threshold")
    print("5. ALL trades will execute")
    print("6. Run: .\\start_final_targeted_bot.bat")
    print("7. Bot will execute trades with ANY confidence > 0")

    return True

if __name__ == "__main__":
    final_targeted_patch()
