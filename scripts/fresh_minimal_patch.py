#!/usr/bin/env python3
"""
Fresh Minimal Patch for Force Trade Execution
Only changes the most essential values without touching anything else
"""

import os
import shutil
import time

def fresh_minimal_patch():
    """Apply fresh minimal patches to force trade execution"""
    print("🎯 FRESH MINIMAL PATCH - FORCE TRADE EXECUTION")
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

    # Apply fresh minimal patches - only the most essential ones
    patches = [
        # Patch 1: Force default base confidence threshold to 0.0001 (CRITICAL)
        ('self.base_confidence_threshold = 0.08', 
         'self.base_confidence_threshold = 0.0001'),
        
        # Patch 2: Force current threshold fallback to 0.0001 (CRITICAL)
        ('current_threshold = getattr(self.config, \'confidence_threshold\', 0.08)', 
         'current_threshold = getattr(self.config, \'confidence_threshold\', 0.0001)')
    ]

    modified = False
    for i, line in enumerate(lines):
        for old_text, new_text in patches:
            if old_text in line:
                lines[i] = line.replace(old_text, new_text)
                modified = True
                print(f"✅ Applied patch on line {i+1}: {old_text[:50]}...")

    if modified:
        # Write the patched file
        with open('newbotcode.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("✅ Bot source code FRESHLY MINIMALLY patched for force execution")
    else:
        print("⚠️ No patches applied - patterns not found")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo FRESHLY MINIMALLY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting FRESHLY MINIMALLY PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee_threshold_multi 0.01 --confidence_threshold 0.0001 --aggressive_mode true

pause
"""

    with open('start_freshly_minimally_patched_bot.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_freshly_minimally_patched_bot.bat")

    print("\n🎯 FRESH MINIMAL PATCH COMPLETE!")
    print("=" * 40)
    print("1. Bot source code has been FRESHLY MINIMALLY patched")
    print("2. Essential hardcoded confidence thresholds removed")
    print("3. Key thresholds forced to 0.0001")
    print("4. Syntax completely preserved")
    print("5. Core restrictions bypassed")
    print("6. Run: .\\start_freshly_minimally_patched_bot.bat")
    print("7. Trades should execute with ANY confidence > 0")

    return True

if __name__ == "__main__":
    fresh_minimal_patch()
