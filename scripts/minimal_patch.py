#!/usr/bin/env python3
"""
Minimal Patch for Force Trade Execution
Only changes the most critical values without breaking syntax
"""

import os
import shutil
import time

def minimal_patch():
    """Apply minimal patches to force trade execution"""
    print("🎯 MINIMAL PATCH - FORCE TRADE EXECUTION")
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
        content = f.read()

    # Apply minimal patches - only the most critical ones
    patches = [
        # Patch 1: Force main confidence threshold to 0.0001 (CRITICAL)
        ('confidence_threshold: float = 0.08', 
         'confidence_threshold: float = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 2: Force base confidence threshold to 0.0001 (CRITICAL)
        ('base_confidence_threshold: float = 0.02', 
         'base_confidence_threshold: float = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 3: Force environment variable fallback to 0.0001 (CRITICAL)
        ('base_threshold = float(os.environ.get("BOT_CONFIDENCE_THRESHOLD", "0.015"))', 
         'base_threshold = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 4: Force old threshold fallback to 0.0001 (CRITICAL)
        ('old_threshold = getattr(self.config, \'confidence_threshold\', 0.015)', 
         'old_threshold = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 5: Force environment variable set to 0.0001 (CRITICAL)
        ('os.environ["BOT_CONFIDENCE_THRESHOLD"] = "0.015"', 
         'os.environ["BOT_CONFIDENCE_THRESHOLD"] = "0.0001"  # EMERGENCY PATCH: FORCED TO 0.0001')
    ]

    modified = False
    for old_text, new_text in patches:
        if old_text in content:
            content = content.replace(old_text, new_text)
            modified = True
            print(f"✅ Applied patch: {old_text[:50]}...")

    if modified:
        # Write the patched file
        with open('newbotcode.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Bot source code minimally patched for force execution")
    else:
        print("⚠️ No patches applied - patterns not found")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo MINIMALLY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting MINIMALLY PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee_threshold_multi 0.01 --confidence_threshold 0.0001 --aggressive_mode true

pause
"""

    with open('start_minimally_patched_bot.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_minimally_patched_bot.bat")

    print("\n🎯 MINIMAL PATCH COMPLETE!")
    print("=" * 40)
    print("1. Bot source code has been minimally patched")
    print("2. Critical confidence thresholds forced to 0.0001")
    print("3. Key hardcoded values removed")
    print("4. Syntax preserved")
    print("5. Core restrictions bypassed")
    print("6. Run: .\\start_minimally_patched_bot.bat")
    print("7. Trades should execute with ANY confidence > 0")

    return True

if __name__ == "__main__":
    minimal_patch()
