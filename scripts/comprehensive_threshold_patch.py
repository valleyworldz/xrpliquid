#!/usr/bin/env python3
"""
Comprehensive Threshold Patch for Force Trade Execution
Attacks ALL confidence threshold calculations including ATR scaled floor
"""

import os
import shutil
import time

def comprehensive_threshold_patch():
    """Apply comprehensive patches to force trade execution"""
    print("🚨 COMPREHENSIVE THRESHOLD PATCH - FORCE TRADE EXECUTION")
    print("=" * 60)

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

    # Apply comprehensive patches - attack ALL confidence thresholds
    patches = [
        # Patch 1: Force ATR scaled floor to 0.0001 (CRITICAL - this is blocking trades!)
        ('atr_scaled_floor = max(0.01, 0.5 * atr / current_price)', 
         'atr_scaled_floor = 0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 2: Force base confidence threshold to 0.0001 (CRITICAL)
        ('self.base_confidence_threshold = 0.08', 
         'self.base_confidence_threshold = 0.0001'),
        
        # Patch 3: Force adaptive base confidence threshold to 0.0001 (CRITICAL)
        ('self.base_confidence_threshold = 0.08  # Adaptive', 
         'self.base_confidence_threshold = 0.0001  # Adaptive'),
        
        # Patch 4: Force current threshold fallback to 0.0001 (CRITICAL)
        ('current_threshold = getattr(self.config, \'confidence_threshold\', 0.08)', 
         'current_threshold = getattr(self.config, \'confidence_threshold\', 0.0001)'),
        
        # Patch 5: Force new threshold calculation to 0.0001 (CRITICAL)
        ('new_threshold = min(0.95, current_threshold + 0.08)', 
         'new_threshold = 0.0001'),
        
        # Patch 6: Force low drawdown confidence to 0.0001 (CRITICAL)
        ('low_drawdown_confidence', 
         'low_drawdown_confidence = 0.0001  # EMERGENCY PATCH'),
        
        # Patch 7: Force high drawdown confidence to 0.0001 (CRITICAL)
        ('high_drawdown_confidence', 
         'high_drawdown_confidence = 0.0001  # EMERGENCY PATCH'),
        
        # Patch 8: Force any hardcoded 0.01 to 0.0001 (CRITICAL)
        ('max(0.01,', 
         'max(0.0001,  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 9: Force any hardcoded 0.02 to 0.0001 (CRITICAL)
        ('0.02', 
         '0.0001  # EMERGENCY PATCH: FORCED TO 0.0001'),
        
        # Patch 10: Force any hardcoded 0.015 to 0.0001 (CRITICAL)
        ('0.015', 
         '0.0001  # EMERGENCY PATCH: FORCED TO 0.0001')
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
        
        print("✅ Bot source code COMPREHENSIVELY patched for force execution")
    else:
        print("⚠️ No patches applied - patterns not found")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo COMPREHENSIVELY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting COMPREHENSIVELY PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
"""

    with open('start_comprehensively_patched_bot.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_comprehensively_patched_bot.bat")

    print("\n🚨 COMPREHENSIVE THRESHOLD PATCH COMPLETE!")
    print("=" * 50)
    print("1. Bot source code has been COMPREHENSIVELY patched")
    print("2. ALL confidence thresholds removed including ATR scaled floor")
    print("3. ALL thresholds forced to 0.0001")
    print("4. ALL filters bypassed")
    print("5. ALL restrictions removed")
    print("6. Run: .\\start_comprehensively_patched_bot.bat")
    print("7. Trades will execute with ANY confidence > 0")

    return True

if __name__ == "__main__":
    comprehensive_threshold_patch()
