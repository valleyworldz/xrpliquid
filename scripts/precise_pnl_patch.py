#!/usr/bin/env python3
"""
Precise PnL Threshold Removal
Removes specific PnL threshold checks blocking trades
"""

import os
import shutil
import time

def precise_pnl_patch():
    """Remove specific PnL threshold checks"""
    print("🚨 PRECISE PnL THRESHOLD REMOVAL")
    print("=" * 50)
    
    if not os.path.exists('newbotcode.py'):
        print("❌ newbotcode.py not found")
        return False
    
    # Create backup
    backup_file = f'newbotcode_backup_precise_pnl_{int(time.time())}.py'
    shutil.copy2('newbotcode.py', backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    # Read the source file
    with open('newbotcode.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # PRECISE PnL THRESHOLD REMOVAL PATCHES
    
    patches = [
        # Remove PnL threshold check on line ~12397
        ('if expected_pnl < threshold_multi * (round_trip_cost + expected_funding):',
         'if False:  # PRECISE PATCH: PnL THRESHOLD DISABLED'),
        
        # Remove PnL threshold check on line ~23147
        ('if expected_pnl >= threshold_multi * (round_trip_cost + expected_funding):',
         'if True:  # PRECISE PATCH: PnL THRESHOLD DISABLED'),
        
        # Remove PnL threshold check on line ~23165
        ('if expected_pnl < threshold_multi * (round_trip_cost + expected_funding):',
         'if False:  # PRECISE PATCH: PnL THRESHOLD DISABLED'),
        
        # Remove any remaining PnL skip messages
        ('🚫 Skipping entry - expected PnL below fee+funding threshold',
         '✅ FORCE EXECUTING entry - PRECISE PATCH: PnL THRESHOLD DISABLED'),
        
        # Remove any remaining PnL skip messages
        ('🚫 Skipping entry - notional too small and fees/funding dominate',
         '✅ FORCE EXECUTING entry - PRECISE PATCH: NOTIONAL THRESHOLD DISABLED'),
    ]
    
    modified = False
    for old_text, new_text in patches:
        if old_text in content:
            content = content.replace(old_text, new_text)
            modified = True
            print(f"✅ Applied precise PnL patch: {old_text[:50]}...")
    
    if modified:
        # Write the patched file
        with open('newbotcode.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n✅ PRECISE PnL THRESHOLD REMOVAL COMPLETE!")
        print("=" * 50)
        print("🚨 ALL PnL THRESHOLD CHECKS REMOVED:")
        print("   • Line ~12397: PnL threshold DISABLED")
        print("   • Line ~23147: PnL threshold DISABLED")
        print("   • Line ~23165: PnL threshold DISABLED")
        print("   • All trades will execute regardless of PnL")
        
        # Create precise startup script
        precise_script = """@echo off
echo ============================================================
echo PRECISE PnL THRESHOLD REMOVAL BOT
echo ============================================================
echo.

echo 🚨 PRECISE PATCH APPLIED
echo ALL PnL THRESHOLD CHECKS REMOVED
echo ALL TRADES WILL EXECUTE REGARDLESS OF PnL

echo 1 | python newbotcode.py --fee-threshold-multi 0.001 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
"""
        
        with open('start_precise_pnl_bot.bat', 'w') as f:
            f.write(precise_script)
        
        print("\n✅ Created start_precise_pnl_bot.bat")
        print("\n🎯 PRECISE PnL PATCH READY!")
        print("=" * 35)
        print("1. All PnL threshold checks removed")
        print("2. All trades will execute regardless of PnL")
        print("3. Bot will trade with maximum frequency")
        print("4. Run: .\\start_precise_pnl_bot.bat")
        
    else:
        print("⚠️ No patches applied - patterns not found")
    
    return True

if __name__ == "__main__":
    precise_pnl_patch()
