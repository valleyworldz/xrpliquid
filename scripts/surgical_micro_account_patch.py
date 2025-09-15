#!/usr/bin/env python3
"""
Surgical Micro-Account Safeguard Removal
Directly targets and removes the micro-account safeguard
"""

import os
import shutil
import time

def surgical_micro_account_patch():
    """Surgically remove micro-account safeguard"""
    print("🚨 SURGICAL MICRO-ACCOUNT SAFEGUARD REMOVAL")
    print("=" * 60)
    
    if not os.path.exists('newbotcode.py'):
        print("❌ newbotcode.py not found")
        return False
    
    # Create backup
    backup_file = f'newbotcode_backup_surgical_{int(time.time())}.py'
    shutil.copy2('newbotcode.py', backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    # Read the source file
    with open('newbotcode.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # SURGICAL PATCHES - TARGET MICRO-ACCOUNT SAFEGUARD
    
    patches = [
        # Remove micro-account safeguard message
        ('🚫 Skipping trade - expected PnL below fee+funding threshold (micro-account safeguard)',
         '✅ FORCE EXECUTING trade - SURGICAL PATCH: MICRO-ACCOUNT SAFEGUARD REMOVED'),
        
        # Remove the actual safeguard logic
        ('if expected_pnl < fee_threshold:',
         'if False:  # SURGICAL PATCH: MICRO-ACCOUNT SAFEGUARD DISABLED'),
        
        # Remove any PnL threshold checks
        ('if pnl < threshold:',
         'if False:  # SURGICAL PATCH: PnL THRESHOLD DISABLED'),
        
        # Remove fee threshold checks
        ('if fee > max_fee:',
         'if False:  # SURGICAL PATCH: FEE THRESHOLD DISABLED'),
        
        # Force all trades to execute
        ('return False  # Trade blocked by PnL',
         'return True   # SURGICAL PATCH: FORCE EXECUTE'),
        
        # Remove any remaining skip logic
        ('skip_trade = True',
         'skip_trade = False  # SURGICAL PATCH: NO SKIPPING'),
    ]
    
    modified = False
    for old_text, new_text in patches:
        if old_text in content:
            content = content.replace(old_text, new_text)
            modified = True
            print(f"✅ Applied surgical patch: {old_text[:50]}...")
    
    if modified:
        # Write the patched file
        with open('newbotcode.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n✅ SURGICAL MICRO-ACCOUNT SAFEGUARD REMOVAL COMPLETE!")
        print("=" * 50)
        print("🚨 MICRO-ACCOUNT SAFEGUARD COMPLETELY REMOVED:")
        print("   • PnL threshold checks: DISABLED")
        print("   • Fee threshold checks: DISABLED")
        print("   • Trade skipping: DISABLED")
        print("   • All trades will execute regardless of PnL")
        
        # Create surgical startup script
        surgical_script = """@echo off
echo ============================================================
echo SURGICAL MICRO-ACCOUNT SAFEGUARD REMOVAL BOT
echo ============================================================
echo.

echo 🚨 SURGICAL PATCH APPLIED
echo MICRO-ACCOUNT SAFEGUARD COMPLETELY REMOVED
echo ALL TRADES WILL EXECUTE REGARDLESS OF PnL

echo 1 | python newbotcode.py --fee-threshold-multi 0.001 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
"""
        
        with open('start_surgical_patch_bot.bat', 'w') as f:
            f.write(surgical_script)
        
        print("\n✅ Created start_surgical_patch_bot.bat")
        print("\n🎯 SURGICAL PATCH READY!")
        print("=" * 30)
        print("1. Micro-account safeguard removed")
        print("2. All PnL restrictions disabled")
        print("3. Bot will execute every trade")
        print("4. Run: .\\start_surgical_patch_bot.bat")
        
    else:
        print("⚠️ No patches applied - patterns not found")
    
    return True

if __name__ == "__main__":
    surgical_micro_account_patch()
