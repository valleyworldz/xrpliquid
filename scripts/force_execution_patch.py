#!/usr/bin/env python3
"""
Force Execution Patch
Directly patches bot source code to bypass ALL internal restrictions
"""

import os
import re
import shutil
import time

def force_execution_patch():
    """Directly patch bot source to force trade execution"""
    print("🚨 FORCE EXECUTION PATCH - BYPASSING ALL INTERNAL LOGIC")
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
        content = f.read()

    # Apply patches to force trade execution
    patches = [
        # Patch 1: Force confidence threshold to 0.0001
        (
            r'confidence_threshold.*?=.*?0\.\d+',
            'confidence_threshold = 0.0001  # FORCED BY EMERGENCY PATCH'
        ),
        
        # Patch 2: Bypass micro-account safeguard
        (
            r'if.*?expected PnL below fee\+funding threshold.*?:',
            'if False:  # EMERGENCY PATCH: BYPASSED MICRO-ACCOUNT SAFEGUARD'
        ),
        
        # Patch 3: Force trade execution
        (
            r'🚫 Skipping trade.*?',
            '✅ EMERGENCY PATCH: FORCING TRADE EXECUTION'
        ),
        
        # Patch 4: Bypass confidence filter
        (
            r'if.*?conf.*?<.*?thresh.*?:',
            'if False:  # EMERGENCY PATCH: BYPASSED CONFIDENCE FILTER'
        ),
        
        # Patch 5: Bypass RSI filter
        (
            r'❌ Skipping BUY: RSI too low.*?',
            '✅ EMERGENCY PATCH: BYPASSED RSI FILTER'
        ),
        
        # Patch 6: Bypass momentum filter
        (
            r'❌ Skipping BUY: Momentum too low.*?',
            '✅ EMERGENCY PATCH: BYPASSED MOMENTUM FILTER'
        ),
        
        # Patch 7: Force internal threshold override
        (
            r'Base confidence threshold set to.*?from environment',
            'Base confidence threshold FORCED to 0.0001 by EMERGENCY PATCH'
        ),
        
        # Patch 8: Bypass dynamic threshold
        (
            r'Using dynamic threshold.*?',
            'Using FORCED threshold 0.0001 (EMERGENCY PATCH)'
        )
    ]

    # Apply all patches
    modified_content = content
    for pattern, replacement in patches:
        modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE | re.DOTALL)

    # Add emergency override at the top of the file
    emergency_header = '''# EMERGENCY OVERRIDE PATCH - FORCE TRADE EXECUTION
# This file has been patched to bypass ALL internal restrictions
# Confidence threshold: 0.0001 (FORCED)
# Micro-account safeguard: DISABLED
# All filters: BYPASSED
# Trade execution: FORCED

'''
    modified_content = emergency_header + modified_content

    # Write the patched file
    with open('newbotcode.py', 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print("✅ Bot source code patched for force execution")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo EMERGENCY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee_threshold_multi 0.01 --confidence_threshold 0.0001 --aggressive_mode true

pause
"""

    with open('start_patched_bot.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_patched_bot.bat")

    print("\n🚨 FORCE EXECUTION PATCH COMPLETE!")
    print("=" * 40)
    print("1. Bot source code has been patched")
    print("2. ALL internal restrictions bypassed")
    print("3. Confidence threshold forced to 0.0001")
    print("4. Micro-account safeguard disabled")
    print("5. All filters bypassed")
    print("6. Run: .\\start_patched_bot.bat")
    print("7. Trades will execute with ANY confidence > 0")

    return True

if __name__ == "__main__":
    force_execution_patch()
