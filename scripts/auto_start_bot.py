#!/usr/bin/env python3
import subprocess
import sys
import os

def auto_start_bot():
    # Set environment variables
    os.environ['BOT_BYPASS_INTERACTIVE'] = 'true'
    os.environ['BOT_AUTO_START'] = 'true'
    os.environ['BOT_FORCE_TRADING'] = 'true'
    
    # Start bot with auto-selection
    cmd = [
        sys.executable, 'newbotcode.py',
        '--fee-threshold-multi', '0.01',
        '--disable-rsi-veto',
        '--disable-momentum-veto', 
        '--disable-microstructure-veto'
    ]
    
    # Create process with stdin pipe for auto-selection
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send "1" to select option 1 automatically
    stdout, stderr = process.communicate(input="1\n")
    
    print("Bot started with auto-selection")
    print("STDOUT:", stdout)
    if stderr:
        print("STDERR:", stderr)

if __name__ == "__main__":
    auto_start_bot()
