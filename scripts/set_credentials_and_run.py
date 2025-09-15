#!/usr/bin/env python3
"""
Set credentials from encrypted file and run the bot
"""

import os
import sys
import subprocess
from pathlib import Path

def set_credentials():
    """Set credentials from encrypted file"""
    try:
        # Read the encrypted credentials file
        cred_file = Path("credentials/encrypted_credentials.dat")
        if not cred_file.exists():
            print("❌ Credentials file not found")
            return False
            
        with open(cred_file, 'r') as f:
            lines = f.readlines()
            
        # Extract credentials from the last two lines
        address = None
        private_key = None
        
        for line in lines:
            if line.startswith('address='):
                address = line.split('=')[1].strip()
            elif line.startswith('private_key='):
                private_key = line.split('=')[1].strip()
                
        if not address or not private_key:
            print("❌ Could not extract credentials from file")
            return False
            
        # Set environment variables
        os.environ['HYPERLIQUID_WALLET_ADDRESS'] = address
        os.environ['HYPERLIQUID_PRIVATE_KEY'] = private_key
        
        print(f"✅ Credentials set:")
        print(f"   Address: {address[:10]}...{address[-10:]}")
        print(f"   Private Key: {private_key[:10]}...{private_key[-10:]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting credentials: {e}")
        return False

def run_bot():
    """Run the ultimate master bot"""
    try:
        print("🚀 Starting Ultimate Master Bot...")
        
        # Run the bot
        result = subprocess.run([
            sys.executable, 
            "scripts/run_ultimate_master_bot.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Bot started successfully")
            print(result.stdout)
        else:
            print("❌ Bot failed to start")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"❌ Error running bot: {e}")

def main():
    """Main function"""
    print("🔐 Setting up credentials and running bot...")
    print("=" * 50)
    
    # Set credentials
    if not set_credentials():
        return
        
    # Run bot
    run_bot()

if __name__ == "__main__":
    main() 