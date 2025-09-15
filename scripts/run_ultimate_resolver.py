#!/usr/bin/env python3
"""
Run Ultimate Position Resolver with proper credential loading
"""

import os
import sys
import subprocess

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.utils.credential_manager import CredentialManager

def main():
    print("🔐 Loading credentials and setting environment variables...")
    
    try:
        # Load credentials
        cm = CredentialManager()
        private_key = cm.get_credential("HYPERLIQUID_PRIVATE_KEY")
        address = cm.get_credential("HYPERLIQUID_API_KEY")
        
        if not private_key or not address:
            print("❌ No credentials found")
            return False
        
        # Set environment variables
        os.environ['HYPERLIQUID_PRIVATE_KEY'] = private_key
        os.environ['HYPERLIQUID_API_KEY'] = address
        
        print("✅ Credentials loaded and environment variables set")
        print(f"📍 Address: {address[:10]}...")
        print("🔐 Private key: [LOADED]")
        
        # Run the ultimate position resolver
        print("\n🚀 Running Ultimate Position Resolver...")
        print("=" * 50)
        
        # Import and run the resolver directly
        from ultimate_position_resolver import main as run_resolver
        run_resolver()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 