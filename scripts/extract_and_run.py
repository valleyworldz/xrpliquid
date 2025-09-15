#!/usr/bin/env python3
"""
Extract credentials from encrypted file and run ultimate position resolver
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def extract_credentials():
    """Extract credentials from the encrypted file"""
    try:
        with open('credentials/encrypted_credentials.dat', 'r') as f:
            lines = f.readlines()
        
        # Extract credentials from the last two lines
        address = None
        private_key = None
        
        for line in lines:
            if line.startswith('address='):
                address = line.split('=')[1].strip()
            elif line.startswith('private_key='):
                private_key = line.split('=')[1].strip()
        
        return private_key, address
        
    except Exception as e:
        print(f"Error reading credentials: {e}")
        return None, None

def main():
    print("🔐 Extracting credentials from encrypted file...")
    
    # Extract credentials
    private_key, address = extract_credentials()
    
    if not private_key or not address:
        print("❌ Failed to extract credentials")
        return False
    
    # Set environment variables
    os.environ['HYPERLIQUID_PRIVATE_KEY'] = private_key
    os.environ['HYPERLIQUID_API_KEY'] = address
    
    print("✅ Credentials extracted and environment variables set")
    print(f"📍 Address: {address[:10]}...")
    print("🔐 Private key: [LOADED]")
    
    # Run the ultimate position resolver
    print("\n🚀 Running Ultimate Position Resolver...")
    print("=" * 50)
    
    # Import and run the resolver directly
    from ultimate_position_resolver import main as run_resolver
    run_resolver()
    
    return True

if __name__ == "__main__":
    main() 