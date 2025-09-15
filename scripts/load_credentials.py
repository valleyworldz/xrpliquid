#!/usr/bin/env python3
"""
Load encrypted credentials and set environment variables
"""

import os
import base64
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def load_credentials():
    """Load credentials from encrypted file and set environment variables"""
    try:
        # Path to encrypted credentials file
        creds_file = os.path.join(os.path.dirname(__file__), 'credentials/encrypted_credentials.dat')
        
        if not os.path.exists(creds_file):
            print("❌ Encrypted credentials file not found")
            return False
        
        # Read encrypted data
        with open(creds_file, 'r') as f:
            content = f.read().strip()
        
        # Parse the content - it has both encrypted data and plain text
        lines = content.split('\n')
        encrypted_data = lines[0]
        
        # Extract plain text credentials
        address = None
        private_key = None
        
        for line in lines[1:]:
            if line.startswith('address='):
                address = line.split('=')[1]
            elif line.startswith('private_key='):
                private_key = line.split('=')[1]
        
        if address and private_key:
            # Set environment variables
            os.environ['HYPERLIQUID_API_KEY'] = address
            os.environ['HYPERLIQUID_PRIVATE_KEY'] = private_key
            
            print("✅ Credentials loaded successfully")
            print(f"📍 Address: {address[:10]}...{address[-10:]}")
            print("🔐 Private key: [LOADED - NOT DISPLAYED]")
            return True
        else:
            print("❌ Could not extract credentials from file")
            return False
            
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return False

if __name__ == "__main__":
    success = load_credentials()
    if success:
        print("🚀 Ready to run bot with loaded credentials")
    else:
        print("❌ Failed to load credentials") 