#!/usr/bin/env python3
"""
Simple API Test
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.utils.credential_handler import SecureCredentialHandler
import requests
import json

def test_credentials():
    """Test credential loading"""
    print("Testing credential loading...")
    
    handler = SecureCredentialHandler()
    credentials = handler.load_credentials()
    
    if credentials:
        print(" Credentials loaded successfully")
        print(f"Address: {credentials['address']}")
        return credentials
    else:
        print(" Failed to load credentials")
        return None

def test_api_connectivity():
    """Test API connectivity"""
    print("\nTesting API connectivity...")
    
    try:
        response = requests.get("https://api.hyperliquid.xyz/info", timeout=10)
        if response.status_code == 200:
            print(" API connectivity successful")
            return True
        else:
            print(f" API connectivity failed: {response.status_code}")
            return False
    except Exception as e:
        print(f" API connectivity error: {e}")
        return False

def main():
    """Main function"""
    print(" Starting Simple API Test")
    print("=" * 40)
    
    # Test credentials
    credentials = test_credentials()
    
    # Test API connectivity
    api_ok = test_api_connectivity()
    
    if credentials and api_ok:
        print("\n All tests passed!")
        
        # Try to get user state
        if credentials:
            print("\nTesting user state...")
            try:
                payload = {
                    "type": "clearinghouseState",
                    "user": credentials['address']
                }
                
                response = requests.post(
                    "https://api.hyperliquid.xyz/info",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(" User state retrieved successfully")
                    print(f"Response: {json.dumps(data, indent=2)}")
                else:
                    print(f" User state failed: {response.status_code}")
            except Exception as e:
                print(f" User state error: {e}")
    else:
        print("\n Some tests failed")

if __name__ == "__main__":
    main()
