#!/usr/bin/env python3
"""
Fixed API Test with Env Loader and Explicit Password
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.utils import load_env_from_file
from core.utils.credential_handler import SecureCredentialHandler
import requests
import json

# Load environment variables from file if present
load_env_from_file()

PASSWORD = "HyperLiquidSecure2025!"

def test_credentials():
    """Test credential loading"""
    print("Testing credential loading...")
    
    handler = SecureCredentialHandler()
    credentials = handler.load_credentials(PASSWORD)
    
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
        # Use POST with a simple payload for the info endpoint
        payload = {"type": "meta"}
        response = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=10)
        
        if response.status_code == 200:
            print(" API connectivity successful")
            return True
        else:
            print(f" API connectivity failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f" API connectivity error: {e}")
        return False

def test_user_state(credentials):
    """Test getting user state"""
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
            print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            return data
        else:
            print(f" User state failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f" User state error: {e}")
        return None

def main():
    """Main function"""
    print(" Starting Fixed API Test with Env Loader and Explicit Password")
    print("=" * 40)
    
    # Test credentials
    credentials = test_credentials()
    
    # Test API connectivity
    api_ok = test_api_connectivity()
    
    if credentials and api_ok:
        print("\n Basic tests passed!")
        
        # Test user state
        user_state = test_user_state(credentials)
        
        if user_state:
            print("\n All tests passed! System is ready for trading.")
        else:
            print("\n  User state test failed, but basic connectivity works.")
    else:
        print("\n Basic tests failed")

if __name__ == "__main__":
    main()
