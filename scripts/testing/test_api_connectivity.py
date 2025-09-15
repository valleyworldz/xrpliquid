#!/usr/bin/env python3
"""
API Connectivity Test - Liquid Manus Trading System
=================================================

This script tests the connectivity to the HyperLiquid API using our secure credentials.
It verifies that we can:
1. Load credentials securely
2. Connect to the API
3. Get user state
4. Get account information

Author: Liquid Manus Development Team
Version: 1.0.0
Last Updated: 2025-01-27
"""

import requests
import json
import time
from typing import Dict, Optional, Any
from core.utils.credential_handler import SecureCredentialHandler

class HyperLiquidAPITester:
    """Test HyperLiquid API connectivity"""
    
    def __init__(self):
        """Initialize the API tester"""
        self.base_url = "https://api.hyperliquid.xyz"
        self.credentials = None
        self.handler = SecureCredentialHandler()
    
    def load_credentials(self) -> bool:
        """Load credentials securely"""
        try:
            print(" Loading credentials...")
            self.credentials = self.handler.load_credentials()
            
            if self.credentials is None:
                print(" Failed to load credentials")
                return False
            
            print(" Credentials loaded successfully")
            print(f"  Address: {self.credentials['address'][:10]}...{self.credentials['address'][-8:]}")
            return True
        except Exception as e:
            print(f" Error loading credentials: {e}")
            return False
    
    def test_basic_connectivity(self) -> bool:
        """Test basic API connectivity"""
        try:
            print("\n Testing basic API connectivity...")
            
            # Test the info endpoint
            response = requests.get(f"{self.base_url}/info", timeout=10)
            
            if response.status_code == 200:
                print(" Basic connectivity successful")
                return True
            else:
                print(f" Basic connectivity failed: {response.status_code}")
                return False
        except Exception as e:
            print(f" Basic connectivity error: {e}")
            return False
    
    def test_user_state(self) -> Optional[Dict[str, Any]]:
        """Test getting user state"""
        try:
            print("\n Testing user state...")
            
            if not self.credentials:
                print(" No credentials available")
                return None
            
            # Prepare the request payload
            payload = {
                "type": "clearinghouseState",
                "user": self.credentials['address']
            }
            
            response = requests.post(
                f"{self.base_url}/info",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(" User state retrieved successfully")
                return data
            else:
                print(f" User state failed: {response.status_code}")
                return None
        except Exception as e:
            print(f" User state error: {e}")
            return None
    
    def test_account_info(self) -> Optional[Dict[str, Any]]:
        """Test getting account information"""
        try:
            print("\n Testing account information...")
            
            if not self.credentials:
                print(" No credentials available")
                return None
            
            # Prepare the request payload
            payload = {
                "type": "allMids",
                "user": self.credentials['address']
            }
            
            response = requests.post(
                f"{self.base_url}/info",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(" Account information retrieved successfully")
                return data
            else:
                print(f" Account info failed: {response.status_code}")
                return None
        except Exception as e:
            print(f" Account info error: {e}")
            return None
    
    def run_full_test(self) -> bool:
        """Run the complete API connectivity test"""
        print(" Starting HyperLiquid API Connectivity Test")
        print("=" * 50)
        
        # Test 1: Load credentials
        if not self.load_credentials():
            return False
        
        # Test 2: Basic connectivity
        if not self.test_basic_connectivity():
            return False
        
        # Test 3: User state
        user_state = self.test_user_state()
        if user_state:
            print(f"  User State: {json.dumps(user_state, indent=2)}")
        
        # Test 4: Account info
        account_info = self.test_account_info()
        if account_info:
            print(f"  Account Info: {json.dumps(account_info, indent=2)}")
        
        print("\n" + "=" * 50)
        print(" API Connectivity Test Completed Successfully!")
        return True

def main():
    """Main function"""
    tester = HyperLiquidAPITester()
    
    try:
        success = tester.run_full_test()
        if success:
            print("\n All tests passed! The system is ready for trading.")
        else:
            print("\n Some tests failed. Please check the configuration.")
    except KeyboardInterrupt:
        print("\n  Test interrupted by user")
    except Exception as e:
        print(f"\n Unexpected error: {e}")

if __name__ == "__main__":
    main()
