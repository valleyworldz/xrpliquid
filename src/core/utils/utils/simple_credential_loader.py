#!/usr/bin/env python3
"""
Simple Credential Loader
Loads credentials directly from environment variables
No encryption, no complex handling - just simple .env support
"""

import os
from typing import Optional, Dict
from dotenv import load_dotenv

class SimpleCredentialLoader:
    """Simple credential loader that uses environment variables only"""
    
    def __init__(self):
        """Initialize the credential loader"""
        # Load .env file if it exists (with error handling)
        try:
            load_dotenv()
        except Exception:
            # If .env loading fails, continue with environment variables
            pass
        
    def load_credentials(self) -> Optional[Dict[str, str]]:
        """Load credentials from environment variables"""
        try:
            # Get credentials from environment
            address = os.environ.get('HYPERLIQUID_ADDRESS')
            private_key = os.environ.get('HYPERLIQUID_PRIVATE_KEY')
            
            if not address or not private_key:
                print("❌ Missing credentials in environment variables")
                print("   Please set HYPERLIQUID_ADDRESS and HYPERLIQUID_PRIVATE_KEY")
                return None
            
            # Set API key for compatibility
            os.environ['HYPERLIQUID_API_KEY'] = address
            
            credentials = {
                'address': address,
                'private_key': private_key,
                'api_key': address
            }
            
            print("✅ Credentials loaded from environment variables")
            return credentials
            
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return None
    
    def verify_credentials(self) -> bool:
        """Verify that credentials are properly loaded"""
        try:
            creds = self.load_credentials()
            if not creds:
                return False
            
            # Basic validation
            address = creds.get('address', '')
            private_key = creds.get('private_key', '')
            
            if not address.startswith('0x') or len(address) != 42:
                print("❌ Invalid address format")
                return False
            
            if len(private_key) != 64:
                print("❌ Invalid private key length")
                return False
            
            print("✅ Credentials verified successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error verifying credentials: {e}")
            return False
    
    def get_api_config(self) -> Dict[str, str]:
        """Get API configuration from environment"""
        return {
            'api_url': os.environ.get('HYPERLIQUID_API_URL', 'https://api.hyperliquid.xyz'),
            'websocket_url': os.environ.get('HYPERLIQUID_WEBSOCKET_URL', 'wss://api.hyperliquid.xyz/ws'),
            'testnet': str(os.environ.get('HYPERLIQUID_TESTNET', 'false'))
        }
    
    def get_trading_config(self) -> Dict[str, float]:
        """Get trading configuration from environment"""
        return {
            'min_position_size': float(os.environ.get('MIN_POSITION_SIZE', '500')),
            'max_positions': float(os.environ.get('MAX_POSITIONS', '2')),
            'risk_percentage': float(os.environ.get('RISK_PERCENTAGE', '2.0'))
        }

def load_credentials() -> Optional[Dict[str, str]]:
    """Convenience function to load credentials"""
    loader = SimpleCredentialLoader()
    return loader.load_credentials()

def verify_credentials() -> bool:
    """Convenience function to verify credentials"""
    loader = SimpleCredentialLoader()
    return loader.verify_credentials()

if __name__ == "__main__":
    # Test the credential loader
    print("🔐 Testing Simple Credential Loader")
    print("=" * 40)
    
    loader = SimpleCredentialLoader()
    
    # Test credential loading
    creds = loader.load_credentials()
    if creds:
        print(f"✅ Address: {creds['address'][:10]}...")
        print(f"✅ Private Key: {creds['private_key'][:10]}...")
    else:
        print("❌ Failed to load credentials")
    
    # Test verification
    if loader.verify_credentials():
        print("✅ Credentials verified")
    else:
        print("❌ Credentials verification failed")
    
    # Test API config
    api_config = loader.get_api_config()
    print(f"✅ API URL: {api_config['api_url']}")
    
    # Test trading config
    trading_config = loader.get_trading_config()
    print(f"✅ Min Position: ${trading_config['min_position_size']}")
    print(f"✅ Max Positions: {trading_config['max_positions']}") 