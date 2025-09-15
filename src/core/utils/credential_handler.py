#!/usr/bin/env python3
"""
Secure Credential Handler - Liquid Manus Trading System
====================================================

This module provides secure credential handling for the Liquid Manus trading system.
It handles loading, decryption, and validation of credentials.

Author: Liquid Manus Development Team
Version: 3.0.0
Last Updated: 2025-01-27
"""

import os
import json
import base64
from typing import Dict, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecureCredentialHandler:
    """Handles secure credential management"""
    
    def __init__(self):
        """Initialize the credential handler"""
        self.credentials: Optional[Dict[str, str]] = None
        self.is_initialized: bool = False
        self.encryption_key: Optional[bytes] = None
    
    def _derive_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = b'HyperLiquidSecureSalt2025'  # Fixed salt for development
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def initialize(self, password: Optional[str] = None) -> bool:
        """Initialize the credential handler"""
        try:
            if password is None:
                password = os.environ.get('HL_CRED_PASSWORD', 'HyperLiquidSecure2025!')
            
            self.encryption_key = self._derive_key(password)
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"❌ Failed to initialize credential handler: {e}")
            return False
    
    def _validate_credential_format(self, address: str, private_key: str) -> bool:
        """Validate credential format"""
        try:
            # Validate address format
            if not address.startswith('0x') or len(address) != 42:
                print("❌ Invalid address format")
                return False
            
            # Validate private key format (64 hex characters)
            if len(private_key) != 64:
                print("❌ Invalid private key format")
                return False
            
            # Try to convert private key to int to validate hex
            try:
                int(private_key, 16)
            except ValueError:
                print("❌ Invalid private key format (not hex)")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False
    
    def store_credentials_securely(self, address: str, private_key: str, password: Optional[str] = None) -> bool:
        """Store credentials securely"""
        try:
            # Validate credentials first
            if not self._validate_credential_format(address, private_key):
                raise ValueError("Invalid credential format")
            
            if not self.is_initialized:
                if not self.initialize(password):
                    raise ValueError("Failed to initialize credential handler")
            
            if self.encryption_key is None:
                raise ValueError("Encryption key not initialized")
            
            # Create Fernet cipher
            cipher = Fernet(self.encryption_key)
            
            # Encrypt credentials
            credentials_data = {
                'address': address,
                'private_key': private_key
            }
            
            encrypted_data = cipher.encrypt(json.dumps(credentials_data).encode())
            encrypted_str = base64.b64encode(encrypted_data).decode()
            
            # Store only encrypted data in environment
            os.environ['HL_ENCRYPTED_CREDS'] = encrypted_str
            
            # Save only encrypted data to file
            with open('secure_creds.env', 'w') as f:
                f.write(f"export HL_ENCRYPTED_CREDS='{encrypted_str}'\n")
                f.write(f"export HL_CRED_PASSWORD='{password or 'HyperLiquidSecure2025!'}'\n")
            
            self.credentials = credentials_data
            return True
        except Exception as e:
            print(f"❌ Failed to store credentials: {e}")
            raise  # Re-raise the exception for proper error handling
    
    def _find_creds_file(self) -> str:
        """Find the secure_creds.env file in current or project root directory"""
        local_path = os.path.join(os.getcwd(), 'secure_creds.env')
        if os.path.exists(local_path):
            return local_path
        # Try project root (assume this file is always in core/utils/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        root_path = os.path.join(project_root, 'secure_creds.env')
        if os.path.exists(root_path):
            return root_path
        return None

    def load_credentials(self, password: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Load credentials from environment or file"""
        try:
            # Check initialization state first
            if not self.is_initialized:
                # Try to initialize if password provided
                if password is not None:
                    if not self.initialize(password):
                        return None
                else:
                    # No password and not initialized - return None
                    return None
            
            if self.encryption_key is None:
                return None
            
            # Try loading from environment first
            encrypted_creds = os.environ.get('HL_ENCRYPTED_CREDS')
            if not encrypted_creds:
                # Try loading from file (search both current dir and project root)
                creds_file = self._find_creds_file()
                if creds_file:
                    with open(creds_file, 'r') as f:
                        for line in f:
                            if line.startswith('export HL_ENCRYPTED_CREDS='):
                                encrypted_creds = line.split('=', 1)[1].strip().strip("'")
                                break
            
            if not encrypted_creds:
                print("❌ No encrypted credentials found")
                return None
            
            try:
                # Decrypt credentials
                cipher = Fernet(self.encryption_key)
                decrypted_data = cipher.decrypt(base64.b64decode(encrypted_creds))
                decrypted_creds = json.loads(decrypted_data.decode())
                
                # Type check and validate decrypted credentials
                if not isinstance(decrypted_creds, dict):
                    print("❌ Invalid credential format (not a dictionary)")
                    return None
                
                address = decrypted_creds.get('address')
                private_key = decrypted_creds.get('private_key')
                
                if not isinstance(address, str) or not isinstance(private_key, str):
                    print("❌ Invalid credential format (missing required fields)")
                    return None
                
                # Validate decrypted credentials
                if not self._validate_credential_format(address, private_key):
                    return None
                
                self.credentials = decrypted_creds
                return self.credentials
            except Exception:
                print("❌ Failed to decrypt credentials (wrong password?)")
                return None
                
        except Exception as e:
            print(f"❌ Failed to load credentials: {e}")
            return None
    
    def get_credentials(self) -> Optional[Dict[str, str]]:
        """Get the current credentials"""
        return self.credentials if self.credentials else None
    
    def validate_credentials(self) -> bool:
        """Validate the current credentials"""
        try:
            if not self.credentials:
                return False
            
            return self._validate_credential_format(
                self.credentials['address'],
                self.credentials['private_key']
            )
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False

def main():
    """Main function for testing"""
    handler = SecureCredentialHandler()
    
    # Test storing credentials from environment
    try:
        address = os.environ.get('WALLET_ADDRESS', '')
        private_key = os.environ.get('PRIVATE_KEY', '')
        
        if not address or not private_key:
            print("❌ WALLET_ADDRESS and PRIVATE_KEY not found in environment")
            print("Please set these in your .env file")
            return
        
        success = handler.store_credentials_securely(address, private_key)
        
        if success:
            print("✅ Credentials stored successfully")
            
            # Test loading credentials
            credentials = handler.load_credentials()
            if credentials:
                print("✅ Credentials loaded successfully")
                print(f"  Address: {credentials['address'][:10]}...{credentials['address'][-8:]}")
                print("  Private Key: [ENCRYPTED]")
                
                # Test validation
                if handler.validate_credentials():
                    print("✅ Credentials validated successfully")
                else:
                    print("❌ Credential validation failed")
            else:
                print("❌ Failed to load credentials")
    except Exception as e:
        print(f"❌ Failed to store credentials: {e}")

if __name__ == "__main__":
    main() 