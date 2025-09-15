
#!/usr/bin/env python3
"""
Enhanced Credential Handler - Fixed Version
==========================================
Handles secure credential management with proper error handling
"""

import os
import json
import base64
from typing import Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class EnhancedCredentialHandler:
    """Enhanced credential handler with proper error handling"""
    
    def __init__(self):
        self.credentials = None
        self.is_initialized = False
        self.encryption_key = None
    
    def initialize(self, password=None):
        """Initialize with proper error handling"""
        try:
            if password is None:
                password = os.environ.get('HL_CRED_PASSWORD', 'HyperLiquidSecure2025!')
            
            # Derive key with fixed salt
            salt = b'HyperLiquidSecureSalt2025'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            
            self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Credential initialization error: {e}")
            return False
    
    def load_credentials(self):
        """Load credentials with enhanced error handling"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return None
            
            # Get encrypted credentials
            encrypted_creds = os.environ.get('HL_ENCRYPTED_CREDS')
            if not encrypted_creds:
                print("No encrypted credentials found in environment")
                return None
            
            # Decrypt credentials
            cipher = Fernet(self.encryption_key)
            decrypted_data = cipher.decrypt(base64.b64decode(encrypted_creds))
            credentials = json.loads(decrypted_data.decode())
            
            # Validate credentials
            if self._validate_credentials(credentials):
                self.credentials = credentials
                return credentials
            else:
                print("Invalid credentials format")
                return None
                
        except Exception as e:
            print(f"Credential loading error: {e}")
            return None
    
    def _validate_credentials(self, credentials):
        """Validate credential format"""
        try:
            if not isinstance(credentials, dict):
                return False
            
            address = credentials.get('address')
            private_key = credentials.get('private_key')
            
            if not address or not private_key:
                return False
            
            # Basic format validation
            if not address.startswith('0x') or len(address) != 42:
                return False
            
            if len(private_key) != 64:
                return False
            
            # Validate hex format
            try:
                int(private_key, 16)
            except ValueError:
                return False
            
            return True
        except Exception:
            return False
    
    def get_credentials(self):
        """Get current credentials"""
        return self.credentials
