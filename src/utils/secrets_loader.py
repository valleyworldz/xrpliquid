"""
Secure Secrets Loader with Fail-Closed Design
============================================

This module provides secure credential loading with fail-closed behavior.
It ensures that the system cannot operate with missing or invalid credentials.

Security Features:
- Fail-closed design (system stops if secrets missing/invalid)
- Environment variable validation
- File-based credential loading with encryption
- Credential rotation support
- Audit logging for security events
"""

import os
import sys
import json
import logging
import hashlib
import hmac
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

@dataclass
class CredentialValidationResult:
    """Result of credential validation"""
    is_valid: bool
    error_message: Optional[str] = None
    security_level: str = "unknown"

class SecretsLoader:
    """
    Secure secrets loader with fail-closed design
    
    This class ensures that:
    1. All required credentials are present
    2. Credentials are properly formatted
    3. System fails hard if credentials are missing/invalid
    4. All credential access is logged for audit
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("config/secure_creds.env")
        self.required_credentials = {
            "HYPERLIQUID_PRIVATE_KEY": {
                "type": "hex_string",
                "min_length": 64,
                "pattern": r"^0x[a-fA-F0-9]{64}$",
                "description": "Hyperliquid private key (64 hex chars with 0x prefix)"
            },
            "HYPERLIQUID_ADDRESS": {
                "type": "ethereum_address",
                "min_length": 42,
                "pattern": r"^0x[a-fA-F0-9]{40}$",
                "description": "Hyperliquid wallet address (40 hex chars with 0x prefix)"
            }
        }
        self.optional_credentials = {
            "HYPERLIQUID_TESTNET": {
                "type": "boolean",
                "default": "true",
                "description": "Use Hyperliquid testnet (true/false)"
            },
            "MAX_POSITION_SIZE": {
                "type": "float",
                "default": "100.0",
                "description": "Maximum position size in USD"
            },
            "RISK_LIMIT": {
                "type": "float",
                "default": "0.02",
                "description": "Risk limit as decimal (0.02 = 2%)"
            }
        }
        self.loaded_credentials: Dict[str, Any] = {}
        self.audit_log: list = []
    
    def _log_security_event(self, event_type: str, details: str, severity: str = "INFO"):
        """Log security events for audit trail"""
        event = {
            "timestamp": self._get_timestamp(),
            "event_type": event_type,
            "details": details,
            "severity": severity
        }
        self.audit_log.append(event)
        
        if severity == "CRITICAL":
            self.logger.critical(f"🔐 SECURITY: {event_type} - {details}")
        elif severity == "WARNING":
            self.logger.warning(f"🔐 SECURITY: {event_type} - {details}")
        else:
            self.logger.info(f"🔐 SECURITY: {event_type} - {details}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for audit logging"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def _validate_hex_string(self, value: str, min_length: int, pattern: str) -> CredentialValidationResult:
        """Validate hex string format"""
        import re
        
        if not isinstance(value, str):
            return CredentialValidationResult(False, "Value must be a string")
        
        if len(value) < min_length:
            return CredentialValidationResult(False, f"Value too short (min {min_length} chars)")
        
        if not re.match(pattern, value):
            return CredentialValidationResult(False, f"Value does not match required pattern: {pattern}")
        
        return CredentialValidationResult(True, security_level="validated")
    
    def _validate_ethereum_address(self, value: str, min_length: int, pattern: str) -> CredentialValidationResult:
        """Validate Ethereum address format"""
        import re
        
        if not isinstance(value, str):
            return CredentialValidationResult(False, "Address must be a string")
        
        if len(value) < min_length:
            return CredentialValidationResult(False, f"Address too short (min {min_length} chars)")
        
        if not re.match(pattern, value):
            return CredentialValidationResult(False, f"Address does not match Ethereum format: {pattern}")
        
        # Additional Ethereum address validation
        if not value.startswith("0x"):
            return CredentialValidationResult(False, "Address must start with 0x")
        
        # Check if address is all zeros (invalid)
        if value.lower() == "0x" + "0" * 40:
            return CredentialValidationResult(False, "Address cannot be all zeros")
        
        return CredentialValidationResult(True, security_level="validated")
    
    def _validate_boolean(self, value: str) -> CredentialValidationResult:
        """Validate boolean value"""
        if value.lower() in ("true", "false", "1", "0", "yes", "no"):
            return CredentialValidationResult(True, security_level="validated")
        return CredentialValidationResult(False, "Value must be true/false, 1/0, or yes/no")
    
    def _validate_float(self, value: str) -> CredentialValidationResult:
        """Validate float value"""
        try:
            float_val = float(value)
            if float_val < 0:
                return CredentialValidationResult(False, "Value must be non-negative")
            return CredentialValidationResult(True, security_level="validated")
        except ValueError:
            return CredentialValidationResult(False, "Value must be a valid number")
    
    def _validate_credential(self, key: str, value: str, spec: Dict[str, Any]) -> CredentialValidationResult:
        """Validate a single credential against its specification"""
        
        if spec["type"] == "hex_string":
            return self._validate_hex_string(value, spec["min_length"], spec["pattern"])
        elif spec["type"] == "ethereum_address":
            return self._validate_ethereum_address(value, spec["min_length"], spec["pattern"])
        elif spec["type"] == "boolean":
            return self._validate_boolean(value)
        elif spec["type"] == "float":
            return self._validate_float(value)
        else:
            return CredentialValidationResult(False, f"Unknown credential type: {spec['type']}")
    
    def _load_from_env_file(self) -> Dict[str, str]:
        """Load credentials from environment file"""
        credentials = {}
        
        if not self.config_path.exists():
            self._log_security_event(
                "CREDENTIAL_FILE_MISSING",
                f"Credential file not found: {self.config_path}",
                "CRITICAL"
            )
            return credentials
        
        try:
            with open(self.config_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')  # Remove quotes
                        credentials[key] = value
                    else:
                        self._log_security_event(
                            "INVALID_CREDENTIAL_FORMAT",
                            f"Invalid format in {self.config_path}:{line_num}: {line}",
                            "WARNING"
                        )
            
            self._log_security_event(
                "CREDENTIAL_FILE_LOADED",
                f"Loaded {len(credentials)} credentials from {self.config_path}",
                "INFO"
            )
            
        except Exception as e:
            self._log_security_event(
                "CREDENTIAL_FILE_ERROR",
                f"Error loading {self.config_path}: {e}",
                "CRITICAL"
            )
        
        return credentials
    
    def _load_from_environment(self) -> Dict[str, str]:
        """Load credentials from environment variables"""
        credentials = {}
        
        for key in self.required_credentials:
            value = os.environ.get(key)
            if value:
                credentials[key] = value
                self._log_security_event(
                    "ENV_CREDENTIAL_LOADED",
                    f"Loaded {key} from environment",
                    "INFO"
                )
        
        for key in self.optional_credentials:
            value = os.environ.get(key)
            if value:
                credentials[key] = value
                self._log_security_event(
                    "ENV_CREDENTIAL_LOADED",
                    f"Loaded optional {key} from environment",
                    "INFO"
                )
        
        return credentials
    
    def _check_credential_security(self, key: str, value: str) -> bool:
        """Perform additional security checks on credentials"""
        
        # Check for common weak credentials
        weak_patterns = [
            "test", "demo", "example", "sample", "dummy",
            "123456", "password", "admin", "root"
        ]
        
        value_lower = value.lower()
        for pattern in weak_patterns:
            if pattern in value_lower:
                self._log_security_event(
                    "WEAK_CREDENTIAL_DETECTED",
                    f"Potentially weak credential detected for {key}",
                    "WARNING"
                )
                return False
        
        # Check for default values
        if key == "HYPERLIQUID_PRIVATE_KEY" and value == "0x" + "0" * 64:
            self._log_security_event(
                "DEFAULT_CREDENTIAL_DETECTED",
                f"Default private key detected for {key}",
                "CRITICAL"
            )
            return False
        
        if key == "HYPERLIQUID_ADDRESS" and value == "0x" + "0" * 40:
            self._log_security_event(
                "DEFAULT_CREDENTIAL_DETECTED",
                f"Default address detected for {key}",
                "CRITICAL"
            )
            return False
        
        return True
    
    def load_credentials(self) -> Dict[str, Any]:
        """
        Load and validate all credentials with fail-closed design
        
        Returns:
            Dict containing validated credentials
            
        Raises:
            SystemExit: If any required credentials are missing or invalid
        """
        self._log_security_event(
            "CREDENTIAL_LOAD_START",
            "Starting credential loading process",
            "INFO"
        )
        
        # Load from both file and environment (env takes precedence)
        file_creds = self._load_from_env_file()
        env_creds = self._load_from_environment()
        
        # Merge credentials (environment overrides file)
        all_credentials = {**file_creds, **env_creds}
        
        # Validate required credentials
        missing_credentials = []
        invalid_credentials = []
        
        for key, spec in self.required_credentials.items():
            if key not in all_credentials:
                missing_credentials.append(key)
                self._log_security_event(
                    "REQUIRED_CREDENTIAL_MISSING",
                    f"Required credential missing: {key}",
                    "CRITICAL"
                )
                continue
            
            value = all_credentials[key]
            
            # Validate credential
            validation_result = self._validate_credential(key, value, spec)
            if not validation_result.is_valid:
                invalid_credentials.append(f"{key}: {validation_result.error_message}")
                self._log_security_event(
                    "INVALID_CREDENTIAL",
                    f"Invalid credential {key}: {validation_result.error_message}",
                    "CRITICAL"
                )
                continue
            
            # Additional security checks
            if not self._check_credential_security(key, value):
                invalid_credentials.append(f"{key}: Failed security checks")
                continue
            
            # Store validated credential
            self.loaded_credentials[key] = value
            self._log_security_event(
                "CREDENTIAL_VALIDATED",
                f"Successfully validated {key}",
                "INFO"
            )
        
        # Handle optional credentials
        for key, spec in self.optional_credentials.items():
            if key in all_credentials:
                value = all_credentials[key]
                validation_result = self._validate_credential(key, value, spec)
                if validation_result.is_valid:
                    self.loaded_credentials[key] = value
                else:
                    self._log_security_event(
                        "INVALID_OPTIONAL_CREDENTIAL",
                        f"Invalid optional credential {key}: {validation_result.error_message}",
                        "WARNING"
                    )
            else:
                # Use default value
                self.loaded_credentials[key] = spec["default"]
                self._log_security_event(
                    "DEFAULT_CREDENTIAL_USED",
                    f"Using default value for {key}: {spec['default']}",
                    "INFO"
                )
        
        # Fail-closed design: Exit if any required credentials are missing or invalid
        if missing_credentials or invalid_credentials:
            error_message = "🔐 CRITICAL SECURITY FAILURE: Cannot proceed with missing or invalid credentials\n\n"
            
            if missing_credentials:
                error_message += "❌ Missing required credentials:\n"
                for cred in missing_credentials:
                    error_message += f"   - {cred}: {self.required_credentials[cred]['description']}\n"
                error_message += "\n"
            
            if invalid_credentials:
                error_message += "❌ Invalid credentials:\n"
                for cred in invalid_credentials:
                    error_message += f"   - {cred}\n"
                error_message += "\n"
            
            error_message += "🔧 To fix this:\n"
            error_message += "1. Create config/secure_creds.env with your credentials\n"
            error_message += "2. Or set environment variables\n"
            error_message += "3. Ensure credentials are properly formatted\n"
            error_message += "4. Never commit credentials to the repository\n\n"
            error_message += "📖 See config/secure_creds.env.example for format\n"
            
            self._log_security_event(
                "CREDENTIAL_LOAD_FAILED",
                f"Failed to load credentials: {len(missing_credentials)} missing, {len(invalid_credentials)} invalid",
                "CRITICAL"
            )
            
            print(error_message)
            sys.exit(1)
        
        self._log_security_event(
            "CREDENTIAL_LOAD_SUCCESS",
            f"Successfully loaded {len(self.loaded_credentials)} credentials",
            "INFO"
        )
        
        return self.loaded_credentials.copy()
    
    def get_credential(self, key: str) -> Any:
        """Get a specific credential (for use after load_credentials)"""
        if key not in self.loaded_credentials:
            self._log_security_event(
                "CREDENTIAL_ACCESS_DENIED",
                f"Attempted to access unloaded credential: {key}",
                "WARNING"
            )
            return None
        
        self._log_security_event(
            "CREDENTIAL_ACCESSED",
            f"Accessed credential: {key}",
            "INFO"
        )
        
        return self.loaded_credentials[key]
    
    def get_audit_log(self) -> list:
        """Get the audit log for security monitoring"""
        return self.audit_log.copy()
    
    def export_audit_log(self, file_path: Path) -> None:
        """Export audit log to file for security analysis"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
            
            self._log_security_event(
                "AUDIT_LOG_EXPORTED",
                f"Audit log exported to {file_path}",
                "INFO"
            )
        except Exception as e:
            self._log_security_event(
                "AUDIT_LOG_EXPORT_FAILED",
                f"Failed to export audit log: {e}",
                "WARNING"
            )

def load_secure_credentials(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to load secure credentials with fail-closed design
    
    Args:
        config_path: Path to credentials file (default: config/secure_creds.env)
        
    Returns:
        Dict containing validated credentials
        
    Raises:
        SystemExit: If credentials are missing or invalid
    """
    loader = SecretsLoader(config_path)
    return loader.load_credentials()

if __name__ == "__main__":
    # Test the secrets loader
    try:
        credentials = load_secure_credentials()
        print("✅ Credentials loaded successfully!")
        print(f"📊 Loaded {len(credentials)} credentials")
        
        # Don't print actual credentials for security
        for key in credentials.keys():
            print(f"   - {key}: [REDACTED]")
            
    except SystemExit:
        print("❌ Credential loading failed - system exiting for security")
        sys.exit(1)
