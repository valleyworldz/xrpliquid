import os

# Try to import dotenv, but handle gracefully if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(dotenv_path=None):
        """Dummy function when dotenv is not available"""
        pass

from core.utils.secure_credential_handler import SecureCredentialHandler

class CredentialManager:
    def __init__(self):
        # Load environment variables from .env file in the configs directory
        # This will load HYPERLIQUID_API_KEY and HYPERLIQUID_SECRET_KEY if they exist
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../configs/secrets.env'))
        
        # Also try to load from secure_creds.env if it exists (created by setup_credentials.py)
        secure_creds_path = os.path.join(os.path.dirname(__file__), '../../secure_creds.env')
        if os.path.exists(secure_creds_path):
            load_dotenv(dotenv_path=secure_creds_path)
            print("✅ Credentials loaded from environment file")

        self.secure_handler = SecureCredentialHandler()
        self.decrypted_creds = None
        
        # Attempt to load and decrypt credentials immediately
        try:
            self.decrypted_creds = self.secure_handler.load_credentials_securely()
        except Exception as e:
            print(f"Info: Secure credential loading not available: {e}")

    def get_credential(self, key):
        """
        Retrieves a credential (e.g., API key, secret key) by its key.

        Prioritizes decrypted credentials, then direct environment variables, then .env file.

        Args:
            key (str): The name of the credential.

        Returns:
            str: The value of the credential, or None if not found.
        """
        # First try decrypted credentials
        if self.decrypted_creds:
            if key == "HYPERLIQUID_PRIVATE_KEY":
                return self.decrypted_creds.get("private_key")
            elif key == "HYPERLIQUID_API_KEY":
                return self.decrypted_creds.get("address")

        # Check for direct environment variables first
        credential = os.getenv(key)
        if credential:
            return credential
        
        # Check for alternative key names
        if key == "HYPERLIQUID_PRIVATE_KEY":
            # Check alternative environment variable names
            alt_keys = ["ENCRYPTED_PRIVATE_KEY", "PRIVATE_KEY", "HL_PRIVATE_KEY"]
            for alt_key in alt_keys:
                credential = os.getenv(alt_key)
                if credential:
                    # Ensure proper format (remove 0x if present)
                    if credential.startswith('0x'):
                        credential = credential[2:]
                    return credential
        
        elif key == "HYPERLIQUID_API_KEY":
            # Check alternative environment variable names
            alt_keys = ["ENCRYPTED_ADDRESS", "WALLET_ADDRESS", "HL_ADDRESS"]
            for alt_key in alt_keys:
                credential = os.getenv(alt_key)
                if credential:
                    return credential
        
        # If nothing found, log warning
        if credential is None:
            print(f"Warning: Credential for key '{key}' not found in environment variables.")
        
        return credential

    def check_credentials_valid(self, required_keys):
        """
        Checks if all required credentials are present and not empty.

        Args:
            required_keys (list): A list of credential keys that must be present.

        Returns:
            bool: True if all required credentials are valid, False otherwise.
        """
        for key in required_keys:
            if not self.get_credential(key):
                print(f"Error: Missing or empty required credential: {key}")
                return False
        return True


