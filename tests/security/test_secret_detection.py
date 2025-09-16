"""
🔒 SECRET DETECTION TESTS
=========================
Tests to ensure no secrets are exposed in logs or code.
"""

import pytest
import re
import os
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.security.secret_detection import SECRET_PATTERNS, scan_file

class TestSecretDetection:
    """Test secret detection functionality"""
    
    def test_api_key_detection(self):
        """Test API key detection"""
        test_content = """
        API_KEY = "sk-1234567890abcdef1234567890abcdef"
        api_key = "ak_test_1234567890abcdef"
        """
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) >= 2
        assert any(f['pattern'] == 'api_key' for f in findings)
    
    def test_secret_key_detection(self):
        """Test secret key detection"""
        test_content = """
        SECRET_KEY = "secret_1234567890abcdef1234567890abcdef"
        secret_key = "sk_test_1234567890abcdef"
        """
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) >= 2
        assert any(f['pattern'] == 'secret_key' for f in findings)
    
    def test_mnemonic_detection(self):
        """Test mnemonic detection"""
        test_content = """
        MNEMONIC = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        seed_phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        """
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) >= 2
        assert any(f['pattern'] == 'mnemonic' for f in findings)
    
    def test_password_detection(self):
        """Test password detection"""
        test_content = """
        PASSWORD = "mypassword123"
        passwd = "secretpass"
        """
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) >= 2
        assert any(f['pattern'] == 'password' for f in findings)
    
    def test_token_detection(self):
        """Test token detection"""
        test_content = """
        TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        bearer_token = "bearer_1234567890abcdef"
        """
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) >= 2
        assert any(f['pattern'] == 'token' for f in findings)
    
    def test_webhook_detection(self):
        """Test webhook URL detection"""
        test_content = """
        WEBHOOK_URL = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        webhook = "https://discord.com/api/webhooks/123456789/abcdefghijklmnopqrstuvwxyz"
        """
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) >= 2
        assert any(f['pattern'] == 'webhook' for f in findings)
    
    def test_no_secrets_detected(self):
        """Test that no secrets are detected in clean code"""
        test_content = """
        # This is a comment
        def function_name():
            variable = "normal_string"
            number = 12345
            return variable
        """
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) == 0
    
    def test_secret_patterns_completeness(self):
        """Test that all secret patterns are defined"""
        required_patterns = [
            'api_key', 'secret_key', 'private_key', 
            'mnemonic', 'password', 'token', 'webhook'
        ]
        
        for pattern in required_patterns:
            assert pattern in SECRET_PATTERNS
            assert isinstance(SECRET_PATTERNS[pattern], str)
            assert len(SECRET_PATTERNS[pattern]) > 0
    
    def test_pattern_regex_validity(self):
        """Test that all patterns are valid regex"""
        for pattern_name, pattern in SECRET_PATTERNS.items():
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex for {pattern_name}: {e}")
    
    def test_finding_structure(self):
        """Test that findings have correct structure"""
        test_content = 'API_KEY = "test_key_1234567890abcdef"'
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) > 0
        
        finding = findings[0]
        required_fields = ['file', 'line', 'pattern', 'match', 'severity']
        
        for field in required_fields:
            assert field in finding
        
        assert finding['severity'] == 'HIGH'
        assert finding['pattern'] == 'api_key'
        assert finding['match'] == 'API_KEY = "test_key_1234567890abcdef"'
    
    def test_file_error_handling(self):
        """Test error handling for file reading"""
        with patch("builtins.open", side_effect=IOError("File not found")):
            findings = scan_file(Path("nonexistent.py"))
            
        assert len(findings) == 0
    
    def test_encoding_error_handling(self):
        """Test error handling for encoding issues"""
        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")):
            findings = scan_file(Path("test.py"))
            
        assert len(findings) == 0

class TestLogRedaction:
    """Test log redaction functionality"""
    
    def test_api_key_redaction(self):
        """Test that API keys are redacted in logs"""
        from src.core.utils.logger import Logger
        
        logger = Logger()
        
        # Mock log output
        with patch('builtins.print') as mock_print:
            logger.info("API key: sk-1234567890abcdef")
            
            # Check that API key was redacted
            mock_print.assert_called()
            log_output = str(mock_print.call_args)
            assert "sk-1234567890abcdef" not in log_output
            assert "***REDACTED***" in log_output or "***" in log_output
    
    def test_secret_redaction(self):
        """Test that secrets are redacted in logs"""
        from src.core.utils.logger import Logger
        
        logger = Logger()
        
        # Mock log output
        with patch('builtins.print') as mock_print:
            logger.info("Secret: secret_1234567890abcdef")
            
            # Check that secret was redacted
            mock_print.assert_called()
            log_output = str(mock_print.call_args)
            assert "secret_1234567890abcdef" not in log_output
            assert "***REDACTED***" in log_output or "***" in log_output
    
    def test_mnemonic_redaction(self):
        """Test that mnemonics are redacted in logs"""
        from src.core.utils.logger import Logger
        
        logger = Logger()
        
        # Mock log output
        with patch('builtins.print') as mock_print:
            logger.info("Mnemonic: abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about")
            
            # Check that mnemonic was redacted
            mock_print.assert_called()
            log_output = str(mock_print.call_args)
            assert "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" not in log_output
            assert "***REDACTED***" in log_output or "***" in log_output

if __name__ == "__main__":
    pytest.main([__file__])
