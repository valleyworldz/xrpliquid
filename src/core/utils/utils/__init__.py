"""
Core Utilities - Liquid Manus Trading System
==========================================

This package contains core utility modules for the Liquid Manus trading system.

Author: Liquid Manus Development Team
Version: 3.0.0
Last Updated: 2025-01-27
"""

from .credential_handler import SecureCredentialHandler
import os

__all__ = ['SecureCredentialHandler']

def load_env_from_file():
    """Load environment variables from secure_creds.env if present"""
    possible_paths = [
        os.path.join(os.getcwd(), 'secure_creds.env'),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../secure_creds.env'))
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('export '):
                        keyval = line[len('export '):].strip().split('=', 1)
                        if len(keyval) == 2:
                            key, val = keyval
                            os.environ[key] = val.strip("'\"")
