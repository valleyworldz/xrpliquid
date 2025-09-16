#!/usr/bin/env python3
"""
🔒 SECRET DETECTION SCRIPT
==========================
Comprehensive secret detection for trading bot security.
"""

import re
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Secret patterns
SECRET_PATTERNS = {
    'api_key': r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
    'secret_key': r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
    'private_key': r'(?i)(private[_-]?key|privatekey)\s*[=:]\s*["\']?([a-zA-Z0-9_-]{40,})["\']?',
    'mnemonic': r'(?i)(mnemonic|seed[_-]?phrase)\s*[=:]\s*["\']?([a-zA-Z0-9\s]{50,})["\']?',
    'password': r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\']{8,})["\']?',
    'token': r'(?i)(token|bearer)\s*[=:]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
    'webhook': r'(?i)(webhook[_-]?url|webhook)\s*[=:]\s*["\']?(https?://[^\s"\']+)["\']?',
}

def scan_file(file_path: Path) -> List[Dict[str, Any]]:
    """Scan file for secrets"""
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            line_num = 0
            
            for line in content.split('\n'):
                line_num += 1
                
                for pattern_name, pattern in SECRET_PATTERNS.items():
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        findings.append({
                            'file': str(file_path),
                            'line': line_num,
                            'pattern': pattern_name,
                            'match': match.group(0),
                            'severity': 'HIGH'
                        })
    
    except Exception as e:
        print(f"Error scanning {file_path}: {e}")
    
    return findings

def main():
    """Main secret detection function"""
    findings = []
    
    # Scan Python files
    for py_file in Path('.').rglob('*.py'):
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
        findings.extend(scan_file(py_file))
    
    # Scan config files
    for config_file in Path('.').rglob('*.{json,yaml,yml,env,ini}'):
        if 'venv' in str(config_file) or '__pycache__' in str(config_file):
            continue
        findings.extend(scan_file(config_file))
    
    if findings:
        print("🚨 SECRET DETECTION FAILED!")
        print(f"Found {len(findings)} potential secrets:")
        
        for finding in findings:
            print(f"  {finding['file']}:{finding['line']} - {finding['pattern']}")
        
        sys.exit(1)
    else:
        print("✅ No secrets detected")

if __name__ == "__main__":
    main()
