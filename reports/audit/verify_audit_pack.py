#!/usr/bin/env python3
"""
AuditPack Verification Script
Verifies the integrity and completeness of the audit package.
"""

import json
import hashlib
import os
from pathlib import Path

def verify_checksums(manifest_file):
    """Verify all file checksums."""
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    checksums = manifest['checksums']
    artifacts = manifest['artifacts']
    
    print("🔍 Verifying checksums...")
    
    for artifact_path, expected_checksum in checksums.items():
        if os.path.exists(artifact_path):
            # Calculate actual checksum
            sha256_hash = hashlib.sha256()
            with open(artifact_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            actual_checksum = sha256_hash.hexdigest()
            
            if actual_checksum == expected_checksum:
                print(f"✅ {artifact_path}: Checksum verified")
            else:
                print(f"❌ {artifact_path}: Checksum mismatch")
                return False
        else:
            print(f"❌ {artifact_path}: File not found")
            return False
    
    print("✅ All checksums verified successfully")
    return True

def verify_completeness(manifest_file):
    """Verify all required artifacts are present."""
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    required_artifacts = [
        'reports/hash_manifest.json',
        'reports/final_system_status.json',
        'reports/executive_dashboard.html',
        'reports/tearsheets/comprehensive_tearsheet.html',
        'sbom.json',
        'docs/ARCHITECTURE.md',
        'docs/RUNBOOK.md',
        'docs/SECURITY.md'
    ]
    
    print("🔍 Verifying completeness...")
    
    for artifact in required_artifacts:
        if artifact in manifest['artifacts']:
            print(f"✅ {artifact}: Present")
        else:
            print(f"❌ {artifact}: Missing")
            return False
    
    print("✅ All required artifacts present")
    return True

def main():
    """Main verification function."""
    print("🚀 Starting audit package verification...")
    
    manifest_file = "audit_manifest.json"
    
    if not os.path.exists(manifest_file):
        print(f"❌ Manifest file not found: {manifest_file}")
        return False
    
    # Verify checksums
    if not verify_checksums(manifest_file):
        print("❌ Checksum verification failed")
        return False
    
    # Verify completeness
    if not verify_completeness(manifest_file):
        print("❌ Completeness verification failed")
        return False
    
    print("✅ Audit package verification completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
