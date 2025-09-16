"""
AuditPack Generator
Creates comprehensive audit package for third-party verification.
"""

import json
import os
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditPackGenerator:
    """Generates comprehensive audit package."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        self.audit_dir = self.reports_dir / "audit"
        
        # Create audit directory
        self.audit_dir.mkdir(exist_ok=True)
    
    def collect_artifacts(self) -> Dict:
        """Collect all artifacts for audit package."""
        logger.info("📦 Collecting artifacts for audit package...")
        
        artifacts = {
            'timestamp': datetime.now().isoformat(),
            'artifacts': {},
            'checksums': {},
            'metadata': {}
        }
        
        # Collect key artifacts
        artifact_paths = [
            'reports/hash_manifest.json',
            'reports/final_system_status.json',
            'reports/executive_dashboard.html',
            'reports/tearsheets/comprehensive_tearsheet.html',
            'reports/latency/latency_analysis.json',
            'reports/ledgers/trades.parquet',
            'reports/risk/var_es.json',
            'reports/reconciliation/exchange_vs_ledger.json',
            'sbom.json',
            'config/sizing_by_regime.json',
            'docs/ARCHITECTURE.md',
            'docs/RUNBOOK.md',
            'docs/SLOs.md',
            'docs/SECURITY.md',
            'docs/ONBOARDING.md',
            'CHANGELOG.md'
        ]
        
        for artifact_path in artifact_paths:
            full_path = self.repo_root / artifact_path
            if full_path.exists():
                # Calculate checksum
                checksum = self.calculate_file_checksum(full_path)
                
                artifacts['artifacts'][artifact_path] = {
                    'path': str(full_path),
                    'size': full_path.stat().st_size,
                    'checksum': checksum,
                    'modified': datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()
                }
                
                artifacts['checksums'][artifact_path] = checksum
        
        # Collect metadata
        artifacts['metadata'] = {
            'repo_url': 'https://github.com/valleyworldz/xrpliquid',
            'version': '1.0.0',
            'generated_by': 'AuditPack Generator',
            'generation_time': datetime.now().isoformat(),
            'total_artifacts': len(artifacts['artifacts']),
            'total_size': sum(artifacts['artifacts'][path]['size'] for path in artifacts['artifacts'])
        }
        
        return artifacts
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def create_audit_manifest(self, artifacts: Dict) -> Path:
        """Create audit manifest file."""
        logger.info("📋 Creating audit manifest...")
        
        manifest = {
            'audit_pack_info': {
                'version': '1.0.0',
                'generated_at': datetime.now().isoformat(),
                'generator': 'XRPLiquid AuditPack Generator',
                'purpose': 'Third-party verification and audit'
            },
            'artifacts': artifacts['artifacts'],
            'checksums': artifacts['checksums'],
            'metadata': artifacts['metadata'],
            'verification_instructions': {
                'step_1': 'Verify all checksums match',
                'step_2': 'Check artifact completeness',
                'step_3': 'Validate data integrity',
                'step_4': 'Review documentation',
                'step_5': 'Perform security audit'
            }
        }
        
        manifest_file = self.audit_dir / "audit_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✅ Audit manifest created: {manifest_file}")
        return manifest_file
    
    def create_verification_script(self) -> Path:
        """Create verification script for audit package."""
        logger.info("🔍 Creating verification script...")
        
        verification_script = '''#!/usr/bin/env python3
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
'''
        
        script_file = self.audit_dir / "verify_audit_pack.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(verification_script)
        
        # Make script executable
        os.chmod(script_file, 0o755)
        
        logger.info(f"✅ Verification script created: {script_file}")
        return script_file
    
    def create_audit_package(self, artifacts: Dict) -> Path:
        """Create the audit package ZIP file."""
        logger.info("📦 Creating audit package...")
        
        # Create manifest
        manifest_file = self.create_audit_manifest(artifacts)
        
        # Create verification script
        script_file = self.create_verification_script()
        
        # Create ZIP package
        package_name = f"xrpliquid_audit_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        package_path = self.audit_dir / package_name
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add manifest
            zipf.write(manifest_file, "audit_manifest.json")
            
            # Add verification script
            zipf.write(script_file, "verify_audit_pack.py")
            
            # Add all artifacts
            for artifact_path, artifact_info in artifacts['artifacts'].items():
                full_path = self.repo_root / artifact_path
                if full_path.exists():
                    zipf.write(full_path, artifact_path)
        
        logger.info(f"✅ Audit package created: {package_path}")
        return package_path
    
    def generate_audit_pack(self) -> Dict:
        """Generate complete audit package."""
        logger.info("🚀 Generating audit package...")
        
        # Collect artifacts
        artifacts = self.collect_artifacts()
        
        # Create audit package
        package_path = self.create_audit_package(artifacts)
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'package_path': str(package_path),
            'package_size': package_path.stat().st_size,
            'total_artifacts': len(artifacts['artifacts']),
            'total_size': sum(artifacts['artifacts'][path]['size'] for path in artifacts['artifacts']),
            'verification_script': 'verify_audit_pack.py',
            'manifest_file': 'audit_manifest.json'
        }
        
        # Save summary
        summary_file = self.audit_dir / "audit_pack_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Audit package generation completed: {package_path}")
        return summary


def main():
    """Main function to generate audit package."""
    generator = AuditPackGenerator()
    summary = generator.generate_audit_pack()
    
    print(f"Audit Package: {summary['package_path']}")
    print(f"Package Size: {summary['package_size']:,} bytes")
    print(f"Total Artifacts: {summary['total_artifacts']}")
    print(f"Verification Script: {summary['verification_script']}")
    
    print("✅ Audit package generation completed")


if __name__ == "__main__":
    main()
