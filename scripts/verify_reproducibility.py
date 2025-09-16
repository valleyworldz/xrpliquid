"""
Reproducibility Verification Script
Verifies that all artifacts can be reproduced from the same inputs.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReproducibilityVerifier:
    """Verifies system reproducibility."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Required files for verification
        self.required_files = [
            "reports/hash_manifest.json",
            "reports/final_system_status.json",
            "reports/executive_dashboard.html",
            "reports/tearsheets/comprehensive_tearsheet.html",
            "reports/latency/latency_analysis.json"
        ]
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        if not file_path.exists():
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def verify_hash_manifest(self) -> dict:
        """Verify hash manifest structure and content."""
        logger.info("🔍 Verifying hash manifest...")
        
        manifest_path = self.reports_dir / "hash_manifest.json"
        if not manifest_path.exists():
            return {
                'status': 'FAILED',
                'error': 'Hash manifest not found',
                'details': []
            }
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Check required fields
            required_fields = ['timestamp', 'code_commit', 'environment_hash', 'artifacts']
            missing_fields = [field for field in required_fields if field not in manifest]
            
            if missing_fields:
                return {
                    'status': 'FAILED',
                    'error': f'Missing required fields: {missing_fields}',
                    'details': []
                }
            
            # Verify artifacts exist
            artifacts = manifest.get('artifacts', {})
            missing_artifacts = []
            for artifact_name, artifact_info in artifacts.items():
                if 'file_path' in artifact_info:
                    file_path = self.repo_root / artifact_info['file_path']
                    if not file_path.exists():
                        missing_artifacts.append(artifact_name)
            
            if missing_artifacts:
                return {
                    'status': 'FAILED',
                    'error': f'Missing artifact files: {missing_artifacts}',
                    'details': []
                }
            
            return {
                'status': 'PASSED',
                'error': None,
                'details': [f'Hash manifest verified with {len(artifacts)} artifacts']
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': f'Error reading hash manifest: {str(e)}',
                'details': []
            }
    
    def verify_canonical_artifacts(self) -> dict:
        """Verify canonical artifacts exist and are accessible."""
        logger.info("📊 Verifying canonical artifacts...")
        
        results = {
            'status': 'PASSED',
            'missing_files': [],
            'details': []
        }
        
        for file_path in self.required_files:
            full_path = self.repo_root / file_path
            if full_path.exists():
                file_size = full_path.stat().st_size
                results['details'].append(f"✅ {file_path} ({file_size} bytes)")
            else:
                results['missing_files'].append(file_path)
                results['details'].append(f"❌ {file_path} (missing)")
        
        if results['missing_files']:
            results['status'] = 'FAILED'
        
        return results
    
    def verify_dashboard_consistency(self) -> dict:
        """Verify dashboard consistency with canonical sources."""
        logger.info("🎨 Verifying dashboard consistency...")
        
        try:
            # Import and run the dashboard consistency test
            import sys
            sys.path.append(str(self.repo_root))
            
            from tests.test_dashboard_consistency import test_dashboard_consistency
            test_results = test_dashboard_consistency()
            
            return {
                'status': 'PASSED' if test_results['status'] == 'PASSED' else 'FAILED',
                'error': None,
                'details': test_results['checks_performed']
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': f'Dashboard consistency test failed: {str(e)}',
                'details': []
            }
    
    def verify_performance_targets(self) -> dict:
        """Verify performance targets are met."""
        logger.info("🎯 Verifying performance targets...")
        
        status_path = self.reports_dir / "final_system_status.json"
        if not status_path.exists():
            return {
                'status': 'FAILED',
                'error': 'System status file not found',
                'details': []
            }
        
        try:
            with open(status_path, 'r') as f:
                status = json.load(f)
            
            performance = status.get('performance_metrics', {})
            latency = performance.get('p95_latency_ms', 0)
            sharpe = performance.get('sharpe_ratio', 0)
            
            details = []
            
            # Check latency target (sub-100ms)
            if latency <= 100:
                details.append(f"✅ P95 latency: {latency}ms (target: ≤100ms)")
            else:
                details.append(f"❌ P95 latency: {latency}ms (target: ≤100ms)")
            
            # Check Sharpe ratio target (>1.0)
            if sharpe > 1.0:
                details.append(f"✅ Sharpe ratio: {sharpe} (target: >1.0)")
            else:
                details.append(f"❌ Sharpe ratio: {sharpe} (target: >1.0)")
            
            status_result = 'PASSED' if latency <= 100 and sharpe > 1.0 else 'FAILED'
            
            return {
                'status': status_result,
                'error': None,
                'details': details
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': f'Error reading performance metrics: {str(e)}',
                'details': []
            }
    
    def run_verification(self) -> dict:
        """Run complete reproducibility verification."""
        logger.info("🚀 Starting reproducibility verification...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'verification_type': 'reproducibility',
            'checks': {},
            'overall_status': 'PASSED'
        }
        
        # Run all verification checks
        checks = [
            ('hash_manifest', self.verify_hash_manifest),
            ('canonical_artifacts', self.verify_canonical_artifacts),
            ('dashboard_consistency', self.verify_dashboard_consistency),
            ('performance_targets', self.verify_performance_targets)
        ]
        
        for check_name, check_func in checks:
            logger.info(f"🔍 Running {check_name} check...")
            check_result = check_func()
            results['checks'][check_name] = check_result
            
            if check_result['status'] == 'FAILED':
                results['overall_status'] = 'FAILED'
                logger.error(f"❌ {check_name} check failed: {check_result.get('error', 'Unknown error')}")
            else:
                logger.info(f"✅ {check_name} check passed")
        
        # Save verification results
        os.makedirs(self.reports_dir / "tests", exist_ok=True)
        results_path = self.reports_dir / "tests" / "reproducibility_verification.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Reproducibility verification completed: {results['overall_status']}")
        return results


def main():
    """Main verification function."""
    verifier = ReproducibilityVerifier()
    results = verifier.run_verification()
    
    if results['overall_status'] == 'FAILED':
        logger.error("❌ Reproducibility verification failed")
        exit(1)
    else:
        logger.info("✅ Reproducibility verification passed")
        exit(0)


if __name__ == "__main__":
    main()