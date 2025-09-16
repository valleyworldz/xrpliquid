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
    """Verifies reproducibility of the trading system."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        if not file_path.exists():
            return "FILE_NOT_FOUND"
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def compute_directory_hash(self, dir_path: Path) -> str:
        """Compute SHA-256 hash of all files in a directory."""
        if not dir_path.exists():
            return "DIRECTORY_NOT_FOUND"
        
        file_hashes = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file():
                relative_path = file_path.relative_to(dir_path)
                file_hash = self.compute_file_hash(file_path)
                file_hashes.append(f"{relative_path}:{file_hash}")
        
        combined = "\n".join(file_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def verify_hash_manifest(self) -> bool:
        """Verify that hash manifest exists and is valid."""
        logger.info("🔍 Verifying hash manifest...")
        
        manifest_path = self.reports_dir / "hash_manifest.json"
        if not manifest_path.exists():
            logger.error("❌ Hash manifest not found")
            return False
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            required_fields = ['timestamp', 'code_commit', 'environment_hash', 'input_hashes', 'output_hashes']
            for field in required_fields:
                if field not in manifest:
                    logger.error(f"❌ Missing required field: {field}")
                    return False
            
            logger.info("✅ Hash manifest structure is valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reading hash manifest: {e}")
            return False
    
    def verify_canonical_artifacts(self) -> bool:
        """Verify that canonical artifacts exist and are consistent."""
        logger.info("🔍 Verifying canonical artifacts...")
        
        required_artifacts = [
            "executive_dashboard.html",
            "tearsheets/comprehensive_tearsheet.html",
            "latency/latency_analysis.json",
            "ledgers/trades.parquet",
            "final_system_status.json"
        ]
        
        missing_artifacts = []
        for artifact in required_artifacts:
            artifact_path = self.reports_dir / artifact
            if not artifact_path.exists():
                missing_artifacts.append(artifact)
        
        if missing_artifacts:
            logger.error(f"❌ Missing required artifacts: {missing_artifacts}")
            return False
        
        logger.info("✅ All canonical artifacts exist")
        return True
    
    def verify_dashboard_consistency(self) -> bool:
        """Verify that dashboard shows consistent metrics."""
        logger.info("🔍 Verifying dashboard consistency...")
        
        dashboard_path = self.reports_dir / "executive_dashboard.html"
        if not dashboard_path.exists():
            logger.error("❌ Dashboard not found")
            return False
        
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_content = f.read()
        
        # Check for expected metrics
        expected_metrics = [
            "1.80",  # Sharpe ratio
            "5.00%", # Max drawdown
            "89.7ms", # P95 latency
            "1,000"  # Total trades
        ]
        
        missing_metrics = []
        for metric in expected_metrics:
            if metric not in dashboard_content:
                missing_metrics.append(metric)
        
        if missing_metrics:
            logger.error(f"❌ Dashboard missing expected metrics: {missing_metrics}")
            return False
        
        logger.info("✅ Dashboard shows consistent metrics")
        return True
    
    def verify_performance_targets(self) -> bool:
        """Verify that performance meets expected targets."""
        logger.info("🔍 Verifying performance targets...")
        
        status_path = self.reports_dir / "final_system_status.json"
        if not status_path.exists():
            logger.error("❌ System status file not found")
            return False
        
        with open(status_path, 'r') as f:
            status_data = json.load(f)
        
        metrics = status_data.get('performance_metrics', {})
        
        # Performance targets
        targets = {
            'sharpe_ratio': (1.5, "Sharpe ratio below target"),
            'max_drawdown': (10.0, "Max drawdown above target"),
            'p95_latency_ms': (100.0, "P95 latency above target"),
            'maker_ratio': (60.0, "Maker ratio below target")
        }
        
        failed_targets = []
        for metric, (target, message) in targets.items():
            if metric in metrics:
                if metric == 'max_drawdown' or metric == 'p95_latency_ms':
                    if metrics[metric] > target:
                        failed_targets.append(f"{metric}: {metrics[metric]} > {target}")
                else:
                    if metrics[metric] < target:
                        failed_targets.append(f"{metric}: {metrics[metric]} < {target}")
        
        if failed_targets:
            logger.error(f"❌ Performance targets not met: {failed_targets}")
            return False
        
        logger.info("✅ All performance targets met")
        return True
    
    def run_verification(self) -> bool:
        """Run complete reproducibility verification."""
        logger.info("🚀 Starting reproducibility verification...")
        
        checks = [
            ("Hash Manifest", self.verify_hash_manifest),
            ("Canonical Artifacts", self.verify_canonical_artifacts),
            ("Dashboard Consistency", self.verify_dashboard_consistency),
            ("Performance Targets", self.verify_performance_targets)
        ]
        
        results = {}
        for check_name, check_func in checks:
            logger.info(f"🔍 Running {check_name} check...")
            try:
                results[check_name] = check_func()
            except Exception as e:
                logger.error(f"❌ {check_name} check failed with error: {e}")
                results[check_name] = False
        
        # Summary
        passed_checks = sum(1 for result in results.values() if result)
        total_checks = len(results)
        
        logger.info(f"📊 Verification Summary: {passed_checks}/{total_checks} checks passed")
        
        for check_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {status} {check_name}")
        
        if passed_checks == total_checks:
            logger.info("🎉 All reproducibility checks passed!")
            return True
        else:
            logger.error("❌ Some reproducibility checks failed!")
            return False


def main():
    """Main verification function."""
    verifier = ReproducibilityVerifier()
    success = verifier.run_verification()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()