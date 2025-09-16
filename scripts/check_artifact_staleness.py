"""
Artifact Freshness Checker
Ensures reports are regenerated after code/config changes.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArtifactFreshnessChecker:
    """Checks if artifacts are fresh relative to code/config changes."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Artifact directories to check
        self.artifact_dirs = [
            "reports/tearsheets/",
            "reports/ledgers/",
            "reports/latency/",
            "reports/executive_dashboard.html"
        ]
        
        # Code/config directories that should trigger regeneration
        self.source_dirs = [
            "src/",
            "config/",
            "scripts/",
            "tests/"
        ]
    
    def get_latest_file_mtime(self, directory: Path) -> datetime:
        """Get the latest modification time of files in a directory."""
        if not directory.exists():
            return datetime.min
        
        latest_mtime = datetime.min
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime > latest_mtime:
                    latest_mtime = mtime
        
        return latest_mtime
    
    def check_artifact_freshness(self) -> dict:
        """Check if artifacts are fresh relative to source changes."""
        logger.info("🔍 Checking artifact freshness...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'artifact_freshness',
            'checks': {},
            'overall_status': 'PASSED',
            'stale_artifacts': []
        }
        
        # Get latest source modification time
        latest_source_mtime = datetime.min
        for source_dir in self.source_dirs:
            source_path = self.repo_root / source_dir
            if source_path.exists():
                dir_mtime = self.get_latest_file_mtime(source_path)
                if dir_mtime > latest_source_mtime:
                    latest_source_mtime = dir_mtime
        
        logger.info(f"📅 Latest source modification: {latest_source_mtime}")
        
        # Check each artifact directory
        for artifact_path in self.artifact_dirs:
            full_path = self.repo_root / artifact_path
            
            if full_path.is_file():
                # Single file
                if full_path.exists():
                    artifact_mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
                    is_fresh = artifact_mtime >= latest_source_mtime
                    
                    results['checks'][artifact_path] = {
                        'type': 'file',
                        'mtime': artifact_mtime.isoformat(),
                        'is_fresh': is_fresh,
                        'status': 'PASSED' if is_fresh else 'FAILED'
                    }
                    
                    if not is_fresh:
                        results['stale_artifacts'].append(artifact_path)
                        results['overall_status'] = 'FAILED'
                        logger.warning(f"⚠️ Stale artifact: {artifact_path}")
                    else:
                        logger.info(f"✅ Fresh artifact: {artifact_path}")
            
            elif full_path.is_dir():
                # Directory
                if full_path.exists():
                    artifact_mtime = self.get_latest_file_mtime(full_path)
                    is_fresh = artifact_mtime >= latest_source_mtime
                    
                    results['checks'][artifact_path] = {
                        'type': 'directory',
                        'mtime': artifact_mtime.isoformat(),
                        'is_fresh': is_fresh,
                        'status': 'PASSED' if is_fresh else 'FAILED'
                    }
                    
                    if not is_fresh:
                        results['stale_artifacts'].append(artifact_path)
                        results['overall_status'] = 'FAILED'
                        logger.warning(f"⚠️ Stale artifact directory: {artifact_path}")
                    else:
                        logger.info(f"✅ Fresh artifact directory: {artifact_path}")
                else:
                    results['checks'][artifact_path] = {
                        'type': 'directory',
                        'mtime': None,
                        'is_fresh': False,
                        'status': 'FAILED'
                    }
                    results['stale_artifacts'].append(artifact_path)
                    results['overall_status'] = 'FAILED'
                    logger.error(f"❌ Missing artifact directory: {artifact_path}")
        
        return results
    
    def check_critical_artifacts(self) -> dict:
        """Check if critical artifacts exist and are recent."""
        logger.info("🎯 Checking critical artifacts...")
        
        critical_artifacts = [
            "reports/final_system_status.json",
            "reports/executive_dashboard.html",
            "reports/tearsheets/comprehensive_tearsheet.html",
            "reports/latency/latency_analysis.json",
            "reports/hash_manifest.json"
        ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'critical_artifacts',
            'artifacts': {},
            'overall_status': 'PASSED',
            'missing_artifacts': []
        }
        
        for artifact_path in critical_artifacts:
            full_path = self.repo_root / artifact_path
            
            if full_path.exists():
                mtime = datetime.fromtimestamp(full_path.stat().st_mtime)
                age_hours = (datetime.now() - mtime).total_seconds() / 3600
                
                results['artifacts'][artifact_path] = {
                    'exists': True,
                    'mtime': mtime.isoformat(),
                    'age_hours': round(age_hours, 2),
                    'status': 'PASSED' if age_hours < 24 else 'WARNING'
                }
                
                if age_hours > 24:
                    logger.warning(f"⚠️ Old artifact: {artifact_path} ({age_hours:.1f} hours old)")
                else:
                    logger.info(f"✅ Recent artifact: {artifact_path} ({age_hours:.1f} hours old)")
            else:
                results['artifacts'][artifact_path] = {
                    'exists': False,
                    'mtime': None,
                    'age_hours': None,
                    'status': 'FAILED'
                }
                results['missing_artifacts'].append(artifact_path)
                results['overall_status'] = 'FAILED'
                logger.error(f"❌ Missing critical artifact: {artifact_path}")
        
        return results
    
    def run_freshness_check(self) -> dict:
        """Run complete artifact freshness check."""
        logger.info("🚀 Starting artifact freshness check...")
        
        # Check artifact freshness
        freshness_results = self.check_artifact_freshness()
        
        # Check critical artifacts
        critical_results = self.check_critical_artifacts()
        
        # Combine results
        combined_results = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'artifact_freshness_complete',
            'freshness_check': freshness_results,
            'critical_artifacts': critical_results,
            'overall_status': 'PASSED'
        }
        
        # Determine overall status
        if (freshness_results['overall_status'] == 'FAILED' or 
            critical_results['overall_status'] == 'FAILED'):
            combined_results['overall_status'] = 'FAILED'
        
        # Save results
        os.makedirs(self.reports_dir / "tests", exist_ok=True)
        results_path = self.reports_dir / "tests" / "artifact_freshness_check.json"
        with open(results_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        logger.info(f"📊 Artifact freshness check completed: {combined_results['overall_status']}")
        return combined_results


def main():
    """Main freshness check function."""
    checker = ArtifactFreshnessChecker()
    results = checker.run_freshness_check()
    
    if results['overall_status'] == 'FAILED':
        logger.error("❌ Artifact freshness check failed")
        logger.error(f"Stale artifacts: {results['freshness_check']['stale_artifacts']}")
        logger.error(f"Missing artifacts: {results['critical_artifacts']['missing_artifacts']}")
        exit(1)
    else:
        logger.info("✅ Artifact freshness check passed")
        exit(0)


if __name__ == "__main__":
    main()