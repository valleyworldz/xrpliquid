"""
Artifact Staleness Checker
Checks if critical artifacts are stale and need regeneration.
"""

import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArtifactStalenessChecker:
    """Checks for stale artifacts that need regeneration."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Critical artifacts that must be fresh
        self.critical_artifacts = [
            "tearsheets/comprehensive_tearsheet.html",
            "ledgers/trades.parquet",
            "ledgers/trades.csv", 
            "executive_dashboard.html",
            "latency/latency_analysis.json",
            "final_system_status.json"
        ]
        
        # Maximum age for artifacts (1 hour)
        self.max_age_hours = 1.0
    
    def get_file_age_hours(self, file_path: Path) -> float:
        """Get age of file in hours."""
        if not file_path.exists():
            return float('inf')  # Missing files are considered infinitely old
        
        mtime = file_path.stat().st_mtime
        age_seconds = time.time() - mtime
        return age_seconds / 3600.0
    
    def check_artifact_freshness(self) -> dict:
        """Check freshness of all critical artifacts."""
        logger.info("🔍 Checking artifact freshness...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'max_age_hours': self.max_age_hours,
            'artifacts': {},
            'stale_count': 0,
            'missing_count': 0,
            'all_fresh': True
        }
        
        for artifact_rel_path in self.critical_artifacts:
            artifact_path = self.reports_dir / artifact_rel_path
            
            if not artifact_path.exists():
                results['artifacts'][artifact_rel_path] = {
                    'status': 'missing',
                    'age_hours': None,
                    'fresh': False
                }
                results['missing_count'] += 1
                results['all_fresh'] = False
                logger.warning(f"❌ Missing artifact: {artifact_rel_path}")
                continue
            
            age_hours = self.get_file_age_hours(artifact_path)
            is_fresh = age_hours <= self.max_age_hours
            
            results['artifacts'][artifact_rel_path] = {
                'status': 'stale' if not is_fresh else 'fresh',
                'age_hours': round(age_hours, 2),
                'fresh': is_fresh,
                'last_modified': datetime.fromtimestamp(artifact_path.stat().st_mtime).isoformat()
            }
            
            if not is_fresh:
                results['stale_count'] += 1
                results['all_fresh'] = False
                logger.warning(f"⚠️  Stale artifact: {artifact_rel_path} (age: {age_hours:.1f}h)")
            else:
                logger.info(f"✅ Fresh artifact: {artifact_rel_path} (age: {age_hours:.1f}h)")
        
        return results
    
    def check_code_config_changes(self) -> dict:
        """Check if code or config files have changed recently."""
        logger.info("🔍 Checking for recent code/config changes...")
        
        # Files that should trigger artifact regeneration
        trigger_paths = [
            "src/",
            "config/",
            "scripts/",
            "requirements.txt",
            ".pre-commit-config.yaml"
        ]
        
        results = {
            'recent_changes': [],
            'latest_change_hours': 0,
            'has_recent_changes': False
        }
        
        current_time = time.time()
        max_change_age_hours = 2.0  # Consider changes within 2 hours as "recent"
        
        for trigger_path in trigger_paths:
            trigger_file = self.repo_root / trigger_path
            
            if trigger_file.is_file():
                # Single file
                mtime = trigger_file.stat().st_mtime
                age_hours = (current_time - mtime) / 3600.0
                
                if age_hours <= max_change_age_hours:
                    results['recent_changes'].append({
                        'path': trigger_path,
                        'age_hours': round(age_hours, 2),
                        'last_modified': datetime.fromtimestamp(mtime).isoformat()
                    })
                    results['latest_change_hours'] = max(results['latest_change_hours'], age_hours)
                    results['has_recent_changes'] = True
                    
            elif trigger_file.is_dir():
                # Directory - check all files recursively
                for file_path in trigger_file.rglob("*"):
                    if file_path.is_file():
                        mtime = file_path.stat().st_mtime
                        age_hours = (current_time - mtime) / 3600.0
                        
                        if age_hours <= max_change_age_hours:
                            rel_path = file_path.relative_to(self.repo_root)
                            results['recent_changes'].append({
                                'path': str(rel_path),
                                'age_hours': round(age_hours, 2),
                                'last_modified': datetime.fromtimestamp(mtime).isoformat()
                            })
                            results['latest_change_hours'] = max(results['latest_change_hours'], age_hours)
                            results['has_recent_changes'] = True
        
        if results['recent_changes']:
            logger.info(f"📝 Found {len(results['recent_changes'])} recent code/config changes")
            for change in results['recent_changes'][:5]:  # Show first 5
                logger.info(f"  - {change['path']} ({change['age_hours']:.1f}h ago)")
        else:
            logger.info("✅ No recent code/config changes detected")
        
        return results
    
    def generate_staleness_report(self) -> dict:
        """Generate comprehensive staleness report."""
        logger.info("📊 Generating staleness report...")
        
        artifact_results = self.check_artifact_freshness()
        change_results = self.check_code_config_changes()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'artifact_staleness',
            'artifact_freshness': artifact_results,
            'code_config_changes': change_results,
            'recommendations': []
        }
        
        # Generate recommendations
        if not artifact_results['all_fresh']:
            if artifact_results['missing_count'] > 0:
                report['recommendations'].append(
                    f"Regenerate {artifact_results['missing_count']} missing artifacts"
                )
            if artifact_results['stale_count'] > 0:
                report['recommendations'].append(
                    f"Regenerate {artifact_results['stale_count']} stale artifacts"
                )
        
        if change_results['has_recent_changes'] and not artifact_results['all_fresh']:
            report['recommendations'].append(
                "Code/config changes detected - artifacts should be regenerated"
            )
        
        if not report['recommendations']:
            report['recommendations'].append("All artifacts are fresh - no action needed")
        
        return report
    
    def save_report(self, report: dict) -> Path:
        """Save staleness report to file."""
        os.makedirs(self.reports_dir / "freshness", exist_ok=True)
        report_path = self.reports_dir / "freshness" / "staleness_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Staleness report saved: {report_path}")
        return report_path
    
    def run_check(self) -> bool:
        """Run complete staleness check."""
        logger.info("🚀 Starting artifact staleness check...")
        
        report = self.generate_staleness_report()
        self.save_report(report)
        
        # Summary
        artifact_fresh = report['artifact_freshness']['all_fresh']
        has_recent_changes = report['code_config_changes']['has_recent_changes']
        
        logger.info("📊 Staleness Check Summary:")
        logger.info(f"  Artifacts fresh: {'✅' if artifact_fresh else '❌'}")
        logger.info(f"  Recent changes: {'✅' if has_recent_changes else '❌'}")
        logger.info(f"  Recommendations: {len(report['recommendations'])}")
        
        for rec in report['recommendations']:
            logger.info(f"    - {rec}")
        
        # Return True if all artifacts are fresh
        return artifact_fresh


def main():
    """Main staleness check function."""
    checker = ArtifactStalenessChecker()
    success = checker.run_check()
    
    if not success:
        logger.error("❌ Staleness check failed - artifacts need regeneration")
        exit(1)
    else:
        logger.info("✅ All artifacts are fresh")


if __name__ == "__main__":
    main()