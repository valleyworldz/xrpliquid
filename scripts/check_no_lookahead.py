"""
No-Lookahead Bias Checker
Validates that features don't use future data and train/test splits are properly isolated.
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoLookaheadChecker:
    """Checks for lookahead bias in features and data splits."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Patterns that indicate potential lookahead bias
        self.lookahead_patterns = [
            r'ts\s*[><=]+\s*decision_ts',
            r'future.*price',
            r'next.*candle',
            r'lookahead',
            r'peek.*ahead',
            r'\.shift\(-\d+\)',  # Negative shift (future data) - only negative shifts
            r'\.loc\[.*future.*\]',  # Explicit future references
        ]
        
        # Feature files to check
        self.feature_files = [
            "src/core/strategies/",
            "src/core/ml/",
            "src/core/analytics/",
            "src/core/backtesting/"
        ]
    
    def check_code_patterns(self) -> dict:
        """Check code for lookahead bias patterns."""
        logger.info("🔍 Checking code for lookahead bias patterns...")
        
        results = {
            'files_checked': 0,
            'violations': [],
            'total_violations': 0
        }
        
        # Check all Python files in feature directories
        for feature_dir in self.feature_files:
            feature_path = self.repo_root / feature_dir
            if feature_path.exists():
                for py_file in feature_path.rglob("*.py"):
                    results['files_checked'] += 1
                    violations = self._check_file_for_lookahead(py_file)
                    if violations:
                        results['violations'].extend(violations)
                        results['total_violations'] += len(violations)
        
        return results
    
    def _check_file_for_lookahead(self, file_path: Path) -> list:
        """Check a single file for lookahead bias patterns."""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for pattern in self.lookahead_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append({
                            'file': str(file_path.relative_to(self.repo_root)),
                            'line': line_num,
                            'pattern': pattern,
                            'content': line.strip()
                        })
        
        except Exception as e:
            logger.warning(f"Could not check {file_path}: {e}")
        
        return violations
    
    def check_data_lineage(self) -> dict:
        """Check data lineage and provenance."""
        logger.info("🔍 Checking data lineage and provenance...")
        
        results = {
            'provenance_exists': False,
            'splits_exist': False,
            'provenance_valid': False,
            'splits_valid': False,
            'issues': []
        }
        
        # Check provenance file
        provenance_path = self.repo_root / "data" / "warehouse" / "2025-09-16" / "_provenance.json"
        if provenance_path.exists():
            results['provenance_exists'] = True
            try:
                with open(provenance_path, 'r') as f:
                    provenance = json.load(f)
                
                required_fields = ['timestamp', 'provenance', 'feed_metadata', 'capture_metadata']
                if all(field in provenance for field in required_fields):
                    results['provenance_valid'] = True
                else:
                    results['issues'].append("Provenance file missing required fields")
            except Exception as e:
                results['issues'].append(f"Error reading provenance: {e}")
        else:
            results['issues'].append("Provenance file missing")
        
        # Check train/test splits
        splits_path = self.reports_dir / "splits" / "train_test_splits.json"
        if splits_path.exists():
            results['splits_exist'] = True
            try:
                with open(splits_path, 'r') as f:
                    splits = json.load(f)
                
                required_fields = ['splits', 'validation_metadata', 'total_period']
                if all(field in splits for field in required_fields):
                    # Check that splits have proper wall-clock timestamps
                    for split in splits.get('splits', []):
                        if not all(field in split for field in ['wall_clock_train_start', 'wall_clock_test_start']):
                            results['issues'].append("Split missing wall-clock timestamps")
                            break
                    else:
                        results['splits_valid'] = True
                else:
                    results['issues'].append("Splits file missing required fields")
            except Exception as e:
                results['issues'].append(f"Error reading splits: {e}")
        else:
            results['issues'].append("Train/test splits file missing")
        
        return results
    
    def check_feature_timestamps(self) -> dict:
        """Check that features respect decision timestamps."""
        logger.info("🔍 Checking feature timestamp validation...")
        
        results = {
            'timestamp_validation': False,
            'decision_ts_respected': False,
            'issues': []
        }
        
        # Look for timestamp validation in feature engineering
        feature_files = list((self.repo_root / "src").rglob("*.py"))
        
        timestamp_validation_found = False
        decision_ts_validation_found = False
        
        for file_path in feature_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for timestamp validation
                if re.search(r'timestamp.*validation|validate.*timestamp', content, re.IGNORECASE):
                    timestamp_validation_found = True
                
                # Check for decision timestamp respect
                if re.search(r'decision_ts|decision_timestamp', content, re.IGNORECASE):
                    decision_ts_validation_found = True
                
            except Exception as e:
                logger.warning(f"Could not check {file_path}: {e}")
        
        results['timestamp_validation'] = timestamp_validation_found
        results['decision_ts_respected'] = decision_ts_validation_found
        
        if not timestamp_validation_found:
            results['issues'].append("No timestamp validation found in feature engineering")
        if not decision_ts_validation_found:
            results['issues'].append("No decision timestamp validation found")
        
        return results
    
    def generate_validation_report(self) -> dict:
        """Generate comprehensive no-lookahead validation report."""
        logger.info("📊 Generating no-lookahead validation report...")
        
        code_results = self.check_code_patterns()
        lineage_results = self.check_data_lineage()
        timestamp_results = self.check_feature_timestamps()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'no_lookahead_bias',
            'code_analysis': code_results,
            'data_lineage': lineage_results,
            'timestamp_validation': timestamp_results,
            'overall_status': 'PASSED',
            'recommendations': []
        }
        
        # Determine overall status
        if code_results['total_violations'] > 0:
            report['overall_status'] = 'FAILED'
            report['recommendations'].append(f"Fix {code_results['total_violations']} lookahead bias violations")
        
        if lineage_results['issues']:
            report['overall_status'] = 'FAILED'
            report['recommendations'].extend(lineage_results['issues'])
        
        if timestamp_results['issues']:
            report['overall_status'] = 'FAILED'
            report['recommendations'].extend(timestamp_results['issues'])
        
        if not report['recommendations']:
            report['recommendations'].append("No lookahead bias detected - validation passed")
        
        return report
    
    def save_report(self, report: dict) -> Path:
        """Save validation report to file."""
        os.makedirs(self.reports_dir / "validation", exist_ok=True)
        report_path = self.reports_dir / "validation" / "no_lookahead_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 No-lookahead validation report saved: {report_path}")
        return report_path
    
    def run_validation(self) -> bool:
        """Run complete no-lookahead validation."""
        logger.info("🚀 Starting no-lookahead bias validation...")
        
        report = self.generate_validation_report()
        self.save_report(report)
        
        # Summary
        status = report['overall_status']
        violations = report['code_analysis']['total_violations']
        issues = len(report['recommendations']) - 1  # Subtract 1 for "no issues" message
        
        logger.info("📊 No-Lookahead Validation Summary:")
        logger.info(f"  Overall Status: {'✅' if status == 'PASSED' else '❌'} {status}")
        logger.info(f"  Code Violations: {violations}")
        logger.info(f"  Data Lineage Issues: {len(report['data_lineage']['issues'])}")
        logger.info(f"  Timestamp Issues: {len(report['timestamp_validation']['issues'])}")
        
        for rec in report['recommendations']:
            logger.info(f"    - {rec}")
        
        return status == 'PASSED'


def main():
    """Main no-lookahead validation function."""
    checker = NoLookaheadChecker()
    success = checker.run_validation()
    
    if not success:
        logger.error("❌ No-lookahead validation failed")
        exit(1)
    else:
        logger.info("✅ No-lookahead validation passed")


if __name__ == "__main__":
    main()
