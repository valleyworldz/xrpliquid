"""
No-Lookahead Bias Checker
Detects potential lookahead bias in code and data.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoLookaheadChecker:
    """Checks for lookahead bias in code and data."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Patterns that indicate potential lookahead bias
        self.lookahead_patterns = [
            r'ts\s*[><=]+\s*decision_ts',
            r'future.*price',
            r'next.*candle',
            r'peek.*ahead',
            r'\.shift\(-\d+\)',  # Negative shift (future data)
            r'\.loc\[.*future.*\]',  # Explicit future references
            r'\.iloc\[.*:\s*-\d+\]',  # Negative indexing for future
        ]
        
        # Directories to check
        self.check_directories = [
            "src/",
            "scripts/",
            "tests/"
        ]
        
        # File extensions to check
        self.check_extensions = [".py", ".ipynb"]
    
    def check_file_for_lookahead(self, file_path: Path) -> list:
        """Check a single file for lookahead bias patterns."""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern in self.lookahead_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip comments and docstrings
                        stripped_line = line.strip()
                        if not (stripped_line.startswith('#') or 
                               stripped_line.startswith('"""') or
                               stripped_line.startswith("'''")):
                            violations.append({
                                'line_number': line_num,
                                'line_content': line.strip(),
                                'pattern': pattern,
                                'severity': 'WARNING'
                            })
        
        except Exception as e:
            logger.warning(f"Could not check {file_path}: {e}")
        
        return violations
    
    def check_codebase_for_lookahead(self) -> dict:
        """Check entire codebase for lookahead bias."""
        logger.info("🔍 Checking codebase for lookahead bias...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'no_lookahead_bias',
            'files_checked': 0,
            'violations_found': [],
            'overall_status': 'PASSED'
        }
        
        for directory in self.check_directories:
            dir_path = self.repo_root / directory
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in self.check_extensions:
                        results['files_checked'] += 1
                        violations = self.check_file_for_lookahead(file_path)
                        
                        if violations:
                            results['violations_found'].extend([{
                                'file': str(file_path.relative_to(self.repo_root)),
                                'violations': violations
                            }])
                            results['overall_status'] = 'FAILED'
        
        return results
    
    def check_data_splits(self) -> dict:
        """Check train/test splits for lookahead bias."""
        logger.info("📊 Checking data splits for lookahead bias...")
        
        splits_file = self.reports_dir / "splits" / "train_test_splits.json"
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'data_splits_lookahead',
            'splits_file_exists': splits_file.exists(),
            'validation_rules': {},
            'overall_status': 'PASSED'
        }
        
        if splits_file.exists():
            try:
                with open(splits_file, 'r') as f:
                    splits_data = json.load(f)
                
                validation_rules = splits_data.get('validation_rules', {})
                results['validation_rules'] = validation_rules
                
                # Check for no-lookahead rule
                if not validation_rules.get('no_lookahead', False):
                    results['overall_status'] = 'FAILED'
                    logger.error("❌ No-lookahead rule not enforced in splits")
                else:
                    logger.info("✅ No-lookahead rule enforced in splits")
                
                # Check split timing
                splits = splits_data.get('splits', [])
                for split in splits:
                    train_end = split.get('train_end')
                    test_start = split.get('test_start')
                    
                    if train_end and test_start:
                        if train_end >= test_start:
                            results['overall_status'] = 'FAILED'
                            logger.error(f"❌ Lookahead bias detected: train_end >= test_start in split {split.get('split_id')}")
                        else:
                            logger.info(f"✅ No lookahead bias in split {split.get('split_id')}")
            
            except Exception as e:
                results['overall_status'] = 'FAILED'
                logger.error(f"❌ Error reading splits file: {e}")
        else:
            results['overall_status'] = 'FAILED'
            logger.error("❌ Train/test splits file not found")
        
        return results
    
    def check_provenance_data(self) -> dict:
        """Check data provenance for lookahead bias indicators."""
        logger.info("📋 Checking data provenance...")
        
        provenance_dir = self.repo_root / "data" / "warehouse"
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'provenance_lookahead',
            'provenance_files': [],
            'overall_status': 'PASSED'
        }
        
        if provenance_dir.exists():
            for provenance_file in provenance_dir.rglob("_provenance.json"):
                results['provenance_files'].append(str(provenance_file.relative_to(self.repo_root)))
                
                try:
                    with open(provenance_file, 'r') as f:
                        provenance_data = json.load(f)
                    
                    # Check for future data references
                    data_provenance = provenance_data.get('data_provenance', {})
                    if 'future_data' in str(data_provenance).lower():
                        results['overall_status'] = 'FAILED'
                        logger.error(f"❌ Future data reference in {provenance_file}")
                    else:
                        logger.info(f"✅ No future data references in {provenance_file}")
                
                except Exception as e:
                    logger.warning(f"Could not read {provenance_file}: {e}")
        else:
            results['overall_status'] = 'FAILED'
            logger.error("❌ Data provenance directory not found")
        
        return results
    
    def run_no_lookahead_check(self) -> dict:
        """Run complete no-lookahead bias check."""
        logger.info("🚀 Starting no-lookahead bias check...")
        
        # Check codebase
        codebase_results = self.check_codebase_for_lookahead()
        
        # Check data splits
        splits_results = self.check_data_splits()
        
        # Check provenance
        provenance_results = self.check_provenance_data()
        
        # Combine results
        combined_results = {
            'timestamp': datetime.now().isoformat(),
            'check_type': 'no_lookahead_complete',
            'codebase_check': codebase_results,
            'splits_check': splits_results,
            'provenance_check': provenance_results,
            'overall_status': 'PASSED'
        }
        
        # Determine overall status
        if (codebase_results['overall_status'] == 'FAILED' or 
            splits_results['overall_status'] == 'FAILED' or
            provenance_results['overall_status'] == 'FAILED'):
            combined_results['overall_status'] = 'FAILED'
        
        # Save results
        os.makedirs(self.reports_dir / "tests", exist_ok=True)
        results_path = self.reports_dir / "tests" / "no_lookahead_check.json"
        with open(results_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        logger.info(f"📊 No-lookahead bias check completed: {combined_results['overall_status']}")
        return combined_results


def main():
    """Main no-lookahead check function."""
    checker = NoLookaheadChecker()
    results = checker.run_no_lookahead_check()
    
    if results['overall_status'] == 'FAILED':
        logger.error("❌ No-lookahead bias check failed")
        exit(1)
    else:
        logger.info("✅ No-lookahead bias check passed")
        exit(0)


if __name__ == "__main__":
    main()