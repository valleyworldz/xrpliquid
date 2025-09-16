"""
Leak Canary Detector
Detects if fake secrets (canaries) appear in logs/artifacts.
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


class LeakCanaryDetector:
    """Detects leak canaries in logs and artifacts."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Leak canaries (fake secrets for testing)
        self.canaries = [
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "sk_test_51234567890abcdef",
            "AIzaSyBOti4mM-6x9WDnZIjIey21xX4pI8pYEXAMPLE",
            "ghp_1234567890abcdefghijklmnopqrstuvwxyz1234",
            "fake_api_key_12345",
            "test_secret_67890",
            "dummy_token_abcdef"
        ]
        
        # Directories to scan
        self.scan_directories = [
            "logs/",
            "reports/",
            "src/",
            "scripts/"
        ]
        
        # File extensions to scan
        self.scan_extensions = [".log", ".json", ".txt", ".py", ".md"]
    
    def scan_for_canaries(self) -> Dict:
        """Scan for leak canaries in the codebase."""
        logger.info("🔍 Scanning for leak canaries...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'scan_type': 'leak_canary_detection',
            'canaries_found': [],
            'files_scanned': 0,
            'total_canaries': 0,
            'scan_status': 'PASSED'
        }
        
        for directory in self.scan_directories:
            dir_path = self.repo_root / directory
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in self.scan_extensions:
                        results['files_scanned'] += 1
                        canaries_in_file = self.scan_file_for_canaries(file_path)
                        
                        if canaries_in_file:
                            results['canaries_found'].extend(canaries_in_file)
                            results['total_canaries'] += len(canaries_in_file)
                            results['scan_status'] = 'FAILED'
        
        return results
    
    def scan_file_for_canaries(self, file_path: Path) -> List[Dict]:
        """Scan a single file for leak canaries."""
        canaries_found = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for canary in self.canaries:
                if canary in content:
                    # Find line numbers where canary appears
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        if canary in line:
                            canaries_found.append({
                                'file': str(file_path.relative_to(self.repo_root)),
                                'line_number': line_num,
                                'canary': canary,
                                'line_content': line.strip(),
                                'severity': 'CRITICAL'
                            })
        
        except Exception as e:
            logger.warning(f"Could not scan {file_path}: {e}")
        
        return canaries_found
    
    def create_canary_test_file(self) -> Path:
        """Create a test file with canaries for testing detection."""
        logger.info("🧪 Creating canary test file...")
        
        test_dir = self.repo_root / "tests"
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / "leak_canaries.txt"
        
        with open(test_file, 'w') as f:
            f.write("# Leak Canary Test File\n")
            f.write("# This file contains fake secrets for testing leak detection\n\n")
            
            for canary in self.canaries:
                f.write(f"# Test canary: {canary}\n")
                f.write(f"fake_secret = \"{canary}\"\n\n")
        
        logger.info(f"✅ Canary test file created: {test_file}")
        return test_file
    
    def run_canary_detection_test(self) -> Dict:
        """Run a complete canary detection test."""
        logger.info("🚀 Running leak canary detection test...")
        
        # Create test file with canaries
        test_file = self.create_canary_test_file()
        
        # Scan for canaries
        scan_results = self.scan_for_canaries()
        
        # Check if test canaries were detected
        test_canaries_detected = 0
        for canary_found in scan_results['canaries_found']:
            if canary_found['file'] == str(test_file.relative_to(self.repo_root)):
                test_canaries_detected += 1
        
        # Generate test report
        test_report = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'leak_canary_detection_test',
            'scan_results': scan_results,
            'test_canaries_created': len(self.canaries),
            'test_canaries_detected': test_canaries_detected,
            'detection_rate': test_canaries_detected / len(self.canaries) if self.canaries else 0,
            'test_status': 'PASSED' if test_canaries_detected > 0 else 'FAILED'
        }
        
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
            logger.info("🧹 Test file cleaned up")
        
        return test_report
    
    def save_canary_detection_report(self, report: Dict) -> Path:
        """Save canary detection report."""
        security_dir = self.reports_dir / "security"
        security_dir.mkdir(exist_ok=True)
        
        report_path = security_dir / "leak_canary_detection.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Canary detection report saved: {report_path}")
        return report_path
    
    def check_canary_detection_ci(self) -> bool:
        """Check if canary detection should fail CI."""
        scan_results = self.scan_for_canaries()
        
        # Fail CI if any canaries are found (excluding test files)
        for canary_found in scan_results['canaries_found']:
            if 'test' not in canary_found['file'].lower():
                logger.error(f"❌ Leak canary detected in production code: {canary_found['file']}")
                return False
        
        logger.info("✅ No leak canaries detected in production code")
        return True


def main():
    """Main function to demonstrate leak canary detection."""
    detector = LeakCanaryDetector()
    
    # Run canary detection test
    test_report = detector.run_canary_detection_test()
    
    # Save report
    detector.save_canary_detection_report(test_report)
    
    # Check CI status
    ci_passed = detector.check_canary_detection_ci()
    
    print(f"Canary Detection Test: {test_report['test_status']}")
    print(f"Detection Rate: {test_report['detection_rate']:.1%}")
    print(f"CI Status: {'PASSED' if ci_passed else 'FAILED'}")
    
    print("✅ Leak canary detection demonstration completed")


if __name__ == "__main__":
    main()
