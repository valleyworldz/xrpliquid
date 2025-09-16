"""
Lookahead Guard - Prevents Data Leakage in Features
Implements CI tests to ensure no features use future data.
"""

import ast
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class LookaheadViolation:
    """Represents a lookahead violation."""
    file_path: str
    line_number: int
    violation_type: str
    description: str
    severity: str


class LookaheadGuard:
    """Guards against lookahead bias in feature engineering."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.validation_dir = self.reports_dir / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Lookahead patterns to detect
        self.lookahead_patterns = {
            "future_data_access": [
                "df.shift(-1)", "df.shift(-2)", "df.shift(-3)",
                "df.rolling(window=5).shift(-1)",
                "df.expanding().shift(-1)"
            ],
            "future_calculations": [
                "df.pct_change().shift(-1)",
                "df.diff().shift(-1)",
                "df.cumsum().shift(-1)"
            ],
            "future_indicators": [
                "ta.RSI(period=14).shift(-1)",
                "ta.MACD().shift(-1)",
                "ta.BBANDS().shift(-1)"
            ],
            "future_aggregations": [
                "df.groupby().transform().shift(-1)",
                "df.resample().shift(-1)",
                "df.rolling().mean().shift(-1)"
            ]
        }
        
        # Time-based lookahead patterns
        self.time_lookahead_patterns = [
            "timestamp > decision_timestamp",
            "ts > decision_ts",
            "time > decision_time",
            "date > decision_date"
        ]
    
    def scan_codebase_for_lookahead(self, codebase_path: str = ".") -> List[LookaheadViolation]:
        """Scan codebase for lookahead violations."""
        
        violations = []
        codebase_path = Path(codebase_path)
        
        # Scan Python files
        for py_file in codebase_path.rglob("*.py"):
            if self._should_scan_file(py_file):
                file_violations = self._scan_file_for_lookahead(py_file)
                violations.extend(file_violations)
        
        return violations
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Determine if file should be scanned."""
        
        # Skip certain directories
        skip_dirs = {
            "__pycache__", ".git", ".pytest_cache", "node_modules",
            "venv", "env", ".venv", ".env"
        }
        
        for part in file_path.parts:
            if part in skip_dirs:
                return False
        
        # Skip certain file patterns
        skip_patterns = {
            "test_", "_test.py", "conftest.py", "setup.py", "requirements"
        }
        
        for pattern in skip_patterns:
            if pattern in file_path.name:
                return False
        
        return True
    
    def _scan_file_for_lookahead(self, file_path: Path) -> List[LookaheadViolation]:
        """Scan a single file for lookahead violations."""
        
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Scan for lookahead patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    violation = self._check_call_for_lookahead(node, file_path, content)
                    if violation:
                        violations.append(violation)
                
                elif isinstance(node, ast.Compare):
                    violation = self._check_compare_for_lookahead(node, file_path, content)
                    if violation:
                        violations.append(violation)
                
                elif isinstance(node, ast.Assign):
                    violation = self._check_assign_for_lookahead(node, file_path, content)
                    if violation:
                        violations.append(violation)
        
        except Exception as e:
            # If parsing fails, do basic string search
            violations.extend(self._basic_string_search(file_path, content))
        
        return violations
    
    def _check_call_for_lookahead(self, node: ast.Call, file_path: Path, content: str) -> Optional[LookaheadViolation]:
        """Check function call for lookahead patterns."""
        
        # Get line number
        line_number = node.lineno
        
        # Get the line content
        lines = content.split('\n')
        if line_number <= len(lines):
            line_content = lines[line_number - 1]
        else:
            return None
        
        # Check for shift(-1) patterns
        if "shift(-1)" in line_content or "shift(-2)" in line_content or "shift(-3)" in line_content:
            return LookaheadViolation(
                file_path=str(file_path),
                line_number=line_number,
                violation_type="future_data_access",
                description=f"Negative shift detected: {line_content.strip()}",
                severity="high"
            )
        
        # Check for future calculations
        if any(pattern in line_content for pattern in ["pct_change().shift(-1)", "diff().shift(-1)"]):
            return LookaheadViolation(
                file_path=str(file_path),
                line_number=line_number,
                violation_type="future_calculations",
                description=f"Future calculation detected: {line_content.strip()}",
                severity="high"
            )
        
        return None
    
    def _check_compare_for_lookahead(self, node: ast.Compare, file_path: Path, content: str) -> Optional[LookaheadViolation]:
        """Check comparison for time-based lookahead."""
        
        line_number = node.lineno
        lines = content.split('\n')
        if line_number <= len(lines):
            line_content = lines[line_number - 1]
        else:
            return None
        
        # Check for time-based lookahead patterns
        for pattern in self.time_lookahead_patterns:
            if pattern in line_content:
                return LookaheadViolation(
                    file_path=str(file_path),
                    line_number=line_number,
                    violation_type="time_lookahead",
                    description=f"Time-based lookahead detected: {line_content.strip()}",
                    severity="critical"
                )
        
        return None
    
    def _check_assign_for_lookahead(self, node: ast.Assign, file_path: Path, content: str) -> Optional[LookaheadViolation]:
        """Check assignment for lookahead patterns."""
        
        line_number = node.lineno
        lines = content.split('\n')
        if line_number <= len(lines):
            line_content = lines[line_number - 1]
        else:
            return None
        
        # Check for future data assignments
        if any(pattern in line_content for pattern in ["shift(-1)", "shift(-2)", "shift(-3)"]):
            return LookaheadViolation(
                file_path=str(file_path),
                line_number=line_number,
                violation_type="future_data_assignment",
                description=f"Future data assignment detected: {line_content.strip()}",
                severity="high"
            )
        
        return None
    
    def _basic_string_search(self, file_path: Path, content: str) -> List[LookaheadViolation]:
        """Basic string search for lookahead patterns."""
        
        violations = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check for negative shifts
            if "shift(-1)" in line or "shift(-2)" in line or "shift(-3)" in line:
                violations.append(LookaheadViolation(
                    file_path=str(file_path),
                    line_number=line_num,
                    violation_type="future_data_access",
                    description=f"Negative shift detected: {line.strip()}",
                    severity="high"
                ))
            
            # Check for time-based lookahead
            for pattern in self.time_lookahead_patterns:
                if pattern in line:
                    violations.append(LookaheadViolation(
                        file_path=str(file_path),
                        line_number=line_num,
                        violation_type="time_lookahead",
                        description=f"Time-based lookahead detected: {line.strip()}",
                        severity="critical"
                    ))
        
        return violations
    
    def validate_feature_engineering(self, feature_file: str) -> Dict[str, Any]:
        """Validate feature engineering file for lookahead."""
        
        feature_path = Path(feature_file)
        if not feature_path.exists():
            return {
                "status": "error",
                "message": f"Feature file not found: {feature_file}"
            }
        
        # Scan file for violations
        violations = self._scan_file_for_lookahead(feature_path)
        
        # Categorize violations
        violation_summary = {
            "total_violations": len(violations),
            "critical_violations": len([v for v in violations if v.severity == "critical"]),
            "high_violations": len([v for v in violations if v.severity == "high"]),
            "medium_violations": len([v for v in violations if v.severity == "medium"]),
            "low_violations": len([v for v in violations if v.severity == "low"])
        }
        
        # Determine overall status
        if violation_summary["critical_violations"] > 0:
            status = "failed"
        elif violation_summary["high_violations"] > 0:
            status = "warning"
        else:
            status = "passed"
        
        return {
            "status": status,
            "feature_file": str(feature_path),
            "violation_summary": violation_summary,
            "violations": [
                {
                    "line_number": v.line_number,
                    "violation_type": v.violation_type,
                    "description": v.description,
                    "severity": v.severity
                }
                for v in violations
            ],
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def run_ci_lookahead_test(self, codebase_path: str = ".") -> Dict[str, Any]:
        """Run CI lookahead test."""
        
        # Scan entire codebase
        violations = self.scan_codebase_for_lookahead(codebase_path)
        
        # Generate report
        report = {
            "test_name": "lookahead_guard",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "codebase_path": codebase_path,
            "total_files_scanned": len(list(Path(codebase_path).rglob("*.py"))),
            "total_violations": len(violations),
            "violation_summary": {
                "critical": len([v for v in violations if v.severity == "critical"]),
                "high": len([v for v in violations if v.severity == "high"]),
                "medium": len([v for v in violations if v.severity == "medium"]),
                "low": len([v for v in violations if v.severity == "low"])
            },
            "violations_by_file": {},
            "test_result": "passed"
        }
        
        # Group violations by file
        for violation in violations:
            file_path = violation.file_path
            if file_path not in report["violations_by_file"]:
                report["violations_by_file"][file_path] = []
            
            report["violations_by_file"][file_path].append({
                "line_number": violation.line_number,
                "violation_type": violation.violation_type,
                "description": violation.description,
                "severity": violation.severity
            })
        
        # Determine test result
        if report["violation_summary"]["critical"] > 0:
            report["test_result"] = "failed"
        elif report["violation_summary"]["high"] > 0:
            report["test_result"] = "warning"
        
        # Save report
        report_file = self.validation_dir / f"lookahead_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def create_lookahead_test_data(self) -> pd.DataFrame:
        """Create test data for lookahead validation."""
        
        # Create sample data with timestamps
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'timestamp': dates,
            'price': [100 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)],
            'decision_timestamp': dates  # This should be used for lookahead checks
        })
        
        return data


def main():
    """Test lookahead guard functionality."""
    guard = LookaheadGuard()
    
    # Test feature validation
    test_feature_file = "src/core/features/test_features.py"
    
    # Create a test feature file with lookahead violations
    test_content = '''
import pandas as pd

def create_features(df):
    # This is a lookahead violation
    df['future_price'] = df['price'].shift(-1)
    
    # This is also a lookahead violation
    df['future_return'] = df['price'].pct_change().shift(-1)
    
    # This is a time-based lookahead violation
    df['future_signal'] = df['timestamp'] > df['decision_timestamp']
    
    return df
'''
    
    # Write test file
    Path(test_feature_file).parent.mkdir(parents=True, exist_ok=True)
    with open(test_feature_file, 'w') as f:
        f.write(test_content)
    
    # Validate feature file
    validation_result = guard.validate_feature_engineering(test_feature_file)
    print(f"✅ Feature validation: {validation_result['status']}")
    print(f"✅ Violations found: {validation_result['violation_summary']['total_violations']}")
    
    # Run CI test
    ci_result = guard.run_ci_lookahead_test("src/core/features")
    print(f"✅ CI test result: {ci_result['test_result']}")
    print(f"✅ Total violations: {ci_result['total_violations']}")
    
    # Clean up test file
    Path(test_feature_file).unlink(missing_ok=True)
    
    print("✅ Lookahead guard testing completed")


if __name__ == "__main__":
    main()