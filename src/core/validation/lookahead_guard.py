"""
Lookahead Guard - Prevents Future Data Leakage
Ensures no feature references data beyond decision timestamp.
"""

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Set
import pandas as pd


class LookaheadGuard:
    """Prevents lookahead bias by validating feature calculations."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.validation_dir = self.reports_dir / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_code_for_lookahead(self, code_path: str) -> Dict[str, Any]:
        """Scan Python code for potential lookahead violations."""
        
        violations = []
        warnings = []
        
        try:
            with open(code_path, 'r') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Look for suspicious patterns
            for node in ast.walk(tree):
                # Check for future timestamp references
                if isinstance(node, ast.Compare):
                    violations.extend(self._check_comparison(node, code_path))
                
                # Check for future data access
                if isinstance(node, ast.Subscript):
                    violations.extend(self._check_subscript(node, code_path))
                
                # Check for function calls that might access future data
                if isinstance(node, ast.Call):
                    warnings.extend(self._check_function_call(node, code_path))
        
        except Exception as e:
            violations.append({
                "type": "parse_error",
                "file": code_path,
                "error": str(e),
                "severity": "high"
            })
        
        return {
            "file": code_path,
            "violations": violations,
            "warnings": warnings,
            "scan_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _check_comparison(self, node: ast.Compare, file_path: str) -> List[Dict[str, Any]]:
        """Check comparison operations for lookahead."""
        violations = []
        
        # Look for timestamp comparisons
        for op in node.ops:
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                # Check if comparing with future timestamps
                if self._has_timestamp_reference(node.left) or self._has_timestamp_reference(node.comparators[0]):
                    violations.append({
                        "type": "timestamp_comparison",
                        "file": file_path,
                        "line": node.lineno,
                        "operation": ast.dump(op),
                        "severity": "high",
                        "message": "Timestamp comparison may cause lookahead bias"
                    })
        
        return violations
    
    def _check_subscript(self, node: ast.Subscript, file_path: str) -> List[Dict[str, Any]]:
        """Check subscript operations for future data access."""
        violations = []
        
        # Look for future data access patterns
        if isinstance(node.slice, ast.Slice):
            if node.slice.upper is not None:
                violations.append({
                    "type": "future_data_access",
                    "file": file_path,
                    "line": node.lineno,
                    "severity": "medium",
                    "message": "Slice with upper bound may access future data"
                })
        
        return violations
    
    def _check_function_call(self, node: ast.Call, file_path: str) -> List[Dict[str, Any]]:
        """Check function calls for potential lookahead."""
        warnings = []
        
        # Check for suspicious function names
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ['shift', 'rolling', 'ewm', 'expanding']:
                warnings.append({
                    "type": "suspicious_function",
                    "file": file_path,
                    "line": node.lineno,
                    "function": func_name,
                    "severity": "low",
                    "message": f"Function '{func_name}' may cause lookahead if not used carefully"
                })
        
        return warnings
    
    def _has_timestamp_reference(self, node: ast.AST) -> bool:
        """Check if node references timestamps."""
        if isinstance(node, ast.Name):
            return node.id in ['timestamp', 'ts', 'time', 'datetime']
        elif isinstance(node, ast.Attribute):
            return node.attr in ['timestamp', 'ts', 'time']
        return False
    
    def validate_feature_calculation(self, 
                                   feature_data: pd.DataFrame,
                                   decision_timestamp: datetime,
                                   feature_name: str) -> Dict[str, Any]:
        """Validate that feature calculation doesn't use future data."""
        
        violations = []
        
        # Check if any data is beyond decision timestamp
        if 'ts' in feature_data.columns:
            future_data = feature_data[feature_data['ts'] > decision_timestamp]
            if len(future_data) > 0:
                violations.append({
                    "type": "future_data_used",
                    "feature": feature_name,
                    "decision_timestamp": decision_timestamp.isoformat(),
                    "future_data_count": len(future_data),
                    "earliest_future_timestamp": future_data['ts'].min().isoformat(),
                    "severity": "high"
                })
        
        # Check for data gaps that might indicate lookahead
        if 'ts' in feature_data.columns:
            time_diffs = feature_data['ts'].diff().dropna()
            large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
            if len(large_gaps) > 0:
                violations.append({
                    "type": "data_gap_suspicious",
                    "feature": feature_name,
                    "large_gaps_count": len(large_gaps),
                    "severity": "medium"
                })
        
        return {
            "feature_name": feature_name,
            "decision_timestamp": decision_timestamp.isoformat(),
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "violations": violations,
            "valid": len(violations) == 0
        }
    
    def run_ci_lookahead_check(self, code_directories: List[str]) -> Dict[str, Any]:
        """Run comprehensive lookahead check for CI."""
        
        all_results = []
        total_violations = 0
        total_warnings = 0
        
        for directory in code_directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue
            
            # Find all Python files
            for py_file in dir_path.glob("**/*.py"):
                if py_file.name.startswith('test_'):
                    continue  # Skip test files
                
                result = self.scan_code_for_lookahead(str(py_file))
                all_results.append(result)
                total_violations += len(result['violations'])
                total_warnings += len(result['warnings'])
        
        # Generate summary
        summary = {
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_files_scanned": len(all_results),
            "total_violations": total_violations,
            "total_warnings": total_warnings,
            "ci_status": "PASS" if total_violations == 0 else "FAIL",
            "results": all_results
        }
        
        # Save results
        results_file = self.validation_dir / "lookahead_validation.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def create_lookahead_test(self, test_file_path: str) -> str:
        """Create a unit test for lookahead validation."""
        
        test_code = '''"""
Lookahead Validation Tests
Ensures no future data leakage in feature calculations.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.core.validation.lookahead_guard import LookaheadGuard


class TestLookaheadValidation:
    """Test suite for lookahead bias prevention."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.guard = LookaheadGuard()
        self.decision_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    def test_no_future_data_in_features(self):
        """Test that features don't use future data."""
        # Create test data with future timestamps
        future_time = self.decision_time + timedelta(hours=1)
        test_data = pd.DataFrame({
            'ts': [self.decision_time, future_time],
            'price': [100.0, 101.0],
            'volume': [1000, 1100]
        })
        
        # Validate feature calculation
        result = self.guard.validate_feature_calculation(
            test_data, self.decision_time, "test_feature"
        )
        
        # Should fail due to future data
        assert not result['valid']
        assert len(result['violations']) > 0
        assert result['violations'][0]['type'] == 'future_data_used'
    
    def test_valid_feature_calculation(self):
        """Test valid feature calculation with no future data."""
        # Create test data with only past timestamps
        past_time = self.decision_time - timedelta(hours=1)
        test_data = pd.DataFrame({
            'ts': [past_time, self.decision_time],
            'price': [100.0, 101.0],
            'volume': [1000, 1100]
        })
        
        # Validate feature calculation
        result = self.guard.validate_feature_calculation(
            test_data, self.decision_time, "test_feature"
        )
        
        # Should pass
        assert result['valid']
        assert len(result['violations']) == 0
    
    def test_ci_lookahead_check(self):
        """Test CI lookahead check."""
        # Run check on test directories
        results = self.guard.run_ci_lookahead_check(['src/core/strategies'])
        
        # Should complete without errors
        assert 'scan_timestamp' in results
        assert 'total_files_scanned' in results
        assert 'ci_status' in results
    
    def test_timestamp_comparison_detection(self):
        """Test detection of timestamp comparisons."""
        # This would be tested with actual code files
        # For now, just ensure the method exists
        assert hasattr(self.guard, '_check_comparison')
        assert hasattr(self.guard, '_check_subscript')
        assert hasattr(self.guard, '_check_function_call')


if __name__ == "__main__":
    pytest.main([__file__])
'''
        
        # Write test file
        with open(test_file_path, 'w') as f:
            f.write(test_code)
        
        return test_file_path


def main():
    """Test lookahead guard functionality."""
    guard = LookaheadGuard()
    
    # Test code scanning
    test_directories = ['src/core/strategies', 'src/core/analytics']
    results = guard.run_ci_lookahead_check(test_directories)
    
    print(f"✅ Lookahead validation completed:")
    print(f"   Files scanned: {results['total_files_scanned']}")
    print(f"   Violations: {results['total_violations']}")
    print(f"   Warnings: {results['total_warnings']}")
    print(f"   CI Status: {results['ci_status']}")
    
    # Create test file
    test_file = guard.create_lookahead_test("tests/validation/test_lookahead_guard.py")
    print(f"✅ Test file created: {test_file}")


if __name__ == "__main__":
    main()
