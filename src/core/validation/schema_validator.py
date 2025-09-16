"""
Schema Validator - Data Integrity & Quality Assurance
Implements Great-Expectations-style schema tests and outlier detection.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import hashlib


class ValidationResult(Enum):
    """Validation result enumeration."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    ERROR = "error"


@dataclass
class ValidationCheck:
    """Represents a validation check result."""
    check_name: str
    result: ValidationResult
    message: str
    details: Dict[str, Any]
    timestamp: datetime


class SchemaValidator:
    """Comprehensive schema validation and data quality assurance."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.validation_dir = self.reports_dir / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Schema definitions
        self.schemas = self._define_schemas()
        
        # Outlier policies
        self.outlier_policies = self._define_outlier_policies()
        
        # Validation history
        self.validation_history = []
    
    def _define_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Define schema specifications for different data types."""
        
        schemas = {
            "trade_data": {
                "required_columns": ["ts", "symbol", "side", "qty", "price", "order_id"],
                "column_types": {
                    "ts": "datetime64[ns]",
                    "symbol": "object",
                    "side": "object",
                    "qty": "float64",
                    "price": "float64",
                    "order_id": "object"
                },
                "constraints": {
                    "ts": {"not_null": True, "unique": False},
                    "symbol": {"not_null": True, "allowed_values": ["XRP", "BTC", "ETH"]},
                    "side": {"not_null": True, "allowed_values": ["buy", "sell"]},
                    "qty": {"not_null": True, "min_value": 0.0, "max_value": 1000000.0},
                    "price": {"not_null": True, "min_value": 0.0, "max_value": 1000000.0},
                    "order_id": {"not_null": True, "unique": True}
                }
            },
            
            "market_data": {
                "required_columns": ["ts", "symbol", "bid", "ask", "bid_size", "ask_size"],
                "column_types": {
                    "ts": "datetime64[ns]",
                    "symbol": "object",
                    "bid": "float64",
                    "ask": "float64",
                    "bid_size": "float64",
                    "ask_size": "float64"
                },
                "constraints": {
                    "ts": {"not_null": True, "unique": False},
                    "symbol": {"not_null": True, "allowed_values": ["XRP", "BTC", "ETH"]},
                    "bid": {"not_null": True, "min_value": 0.0},
                    "ask": {"not_null": True, "min_value": 0.0},
                    "bid_size": {"not_null": True, "min_value": 0.0},
                    "ask_size": {"not_null": True, "min_value": 0.0}
                }
            },
            
            "order_book": {
                "required_columns": ["ts", "symbol", "level", "side", "price", "size"],
                "column_types": {
                    "ts": "datetime64[ns]",
                    "symbol": "object",
                    "level": "int64",
                    "side": "object",
                    "price": "float64",
                    "size": "float64"
                },
                "constraints": {
                    "ts": {"not_null": True},
                    "symbol": {"not_null": True},
                    "level": {"not_null": True, "min_value": 0, "max_value": 20},
                    "side": {"not_null": True, "allowed_values": ["bid", "ask"]},
                    "price": {"not_null": True, "min_value": 0.0},
                    "size": {"not_null": True, "min_value": 0.0}
                }
            }
        }
        
        return schemas
    
    def _define_outlier_policies(self) -> Dict[str, Dict[str, Any]]:
        """Define outlier detection policies."""
        
        policies = {
            "price_outliers": {
                "method": "iqr",
                "threshold": 3.0,
                "action": "flag_and_log",
                "description": "Detect price outliers using IQR method"
            },
            "volume_outliers": {
                "method": "z_score",
                "threshold": 3.0,
                "action": "flag_and_log",
                "description": "Detect volume outliers using Z-score"
            },
            "timestamp_gaps": {
                "method": "time_delta",
                "threshold_seconds": 3600,
                "action": "flag_and_log",
                "description": "Detect large gaps in timestamps"
            },
            "duplicate_records": {
                "method": "exact_match",
                "action": "remove_and_log",
                "description": "Detect and remove duplicate records"
            }
        }
        
        return policies
    
    def validate_schema(self, 
                       data: pd.DataFrame, 
                       schema_name: str) -> List[ValidationCheck]:
        """Validate data against schema specification."""
        
        if schema_name not in self.schemas:
            return [ValidationCheck(
                check_name="schema_exists",
                result=ValidationResult.ERROR,
                message=f"Schema '{schema_name}' not found",
                details={"available_schemas": list(self.schemas.keys())},
                timestamp=datetime.now(timezone.utc)
            )]
        
        schema = self.schemas[schema_name]
        checks = []
        
        # Check required columns
        missing_columns = set(schema["required_columns"]) - set(data.columns)
        if missing_columns:
            checks.append(ValidationCheck(
                check_name="required_columns",
                result=ValidationResult.FAIL,
                message=f"Missing required columns: {missing_columns}",
                details={"missing_columns": list(missing_columns)},
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Check column types
        for column, expected_type in schema["column_types"].items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    checks.append(ValidationCheck(
                        check_name=f"column_type_{column}",
                        result=ValidationResult.WARN,
                        message=f"Column '{column}' type mismatch: expected {expected_type}, got {actual_type}",
                        details={"expected": expected_type, "actual": actual_type},
                        timestamp=datetime.now(timezone.utc)
                    ))
        
        # Check constraints
        for column, constraints in schema["constraints"].items():
            if column in data.columns:
                column_checks = self._validate_column_constraints(
                    data[column], column, constraints
                )
                checks.extend(column_checks)
        
        return checks
    
    def _validate_column_constraints(self, 
                                   series: pd.Series, 
                                   column_name: str, 
                                   constraints: Dict[str, Any]) -> List[ValidationCheck]:
        """Validate column constraints."""
        
        checks = []
        
        # Check null constraints
        if constraints.get("not_null", False):
            null_count = series.isnull().sum()
            if null_count > 0:
                checks.append(ValidationCheck(
                    check_name=f"not_null_{column_name}",
                    result=ValidationResult.FAIL,
                    message=f"Column '{column_name}' has {null_count} null values",
                    details={"null_count": int(null_count), "total_count": len(series)},
                    timestamp=datetime.now(timezone.utc)
                ))
        
        # Check unique constraints
        if constraints.get("unique", False):
            duplicate_count = series.duplicated().sum()
            if duplicate_count > 0:
                checks.append(ValidationCheck(
                    check_name=f"unique_{column_name}",
                    result=ValidationResult.FAIL,
                    message=f"Column '{column_name}' has {duplicate_count} duplicate values",
                    details={"duplicate_count": int(duplicate_count)},
                    timestamp=datetime.now(timezone.utc)
                ))
        
        # Check allowed values
        if "allowed_values" in constraints:
            invalid_values = series[~series.isin(constraints["allowed_values"])]
            if len(invalid_values) > 0:
                checks.append(ValidationCheck(
                    check_name=f"allowed_values_{column_name}",
                    result=ValidationResult.FAIL,
                    message=f"Column '{column_name}' has {len(invalid_values)} invalid values",
                    details={"invalid_values": invalid_values.unique().tolist()},
                    timestamp=datetime.now(timezone.utc)
                ))
        
        # Check numeric constraints
        if series.dtype in ['int64', 'float64']:
            if "min_value" in constraints:
                below_min = series[series < constraints["min_value"]]
                if len(below_min) > 0:
                    checks.append(ValidationCheck(
                        check_name=f"min_value_{column_name}",
                        result=ValidationResult.FAIL,
                        message=f"Column '{column_name}' has {len(below_min)} values below minimum",
                        details={"min_value": constraints["min_value"], "violations": len(below_min)},
                        timestamp=datetime.now(timezone.utc)
                    ))
            
            if "max_value" in constraints:
                above_max = series[series > constraints["max_value"]]
                if len(above_max) > 0:
                    checks.append(ValidationCheck(
                        check_name=f"max_value_{column_name}",
                        result=ValidationResult.FAIL,
                        message=f"Column '{column_name}' has {len(above_max)} values above maximum",
                        details={"max_value": constraints["max_value"], "violations": len(above_max)},
                        timestamp=datetime.now(timezone.utc)
                    ))
        
        return checks
    
    def detect_outliers(self, 
                       data: pd.DataFrame, 
                       column: str, 
                       policy_name: str) -> Dict[str, Any]:
        """Detect outliers using specified policy."""
        
        if policy_name not in self.outlier_policies:
            return {
                "error": f"Policy '{policy_name}' not found",
                "available_policies": list(self.outlier_policies.keys())
            }
        
        policy = self.outlier_policies[policy_name]
        
        if column not in data.columns:
            return {"error": f"Column '{column}' not found in data"}
        
        series = data[column].dropna()
        
        if len(series) == 0:
            return {"error": "No data to analyze"}
        
        outliers = []
        
        if policy["method"] == "iqr":
            outliers = self._detect_iqr_outliers(series, policy["threshold"])
        elif policy["method"] == "z_score":
            outliers = self._detect_zscore_outliers(series, policy["threshold"])
        elif policy["method"] == "time_delta":
            outliers = self._detect_time_delta_outliers(data, column, policy["threshold_seconds"])
        elif policy["method"] == "exact_match":
            outliers = self._detect_duplicate_outliers(data)
        
        result = {
            "policy_name": policy_name,
            "column": column,
            "method": policy["method"],
            "threshold": policy.get("threshold", policy.get("threshold_seconds")),
            "total_records": len(data),
            "outliers_detected": len(outliers),
            "outlier_indices": outliers,
            "action": policy["action"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return result
    
    def _detect_iqr_outliers(self, series: pd.Series, threshold: float) -> List[int]:
        """Detect outliers using IQR method."""
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index.tolist()
    
    def _detect_zscore_outliers(self, series: pd.Series, threshold: float) -> List[int]:
        """Detect outliers using Z-score method."""
        
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold]
        return outliers.index.tolist()
    
    def _detect_time_delta_outliers(self, data: pd.DataFrame, column: str, threshold_seconds: int) -> List[int]:
        """Detect large time gaps."""
        
        if column not in data.columns:
            return []
        
        time_series = pd.to_datetime(data[column])
        time_diffs = time_series.diff().dropna()
        
        large_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=threshold_seconds)]
        return large_gaps.index.tolist()
    
    def _detect_duplicate_outliers(self, data: pd.DataFrame) -> List[int]:
        """Detect duplicate records."""
        
        duplicates = data.duplicated()
        return duplicates[duplicates].index.tolist()
    
    def apply_outlier_policy(self, 
                           data: pd.DataFrame, 
                           outlier_result: Dict[str, Any]) -> pd.DataFrame:
        """Apply outlier policy to data."""
        
        action = outlier_result.get("action", "flag_and_log")
        outlier_indices = outlier_result.get("outlier_indices", [])
        
        if action == "remove_and_log":
            # Remove outliers
            cleaned_data = data.drop(index=outlier_indices)
            
            # Log removal
            self._log_outlier_action(outlier_result, "removed", len(outlier_indices))
            
            return cleaned_data
        
        elif action == "flag_and_log":
            # Flag outliers but keep data
            data_copy = data.copy()
            data_copy["_outlier_flag"] = False
            data_copy.loc[outlier_indices, "_outlier_flag"] = True
            
            # Log flagging
            self._log_outlier_action(outlier_result, "flagged", len(outlier_indices))
            
            return data_copy
        
        else:
            # Default: just log
            self._log_outlier_action(outlier_result, "logged", len(outlier_indices))
            return data
    
    def _log_outlier_action(self, outlier_result: Dict[str, Any], action: str, count: int):
        """Log outlier action for monitoring."""
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy_name": outlier_result["policy_name"],
            "column": outlier_result["column"],
            "action": action,
            "outlier_count": count,
            "total_records": outlier_result["total_records"]
        }
        
        # Save to outlier log
        outlier_log_file = self.validation_dir / "outlier_actions.jsonl"
        with open(outlier_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def calculate_data_checksum(self, data: pd.DataFrame) -> str:
        """Calculate SHA-256 checksum of data."""
        
        # Convert data to string representation for hashing
        data_str = data.to_string()
        
        # Calculate SHA-256 hash
        sha256_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        return sha256_hash
    
    def validate_data_integrity(self, 
                              data: pd.DataFrame, 
                              schema_name: str,
                              expected_checksum: str = None) -> Dict[str, Any]:
        """Comprehensive data integrity validation."""
        
        validation_start = datetime.now(timezone.utc)
        
        # Schema validation
        schema_checks = self.validate_schema(data, schema_name)
        
        # Outlier detection
        outlier_results = {}
        for policy_name in self.outlier_policies.keys():
            # Apply to numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                outlier_result = self.detect_outliers(data, column, policy_name)
                if "error" not in outlier_result:
                    outlier_results[f"{policy_name}_{column}"] = outlier_result
        
        # Checksum validation
        checksum_validation = {}
        if expected_checksum:
            actual_checksum = self.calculate_data_checksum(data)
            checksum_validation = {
                "expected": expected_checksum,
                "actual": actual_checksum,
                "matches": expected_checksum == actual_checksum
            }
        
        # Overall validation result
        schema_failures = [check for check in schema_checks if check.result == ValidationResult.FAIL]
        overall_status = "PASS" if len(schema_failures) == 0 else "FAIL"
        
        validation_result = {
            "validation_timestamp": validation_start.isoformat(),
            "schema_name": schema_name,
            "data_shape": data.shape,
            "overall_status": overall_status,
            "schema_validation": {
                "total_checks": len(schema_checks),
                "passed": len([c for c in schema_checks if c.result == ValidationResult.PASS]),
                "failed": len([c for c in schema_checks if c.result == ValidationResult.FAIL]),
                "warnings": len([c for c in schema_checks if c.result == ValidationResult.WARN]),
                "checks": [
                    {
                        "name": check.check_name,
                        "result": check.result.value,
                        "message": check.message,
                        "details": check.details
                    }
                    for check in schema_checks
                ]
            },
            "outlier_detection": outlier_results,
            "checksum_validation": checksum_validation,
            "data_checksum": self.calculate_data_checksum(data)
        }
        
        # Save validation result
        validation_file = self.validation_dir / f"validation_{schema_name}_{validation_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_result, f, indent=2)
        
        return validation_result


def main():
    """Test schema validator functionality."""
    validator = SchemaValidator()
    
    # Create sample trade data
    sample_data = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'symbol': ['XRP'] * 100,
        'side': ['buy', 'sell'] * 50,
        'qty': np.random.uniform(10, 1000, 100),
        'price': np.random.uniform(0.4, 0.6, 100),
        'order_id': [f'order_{i}' for i in range(100)]
    })
    
    # Test schema validation
    schema_checks = validator.validate_schema(sample_data, "trade_data")
    print(f"✅ Schema validation: {len(schema_checks)} checks performed")
    
    # Test outlier detection
    outlier_result = validator.detect_outliers(sample_data, "price", "price_outliers")
    print(f"✅ Outlier detection: {outlier_result['outliers_detected']} outliers found")
    
    # Test data integrity validation
    integrity_result = validator.validate_data_integrity(sample_data, "trade_data")
    print(f"✅ Data integrity validation: {integrity_result['overall_status']}")
    
    print("✅ Schema validator testing completed")


if __name__ == "__main__":
    main()
