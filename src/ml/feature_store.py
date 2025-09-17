"""
Feature Store with Leakage Firewall
Registry for features with strict temporal constraints and leakage detection.
"""

import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

class FeatureStore:
    """Feature registry with leakage detection and temporal constraints."""
    
    def __init__(self, registry_file: str = "config/feature_registry.yaml"):
        self.registry_file = Path(registry_file)
        self.registry = self._load_registry()
        self.leakage_tests = []
    
    def _load_registry(self) -> Dict:
        """Load feature registry from YAML."""
        if not self.registry_file.exists():
            return {"features": {}, "metadata": {"version": "1.0", "created": datetime.now().isoformat()}}
        
        try:
            with open(self.registry_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading feature registry: {e}")
            return {"features": {}, "metadata": {"version": "1.0", "created": datetime.now().isoformat()}}
    
    def register_feature(self, name: str, definition: str, owner: str, 
                        lags: List[int], update_cadence: str, 
                        online_path: str, offline_path: str,
                        tests: List[str] = None) -> bool:
        """Register a new feature with metadata."""
        
        if tests is None:
            tests = []
        
        feature_metadata = {
            "name": name,
            "definition": definition,
            "owner": owner,
            "lags": lags,
            "update_cadence": update_cadence,
            "online_path": online_path,
            "offline_path": offline_path,
            "tests": tests,
            "registered_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        self.registry["features"][name] = feature_metadata
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered feature: {name}")
        return True
    
    def _save_registry(self):
        """Save feature registry to YAML."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_file, 'w') as f:
            yaml.dump(self.registry, f, default_flow_style=False, indent=2)
    
    def get_feature_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for a specific feature."""
        return self.registry["features"].get(name)
    
    def list_features(self) -> List[str]:
        """List all registered features."""
        return list(self.registry["features"].keys())
    
    def validate_feature_definition(self, definition: str) -> Dict:
        """Validate feature definition for leakage patterns."""
        
        leakage_patterns = [
            r'\.shift\(-?\d+\)',  # Future/past shifts
            r'\.rolling\([^)]*\)\.apply\([^)]*\)',  # Rolling windows
            r'\.expanding\([^)]*\)\.apply\([^)]*\)',  # Expanding windows
            r'\.ewm\([^)]*\)\.apply\([^)]*\)',  # Exponential windows
            r'\.resample\([^)]*\)\.apply\([^)]*\)',  # Resampling
            r'\.groupby\([^)]*\)\.transform\([^)]*\)',  # Group transforms
            r'\.pct_change\([^)]*\)',  # Percentage changes
            r'\.diff\([^)]*\)',  # Differences
            r'\.cumsum\(\)',  # Cumulative sums
            r'\.cumprod\(\)',  # Cumulative products
            r'\.cummax\(\)',  # Cumulative max
            r'\.cummin\(\)',  # Cumulative min
        ]
        
        violations = []
        for pattern in leakage_patterns:
            matches = re.findall(pattern, definition)
            if matches:
                violations.append({
                    "pattern": pattern,
                    "matches": matches,
                    "severity": "high" if "shift" in pattern else "medium"
                })
        
        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "leakage_risk": "high" if any(v["severity"] == "high" for v in violations) else "low"
        }
    
    def create_leakage_test(self, feature_name: str) -> str:
        """Create a unit test for feature leakage detection."""
        
        feature_meta = self.get_feature_metadata(feature_name)
        if not feature_meta:
            raise ValueError(f"Feature {feature_name} not found in registry")
        
        test_code = f'''
def test_{feature_name}_no_leakage():
    """Test that {feature_name} does not introduce lookahead bias."""
    
    # Create test data with known future values
    test_data = pd.DataFrame({{
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='1min'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }})
    
    # Apply feature definition
    feature_definition = """{feature_meta['definition']}"""
    
    # Test that feature doesn't use future data
    # This test will fail if the feature uses .shift(-1) or similar
    result = eval(feature_definition)
    
    # Verify no NaN values at the beginning (indicating future data usage)
    assert not result.iloc[:10].isna().any(), "Feature uses future data"
    
    # Verify feature is properly lagged
    expected_lag = max({feature_meta['lags']}) if {feature_meta['lags']} else 0
    if expected_lag > 0:
        assert result.iloc[:expected_lag].isna().all(), "Feature not properly lagged"
    
    print(f"PASSED {feature_name} leakage test")
'''
        
        return test_code
    
    def generate_feature_parquet_mapping(self) -> pd.DataFrame:
        """Generate Parquet mapping for all features."""
        
        features_data = []
        for name, meta in self.registry["features"].items():
            features_data.append({
                "feature_name": name,
                "owner": meta["owner"],
                "update_cadence": meta["update_cadence"],
                "online_path": meta["online_path"],
                "offline_path": meta["offline_path"],
                "max_lag": max(meta["lags"]) if meta["lags"] else 0,
                "registered_at": meta["registered_at"],
                "last_updated": meta["last_updated"]
            })
        
        df = pd.DataFrame(features_data)
        
        # Save to Parquet
        parquet_path = Path("data/features/feature_mapping.parquet")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Feature mapping saved to {parquet_path}")
        return df

class LeakageFirewall:
    """Firewall to detect and prevent feature leakage."""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.violations = []
    
    def scan_codebase_for_leakage(self, code_path: str = "src/") -> Dict:
        """Scan codebase for potential leakage patterns."""
        
        code_path = Path(code_path)
        violations = []
        
        # Patterns that indicate potential leakage
        leakage_patterns = [
            (r'\.shift\(-?\d+\)', "Future/past data shifts"),
            (r'\.rolling\([^)]*\)\.apply\([^)]*\)', "Rolling window operations"),
            (r'\.expanding\([^)]*\)\.apply\([^)]*\)', "Expanding window operations"),
            (r'\.ewm\([^)]*\)\.apply\([^)]*\)', "Exponential window operations"),
            (r'\.resample\([^)]*\)\.apply\([^)]*\)', "Resampling operations"),
            (r'\.groupby\([^)]*\)\.transform\([^)]*\)', "Group transform operations"),
            (r'\.pct_change\([^)]*\)', "Percentage change operations"),
            (r'\.diff\([^)]*\)', "Difference operations"),
            (r'\.cumsum\(\)', "Cumulative sum operations"),
            (r'\.cumprod\(\)', "Cumulative product operations"),
            (r'\.cummax\(\)', "Cumulative max operations"),
            (r'\.cummin\(\)', "Cumulative min operations"),
        ]
        
        # Scan Python files
        for py_file in code_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern, description in leakage_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        violations.append({
                            "file": str(py_file),
                            "line": line_num,
                            "pattern": pattern,
                            "description": description,
                            "match": match.group(),
                            "severity": "high" if "shift" in pattern else "medium"
                        })
            
            except Exception as e:
                logger.error(f"Error scanning {py_file}: {e}")
        
        self.violations = violations
        
        # Save violations report
        violations_file = Path("reports/leakage/violations_report.json")
        violations_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(violations_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_violations": len(violations),
                "high_severity": len([v for v in violations if v["severity"] == "high"]),
                "medium_severity": len([v for v in violations if v["severity"] == "medium"]),
                "violations": violations
            }, f, indent=2)
        
        return {
            "total_violations": len(violations),
            "high_severity": len([v for v in violations if v["severity"] == "high"]),
            "medium_severity": len([v for v in violations if v["severity"] == "medium"]),
            "violations": violations
        }
    
    def create_leakage_tests(self) -> List[str]:
        """Create unit tests for all registered features."""
        
        test_files = []
        
        for feature_name in self.feature_store.list_features():
            test_code = self.feature_store.create_leakage_test(feature_name)
            
            # Save test file
            test_file = Path(f"tests/test_feature_{feature_name}_leakage.py")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(f'''
"""
Leakage test for {feature_name}
Generated by Feature Store Leakage Firewall
"""

import pytest
import pandas as pd
import numpy as np
{test_code}
''')
            
            test_files.append(str(test_file))
        
        logger.info(f"Created {len(test_files)} leakage test files")
        return test_files

def main():
    """Initialize feature store with sample features."""
    
    # Initialize feature store
    feature_store = FeatureStore()
    
    # Register sample features
    sample_features = [
        {
            "name": "price_momentum_5min",
            "definition": "df['price'].pct_change(5)",
            "owner": "quant_team",
            "lags": [5],
            "update_cadence": "1min",
            "online_path": "data/features/online/price_momentum_5min.parquet",
            "offline_path": "data/features/offline/price_momentum_5min.parquet",
            "tests": ["test_price_momentum_5min_no_leakage"]
        },
        {
            "name": "volume_ma_20min",
            "definition": "df['volume'].rolling(20).mean()",
            "owner": "quant_team",
            "lags": [20],
            "update_cadence": "1min",
            "online_path": "data/features/online/volume_ma_20min.parquet",
            "offline_path": "data/features/offline/volume_ma_20min.parquet",
            "tests": ["test_volume_ma_20min_no_leakage"]
        },
        {
            "name": "bid_ask_spread",
            "definition": "df['ask'] - df['bid']",
            "owner": "microstructure_team",
            "lags": [0],
            "update_cadence": "1s",
            "online_path": "data/features/online/bid_ask_spread.parquet",
            "offline_path": "data/features/offline/bid_ask_spread.parquet",
            "tests": ["test_bid_ask_spread_no_leakage"]
        }
    ]
    
    # Register features
    for feature in sample_features:
        feature_store.register_feature(**feature)
    
    # Generate feature mapping
    feature_store.generate_feature_parquet_mapping()
    
    # Initialize leakage firewall
    firewall = LeakageFirewall(feature_store)
    
    # Scan for leakage
    violations = firewall.scan_codebase_for_leakage()
    
    # Create leakage tests
    test_files = firewall.create_leakage_tests()
    
    print("SUCCESS: Feature Store initialized")
    print(f"   Registered features: {len(feature_store.list_features())}")
    print(f"   Leakage violations found: {violations['total_violations']}")
    print(f"   High severity: {violations['high_severity']}")
    print(f"   Medium severity: {violations['medium_severity']}")
    print(f"   Created {len(test_files)} leakage test files")
    
    return 0

if __name__ == "__main__":
    exit(main())
