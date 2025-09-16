"""
Adaptive Governance - ML Model Management & Deployment
Implements time-blocked CV, shadow mode, and feature store governance.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib


class ModelStatus(Enum):
    """Model status enumeration."""
    TRAINING = "training"
    VALIDATED = "validated"
    SHADOW = "shadow"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Represents a model version."""
    version_id: str
    model_type: str
    status: ModelStatus
    created_at: datetime
    training_data_hash: str
    validation_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    performance_threshold: float


class AdaptiveGovernance:
    """Governs ML model lifecycle and adaptive system deployment."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.ml_dir = self.reports_dir / "ml_governance"
        self.ml_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_versions: List[ModelVersion] = []
        
        # Feature store
        self.feature_store = {}
        
        # Governance rules
        self.governance_rules = self._define_governance_rules()
        
        # Load existing state
        self._load_state()
    
    def _define_governance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define governance rules for ML model deployment."""
        
        rules = {
            "time_blocked_cv": {
                "enabled": True,
                "min_train_period_days": 30,
                "max_train_period_days": 365,
                "test_period_days": 7,
                "gap_days": 1,
                "min_folds": 3,
                "max_folds": 10
            },
            "shadow_mode": {
                "enabled": True,
                "min_shadow_period_days": 7,
                "max_shadow_period_days": 30,
                "performance_threshold": 0.05,  # 5% improvement required
                "statistical_significance": 0.95,
                "min_samples": 100
            },
            "feature_store": {
                "enabled": True,
                "max_feature_age_days": 7,
                "drift_threshold": 0.1,
                "correlation_threshold": 0.95,
                "missing_value_threshold": 0.1
            },
            "model_deployment": {
                "enabled": True,
                "approval_required": True,
                "rollback_threshold": 0.1,  # 10% performance degradation
                "monitoring_period_hours": 24,
                "max_rollback_attempts": 3
            }
        }
        
        return rules
    
    def create_time_blocked_cv(self, 
                             data: pd.DataFrame,
                             target_column: str,
                             time_column: str,
                             model_type: str = "regression") -> Dict[str, Any]:
        """Create time-blocked cross-validation splits."""
        
        rules = self.governance_rules["time_blocked_cv"]
        
        if not rules["enabled"]:
            return {"error": "Time-blocked CV is disabled"}
        
        # Sort data by time
        data_sorted = data.sort_values(time_column)
        
        # Calculate date range
        start_date = pd.to_datetime(data_sorted[time_column].min())
        end_date = pd.to_datetime(data_sorted[time_column].max())
        total_days = (end_date - start_date).days
        
        # Validate minimum training period
        if total_days < rules["min_train_period_days"]:
            return {
                "error": f"Insufficient data: {total_days} days < {rules['min_train_period_days']} required"
            }
        
        # Calculate optimal number of folds
        max_possible_folds = total_days // (rules["min_train_period_days"] + rules["test_period_days"] + rules["gap_days"])
        num_folds = min(max(rules["min_folds"], max_possible_folds), rules["max_folds"])
        
        # Create CV splits
        cv_splits = []
        train_period_days = min(rules["max_train_period_days"], total_days // 2)
        
        for fold in range(num_folds):
            # Calculate split dates
            test_start = start_date + timedelta(days=fold * (rules["test_period_days"] + rules["gap_days"]))
            test_end = test_start + timedelta(days=rules["test_period_days"])
            train_start = max(start_date, test_start - timedelta(days=train_period_days))
            train_end = test_start - timedelta(days=rules["gap_days"])
            
            # Skip if not enough data
            if train_end <= train_start or test_end > end_date:
                continue
            
            # Create masks
            train_mask = (data_sorted[time_column] >= train_start) & (data_sorted[time_column] < train_end)
            test_mask = (data_sorted[time_column] >= test_start) & (data_sorted[time_column] < test_end)
            
            # Extract splits
            train_data = data_sorted[train_mask]
            test_data = data_sorted[test_mask]
            
            if len(train_data) > 0 and len(test_data) > 0:
                cv_splits.append({
                    "fold": fold,
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "train_samples": len(train_data),
                    "test_samples": len(test_data),
                    "train_data_hash": self._calculate_data_hash(train_data),
                    "test_data_hash": self._calculate_data_hash(test_data)
                })
        
        cv_result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "total_samples": len(data_sorted),
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "total_days": total_days
            },
            "cv_config": {
                "num_folds": len(cv_splits),
                "train_period_days": train_period_days,
                "test_period_days": rules["test_period_days"],
                "gap_days": rules["gap_days"]
            },
            "splits": cv_splits
        }
        
        # Save CV result
        cv_file = self.ml_dir / f"time_blocked_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(cv_file, 'w') as f:
            json.dump(cv_result, f, indent=2)
        
        return cv_result
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for integrity checking."""
        data_str = data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def deploy_shadow_mode(self, 
                          model_version: ModelVersion,
                          baseline_model: ModelVersion,
                          shadow_data: pd.DataFrame) -> Dict[str, Any]:
        """Deploy model in shadow mode for A/B testing."""
        
        rules = self.governance_rules["shadow_mode"]
        
        if not rules["enabled"]:
            return {"error": "Shadow mode is disabled"}
        
        # Validate shadow data
        if len(shadow_data) < rules["min_samples"]:
            return {
                "error": f"Insufficient shadow data: {len(shadow_data)} < {rules['min_samples']} required"
            }
        
        # Create shadow deployment
        shadow_deployment = {
            "deployment_id": f"shadow_{model_version.version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_version": model_version.version_id,
            "baseline_model": baseline_model.version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "shadow_data_samples": len(shadow_data),
            "performance_threshold": rules["performance_threshold"],
            "statistical_significance": rules["statistical_significance"],
            "min_shadow_period_days": rules["min_shadow_period_days"],
            "max_shadow_period_days": rules["max_shadow_period_days"]
        }
        
        # Save shadow deployment
        shadow_file = self.ml_dir / f"shadow_deployment_{shadow_deployment['deployment_id']}.json"
        with open(shadow_file, 'w') as f:
            json.dump(shadow_deployment, f, indent=2)
        
        return shadow_deployment
    
    def evaluate_shadow_performance(self, 
                                  deployment_id: str,
                                  shadow_predictions: List[float],
                                  baseline_predictions: List[float],
                                  actual_values: List[float]) -> Dict[str, Any]:
        """Evaluate shadow mode performance against baseline."""
        
        if len(shadow_predictions) != len(baseline_predictions) or len(shadow_predictions) != len(actual_values):
            return {"error": "Prediction arrays must have same length"}
        
        # Calculate performance metrics
        shadow_mae = np.mean(np.abs(np.array(shadow_predictions) - np.array(actual_values)))
        baseline_mae = np.mean(np.abs(np.array(baseline_predictions) - np.array(actual_values)))
        
        shadow_rmse = np.sqrt(np.mean((np.array(shadow_predictions) - np.array(actual_values))**2))
        baseline_rmse = np.sqrt(np.mean((np.array(baseline_predictions) - np.array(actual_values))**2))
        
        # Calculate improvement
        mae_improvement = (baseline_mae - shadow_mae) / baseline_mae
        rmse_improvement = (baseline_rmse - shadow_rmse) / baseline_rmse
        
        # Statistical significance test (simplified)
        shadow_errors = np.abs(np.array(shadow_predictions) - np.array(actual_values))
        baseline_errors = np.abs(np.array(baseline_predictions) - np.array(actual_values))
        
        # Paired t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(baseline_errors, shadow_errors)
        
        # Determine if improvement is significant
        rules = self.governance_rules["shadow_mode"]
        significant_improvement = (
            mae_improvement > rules["performance_threshold"] and 
            p_value < (1 - rules["statistical_significance"])
        )
        
        evaluation_result = {
            "deployment_id": deployment_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "sample_size": len(shadow_predictions),
            "shadow_performance": {
                "mae": float(shadow_mae),
                "rmse": float(shadow_rmse)
            },
            "baseline_performance": {
                "mae": float(baseline_mae),
                "rmse": float(baseline_rmse)
            },
            "improvement": {
                "mae_improvement": float(mae_improvement),
                "rmse_improvement": float(rmse_improvement)
            },
            "statistical_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": significant_improvement
            },
            "promotion_recommended": significant_improvement
        }
        
        # Save evaluation result
        eval_file = self.ml_dir / f"shadow_evaluation_{deployment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_result, f, indent=2)
        
        return evaluation_result
    
    def manage_feature_store(self, 
                           feature_name: str,
                           feature_data: pd.DataFrame,
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Manage feature store with drift detection and validation."""
        
        rules = self.governance_rules["feature_store"]
        
        if not rules["enabled"]:
            return {"error": "Feature store is disabled"}
        
        # Check feature age
        if feature_name in self.feature_store:
            existing_feature = self.feature_store[feature_name]
            age_days = (datetime.now(timezone.utc) - existing_feature["last_updated"]).days
            
            if age_days > rules["max_feature_age_days"]:
                return {
                    "error": f"Feature '{feature_name}' is too old: {age_days} days > {rules['max_feature_age_days']} limit"
                }
        
        # Validate feature data
        validation_result = self._validate_feature_data(feature_data, rules)
        
        if not validation_result["valid"]:
            return {
                "error": f"Feature validation failed: {validation_result['errors']}"
            }
        
        # Check for drift if feature exists
        drift_detected = False
        if feature_name in self.feature_store:
            drift_result = self._detect_feature_drift(
                self.feature_store[feature_name]["data"],
                feature_data,
                rules["drift_threshold"]
            )
            drift_detected = drift_result["drift_detected"]
        
        # Store feature
        feature_entry = {
            "feature_name": feature_name,
            "data_hash": self._calculate_data_hash(feature_data),
            "data_shape": feature_data.shape,
            "data_types": feature_data.dtypes.to_dict(),
            "statistics": {
                "mean": float(feature_data.mean().mean()) if feature_data.select_dtypes(include=[np.number]).size > 0 else None,
                "std": float(feature_data.std().mean()) if feature_data.select_dtypes(include=[np.number]).size > 0 else None,
                "missing_ratio": float(feature_data.isnull().sum().sum() / feature_data.size)
            },
            "metadata": metadata or {},
            "last_updated": datetime.now(timezone.utc),
            "drift_detected": drift_detected
        }
        
        self.feature_store[feature_name] = feature_entry
        
        # Save feature store
        self._save_feature_store()
        
        return {
            "feature_name": feature_name,
            "status": "stored",
            "validation_result": validation_result,
            "drift_detected": drift_detected,
            "feature_entry": feature_entry
        }
    
    def _validate_feature_data(self, 
                             data: pd.DataFrame, 
                             rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature data against governance rules."""
        
        errors = []
        
        # Check missing values
        missing_ratio = data.isnull().sum().sum() / data.size
        if missing_ratio > rules["missing_value_threshold"]:
            errors.append(f"Too many missing values: {missing_ratio:.2%} > {rules['missing_value_threshold']:.2%}")
        
        # Check for high correlation (if numeric)
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > rules["correlation_threshold"]:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                errors.append(f"High correlation detected: {len(high_corr_pairs)} pairs > {rules['correlation_threshold']}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "missing_ratio": missing_ratio,
            "high_correlation_pairs": len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0
        }
    
    def _detect_feature_drift(self, 
                            old_data: pd.DataFrame, 
                            new_data: pd.DataFrame, 
                            threshold: float) -> Dict[str, Any]:
        """Detect feature drift between old and new data."""
        
        # Simple drift detection using statistical tests
        drift_detected = False
        drift_details = []
        
        # Compare numeric columns
        numeric_cols = old_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in new_data.columns:
                old_values = old_data[col].dropna()
                new_values = new_data[col].dropna()
                
                if len(old_values) > 0 and len(new_values) > 0:
                    # Kolmogorov-Smirnov test
                    from scipy import stats
                    ks_stat, ks_pvalue = stats.ks_2samp(old_values, new_values)
                    
                    if ks_pvalue < 0.05:  # Significant difference
                        drift_detected = True
                        drift_details.append({
                            "column": col,
                            "test": "ks_test",
                            "statistic": float(ks_stat),
                            "p_value": float(ks_pvalue),
                            "drift_detected": True
                        })
        
        return {
            "drift_detected": drift_detected,
            "threshold": threshold,
            "drift_details": drift_details
        }
    
    def _save_feature_store(self):
        """Save feature store to disk."""
        
        # Convert datetime objects to strings for JSON serialization
        feature_store_serializable = {}
        for name, entry in self.feature_store.items():
            entry_copy = entry.copy()
            entry_copy["last_updated"] = entry_copy["last_updated"].isoformat()
            feature_store_serializable[name] = entry_copy
        
        feature_store_file = self.ml_dir / "feature_store.json"
        with open(feature_store_file, 'w') as f:
            json.dump(feature_store_serializable, f, indent=2)
    
    def _load_state(self):
        """Load state from disk."""
        
        # Load feature store
        feature_store_file = self.ml_dir / "feature_store.json"
        if feature_store_file.exists():
            try:
                with open(feature_store_file, 'r') as f:
                    feature_store_data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                for name, entry in feature_store_data.items():
                    entry["last_updated"] = datetime.fromisoformat(entry["last_updated"])
                    self.feature_store[name] = entry
                    
            except Exception as e:
                print(f"Warning: Could not load feature store: {e}")


def main():
    """Test adaptive governance functionality."""
    governance = AdaptiveGovernance()
    
    # Test time-blocked CV
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    cv_result = governance.create_time_blocked_cv(sample_data, 'target', 'date')
    print(f"✅ Time-blocked CV: {cv_result['cv_config']['num_folds']} folds created")
    
    # Test feature store
    feature_result = governance.manage_feature_store('test_feature', sample_data[['feature1', 'feature2']])
    print(f"✅ Feature store: {feature_result['status']}")
    
    # Test shadow mode (simplified)
    model_version = ModelVersion(
        version_id="v1.0.0",
        model_type="regression",
        status=ModelStatus.VALIDATED,
        created_at=datetime.now(timezone.utc),
        training_data_hash="abc123",
        validation_metrics={"mae": 0.1, "rmse": 0.15},
        feature_importance={"feature1": 0.6, "feature2": 0.4},
        hyperparameters={"learning_rate": 0.01},
        performance_threshold=0.05
    )
    
    baseline_model = ModelVersion(
        version_id="v0.9.0",
        model_type="regression",
        status=ModelStatus.PRODUCTION,
        created_at=datetime.now(timezone.utc),
        training_data_hash="def456",
        validation_metrics={"mae": 0.12, "rmse": 0.18},
        feature_importance={"feature1": 0.5, "feature2": 0.5},
        hyperparameters={"learning_rate": 0.005},
        performance_threshold=0.05
    )
    
    shadow_deployment = governance.deploy_shadow_mode(model_version, baseline_model, sample_data)
    print(f"✅ Shadow deployment: {shadow_deployment['deployment_id']}")
    
    print("✅ Adaptive governance testing completed")


if __name__ == "__main__":
    main()
