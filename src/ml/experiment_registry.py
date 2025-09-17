"""
Experiment/Model Registry
Lightweight registry for tracking ML experiments with full traceability.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)

class ExperimentRegistry:
    """Registry for tracking ML experiments and model versions."""
    
    def __init__(self, registry_file: str = "data/ml/experiment_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load experiment registry from JSON."""
        if not self.registry_file.exists():
            return {
                "experiments": {},
                "models": {},
                "metadata": {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
            }
        
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading experiment registry: {e}")
            return {
                "experiments": {},
                "models": {},
                "metadata": {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
            }
    
    def _save_registry(self):
        """Save experiment registry to JSON."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.registry["metadata"]["last_updated"] = datetime.now().isoformat()
        
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_experiment(self, name: str, description: str, 
                          params: Dict, commit_sha: str,
                          dataset_version: str, metrics: Dict,
                          artifacts: List[str], status: str = "running") -> str:
        """Register a new experiment."""
        
        experiment_id = self._generate_experiment_id(name, params, commit_sha)
        
        experiment_data = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "params": params,
            "commit_sha": commit_sha,
            "dataset_version": dataset_version,
            "metrics": metrics,
            "artifacts": artifacts,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.registry["experiments"][experiment_id] = experiment_data
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered experiment: {name} ({experiment_id})")
        return experiment_id
    
    def _generate_experiment_id(self, name: str, params: Dict, commit_sha: str) -> str:
        """Generate unique experiment ID."""
        content = f"{name}_{json.dumps(params, sort_keys=True)}_{commit_sha}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def update_experiment_metrics(self, experiment_id: str, metrics: Dict) -> bool:
        """Update experiment metrics."""
        
        if experiment_id not in self.registry["experiments"]:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        self.registry["experiments"][experiment_id]["metrics"].update(metrics)
        self.registry["experiments"][experiment_id]["updated_at"] = datetime.now().isoformat()
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Updated metrics for experiment {experiment_id}")
        return True
    
    def promote_experiment_to_model(self, experiment_id: str, 
                                  model_name: str, promotion_reason: str) -> str:
        """Promote experiment to production model."""
        
        if experiment_id not in self.registry["experiments"]:
            logger.error(f"Experiment {experiment_id} not found")
            return None
        
        experiment = self.registry["experiments"][experiment_id]
        
        # Create model entry
        model_id = f"model_{experiment_id}"
        
        model_data = {
            "model_id": model_id,
            "model_name": model_name,
            "experiment_id": experiment_id,
            "promotion_reason": promotion_reason,
            "params": experiment["params"],
            "commit_sha": experiment["commit_sha"],
            "dataset_version": experiment["dataset_version"],
            "metrics": experiment["metrics"],
            "artifacts": experiment["artifacts"],
            "status": "active",
            "promoted_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        self.registry["models"][model_id] = model_data
        
        # Update experiment status
        self.registry["experiments"][experiment_id]["status"] = "promoted"
        self.registry["experiments"][experiment_id]["promoted_to"] = model_id
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Promoted experiment {experiment_id} to model {model_id}")
        return model_id
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment data by ID."""
        return self.registry["experiments"].get(experiment_id)
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model data by ID."""
        return self.registry["models"].get(model_id)
    
    def list_experiments(self, status: str = None) -> List[Dict]:
        """List experiments, optionally filtered by status."""
        experiments = list(self.registry["experiments"].values())
        
        if status:
            experiments = [exp for exp in experiments if exp["status"] == status]
        
        return experiments
    
    def list_models(self, status: str = None) -> List[Dict]:
        """List models, optionally filtered by status."""
        models = list(self.registry["models"].values())
        
        if status:
            models = [model for model in models if model["status"] == status]
        
        return models
    
    def generate_experiment_report(self) -> pd.DataFrame:
        """Generate experiment report as DataFrame."""
        
        experiments_data = []
        for exp_id, exp_data in self.registry["experiments"].items():
            experiments_data.append({
                "experiment_id": exp_id,
                "name": exp_data["name"],
                "status": exp_data["status"],
                "commit_sha": exp_data["commit_sha"],
                "dataset_version": exp_data["dataset_version"],
                "sharpe_ratio": exp_data["metrics"].get("sharpe_ratio", None),
                "sortino_ratio": exp_data["metrics"].get("sortino_ratio", None),
                "psr_confidence": exp_data["metrics"].get("psr_confidence", None),
                "max_drawdown": exp_data["metrics"].get("max_drawdown", None),
                "total_return": exp_data["metrics"].get("total_return", None),
                "created_at": exp_data["created_at"],
                "updated_at": exp_data["updated_at"]
            })
        
        df = pd.DataFrame(experiments_data)
        
        # Save to CSV
        csv_path = Path("reports/ml/experiment_report.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Experiment report saved to {csv_path}")
        return df
    
    def generate_model_report(self) -> pd.DataFrame:
        """Generate model report as DataFrame."""
        
        models_data = []
        for model_id, model_data in self.registry["models"].items():
            models_data.append({
                "model_id": model_id,
                "model_name": model_data["model_name"],
                "status": model_data["status"],
                "experiment_id": model_data["experiment_id"],
                "commit_sha": model_data["commit_sha"],
                "dataset_version": model_data["dataset_version"],
                "sharpe_ratio": model_data["metrics"].get("sharpe_ratio", None),
                "sortino_ratio": model_data["metrics"].get("sortino_ratio", None),
                "psr_confidence": model_data["metrics"].get("psr_confidence", None),
                "max_drawdown": model_data["metrics"].get("max_drawdown", None),
                "total_return": model_data["metrics"].get("total_return", None),
                "promoted_at": model_data["promoted_at"],
                "created_at": model_data["created_at"]
            })
        
        df = pd.DataFrame(models_data)
        
        # Save to CSV
        csv_path = Path("reports/ml/model_report.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Model report saved to {csv_path}")
        return df

class ModelVersionManager:
    """Manages model versions and deployments."""
    
    def __init__(self, experiment_registry: ExperimentRegistry):
        self.registry = experiment_registry
        self.active_models = {}
    
    def deploy_model(self, model_id: str, environment: str = "production") -> bool:
        """Deploy model to specified environment."""
        
        model_data = self.registry.get_model(model_id)
        if not model_data:
            logger.error(f"Model {model_id} not found")
            return False
        
        # Update model status
        model_data["status"] = "deployed"
        model_data["deployed_at"] = datetime.now().isoformat()
        model_data["environment"] = environment
        
        # Track active models
        self.active_models[environment] = model_id
        
        # Save registry
        self.registry._save_registry()
        
        logger.info(f"Deployed model {model_id} to {environment}")
        return True
    
    def rollback_model(self, environment: str, previous_model_id: str) -> bool:
        """Rollback to previous model version."""
        
        if environment not in self.active_models:
            logger.error(f"No active model in {environment}")
            return False
        
        current_model_id = self.active_models[environment]
        
        # Update current model status
        current_model = self.registry.get_model(current_model_id)
        if current_model:
            current_model["status"] = "rolled_back"
            current_model["rolled_back_at"] = datetime.now().isoformat()
        
        # Deploy previous model
        success = self.deploy_model(previous_model_id, environment)
        
        if success:
            logger.info(f"Rolled back {environment} from {current_model_id} to {previous_model_id}")
        
        return success
    
    def get_model_performance_history(self, model_id: str) -> List[Dict]:
        """Get performance history for a model."""
        
        model_data = self.registry.get_model(model_id)
        if not model_data:
            return []
        
        # Get experiment history
        experiment_id = model_data["experiment_id"]
        experiment = self.registry.get_experiment(experiment_id)
        
        if not experiment:
            return []
        
        # Build performance history
        history = [
            {
                "timestamp": experiment["created_at"],
                "stage": "experiment",
                "metrics": experiment["metrics"]
            },
            {
                "timestamp": model_data["promoted_at"],
                "stage": "promotion",
                "metrics": model_data["metrics"]
            }
        ]
        
        if "deployed_at" in model_data:
            history.append({
                "timestamp": model_data["deployed_at"],
                "stage": "deployment",
                "metrics": model_data["metrics"]
            })
        
        return history

def main():
    """Initialize experiment registry with sample data."""
    
    # Initialize registry
    registry = ExperimentRegistry()
    
    # Register sample experiments
    sample_experiments = [
        {
            "name": "xrp_momentum_v1",
            "description": "XRP momentum strategy with 5-minute lookback",
            "params": {
                "lookback_minutes": 5,
                "threshold": 0.001,
                "max_position": 0.1
            },
            "commit_sha": "abc123def456",
            "dataset_version": "2025-09-16",
            "metrics": {
                "sharpe_ratio": 1.85,
                "sortino_ratio": 2.12,
                "psr_confidence": 95.5,
                "max_drawdown": 0.032,
                "total_return": 0.124
            },
            "artifacts": [
                "models/xrp_momentum_v1.pkl",
                "reports/backtest_xrp_momentum_v1.html"
            ]
        },
        {
            "name": "xrp_mean_reversion_v2",
            "description": "XRP mean reversion with Bollinger Bands",
            "params": {
                "bb_period": 20,
                "bb_std": 2.0,
                "reversion_threshold": 0.5
            },
            "commit_sha": "def456ghi789",
            "dataset_version": "2025-09-16",
            "metrics": {
                "sharpe_ratio": 1.72,
                "sortino_ratio": 1.98,
                "psr_confidence": 92.3,
                "max_drawdown": 0.028,
                "total_return": 0.108
            },
            "artifacts": [
                "models/xrp_mean_reversion_v2.pkl",
                "reports/backtest_xrp_mean_reversion_v2.html"
            ]
        }
    ]
    
    # Register experiments
    experiment_ids = []
    for exp in sample_experiments:
        exp_id = registry.register_experiment(**exp)
        experiment_ids.append(exp_id)
    
    # Promote first experiment to model
    model_id = registry.promote_experiment_to_model(
        experiment_ids[0], 
        "xrp_momentum_prod", 
        "Best Sharpe ratio and PSR confidence"
    )
    
    # Deploy model
    version_manager = ModelVersionManager(registry)
    version_manager.deploy_model(model_id, "production")
    
    # Generate reports
    exp_report = registry.generate_experiment_report()
    model_report = registry.generate_model_report()
    
    print("✅ Experiment Registry initialized")
    print(f"   Registered experiments: {len(registry.list_experiments())}")
    print(f"   Active models: {len(registry.list_models())}")
    print(f"   Promoted model: {model_id}")
    print(f"   Experiment report: {len(exp_report)} rows")
    print(f"   Model report: {len(model_report)} rows")
    
    return 0

if __name__ == "__main__":
    exit(main())
