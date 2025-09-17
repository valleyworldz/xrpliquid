"""
Model Governance - Model Cards + Drift Timelines
Full audit trail of AI/ML decisions with model cards and drift monitoring
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle
import warnings
warnings.filterwarnings('ignore')

class ModelType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENSEMBLE = "ensemble"

class ModelStatus(Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class DriftType(Enum):
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    FEATURE_DRIFT = "feature_drift"

@dataclass
class ModelCard:
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_date: str
    last_updated: str
    description: str
    training_data: Dict[str, Any]
    features: List[str]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    fairness_metrics: Dict[str, float]
    limitations: List[str]
    use_cases: List[str]
    training_environment: Dict[str, str]
    model_hash: str
    dependencies: List[str]
    maintainer: str
    approval_status: str

@dataclass
class DriftEvent:
    event_id: str
    model_id: str
    drift_type: DriftType
    severity: float  # 0-1 scale
    detected_at: str
    features_affected: List[str]
    drift_score: float
    baseline_period: str
    current_period: str
    statistical_test: str
    p_value: float
    threshold: float
    action_taken: str
    resolved_at: Optional[str]

@dataclass
class ModelDecision:
    decision_id: str
    model_id: str
    timestamp: str
    input_features: Dict[str, float]
    prediction: Any
    confidence: float
    explanation: Dict[str, Any]
    context: Dict[str, Any]
    audit_trail: List[str]

class ModelGovernanceManager:
    """
    Manages model governance with model cards and drift monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_cards: Dict[str, ModelCard] = {}
        self.drift_events: List[DriftEvent] = []
        self.model_decisions: List[ModelDecision] = []
        self.drift_monitors: Dict[str, Any] = {}
        
        # Create reports directory
        self.reports_dir = Path("reports/model_governance")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def register_model(self, model_card: ModelCard):
        """Register a new model with governance"""
        try:
            # Calculate model hash
            model_card.model_hash = self._calculate_model_hash(model_card)
            
            # Store model card
            self.model_cards[model_card.model_id] = model_card
            
            # Initialize drift monitoring
            self._initialize_drift_monitoring(model_card.model_id)
            
            self.logger.info(f"✅ Registered model: {model_card.name} v{model_card.version}")
            
        except Exception as e:
            self.logger.error(f"❌ Model registration error: {e}")
    
    def _calculate_model_hash(self, model_card: ModelCard) -> str:
        """Calculate hash for model identification"""
        try:
            # Create hash from model metadata
            hash_input = f"{model_card.name}_{model_card.version}_{model_card.created_date}_{json.dumps(model_card.hyperparameters, sort_keys=True)}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            
        except Exception as e:
            self.logger.error(f"❌ Model hash calculation error: {e}")
            return "unknown"
    
    def _initialize_drift_monitoring(self, model_id: str):
        """Initialize drift monitoring for a model"""
        try:
            # Create drift monitor instance
            monitor = {
                "model_id": model_id,
                "baseline_data": None,
                "drift_threshold": 0.1,
                "monitoring_enabled": True,
                "last_check": None,
                "drift_history": []
            }
            
            self.drift_monitors[model_id] = monitor
            
        except Exception as e:
            self.logger.error(f"❌ Drift monitoring initialization error: {e}")
    
    def log_model_decision(self, decision: ModelDecision):
        """Log a model decision for audit trail"""
        try:
            # Validate model exists
            if decision.model_id not in self.model_cards:
                self.logger.error(f"❌ Model {decision.model_id} not found")
                return
            
            # Add to decisions log
            self.model_decisions.append(decision)
            
            # Keep only recent decisions (last 10000)
            if len(self.model_decisions) > 10000:
                self.model_decisions = self.model_decisions[-10000:]
            
            self.logger.debug(f"📝 Logged decision for model {decision.model_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Decision logging error: {e}")
    
    def check_for_drift(self, model_id: str, current_data: pd.DataFrame) -> Optional[DriftEvent]:
        """Check for drift in model performance or data"""
        try:
            if model_id not in self.drift_monitors:
                self.logger.error(f"❌ Drift monitor not found for model {model_id}")
                return None
            
            monitor = self.drift_monitors[model_id]
            
            if not monitor["monitoring_enabled"]:
                return None
            
            # Get baseline data
            baseline_data = monitor["baseline_data"]
            if baseline_data is None:
                # Set current data as baseline
                monitor["baseline_data"] = current_data.copy()
                monitor["last_check"] = datetime.now().isoformat()
                return None
            
            # Check for data drift
            drift_event = self._detect_data_drift(model_id, baseline_data, current_data)
            
            if drift_event:
                self.drift_events.append(drift_event)
                monitor["drift_history"].append(drift_event.event_id)
                
                # Update baseline if drift is significant
                if drift_event.severity > 0.5:
                    monitor["baseline_data"] = current_data.copy()
            
            monitor["last_check"] = datetime.now().isoformat()
            return drift_event
            
        except Exception as e:
            self.logger.error(f"❌ Drift check error: {e}")
            return None
    
    def _detect_data_drift(self, model_id: str, baseline_data: pd.DataFrame, current_data: pd.DataFrame) -> Optional[DriftEvent]:
        """Detect data drift between baseline and current data"""
        try:
            # Ensure same columns
            common_columns = list(set(baseline_data.columns) & set(current_data.columns))
            if not common_columns:
                return None
            
            baseline_subset = baseline_data[common_columns]
            current_subset = current_data[common_columns]
            
            # Calculate drift for each feature
            drift_scores = {}
            features_affected = []
            
            for column in common_columns:
                if baseline_subset[column].dtype in ['float64', 'int64']:
                    # Numerical drift detection (KS test)
                    drift_score = self._calculate_ks_drift(baseline_subset[column], current_subset[column])
                    drift_scores[column] = drift_score
                    
                    if drift_score > 0.1:  # Threshold for drift
                        features_affected.append(column)
            
            # Calculate overall drift severity
            if drift_scores:
                overall_drift = max(drift_scores.values())
                
                if overall_drift > 0.1:  # Drift threshold
                    return DriftEvent(
                        event_id=f"drift_{model_id}_{int(datetime.now().timestamp())}",
                        model_id=model_id,
                        drift_type=DriftType.DATA_DRIFT,
                        severity=overall_drift,
                        detected_at=datetime.now().isoformat(),
                        features_affected=features_affected,
                        drift_score=overall_drift,
                        baseline_period=f"{baseline_data.index[0]} to {baseline_data.index[-1]}",
                        current_period=f"{current_data.index[0]} to {current_data.index[-1]}",
                        statistical_test="Kolmogorov-Smirnov",
                        p_value=0.001,  # Simulated
                        threshold=0.1,
                        action_taken="Monitoring",
                        resolved_at=None
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Data drift detection error: {e}")
            return None
    
    def _calculate_ks_drift(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate Kolmogorov-Smirnov drift score"""
        try:
            # Remove NaN values
            baseline_clean = baseline.dropna()
            current_clean = current.dropna()
            
            if len(baseline_clean) == 0 or len(current_clean) == 0:
                return 0.0
            
            # Calculate KS statistic
            from scipy import stats
            
            ks_statistic, p_value = stats.ks_2samp(baseline_clean, current_clean)
            
            # Convert to drift score (0-1)
            drift_score = min(1.0, ks_statistic * 2)  # Scale KS statistic
            
            return drift_score
            
        except Exception as e:
            self.logger.error(f"❌ KS drift calculation error: {e}")
            return 0.0
    
    def update_model_performance(self, model_id: str, performance_metrics: Dict[str, float]):
        """Update model performance metrics"""
        try:
            if model_id not in self.model_cards:
                self.logger.error(f"❌ Model {model_id} not found")
                return
            
            # Update performance metrics
            self.model_cards[model_id].performance_metrics.update(performance_metrics)
            self.model_cards[model_id].last_updated = datetime.now().isoformat()
            
            # Check for performance drift
            self._check_performance_drift(model_id, performance_metrics)
            
            self.logger.info(f"📊 Updated performance for model {model_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Performance update error: {e}")
    
    def _check_performance_drift(self, model_id: str, current_metrics: Dict[str, float]):
        """Check for performance drift"""
        try:
            model_card = self.model_cards[model_id]
            baseline_metrics = model_card.performance_metrics
            
            # Check for significant performance degradation
            performance_drift_detected = False
            drift_details = []
            
            for metric, current_value in current_metrics.items():
                if metric in baseline_metrics:
                    baseline_value = baseline_metrics[metric]
                    
                    # Calculate relative change
                    if baseline_value != 0:
                        relative_change = abs(current_value - baseline_value) / abs(baseline_value)
                        
                        # Check for significant degradation (20% threshold)
                        if relative_change > 0.2:
                            performance_drift_detected = True
                            drift_details.append(f"{metric}: {baseline_value:.3f} -> {current_value:.3f}")
            
            if performance_drift_detected:
                drift_event = DriftEvent(
                    event_id=f"perf_drift_{model_id}_{int(datetime.now().timestamp())}",
                    model_id=model_id,
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=0.7,  # High severity for performance drift
                    detected_at=datetime.now().isoformat(),
                    features_affected=[],
                    drift_score=0.7,
                    baseline_period="Training period",
                    current_period="Current evaluation",
                    statistical_test="Performance comparison",
                    p_value=0.01,
                    threshold=0.2,
                    action_taken="Performance monitoring",
                    resolved_at=None
                )
                
                self.drift_events.append(drift_event)
                self.logger.warning(f"⚠️ Performance drift detected for model {model_id}: {drift_details}")
            
        except Exception as e:
            self.logger.error(f"❌ Performance drift check error: {e}")
    
    def generate_model_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive model report"""
        try:
            if model_id not in self.model_cards:
                return {"error": f"Model {model_id} not found"}
            
            model_card = self.model_cards[model_id]
            
            # Get model decisions
            model_decisions = [d for d in self.model_decisions if d.model_id == model_id]
            
            # Get drift events
            model_drift_events = [e for e in self.drift_events if e.model_id == model_id]
            
            # Calculate decision statistics
            decision_stats = self._calculate_decision_statistics(model_decisions)
            
            # Calculate drift statistics
            drift_stats = self._calculate_drift_statistics(model_drift_events)
            
            report = {
                "model_card": asdict(model_card),
                "decision_statistics": decision_stats,
                "drift_statistics": drift_stats,
                "recent_decisions": [asdict(d) for d in model_decisions[-10:]],
                "recent_drift_events": [asdict(e) for e in model_drift_events[-5:]],
                "governance_status": self._assess_governance_status(model_card, model_drift_events),
                "report_generated": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Model report generation error: {e}")
            return {"error": str(e)}
    
    def _calculate_decision_statistics(self, decisions: List[ModelDecision]) -> Dict[str, Any]:
        """Calculate statistics for model decisions"""
        try:
            if not decisions:
                return {"total_decisions": 0}
            
            # Calculate confidence statistics
            confidences = [d.confidence for d in decisions]
            
            # Calculate prediction distribution
            predictions = [str(d.prediction) for d in decisions]
            prediction_counts = {}
            for pred in predictions:
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            return {
                "total_decisions": len(decisions),
                "average_confidence": np.mean(confidences),
                "confidence_std": np.std(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences),
                "prediction_distribution": prediction_counts,
                "decision_frequency": len(decisions) / max(1, (datetime.now() - datetime.fromisoformat(decisions[0].timestamp)).days)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Decision statistics calculation error: {e}")
            return {"error": str(e)}
    
    def _calculate_drift_statistics(self, drift_events: List[DriftEvent]) -> Dict[str, Any]:
        """Calculate statistics for drift events"""
        try:
            if not drift_events:
                return {"total_drift_events": 0}
            
            # Group by drift type
            drift_by_type = {}
            for event in drift_events:
                drift_type = event.drift_type.value
                if drift_type not in drift_by_type:
                    drift_by_type[drift_type] = []
                drift_by_type[drift_type].append(event)
            
            # Calculate severity statistics
            severities = [e.severity for e in drift_events]
            
            return {
                "total_drift_events": len(drift_events),
                "drift_by_type": {k: len(v) for k, v in drift_by_type.items()},
                "average_severity": np.mean(severities),
                "max_severity": np.max(severities),
                "unresolved_events": len([e for e in drift_events if e.resolved_at is None]),
                "recent_drift_rate": len([e for e in drift_events if datetime.fromisoformat(e.detected_at) > datetime.now() - timedelta(days=30)])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Drift statistics calculation error: {e}")
            return {"error": str(e)}
    
    def _assess_governance_status(self, model_card: ModelCard, drift_events: List[DriftEvent]) -> Dict[str, Any]:
        """Assess overall governance status"""
        try:
            # Check model status
            status_ok = model_card.status in [ModelStatus.PRODUCTION, ModelStatus.STAGING]
            
            # Check for recent drift
            recent_drift = [e for e in drift_events if 
                          datetime.fromisoformat(e.detected_at) > datetime.now() - timedelta(days=7)]
            
            # Check performance metrics
            performance_ok = len(model_card.performance_metrics) > 0
            
            # Overall assessment
            governance_score = 0
            if status_ok:
                governance_score += 1
            if len(recent_drift) == 0:
                governance_score += 1
            if performance_ok:
                governance_score += 1
            
            return {
                "governance_score": governance_score / 3,
                "status": "GOOD" if governance_score >= 2 else "NEEDS_ATTENTION",
                "model_status_ok": status_ok,
                "no_recent_drift": len(recent_drift) == 0,
                "performance_tracked": performance_ok,
                "recommendations": self._generate_governance_recommendations(model_card, drift_events)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Governance status assessment error: {e}")
            return {"error": str(e)}
    
    def _generate_governance_recommendations(self, model_card: ModelCard, drift_events: List[DriftEvent]) -> List[str]:
        """Generate governance recommendations"""
        recommendations = []
        
        try:
            # Check model status
            if model_card.status == ModelStatus.DEVELOPMENT:
                recommendations.append("Consider moving model to validation stage")
            
            # Check for drift
            unresolved_drift = [e for e in drift_events if e.resolved_at is None]
            if unresolved_drift:
                recommendations.append(f"Address {len(unresolved_drift)} unresolved drift events")
            
            # Check performance metrics
            if len(model_card.performance_metrics) == 0:
                recommendations.append("Add performance metrics to model card")
            
            # Check fairness metrics
            if len(model_card.fairness_metrics) == 0:
                recommendations.append("Add fairness metrics to model card")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Recommendation generation error: {e}")
            return ["Check system logs for detailed analysis"]
    
    async def save_governance_report(self):
        """Save comprehensive governance report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "model_cards": {k: asdict(v) for k, v in self.model_cards.items()},
                "drift_events": [asdict(e) for e in self.drift_events],
                "total_decisions": len(self.model_decisions),
                "governance_summary": self._get_governance_summary()
            }
            
            # Convert enums to strings
            for model_id, model_card in report["model_cards"].items():
                model_card["model_type"] = model_card["model_type"].value
                model_card["status"] = model_card["status"].value
            
            for event in report["drift_events"]:
                event["drift_type"] = event["drift_type"].value
            
            report_file = self.reports_dir / f"governance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"💾 Governance report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save governance report: {e}")
    
    def _get_governance_summary(self) -> Dict[str, Any]:
        """Get governance summary"""
        try:
            total_models = len(self.model_cards)
            production_models = len([m for m in self.model_cards.values() if m.status == ModelStatus.PRODUCTION])
            recent_drift = len([e for e in self.drift_events if 
                              datetime.fromisoformat(e.detected_at) > datetime.now() - timedelta(days=30)])
            
            return {
                "total_models": total_models,
                "production_models": production_models,
                "total_drift_events": len(self.drift_events),
                "recent_drift_events": recent_drift,
                "total_decisions": len(self.model_decisions),
                "models_with_drift": len(set(e.model_id for e in self.drift_events)),
                "governance_compliance": "COMPLIANT" if recent_drift == 0 else "NEEDS_ATTENTION"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Governance summary error: {e}")
            return {"error": str(e)}

# Demo function
async def demo_model_governance():
    """Demo the model governance system"""
    print("🤖 Model Governance System Demo")
    print("=" * 50)
    
    # Create governance manager
    governance = ModelGovernanceManager()
    
    # Register sample models
    models = [
        ModelCard(
            model_id="momentum_classifier_v1",
            name="Momentum Classifier",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.PRODUCTION,
            created_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            description="Classifies market momentum for trading decisions",
            training_data={"samples": 10000, "features": 20, "period": "2024-01-01 to 2024-12-31"},
            features=["price_change", "volume_ratio", "rsi", "macd", "bollinger_position"],
            hyperparameters={"learning_rate": 0.01, "max_depth": 10, "n_estimators": 100},
            performance_metrics={"accuracy": 0.75, "precision": 0.72, "recall": 0.78, "f1_score": 0.75},
            fairness_metrics={"demographic_parity": 0.95, "equalized_odds": 0.92},
            limitations=["Limited to liquid markets", "Requires sufficient historical data"],
            use_cases=["Trading signal generation", "Risk assessment"],
            training_environment={"python": "3.9", "scikit-learn": "1.0.0", "pandas": "1.3.0"},
            model_hash="",
            dependencies=["numpy", "pandas", "scikit-learn"],
            maintainer="ML Team",
            approval_status="approved"
        ),
        ModelCard(
            model_id="risk_regressor_v1",
            name="Risk Regressor",
            version="1.0.0",
            model_type=ModelType.REGRESSION,
            status=ModelStatus.STAGING,
            created_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            description="Predicts portfolio risk metrics",
            training_data={"samples": 5000, "features": 15, "period": "2024-01-01 to 2024-12-31"},
            features=["volatility", "correlation", "liquidity", "market_cap", "sector"],
            hyperparameters={"learning_rate": 0.005, "max_depth": 8, "n_estimators": 200},
            performance_metrics={"mse": 0.02, "mae": 0.12, "r2_score": 0.85},
            fairness_metrics={},
            limitations=["Market regime dependent", "Requires real-time data"],
            use_cases=["Risk management", "Portfolio optimization"],
            training_environment={"python": "3.9", "xgboost": "1.5.0", "pandas": "1.3.0"},
            model_hash="",
            dependencies=["numpy", "pandas", "xgboost"],
            maintainer="Risk Team",
            approval_status="pending"
        )
    ]
    
    # Register models
    for model in models:
        governance.register_model(model)
    
    # Simulate model decisions
    print("📝 Simulating model decisions...")
    for i in range(100):
        decision = ModelDecision(
            decision_id=f"decision_{i}",
            model_id="momentum_classifier_v1",
            timestamp=datetime.now().isoformat(),
            input_features={"price_change": np.random.normal(0, 0.02), "volume_ratio": np.random.uniform(0.5, 2.0)},
            prediction=np.random.choice(["buy", "sell", "hold"]),
            confidence=np.random.uniform(0.6, 0.95),
            explanation={"feature_importance": {"price_change": 0.6, "volume_ratio": 0.4}},
            context={"market_regime": "normal", "volatility": "medium"},
            audit_trail=[f"Decision {i} generated"]
        )
        governance.log_model_decision(decision)
    
    # Simulate drift detection
    print("🔍 Simulating drift detection...")
    np.random.seed(42)
    
    # Create baseline data
    baseline_data = pd.DataFrame({
        "price_change": np.random.normal(0, 0.02, 1000),
        "volume_ratio": np.random.uniform(0.5, 2.0, 1000),
        "rsi": np.random.uniform(20, 80, 1000)
    })
    
    # Create current data with drift
    current_data = pd.DataFrame({
        "price_change": np.random.normal(0.01, 0.03, 1000),  # Slight drift
        "volume_ratio": np.random.uniform(0.3, 2.5, 1000),   # More drift
        "rsi": np.random.uniform(15, 85, 1000)               # Some drift
    })
    
    # Check for drift
    drift_event = governance.check_for_drift("momentum_classifier_v1", current_data)
    if drift_event:
        print(f"⚠️ Drift detected: {drift_event.drift_type.value} (severity: {drift_event.severity:.3f})")
    
    # Update model performance
    governance.update_model_performance("momentum_classifier_v1", {
        "accuracy": 0.73,  # Slight degradation
        "precision": 0.70,
        "recall": 0.75,
        "f1_score": 0.72
    })
    
    # Generate model reports
    print("📊 Generating model reports...")
    for model_id in governance.model_cards.keys():
        report = governance.generate_model_report(model_id)
        print(f"\n📋 Model Report: {model_id}")
        print(f"  Status: {report['governance_status']['status']}")
        print(f"  Governance Score: {report['governance_status']['governance_score']:.2f}")
        print(f"  Total Decisions: {report['decision_statistics']['total_decisions']}")
        if 'average_confidence' in report['decision_statistics']:
            print(f"  Average Confidence: {report['decision_statistics']['average_confidence']:.3f}")
        else:
            print(f"  Average Confidence: N/A")
        print(f"  Drift Events: {report['drift_statistics']['total_drift_events']}")
    
    # Save governance report
    await governance.save_governance_report()
    
    # Get governance summary
    summary = governance._get_governance_summary()
    print(f"\n📈 Governance Summary:")
    print(f"Total Models: {summary['total_models']}")
    print(f"Production Models: {summary['production_models']}")
    print(f"Total Drift Events: {summary['total_drift_events']}")
    print(f"Recent Drift Events: {summary['recent_drift_events']}")
    print(f"Total Decisions: {summary['total_decisions']}")
    print(f"Governance Compliance: {summary['governance_compliance']}")
    
    print("\n✅ Model Governance Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_model_governance())
