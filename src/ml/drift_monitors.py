"""
Drift Monitors with Auto De-risk
Monitors data drift and concept drift with automatic risk reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy import stats
from sklearn.metrics import mutual_info_score

logger = logging.getLogger(__name__)

@dataclass
class DriftAlert:
    """Drift alert with severity and action taken."""
    timestamp: datetime
    drift_type: str
    feature_name: str
    severity: str
    drift_score: float
    threshold: float
    action_taken: str
    details: Dict

class DataDriftMonitor:
    """Monitors data drift using statistical tests."""
    
    def __init__(self, reference_data: pd.DataFrame, 
                 drift_threshold: float = 0.1,
                 significance_level: float = 0.05):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.drift_alerts = []
        self.feature_stats = {}
        
        # Compute reference statistics
        self._compute_reference_stats()
    
    def _compute_reference_stats(self):
        """Compute reference statistics for all features."""
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ['float64', 'int64']:
                self.feature_stats[column] = {
                    'mean': self.reference_data[column].mean(),
                    'std': self.reference_data[column].std(),
                    'skewness': self.reference_data[column].skew(),
                    'kurtosis': self.reference_data[column].kurtosis(),
                    'percentiles': self.reference_data[column].quantile([0.25, 0.5, 0.75]).to_dict()
                }
    
    def detect_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """Detect data drift in current data."""
        alerts = []
        
        for column in current_data.columns:
            if column not in self.feature_stats:
                continue
            
            # Skip if not enough data
            if len(current_data[column].dropna()) < 30:
                continue
            
            # Detect drift using multiple methods
            drift_score, drift_type = self._compute_drift_score(
                self.reference_data[column], 
                current_data[column]
            )
            
            if drift_score > self.drift_threshold:
                severity = self._determine_severity(drift_score)
                action = self._determine_action(severity, column)
                
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type=drift_type,
                    feature_name=column,
                    severity=severity,
                    drift_score=drift_score,
                    threshold=self.drift_threshold,
                    action_taken=action,
                    details={
                        'reference_mean': self.feature_stats[column]['mean'],
                        'current_mean': current_data[column].mean(),
                        'reference_std': self.feature_stats[column]['std'],
                        'current_std': current_data[column].std()
                    }
                )
                
                alerts.append(alert)
                self.drift_alerts.append(alert)
        
        return alerts
    
    def _compute_drift_score(self, reference: pd.Series, current: pd.Series) -> Tuple[float, str]:
        """Compute drift score using multiple statistical tests."""
        
        # Remove NaN values
        reference = reference.dropna()
        current = current.dropna()
        
        if len(reference) < 30 or len(current) < 30:
            return 0.0, "insufficient_data"
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
        ks_drift = 1 - ks_pvalue
        
        # Population Stability Index (PSI)
        psi_score = self._compute_psi(reference, current)
        
        # Wasserstein distance (normalized)
        wasserstein_dist = stats.wasserstein_distance(reference, current)
        wasserstein_drift = min(wasserstein_dist / reference.std(), 1.0)
        
        # Combine scores
        drift_score = max(ks_drift, psi_score, wasserstein_drift)
        
        # Determine drift type
        if psi_score > 0.2:
            drift_type = "distribution_shift"
        elif ks_drift > 0.1:
            drift_type = "statistical_drift"
        else:
            drift_type = "wasserstein_drift"
        
        return drift_score, drift_type
    
    def _compute_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """Compute Population Stability Index."""
        
        # Create bins
        bins = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            11
        )
        
        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bins)
        
        # Normalize to probabilities
        ref_prob = ref_hist / len(reference)
        curr_prob = curr_hist / len(current)
        
        # Compute PSI
        psi = 0.0
        for i in range(len(ref_prob)):
            if ref_prob[i] > 0 and curr_prob[i] > 0:
                psi += (curr_prob[i] - ref_prob[i]) * np.log(curr_prob[i] / ref_prob[i])
        
        return psi
    
    def _determine_severity(self, drift_score: float) -> str:
        """Determine severity level based on drift score."""
        if drift_score > 0.5:
            return "critical"
        elif drift_score > 0.3:
            return "high"
        elif drift_score > 0.1:
            return "medium"
        else:
            return "low"
    
    def _determine_action(self, severity: str, feature_name: str) -> str:
        """Determine action to take based on severity."""
        if severity == "critical":
            return "pause_feature"
        elif severity == "high":
            return "reduce_weight"
        elif severity == "medium":
            return "increase_monitoring"
        else:
            return "log_only"

class ConceptDriftMonitor:
    """Monitors concept drift in model performance."""
    
    def __init__(self, window_size: int = 100, 
                 performance_threshold: float = 0.1):
        self.window_size = window_size
        self.performance_threshold = performance_threshold
        self.performance_history = []
        self.drift_alerts = []
    
    def add_performance_measure(self, timestamp: datetime, 
                              performance_metrics: Dict[str, float]):
        """Add performance measurement."""
        
        self.performance_history.append({
            'timestamp': timestamp,
            'metrics': performance_metrics
        })
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size * 2:
            self.performance_history = self.performance_history[-self.window_size:]
    
    def detect_concept_drift(self) -> List[DriftAlert]:
        """Detect concept drift in performance."""
        alerts = []
        
        if len(self.performance_history) < self.window_size:
            return alerts
        
        # Get recent and historical performance
        recent_data = self.performance_history[-self.window_size:]
        historical_data = self.performance_history[-self.window_size*2:-self.window_size]
        
        # Check each performance metric
        for metric_name in recent_data[0]['metrics'].keys():
            recent_values = [d['metrics'][metric_name] for d in recent_data]
            historical_values = [d['metrics'][metric_name] for d in historical_data]
            
            # Compute performance degradation
            recent_mean = np.mean(recent_values)
            historical_mean = np.mean(historical_values)
            
            if historical_mean != 0:
                degradation = (historical_mean - recent_mean) / abs(historical_mean)
            else:
                degradation = 0.0
            
            if degradation > self.performance_threshold:
                severity = self._determine_severity(degradation)
                action = self._determine_action(severity, metric_name)
                
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type="concept_drift",
                    feature_name=metric_name,
                    severity=severity,
                    drift_score=degradation,
                    threshold=self.performance_threshold,
                    action_taken=action,
                    details={
                        'recent_mean': recent_mean,
                        'historical_mean': historical_mean,
                        'degradation': degradation
                    }
                )
                
                alerts.append(alert)
                self.drift_alerts.append(alert)
        
        return alerts
    
    def _determine_severity(self, degradation: float) -> str:
        """Determine severity based on performance degradation."""
        if degradation > 0.5:
            return "critical"
        elif degradation > 0.3:
            return "high"
        elif degradation > 0.1:
            return "medium"
        else:
            return "low"
    
    def _determine_action(self, severity: str, metric_name: str) -> str:
        """Determine action based on severity and metric."""
        if severity == "critical":
            return "pause_strategy"
        elif severity == "high":
            return "reduce_position_size"
        elif severity == "medium":
            return "increase_monitoring"
        else:
            return "log_only"

class AutoDeRiskManager:
    """Manages automatic risk reduction based on drift alerts."""
    
    def __init__(self):
        self.drift_alerts = []
        self.risk_actions = []
        self.current_risk_level = "normal"
        self.position_multiplier = 1.0
        self.feature_weights = {}
        self.paused_features = set()
        self.paused_strategies = set()
    
    def process_drift_alerts(self, alerts: List[DriftAlert]):
        """Process drift alerts and take appropriate actions."""
        
        for alert in alerts:
            self.drift_alerts.append(alert)
            
            # Take action based on alert
            action_result = self._take_action(alert)
            
            # Log action
            self.risk_actions.append({
                'timestamp': alert.timestamp,
                'alert': alert,
                'action_result': action_result
            })
            
            logger.info(f"Processed drift alert: {alert.feature_name} - {alert.action_taken}")
    
    def _take_action(self, alert: DriftAlert) -> Dict:
        """Take appropriate action based on drift alert."""
        
        action_result = {
            'action': alert.action_taken,
            'success': True,
            'details': {}
        }
        
        if alert.action_taken == "pause_feature":
            self.paused_features.add(alert.feature_name)
            action_result['details']['paused_features'] = list(self.paused_features)
            
        elif alert.action_taken == "reduce_weight":
            # Reduce feature weight by 50%
            current_weight = self.feature_weights.get(alert.feature_name, 1.0)
            self.feature_weights[alert.feature_name] = current_weight * 0.5
            action_result['details']['new_weight'] = self.feature_weights[alert.feature_name]
            
        elif alert.action_taken == "reduce_position_size":
            # Reduce position size by 25%
            self.position_multiplier *= 0.75
            action_result['details']['new_multiplier'] = self.position_multiplier
            
        elif alert.action_taken == "pause_strategy":
            # Pause affected strategy
            strategy_name = f"strategy_{alert.feature_name}"
            self.paused_strategies.add(strategy_name)
            action_result['details']['paused_strategies'] = list(self.paused_strategies)
            
        elif alert.action_taken == "increase_monitoring":
            # Increase monitoring frequency
            action_result['details']['monitoring_increased'] = True
            
        return action_result
    
    def get_risk_status(self) -> Dict:
        """Get current risk status and actions taken."""
        
        # Determine overall risk level
        critical_alerts = [a for a in self.drift_alerts if a.severity == "critical"]
        high_alerts = [a for a in self.drift_alerts if a.severity == "high"]
        
        if critical_alerts:
            self.current_risk_level = "critical"
        elif high_alerts:
            self.current_risk_level = "high"
        else:
            self.current_risk_level = "normal"
        
        return {
            'current_risk_level': self.current_risk_level,
            'position_multiplier': self.position_multiplier,
            'paused_features': list(self.paused_features),
            'paused_strategies': list(self.paused_strategies),
            'feature_weights': self.feature_weights,
            'total_alerts': len(self.drift_alerts),
            'critical_alerts': len(critical_alerts),
            'high_alerts': len(high_alerts),
            'recent_actions': self.risk_actions[-10:] if self.risk_actions else []
        }
    
    def reset_risk_level(self):
        """Reset risk level to normal (manual intervention)."""
        self.current_risk_level = "normal"
        self.position_multiplier = 1.0
        self.feature_weights = {}
        self.paused_features.clear()
        self.paused_strategies.clear()
        
        logger.info("Risk level reset to normal")

class DriftMonitorSystem:
    """Complete drift monitoring system."""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.data_drift_monitor = DataDriftMonitor(reference_data)
        self.concept_drift_monitor = ConceptDriftMonitor()
        self.auto_de_risk_manager = AutoDeRiskManager()
        
        self.monitoring_active = True
        self.last_check = datetime.now()
    
    def check_drift(self, current_data: pd.DataFrame, 
                   performance_metrics: Dict[str, float] = None) -> Dict:
        """Check for both data and concept drift."""
        
        if not self.monitoring_active:
            return {"status": "monitoring_disabled"}
        
        # Check data drift
        data_drift_alerts = self.data_drift_monitor.detect_drift(current_data)
        
        # Check concept drift
        concept_drift_alerts = []
        if performance_metrics:
            self.concept_drift_monitor.add_performance_measure(
                datetime.now(), performance_metrics
            )
            concept_drift_alerts = self.concept_drift_monitor.detect_concept_drift()
        
        # Process all alerts
        all_alerts = data_drift_alerts + concept_drift_alerts
        self.auto_de_risk_manager.process_drift_alerts(all_alerts)
        
        # Get risk status
        risk_status = self.auto_de_risk_manager.get_risk_status()
        
        self.last_check = datetime.now()
        
        return {
            'timestamp': self.last_check.isoformat(),
            'data_drift_alerts': len(data_drift_alerts),
            'concept_drift_alerts': len(concept_drift_alerts),
            'total_alerts': len(all_alerts),
            'risk_status': risk_status,
            'alerts': [
                {
                    'type': alert.drift_type,
                    'feature': alert.feature_name,
                    'severity': alert.severity,
                    'score': alert.drift_score,
                    'action': alert.action_taken
                }
                for alert in all_alerts
            ]
        }
    
    def save_drift_report(self, filepath: str = "reports/ml/drift_report.json"):
        """Save comprehensive drift report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': {
                'active': self.monitoring_active,
                'last_check': self.last_check.isoformat()
            },
            'data_drift_summary': {
                'total_alerts': len(self.data_drift_monitor.drift_alerts),
                'features_monitored': len(self.data_drift_monitor.feature_stats)
            },
            'concept_drift_summary': {
                'total_alerts': len(self.concept_drift_monitor.drift_alerts),
                'performance_history_length': len(self.concept_drift_monitor.performance_history)
            },
            'risk_status': self.auto_de_risk_manager.get_risk_status(),
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'type': alert.drift_type,
                    'feature': alert.feature_name,
                    'severity': alert.severity,
                    'score': alert.drift_score,
                    'action': alert.action_taken
                }
                for alert in self.auto_de_risk_manager.drift_alerts[-20:]
            ]
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Drift report saved to {filepath}")

def main():
    """Demonstrate drift monitoring system."""
    
    # Create sample reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'price': np.random.normal(0.52, 0.01, 1000),
        'volume': np.random.lognormal(8, 1, 1000),
        'spread': np.random.exponential(0.001, 1000)
    })
    
    # Initialize drift monitoring system
    drift_system = DriftMonitorSystem(reference_data)
    
    # Simulate some drift scenarios
    scenarios = [
        {
            'name': 'Normal data',
            'data': pd.DataFrame({
                'price': np.random.normal(0.52, 0.01, 100),
                'volume': np.random.lognormal(8, 1, 100),
                'spread': np.random.exponential(0.001, 100)
            }),
            'performance': {'sharpe_ratio': 1.8, 'max_drawdown': 0.03}
        },
        {
            'name': 'Price drift',
            'data': pd.DataFrame({
                'price': np.random.normal(0.55, 0.01, 100),  # Price drift
                'volume': np.random.lognormal(8, 1, 100),
                'spread': np.random.exponential(0.001, 100)
            }),
            'performance': {'sharpe_ratio': 1.5, 'max_drawdown': 0.04}
        },
        {
            'name': 'Performance degradation',
            'data': pd.DataFrame({
                'price': np.random.normal(0.52, 0.01, 100),
                'volume': np.random.lognormal(8, 1, 100),
                'spread': np.random.exponential(0.001, 100)
            }),
            'performance': {'sharpe_ratio': 1.2, 'max_drawdown': 0.06}  # Performance drop
        }
    ]
    
    # Test each scenario
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        
        result = drift_system.check_drift(
            scenario['data'], 
            scenario['performance']
        )
        
        print(f"  Data drift alerts: {result['data_drift_alerts']}")
        print(f"  Concept drift alerts: {result['concept_drift_alerts']}")
        print(f"  Risk level: {result['risk_status']['current_risk_level']}")
        print(f"  Position multiplier: {result['risk_status']['position_multiplier']:.2f}")
    
    # Save comprehensive report
    drift_system.save_drift_report()
    
    print("\n✅ Drift monitoring system demo completed")
    print("   Drift report saved to reports/ml/drift_report.json")
    
    return 0

if __name__ == "__main__":
    exit(main())
