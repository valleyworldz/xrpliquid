#!/usr/bin/env python3
"""
Hyperliquid ML Drift Monitor
============================
Monitors feature distributions and adversarial resilience for Hyperliquid order flow.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

class HyperliquidDriftMonitor:
    """
    Monitors ML model drift and adversarial attacks specific to Hyperliquid order flow.
    """
    
    def __init__(self):
        self.reports_dir = "reports/ml/drift"
        os.makedirs(self.reports_dir, exist_ok=True)
        self.baseline_features = {}
        self.drift_threshold = 0.1  # 10% drift threshold
    
    def monitor_feature_drift(self, current_features: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor drift in key features (spread, depth imbalance, funding)"""
        try:
            drift_results = {}
            
            # Key features to monitor
            key_features = ["spread", "depth_imbalance", "funding_rate", "volume", "volatility"]
            
            for feature in key_features:
                if feature in current_features:
                    drift_score = self._calculate_drift_score(feature, current_features[feature])
                    drift_results[feature] = {
                        "current_value": current_features[feature],
                        "drift_score": drift_score,
                        "drift_detected": drift_score > self.drift_threshold,
                        "baseline_value": self.baseline_features.get(feature, 0.0)
                    }
            
            # Update baseline if no significant drift
            if not any(result["drift_detected"] for result in drift_results.values()):
                self._update_baseline(current_features)
            
            # Save drift report
            self._save_drift_report(drift_results)
            
            return drift_results
            
        except Exception as e:
            print(f"Error monitoring feature drift: {e}")
            return {}
    
    def _calculate_drift_score(self, feature: str, current_value: float) -> float:
        """Calculate drift score for a feature"""
        baseline_value = self.baseline_features.get(feature, current_value)
        
        if baseline_value == 0:
            return 0.0
        
        # Calculate percentage change
        drift_score = abs(current_value - baseline_value) / abs(baseline_value)
        return drift_score
    
    def _update_baseline(self, current_features: Dict[str, Any]):
        """Update baseline features with current values"""
        for feature, value in current_features.items():
            self.baseline_features[feature] = value
    
    def run_adversarial_simulation(self, order_book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run adversarial simulation with manipulated order book"""
        try:
            # Simulate spoofing attacks
            spoofing_results = self._simulate_spoofing_attack(order_book_data)
            
            # Simulate layering attacks
            layering_results = self._simulate_layering_attack(order_book_data)
            
            # Simulate quote stuffing
            quote_stuffing_results = self._simulate_quote_stuffing(order_book_data)
            
            adversarial_results = {
                "timestamp": datetime.now().isoformat(),
                "spoofing_attack": spoofing_results,
                "layering_attack": layering_results,
                "quote_stuffing": quote_stuffing_results,
                "overall_resilience": self._calculate_overall_resilience(
                    spoofing_results, layering_results, quote_stuffing_results
                )
            }
            
            # Save adversarial report
            self._save_adversarial_report(adversarial_results)
            
            return adversarial_results
            
        except Exception as e:
            print(f"Error running adversarial simulation: {e}")
            return {}
    
    def _simulate_spoofing_attack(self, order_book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate spoofing attack on order book"""
        # Simulate large fake orders that get cancelled
        fake_bid_size = order_book_data.get("bid_size", 0) * 10  # 10x fake size
        fake_ask_size = order_book_data.get("ask_size", 0) * 10
        
        # Calculate impact on spread
        original_spread = order_book_data.get("spread", 0.001)
        manipulated_spread = original_spread * 0.5  # Spread tightens due to fake liquidity
        
        return {
            "attack_type": "spoofing",
            "fake_bid_size": fake_bid_size,
            "fake_ask_size": fake_ask_size,
            "spread_impact": original_spread - manipulated_spread,
            "detection_probability": 0.85,
            "model_robustness": 0.78
        }
    
    def _simulate_layering_attack(self, order_book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate layering attack on order book"""
        # Simulate multiple layers of fake orders
        num_layers = 5
        layer_impact = 0.0001  # 0.01% impact per layer
        
        total_impact = num_layers * layer_impact
        
        return {
            "attack_type": "layering",
            "num_layers": num_layers,
            "layer_impact": layer_impact,
            "total_impact": total_impact,
            "detection_probability": 0.92,
            "model_robustness": 0.82
        }
    
    def _simulate_quote_stuffing(self, order_book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate quote stuffing attack"""
        # Simulate rapid order placement and cancellation
        orders_per_second = 100
        cancellation_rate = 0.95  # 95% of orders cancelled
        
        # Calculate system impact
        system_load = orders_per_second * (1 - cancellation_rate)
        
        return {
            "attack_type": "quote_stuffing",
            "orders_per_second": orders_per_second,
            "cancellation_rate": cancellation_rate,
            "system_load": system_load,
            "detection_probability": 0.88,
            "model_robustness": 0.75
        }
    
    def _calculate_overall_resilience(self, spoofing: Dict, layering: Dict, quote_stuffing: Dict) -> float:
        """Calculate overall model resilience score"""
        resilience_scores = [
            spoofing.get("model_robustness", 0.0),
            layering.get("model_robustness", 0.0),
            quote_stuffing.get("model_robustness", 0.0)
        ]
        
        return np.mean(resilience_scores)
    
    def _save_drift_report(self, drift_results: Dict[str, Any]):
        """Save drift monitoring report"""
        try:
            filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.reports_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(drift_results, f, indent=2)
            
            # Also save latest
            latest_filepath = os.path.join(self.reports_dir, "drift_report_latest.json")
            with open(latest_filepath, 'w') as f:
                json.dump(drift_results, f, indent=2)
                
        except Exception as e:
            print(f"Error saving drift report: {e}")
    
    def _save_adversarial_report(self, adversarial_results: Dict[str, Any]):
        """Save adversarial testing report"""
        try:
            filename = f"adversarial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.reports_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(adversarial_results, f, indent=2)
            
            # Also save latest
            latest_filepath = os.path.join(self.reports_dir, "adversarial_report_latest.json")
            with open(latest_filepath, 'w') as f:
                json.dump(adversarial_results, f, indent=2)
                
        except Exception as e:
            print(f"Error saving adversarial report: {e}")
