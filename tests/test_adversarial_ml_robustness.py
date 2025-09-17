"""
Adversarial ML Robustness Tests
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AdversarialMLTester:
    def __init__(self):
        self.test_results = []
        self.safe_baseline_actions = ['HOLD', 'CLOSE_POSITION', 'REDUCE_SIZE']
        
    def generate_nan_inf_inputs(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'all_nan_features',
                'features': {
                    'price': float('nan'),
                    'volume': float('nan'),
                    'rsi': float('nan')
                }
            },
            {
                'name': 'mixed_nan_inf',
                'features': {
                    'price': float('inf'),
                    'volume': float('-inf'),
                    'rsi': float('nan')
                }
            }
        ]
    
    def test_ml_model_robustness(self, model_predict_func, test_name: str = "ML Model") -> Dict[str, Any]:
        results = {
            'test_name': test_name,
            'total_tests': 0,
            'passed_tests': 0,
            'safe_responses': 0
        }
        
        all_inputs = self.generate_nan_inf_inputs()
        
        for input_data in all_inputs:
            results['total_tests'] += 1
            
            try:
                prediction = model_predict_func(input_data['features'])
                is_safe = self.is_safe_prediction(prediction)
                
                if is_safe:
                    results['safe_responses'] += 1
                    results['passed_tests'] += 1
            except Exception:
                pass
        
        return results
    
    def is_safe_prediction(self, prediction: Any) -> bool:
        if prediction is None:
            return True
        
        pred_str = str(prediction).upper()
        for safe_action in self.safe_baseline_actions:
            if safe_action in pred_str:
                return True
        return False

def mock_ml_predict(features: Dict[str, Any]) -> str:
    try:
        for key, value in features.items():
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    return "HOLD"
        return "BUY" if features.get('momentum', 0) > 0 else "SELL"
    except Exception:
        return "HOLD"

def test_adversarial_ml_suite():
    print("🛡️ Adversarial ML Robustness Test Suite")
    print("=" * 50)
    
    tester = AdversarialMLTester()
    ml_results = tester.test_ml_model_robustness(mock_ml_predict, "Mock ML Model")
    
    total_tests = ml_results['total_tests']
    total_passed = ml_results['passed_tests']
    total_safe = ml_results['safe_responses']
    
    print(f"📊 Overall Results:")
    print(f"Total Tests: {total_tests}")
    print(f"Total Passed: {total_passed}")
    print(f"Total Safe Responses: {total_safe}")
    print(f"Overall Pass Rate: {total_passed/total_tests:.2%}")
    print(f"Overall Safe Rate: {total_safe/total_tests:.2%}")
    
    assert total_passed/total_tests >= 0.8, "Overall pass rate below 80%"
    assert total_safe/total_tests >= 0.8, "Overall safe rate below 80%"
    
    print("✅ Adversarial ML Test Suite Complete")

if __name__ == "__main__":
    test_adversarial_ml_suite()
