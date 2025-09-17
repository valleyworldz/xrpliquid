"""
Promotion Pipeline (Simplified)
Enforces dev → paper-trade → low-risk prod gates with auto-generated reports.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class PromotionStage(Enum):
    """Promotion stages."""
    DEVELOPMENT = "development"
    PAPER_TRADE = "paper_trade"
    LOW_RISK_PROD = "low_risk_prod"
    FULL_PROD = "full_prod"

class PromotionStatus(Enum):
    """Promotion status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class PromotionGate:
    """Individual promotion gate with criteria."""
    gate_name: str
    stage: PromotionStage
    criteria: Dict[str, Any]
    required_metrics: List[str]
    thresholds: Dict[str, float]
    duration_days: int
    min_observations: int

@dataclass
class PromotionResult:
    """Result of promotion gate evaluation."""
    gate_name: str
    stage: PromotionStage
    status: PromotionStatus
    score: float
    passed_criteria: List[str]
    failed_criteria: List[str]
    metrics: Dict[str, float]
    evaluation_date: datetime
    next_evaluation: datetime

class PromotionPipeline:
    """Main promotion pipeline manager."""
    
    def __init__(self, config_file: str = "config/promotion_pipeline.json"):
        self.config_file = Path(config_file)
        self.gates = {}
        self.promotion_history = []
        self.current_stage = PromotionStage.DEVELOPMENT
        
        # Load promotion gates
        self._load_promotion_gates()
    
    def _load_promotion_gates(self):
        """Load promotion gates from configuration."""
        
        if not self.config_file.exists():
            self._create_default_gates()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            for gate_config in config.get('gates', []):
                gate = PromotionGate(
                    gate_name=gate_config['name'],
                    stage=PromotionStage(gate_config['stage']),
                    criteria=gate_config['criteria'],
                    required_metrics=gate_config['required_metrics'],
                    thresholds=gate_config['thresholds'],
                    duration_days=gate_config['duration_days'],
                    min_observations=gate_config['min_observations']
                )
                self.gates[gate.gate_name] = gate
            
            logger.info(f"Loaded {len(self.gates)} promotion gates")
            
        except Exception as e:
            logger.error(f"Error loading promotion gates: {e}")
            self._create_default_gates()
    
    def _create_default_gates(self):
        """Create default promotion gates."""
        
        default_gates = [
            {
                'name': 'development_validation',
                'stage': 'development',
                'criteria': {
                    'backtest_sharpe': '>= 1.5',
                    'max_drawdown': '<= 0.1',
                    'win_rate': '>= 0.6'
                },
                'required_metrics': ['sharpe_ratio', 'max_drawdown', 'win_rate'],
                'thresholds': {
                    'sharpe_ratio': 1.5,
                    'max_drawdown': 0.1,
                    'win_rate': 0.6
                },
                'duration_days': 7,
                'min_observations': 100
            },
            {
                'name': 'paper_trade_validation',
                'stage': 'paper_trade',
                'criteria': {
                    'paper_sharpe': '>= 1.2',
                    'paper_drawdown': '<= 0.08',
                    'execution_quality': '>= 0.8'
                },
                'required_metrics': ['sharpe_ratio', 'max_drawdown', 'execution_quality'],
                'thresholds': {
                    'sharpe_ratio': 1.2,
                    'max_drawdown': 0.08,
                    'execution_quality': 0.8
                },
                'duration_days': 14,
                'min_observations': 200
            },
            {
                'name': 'low_risk_prod_validation',
                'stage': 'low_risk_prod',
                'criteria': {
                    'prod_sharpe': '>= 1.0',
                    'prod_drawdown': '<= 0.06',
                    'risk_controls': '>= 0.9'
                },
                'required_metrics': ['sharpe_ratio', 'max_drawdown', 'risk_controls'],
                'thresholds': {
                    'sharpe_ratio': 1.0,
                    'max_drawdown': 0.06,
                    'risk_controls': 0.9
                },
                'duration_days': 21,
                'min_observations': 500
            }
        ]
        
        for gate_config in default_gates:
            gate = PromotionGate(
                gate_name=gate_config['name'],
                stage=PromotionStage(gate_config['stage']),
                criteria=gate_config['criteria'],
                required_metrics=gate_config['required_metrics'],
                thresholds=gate_config['thresholds'],
                duration_days=gate_config['duration_days'],
                min_observations=gate_config['min_observations']
            )
            self.gates[gate.gate_name] = gate
        
        # Save default gates
        self._save_promotion_gates()
        
        logger.info("Created default promotion gates")
    
    def _save_promotion_gates(self):
        """Save promotion gates to configuration file."""
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'gates': [
                {
                    'name': gate.gate_name,
                    'stage': gate.stage.value,
                    'criteria': gate.criteria,
                    'required_metrics': gate.required_metrics,
                    'thresholds': gate.thresholds,
                    'duration_days': gate.duration_days,
                    'min_observations': gate.min_observations
                }
                for gate in self.gates.values()
            ],
            'metadata': {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def evaluate_promotion_gate(self, gate_name: str, metrics: Dict[str, float]) -> PromotionResult:
        """Evaluate a specific promotion gate."""
        
        if gate_name not in self.gates:
            raise ValueError(f"Gate {gate_name} not found")
        
        gate = self.gates[gate_name]
        
        # Check if we have enough observations
        if len(metrics) < gate.min_observations:
            return PromotionResult(
                gate_name=gate_name,
                stage=gate.stage,
                status=PromotionStatus.BLOCKED,
                score=0.0,
                passed_criteria=[],
                failed_criteria=[f"insufficient_observations: {len(metrics)} < {gate.min_observations}"],
                metrics=metrics,
                evaluation_date=datetime.now(),
                next_evaluation=datetime.now() + timedelta(days=1)
            )
        
        # Evaluate each criterion
        passed_criteria = []
        failed_criteria = []
        total_score = 0.0
        
        for criterion, threshold in gate.thresholds.items():
            if criterion in metrics:
                actual_value = metrics[criterion]
                
                # Check if criterion is met
                if self._evaluate_criterion(actual_value, threshold, gate.criteria.get(criterion, '>=')):
                    passed_criteria.append(criterion)
                    total_score += 1.0
                else:
                    failed_criteria.append(f"{criterion}: {actual_value:.4f} vs {threshold:.4f}")
            else:
                failed_criteria.append(f"missing_metric: {criterion}")
        
        # Calculate overall score
        score = total_score / len(gate.thresholds) if gate.thresholds else 0.0
        
        # Determine status
        if score >= 0.8:  # 80% of criteria must pass
            status = PromotionStatus.PASSED
        elif score >= 0.6:  # 60% threshold for partial pass
            status = PromotionStatus.IN_PROGRESS
        else:
            status = PromotionStatus.FAILED
        
        # Calculate next evaluation date
        next_evaluation = datetime.now() + timedelta(days=gate.duration_days)
        
        result = PromotionResult(
            gate_name=gate_name,
            stage=gate.stage,
            status=status,
            score=score,
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            metrics=metrics,
            evaluation_date=datetime.now(),
            next_evaluation=next_evaluation
        )
        
        # Store result
        self.promotion_history.append(result)
        
        return result
    
    def _evaluate_criterion(self, actual_value: float, threshold: float, operator: str) -> bool:
        """Evaluate a single criterion."""
        
        if operator == '>=':
            return actual_value >= threshold
        elif operator == '>':
            return actual_value > threshold
        elif operator == '<=':
            return actual_value <= threshold
        elif operator == '<':
            return actual_value < threshold
        elif operator == '==':
            return abs(actual_value - threshold) < 1e-6
        else:
            return actual_value >= threshold  # Default to >=
    
    def get_next_promotion_stage(self) -> Optional[PromotionStage]:
        """Get the next promotion stage to evaluate."""
        
        current_stage = self.current_stage
        
        if current_stage == PromotionStage.DEVELOPMENT:
            return PromotionStage.PAPER_TRADE
        elif current_stage == PromotionStage.PAPER_TRADE:
            return PromotionStage.LOW_RISK_PROD
        elif current_stage == PromotionStage.LOW_RISK_PROD:
            return PromotionStage.FULL_PROD
        else:
            return None  # Already at full production
    
    def promote_to_next_stage(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Attempt to promote to the next stage."""
        
        next_stage = self.get_next_promotion_stage()
        if not next_stage:
            return False, "Already at full production stage"
        
        # Find the gate for the next stage
        next_gate = None
        for gate in self.gates.values():
            if gate.stage == next_stage:
                next_gate = gate
                break
        
        if not next_gate:
            return False, f"No gate found for stage {next_stage.value}"
        
        # Evaluate the gate
        result = self.evaluate_promotion_gate(next_gate.gate_name, metrics)
        
        if result.status == PromotionStatus.PASSED:
            self.current_stage = next_stage
            message = f"Successfully promoted to {next_stage.value}"
            logger.info(message)
            return True, message
        else:
            message = f"Promotion to {next_stage.value} failed: {', '.join(result.failed_criteria)}"
            logger.warning(message)
            return False, message
    
    def generate_promotion_report(self) -> Dict:
        """Generate comprehensive promotion report."""
        
        if not self.promotion_history:
            return {'message': 'No promotion history available'}
        
        # Analyze recent promotions
        recent_results = self.promotion_history[-20:]  # Last 20 evaluations
        
        # Calculate success rates by stage
        stage_success_rates = {}
        for stage in PromotionStage:
            stage_results = [r for r in recent_results if r.stage == stage]
            if stage_results:
                success_rate = len([r for r in stage_results if r.status == PromotionStatus.PASSED]) / len(stage_results)
                stage_success_rates[stage.value] = success_rate
        
        # Calculate average scores by stage
        stage_avg_scores = {}
        for stage in PromotionStage:
            stage_results = [r for r in recent_results if r.stage == stage]
            if stage_results:
                avg_score = np.mean([r.score for r in stage_results])
                stage_avg_scores[stage.value] = avg_score
        
        # Get current status
        current_status = {
            'current_stage': self.current_stage.value,
            'next_stage': self.get_next_promotion_stage().value if self.get_next_promotion_stage() else None,
            'total_evaluations': len(self.promotion_history),
            'recent_evaluations': len(recent_results)
        }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_status': current_status,
            'stage_success_rates': stage_success_rates,
            'stage_avg_scores': stage_avg_scores,
            'recent_results': [
                {
                    'gate_name': r.gate_name,
                    'stage': r.stage.value,
                    'status': r.status.value,
                    'score': r.score,
                    'evaluation_date': r.evaluation_date.isoformat(),
                    'passed_criteria': r.passed_criteria,
                    'failed_criteria': r.failed_criteria
                }
                for r in recent_results[-10:]  # Last 10 results
            ]
        }
        
        return report
    
    def save_promotion_report(self, filepath: str = "reports/ml/promotion_report.json"):
        """Save promotion report to file."""
        
        report = self.generate_promotion_report()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Promotion report saved to {filepath}")

def main():
    """Demonstrate promotion pipeline."""
    
    # Initialize promotion pipeline
    pipeline = PromotionPipeline()
    
    # Simulate some promotion evaluations
    print("🧪 Testing Promotion Pipeline")
    print("=" * 50)
    
    # Development stage metrics
    dev_metrics = {
        'sharpe_ratio': 1.8,
        'max_drawdown': 0.08,
        'win_rate': 0.65
    }
    
    # Evaluate development gate
    dev_result = pipeline.evaluate_promotion_gate('development_validation', dev_metrics)
    print(f"Development Gate: {dev_result.status.value} (Score: {dev_result.score:.2f})")
    
    # Try to promote to paper trade
    success, message = pipeline.promote_to_next_stage(dev_metrics)
    print(f"Promotion to Paper Trade: {message}")
    
    # Paper trade stage metrics
    paper_metrics = {
        'sharpe_ratio': 1.4,
        'max_drawdown': 0.06,
        'execution_quality': 0.85
    }
    
    # Evaluate paper trade gate
    paper_result = pipeline.evaluate_promotion_gate('paper_trade_validation', paper_metrics)
    print(f"Paper Trade Gate: {paper_result.status.value} (Score: {paper_result.score:.2f})")
    
    # Try to promote to low risk prod
    success, message = pipeline.promote_to_next_stage(paper_metrics)
    print(f"Promotion to Low Risk Prod: {message}")
    
    # Low risk prod stage metrics
    prod_metrics = {
        'sharpe_ratio': 1.1,
        'max_drawdown': 0.05,
        'risk_controls': 0.92
    }
    
    # Evaluate low risk prod gate
    prod_result = pipeline.evaluate_promotion_gate('low_risk_prod_validation', prod_metrics)
    print(f"Low Risk Prod Gate: {prod_result.status.value} (Score: {prod_result.score:.2f})")
    
    # Try to promote to full prod
    success, message = pipeline.promote_to_next_stage(prod_metrics)
    print(f"Promotion to Full Prod: {message}")
    
    # Generate reports
    pipeline.save_promotion_report()
    
    print(f"\n📊 Promotion Pipeline Summary:")
    print(f"   Current Stage: {pipeline.current_stage.value}")
    print(f"   Total Evaluations: {len(pipeline.promotion_history)}")
    
    print("\n✅ Promotion pipeline demo completed")
    
    return 0

if __name__ == "__main__":
    exit(main())
