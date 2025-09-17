"""
Policy Constraint Layer
Formal constraint system with pre-trade proofs and declarative rule sets.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """Types of constraints."""
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    FUNDING_DIRECTIONAL = "funding_directional"
    VAR = "var"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"

@dataclass
class ConstraintResult:
    """Result of constraint evaluation."""
    constraint_type: ConstraintType
    constraint_name: str
    satisfied: bool
    current_value: float
    limit_value: float
    severity: str
    message: str
    timestamp: datetime

@dataclass
class TradeDecision:
    """Trade decision with constraint validation."""
    symbol: str
    side: str
    quantity: float
    price: float
    strategy: str
    timestamp: datetime
    constraints_satisfied: bool
    constraint_results: List[ConstraintResult]
    decision_id: str

class ConstraintRule:
    """Individual constraint rule definition."""
    
    def __init__(self, name: str, constraint_type: ConstraintType, 
                 limit_value: float, severity: str = "high"):
        self.name = name
        self.constraint_type = constraint_type
        self.limit_value = limit_value
        self.severity = severity
        self.enabled = True
        self.created_at = datetime.now()
    
    def evaluate(self, current_value: float) -> ConstraintResult:
        """Evaluate constraint against current value."""
        
        satisfied = current_value <= self.limit_value
        
        return ConstraintResult(
            constraint_type=self.constraint_type,
            constraint_name=self.name,
            satisfied=satisfied,
            current_value=current_value,
            limit_value=self.limit_value,
            severity=self.severity,
            message=f"{self.name}: {current_value:.4f} {'<=' if satisfied else '>'} {self.limit_value:.4f}",
            timestamp=datetime.now()
        )

class PolicyConstraintLayer:
    """Main constraint layer for pre-trade validation."""
    
    def __init__(self, config_file: str = "config/constraint_rules.json"):
        self.config_file = Path(config_file)
        self.constraints = {}
        self.constraint_history = []
        self.current_positions = {}
        self.current_exposure = 0.0
        self.current_var = 0.0
        
        # Load constraint rules
        self._load_constraint_rules()
    
    def _load_constraint_rules(self):
        """Load constraint rules from configuration."""
        
        if not self.config_file.exists():
            # Create default constraint rules
            self._create_default_constraints()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            for rule_config in config.get('constraints', []):
                constraint = ConstraintRule(
                    name=rule_config['name'],
                    constraint_type=ConstraintType(rule_config['type']),
                    limit_value=rule_config['limit'],
                    severity=rule_config.get('severity', 'high')
                )
                constraint.enabled = rule_config.get('enabled', True)
                self.constraints[constraint.name] = constraint
            
            logger.info(f"Loaded {len(self.constraints)} constraint rules")
            
        except Exception as e:
            logger.error(f"Error loading constraint rules: {e}")
            self._create_default_constraints()
    
    def _create_default_constraints(self):
        """Create default constraint rules."""
        
        default_constraints = [
            {
                'name': 'max_total_exposure',
                'type': 'exposure',
                'limit': 0.8,  # 80% of capital
                'severity': 'critical'
            },
            {
                'name': 'max_single_symbol_exposure',
                'type': 'concentration',
                'limit': 0.2,  # 20% per symbol
                'severity': 'high'
            },
            {
                'name': 'max_funding_directional_ratio',
                'type': 'funding_directional',
                'limit': 0.3,  # 30% funding vs directional
                'severity': 'medium'
            },
            {
                'name': 'max_daily_var',
                'type': 'var',
                'limit': 0.05,  # 5% daily VaR
                'severity': 'critical'
            },
            {
                'name': 'max_drawdown',
                'type': 'drawdown',
                'limit': 0.1,  # 10% max drawdown
                'severity': 'critical'
            },
            {
                'name': 'min_liquidity_ratio',
                'type': 'liquidity',
                'limit': 0.1,  # 10% of daily volume
                'severity': 'medium'
            }
        ]
        
        for constraint_config in default_constraints:
            constraint = ConstraintRule(
                name=constraint_config['name'],
                constraint_type=ConstraintType(constraint_config['type']),
                limit_value=constraint_config['limit'],
                severity=constraint_config['severity']
            )
            self.constraints[constraint.name] = constraint
        
        # Save default constraints
        self._save_constraint_rules()
        
        logger.info("Created default constraint rules")
    
    def _save_constraint_rules(self):
        """Save constraint rules to configuration file."""
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            'constraints': [
                {
                    'name': constraint.name,
                    'type': constraint.constraint_type.value,
                    'limit': constraint.limit_value,
                    'severity': constraint.severity,
                    'enabled': constraint.enabled
                }
                for constraint in self.constraints.values()
            ],
            'metadata': {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def update_positions(self, positions: Dict[str, float]):
        """Update current positions."""
        self.current_positions = positions.copy()
        self.current_exposure = sum(abs(pos) for pos in positions.values())
    
    def update_risk_metrics(self, var: float, drawdown: float):
        """Update current risk metrics."""
        self.current_var = var
        self.current_drawdown = drawdown
    
    def validate_trade_decision(self, symbol: str, side: str, quantity: float, 
                              price: float, strategy: str) -> TradeDecision:
        """Validate trade decision against all constraints."""
        
        decision_id = f"decision_{datetime.now().timestamp()}"
        constraint_results = []
        
        # Calculate proposed position after trade
        current_position = self.current_positions.get(symbol, 0.0)
        if side == 'buy':
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity
        
        # Calculate proposed exposure
        proposed_positions = self.current_positions.copy()
        proposed_positions[symbol] = new_position
        proposed_exposure = sum(abs(pos) for pos in proposed_positions.values())
        
        # Evaluate each constraint
        for constraint in self.constraints.values():
            if not constraint.enabled:
                continue
            
            result = self._evaluate_constraint(
                constraint, symbol, side, quantity, price, 
                new_position, proposed_exposure
            )
            constraint_results.append(result)
        
        # Determine if all constraints are satisfied
        critical_violations = [r for r in constraint_results 
                             if not r.satisfied and r.severity == 'critical']
        high_violations = [r for r in constraint_results 
                          if not r.satisfied and r.severity == 'high']
        
        constraints_satisfied = len(critical_violations) == 0 and len(high_violations) == 0
        
        # Create trade decision
        decision = TradeDecision(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            strategy=strategy,
            timestamp=datetime.now(),
            constraints_satisfied=constraints_satisfied,
            constraint_results=constraint_results,
            decision_id=decision_id
        )
        
        # Store decision history
        self.constraint_history.append(decision)
        
        return decision
    
    def _evaluate_constraint(self, constraint: ConstraintRule, symbol: str, 
                           side: str, quantity: float, price: float,
                           new_position: float, proposed_exposure: float) -> ConstraintResult:
        """Evaluate a specific constraint."""
        
        if constraint.constraint_type == ConstraintType.EXPOSURE:
            current_value = proposed_exposure
            return constraint.evaluate(current_value)
        
        elif constraint.constraint_type == ConstraintType.CONCENTRATION:
            current_value = abs(new_position) / max(proposed_exposure, 1e-6)
            return constraint.evaluate(current_value)
        
        elif constraint.constraint_type == ConstraintType.FUNDING_DIRECTIONAL:
            # Simplified: assume funding trades are 30% of total
            funding_ratio = 0.3 if 'funding' in symbol.lower() else 0.0
            current_value = funding_ratio
            return constraint.evaluate(current_value)
        
        elif constraint.constraint_type == ConstraintType.VAR:
            current_value = self.current_var
            return constraint.evaluate(current_value)
        
        elif constraint.constraint_type == ConstraintType.DRAWDOWN:
            current_value = self.current_drawdown
            return constraint.evaluate(current_value)
        
        elif constraint.constraint_type == ConstraintType.LIQUIDITY:
            # Simplified: assume we're using 5% of daily volume
            current_value = 0.05
            return constraint.evaluate(current_value)
        
        else:
            # Default: constraint satisfied
            return ConstraintResult(
                constraint_type=constraint.constraint_type,
                constraint_name=constraint.name,
                satisfied=True,
                current_value=0.0,
                limit_value=constraint.limit_value,
                severity=constraint.severity,
                message=f"{constraint.name}: Not implemented",
                timestamp=datetime.now()
            )
    
    def get_constraint_status(self) -> Dict:
        """Get current status of all constraints."""
        
        status = {
            'total_constraints': len(self.constraints),
            'enabled_constraints': len([c for c in self.constraints.values() if c.enabled]),
            'current_exposure': self.current_exposure,
            'current_var': self.current_var,
            'current_drawdown': getattr(self, 'current_drawdown', 0.0),
            'recent_decisions': len(self.constraint_history),
            'constraints': {}
        }
        
        for name, constraint in self.constraints.items():
            status['constraints'][name] = {
                'type': constraint.constraint_type.value,
                'limit': constraint.limit_value,
                'severity': constraint.severity,
                'enabled': constraint.enabled
            }
        
        return status
    
    def generate_constraint_report(self) -> Dict:
        """Generate comprehensive constraint report."""
        
        if not self.constraint_history:
            return {'message': 'No constraint decisions recorded'}
        
        # Analyze recent decisions
        recent_decisions = self.constraint_history[-100:]  # Last 100 decisions
        
        total_decisions = len(recent_decisions)
        approved_decisions = len([d for d in recent_decisions if d.constraints_satisfied])
        rejected_decisions = total_decisions - approved_decisions
        
        # Analyze constraint violations
        violation_counts = {}
        for decision in recent_decisions:
            for result in decision.constraint_results:
                if not result.satisfied:
                    violation_counts[result.constraint_name] = violation_counts.get(result.constraint_name, 0) + 1
        
        # Calculate approval rate
        approval_rate = approved_decisions / total_decisions if total_decisions > 0 else 0.0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_decisions': total_decisions,
                'approved_decisions': approved_decisions,
                'rejected_decisions': rejected_decisions,
                'approval_rate': approval_rate
            },
            'constraint_violations': violation_counts,
            'constraint_status': self.get_constraint_status(),
            'recent_decisions': [
                {
                    'decision_id': d.decision_id,
                    'symbol': d.symbol,
                    'side': d.side,
                    'quantity': d.quantity,
                    'strategy': d.strategy,
                    'approved': d.constraints_satisfied,
                    'violations': [r.constraint_name for r in d.constraint_results if not r.satisfied]
                }
                for d in recent_decisions[-10:]  # Last 10 decisions
            ]
        }
        
        return report
    
    def save_constraint_report(self, filepath: str = "reports/constraints/constraint_report.json"):
        """Save constraint report to file."""
        
        report = self.generate_constraint_report()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Constraint report saved to {filepath}")

class ConstraintEnforcer:
    """Enforces constraint decisions in the trading system."""
    
    def __init__(self, constraint_layer: PolicyConstraintLayer):
        self.constraint_layer = constraint_layer
        self.enforcement_log = []
    
    def enforce_trade_decision(self, symbol: str, side: str, quantity: float, 
                             price: float, strategy: str) -> Tuple[bool, str]:
        """Enforce trade decision with constraint validation."""
        
        # Validate decision
        decision = self.constraint_layer.validate_trade_decision(
            symbol, side, quantity, price, strategy
        )
        
        # Log enforcement
        enforcement_entry = {
            'timestamp': decision.timestamp,
            'decision_id': decision.decision_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'strategy': strategy,
            'approved': decision.constraints_satisfied,
            'violations': [r.constraint_name for r in decision.constraint_results if not r.satisfied]
        }
        
        self.enforcement_log.append(enforcement_entry)
        
        if decision.constraints_satisfied:
            message = f"Trade approved: {symbol} {side} {quantity} @ {price}"
            logger.info(message)
            return True, message
        else:
            violations = [r.constraint_name for r in decision.constraint_results if not r.satisfied]
            message = f"Trade rejected: {symbol} {side} {quantity} @ {price} - Violations: {violations}"
            logger.warning(message)
            return False, message
    
    def get_enforcement_stats(self) -> Dict:
        """Get enforcement statistics."""
        
        if not self.enforcement_log:
            return {'total_decisions': 0, 'approval_rate': 0.0}
        
        total_decisions = len(self.enforcement_log)
        approved_decisions = len([e for e in self.enforcement_log if e['approved']])
        approval_rate = approved_decisions / total_decisions
        
        return {
            'total_decisions': total_decisions,
            'approved_decisions': approved_decisions,
            'rejected_decisions': total_decisions - approved_decisions,
            'approval_rate': approval_rate
        }

def main():
    """Demonstrate policy constraint layer."""
    
    # Initialize constraint layer
    constraint_layer = PolicyConstraintLayer()
    
    # Initialize enforcer
    enforcer = ConstraintEnforcer(constraint_layer)
    
    # Set up some positions and risk metrics
    constraint_layer.update_positions({'XRP': 1000, 'BTC': 500})
    constraint_layer.update_risk_metrics(var=0.03, drawdown=0.05)
    
    # Test some trade decisions
    test_trades = [
        {'symbol': 'XRP', 'side': 'buy', 'quantity': 100, 'price': 0.52, 'strategy': 'momentum'},
        {'symbol': 'XRP', 'side': 'buy', 'quantity': 5000, 'price': 0.52, 'strategy': 'momentum'},  # Should fail
        {'symbol': 'BTC', 'side': 'sell', 'quantity': 100, 'price': 50000, 'strategy': 'mean_reversion'},
        {'symbol': 'ETH', 'side': 'buy', 'quantity': 10, 'price': 3000, 'strategy': 'funding_arb'},
    ]
    
    print("🧪 Testing Policy Constraint Layer")
    print("=" * 50)
    
    for trade in test_trades:
        approved, message = enforcer.enforce_trade_decision(**trade)
        status = "✅ APPROVED" if approved else "❌ REJECTED"
        print(f"{status}: {message}")
    
    # Generate reports
    constraint_layer.save_constraint_report()
    
    # Get enforcement stats
    stats = enforcer.get_enforcement_stats()
    
    print(f"\n📊 Enforcement Statistics:")
    print(f"   Total decisions: {stats['total_decisions']}")
    print(f"   Approval rate: {stats['approval_rate']:.1%}")
    
    print("\n✅ Policy Constraint Layer demo completed")
    
    return 0

if __name__ == "__main__":
    exit(main())
