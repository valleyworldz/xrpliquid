"""
Contextual Bandits for Execution Choices
Bandit policy to pick order type conditioned on market context with off-policy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order execution types."""
    POST_ONLY = "post_only"
    TAKER = "taker"
    TWAP = "twap"
    ICEBERG = "iceberg"

@dataclass
class MarketContext:
    """Market context for bandit decision making."""
    spread_bps: float
    depth_ratio: float
    volatility: float
    urgency: float
    time_to_close: float
    volume_profile: float

@dataclass
class ExecutionResult:
    """Result of order execution."""
    order_type: OrderType
    fill_price: float
    fill_quantity: float
    slippage_bps: float
    fill_time_ms: float
    rebate_earned: float
    success: bool

class ContextualBandit:
    """Contextual bandit for execution choice optimization."""
    
    def __init__(self, n_arms: int = 4, context_dim: int = 6):
        self.n_arms = n_arms  # Number of order types
        self.context_dim = context_dim  # Dimension of context vector
        self.arm_names = [OrderType.POST_ONLY, OrderType.TAKER, OrderType.TWAP, OrderType.ICEBERG]
        
        # Bandit parameters
        self.alpha = 1.0  # Exploration parameter
        self.beta = 1.0   # Exploration parameter
        self.learning_rate = 0.01
        
        # Initialize arm statistics
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.arm_contexts = np.zeros((n_arms, context_dim))
        self.arm_reward_sums = np.zeros(n_arms)
        
        # Contextual features
        self.feature_weights = np.random.normal(0, 0.1, (n_arms, context_dim))
        self.feature_counts = np.zeros((n_arms, context_dim))
        
        # Safety constraints
        self.max_slippage_bps = 5.0
        self.min_fill_probability = 0.8
        self.max_urgency_threshold = 0.9
    
    def _context_to_features(self, context: MarketContext) -> np.ndarray:
        """Convert market context to feature vector."""
        return np.array([
            context.spread_bps,
            context.depth_ratio,
            context.volatility,
            context.urgency,
            context.time_to_close,
            context.volume_profile
        ])
    
    def _compute_expected_reward(self, arm: int, context: MarketContext) -> float:
        """Compute expected reward for an arm given context."""
        features = self._context_to_features(context)
        
        # Linear model: reward = w^T * features + bias
        linear_reward = np.dot(self.feature_weights[arm], features)
        
        # Add exploration bonus (UCB-style)
        if self.arm_counts[arm] > 0:
            exploration_bonus = self.alpha * np.sqrt(
                np.log(np.sum(self.arm_counts)) / self.arm_counts[arm]
            )
        else:
            exploration_bonus = self.alpha * 10.0  # High bonus for unexplored arms
        
        return linear_reward + exploration_bonus
    
    def _check_safety_constraints(self, arm: int, context: MarketContext) -> bool:
        """Check if arm satisfies safety constraints."""
        
        # Check urgency constraint
        if context.urgency > self.max_urgency_threshold:
            # High urgency: only allow taker orders
            return self.arm_names[arm] == OrderType.TAKER
        
        # Check spread constraint
        if context.spread_bps > 10.0:  # Wide spread
            # Wide spread: prefer post-only to avoid slippage
            return self.arm_names[arm] in [OrderType.POST_ONLY, OrderType.TWAP]
        
        # Check depth constraint
        if context.depth_ratio < 0.1:  # Low depth
            # Low depth: avoid large orders
            return self.arm_names[arm] in [OrderType.POST_ONLY, OrderType.ICEBERG]
        
        return True
    
    def select_arm(self, context: MarketContext) -> Tuple[int, OrderType]:
        """Select best arm given market context."""
        
        # Compute expected rewards for all arms
        expected_rewards = []
        valid_arms = []
        
        for arm in range(self.n_arms):
            if self._check_safety_constraints(arm, context):
                reward = self._compute_expected_reward(arm, context)
                expected_rewards.append(reward)
                valid_arms.append(arm)
            else:
                expected_rewards.append(-np.inf)  # Invalid arm
        
        # Select arm with highest expected reward
        if valid_arms:
            best_arm_idx = np.argmax([expected_rewards[arm] for arm in valid_arms])
            best_arm = valid_arms[best_arm_idx]
        else:
            # Fallback to taker if no valid arms
            best_arm = 1  # TAKER arm
        
        return best_arm, self.arm_names[best_arm]
    
    def update(self, arm: int, context: MarketContext, reward: float):
        """Update bandit with new observation."""
        
        # Update arm statistics
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.arm_reward_sums[arm] += reward
        
        # Update contextual features
        features = self._context_to_features(context)
        self.arm_contexts[arm] = features
        
        # Update feature weights using gradient descent
        if self.arm_counts[arm] > 1:
            # Compute prediction error
            predicted_reward = np.dot(self.feature_weights[arm], features)
            error = reward - predicted_reward
            
            # Update weights
            self.feature_weights[arm] += self.learning_rate * error * features
            
            # Update feature counts
            self.feature_counts[arm] += 1
        
        logger.debug(f"Updated arm {arm} with reward {reward:.4f}")
    
    def get_arm_statistics(self) -> Dict:
        """Get statistics for all arms."""
        stats = {}
        
        for arm in range(self.n_arms):
            if self.arm_counts[arm] > 0:
                stats[self.arm_names[arm].value] = {
                    "count": int(self.arm_counts[arm]),
                    "avg_reward": self.arm_rewards[arm] / self.arm_counts[arm],
                    "total_reward": self.arm_rewards[arm],
                    "feature_weights": self.feature_weights[arm].tolist()
                }
            else:
                stats[self.arm_names[arm].value] = {
                    "count": 0,
                    "avg_reward": 0.0,
                    "total_reward": 0.0,
                    "feature_weights": self.feature_weights[arm].tolist()
                }
        
        return stats

class OffPolicyEvaluator:
    """Off-policy evaluation for bandit performance estimation."""
    
    def __init__(self, bandit: ContextualBandit):
        self.bandit = bandit
        self.evaluation_data = []
    
    def add_observation(self, context: MarketContext, chosen_arm: int, 
                       reward: float, action_probability: float):
        """Add observation for off-policy evaluation."""
        
        self.evaluation_data.append({
            "context": context,
            "chosen_arm": chosen_arm,
            "reward": reward,
            "action_probability": action_probability,
            "timestamp": pd.Timestamp.now()
        })
    
    def estimate_policy_value(self, target_policy: str = "current") -> Dict:
        """Estimate value of target policy using off-policy evaluation."""
        
        if not self.evaluation_data:
            return {"policy_value": 0.0, "confidence_interval": (0.0, 0.0)}
        
        # Compute importance sampling weights
        weights = []
        rewards = []
        
        for obs in self.evaluation_data:
            context = obs["context"]
            chosen_arm = obs["chosen_arm"]
            reward = obs["reward"]
            action_prob = obs["action_probability"]
            
            # Compute target policy probability
            if target_policy == "current":
                # Use current bandit policy
                expected_rewards = [
                    self.bandit._compute_expected_reward(arm, context) 
                    for arm in range(self.bandit.n_arms)
                ]
                target_prob = np.exp(expected_rewards[chosen_arm]) / np.sum(np.exp(expected_rewards))
            else:
                # Use uniform policy for comparison
                target_prob = 1.0 / self.bandit.n_arms
            
            # Importance sampling weight
            weight = target_prob / action_prob if action_prob > 0 else 0.0
            weights.append(weight)
            rewards.append(reward)
        
        # Compute weighted average
        weights = np.array(weights)
        rewards = np.array(rewards)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Compute policy value
        policy_value = np.sum(weights * rewards)
        
        # Compute confidence interval (bootstrap)
        n_bootstrap = 1000
        bootstrap_values = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(rewards), size=len(rewards), replace=True)
            bootstrap_value = np.sum(weights[indices] * rewards[indices])
            bootstrap_values.append(bootstrap_value)
        
        confidence_interval = (
            np.percentile(bootstrap_values, 2.5),
            np.percentile(bootstrap_values, 97.5)
        )
        
        return {
            "policy_value": policy_value,
            "confidence_interval": confidence_interval,
            "n_observations": len(self.evaluation_data),
            "effective_sample_size": np.sum(weights) ** 2 / np.sum(weights ** 2)
        }
    
    def compare_policies(self) -> Dict:
        """Compare current policy with baseline policies."""
        
        # Estimate current policy value
        current_value = self.estimate_policy_value("current")
        
        # Estimate uniform policy value
        uniform_value = self.estimate_policy_value("uniform")
        
        # Estimate greedy policy value (always choose best arm)
        greedy_value = self.estimate_policy_value("greedy")
        
        return {
            "current_policy": current_value,
            "uniform_policy": uniform_value,
            "greedy_policy": greedy_value,
            "improvement_vs_uniform": current_value["policy_value"] - uniform_value["policy_value"],
            "improvement_vs_greedy": current_value["policy_value"] - greedy_value["policy_value"]
        }

class ExecutionBanditManager:
    """Manager for execution bandit system."""
    
    def __init__(self):
        self.bandit = ContextualBandit()
        self.evaluator = OffPolicyEvaluator(self.bandit)
        self.execution_history = []
    
    def make_execution_decision(self, context: MarketContext) -> Tuple[OrderType, float]:
        """Make execution decision using bandit."""
        
        # Select arm
        arm, order_type = self.bandit.select_arm(context)
        
        # Compute action probability for off-policy evaluation
        expected_rewards = [
            self.bandit._compute_expected_reward(a, context) 
            for a in range(self.bandit.n_arms)
        ]
        action_prob = np.exp(expected_rewards[arm]) / np.sum(np.exp(expected_rewards))
        
        # Log decision
        decision = {
            "timestamp": pd.Timestamp.now(),
            "context": context,
            "chosen_arm": arm,
            "order_type": order_type,
            "action_probability": action_prob,
            "expected_rewards": expected_rewards
        }
        
        self.execution_history.append(decision)
        
        return order_type, action_prob
    
    def update_with_result(self, context: MarketContext, order_type: OrderType, 
                          result: ExecutionResult):
        """Update bandit with execution result."""
        
        # Find corresponding decision
        decision = None
        for d in reversed(self.execution_history):
            if d["order_type"] == order_type and d["context"] == context:
                decision = d
                break
        
        if not decision:
            logger.warning("No matching decision found for result")
            return
        
        # Compute reward
        reward = self._compute_reward(result)
        
        # Update bandit
        arm = decision["chosen_arm"]
        self.bandit.update(arm, context, reward)
        
        # Update evaluator
        self.evaluator.add_observation(
            context, arm, reward, decision["action_probability"]
        )
        
        logger.info(f"Updated bandit with reward {reward:.4f} for {order_type.value}")
    
    def _compute_reward(self, result: ExecutionResult) -> float:
        """Compute reward from execution result."""
        
        if not result.success:
            return -1.0  # Penalty for failed execution
        
        # Base reward from fill
        base_reward = 1.0
        
        # Slippage penalty
        slippage_penalty = -result.slippage_bps / 100.0
        
        # Speed bonus
        speed_bonus = max(0, (1000 - result.fill_time_ms) / 1000.0)
        
        # Rebate bonus
        rebate_bonus = result.rebate_earned * 10.0
        
        total_reward = base_reward + slippage_penalty + speed_bonus + rebate_bonus
        
        return total_reward
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        
        # Get bandit statistics
        bandit_stats = self.bandit.get_arm_statistics()
        
        # Get off-policy evaluation
        policy_comparison = self.evaluator.compare_policies()
        
        # Compute execution metrics
        total_decisions = len(self.execution_history)
        successful_executions = len([d for d in self.execution_history if d.get("success", False)])
        success_rate = successful_executions / total_decisions if total_decisions > 0 else 0.0
        
        return {
            "bandit_statistics": bandit_stats,
            "policy_comparison": policy_comparison,
            "execution_metrics": {
                "total_decisions": total_decisions,
                "successful_executions": successful_executions,
                "success_rate": success_rate
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    def save_report(self, filepath: str = "reports/ml/execution_bandit_report.json"):
        """Save performance report to file."""
        
        report = self.get_performance_report()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Execution bandit report saved to {filepath}")

def main():
    """Demonstrate contextual bandit system."""
    
    # Initialize bandit manager
    manager = ExecutionBanditManager()
    
    # Simulate some execution decisions
    for i in range(100):
        # Create random market context
        context = MarketContext(
            spread_bps=np.random.uniform(0.5, 5.0),
            depth_ratio=np.random.uniform(0.1, 1.0),
            volatility=np.random.uniform(0.01, 0.05),
            urgency=np.random.uniform(0.0, 1.0),
            time_to_close=np.random.uniform(0.0, 1.0),
            volume_profile=np.random.uniform(0.5, 2.0)
        )
        
        # Make decision
        order_type, action_prob = manager.make_execution_decision(context)
        
        # Simulate execution result
        result = ExecutionResult(
            order_type=order_type,
            fill_price=0.52 + np.random.normal(0, 0.001),
            fill_quantity=1000,
            slippage_bps=np.random.uniform(0.1, 2.0),
            fill_time_ms=np.random.uniform(50, 500),
            rebate_earned=0.0005 if order_type == OrderType.POST_ONLY else 0.0,
            success=np.random.random() > 0.1  # 90% success rate
        )
        
        # Update bandit
        manager.update_with_result(context, order_type, result)
    
    # Generate and save report
    manager.save_report()
    
    # Print summary
    report = manager.get_performance_report()
    print("✅ Contextual Bandit System Demo")
    print(f"   Total decisions: {report['execution_metrics']['total_decisions']}")
    print(f"   Success rate: {report['execution_metrics']['success_rate']:.1%}")
    print(f"   Policy improvement vs uniform: {report['policy_comparison']['improvement_vs_uniform']:.4f}")
    
    return 0

if __name__ == "__main__":
    exit(main())
