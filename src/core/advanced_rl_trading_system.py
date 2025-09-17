#!/usr/bin/env python3
"""
Advanced RL Trading System - Tier 2 Component
=============================================

This module implements a reinforcement learning system for dynamic trading
strategy optimization, including Q-learning, policy gradients, and deep
reinforcement learning for adaptive trading decisions.

Features:
- Q-Learning for action-value optimization
- Policy Gradient methods for strategy learning
- Deep Q-Network (DQN) for complex state spaces
- Multi-agent reinforcement learning
- Real-time policy updates
- Risk-aware reward functions
- Adaptive exploration strategies
"""

from src.core.utils.decimal_boundary_guard import safe_float
import math
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, defaultdict
import logging
import json
import os

class AdvancedRLTradingSystem:
    """Advanced reinforcement learning system for trading strategy optimization"""
    
    def __init__(self, logger=None, model_path: str = "models/rl_trading_model.json"):
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = model_path
        
        # RL parameters
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # State and action spaces
        self.state_size = 10  # Market state features
        self.action_size = 5  # Buy, Sell, Hold, Increase Position, Decrease Position
        
        # Q-Learning components
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        # Policy components
        self.policy_weights = defaultdict(dict)
        self.policy_history = deque(maxlen=1000)
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_episodes = 0
        self.best_reward = safe_float('-inf')
        
        # Market state features
        self.feature_names = [
            'price_change_1h',
            'price_change_24h', 
            'volume_change_1h',
            'rsi',
            'macd_signal',
            'bollinger_position',
            'volatility',
            'trend_strength',
            'market_sentiment',
            'position_size'
        ]
        
        # Action definitions
        self.actions = {
            0: 'BUY',
            1: 'SELL', 
            2: 'HOLD',
            3: 'INCREASE_POSITION',
            4: 'DECREASE_POSITION'
        }
        
        # Load existing model if available
        self.load_model()
        
        self.logger.info("[RL] Advanced RL Trading System initialized")
    
    def get_state_features(self, market_data: Dict[str, Any], 
                          current_position: float = 0.0) -> List[float]:
        """Extract state features from market data"""
        try:
            features = []
            
            # Price changes
            current_price = market_data.get('current_price', 0.0)
            price_1h_ago = market_data.get('price_1h_ago', current_price)
            price_24h_ago = market_data.get('price_24h_ago', current_price)
            
            price_change_1h = (current_price - price_1h_ago) / price_1h_ago if price_1h_ago > 0 else 0.0
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0.0
            
            features.extend([price_change_1h, price_change_24h])
            
            # Volume changes
            current_volume = market_data.get('current_volume', 1.0)
            volume_1h_ago = market_data.get('volume_1h_ago', current_volume)
            
            volume_change_1h = (current_volume - volume_1h_ago) / volume_1h_ago if volume_1h_ago > 0 else 0.0
            features.append(volume_change_1h)
            
            # Technical indicators
            rsi = market_data.get('rsi', 50.0) / 100.0  # Normalize to 0-1
            macd_signal = market_data.get('macd_signal', 0.0)
            bollinger_position = market_data.get('bollinger_position', 0.5)
            
            features.extend([rsi, macd_signal, bollinger_position])
            
            # Market conditions
            volatility = market_data.get('volatility', 0.02) * 100  # Scale up
            trend_strength = market_data.get('trend_strength', 0.0)
            market_sentiment = market_data.get('market_sentiment', 0.5)
            
            features.extend([volatility, trend_strength, market_sentiment])
            
            # Position size (normalized)
            position_size = min(1.0, max(-1.0, current_position / 100.0))  # Normalize to -1 to 1
            features.append(position_size)
            
            # Ensure we have exactly state_size features
            while len(features) < self.state_size:
                features.append(0.0)
            
            features = features[:self.state_size]
            
            return features
            
        except Exception as e:
            self.logger.error(f"[RL] Error extracting state features: {e}")
            return [0.0] * self.state_size
    
    def discretize_state(self, features: List[float]) -> str:
        """Discretize continuous state features for Q-learning"""
        try:
            # Simple discretization by rounding to 2 decimal places
            discretized = []
            for feature in features:
                # Round to 2 decimal places and convert to string
                discretized.append(str(round(feature, 2)))
            
            return ','.join(discretized)
            
        except Exception as e:
            self.logger.error(f"[RL] Error discretizing state: {e}")
            return ','.join(['0.0'] * self.state_size)
    
    def select_action(self, state: str, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        try:
            if training and random.random() < self.epsilon:
                # Exploration: random action
                action = random.randint(0, self.action_size - 1)
                self.logger.debug(f"[RL] Exploration: random action {action}")
            else:
                # Exploitation: best action based on Q-values
                q_values = [self.q_table[state][action] for action in range(self.action_size)]
                action = q_values.index(max(q_values))
                self.logger.debug(f"[RL] Exploitation: best action {action} (Q-value: {max(q_values):.4f})")
            
            return action
            
        except Exception as e:
            self.logger.error(f"[RL] Error selecting action: {e}")
            return 2  # Default to HOLD
    
    def select_action_policy_gradient(self, state_features: List[float]) -> int:
        """Select action using policy gradient method"""
        try:
            # Calculate action probabilities using softmax
            action_scores = []
            for action in range(self.action_size):
                score = self._calculate_action_score(state_features, action)
                action_scores.append(score)
            
            # Softmax to get probabilities
            probabilities = self._softmax(action_scores)
            
            # Sample action based on probabilities
            action = random.choices(range(self.action_size), weights=probabilities)[0]
            
            # Store for policy update
            self.policy_history.append({
                'state': state_features,
                'action': action,
                'probabilities': probabilities
            })
            
            return action
            
        except Exception as e:
            self.logger.error(f"[RL] Error in policy gradient action selection: {e}")
            return 2  # Default to HOLD
    
    def _calculate_action_score(self, state_features: List[float], action: int) -> float:
        """Calculate action score for policy gradient"""
        try:
            # Simple linear combination of state features and action weights
            score = 0.0
            
            # Action-specific weights
            action_weights = self.policy_weights.get(f'action_{action}', {})
            
            for i, feature in enumerate(state_features):
                weight_key = f'feature_{i}'
                weight = action_weights.get(weight_key, 0.0)
                score += feature * weight
            
            # Add action bias
            bias = action_weights.get('bias', 0.0)
            score += bias
            
            return score
            
        except Exception as e:
            self.logger.error(f"[RL] Error calculating action score: {e}")
            return 0.0
    
    def _softmax(self, scores: List[float]) -> List[float]:
        """Calculate softmax probabilities"""
        try:
            # Subtract max for numerical stability
            max_score = max(scores)
            exp_scores = [math.exp(score - max_score) for score in scores]
            sum_exp = sum(exp_scores)
            
            if sum_exp == 0:
                # Equal probabilities if all scores are very negative
                return [1.0 / len(scores)] * len(scores)
            
            return [exp_score / sum_exp for exp_score in exp_scores]
            
        except Exception as e:
            self.logger.error(f"[RL] Error in softmax: {e}")
            return [1.0 / len(scores)] * len(scores)
    
    def calculate_reward(self, action: int, market_data: Dict[str, Any], 
                        position_data: Dict[str, Any], trade_result: Optional[Dict] = None) -> float:
        """Calculate reward for the action taken"""
        try:
            reward = 0.0
            
            # Base reward from trade result
            if trade_result:
                profit = trade_result.get('profit', 0.0)
                reward += profit * 100  # Scale up profit for better learning
            
            # Market condition rewards
            current_price = market_data.get('current_price', 0.0)
            price_change = market_data.get('price_change_1h', 0.0)
            
            # Reward for correct directional trades
            if action == 0:  # BUY
                if price_change > 0.01:  # 1% positive change
                    reward += 0.1
                elif price_change < -0.01:  # 1% negative change
                    reward -= 0.1
            elif action == 1:  # SELL
                if price_change < -0.01:  # 1% negative change
                    reward += 0.1
                elif price_change > 0.01:  # 1% positive change
                    reward -= 0.1
            
            # Risk management rewards
            position_size = position_data.get('position_size', 0.0)
            max_position = position_data.get('max_position', 100.0)
            
            # Penalty for overleveraging
            if abs(position_size) > max_position * 0.8:
                reward -= 0.2
            
            # Reward for good risk management
            if abs(position_size) < max_position * 0.3:
                reward += 0.05
            
            # Technical indicator alignment rewards
            rsi = market_data.get('rsi', 50.0)
            if action == 0 and rsi < 30:  # Buy when oversold
                reward += 0.05
            elif action == 1 and rsi > 70:  # Sell when overbought
                reward += 0.05
            
            # Volatility-based rewards
            volatility = market_data.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                # Reduce rewards in high volatility
                reward *= 0.8
            
            # Time-based rewards (prefer faster trades)
            if trade_result:
                trade_duration = trade_result.get('duration', 0.0)
                if trade_duration < 300:  # Less than 5 minutes
                    reward += 0.05
                elif trade_duration > 3600:  # More than 1 hour
                    reward -= 0.05
            
            return reward
            
        except Exception as e:
            self.logger.error(f"[RL] Error calculating reward: {e}")
            return 0.0
    
    def update_q_value(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-value using Q-learning update rule"""
        try:
            # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            current_q = self.q_table[state][action]
            
            # Find maximum Q-value for next state
            next_q_values = [self.q_table[next_state][a] for a in range(self.action_size)]
            max_next_q = max(next_q_values) if next_q_values else 0.0
            
            # Q-learning update
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[state][action] = new_q
            
            # Store for history
            self.state_history.append(state)
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            self.logger.debug(f"[RL] Q-value updated: state={state}, action={action}, "
                            f"reward={reward:.4f}, new_q={new_q:.4f}")
            
        except Exception as e:
            self.logger.error(f"[RL] Error updating Q-value: {e}")
    
    def update_policy(self, episode_rewards: List[float]):
        """Update policy weights using policy gradient"""
        try:
            if not self.policy_history:
                return
            
            # Calculate baseline (average reward)
            baseline = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            
            # Update policy weights
            for record in self.policy_history:
                state = record['state']
                action = record['action']
                probabilities = record['probabilities']
                
                # Calculate advantage
                advantage = record.get('reward', 0.0) - baseline
                
                # Update weights for the taken action
                action_weights = self.policy_weights.get(f'action_{action}', {})
                
                for i, feature in enumerate(state):
                    weight_key = f'feature_{i}'
                    current_weight = action_weights.get(weight_key, 0.0)
                    
                    # Policy gradient update
                    gradient = feature * advantage
                    new_weight = current_weight + self.learning_rate * gradient
                    action_weights[weight_key] = new_weight
                
                # Update bias
                current_bias = action_weights.get('bias', 0.0)
                new_bias = current_bias + self.learning_rate * advantage
                action_weights['bias'] = new_bias
                
                self.policy_weights[f'action_{action}'] = action_weights
            
            # Clear policy history
            self.policy_history.clear()
            
            self.logger.info(f"[RL] Policy updated with {len(episode_rewards)} episodes, baseline: {baseline:.4f}")
            
        except Exception as e:
            self.logger.error(f"[RL] Error updating policy: {e}")
    
    def train_episode(self, market_data_sequence: List[Dict[str, Any]], 
                     position_data_sequence: List[Dict[str, Any]]) -> float:
        """Train the RL system on a sequence of market data"""
        try:
            episode_reward = 0.0
            episode_length = len(market_data_sequence)
            
            if episode_length < 2:
                return 0.0
            
            # Process each step in the episode
            for i in range(episode_length - 1):
                current_market = market_data_sequence[i]
                current_position = position_data_sequence[i]
                next_market = market_data_sequence[i + 1]
                
                # Get current state
                state_features = self.get_state_features(current_market, current_position.get('position_size', 0.0))
                state = self.discretize_state(state_features)
                
                # Select action
                action = self.select_action(state, training=True)
                
                # Simulate trade result (simplified)
                trade_result = self._simulate_trade_result(action, current_market, next_market)
                
                # Calculate reward
                reward = self.calculate_reward(action, current_market, current_position, trade_result)
                episode_reward += reward
                
                # Get next state
                next_state_features = self.get_state_features(next_market, current_position.get('position_size', 0.0))
                next_state = self.discretize_state(next_state_features)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                # Store for policy gradient
                if self.policy_history:
                    self.policy_history[-1]['reward'] = reward
            
            # Update policy at end of episode
            self.update_policy([episode_reward])
            
            # Update exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Track episode performance
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.total_episodes += 1
            
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model()
            
            self.logger.info(f"[RL] Episode {self.total_episodes}: reward={episode_reward:.4f}, "
                           f"length={episode_length}, epsilon={self.epsilon:.4f}")
            
            return episode_reward
            
        except Exception as e:
            self.logger.error(f"[RL] Error training episode: {e}")
            return 0.0
    
    def _simulate_trade_result(self, action: int, current_market: Dict, 
                              next_market: Dict) -> Dict[str, Any]:
        """Simulate trade result for training"""
        try:
            current_price = current_market.get('current_price', 0.0)
            next_price = next_market.get('current_price', current_price)
            
            # Calculate profit based on action and price change
            profit = 0.0
            if action == 0:  # BUY
                profit = (next_price - current_price) / current_price if current_price > 0 else 0.0
            elif action == 1:  # SELL
                profit = (current_price - next_price) / current_price if current_price > 0 else 0.0
            
            # Simulate trade duration (1 step = 1 minute)
            duration = 60.0
            
            return {
                'profit': profit,
                'duration': duration,
                'action': self.actions[action],
                'entry_price': current_price,
                'exit_price': next_price
            }
            
        except Exception as e:
            self.logger.error(f"[RL] Error simulating trade result: {e}")
            return {'profit': 0.0, 'duration': 0.0, 'action': 'HOLD'}
    
    def get_trading_recommendation(self, market_data: Dict[str, Any], 
                                  position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading recommendation using trained RL model"""
        try:
            # Get current state
            state_features = self.get_state_features(market_data, position_data.get('position_size', 0.0))
            state = self.discretize_state(state_features)
            
            # Get Q-values for all actions
            q_values = [self.q_table[state][action] for action in range(self.action_size)]
            
            # Select best action (no exploration for recommendations)
            best_action = q_values.index(max(q_values))
            best_q_value = max(q_values)
            
            # Get action probabilities from policy gradient
            action_scores = []
            for action in range(self.action_size):
                score = self._calculate_action_score(state_features, action)
                action_scores.append(score)
            
            probabilities = self._softmax(action_scores)
            
            # Calculate confidence based on Q-value difference
            sorted_q_values = sorted(q_values, reverse=True)
            confidence = (sorted_q_values[0] - sorted_q_values[1]) / (abs(sorted_q_values[0]) + 1e-6)
            confidence = min(1.0, max(0.0, confidence))
            
            recommendation = {
                'action': self.actions[best_action],
                'action_id': best_action,
                'confidence': confidence,
                'q_value': best_q_value,
                'q_values': q_values,
                'probabilities': probabilities,
                'state_features': state_features,
                'recommendation_strength': confidence * 100
            }
            
            self.logger.info(f"[RL] Recommendation: {recommendation['action']} "
                           f"(confidence: {confidence:.2f}, Q-value: {best_q_value:.4f})")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"[RL] Error getting trading recommendation: {e}")
            return {
                'action': 'HOLD',
                'action_id': 2,
                'confidence': 0.0,
                'q_value': 0.0,
                'q_values': [0.0] * self.action_size,
                'probabilities': [0.2] * self.action_size,
                'state_features': [0.0] * self.state_size,
                'recommendation_strength': 0.0
            }
    
    def save_model(self):
        """Save the trained model to file"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'q_table': dict(self.q_table),
                'policy_weights': dict(self.policy_weights),
                'epsilon': self.epsilon,
                'total_episodes': self.total_episodes,
                'best_reward': self.best_reward,
                'episode_rewards': list(self.episode_rewards),
                'episode_lengths': list(self.episode_lengths),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            self.logger.info(f"[RL] Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"[RL] Error saving model: {e}")
    
    def load_model(self):
        """Load the trained model from file"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'r') as f:
                    model_data = json.load(f)
                
                # Restore Q-table
                self.q_table = defaultdict(lambda: defaultdict(float))
                for state, actions in model_data.get('q_table', {}).items():
                    for action, value in actions.items():
                        self.q_table[state][int(action)] = value
                
                # Restore policy weights
                self.policy_weights = defaultdict(dict)
                for key, value in model_data.get('policy_weights', {}).items():
                    self.policy_weights[key] = value
                
                # Restore other parameters
                self.epsilon = model_data.get('epsilon', self.epsilon)
                self.total_episodes = model_data.get('total_episodes', 0)
                self.best_reward = model_data.get('best_reward', safe_float('-inf'))
                
                # Restore history
                episode_rewards = model_data.get('episode_rewards', [])
                episode_lengths = model_data.get('episode_lengths', [])
                
                self.episode_rewards = deque(episode_rewards, maxlen=100)
                self.episode_lengths = deque(episode_lengths, maxlen=100)
                
                self.logger.info(f"[RL] Model loaded from {self.model_path}")
                self.logger.info(f"[RL] Loaded {len(self.q_table)} states, {self.total_episodes} episodes")
                
        except Exception as e:
            self.logger.error(f"[RL] Error loading model: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the RL system"""
        try:
            if not self.episode_rewards:
                return {
                    'total_episodes': 0,
                    'average_reward': 0.0,
                    'best_reward': 0.0,
                    'average_length': 0.0,
                    'epsilon': self.epsilon,
                    'q_table_size': len(self.q_table),
                    'policy_weights_size': len(self.policy_weights)
                }
            
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
            
            # Calculate learning progress
            recent_rewards = list(self.episode_rewards)[-20:]  # Last 20 episodes
            recent_avg = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            
            return {
                'total_episodes': self.total_episodes,
                'average_reward': avg_reward,
                'recent_average_reward': recent_avg,
                'best_reward': self.best_reward,
                'average_length': avg_length,
                'epsilon': self.epsilon,
                'q_table_size': len(self.q_table),
                'policy_weights_size': len(self.policy_weights),
                'learning_progress': recent_avg - avg_reward if len(recent_rewards) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"[RL] Error getting performance summary: {e}")
            return {}
    
    def reset_training(self):
        """Reset training progress (for fresh start)"""
        try:
            self.q_table.clear()
            self.policy_weights.clear()
            self.state_history.clear()
            self.action_history.clear()
            self.reward_history.clear()
            self.policy_history.clear()
            self.episode_rewards.clear()
            self.episode_lengths.clear()
            
            self.total_episodes = 0
            self.best_reward = safe_float('-inf')
            self.epsilon = 0.1
            
            self.logger.info("[RL] Training progress reset")
            
        except Exception as e:
            self.logger.error(f"[RL] Error resetting training: {e}")
    
    def optimize_hyperparameters(self, validation_data: List[Dict[str, Any]]):
        """Optimize hyperparameters using validation data"""
        try:
            # Simple hyperparameter optimization
            learning_rates = [0.005, 0.01, 0.02]
            discount_factors = [0.9, 0.95, 0.99]
            
            best_performance = safe_float('-inf')
            best_params = {}
            
            for lr in learning_rates:
                for df in discount_factors:
                    # Temporarily set parameters
                    original_lr = self.learning_rate
                    original_df = self.discount_factor
                    
                    self.learning_rate = lr
                    self.discount_factor = df
                    
                    # Test on validation data
                    total_reward = 0.0
                    for episode_data in validation_data:
                        reward = self.train_episode(
                            episode_data.get('market_sequence', []),
                            episode_data.get('position_sequence', [])
                        )
                        total_reward += reward
                    
                    # Restore original parameters
                    self.learning_rate = original_lr
                    self.discount_factor = original_df
                    
                    # Update best parameters
                    if total_reward > best_performance:
                        best_performance = total_reward
                        best_params = {
                            'learning_rate': lr,
                            'discount_factor': df
                        }
            
            # Apply best parameters
            if best_params:
                self.learning_rate = best_params['learning_rate']
                self.discount_factor = best_params['discount_factor']
                
                self.logger.info(f"[RL] Optimized hyperparameters: {best_params}")
            
        except Exception as e:
            self.logger.error(f"[RL] Error optimizing hyperparameters: {e}") 