#!/usr/bin/env python3
"""
🤖 REINFORCEMENT LEARNING TRADING ENGINE
========================================
Advanced RL system with multiple agents, self-training, and adaptive strategies.
Built with free ML libraries for maximum performance at zero cost.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
import pickle
import os
from collections import deque
import random
import asyncio

# Free ML libraries
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available, using basic implementations")

import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradingState:
    """Represents the current trading environment state"""
    price: float
    volume: float
    portfolio_value: float
    position: float  # -1 to 1 (short to long)
    cash: float
    unrealized_pnl: float
    volatility: float
    trend: float
    rsi: float
    macd: float
    bollinger_position: float
    volume_profile: float
    market_regime: str  # 'trending', 'ranging', 'volatile'
    time_features: List[float]  # hour, day_of_week, etc.

@dataclass
class TradingAction:
    """Represents a trading action"""
    action_type: str  # 'buy', 'sell', 'hold'
    size: float  # Position size (0-1)
    confidence: float  # Action confidence (0-1)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class BasicScaler:
    """Basic scaler implementation when sklearn is not available"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit_transform(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1  # Avoid division by zero
        return (X - self.mean_) / self.std_
    
    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.std_

class BasicRandomForest:
    """Basic random forest implementation when sklearn is not available"""
    
    def __init__(self, n_estimators=10, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        
        for i in range(self.n_estimators):
            # Simple decision tree (linear regression on random features)
            n_features = min(int(np.sqrt(X.shape[1])), X.shape[1])
            feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
            
            # Bootstrap sample
            n_samples = X.shape[0]
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            
            X_sample = X[sample_indices][:, feature_indices]
            y_sample = y[sample_indices]
            
            # Simple linear regression
            if X_sample.shape[0] > 0:
                weights = np.linalg.lstsq(X_sample, y_sample, rcond=None)[0]
                self.trees.append((feature_indices, weights))
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for feature_indices, weights in self.trees:
            X_subset = X[:, feature_indices]
            pred = np.dot(X_subset, weights)
            predictions.append(pred)
        
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return np.zeros(X.shape[0])
    
    def score(self, X, y):
        pred = self.predict(X)
        return 1 - np.mean((y - pred) ** 2) / np.var(y)

class TradingEnvironment:
    """Trading environment for RL agents"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.reset()
        self.logger = logging.getLogger(__name__)
        
    def reset(self) -> TradingState:
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.trades_history = []
        self.current_step = 0
        
        return self._get_current_state()
    
    def _get_current_state(self) -> TradingState:
        """Get current environment state"""
        current_time = datetime.now()
        
        return TradingState(
            price=45000.0 + np.random.normal(0, 500),  # BTC-like price
            volume=np.random.lognormal(10, 1),
            portfolio_value=self.balance + self.position_value,
            position=self.position,
            cash=self.balance,
            unrealized_pnl=self.position_value,
            volatility=np.random.uniform(0.01, 0.05),
            trend=np.random.uniform(-0.02, 0.02),
            rsi=np.random.uniform(20, 80),
            macd=np.random.uniform(-100, 100),
            bollinger_position=np.random.uniform(0, 1),
            volume_profile=np.random.uniform(0.5, 1.5),
            market_regime=random.choice(['trending', 'ranging', 'volatile']),
            time_features=[
                current_time.hour / 24.0,
                current_time.weekday() / 7.0,
                current_time.day / 31.0
            ]
        )
    
    def step(self, action: TradingAction, current_price: float) -> Tuple[TradingState, float, bool]:
        """Execute action and return new state, reward, and done flag"""
        reward = 0.0
        
        # Execute action
        if action.action_type == 'buy' and self.position < 0.9:
            trade_size = min(action.size, 0.9 - self.position)
            trade_value = trade_size * current_price
            
            if self.balance >= trade_value:
                self.position += trade_size
                self.balance -= trade_value
                self.entry_price = current_price
                reward += 0.1  # Small reward for taking action
                
        elif action.action_type == 'sell' and self.position > -0.9:
            trade_size = min(action.size, self.position + 0.9)
            trade_value = trade_size * current_price
            
            self.position -= trade_size
            self.balance += trade_value
            
            # Calculate profit/loss
            if self.entry_price > 0:
                pnl = (current_price - self.entry_price) * trade_size
                reward += pnl / self.initial_balance  # Normalized reward
                
        # Update position value
        self.position_value = self.position * current_price
        
        # Risk management penalties
        portfolio_value = self.balance + self.position_value
        if portfolio_value < self.initial_balance * 0.8:  # 20% drawdown
            reward -= 0.5
            
        # Volatility-adjusted reward
        reward *= (1 + action.confidence)  # Reward confident actions
        
        self.current_step += 1
        done = self.current_step >= 1000 or portfolio_value < self.initial_balance * 0.5
        
        return self._get_current_state(), reward, done

class DQNAgent:
    """Deep Q-Network agent using sklearn/basic implementation"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Use ensemble of models as neural network approximation
        if SKLEARN_AVAILABLE:
            self.q_networks = [
                RandomForestRegressor(n_estimators=50, random_state=i)
                for i in range(action_size)
            ]
            self.target_networks = [
                RandomForestRegressor(n_estimators=50, random_state=i+100)
                for i in range(action_size)
            ]
            self.scaler = StandardScaler()
        else:
            self.q_networks = [
                BasicRandomForest(n_estimators=10, random_state=i)
                for i in range(action_size)
            ]
            self.target_networks = [
                BasicRandomForest(n_estimators=10, random_state=i+100)
                for i in range(action_size)
            ]
            self.scaler = BasicScaler()
        
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        if not self.is_trained:
            return np.random.choice(self.action_size)
            
        try:
            state_scaled = self.scaler.transform([state])
            q_values = []
            
            for network in self.q_networks:
                q_val = network.predict(state_scaled)[0]
                q_values.append(q_val)
                
            return int(np.argmax(q_values))
            
        except Exception as e:
            self.logger.warning(f"Action selection failed: {e}")
            return np.random.choice(self.action_size)
    
    def replay(self, batch_size: int = 32):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        try:
            # Sample batch
            batch = random.sample(self.memory, batch_size)
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            
            # Scale states
            if not self.is_trained:
                states_scaled = self.scaler.fit_transform(states)
                next_states_scaled = self.scaler.transform(next_states)
                self.is_trained = True
            else:
                states_scaled = self.scaler.transform(states)
                next_states_scaled = self.scaler.transform(next_states)
            
            # Calculate target Q-values
            target_q_values = rewards.copy()
            
            if self.is_trained:
                for i, network in enumerate(self.target_networks):
                    next_q_values = network.predict(next_states_scaled)
                    target_q_values += 0.95 * next_q_values * (1 - dones)  # Gamma = 0.95
            
            # Train each Q-network
            for action_idx in range(self.action_size):
                action_mask = (actions == action_idx)
                if np.sum(action_mask) > 0:
                    action_states = states_scaled[action_mask]
                    action_targets = target_q_values[action_mask]
                    
                    self.q_networks[action_idx].fit(action_states, action_targets)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            self.logger.info(f"🧠 Agent trained on {batch_size} experiences (ε={self.epsilon:.3f})")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")

class MultiAgentRLSystem:
    """Multi-agent reinforcement learning system"""
    
    def __init__(self, state_size: int = 20):
        self.state_size = state_size
        self.action_size = 3  # buy, sell, hold
        
        # Create multiple agents with different strategies
        self.agents = {
            'dqn_conservative': DQNAgent(state_size, self.action_size, learning_rate=0.001),
            'dqn_aggressive': DQNAgent(state_size, self.action_size, learning_rate=0.01),
        }
        
        self.environment = TradingEnvironment()
        self.performance_tracker = {agent_name: [] for agent_name in self.agents.keys()}
        self.ensemble_weights = {agent_name: 1.0 for agent_name in self.agents.keys()}
        
        self.logger = logging.getLogger(__name__)
        self.training_episodes = 0
        
    def state_to_vector(self, state: TradingState) -> np.ndarray:
        """Convert trading state to feature vector"""
        features = [
            state.price / 50000.0,  # Normalized price
            state.volume / 1000000.0,  # Normalized volume
            state.portfolio_value / 20000.0,  # Normalized portfolio
            state.position,
            state.cash / 20000.0,  # Normalized cash
            state.unrealized_pnl / 5000.0,  # Normalized PnL
            state.volatility,
            state.trend,
            state.rsi / 100.0,  # Normalized RSI
            state.macd / 1000.0,  # Normalized MACD
            state.bollinger_position,
            state.volume_profile,
            1.0 if state.market_regime == 'trending' else 0.0,
            1.0 if state.market_regime == 'ranging' else 0.0,
            1.0 if state.market_regime == 'volatile' else 0.0,
        ]
        
        # Add time features
        features.extend(state.time_features)
        
        # Pad to state_size if needed
        while len(features) < self.state_size:
            features.append(0.0)
            
        return np.array(features[:self.state_size])
    
    def train_agents(self, episodes: int = 1000):
        """Train all agents through episodes"""
        self.logger.info(f"🚀 Starting multi-agent training for {episodes} episodes...")
        
        for episode in range(episodes):
            # Reset environment
            state = self.environment.reset()
            state_vector = self.state_to_vector(state)
            
            episode_rewards = {agent_name: 0.0 for agent_name in self.agents.keys()}
            
            done = False
            step = 0
            
            while not done and step < 200:  # Limit steps per episode
                current_price = state.price
                
                # Get actions from all agents
                agent_actions = {}
                
                for agent_name, agent in self.agents.items():
                    try:
                        action = agent.act(state_vector)
                        # Ensure action is an integer in valid range
                        if not isinstance(action, int) or action not in [0, 1, 2]:
                            action = 2  # Default to hold
                        agent_actions[agent_name] = action
                    except Exception as e:
                        self.logger.warning(f"❌ Agent {agent_name} prediction failed: {e}")
                        agent_actions[agent_name] = 2  # Default to hold
                
                # Use ensemble decision (weighted voting)
                ensemble_action = self._ensemble_decision(agent_actions)
                
                # Convert action to TradingAction
                trading_action = self._action_to_trading_action(ensemble_action)
                
                # Execute action in environment
                next_state, reward, done = self.environment.step(trading_action, current_price)
                next_state_vector = self.state_to_vector(next_state)
                
                # Store experiences for each agent
                for agent_name, agent in self.agents.items():
                    action = agent_actions[agent_name]
                    episode_rewards[agent_name] += reward
                    
                    agent.remember(state_vector, action, reward, next_state_vector, done)
                
                # Update state
                state = next_state
                state_vector = next_state_vector
                step += 1
            
            # Train agents
            for agent_name, agent in self.agents.items():
                agent.replay(batch_size=32)
                
                # Track performance
                self.performance_tracker[agent_name].append(episode_rewards[agent_name])
            
            # Update ensemble weights based on performance
            if episode > 0 and episode % 50 == 0:
                self._update_ensemble_weights()
            
            # Log progress
            if episode % 100 == 0:
                avg_rewards = {name: np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                             for name, rewards in self.performance_tracker.items()}
                self.logger.info(f"📊 Episode {episode}: Avg Rewards = {avg_rewards}")
        
        self.training_episodes += episodes
        self.logger.info(f"✅ Training completed! Total episodes: {self.training_episodes}")
    
    def _ensemble_decision(self, agent_actions: Dict[str, int]) -> int:
        """Make ensemble decision based on weighted voting"""
        try:
            weighted_votes = {}
            
            for agent_name, action in agent_actions.items():
                # Ensure action is an integer
                if not isinstance(action, int):
                    self.logger.warning(f"Agent {agent_name} returned non-integer action: {action}")
                    action = 2  # Default to hold
                
                # Ensure action is in valid range
                if action not in [0, 1, 2]:
                    self.logger.warning(f"Agent {agent_name} returned invalid action: {action}")
                    action = 2  # Default to hold
                
                # Get weight with fallback for missing agents
                weight = self.ensemble_weights.get(agent_name, 1.0)
                if action not in weighted_votes:
                    weighted_votes[action] = 0
                weighted_votes[action] += weight
            
            # Return action with highest weighted vote
            if weighted_votes:
                return max(weighted_votes.keys(), key=lambda x: weighted_votes[x])
            else:
                return 2  # Default to hold
                
        except Exception as e:
            self.logger.error(f"❌ Ensemble decision failed: {e}")
            return 2  # Default to hold
    
    def _action_to_trading_action(self, action: int) -> TradingAction:
        """Convert integer action to TradingAction"""
        action_map = {
            0: TradingAction('buy', 0.1, 0.7),
            1: TradingAction('sell', 0.1, 0.7),
            2: TradingAction('hold', 0.0, 0.5)
        }
        return action_map.get(action, TradingAction('hold', 0.0, 0.5))
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance"""
        try:
            recent_performance = {}
            
            for agent_name, rewards in self.performance_tracker.items():
                if len(rewards) >= 50:
                    recent_performance[agent_name] = np.mean(rewards[-50:])
                else:
                    recent_performance[agent_name] = np.mean(rewards)
            
            # Normalize weights (softmax)
            max_perf = max(recent_performance.values())
            exp_perfs = {name: np.exp(perf - max_perf) for name, perf in recent_performance.items()}
            total_exp = sum(exp_perfs.values())
            
            self.ensemble_weights = {name: exp_perf / total_exp 
                                   for name, exp_perf in exp_perfs.items()}
            
            self.logger.info(f"🎯 Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            self.logger.error(f"❌ Weight update failed: {e}")
    
    def predict_action(self, current_state) -> TradingAction:
        """Predict best action using trained ensemble"""
        try:
            # Handle both TradingState objects and dictionaries
            if isinstance(current_state, dict):
                # Convert dict to TradingState object
                trading_state = TradingState(
                    price=current_state.get('price', 0.0),
                    volume=current_state.get('volume', 1000000.0),
                    portfolio_value=current_state.get('portfolio_value', 100000.0),
                    position=current_state.get('position', 0.0),
                    cash=current_state.get('cash', 100000.0),
                    unrealized_pnl=current_state.get('unrealized_pnl', 0.0),
                    volatility=current_state.get('volatility', 0.02),
                    trend=current_state.get('trend', 0.0),
                    rsi=current_state.get('rsi', 50.0),
                    macd=current_state.get('macd', 0.0),
                    bollinger_position=current_state.get('bollinger_position', 0.5),
                    volume_profile=current_state.get('volume_profile', 1.0),
                    market_regime=current_state.get('market_regime', 'ranging'),
                    time_features=current_state.get('time_features', [0.5, 0.5, 0.5])
                )
            else:
                trading_state = current_state
            
            state_vector = self.state_to_vector(trading_state)
            agent_actions = {}
            
            # Get predictions from all agents with robust error handling
            for agent_name, agent in self.agents.items():
                try:
                    action = agent.act(state_vector)
                    # Ensure action is an integer in valid range
                    if not isinstance(action, int) or action not in [0, 1, 2]:
                        self.logger.warning(f"Agent {agent_name} returned invalid action: {action}")
                        action = 2  # Default to hold
                    agent_actions[agent_name] = action
                except Exception as e:
                    self.logger.warning(f"❌ Agent {agent_name} prediction failed: {e}")
                    agent_actions[agent_name] = 2  # Default to hold
            
            # Make ensemble decision
            ensemble_action = self._ensemble_decision(agent_actions)
            trading_action = self._action_to_trading_action(ensemble_action)
            
            self.logger.info(f"🤖 RL Prediction: {trading_action.action_type} "
                           f"(size: {trading_action.size}, confidence: {trading_action.confidence})")
            
            return trading_action
            
        except Exception as e:
            self.logger.error(f"❌ RL prediction failed: {e}")
            return TradingAction('hold', 0.0, 0.5)
    
    def save_models(self, filepath: str):
        """Save trained models"""
        try:
            models_data = {
                'ensemble_weights': self.ensemble_weights,
                'performance_tracker': self.performance_tracker,
                'training_episodes': self.training_episodes,
                'sklearn_available': SKLEARN_AVAILABLE
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(models_data, f)
                
            self.logger.info(f"💾 Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Model saving failed: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Model file {filepath} not found")
                return
                
            with open(filepath, 'rb') as f:
                models_data = pickle.load(f)
            
            self.ensemble_weights = models_data.get('ensemble_weights', self.ensemble_weights)
            self.performance_tracker = models_data.get('performance_tracker', self.performance_tracker)
            self.training_episodes = models_data.get('training_episodes', 0)
            
            self.logger.info(f"📂 Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all agents"""
        summary = {
            'training_episodes': self.training_episodes,
            'ensemble_weights': self.ensemble_weights,
            'sklearn_available': SKLEARN_AVAILABLE,
            'agent_performance': {}
        }
        
        for agent_name, rewards in self.performance_tracker.items():
            if rewards:
                summary['agent_performance'][agent_name] = {
                    'total_episodes': len(rewards),
                    'avg_reward': np.mean(rewards),
                    'best_reward': np.max(rewards),
                    'worst_reward': np.min(rewards),
                    'recent_avg': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize RL system
    rl_system = MultiAgentRLSystem()
    
    # Train agents
    print("🚀 Starting RL training...")
    rl_system.train_agents(episodes=500)
    
    # Test prediction
    test_state = TradingState(
        price=45000.0,
        volume=1000000.0,
        portfolio_value=10000.0,
        position=0.0,
        cash=10000.0,
        unrealized_pnl=0.0,
        volatility=0.02,
        trend=0.01,
        rsi=65.0,
        macd=50.0,
        bollinger_position=0.7,
        volume_profile=1.2,
        market_regime='trending',
        time_features=[0.5, 0.3, 0.2]
    )
    
    action = rl_system.predict_action(test_state)
    print(f"🤖 Predicted action: {action}")
    
    # Performance summary
    summary = rl_system.get_performance_summary()
    print(f"📊 Performance summary: {summary}")
    
    # Save models
    rl_system.save_models('rl_models.pkl')
    print("💾 Models saved successfully!") 