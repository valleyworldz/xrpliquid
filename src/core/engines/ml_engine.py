"""
ML Engineer (PhD ML, RL, Deep Learning)
Implements reinforcement learning for dynamic parameter adaptation and strategy optimization.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import threading
import json
import os

@dataclass
class TradingState:
    """Represents the current trading state for RL"""
    price: float
    volume: float
    volatility: float
    trend_strength: float
    momentum: float
    market_regime: str
    position_size: float
    unrealized_pnl: float
    drawdown: float
    confidence: float
    timestamp: float

@dataclass
class TradingAction:
    """Represents a trading action for RL"""
    action_type: str  # 'buy', 'sell', 'hold', 'adjust_position'
    confidence_threshold: float
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    risk_multiplier: float

@dataclass
class TradingReward:
    """Represents the reward for a trading action"""
    pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    risk_adjusted_return: float

class MLEngine:
    """
    ML Engineer: Implements reinforcement learning for dynamic parameter adaptation
    and strategy optimization.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("🧠 [ML_ENGINE] ML Engineer initialized")
        
        # RL Components
        self.state_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        # Dynamic Parameters
        self.current_params = {
            'confidence_threshold': 0.7,
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'risk_multiplier': 1.0,
            'momentum_threshold': 0.5,
            'trend_threshold': 0.6,
            'volatility_threshold': 0.02
        }
        
        # Performance Tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_trade_duration': 0.0
        }
        
        # Learning Parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.discount_factor = 0.95
        
        # State Management
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        
        # Threading
        self.lock = threading.Lock()
        self.is_learning = False
        
        # Model Persistence
        self.model_path = "ml_engine_state.json"
        self.load_model_state()
        
        self.logger.info("🧠 [ML_ENGINE] RL components initialized")
        self.logger.info(f"🧠 [ML_ENGINE] Current parameters: {self.current_params}")
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current dynamic parameters"""
        with self.lock:
            return self.current_params.copy()
    
    def update_state(self, 
                    price: float,
                    volume: float,
                    volatility: float,
                    trend_strength: float,
                    momentum: float,
                    market_regime: str,
                    position_size: float = 0.0,
                    unrealized_pnl: float = 0.0,
                    drawdown: float = 0.0,
                    confidence: float = 0.0) -> None:
        """Update current trading state"""
        try:
            state = TradingState(
                price=price,
                volume=volume,
                volatility=volatility,
                trend_strength=trend_strength,
                momentum=momentum,
                market_regime=market_regime,
                position_size=position_size,
                unrealized_pnl=unrealized_pnl,
                drawdown=drawdown,
                confidence=confidence,
                timestamp=time.time()
            )
            
            with self.lock:
                self.last_state = state
                self.state_history.append(state)
            
            self.logger.debug(f"🧠 [ML_ENGINE] State updated: price={price:.4f}, vol={volatility:.4f}, trend={trend_strength:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error updating state: {e}")
    
    def select_action(self, signal_strength: float, market_conditions: Dict[str, Any]) -> TradingAction:
        """Select optimal action using RL"""
        try:
            if not self.last_state:
                return self._get_default_action()
            
            # Epsilon-greedy exploration
            if np.random.random() < self.exploration_rate:
                action = self._explore_action()
                self.logger.debug(f"🧠 [ML_ENGINE] Exploration action: {action.action_type}")
            else:
                action = self._exploit_action(signal_strength, market_conditions)
                self.logger.debug(f"🧠 [ML_ENGINE] Exploitation action: {action.action_type}")
            
            with self.lock:
                self.last_action = action
                self.action_history.append(action)
            
            return action
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error selecting action: {e}")
            return self._get_default_action()
    
    def _explore_action(self) -> TradingAction:
        """Generate random action for exploration"""
        action_types = ['buy', 'sell', 'hold', 'adjust_position']
        action_type = np.random.choice(action_types)
        
        return TradingAction(
            action_type=action_type,
            confidence_threshold=np.random.uniform(0.3, 0.9),
            position_size_multiplier=np.random.uniform(0.5, 1.5),
            stop_loss_multiplier=np.random.uniform(0.8, 1.2),
            take_profit_multiplier=np.random.uniform(0.8, 1.2),
            risk_multiplier=np.random.uniform(0.8, 1.2)
        )
    
    def _exploit_action(self, signal_strength: float, market_conditions: Dict[str, Any]) -> TradingAction:
        """Generate action based on learned policy"""
        try:
            # Simple policy based on current parameters and market conditions
            volatility = market_conditions.get('volatility', 0.02)
            trend_strength = market_conditions.get('trend_strength', 0.5)
            momentum = market_conditions.get('momentum', 0.5)
            
            # Adjust confidence threshold based on market conditions
            confidence_threshold = self.current_params['confidence_threshold']
            if volatility > self.current_params['volatility_threshold']:
                confidence_threshold *= 1.2  # Higher threshold in high volatility
            if trend_strength > self.current_params['trend_threshold']:
                confidence_threshold *= 0.9  # Lower threshold in strong trends
            
            # Determine action type based on signal strength and conditions
            if signal_strength > confidence_threshold:
                if momentum > self.current_params['momentum_threshold']:
                    action_type = 'buy'
                else:
                    action_type = 'hold'
            else:
                action_type = 'hold'
            
            # Adjust position size based on risk
            position_size_multiplier = self.current_params['position_size_multiplier']
            if volatility > self.current_params['volatility_threshold']:
                position_size_multiplier *= 0.8  # Reduce size in high volatility
            
            return TradingAction(
                action_type=action_type,
                confidence_threshold=confidence_threshold,
                position_size_multiplier=position_size_multiplier,
                stop_loss_multiplier=self.current_params['stop_loss_multiplier'],
                take_profit_multiplier=self.current_params['take_profit_multiplier'],
                risk_multiplier=self.current_params['risk_multiplier']
            )
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error in exploit action: {e}")
            return self._get_default_action()
    
    def _get_default_action(self) -> TradingAction:
        """Get default action when no learning is available"""
        return TradingAction(
            action_type='hold',
            confidence_threshold=0.7,
            position_size_multiplier=1.0,
            stop_loss_multiplier=1.0,
            take_profit_multiplier=1.0,
            risk_multiplier=1.0
        )
    
    def update_reward(self, pnl: float, trade_duration: float, max_drawdown: float) -> None:
        """Update reward for the last action"""
        try:
            # Calculate reward components
            pnl_reward = pnl / 100.0  # Normalize PnL
            duration_penalty = -0.1 if trade_duration > 3600 else 0.0  # Penalize long trades
            drawdown_penalty = -max_drawdown / 100.0  # Penalize drawdown
            
            # Calculate Sharpe-like ratio
            if len(self.reward_history) > 0:
                avg_reward = np.mean([r.pnl for r in self.reward_history])
                reward_std = np.std([r.pnl for r in self.reward_history])
                sharpe_reward = (pnl - avg_reward) / (reward_std + 1e-6)
            else:
                sharpe_reward = 0.0
            
            total_reward = pnl_reward + duration_penalty + drawdown_penalty + sharpe_reward * 0.1
            
            reward = TradingReward(
                pnl=pnl,
                sharpe_ratio=sharpe_reward,
                max_drawdown=max_drawdown,
                win_rate=1.0 if pnl > 0 else 0.0,
                risk_adjusted_return=total_reward
            )
            
            with self.lock:
                self.last_reward = reward
                self.reward_history.append(reward)
            
            # Update performance metrics
            self._update_performance_metrics(pnl, trade_duration, max_drawdown)
            
            self.logger.debug(f"🧠 [ML_ENGINE] Reward updated: pnl={pnl:.4f}, reward={total_reward:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error updating reward: {e}")
    
    def _update_performance_metrics(self, pnl: float, trade_duration: float, max_drawdown: float) -> None:
        """Update performance tracking metrics"""
        try:
            with self.lock:
                self.performance_metrics['total_trades'] += 1
                if pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                
                self.performance_metrics['total_pnl'] += pnl
                self.performance_metrics['max_drawdown'] = max(
                    self.performance_metrics['max_drawdown'], 
                    max_drawdown
                )
                
                # Update win rate
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['winning_trades'] / 
                    self.performance_metrics['total_trades']
                )
                
                # Update average trade duration
                current_avg = self.performance_metrics['avg_trade_duration']
                total_trades = self.performance_metrics['total_trades']
                self.performance_metrics['avg_trade_duration'] = (
                    (current_avg * (total_trades - 1) + trade_duration) / total_trades
                )
                
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error updating performance metrics: {e}")
    
    def learn(self) -> None:
        """Perform learning step using collected experience"""
        try:
            if self.is_learning or len(self.state_history) < 10:
                return
            
            self.is_learning = True
            
            # Simple Q-learning update
            if (self.last_state and self.last_action and self.last_reward and 
                len(self.state_history) > 1):
                
                # Get previous state-action pair
                prev_state = list(self.state_history)[-2]
                prev_action = list(self.action_history)[-2]
                
                # Calculate Q-value update
                current_q = self._get_q_value(prev_state, prev_action)
                next_q = self._get_q_value(self.last_state, self.last_action)
                
                # Q-learning update rule
                new_q = current_q + self.learning_rate * (
                    self.last_reward.risk_adjusted_return + 
                    self.discount_factor * next_q - current_q
                )
                
                # Update parameters based on learning
                self._update_parameters_from_learning(new_q, current_q)
                
                self.logger.info(f"🧠 [ML_ENGINE] Learning step completed: Q={new_q:.4f}")
            
            self.is_learning = False
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error in learning step: {e}")
            self.is_learning = False
    
    def _get_q_value(self, state: TradingState, action: TradingAction) -> float:
        """Get Q-value for state-action pair (simplified)"""
        try:
            # Simple Q-value approximation based on state features
            q_value = (
                state.trend_strength * 0.3 +
                state.momentum * 0.2 +
                (1.0 - state.volatility) * 0.2 +
                (1.0 - abs(state.unrealized_pnl)) * 0.3
            )
            
            # Adjust based on action type
            if action.action_type == 'buy':
                q_value *= 1.1
            elif action.action_type == 'sell':
                q_value *= 0.9
            
            return q_value
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error calculating Q-value: {e}")
            return 0.0
    
    def _update_parameters_from_learning(self, new_q: float, old_q: float) -> None:
        """Update parameters based on learning results"""
        try:
            q_improvement = new_q - old_q
            
            if q_improvement > 0:
                # Positive learning - slightly increase exploration
                self.exploration_rate = min(0.2, self.exploration_rate * 1.05)
                
                # Adjust parameters based on successful learning
                if new_q > 0.5:
                    self.current_params['confidence_threshold'] *= 0.98
                    self.current_params['position_size_multiplier'] *= 1.02
                else:
                    self.current_params['confidence_threshold'] *= 1.02
                    self.current_params['position_size_multiplier'] *= 0.98
            else:
                # Negative learning - increase exploration
                self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
                
                # Adjust parameters conservatively
                self.current_params['confidence_threshold'] *= 1.01
                self.current_params['position_size_multiplier'] *= 0.99
            
            # Ensure parameters stay within reasonable bounds
            self._clamp_parameters()
            
            self.logger.debug(f"🧠 [ML_ENGINE] Parameters updated: Q_improvement={q_improvement:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error updating parameters: {e}")
    
    def _clamp_parameters(self) -> None:
        """Ensure parameters stay within reasonable bounds"""
        try:
            self.current_params['confidence_threshold'] = np.clip(
                self.current_params['confidence_threshold'], 0.3, 0.9
            )
            self.current_params['position_size_multiplier'] = np.clip(
                self.current_params['position_size_multiplier'], 0.5, 1.5
            )
            self.current_params['stop_loss_multiplier'] = np.clip(
                self.current_params['stop_loss_multiplier'], 0.8, 1.2
            )
            self.current_params['take_profit_multiplier'] = np.clip(
                self.current_params['take_profit_multiplier'], 0.8, 1.2
            )
            self.current_params['risk_multiplier'] = np.clip(
                self.current_params['risk_multiplier'], 0.8, 1.2
            )
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error clamping parameters: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        try:
            with self.lock:
                return {
                    'parameters': self.current_params.copy(),
                    'performance': self.performance_metrics.copy(),
                    'learning_stats': {
                        'exploration_rate': self.exploration_rate,
                        'learning_rate': self.learning_rate,
                        'state_history_size': len(self.state_history),
                        'action_history_size': len(self.action_history),
                        'reward_history_size': len(self.reward_history)
                    }
                }
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error getting performance summary: {e}")
            return {}
    
    def save_model_state(self) -> None:
        """Save current model state to file"""
        try:
            state = {
                'current_params': self.current_params,
                'performance_metrics': self.performance_metrics,
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate,
                'discount_factor': self.discount_factor
            }
            
            with open(self.model_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.debug(f"🧠 [ML_ENGINE] Model state saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error saving model state: {e}")
    
    def load_model_state(self) -> None:
        """Load model state from file"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'r') as f:
                    state = json.load(f)
                
                self.current_params = state.get('current_params', self.current_params)
                self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
                self.learning_rate = state.get('learning_rate', self.learning_rate)
                self.exploration_rate = state.get('exploration_rate', self.exploration_rate)
                self.discount_factor = state.get('discount_factor', self.discount_factor)
                
                self.logger.info(f"🧠 [ML_ENGINE] Model state loaded from {self.model_path}")
            else:
                self.logger.info("🧠 [ML_ENGINE] No saved model state found, using defaults")
                
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error loading model state: {e}")
    
    def start_learning_thread(self) -> None:
        """Start background learning thread"""
        try:
            def learning_loop():
                while True:
                    try:
                        self.learn()
                        time.sleep(60)  # Learn every minute
                    except Exception as e:
                        self.logger.error(f"❌ [ML_ENGINE] Error in learning loop: {e}")
                        time.sleep(60)
            
            learning_thread = threading.Thread(target=learning_loop, daemon=True)
            learning_thread.start()
            
            self.logger.info("🧠 [ML_ENGINE] Learning thread started")
            
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error starting learning thread: {e}")
    
    def shutdown(self) -> None:
        """Shutdown ML engine and save state"""
        try:
            self.save_model_state()
            self.logger.info("🧠 [ML_ENGINE] ML Engine shutdown complete")
        except Exception as e:
            self.logger.error(f"❌ [ML_ENGINE] Error during shutdown: {e}")
