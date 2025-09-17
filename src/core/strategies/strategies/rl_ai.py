#!/usr/bin/env python3
"""
🤖 ENHANCED RL AI STRATEGY
=========================

Reinforcement Learning AI strategy with:
- Mock RL model for demonstration
- Advanced feature engineering
- Risk-aware decision making
- Adaptive confidence scoring
- Performance-based learning simulation
"""

from src.core.utils.decimal_boundary_guard import safe_float
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.strategies.base_strategy import TradingStrategy
from core.utils.logger import Logger

class RL_AI_Strategy(TradingStrategy):
    def __init__(self, config=None):
        super().__init__(config)
        self.logger = Logger()
        
        # RL model parameters
        self.model_confidence_threshold = self._get_config("strategies.rl_ai.confidence_threshold", 0.7)
        self.learning_rate = self._get_config("strategies.rl_ai.learning_rate", 0.01)
        self.exploration_rate = self._get_config("strategies.rl_ai.exploration_rate", 0.1)
        self.lookback_period = self._get_config("strategies.rl_ai.lookback_period", 50)
        self.feature_window = self._get_config("strategies.rl_ai.feature_window", 20)
        
        # Model state (simplified RL simulation)
        self.model_weights = self._initialize_model_weights()
        self.model_loaded = True
        self.experience_buffer = []
        self.prediction_history = []
        self.action_values = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        
        # Performance tracking
        self.prediction_accuracy = 0.0
        self.total_predictions = 0
        self.correct_predictions = 0
        
        self.logger.info("[RL_AI] Enhanced RL AI Strategy initialized with mock model")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Safely get config value with fallback"""
        try:
            if self.config and hasattr(self.config, 'get'):
                return self.config.get(key, default)
            return default
        except Exception:
            return default
    
    def _initialize_model_weights(self) -> Dict[str, np.ndarray]:
        """Initialize mock model weights"""
        try:
            np.random.seed(42)  # For reproducible results
            return {
                "price_weights": np.random.randn(10) * 0.1,
                "volume_weights": np.random.randn(5) * 0.1,
                "technical_weights": np.random.randn(8) * 0.1,
                "bias": np.random.randn(3) * 0.01  # For buy, sell, hold
            }
        except Exception as e:
            self.logger.error(f"[RL_AI] Error initializing weights: {e}")
            return {}
    
    def extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for RL model input"""
        try:
            current_price = safe_float(market_data["price"])
            price_history = market_data.get("price_history", [current_price])
            volume_history = market_data.get("volume_history", [1.0])
            
            features = []
            
            # Price-based features
            if len(price_history) >= 10:
                recent_prices = np.array(price_history[-10:], dtype=float)
                
                # Price returns
                returns = np.diff(recent_prices) / recent_prices[:-1]
                features.extend(returns.tolist())
                
                # Price momentum
                short_ma = np.mean(recent_prices[-3:])
                long_ma = np.mean(recent_prices[-10:])
                momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0
                features.append(momentum)
            else:
                features.extend([0.0] * 10)  # Pad with zeros
            
            # Volume-based features
            if len(volume_history) >= 5:
                recent_volumes = np.array(volume_history[-5:], dtype=float)
                volume_ma = np.mean(recent_volumes[:-1]) if len(recent_volumes) > 1 else recent_volumes[0]
                volume_ratio = recent_volumes[-1] / volume_ma if volume_ma > 0 else 1.0
                features.extend([volume_ratio, np.std(recent_volumes), np.mean(recent_volumes)])
            else:
                features.extend([1.0, 0.0, 1.0])
            
            # Technical indicator features
            if len(price_history) >= 14:
                prices = np.array(price_history[-14:], dtype=float)
                
                # RSI approximation
                gains = []
                losses = []
                for i in range(1, len(prices)):
                    change = prices[i] - prices[i-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                else:
                        gains.append(0)
                        losses.append(abs(change))
                
                avg_gain = np.mean(gains) if gains else 0.0
                avg_loss = np.mean(losses) if losses else 0.0
                rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50.0
                
                # Bollinger Band position
                price_ma = np.mean(prices)
                price_std = np.std(prices)
                bb_position = (current_price - price_ma) / price_std if price_std > 0 else 0.0
                
                # Volatility
                volatility = np.std(np.diff(prices) / prices[:-1])
                
                features.extend([rsi / 100.0, bb_position, volatility, price_ma / current_price])
            else:
                features.extend([0.5, 0.0, 0.0, 1.0])
            
            # Market timing features
            current_hour = datetime.now().hour
            features.extend([
                np.sin(2 * np.pi * current_hour / 24),  # Hour cyclical
                np.cos(2 * np.pi * current_hour / 24),
                safe_float(datetime.now().weekday()) / 6.0,  # Day of week
                safe_float(current_price > np.mean(price_history[-10:]) if len(price_history) >= 10 else True)  # Above short MA
            ])
            
            return np.array(features, dtype=float)
            
        except Exception as e:
            self.logger.error(f"[RL_AI] Error extracting features: {e}")
            return np.zeros(25)  # Return zero features as fallback
    
    def predict_action(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict action using mock RL model"""
        try:
            if not self.model_weights or len(features) == 0:
                return {"action": "hold", "confidence": 0.5, "action_values": {"buy": 0.33, "sell": 0.33, "hold": 0.34}}
            
            # Simple linear model simulation
            weights = self.model_weights
            
            # Calculate action values using mock neural network
            price_signal = np.dot(features[:10], weights.get("price_weights", np.zeros(10)))
            volume_signal = np.dot(features[10:13], weights.get("volume_weights", np.zeros(3)))
            technical_signal = np.dot(features[13:17], weights.get("technical_weights", np.zeros(4)))
            
            # Combine signals with bias
            bias = weights.get("bias", np.zeros(3))
            action_scores = np.array([
                price_signal + volume_signal + technical_signal + bias[0],  # Buy
                -price_signal - volume_signal + technical_signal + bias[1],  # Sell  
                abs(technical_signal) * 0.5 + bias[2]  # Hold
            ])
            
            # Apply softmax to get probabilities
            exp_scores = np.exp(action_scores - np.max(action_scores))  # Stability
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Determine action
            action_names = ["buy", "sell", "hold"]
            best_action_idx = np.argmax(probabilities)
            best_action = action_names[best_action_idx]
            confidence = safe_float(probabilities[best_action_idx])
            
            # Add exploration (epsilon-greedy)
            if np.random.random() < self.exploration_rate:
                best_action = np.random.choice(action_names)
                confidence *= 0.5  # Lower confidence for random actions
            
            action_values = {
                "buy": safe_float(probabilities[0]),
                "sell": safe_float(probabilities[1]),
                "hold": safe_float(probabilities[2])
            }
            
            return {
                "action": best_action,
                "confidence": confidence,
                "action_values": action_values,
                "raw_scores": action_scores.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"[RL_AI] Error in prediction: {e}")
            return {"action": "hold", "confidence": 0.5, "action_values": {"buy": 0.33, "sell": 0.33, "hold": 0.34}}
    
    def update_model_weights(self, reward: float, features: np.ndarray, action: str):
        """Update model weights based on reward (simplified learning)"""
        try:
            if not self.model_weights or len(features) == 0:
                return

            # Simple gradient update simulation
            action_idx = {"buy": 0, "sell": 1, "hold": 2}.get(action, 2)
            
            # Update weights based on reward
            if reward > 0:  # Positive reward, strengthen the weights that led to this action
                self.model_weights["price_weights"] += self.learning_rate * reward * features[:10] * 0.1
                self.model_weights["volume_weights"] += self.learning_rate * reward * features[10:13] * 0.1
                self.model_weights["technical_weights"] += self.learning_rate * reward * features[13:17] * 0.1
                self.model_weights["bias"][action_idx] += self.learning_rate * reward * 0.01
            else:  # Negative reward, weaken the weights
                self.model_weights["price_weights"] -= self.learning_rate * abs(reward) * features[:10] * 0.05
                self.model_weights["volume_weights"] -= self.learning_rate * abs(reward) * features[10:13] * 0.05
                self.model_weights["technical_weights"] -= self.learning_rate * abs(reward) * features[13:17] * 0.05
                self.model_weights["bias"][action_idx] -= self.learning_rate * abs(reward) * 0.005
            
            # Clip weights to prevent explosion
            for key in self.model_weights:
                if isinstance(self.model_weights[key], np.ndarray):
                    self.model_weights[key] = np.clip(self.model_weights[key], -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"[RL_AI] Error updating weights: {e}")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal using RL model"""
        try:
            # Validate input data
            if not self.validate_data(market_data):
                return {}
            
            # Extract features
            features = self.extract_features(market_data)
            
            # Get prediction
            prediction = self.predict_action(features)
            
            # Only generate signal if confidence is high enough
            if prediction["confidence"] < self.model_confidence_threshold:
                return {}
            
            current_price = safe_float(market_data["price"])
            symbol = market_data.get("symbol", "UNKNOWN")
            
            # Create signal
            signal = {
                "action": prediction["action"],
                "confidence": prediction["confidence"],
                "action_values": prediction["action_values"],
                "entry_price": current_price,
                "features": features.tolist(),
                "model_confidence": prediction["confidence"],
                "strategy": "rl_ai",
                "timestamp": datetime.now()
            }
            
            # Add to prediction history
            self.prediction_history.append({
                "timestamp": datetime.now(),
                "action": prediction["action"],
                "confidence": prediction["confidence"],
                "price": current_price
            })
            
            self.total_predictions += 1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[RL_AI] Error generating signal: {e}")
            return {}
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update model performance and learning"""
        try:
            super().update_performance(trade_result)
            
            # Calculate reward based on trade result
            pnl = trade_result.get("pnl", 0.0)
            reward = pnl  # Simple reward = PnL
            
            # Update prediction accuracy
            if pnl > 0:
                self.correct_predictions += 1
            
            self.prediction_accuracy = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0
            
            # Update model weights if we have the last prediction
            if self.prediction_history:
                last_prediction = self.prediction_history[-1]
                features = np.array(trade_result.get("features", []))
                if len(features) > 0:
                    self.update_model_weights(reward, features, last_prediction["action"])
            
        except Exception as e:
            self.logger.error(f"[RL_AI] Error updating performance: {e}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get current strategy statistics"""
        base_stats = super().get_strategy_info()
        rl_stats = {
            "model_loaded": self.model_loaded,
            "prediction_accuracy": self.prediction_accuracy,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "confidence_threshold": self.model_confidence_threshold,
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "recent_action_values": self.action_values
        }
        return {**base_stats, **rl_stats}
    
    def run(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Main strategy execution method"""
        try:
            # Validate inputs
            if not self.validate_data(data) or not self.validate_params(params):
                return {}
            
            if not self.is_enabled():
                return {}
            
            if not self.model_loaded:
                self.logger.warning("[RL_AI] Model not loaded, cannot generate signals")
                return {}
            
            # Generate signal
            signal = self.generate_signal(data)
            
            if signal and signal.get("action") != "hold":
                self.last_signal_time = datetime.now()
                self.logger.info(f"[RL_AI] Generated signal for {data.get('symbol', 'UNKNOWN')}: {signal['action']} @ {signal['confidence']:.3f} confidence")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[RL_AI] Error in strategy run: {e}")
            return {}
    
    def __str__(self):
        return f"Enhanced RL AI Strategy (accuracy: {self.prediction_accuracy:.2%}, predictions: {self.total_predictions})"


