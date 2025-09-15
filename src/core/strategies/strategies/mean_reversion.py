#!/usr/bin/env python3
"""
📈 ENHANCED MEAN REVERSION STRATEGY
=================================

Advanced mean reversion strategy with:
- Bollinger Bands for entry/exit signals
- Z-score analysis for mean reversion strength
- Dynamic position sizing based on deviation
- Adaptive stop-loss and take-profit levels
- Risk management integration
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from core.strategies.base_strategy import TradingStrategy
from core.utils.logger import Logger

class MeanReversion(TradingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.logger = Logger()
        
        # Strategy parameters with safe defaults
        self.lookback_period = self._get_config("strategies.mean_reversion.lookback_period", 20)
        self.std_multiplier = self._get_config("strategies.mean_reversion.deviation_threshold", 2.0)
        self.max_position_size = self._get_config("strategies.mean_reversion.max_position_size", 0.05)
        self.profit_target = self._get_config("strategies.mean_reversion.profit_target", 0.02)
        self.stop_loss = self._get_config("strategies.mean_reversion.stop_loss", 0.01)
        self.confidence_threshold = self._get_config("strategies.mean_reversion.confidence_threshold", 0.7)
        self.max_holding_time = self._get_config("strategies.mean_reversion.max_holding_time", 3600)  # 1 hour
        
        # Position tracking
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        
        # Performance tracking
        self.trade_history = []
        self.signal_strength_history = []
        
        self.logger.info(f"[MEAN_REVERSION] Strategy initialized with lookback: {self.lookback_period}, std_multiplier: {self.std_multiplier}")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Safely get config value with fallback"""
        try:
            if self.config and hasattr(self.config, 'get'):
                return self.config.get(key, default)
            return default
        except Exception:
            return default
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = None, std_dev: float = None) -> Dict[str, float]:
        """Calculate Bollinger Bands for mean reversion analysis"""
        try:
            if not prices or len(prices) < 2:
                return {"upper": 0.0, "middle": 0.0, "lower": 0.0, "bandwidth": 0.0}
            
            period = period or self.lookback_period
            std_dev = std_dev or self.std_multiplier
            
            # Use available data if less than period
            actual_period = min(period, len(prices))
            recent_prices = np.array(prices[-actual_period:], dtype=float)
            
            if len(recent_prices) < 2:
                current_price = prices[-1]
                return {"upper": current_price, "middle": current_price, "lower": current_price, "bandwidth": 0.0}
            
            # Calculate moving average and standard deviation
            moving_average = np.mean(recent_prices)
            rolling_std = np.std(recent_prices)
            
            # Calculate bands
            upper_band = moving_average + (std_dev * rolling_std)
            lower_band = moving_average - (std_dev * rolling_std)
            bandwidth = (upper_band - lower_band) / moving_average if moving_average > 0 else 0.0
            
            return {
                "upper": float(upper_band),
                "middle": float(moving_average),
                "lower": float(lower_band),
                "bandwidth": float(bandwidth)
            }
            
        except Exception as e:
            self.logger.error(f"[MEAN_REVERSION] Error calculating Bollinger Bands: {e}")
            current_price = prices[-1] if prices else 0.0
            return {"upper": current_price, "middle": current_price, "lower": current_price, "bandwidth": 0.0}
    
    def calculate_z_score(self, current_price: float, prices: List[float]) -> float:
        """Calculate Z-score for mean reversion strength"""
        try:
            if not prices or len(prices) < 2:
                return 0.0
            
            actual_period = min(self.lookback_period, len(prices))
            recent_prices = np.array(prices[-actual_period:], dtype=float)
            
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            if std_price == 0:
                return 0.0
            
            z_score = (current_price - mean_price) / std_price
            return float(z_score)
            
        except Exception as e:
            self.logger.error(f"[MEAN_REVERSION] Error calculating Z-score: {e}")
            return 0.0
    
    def calculate_mean_reversion_strength(self, current_price: float, prices: List[float], volumes: List[float] = None) -> float:
        """Calculate overall mean reversion signal strength"""
        try:
            if not prices or len(prices) < 5:
                return 0.0
            
            # Calculate Z-score component
            z_score = abs(self.calculate_z_score(current_price, prices))
            z_score_strength = min(z_score / 3.0, 1.0)  # Normalize to 0-1, stronger signal for higher deviation
            
            # Calculate trend consistency (less trending = better for mean reversion)
            if len(prices) >= 5:
                recent_prices = prices[-5:]
                price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
                trend_consistency = 1.0 - (abs(sum(price_changes)) / sum(abs(change) for change in price_changes) if sum(abs(change) for change in price_changes) > 0 else 0.0)
            else:
                trend_consistency = 0.5
            
            # Volume confirmation (if available)
            volume_factor = 1.0
            if volumes and len(volumes) >= 3:
                recent_volumes = volumes[-3:]
                avg_volume = np.mean(recent_volumes[:-1])
                current_volume = recent_volumes[-1]
                if avg_volume > 0:
                    volume_factor = min(current_volume / avg_volume, 2.0) / 2.0  # Normalize to 0-1
            
            # Combined strength
            overall_strength = (z_score_strength * 0.5 + trend_consistency * 0.3 + volume_factor * 0.2)
            return max(0.0, min(1.0, overall_strength))
            
        except Exception as e:
            self.logger.error(f"[MEAN_REVERSION] Error calculating signal strength: {e}")
            return 0.0
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mean reversion trading signal"""
        try:
            # Validate input data
            if not self.validate_data(market_data):
                return {}
            
            current_price = float(market_data["price"])
            symbol = market_data.get("symbol", "UNKNOWN")
            
            # Get price history
            price_history = market_data.get("price_history", [current_price])
            volume_history = market_data.get("volume_history", [])
            
            if len(price_history) < 3:
                return {}  # Need some history for mean reversion
            
            # Calculate Bollinger Bands
            bands = self.calculate_bollinger_bands(price_history)
            
            # Calculate signal strength
            signal_strength = self.calculate_mean_reversion_strength(current_price, price_history, volume_history)
            
            # Determine signal direction and confidence
            signal = {}
            
            # Check for oversold condition (buy signal)
            if current_price < bands["lower"] and signal_strength > self.confidence_threshold:
                distance_from_band = (bands["lower"] - current_price) / bands["lower"]
                confidence = min(signal_strength * (1 + distance_from_band), 1.0)
                
                signal = {
                    "action": "buy",
                    "confidence": confidence,
                    "signal_strength": signal_strength,
                    "entry_price": current_price,
                    "target_price": bands["middle"],
                    "stop_loss_price": current_price * (1 - self.stop_loss),
                    "profit_target": self.profit_target,
                    "bollinger_bands": bands,
                    "z_score": self.calculate_z_score(current_price, price_history),
                    "max_holding_time": self.max_holding_time,
                    "strategy": "mean_reversion",
                    "timestamp": datetime.now()
                }
            
            # Check for overbought condition (sell signal)
            elif current_price > bands["upper"] and signal_strength > self.confidence_threshold:
                distance_from_band = (current_price - bands["upper"]) / bands["upper"]
                confidence = min(signal_strength * (1 + distance_from_band), 1.0)
                
                signal = {
                    "action": "sell",
                    "confidence": confidence,
                    "signal_strength": signal_strength,
                    "entry_price": current_price,
                    "target_price": bands["middle"],
                    "stop_loss_price": current_price * (1 + self.stop_loss),
                    "profit_target": self.profit_target,
                    "bollinger_bands": bands,
                    "z_score": self.calculate_z_score(current_price, price_history),
                    "max_holding_time": self.max_holding_time,
                    "strategy": "mean_reversion",
                    "timestamp": datetime.now()
                }
            
            # Check for position close signal (price near middle band)
            elif abs(current_price - bands["middle"]) / bands["middle"] < 0.005:  # Within 0.5% of middle
                if self.current_position != 0:
                    signal = {
                        "action": "close",
                        "confidence": 0.8,
                        "signal_strength": signal_strength,
                        "entry_price": current_price,
                        "reason": "price_reverted_to_mean",
                        "bollinger_bands": bands,
                        "strategy": "mean_reversion",
                        "timestamp": datetime.now()
                    }
            
            if signal:
                self.signal_strength_history.append(signal_strength)
                self.last_signal_time = datetime.now()
                self.logger.info(f"[MEAN_REVERSION] Generated {signal['action']} signal for {symbol}: confidence={signal['confidence']:.3f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[MEAN_REVERSION] Error generating signal: {e}")
            return {}
    
    def update_position(self, action: str, price: float, quantity: float = None):
        """Update current position tracking"""
        try:
            if action == "buy":
                self.current_position += quantity or self.max_position_size
                self.entry_price = price
                self.entry_time = datetime.now()
            elif action == "sell":
                self.current_position -= quantity or self.max_position_size
                self.entry_price = price
                self.entry_time = datetime.now()
            elif action == "close":
                self.current_position = 0.0
                self.entry_price = 0.0
                self.entry_time = None
            
            self.logger.info(f"[MEAN_REVERSION] Position updated: {self.current_position}")
            
        except Exception as e:
            self.logger.error(f"[MEAN_REVERSION] Error updating position: {e}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get current strategy statistics"""
        base_stats = super().get_strategy_info()
        mean_reversion_stats = {
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "lookback_period": self.lookback_period,
            "std_multiplier": self.std_multiplier,
            "confidence_threshold": self.confidence_threshold,
            "avg_signal_strength": np.mean(self.signal_strength_history) if self.signal_strength_history else 0.0,
            "trade_count": len(self.trade_history)
        }
        return {**base_stats, **mean_reversion_stats}
    
    def run(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Main strategy execution method"""
        try:
            # Validate inputs
            if not self.validate_data(data) or not self.validate_params(params):
                return {}
            
            if not self.is_enabled():
                return {}
            
            # Check for position timeout
            if (self.entry_time and 
                (datetime.now() - self.entry_time).total_seconds() > self.max_holding_time):
                return {
                    "action": "close",
                    "confidence": 0.9,
                    "reason": "max_holding_time_exceeded",
                    "strategy": "mean_reversion",
                    "timestamp": datetime.now()
                }
            
            # Generate signal
            signal = self.generate_signal(data)
            
            if signal and signal.get("action") in ["buy", "sell", "close"]:
                self.logger.info(f"[MEAN_REVERSION] Generated signal for {data.get('symbol', 'UNKNOWN')}: {signal['action']} @ {signal['confidence']:.3f} confidence")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[MEAN_REVERSION] Error in strategy run: {e}")
            return {}
    
    def __str__(self):
        return f"Enhanced Mean Reversion Strategy (lookback: {self.lookback_period}, std: {self.std_multiplier})"


