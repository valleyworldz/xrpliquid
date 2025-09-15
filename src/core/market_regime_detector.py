#!/usr/bin/env python3
"""
Auto-Market Regime Detection System
Detects market conditions and adapts trading strategies accordingly
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

class AutoMarketRegimeDetector:
    """Advanced market regime detection system"""
    
    def __init__(self, logger):
        self.logger = logger
        self.regime_history = []
        self.volatility_history = []
        self.trend_history = []
        self.volume_history = []
        self.current_regime = "NEUTRAL"
        self.regime_confidence = 0.5
        
        # Regime thresholds
        self.trend_threshold = 0.02  # 2% trend strength
        self.volatility_threshold = 0.03  # 3% volatility
        self.volume_threshold = 1.5  # 50% above average volume
        
        self.logger.info("[REGIME] Auto-Market Regime Detection System initialized")
    
    def detect_regime(self, market_data: Dict) -> Tuple[str, float]:
        """Detect current market regime"""
        try:
            # Extract price data
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            
            if len(prices) < 20:
                return "NEUTRAL", 0.5
            
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(prices)
            
            # Calculate volatility
            volatility = self.calculate_volatility(prices)
            
            # Calculate volume ratio
            volume_ratio = self.calculate_volume_ratio(volumes)
            
            # Determine regime
            regime, confidence = self.classify_regime(trend_strength, volatility, volume_ratio)
            
            # Update history
            timestamp = datetime.now()
            self.regime_history.append((timestamp, regime, confidence))
            self.trend_history.append((timestamp, trend_strength))
            self.volatility_history.append((timestamp, volatility))
            self.volume_history.append((timestamp, volume_ratio))
            
            # Keep only recent history
            self._cleanup_history()
            
            self.current_regime = regime
            self.regime_confidence = confidence
            
            self.logger.info(f"[REGIME] Detected regime: {regime} (confidence: {confidence:.2f})")
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"[REGIME] Error detecting regime: {e}")
            return "NEUTRAL", 0.5
    
    def calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength using simple linear regression"""
        try:
            if len(prices) < 10:
                return 0.0
            
            # Use last 20 prices for trend calculation
            recent_prices = prices[-20:]
            n = len(recent_prices)
            
            # Simple linear regression
            x_sum = sum(range(n))
            y_sum = sum(recent_prices)
            xy_sum = sum(i * price for i, price in enumerate(recent_prices))
            x2_sum = sum(i * i for i in range(n))
            
            # Calculate slope
            denominator = n * x2_sum - x_sum * x_sum
            if denominator == 0:
                return 0.0
            
            slope = (n * xy_sum - x_sum * y_sum) / denominator
            
            # Normalize by average price
            avg_price = y_sum / n
            normalized_slope = slope / avg_price if avg_price != 0 else 0
            
            return abs(normalized_slope)
            
        except Exception as e:
            self.logger.error(f"[REGIME] Error calculating trend strength: {e}")
            return 0.0
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if not returns:
                return 0.0
            
            # Calculate standard deviation
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = math.sqrt(variance)
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"[REGIME] Error calculating volatility: {e}")
            return 0.0
    
    def calculate_volume_ratio(self, volumes: List[float]) -> float:
        """Calculate volume ratio compared to average"""
        try:
            if len(volumes) < 10:
                return 1.0
            
            recent_volumes = volumes[-10:]
            current_volume = recent_volumes[-1]
            avg_volume = sum(recent_volumes[:-1]) / (len(recent_volumes) - 1)
            
            if avg_volume == 0:
                return 1.0
            
            return current_volume / avg_volume
            
        except Exception as e:
            self.logger.error(f"[REGIME] Error calculating volume ratio: {e}")
            return 1.0
    
    def classify_regime(self, trend_strength: float, volatility: float, volume_ratio: float) -> Tuple[str, float]:
        """Classify market regime based on indicators"""
        try:
            # Calculate confidence scores
            trend_score = min(1.0, trend_strength / self.trend_threshold)
            vol_score = min(1.0, volatility / self.volatility_threshold)
            volume_score = min(1.0, (volume_ratio - 1.0) / (self.volume_threshold - 1.0)) if volume_ratio > 1.0 else 0.0
            
            # Determine regime
            if trend_score > 0.7 and vol_score > 0.5:
                regime = "TRENDING"
                confidence = (trend_score * 0.6 + vol_score * 0.4)
            elif vol_score > 0.7:
                regime = "VOLATILE"
                confidence = vol_score
            elif volume_score > 0.7:
                regime = "HIGH_VOLUME"
                confidence = volume_score
            elif trend_score < 0.3 and vol_score < 0.3:
                regime = "RANGING"
                confidence = 1.0 - max(trend_score, vol_score)
            else:
                regime = "NEUTRAL"
                confidence = 0.5
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"[REGIME] Error classifying regime: {e}")
            return "NEUTRAL", 0.5
    
    def get_regime_recommendations(self) -> Dict[str, any]:
        """Get trading recommendations based on current regime"""
        try:
            recommendations = {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "profit_target_multiplier": 1.0,
                "entry_strategy": "STANDARD",
                "exit_strategy": "STANDARD"
            }
            
            if self.current_regime == "TRENDING":
                recommendations.update({
                    "position_size_multiplier": 1.2,
                    "stop_loss_multiplier": 1.5,
                    "profit_target_multiplier": 1.3,
                    "entry_strategy": "MOMENTUM",
                    "exit_strategy": "TRAILING"
                })
            elif self.current_regime == "VOLATILE":
                recommendations.update({
                    "position_size_multiplier": 0.7,
                    "stop_loss_multiplier": 0.8,
                    "profit_target_multiplier": 0.9,
                    "entry_strategy": "CAUTIOUS",
                    "exit_strategy": "QUICK"
                })
            elif self.current_regime == "RANGING":
                recommendations.update({
                    "position_size_multiplier": 0.8,
                    "stop_loss_multiplier": 0.7,
                    "profit_target_multiplier": 0.8,
                    "entry_strategy": "MEAN_REVERSION",
                    "exit_strategy": "GRID"
                })
            elif self.current_regime == "HIGH_VOLUME":
                recommendations.update({
                    "position_size_multiplier": 1.1,
                    "stop_loss_multiplier": 1.2,
                    "profit_target_multiplier": 1.1,
                    "entry_strategy": "VOLUME_BREAKOUT",
                    "exit_strategy": "VOLUME_BASED"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"[REGIME] Error getting recommendations: {e}")
            return {
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "profit_target_multiplier": 1.0,
                "entry_strategy": "STANDARD",
                "exit_strategy": "STANDARD"
            }
    
    def get_regime_summary(self) -> Dict[str, any]:
        """Get current regime summary"""
        try:
            return {
                "current_regime": self.current_regime,
                "confidence": self.regime_confidence,
                "regime_history": self.regime_history[-10:],  # Last 10 entries
                "trend_strength": self.trend_history[-1][1] if self.trend_history else 0.0,
                "volatility": self.volatility_history[-1][1] if self.volatility_history else 0.0,
                "volume_ratio": self.volume_history[-1][1] if self.volume_history else 1.0
            }
        except Exception as e:
            self.logger.error(f"[REGIME] Error getting regime summary: {e}")
            return {
                "current_regime": "NEUTRAL",
                "confidence": 0.5,
                "regime_history": [],
                "trend_strength": 0.0,
                "volatility": 0.0,
                "volume_ratio": 1.0
            }
    
    def _cleanup_history(self):
        """Clean up old history entries"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            self.regime_history = [(t, r, c) for t, r, c in self.regime_history if t > cutoff_time]
            self.trend_history = [(t, v) for t, v in self.trend_history if t > cutoff_time]
            self.volatility_history = [(t, v) for t, v in self.volatility_history if t > cutoff_time]
            self.volume_history = [(t, v) for t, v in self.volume_history if t > cutoff_time]
            
        except Exception as e:
            self.logger.error(f"[REGIME] Error cleaning up history: {e}") 