#!/usr/bin/env python3
"""
⏰ MULTI-TIMEFRAME STRATEGY ENGINE
=================================
Advanced multi-timeframe trading with scalping, swing, and position strategies
for maximum profit opportunities across all market conditions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementations
    class RandomForestClassifier:
        def __init__(self, n_estimators=50, random_state=42):
            pass
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.random.choice([0, 1, 2], size=len(X))
        def predict_proba(self, X):
            return np.random.random((len(X), 3))

class TimeFrame(Enum):
    """Trading timeframes"""
    SCALP_1M = "1m"
    SCALP_5M = "5m"
    SWING_15M = "15m"
    SWING_1H = "1h"
    SWING_4H = "4h"
    POSITION_1D = "1d"
    POSITION_1W = "1w"

class StrategyType(Enum):
    """Strategy types"""
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"

@dataclass
class TimeFrameSignal:
    """Signal from a specific timeframe"""
    timeframe: TimeFrame
    strategy_type: StrategyType
    direction: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    expected_duration: int  # minutes
    risk_reward_ratio: float

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    timeframe: TimeFrame
    strategy_type: StrategyType
    enabled: bool = True
    weight: float = 1.0
    min_confidence: float = 0.6
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04

class ScalpingStrategy:
    """High-frequency scalping strategy for 1m-5m timeframes"""
    
    def __init__(self, timeframe: TimeFrame):
        self.timeframe = timeframe
        self.logger = logging.getLogger(f"{__name__}.Scalping.{timeframe.value}")
        
        # Scalping parameters
        self.min_volatility = 0.001  # Minimum volatility for scalping
        self.max_spread = 0.0005     # Maximum spread tolerance
        self.momentum_threshold = 0.002
        self.volume_multiplier = 1.5
        
    def analyze(self, price_data: List[float], volume_data: List[float] = None) -> TimeFrameSignal:
        """Analyze scalping opportunities"""
        try:
            if len(price_data) < 20:
                return self._neutral_signal(price_data[-1] if price_data else 0)
            
            prices = np.array(price_data)
            current_price = prices[-1]
            
            # Calculate indicators
            returns = np.diff(np.log(prices))
            volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0
            
            # Short-term momentum
            momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
            
            # Volume analysis (if available)
            volume_signal = 0
            if volume_data and len(volume_data) >= 10:
                volumes = np.array(volume_data)
                avg_volume = np.mean(volumes[-10:])
                current_volume = volumes[-1]
                volume_signal = min(2.0, current_volume / avg_volume) - 1  # -1 to 1
            
            # RSI-like momentum oscillator
            rsi = self._calculate_rsi(prices, period=14)
            
            # Signal generation
            signal_strength = 0
            direction = 'hold'
            
            # Bullish scalping conditions
            if (momentum_5 > self.momentum_threshold and 
                momentum_10 > 0 and 
                volatility > self.min_volatility and
                rsi < 70 and
                volume_signal > 0.2):
                
                direction = 'buy'
                signal_strength = min(1.0, (momentum_5 / self.momentum_threshold) * 0.4 + 
                                           (volume_signal + 1) * 0.3 + 
                                           (1 - rsi/100) * 0.3)
            
            # Bearish scalping conditions
            elif (momentum_5 < -self.momentum_threshold and 
                  momentum_10 < 0 and 
                  volatility > self.min_volatility and
                  rsi > 30 and
                  volume_signal > 0.2):
                
                direction = 'sell'
                signal_strength = min(1.0, abs(momentum_5) / self.momentum_threshold * 0.4 + 
                                           (volume_signal + 1) * 0.3 + 
                                           (rsi/100) * 0.3)
            
            # Calculate confidence
            confidence = min(1.0, signal_strength * volatility * 50)  # Higher vol = higher confidence for scalping
            
            # Calculate stops and targets
            stop_loss, take_profit = self._calculate_scalping_levels(current_price, direction, volatility)
            
            # Risk-reward ratio
            if direction != 'hold':
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward = reward / risk if risk > 0 else 0
            else:
                risk_reward = 0
            
            return TimeFrameSignal(
                timeframe=self.timeframe,
                strategy_type=StrategyType.SCALPING,
                direction=direction,
                strength=signal_strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                expected_duration=5 if self.timeframe == TimeFrame.SCALP_1M else 15,  # minutes
                risk_reward_ratio=risk_reward
            )
            
        except Exception as e:
            self.logger.error(f"❌ Scalping analysis failed: {e}")
            return self._neutral_signal(price_data[-1] if price_data else 0)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50  # Neutral
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_scalping_levels(self, price: float, direction: str, volatility: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit for scalping"""
        # Tight stops for scalping
        stop_distance = max(volatility * price * 2, price * 0.001)  # Min 0.1% stop
        profit_distance = stop_distance * 2  # 2:1 reward:risk
        
        if direction == 'buy':
            stop_loss = price - stop_distance
            take_profit = price + profit_distance
        elif direction == 'sell':
            stop_loss = price + stop_distance
            take_profit = price - profit_distance
        else:
            stop_loss = price * 0.99
            take_profit = price * 1.01
        
        return stop_loss, take_profit
    
    def _neutral_signal(self, price: float) -> TimeFrameSignal:
        """Return neutral signal"""
        return TimeFrameSignal(
            timeframe=self.timeframe,
            strategy_type=StrategyType.SCALPING,
            direction='hold',
            strength=0,
            confidence=0,
            entry_price=price,
            stop_loss=price * 0.99,
            take_profit=price * 1.01,
            expected_duration=5,
            risk_reward_ratio=0
        )

class SwingStrategy:
    """Medium-term swing trading strategy for 15m-4h timeframes"""
    
    def __init__(self, timeframe: TimeFrame):
        self.timeframe = timeframe
        self.logger = logging.getLogger(f"{__name__}.Swing.{timeframe.value}")
        
        # Swing parameters
        self.trend_threshold = 0.02
        self.pullback_threshold = 0.005
        self.breakout_threshold = 0.015
        
    def analyze(self, price_data: List[float], volume_data: List[float] = None) -> TimeFrameSignal:
        """Analyze swing trading opportunities"""
        try:
            if len(price_data) < 50:
                return self._neutral_signal(price_data[-1] if price_data else 0)
            
            prices = np.array(price_data)
            current_price = prices[-1]
            
            # Calculate moving averages
            ma_20 = np.mean(prices[-20:])
            ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else ma_20
            
            # Trend analysis
            short_trend = (ma_20 - ma_50) / ma_50
            long_trend = (prices[-1] - prices[-50]) / prices[-50] if len(prices) >= 50 else 0
            
            # Support/Resistance levels
            recent_high = np.max(prices[-20:])
            recent_low = np.min(prices[-20:])
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
            
            # MACD
            macd_line, signal_line = self._calculate_macd(prices)
            macd_histogram = macd_line - signal_line
            
            # Signal generation
            signal_strength = 0
            direction = 'hold'
            confidence = 0
            
            # Bullish swing conditions
            if (short_trend > self.trend_threshold and
                current_price > ma_20 and
                current_price < bb_upper * 0.98 and  # Not overbought
                macd_histogram > 0):
                
                direction = 'buy'
                signal_strength = min(1.0, short_trend / self.trend_threshold * 0.4 +
                                           (current_price - ma_20) / ma_20 * 10 * 0.3 +
                                           min(1.0, macd_histogram * 100) * 0.3)
                confidence = min(1.0, signal_strength * 1.2)
            
            # Bearish swing conditions
            elif (short_trend < -self.trend_threshold and
                  current_price < ma_20 and
                  current_price > bb_lower * 1.02 and  # Not oversold
                  macd_histogram < 0):
                
                direction = 'sell'
                signal_strength = min(1.0, abs(short_trend) / self.trend_threshold * 0.4 +
                                           (ma_20 - current_price) / ma_20 * 10 * 0.3 +
                                           min(1.0, abs(macd_histogram) * 100) * 0.3)
                confidence = min(1.0, signal_strength * 1.2)
            
            # Pullback opportunities
            elif abs(short_trend) > self.trend_threshold:
                pullback_ratio = (current_price - recent_low) / (recent_high - recent_low)
                
                if short_trend > 0 and pullback_ratio < 0.4:  # Bullish pullback
                    direction = 'buy'
                    signal_strength = 0.6
                    confidence = 0.7
                elif short_trend < 0 and pullback_ratio > 0.6:  # Bearish pullback
                    direction = 'sell'
                    signal_strength = 0.6
                    confidence = 0.7
            
            # Calculate stops and targets
            stop_loss, take_profit = self._calculate_swing_levels(current_price, direction, recent_high, recent_low)
            
            # Expected duration based on timeframe
            duration_map = {
                TimeFrame.SWING_15M: 60,   # 1 hour
                TimeFrame.SWING_1H: 240,   # 4 hours
                TimeFrame.SWING_4H: 1440   # 1 day
            }
            expected_duration = duration_map.get(self.timeframe, 240)
            
            # Risk-reward ratio
            if direction != 'hold':
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward = reward / risk if risk > 0 else 0
            else:
                risk_reward = 0
            
            return TimeFrameSignal(
                timeframe=self.timeframe,
                strategy_type=StrategyType.SWING,
                direction=direction,
                strength=signal_strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                expected_duration=expected_duration,
                risk_reward_ratio=risk_reward
            )
            
        except Exception as e:
            self.logger.error(f"❌ Swing analysis failed: {e}")
            return self._neutral_signal(price_data[-1] if price_data else 0)
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1] * 1.02, prices[-1] * 0.98
        
        ma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)
        
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return 0, 0
        
        # Exponential moving averages
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        if hasattr(self, '_macd_history'):
            self._macd_history.append(macd_line)
            if len(self._macd_history) > signal:
                self._macd_history = self._macd_history[-signal:]
        else:
            self._macd_history = [macd_line]
        
        signal_line = np.mean(self._macd_history)
        
        return macd_line, signal_line
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        
        for price in prices[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_swing_levels(self, price: float, direction: str, recent_high: float, recent_low: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit for swing trading"""
        range_size = recent_high - recent_low
        
        if direction == 'buy':
            stop_loss = recent_low - (range_size * 0.1)  # Below recent low
            take_profit = price + (range_size * 1.5)     # 1.5x range above entry
        elif direction == 'sell':
            stop_loss = recent_high + (range_size * 0.1)  # Above recent high
            take_profit = price - (range_size * 1.5)      # 1.5x range below entry
        else:
            stop_loss = price * 0.97
            take_profit = price * 1.06
        
        return stop_loss, take_profit
    
    def _neutral_signal(self, price: float) -> TimeFrameSignal:
        """Return neutral signal"""
        return TimeFrameSignal(
            timeframe=self.timeframe,
            strategy_type=StrategyType.SWING,
            direction='hold',
            strength=0,
            confidence=0,
            entry_price=price,
            stop_loss=price * 0.97,
            take_profit=price * 1.06,
            expected_duration=240,
            risk_reward_ratio=0
        )

class PositionStrategy:
    """Long-term position trading strategy for 1d-1w timeframes"""
    
    def __init__(self, timeframe: TimeFrame):
        self.timeframe = timeframe
        self.logger = logging.getLogger(f"{__name__}.Position.{timeframe.value}")
        
        # Position parameters
        self.major_trend_threshold = 0.05
        self.accumulation_threshold = 0.02
        
    def analyze(self, price_data: List[float], volume_data: List[float] = None) -> TimeFrameSignal:
        """Analyze position trading opportunities"""
        try:
            if len(price_data) < 100:
                return self._neutral_signal(price_data[-1] if price_data else 0)
            
            prices = np.array(price_data)
            current_price = prices[-1]
            
            # Long-term moving averages
            ma_50 = np.mean(prices[-50:])
            ma_100 = np.mean(prices[-100:])
            ma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else ma_100
            
            # Major trend analysis
            major_trend = (ma_50 - ma_200) / ma_200 if len(prices) >= 200 else 0
            intermediate_trend = (ma_50 - ma_100) / ma_100
            
            # Price position relative to MAs
            price_vs_ma50 = (current_price - ma_50) / ma_50
            price_vs_ma200 = (current_price - ma_200) / ma_200 if len(prices) >= 200 else 0
            
            # Long-term momentum
            momentum_50 = (prices[-1] - prices[-50]) / prices[-50]
            momentum_100 = (prices[-1] - prices[-100]) / prices[-100]
            
            # Volatility analysis
            returns = np.diff(np.log(prices))
            volatility = np.std(returns[-50:]) if len(returns) >= 50 else 0
            
            # Signal generation
            signal_strength = 0
            direction = 'hold'
            confidence = 0
            
            # Strong bullish position
            if (major_trend > self.major_trend_threshold and
                intermediate_trend > 0 and
                current_price > ma_50 and
                momentum_50 > 0 and
                momentum_100 > 0):
                
                direction = 'buy'
                signal_strength = min(1.0, major_trend / self.major_trend_threshold * 0.4 +
                                           momentum_50 * 5 * 0.3 +
                                           max(0, price_vs_ma50) * 10 * 0.3)
                confidence = min(1.0, signal_strength * 1.1)
            
            # Strong bearish position
            elif (major_trend < -self.major_trend_threshold and
                  intermediate_trend < 0 and
                  current_price < ma_50 and
                  momentum_50 < 0 and
                  momentum_100 < 0):
                
                direction = 'sell'
                signal_strength = min(1.0, abs(major_trend) / self.major_trend_threshold * 0.4 +
                                           abs(momentum_50) * 5 * 0.3 +
                                           max(0, -price_vs_ma50) * 10 * 0.3)
                confidence = min(1.0, signal_strength * 1.1)
            
            # Accumulation opportunities (trend reversal)
            elif (major_trend < 0 and intermediate_trend > -self.accumulation_threshold and
                  current_price > ma_50 and volatility < 0.03):
                
                direction = 'buy'
                signal_strength = 0.6
                confidence = 0.7
            
            # Distribution opportunities (trend reversal)
            elif (major_trend > 0 and intermediate_trend < self.accumulation_threshold and
                  current_price < ma_50 and volatility < 0.03):
                
                direction = 'sell'
                signal_strength = 0.6
                confidence = 0.7
            
            # Calculate stops and targets
            stop_loss, take_profit = self._calculate_position_levels(current_price, direction, ma_200, volatility)
            
            # Expected duration (weeks to months)
            duration_map = {
                TimeFrame.POSITION_1D: 10080,   # 1 week
                TimeFrame.POSITION_1W: 43200    # 1 month
            }
            expected_duration = duration_map.get(self.timeframe, 10080)
            
            # Risk-reward ratio
            if direction != 'hold':
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward = reward / risk if risk > 0 else 0
            else:
                risk_reward = 0
            
            return TimeFrameSignal(
                timeframe=self.timeframe,
                strategy_type=StrategyType.POSITION,
                direction=direction,
                strength=signal_strength,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                expected_duration=expected_duration,
                risk_reward_ratio=risk_reward
            )
            
        except Exception as e:
            self.logger.error(f"❌ Position analysis failed: {e}")
            return self._neutral_signal(price_data[-1] if price_data else 0)
    
    def _calculate_position_levels(self, price: float, direction: str, ma_200: float, volatility: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit for position trading"""
        # Wide stops for position trading
        volatility_stop = max(volatility * price * 10, price * 0.05)  # Min 5% stop
        
        if direction == 'buy':
            stop_loss = max(price - volatility_stop, ma_200 * 0.95)  # Below MA200
            take_profit = price * 1.5  # 50% target
        elif direction == 'sell':
            stop_loss = min(price + volatility_stop, ma_200 * 1.05)  # Above MA200
            take_profit = price * 0.7  # 30% decline target
        else:
            stop_loss = price * 0.9
            take_profit = price * 1.2
        
        return stop_loss, take_profit
    
    def _neutral_signal(self, price: float) -> TimeFrameSignal:
        """Return neutral signal"""
        return TimeFrameSignal(
            timeframe=self.timeframe,
            strategy_type=StrategyType.POSITION,
            direction='hold',
            strength=0,
            confidence=0,
            entry_price=price,
            stop_loss=price * 0.9,
            take_profit=price * 1.2,
            expected_duration=10080,
            risk_reward_ratio=0
        )

class MultiTimeframeEngine:
    """Main engine coordinating multiple timeframe strategies"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        
        # Initialize strategies
        self.strategies = {
            # Scalping strategies
            TimeFrame.SCALP_1M: ScalpingStrategy(TimeFrame.SCALP_1M),
            TimeFrame.SCALP_5M: ScalpingStrategy(TimeFrame.SCALP_5M),
            
            # Swing strategies
            TimeFrame.SWING_15M: SwingStrategy(TimeFrame.SWING_15M),
            TimeFrame.SWING_1H: SwingStrategy(TimeFrame.SWING_1H),
            TimeFrame.SWING_4H: SwingStrategy(TimeFrame.SWING_4H),
            
            # Position strategies
            TimeFrame.POSITION_1D: PositionStrategy(TimeFrame.POSITION_1D),
            TimeFrame.POSITION_1W: PositionStrategy(TimeFrame.POSITION_1W)
        }
        
        # Strategy configurations
        self.strategy_configs = {
            TimeFrame.SCALP_1M: StrategyConfig(TimeFrame.SCALP_1M, StrategyType.SCALPING, weight=0.8),
            TimeFrame.SCALP_5M: StrategyConfig(TimeFrame.SCALP_5M, StrategyType.SCALPING, weight=1.0),
            TimeFrame.SWING_15M: StrategyConfig(TimeFrame.SWING_15M, StrategyType.SWING, weight=1.2),
            TimeFrame.SWING_1H: StrategyConfig(TimeFrame.SWING_1H, StrategyType.SWING, weight=1.5),
            TimeFrame.SWING_4H: StrategyConfig(TimeFrame.SWING_4H, StrategyType.SWING, weight=1.8),
            TimeFrame.POSITION_1D: StrategyConfig(TimeFrame.POSITION_1D, StrategyType.POSITION, weight=2.0),
            TimeFrame.POSITION_1W: StrategyConfig(TimeFrame.POSITION_1W, StrategyType.POSITION, weight=2.5)
        }
        
        # ML ensemble
        self.sklearn_available = SKLEARN_AVAILABLE
        if self.sklearn_available:
            self.ensemble_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
        
        # Signal history
        self.signal_history = {}
        
        self.logger.info("⏰ Multi-Timeframe Engine initialized")
    
    def analyze_all_timeframes(self, symbol: str, price_data: Dict[str, List[float]], 
                              volume_data: Dict[str, List[float]] = None) -> Dict[TimeFrame, TimeFrameSignal]:
        """Analyze all timeframes and return signals"""
        signals = {}
        
        for timeframe, strategy in self.strategies.items():
            config = self.strategy_configs[timeframe]
            
            if not config.enabled:
                continue
            
            try:
                # Get appropriate data for timeframe
                tf_prices = price_data.get(timeframe.value, price_data.get('1m', []))
                tf_volumes = volume_data.get(timeframe.value, []) if volume_data else None
                
                # Analyze timeframe
                signal = strategy.analyze(tf_prices, tf_volumes)
                
                # Apply configuration filters
                if signal.confidence >= config.min_confidence:
                    signals[timeframe] = signal
                    
                    self.logger.debug(f"📊 {timeframe.value}: {signal.direction} "
                                    f"(strength: {signal.strength:.3f}, confidence: {signal.confidence:.3f})")
                
            except Exception as e:
                self.logger.error(f"❌ Analysis failed for {timeframe.value}: {e}")
        
        # Store signal history
        self.signal_history[symbol] = {
            'timestamp': datetime.now(),
            'signals': signals
        }
        
        return signals
    
    def get_ensemble_signal(self, symbol: str, timeframe_signals: Dict[TimeFrame, TimeFrameSignal]) -> Dict[str, Any]:
        """Generate ensemble signal from all timeframes"""
        try:
            if not timeframe_signals:
                return self._neutral_ensemble_signal()
            
            # Separate by strategy type
            scalping_signals = [s for s in timeframe_signals.values() if s.strategy_type == StrategyType.SCALPING]
            swing_signals = [s for s in timeframe_signals.values() if s.strategy_type == StrategyType.SWING]
            position_signals = [s for s in timeframe_signals.values() if s.strategy_type == StrategyType.POSITION]
            
            # Calculate weighted votes
            buy_votes = 0
            sell_votes = 0
            total_weight = 0
            
            for timeframe, signal in timeframe_signals.items():
                config = self.strategy_configs[timeframe]
                weight = config.weight * signal.confidence
                
                if signal.direction == 'buy':
                    buy_votes += weight
                elif signal.direction == 'sell':
                    sell_votes += weight
                
                total_weight += weight
            
            # Determine ensemble direction
            if total_weight == 0:
                return self._neutral_ensemble_signal()
            
            buy_ratio = buy_votes / total_weight
            sell_ratio = sell_votes / total_weight
            
            if buy_ratio > 0.6:
                direction = 'buy'
                strength = buy_ratio
            elif sell_ratio > 0.6:
                direction = 'sell'
                strength = sell_ratio
            else:
                direction = 'hold'
                strength = max(buy_ratio, sell_ratio)
            
            # Calculate ensemble confidence
            signal_count = len(timeframe_signals)
            agreement_bonus = 0.1 * signal_count if signal_count > 1 else 0
            confidence = min(1.0, strength + agreement_bonus)
            
            # Select primary signal for execution parameters
            primary_signal = self._select_primary_signal(timeframe_signals, direction)
            
            # Risk-reward analysis
            avg_risk_reward = np.mean([s.risk_reward_ratio for s in timeframe_signals.values() if s.risk_reward_ratio > 0])
            
            ensemble_signal = {
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'entry_price': primary_signal.entry_price if primary_signal else 0,
                'stop_loss': primary_signal.stop_loss if primary_signal else 0,
                'take_profit': primary_signal.take_profit if primary_signal else 0,
                'expected_duration': primary_signal.expected_duration if primary_signal else 0,
                'risk_reward_ratio': avg_risk_reward if not np.isnan(avg_risk_reward) else 0,
                'timeframe_breakdown': {
                    'scalping': len(scalping_signals),
                    'swing': len(swing_signals),
                    'position': len(position_signals)
                },
                'signal_count': signal_count,
                'primary_timeframe': primary_signal.timeframe.value if primary_signal else 'none',
                'strategy_consensus': self._calculate_strategy_consensus(scalping_signals, swing_signals, position_signals)
            }
            
            self.logger.info(f"⏰ Ensemble signal for {symbol}: {direction} "
                           f"(strength: {strength:.3f}, confidence: {confidence:.3f}, "
                           f"signals: {signal_count})")
            
            return ensemble_signal
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble signal generation failed: {e}")
            return self._neutral_ensemble_signal()
    
    def _select_primary_signal(self, signals: Dict[TimeFrame, TimeFrameSignal], direction: str) -> Optional[TimeFrameSignal]:
        """Select the primary signal for execution parameters"""
        # Filter signals matching the ensemble direction
        matching_signals = [s for s in signals.values() if s.direction == direction]
        
        if not matching_signals:
            return None
        
        # Prioritize by timeframe importance and confidence
        scored_signals = []
        for signal in matching_signals:
            config = self.strategy_configs[signal.timeframe]
            score = config.weight * signal.confidence * signal.strength
            scored_signals.append((score, signal))
        
        # Return highest scored signal
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        return scored_signals[0][1]
    
    def _calculate_strategy_consensus(self, scalping: List, swing: List, position: List) -> Dict[str, str]:
        """Calculate consensus within each strategy type"""
        def get_consensus(signals):
            if not signals:
                return 'none'
            
            buy_count = sum(1 for s in signals if s.direction == 'buy')
            sell_count = sum(1 for s in signals if s.direction == 'sell')
            
            if buy_count > sell_count:
                return 'bullish'
            elif sell_count > buy_count:
                return 'bearish'
            else:
                return 'neutral'
        
        return {
            'scalping': get_consensus(scalping),
            'swing': get_consensus(swing),
            'position': get_consensus(position)
        }
    
    def _neutral_ensemble_signal(self) -> Dict[str, Any]:
        """Return neutral ensemble signal"""
        return {
            'direction': 'hold',
            'strength': 0,
            'confidence': 0,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'expected_duration': 0,
            'risk_reward_ratio': 0,
            'timeframe_breakdown': {'scalping': 0, 'swing': 0, 'position': 0},
            'signal_count': 0,
            'primary_timeframe': 'none',
            'strategy_consensus': {'scalping': 'none', 'swing': 'none', 'position': 'none'}
        }
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance summary of all strategies"""
        performance = {
            'total_strategies': len(self.strategies),
            'enabled_strategies': sum(1 for config in self.strategy_configs.values() if config.enabled),
            'timeframe_weights': {tf.value: config.weight for tf, config in self.strategy_configs.items()},
            'recent_signals': len(self.signal_history),
            'ml_available': self.sklearn_available
        }
        
        return performance

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize multi-timeframe engine
    engine = MultiTimeframeEngine()
    
    # Example price data for different timeframes
    price_data = {
        '1m': [45000 + np.random.normal(0, 100) for _ in range(100)],
        '5m': [45000 + np.random.normal(0, 200) for _ in range(50)],
        '1h': [45000 + np.random.normal(0, 500) for _ in range(100)],
        '1d': [45000 + np.random.normal(0, 1000) for _ in range(200)]
    }
    
    # Analyze all timeframes
    signals = engine.analyze_all_timeframes('BTC', price_data)
    ensemble = engine.get_ensemble_signal('BTC', signals)
    
    print(f"Timeframe Signals: {len(signals)}")
    print(f"Ensemble Signal: {ensemble}")
    print(f"Performance: {engine.get_strategy_performance()}") 