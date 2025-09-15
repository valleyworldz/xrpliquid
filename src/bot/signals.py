#!/usr/bin/env python3
"""
Signal Generation and Pattern Analysis for XRP Trading Bot
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

class SignalGenerator:
    """Advanced signal generation with pattern recognition"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Technical indicator parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.rsi_period = 14
        self.atr_period = 14
        
        # Pattern weights (adaptive)
        self.pattern_weights = {
            'trend_following': 0.4,
            'momentum': 0.3,
            'reversal': 0.2,
            'volatility': 0.1
        }
        
        # Performance tracking
        self.signal_history = []
        self.pattern_performance = {}
        
    def analyze_market_signals(self, price_history: List[float], 
                             volume_history: Optional[List[float]] = None,
                             current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive market signal analysis
        
        Args:
            price_history: List of historical prices
            volume_history: Optional volume data
            current_price: Current market price
            
        Returns:
            Dictionary with signal analysis results
        """
        try:
            if len(price_history) < 50:
                return {'signal': None, 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Calculate technical indicators
            macd_line, signal_line, histogram = self.calculate_macd(price_history)
            rsi = self.calculate_rsi(price_history)
            atr = self.calculate_atr(price_history)
            momentum = self.calculate_momentum(price_history)
            
            # Pattern analysis
            patterns = self.analyze_patterns(price_history, volume_history)
            
            # Signal generation
            signal = self.generate_signal(
                macd_line, signal_line, histogram,
                rsi, atr, momentum, patterns,
                current_price or price_history[-1]
            )
            
            # Update performance tracking
            self.update_signal_history(signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in market signal analysis: {e}")
            return {'signal': None, 'confidence': 0.0, 'reason': f'Analysis error: {e}'}
    
    def calculate_macd(self, prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD series for trend confirmation"""
        try:
            if len(prices) < self.macd_slow:
                return [], [], []
            
            # Calculate EMAs for the entire series
            ema_fast_values = []
            ema_slow_values = []
            
            for i in range(len(prices)):
                if i < self.macd_fast - 1:
                    ema_fast_values.append(prices[i])
                else:
                    ema_fast = self._calculate_ema(prices[:i+1], self.macd_fast)
                    ema_fast_values.append(ema_fast)
                
                if i < self.macd_slow - 1:
                    ema_slow_values.append(prices[i])
                else:
                    ema_slow = self._calculate_ema(prices[:i+1], self.macd_slow)
                    ema_slow_values.append(ema_slow)
            
            # Calculate MACD line series
            macd_line = [fast - slow for fast, slow in zip(ema_fast_values, ema_slow_values)]
            
            # Calculate signal line (EMA of MACD)
            signal_line = []
            for i in range(len(macd_line)):
                if i < self.macd_signal - 1:
                    signal_line.append(macd_line[i])
                else:
                    signal = self._calculate_ema(macd_line[:i+1], self.macd_signal)
                    signal_line.append(signal)
            
            # Calculate histogram series
            histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return [], [], []
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def calculate_momentum(self, prices: List[float], period: int = 5) -> float:
        """Calculate price momentum"""
        try:
            if len(prices) < period:
                return 0.0
            
            current_price = prices[-1]
            past_price = prices[-period]
            
            return (current_price - past_price) / past_price
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            true_ranges = []
            for i in range(1, len(prices)):
                high = prices[i]
                low = prices[i]
                prev_close = prices[i-1]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            # Calculate ATR as EMA of true ranges
            atr = self._calculate_ema(true_ranges, period)
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    def analyze_patterns(self, prices: List[float], 
                        volume_history: Optional[List[float]] = None) -> Dict[str, float]:
        """Analyze price patterns for trading signals"""
        patterns = {}
        
        try:
            # Trend analysis
            patterns['trend_following'] = self._analyze_trend(prices)
            
            # Momentum analysis
            patterns['momentum'] = self._analyze_momentum(prices)
            
            # Reversal analysis
            patterns['reversal'] = self._analyze_reversal(prices)
            
            # Volatility analysis
            patterns['volatility'] = self._analyze_volatility(prices)
            
            # Volume confirmation (if available)
            if volume_history and len(volume_history) >= len(prices):
                patterns['volume_confirmation'] = self._analyze_volume_confirmation(prices, volume_history)
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {e}")
        
        return patterns
    
    def _analyze_trend(self, prices: List[float]) -> float:
        """Analyze trend strength"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Calculate trend using linear regression
            x = np.arange(len(prices))
            y = np.array(prices)
            
            slope, _ = np.polyfit(x, y, 1)
            
            # Normalize slope to confidence score
            price_range = max(prices) - min(prices)
            if price_range == 0:
                return 0.0
            
            trend_strength = abs(slope) / price_range * len(prices)
            return min(trend_strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return 0.0
    
    def _analyze_momentum(self, prices: List[float]) -> float:
        """Analyze momentum strength"""
        try:
            if len(prices) < 10:
                return 0.0
            
            # Calculate momentum over different periods
            short_momentum = self.calculate_momentum(prices, 5)
            medium_momentum = self.calculate_momentum(prices, 10)
            
            # Combine momentum signals
            momentum_score = (short_momentum + medium_momentum) / 2
            return min(abs(momentum_score), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return 0.0
    
    def _analyze_reversal(self, prices: List[float]) -> float:
        """Analyze reversal signals"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Look for divergence between price and momentum
            momentum = self.calculate_momentum(prices, 5)
            price_change = (prices[-1] - prices[-5]) / prices[-5]
            
            # Reversal signal when momentum and price diverge
            if (momentum > 0 and price_change < 0) or (momentum < 0 and price_change > 0):
                return abs(momentum)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error analyzing reversal: {e}")
            return 0.0
    
    def _analyze_volatility(self, prices: List[float]) -> float:
        """Analyze volatility patterns"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Calculate rolling volatility
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(returns[-20:])
            
            # Normalize volatility
            return min(volatility * 10, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            return 0.0
    
    def _analyze_volume_confirmation(self, prices: List[float], 
                                   volume_history: List[float]) -> float:
        """Analyze volume confirmation"""
        try:
            if len(volume_history) < 10:
                return 0.0
            
            # Check if volume supports price movement
            price_change = (prices[-1] - prices[-5]) / prices[-5]
            volume_change = (volume_history[-1] - volume_history[-5]) / volume_history[-5]
            
            # Volume confirmation when both move in same direction
            if (price_change > 0 and volume_change > 0) or (price_change < 0 and volume_change > 0):
                return min(abs(volume_change), 1.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume confirmation: {e}")
            return 0.0
    
    def generate_signal(self, macd_line: List[float], signal_line: List[float],
                       histogram: List[float], rsi: float, atr: float,
                       momentum: float, patterns: Dict[str, float],
                       current_price: float) -> Dict[str, Any]:
        """Generate trading signal from indicators"""
        try:
            if not macd_line or not signal_line:
                return {'signal': None, 'confidence': 0.0, 'reason': 'Insufficient MACD data'}
            
            # Get latest values
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            current_histogram = histogram[-1] if histogram else 0
            
            # Calculate signal strength
            signal_strength = 0.0
            signal_type = None
            reasons = []
            
            # MACD analysis
            if current_macd > current_signal and current_histogram > 0:
                signal_strength += 0.3
                signal_type = 'LONG'
                reasons.append('MACD bullish crossover')
            elif current_macd < current_signal and current_histogram < 0:
                signal_strength += 0.3
                signal_type = 'SHORT'
                reasons.append('MACD bearish crossover')
            
            # RSI analysis
            if rsi < 30:
                signal_strength += 0.2
                if signal_type != 'SHORT':
                    signal_type = 'LONG'
                reasons.append('RSI oversold')
            elif rsi > 70:
                signal_strength += 0.2
                if signal_type != 'LONG':
                    signal_type = 'SHORT'
                reasons.append('RSI overbought')
            
            # Momentum analysis
            if momentum > 0.02:
                signal_strength += 0.15
                if signal_type != 'SHORT':
                    signal_type = 'LONG'
                reasons.append('Positive momentum')
            elif momentum < -0.02:
                signal_strength += 0.15
                if signal_type != 'LONG':
                    signal_type = 'SHORT'
                reasons.append('Negative momentum')
            
            # Pattern analysis
            for pattern_type, strength in patterns.items():
                if strength > 0.5:
                    signal_strength += strength * 0.1
                    reasons.append(f'{pattern_type} pattern')
            
            # Normalize confidence
            confidence = min(signal_strength, 1.0)
            
            # Minimum confidence threshold
            if confidence < 0.6:
                return {'signal': None, 'confidence': confidence, 'reason': 'Low confidence'}
            
            return {
                'signal': signal_type,
                'confidence': confidence,
                'reasons': reasons,
                'indicators': {
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': current_histogram,
                    'rsi': rsi,
                    'atr': atr,
                    'momentum': momentum
                },
                'patterns': patterns,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {'signal': None, 'confidence': 0.0, 'reason': f'Signal generation error: {e}'}
    
    def update_signal_history(self, signal: Dict[str, Any]) -> None:
        """Update signal history for performance tracking"""
        if signal and signal.get('signal'):
            self.signal_history.append(signal)
            
            # Keep only last 100 signals
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
    
    def adapt_pattern_weights(self) -> None:
        """Adapt pattern weights based on performance"""
        try:
            if len(self.signal_history) < 20:
                return
            
            # Analyze pattern performance
            pattern_performance = {}
            for pattern_type in self.pattern_weights.keys():
                correct_signals = 0
                total_signals = 0
                
                for signal in self.signal_history[-20:]:
                    if pattern_type in signal.get('patterns', {}):
                        total_signals += 1
                        # This would need actual trade results to determine correctness
                        # For now, we'll use a simple heuristic
                        if signal.get('confidence', 0) > 0.7:
                            correct_signals += 1
                
                if total_signals > 0:
                    pattern_performance[pattern_type] = correct_signals / total_signals
            
            # Update weights based on performance
            for pattern_type, performance in pattern_performance.items():
                if pattern_type in self.pattern_weights:
                    # Increase weight for better performing patterns
                    self.pattern_weights[pattern_type] *= (1 + performance * 0.1)
            
            # Normalize weights
            total_weight = sum(self.pattern_weights.values())
            for pattern_type in self.pattern_weights:
                self.pattern_weights[pattern_type] /= total_weight
            
            self.logger.info(f"Adapted pattern weights: {self.pattern_weights}")
            
        except Exception as e:
            self.logger.error(f"Error adapting pattern weights: {e}")

# Convenience function for backward compatibility
def analyze_xrp_signals(price_history: List[float], 
                       volume_history: Optional[List[float]] = None,
                       current_price: Optional[float] = None,
                       logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Analyze XRP market signals (convenience function)"""
    signal_generator = SignalGenerator(logger)
    return signal_generator.analyze_market_signals(price_history, volume_history, current_price) 