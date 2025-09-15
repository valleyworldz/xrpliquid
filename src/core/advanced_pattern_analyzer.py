#!/usr/bin/env python3
"""
Advanced Pattern Analyzer - Tier 2 Component
============================================

This module implements advanced pattern recognition for market analysis,
including candlestick patterns, chart patterns, and technical indicators
for enhanced trading signals.

Features:
- Multi-timeframe pattern analysis
- Candlestick pattern recognition
- Chart pattern detection (triangles, flags, etc.)
- Technical indicator patterns
- Pattern strength scoring
- Real-time pattern alerts
"""

import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import logging

class AdvancedPatternAnalyzer:
    """Advanced pattern recognition system for market analysis"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Pattern recognition parameters
        self.min_pattern_bars = 5
        self.max_pattern_bars = 50
        self.pattern_confidence_threshold = 0.7
        
        # Pattern storage
        self.detected_patterns = {}
        self.pattern_history = deque(maxlen=1000)
        self.pattern_performance = defaultdict(lambda: {
            'success_count': 0,
            'total_count': 0,
            'avg_profit': 0.0,
            'success_rate': 0.0
        })
        
        # Technical indicators
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_period = 20
        self.bollinger_std = 2
        
        # Pattern definitions
        self.candlestick_patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'shooting_star': self._detect_shooting_star,
            'engulfing': self._detect_engulfing,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star,
            'three_white_soldiers': self._detect_three_white_soldiers,
            'three_black_crows': self._detect_three_black_crows
        }
        
        self.chart_patterns = {
            'triangle': self._detect_triangle,
            'flag': self._detect_flag,
            'wedge': self._detect_wedge,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'head_shoulders': self._detect_head_shoulders
        }
        
        self.logger.info("[PATTERN] Advanced Pattern Analyzer initialized")
    
    def analyze_patterns(self, symbol: str, price_data: List[float], 
                        volume_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """Comprehensive pattern analysis for a symbol"""
        try:
            if len(price_data) < self.min_pattern_bars:
                return {'status': 'insufficient_data', 'patterns': []}
            
            # Prepare OHLC data (simplified - using price as close)
            ohlc_data = self._prepare_ohlc_data(price_data)
            
            # Detect candlestick patterns
            candlestick_patterns = self._detect_candlestick_patterns(ohlc_data)
            
            # Detect chart patterns
            chart_patterns = self._detect_chart_patterns(price_data)
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(price_data, volume_data)
            
            # Combine all patterns
            all_patterns = candlestick_patterns + chart_patterns
            
            # Score patterns
            scored_patterns = self._score_patterns(all_patterns, technical_indicators)
            
            # Filter high-confidence patterns
            high_confidence_patterns = [
                p for p in scored_patterns 
                if p['confidence'] >= self.pattern_confidence_threshold
            ]
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(high_confidence_patterns, technical_indicators)
            
            # Update pattern history
            self._update_pattern_history(symbol, high_confidence_patterns)
            
            result = {
                'status': 'success',
                'symbol': symbol,
                'patterns': high_confidence_patterns,
                'technical_indicators': technical_indicators,
                'trading_signals': trading_signals,
                'pattern_count': len(high_confidence_patterns),
                'signal_strength': self._calculate_signal_strength(trading_signals)
            }
            
            self.logger.info(f"[PATTERN] {symbol}: {len(high_confidence_patterns)} patterns detected, "
                           f"signal strength: {result['signal_strength']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PATTERN] Error analyzing patterns for {symbol}: {e}")
            return {'status': 'error', 'error': str(e), 'patterns': []}
    
    def _prepare_ohlc_data(self, price_data: List[float]) -> List[Dict]:
        """Prepare OHLC data from price data (simplified)"""
        ohlc_data = []
        
        for i, close_price in enumerate(price_data):
            # Simplified OHLC - in real implementation, you'd have actual OHLC data
            high = close_price * (1 + 0.02)  # 2% range
            low = close_price * (1 - 0.02)   # 2% range
            open_price = close_price * (1 + (0.01 if i % 2 == 0 else -0.01))  # Alternating opens
            
            ohlc_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': 1.0  # Default volume
            })
        
        return ohlc_data
    
    def _detect_candlestick_patterns(self, ohlc_data: List[Dict]) -> List[Dict]:
        """Detect candlestick patterns"""
        patterns = []
        
        if len(ohlc_data) < 3:
            return patterns
        
        # Check each pattern type
        for pattern_name, pattern_func in self.candlestick_patterns.items():
            try:
                pattern_result = pattern_func(ohlc_data)
                if pattern_result:
                    patterns.append(pattern_result)
            except Exception as e:
                self.logger.error(f"[PATTERN] Error detecting {pattern_name}: {e}")
        
        return patterns
    
    def _detect_doji(self, ohlc_data: List[Dict]) -> Optional[Dict]:
        """Detect doji pattern"""
        if len(ohlc_data) < 1:
            return None
        
        current = ohlc_data[-1]
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        
        # Doji: small body relative to total range
        if total_range > 0 and body_size / total_range < 0.1:
            return {
                'type': 'candlestick',
                'pattern': 'doji',
                'direction': 'neutral',
                'confidence': 0.8,
                'position': len(ohlc_data) - 1,
                'description': 'Doji pattern detected - indecision in market'
            }
        
        return None
    
    def _detect_hammer(self, ohlc_data: List[Dict]) -> Optional[Dict]:
        """Detect hammer pattern"""
        if len(ohlc_data) < 1:
            return None
        
        current = ohlc_data[-1]
        body_size = abs(current['close'] - current['open'])
        lower_shadow = min(current['open'], current['close']) - current['low']
        upper_shadow = current['high'] - max(current['open'], current['close'])
        
        # Hammer: small body, long lower shadow, small upper shadow
        if (body_size > 0 and lower_shadow > 2 * body_size and 
            upper_shadow < 0.5 * body_size):
            return {
                'type': 'candlestick',
                'pattern': 'hammer',
                'direction': 'bullish',
                'confidence': 0.75,
                'position': len(ohlc_data) - 1,
                'description': 'Hammer pattern detected - potential reversal'
            }
        
        return None
    
    def _detect_shooting_star(self, ohlc_data: List[Dict]) -> Optional[Dict]:
        """Detect shooting star pattern"""
        if len(ohlc_data) < 1:
            return None
        
        current = ohlc_data[-1]
        body_size = abs(current['close'] - current['open'])
        lower_shadow = min(current['open'], current['close']) - current['low']
        upper_shadow = current['high'] - max(current['open'], current['close'])
        
        # Shooting star: small body, long upper shadow, small lower shadow
        if (body_size > 0 and upper_shadow > 2 * body_size and 
            lower_shadow < 0.5 * body_size):
            return {
                'type': 'candlestick',
                'pattern': 'shooting_star',
                'direction': 'bearish',
                'confidence': 0.75,
                'position': len(ohlc_data) - 1,
                'description': 'Shooting star pattern detected - potential reversal'
            }
        
        return None
    
    def _detect_engulfing(self, ohlc_data: List[Dict]) -> Optional[Dict]:
        """Detect engulfing pattern"""
        if len(ohlc_data) < 2:
            return None
        
        current = ohlc_data[-1]
        previous = ohlc_data[-2]
        
        current_body = abs(current['close'] - current['open'])
        previous_body = abs(previous['close'] - previous['open'])
        
        # Bullish engulfing
        if (current['close'] > current['open'] and  # Current is bullish
            previous['close'] < previous['open'] and  # Previous is bearish
            current['open'] < previous['close'] and  # Current opens below previous close
            current['close'] > previous['open'] and  # Current closes above previous open
            current_body > previous_body):  # Current body engulfs previous
            
            return {
                'type': 'candlestick',
                'pattern': 'bullish_engulfing',
                'direction': 'bullish',
                'confidence': 0.8,
                'position': len(ohlc_data) - 1,
                'description': 'Bullish engulfing pattern detected'
            }
        
        # Bearish engulfing
        elif (current['close'] < current['open'] and  # Current is bearish
              previous['close'] > previous['open'] and  # Previous is bullish
              current['open'] > previous['close'] and  # Current opens above previous close
              current['close'] < previous['open'] and  # Current closes below previous open
              current_body > previous_body):  # Current body engulfs previous
            
            return {
                'type': 'candlestick',
                'pattern': 'bearish_engulfing',
                'direction': 'bearish',
                'confidence': 0.8,
                'position': len(ohlc_data) - 1,
                'description': 'Bearish engulfing pattern detected'
            }
        
        return None
    
    def _detect_morning_star(self, ohlc_data: List[Dict]) -> Optional[Dict]:
        """Detect morning star pattern"""
        if len(ohlc_data) < 3:
            return None
        
        # Simplified morning star detection
        first = ohlc_data[-3]
        second = ohlc_data[-2]
        third = ohlc_data[-1]
        
        # Morning star: bearish, small body, bullish
        if (first['close'] < first['open'] and  # First is bearish
            abs(second['close'] - second['open']) < 0.01 and  # Second is small body
            third['close'] > third['open']):  # Third is bullish
            
            return {
                'type': 'candlestick',
                'pattern': 'morning_star',
                'direction': 'bullish',
                'confidence': 0.85,
                'position': len(ohlc_data) - 1,
                'description': 'Morning star pattern detected - strong reversal signal'
            }
        
        return None
    
    def _detect_evening_star(self, ohlc_data: List[Dict]) -> Optional[Dict]:
        """Detect evening star pattern"""
        if len(ohlc_data) < 3:
            return None
        
        # Simplified evening star detection
        first = ohlc_data[-3]
        second = ohlc_data[-2]
        third = ohlc_data[-1]
        
        # Evening star: bullish, small body, bearish
        if (first['close'] > first['open'] and  # First is bullish
            abs(second['close'] - second['open']) < 0.01 and  # Second is small body
            third['close'] < third['open']):  # Third is bearish
            
            return {
                'type': 'candlestick',
                'pattern': 'evening_star',
                'direction': 'bearish',
                'confidence': 0.85,
                'position': len(ohlc_data) - 1,
                'description': 'Evening star pattern detected - strong reversal signal'
            }
        
        return None
    
    def _detect_three_white_soldiers(self, ohlc_data: List[Dict]) -> Optional[Dict]:
        """Detect three white soldiers pattern"""
        if len(ohlc_data) < 3:
            return None
        
        # Check last three candles
        soldiers = ohlc_data[-3:]
        
        # All should be bullish with increasing closes
        if all(s['close'] > s['open'] for s in soldiers):
            if (soldiers[0]['close'] < soldiers[1]['close'] < soldiers[2]['close']):
                return {
                    'type': 'candlestick',
                    'pattern': 'three_white_soldiers',
                    'direction': 'bullish',
                    'confidence': 0.9,
                    'position': len(ohlc_data) - 1,
                    'description': 'Three white soldiers pattern detected - strong uptrend'
                }
        
        return None
    
    def _detect_three_black_crows(self, ohlc_data: List[Dict]) -> Optional[Dict]:
        """Detect three black crows pattern"""
        if len(ohlc_data) < 3:
            return None
        
        # Check last three candles
        crows = ohlc_data[-3:]
        
        # All should be bearish with decreasing closes
        if all(s['close'] < s['open'] for s in crows):
            if (crows[0]['close'] > crows[1]['close'] > crows[2]['close']):
                return {
                    'type': 'candlestick',
                    'pattern': 'three_black_crows',
                    'direction': 'bearish',
                    'confidence': 0.9,
                    'position': len(ohlc_data) - 1,
                    'description': 'Three black crows pattern detected - strong downtrend'
                }
        
        return None
    
    def _detect_chart_patterns(self, price_data: List[float]) -> List[Dict]:
        """Detect chart patterns"""
        patterns = []
        
        if len(price_data) < 10:
            return patterns
        
        # Check each chart pattern type
        for pattern_name, pattern_func in self.chart_patterns.items():
            try:
                pattern_result = pattern_func(price_data)
                if pattern_result:
                    patterns.append(pattern_result)
            except Exception as e:
                self.logger.error(f"[PATTERN] Error detecting chart pattern {pattern_name}: {e}")
        
        return patterns
    
    def _detect_triangle(self, price_data: List[float]) -> Optional[Dict]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(price_data) < 10:
            return None
        
        # Simplified triangle detection
        # Look for converging trend lines
        highs = []
        lows = []
        
        # Find local highs and lows
        for i in range(1, len(price_data) - 1):
            if price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                highs.append((i, price_data[i]))
            elif price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1]:
                lows.append((i, price_data[i]))
        
        if len(highs) >= 3 and len(lows) >= 3:
            # Check for converging lines
            high_slope = self._calculate_slope([h[0] for h in highs[-3:]], [h[1] for h in highs[-3:]])
            low_slope = self._calculate_slope([l[0] for l in lows[-3:]], [l[1] for l in lows[-3:]])
            
            if abs(high_slope) < 0.1 and abs(low_slope) < 0.1:
                # Symmetrical triangle
                return {
                    'type': 'chart',
                    'pattern': 'symmetrical_triangle',
                    'direction': 'neutral',
                    'confidence': 0.7,
                    'position': len(price_data) - 1,
                    'description': 'Symmetrical triangle pattern detected'
                }
            elif high_slope < -0.05 and low_slope > 0.05:
                # Ascending triangle
                return {
                    'type': 'chart',
                    'pattern': 'ascending_triangle',
                    'direction': 'bullish',
                    'confidence': 0.75,
                    'position': len(price_data) - 1,
                    'description': 'Ascending triangle pattern detected'
                }
            elif high_slope < -0.05 and low_slope < -0.05:
                # Descending triangle
                return {
                    'type': 'chart',
                    'pattern': 'descending_triangle',
                    'direction': 'bearish',
                    'confidence': 0.75,
                    'position': len(price_data) - 1,
                    'description': 'Descending triangle pattern detected'
                }
        
        return None
    
    def _detect_flag(self, price_data: List[float]) -> Optional[Dict]:
        """Detect flag patterns"""
        if len(price_data) < 15:
            return None
        
        # Simplified flag detection
        # Look for strong move followed by consolidation
        recent_prices = price_data[-15:]
        
        # Check for strong initial move
        initial_move = abs(recent_prices[0] - recent_prices[5]) / recent_prices[0]
        
        if initial_move > 0.05:  # 5% move
            # Check for consolidation
            consolidation_range = max(recent_prices[5:]) - min(recent_prices[5:])
            consolidation_pct = consolidation_range / recent_prices[5]
            
            if consolidation_pct < 0.03:  # Less than 3% consolidation
                direction = 'bullish' if recent_prices[5] > recent_prices[0] else 'bearish'
                return {
                    'type': 'chart',
                    'pattern': 'flag',
                    'direction': direction,
                    'confidence': 0.8,
                    'position': len(price_data) - 1,
                    'description': f'{direction.capitalize()} flag pattern detected'
                }
        
        return None
    
    def _detect_wedge(self, price_data: List[float]) -> Optional[Dict]:
        """Detect wedge patterns"""
        if len(price_data) < 10:
            return None
        
        # Simplified wedge detection
        # Look for converging trend lines with same direction
        highs = []
        lows = []
        
        for i in range(1, len(price_data) - 1):
            if price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                highs.append((i, price_data[i]))
            elif price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1]:
                lows.append((i, price_data[i]))
        
        if len(highs) >= 3 and len(lows) >= 3:
            high_slope = self._calculate_slope([h[0] for h in highs[-3:]], [h[1] for h in highs[-3:]])
            low_slope = self._calculate_slope([l[0] for l in lows[-3:]], [l[1] for l in lows[-3:]])
            
            if high_slope < -0.05 and low_slope < -0.05:
                # Rising wedge (bearish)
                return {
                    'type': 'chart',
                    'pattern': 'rising_wedge',
                    'direction': 'bearish',
                    'confidence': 0.7,
                    'position': len(price_data) - 1,
                    'description': 'Rising wedge pattern detected'
                }
            elif high_slope > 0.05 and low_slope > 0.05:
                # Falling wedge (bullish)
                return {
                    'type': 'chart',
                    'pattern': 'falling_wedge',
                    'direction': 'bullish',
                    'confidence': 0.7,
                    'position': len(price_data) - 1,
                    'description': 'Falling wedge pattern detected'
                }
        
        return None
    
    def _detect_double_top(self, price_data: List[float]) -> Optional[Dict]:
        """Detect double top pattern"""
        if len(price_data) < 20:
            return None
        
        # Simplified double top detection
        # Look for two peaks at similar levels
        peaks = []
        
        for i in range(1, len(price_data) - 1):
            if price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                peaks.append((i, price_data[i]))
        
        if len(peaks) >= 2:
            # Check last two peaks
            peak1, peak2 = peaks[-2], peaks[-1]
            
            # Peaks should be at similar levels (within 2%)
            price_diff = abs(peak1[1] - peak2[1]) / peak1[1]
            
            if price_diff < 0.02 and peak2[0] - peak1[0] > 5:
                return {
                    'type': 'chart',
                    'pattern': 'double_top',
                    'direction': 'bearish',
                    'confidence': 0.8,
                    'position': len(price_data) - 1,
                    'description': 'Double top pattern detected'
                }
        
        return None
    
    def _detect_double_bottom(self, price_data: List[float]) -> Optional[Dict]:
        """Detect double bottom pattern"""
        if len(price_data) < 20:
            return None
        
        # Simplified double bottom detection
        # Look for two troughs at similar levels
        troughs = []
        
        for i in range(1, len(price_data) - 1):
            if price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1]:
                troughs.append((i, price_data[i]))
        
        if len(troughs) >= 2:
            # Check last two troughs
            trough1, trough2 = troughs[-2], troughs[-1]
            
            # Troughs should be at similar levels (within 2%)
            price_diff = abs(trough1[1] - trough2[1]) / trough1[1]
            
            if price_diff < 0.02 and trough2[0] - trough1[0] > 5:
                return {
                    'type': 'chart',
                    'pattern': 'double_bottom',
                    'direction': 'bullish',
                    'confidence': 0.8,
                    'position': len(price_data) - 1,
                    'description': 'Double bottom pattern detected'
                }
        
        return None
    
    def _detect_head_shoulders(self, price_data: List[float]) -> Optional[Dict]:
        """Detect head and shoulders pattern"""
        if len(price_data) < 25:
            return None
        
        # Simplified head and shoulders detection
        # Look for three peaks with middle peak higher
        peaks = []
        
        for i in range(1, len(price_data) - 1):
            if price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                peaks.append((i, price_data[i]))
        
        if len(peaks) >= 3:
            # Check last three peaks
            left_shoulder, head, right_shoulder = peaks[-3], peaks[-2], peaks[-1]
            
            # Head should be higher than shoulders
            if (head[1] > left_shoulder[1] and head[1] > right_shoulder[1] and
                abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05):
                
                return {
                    'type': 'chart',
                    'pattern': 'head_shoulders',
                    'direction': 'bearish',
                    'confidence': 0.85,
                    'position': len(price_data) - 1,
                    'description': 'Head and shoulders pattern detected'
                }
        
        return None
    
    def _calculate_technical_indicators(self, price_data: List[float], 
                                      volume_data: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = self._calculate_rsi(price_data)
            
            # MACD
            macd_line, signal_line, histogram = self._calculate_macd(price_data)
            indicators['macd_line'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = histogram
            
            # Bollinger Bands
            upper, middle, lower = self._calculate_bollinger_bands(price_data)
            indicators['bb_upper'] = upper
            indicators['bb_middle'] = middle
            indicators['bb_lower'] = lower
            indicators['bb_position'] = (price_data[-1] - lower) / (upper - lower) if upper != lower else 0.5
            
            # Moving averages
            indicators['sma_20'] = sum(price_data[-20:]) / 20 if len(price_data) >= 20 else price_data[-1]
            indicators['sma_50'] = sum(price_data[-50:]) / 50 if len(price_data) >= 50 else price_data[-1]
            
            # Momentum
            indicators['momentum'] = (price_data[-1] - price_data[-5]) / price_data[-5] if len(price_data) >= 5 else 0
            
        except Exception as e:
            self.logger.error(f"[PATTERN] Error calculating technical indicators: {e}")
        
        return indicators
    
    def _calculate_rsi(self, price_data: List[float]) -> float:
        """Calculate RSI"""
        if len(price_data) < self.rsi_period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(price_data)):
            change = price_data[i] - price_data[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) >= self.rsi_period:
            avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
            avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        return 50.0
    
    def _calculate_macd(self, price_data: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD"""
        if len(price_data) < self.macd_slow:
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(price_data, self.macd_fast)
        ema_slow = self._calculate_ema(price_data, self.macd_slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_values = []
        for i in range(self.macd_slow, len(price_data)):
            fast_ema = self._calculate_ema(price_data[:i+1], self.macd_fast)
            slow_ema = self._calculate_ema(price_data[:i+1], self.macd_slow)
            macd_values.append(fast_ema - slow_ema)
        
        if len(macd_values) >= self.macd_signal:
            signal_line = self._calculate_ema(macd_values, self.macd_signal)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return data[-1] if data else 0.0
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        
        for i in range(period, len(data)):
            ema = (data[i] * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_bands(self, price_data: List[float]) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(price_data) < self.bollinger_period:
            current_price = price_data[-1]
            return current_price * 1.02, current_price, current_price * 0.98
        
        recent_prices = price_data[-self.bollinger_period:]
        middle = sum(recent_prices) / len(recent_prices)
        
        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in recent_prices) / len(recent_prices)
        std_dev = math.sqrt(variance)
        
        upper = middle + (self.bollinger_std * std_dev)
        lower = middle - (self.bollinger_std * std_dev)
        
        return upper, middle, lower
    
    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate slope of line through points"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _score_patterns(self, patterns: List[Dict], indicators: Dict[str, float]) -> List[Dict]:
        """Score patterns based on technical indicators and historical performance"""
        scored_patterns = []
        
        for pattern in patterns:
            base_confidence = pattern['confidence']
            
            # Adjust confidence based on technical indicators
            adjusted_confidence = self._adjust_confidence_with_indicators(
                pattern, indicators, base_confidence
            )
            
            # Adjust confidence based on historical performance
            historical_confidence = self._adjust_confidence_with_history(
                pattern['pattern'], adjusted_confidence
            )
            
            pattern['confidence'] = historical_confidence
            scored_patterns.append(pattern)
        
        return scored_patterns
    
    def _adjust_confidence_with_indicators(self, pattern: Dict, indicators: Dict[str, float], 
                                         base_confidence: float) -> float:
        """Adjust pattern confidence based on technical indicators"""
        adjusted_confidence = base_confidence
        
        # RSI adjustments
        rsi = indicators.get('rsi', 50)
        if pattern['direction'] == 'bullish':
            if rsi < 30:  # Oversold
                adjusted_confidence += 0.1
            elif rsi > 70:  # Overbought
                adjusted_confidence -= 0.1
        elif pattern['direction'] == 'bearish':
            if rsi > 70:  # Overbought
                adjusted_confidence += 0.1
            elif rsi < 30:  # Oversold
                adjusted_confidence -= 0.1
        
        # MACD adjustments
        macd_line = indicators.get('macd_line', 0)
        macd_signal = indicators.get('macd_signal', 0)
        
        if pattern['direction'] == 'bullish' and macd_line > macd_signal:
            adjusted_confidence += 0.05
        elif pattern['direction'] == 'bearish' and macd_line < macd_signal:
            adjusted_confidence += 0.05
        
        # Bollinger Bands adjustments
        bb_position = indicators.get('bb_position', 0.5)
        if pattern['direction'] == 'bullish' and bb_position < 0.2:
            adjusted_confidence += 0.05
        elif pattern['direction'] == 'bearish' and bb_position > 0.8:
            adjusted_confidence += 0.05
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _adjust_confidence_with_history(self, pattern_name: str, base_confidence: float) -> float:
        """Adjust confidence based on historical pattern performance"""
        if pattern_name in self.pattern_performance:
            perf = self.pattern_performance[pattern_name]
            if perf['total_count'] >= 5:
                # Adjust based on success rate
                success_rate = perf['success_rate']
                if success_rate > 0.7:
                    return min(1.0, base_confidence + 0.1)
                elif success_rate < 0.3:
                    return max(0.0, base_confidence - 0.1)
        
        return base_confidence
    
    def _generate_trading_signals(self, patterns: List[Dict], indicators: Dict[str, float]) -> List[Dict]:
        """Generate trading signals from patterns and indicators"""
        signals = []
        
        # Pattern-based signals
        for pattern in patterns:
            signal = {
                'type': 'pattern',
                'pattern': pattern['pattern'],
                'direction': pattern['direction'],
                'confidence': pattern['confidence'],
                'strength': pattern['confidence'] * 100,
                'description': pattern['description']
            }
            signals.append(signal)
        
        # Technical indicator signals
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            signals.append({
                'type': 'indicator',
                'indicator': 'rsi',
                'direction': 'bullish',
                'confidence': 0.8,
                'strength': 80,
                'description': 'RSI oversold - potential reversal'
            })
        elif rsi > 70:
            signals.append({
                'type': 'indicator',
                'indicator': 'rsi',
                'direction': 'bearish',
                'confidence': 0.8,
                'strength': 80,
                'description': 'RSI overbought - potential reversal'
            })
        
        # MACD signals
        macd_line = indicators.get('macd_line', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd_line > macd_signal and macd_line > 0:
            signals.append({
                'type': 'indicator',
                'indicator': 'macd',
                'direction': 'bullish',
                'confidence': 0.7,
                'strength': 70,
                'description': 'MACD bullish crossover'
            })
        elif macd_line < macd_signal and macd_line < 0:
            signals.append({
                'type': 'indicator',
                'indicator': 'macd',
                'direction': 'bearish',
                'confidence': 0.7,
                'strength': 70,
                'description': 'MACD bearish crossover'
            })
        
        return signals
    
    def _calculate_signal_strength(self, signals: List[Dict]) -> float:
        """Calculate overall signal strength"""
        if not signals:
            return 0.0
        
        # Weight signals by confidence and direction
        bullish_strength = 0.0
        bearish_strength = 0.0
        
        for signal in signals:
            strength = signal['confidence'] * signal['strength']
            if signal['direction'] == 'bullish':
                bullish_strength += strength
            elif signal['direction'] == 'bearish':
                bearish_strength += strength
        
        # Return net strength (positive = bullish, negative = bearish)
        return bullish_strength - bearish_strength
    
    def _update_pattern_history(self, symbol: str, patterns: List[Dict]):
        """Update pattern history for performance tracking"""
        timestamp = datetime.now()
        
        for pattern in patterns:
            pattern_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'pattern': pattern['pattern'],
                'direction': pattern['direction'],
                'confidence': pattern['confidence'],
                'outcome': None  # Will be updated when pattern completes
            }
            
            self.pattern_history.append(pattern_record)
            self.detected_patterns[symbol] = pattern_record
    
    def record_pattern_outcome(self, symbol: str, pattern_name: str, was_profitable: bool, profit: float):
        """Record the outcome of a pattern for performance tracking"""
        # Find the pattern in history
        for record in self.pattern_history:
            if (record['symbol'] == symbol and 
                record['pattern'] == pattern_name and 
                record['outcome'] is None):
                
                record['outcome'] = {
                    'profitable': was_profitable,
                    'profit': profit,
                    'completion_time': datetime.now()
                }
                
                # Update performance statistics
                perf = self.pattern_performance[pattern_name]
                perf['total_count'] += 1
                if was_profitable:
                    perf['success_count'] += 1
                
                perf['success_rate'] = perf['success_count'] / perf['total_count']
                perf['avg_profit'] = (perf['avg_profit'] * (perf['total_count'] - 1) + profit) / perf['total_count']
                
                break
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of pattern analysis performance"""
        total_patterns = len(self.pattern_history)
        completed_patterns = len([p for p in self.pattern_history if p['outcome'] is not None])
        
        # Calculate overall success rate
        successful_patterns = len([p for p in self.pattern_history 
                                 if p['outcome'] and p['outcome']['profitable']])
        
        overall_success_rate = successful_patterns / completed_patterns if completed_patterns > 0 else 0.0
        
        # Get top performing patterns
        top_patterns = sorted(
            self.pattern_performance.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )[:5]
        
        return {
            'total_patterns_detected': total_patterns,
            'completed_patterns': completed_patterns,
            'overall_success_rate': overall_success_rate,
            'top_performing_patterns': top_patterns,
            'pattern_performance': dict(self.pattern_performance)
        }
    
    def get_recommended_patterns(self, symbol: str) -> List[Dict]:
        """Get recommended patterns for a symbol based on historical performance"""
        recommendations = []
        
        # Get patterns with high success rates
        high_success_patterns = [
            (name, perf) for name, perf in self.pattern_performance.items()
            if perf['total_count'] >= 3 and perf['success_rate'] >= 0.6
        ]
        
        for pattern_name, performance in high_success_patterns:
            recommendations.append({
                'pattern': pattern_name,
                'success_rate': performance['success_rate'],
                'avg_profit': performance['avg_profit'],
                'total_occurrences': performance['total_count'],
                'recommendation_strength': performance['success_rate'] * 100
            })
        
        # Sort by recommendation strength
        recommendations.sort(key=lambda x: x['recommendation_strength'], reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations 