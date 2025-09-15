import logging
from typing import List, Optional
import math

# Import configuration
try:
    from src.core.config import config
except ImportError:
    class FallbackConfig:
        CONFIDENCE_THRESHOLD = 0.95
        VOLUME_THRESHOLD = 0.8
        PROFIT_TARGET_PCT = 0.035
        STOP_LOSS_PCT = 0.025
    config = FallbackConfig()

class AdvancedPatternAnalyzer:
    """Advanced pattern recognition for XRP trading signals"""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.pattern_history = []
        self.confidence_threshold = getattr(config, 'CONFIDENCE_THRESHOLD', 0.95)

    def analyze_xrp_patterns(self, price_history: List[float], volume_history: Optional[List[float]] = None):
        if len(price_history) < 5:
            return {"signal": "HOLD", "confidence": 0.0, "patterns": []}
        patterns = []
        confidence = 0.0
        price_change = 0.0
        if len(price_history) >= 2:
            price_change = (price_history[-1] - price_history[-2]) / price_history[-2]
        rsi = self._calculate_rsi(price_history)
        if rsi < 30:
            patterns.append({"type": "RSI_OVERSOLD", "confidence": 0.8})
            confidence += 0.5
        elif rsi > 70:
            patterns.append({"type": "RSI_OVERBOUGHT", "confidence": 0.8})
            confidence -= 0.5
        sma_10 = sum(price_history[-10:]) / 10 if len(price_history) >= 10 else sum(price_history) / len(price_history)
        sma_20 = sum(price_history[-20:]) / 20 if len(price_history) >= 20 else sum(price_history) / len(price_history)
        current_price = price_history[-1]
        if current_price > sma_10 > sma_20:
            patterns.append({"type": "STRONG_UPTREND", "confidence": 0.9})
            confidence += 0.5
        elif current_price < sma_10 < sma_20:
            patterns.append({"type": "STRONG_DOWNTREND", "confidence": 0.9})
            confidence -= 0.5
        momentum = self._calculate_momentum(price_history)
        if momentum > 0.005:
            patterns.append({"type": "POSITIVE_MOMENTUM", "confidence": 0.8})
            confidence += 0.5
        elif momentum < -0.005:
            patterns.append({"type": "NEGATIVE_MOMENTUM", "confidence": 0.8})
            confidence -= 0.5
        volatility = self._calculate_volatility(price_history)
        if volatility > 0.005:
            patterns.append({"type": "HIGH_VOLATILITY", "confidence": 0.7})
            confidence *= 1.2
        volume_confirmed = self.analyze_volume_confirmation(price_change, volume_history)
        if volume_confirmed:
            patterns.append({"type": "VOLUME_CONFIRMED", "confidence": 0.8})
            confidence *= 1.3
        else:
            patterns.append({"type": "WEAK_VOLUME", "confidence": 0.3})
            confidence *= 0.8
        if confidence > 0.3:
            signal = "BUY"
        elif confidence < -0.3:
            signal = "SELL"
        else:
            signal = "HOLD"
        return {
            "signal": signal,
            "confidence": abs(confidence),
            "patterns": patterns,
            "rsi": rsi,
            "momentum": momentum,
            "volatility": volatility,
            "volume_confirmed": volume_confirmed
        }

    def analyze_patterns(self, prices):
        """Detect simple bullish/bearish reversal patterns in price data."""
        signals = []
        if len(prices) < 3:
            return signals
        # Simple local min/max pattern
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                signals.append({'type': 'bullish_reversal', 'index': i, 'price': prices[i]})
            elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                signals.append({'type': 'bearish_reversal', 'index': i, 'price': prices[i]})
        return signals

    def analyze_volume_confirmation(self, price_change, volume_history):
        if not volume_history or len(volume_history) < 5:
            return True
        recent_volume = volume_history[-5:]
        avg_volume = sum(recent_volume) / len(recent_volume)
        current_volume = recent_volume[-1]
        volume_threshold = getattr(config, 'VOLUME_THRESHOLD', 0.8)
        if abs(price_change) > 0.001:
            if current_volume > avg_volume * (1 + volume_threshold):
                return True
            elif current_volume > avg_volume:
                return True
            else:
                return False
        return True

    def get_volume_data(self, symbol="XRP"):
        try:
            self.logger.warning("⚠️ get_volume_data called from AdvancedPatternAnalyzer - use main bot class instead")
            return {"volume_24h": 0, "avg_volume": 0}
        except Exception as e:
            self.logger.error(f"❌ Error getting volume data: {e}")
            return {"volume_24h": 0, "avg_volume": 0}

    def _calculate_rsi(self, prices, period=14):
        try:
            if len(prices) < period + 1:
                return 50
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"❌ Error calculating RSI: {e}")
            return 50

    def _calculate_momentum(self, prices, period=5):
        try:
            if len(prices) < period + 1:
                return 0.0
            start_price = prices[-period-1]
            end_price = prices[-1]
            if start_price <= 0:
                return 0.0
            momentum = (end_price - start_price) / start_price
            return momentum
        except Exception as e:
            self.logger.error(f"❌ Error calculating momentum: {e}")
            return 0.0

    def _calculate_volatility(self, prices, period=20):
        if len(prices) < period:
            return 0.0
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] != 0]
        if not returns:
            return 0.0
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    def calculate_volatility(self, prices):
        """Calculate historical volatility as the standard deviation of log returns."""
        if len(prices) < 2:
            return 0.0
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append(math.log(prices[i] / prices[i-1]))
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    def detect_market_regime(self, price_history, volume_history=None):
        if len(price_history) < 10:
            return "UNKNOWN"
        trend_strength = self._calculate_trend_strength(price_history)
        volatility = self._calculate_volatility(price_history)
        price_range = (max(price_history[-10:]) - min(price_history[-10:])) / min(price_history[-10:])
        if trend_strength > 0.3 and volatility > 0.005:
            return "TRENDING"
        elif volatility < 0.008 and price_range < 0.02:
            return "RANGING"
        elif volatility > 0.015:
            return "VOLATILE"
        else:
            return "MIXED"

    def _calculate_trend_strength(self, price_history):
        if len(price_history) < 10:
            return 0.0
        recent_prices = price_history[-10:]
        x_values = list(range(len(recent_prices)))
        n = len(recent_prices)
        sum_x = sum(x_values)
        sum_y = sum(recent_prices)
        sum_xy = sum(x * y for x, y in zip(x_values, recent_prices))
        sum_x2 = sum(x * x for x in x_values)
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        avg_price = sum_y / n
        trend_strength = abs(slope / avg_price) * 100
        return min(trend_strength, 1.0)

    def adjust_parameters_for_regime(self, market_regime):
        regime_params = {
            "TRENDING": {
                "profit_target": getattr(config, 'PROFIT_TARGET_PCT', 0.035),
                "stop_loss": getattr(config, 'STOP_LOSS_PCT', 0.025),
                "confidence_boost": 1.2,
                "position_size_boost": 1.3
            },
            "RANGING": {
                "profit_target": getattr(config, 'PROFIT_TARGET_PCT', 0.035),
                "stop_loss": getattr(config, 'STOP_LOSS_PCT', 0.025),
                "confidence_boost": 0.9,
                "position_size_boost": 0.8
            },
            "VOLATILE": {
                "profit_target": getattr(config, 'PROFIT_TARGET_PCT', 0.035) + 0.02,
                "stop_loss": getattr(config, 'STOP_LOSS_PCT', 0.025) + 0.01,
                "confidence_boost": 0.8,
                "position_size_boost": 0.7
            },
            "MIXED": {
                "profit_target": getattr(config, 'PROFIT_TARGET_PCT', 0.035),
                "stop_loss": getattr(config, 'STOP_LOSS_PCT', 0.025),
                "confidence_boost": 1.0,
                "position_size_boost": 1.0
            }
        }
        return regime_params.get(market_regime, regime_params["MIXED"]) 