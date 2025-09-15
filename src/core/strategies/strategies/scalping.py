#!/usr/bin/env python3
"""
🚀 ENHANCED SCALPING STRATEGY
===========================

High-frequency scalping strategy with advanced features:
- Adaptive spread calculation based on volatility
- Dynamic position sizing with Kelly criterion
- Momentum and volume filters
- Multi-timeframe analysis
- Risk-adjusted profit targets
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.strategies.base_strategy import TradingStrategy
from core.engines.risk_management import RiskManagement
from core.utils.logger import Logger

class Scalping(TradingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.risk_management = RiskManagement(config)
        self.logger = Logger()
        self.name = "scalping"
        
        # Enhanced scalping parameters with safe defaults
        self.min_spread = self._get_config("strategies.scalping.params.min_spread", 0.0005)
        self.max_spread = self._get_config("strategies.scalping.params.max_spread", 0.003)
        self.volatility_lookback = self._get_config("strategies.scalping.params.volatility_lookback", 20)
        self.momentum_threshold = self._get_config("strategies.scalping.params.momentum_threshold", 0.6)
        self.volume_threshold = self._get_config("strategies.scalping.params.volume_threshold", 1.2)
        self.profit_multiplier = self._get_config("strategies.scalping.params.profit_multiplier", 2.5)
        self.max_holding_time = self._get_config("strategies.scalping.params.max_holding_time", 300)  # 5 minutes
        
        # Performance tracking
        self.signal_history = []
        self.trade_count = 0
        self.win_rate = 0.0
        self.avg_profit = 0.0
        
        self.logger.info(f"[SCALPING] Strategy initialized with spread range: {self.min_spread:.4f} - {self.max_spread:.4f}")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Safely get config value with fallback"""
        try:
            if self.config and hasattr(self.config, 'get'):
                return self.config.get(key, default)
            return default
        except Exception:
            return default
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI using pure numpy with enhanced error handling"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            # Ensure we have valid prices
            prices = np.array(prices, dtype=float)
            if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
                return 50.0
            
            deltas = np.diff(prices)
            seed = deltas[:period]
            
            gains = seed[seed >= 0]
            losses = -seed[seed < 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            
            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
            
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            # Continue calculation for remaining periods with smoothing
            for delta in deltas[period:]:
                if delta > 0:
                    avg_gain = (avg_gain * (period - 1) + delta) / period
                    avg_loss = (avg_loss * (period - 1)) / period
                else:
                    avg_gain = (avg_gain * (period - 1)) / period
                    avg_loss = (avg_loss * (period - 1) + abs(delta)) / period
                
                if avg_loss == 0:
                    rsi = 100.0 if avg_gain > 0 else 50.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return max(0.0, min(100.0, rsi))
            
        except Exception as e:
            self.logger.error(f"[SCALPING] Error calculating RSI: {e}")
            return 50.0
    
    def calculate_adaptive_spread(self, price_data: List[float]) -> float:
        """Calculate adaptive spread based on recent price volatility"""
        try:
            if not price_data or len(price_data) < 2:
                return self.min_spread
            
            # Take the most recent data points
            recent_prices = np.array(price_data[-min(self.volatility_lookback, len(price_data)):], dtype=float)
            
            if len(recent_prices) < 2:
                return self.min_spread
            
            # Calculate returns and volatility
            returns = np.diff(recent_prices) / recent_prices[:-1]
            
            # Remove any invalid returns
            valid_returns = returns[np.isfinite(returns)]
            if len(valid_returns) == 0:
                return self.min_spread
            
            volatility = np.std(valid_returns)
            
            # Scale spread based on volatility (more volatile = wider spread)
            volatility_multiplier = min(max(volatility * 100, 1.0), 5.0)
            adaptive_spread = self.min_spread * volatility_multiplier
            
            return min(max(adaptive_spread, self.min_spread), self.max_spread)
            
        except Exception as e:
            self.logger.error(f"[SCALPING] Error calculating adaptive spread: {e}")
            return self.min_spread
    
    def calculate_momentum_score(self, price_data: List[float], volume_data: List[float]) -> float:
        """Calculate momentum score using price and volume with enhanced validation"""
        try:
            if not price_data or not volume_data or len(price_data) < 5 or len(volume_data) < 5:
                return 0.5
            
            # Ensure we have enough data
            min_length = min(len(price_data), len(volume_data), 14)
            prices = np.array(price_data[-min_length:], dtype=float)
            volumes = np.array(volume_data[-min_length:], dtype=float)
            
            # Validate data
            if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
                return 0.5
            if np.any(np.isnan(volumes)) or np.any(np.isinf(volumes)) or np.any(volumes <= 0):
                return 0.5
            
            # RSI momentum using custom implementation
            rsi = self.calculate_rsi(prices, min(14, len(prices) - 1))
            rsi_score = 1.0 - abs(rsi - 50) / 50  # Higher score for RSI near 50 (neutral)
            
            # Volume momentum
            if len(volumes) > 1:
                avg_volume = np.mean(volumes[:-1])
                current_volume = volumes[-1]
                if avg_volume > 0:
                    volume_score = min(current_volume / avg_volume / self.volume_threshold, 1.0)
                else:
                    volume_score = 0.5
            else:
                volume_score = 0.5
            
            # Price momentum (short-term)
            if len(prices) >= 7:
                short_ma = np.mean(prices[-3:])
                long_ma = np.mean(prices[-7:])
                if long_ma > 0:
                    price_momentum = abs(short_ma - long_ma) / long_ma
                    momentum_score = min(price_momentum * 100, 1.0)
                else:
                    momentum_score = 0.0
            else:
                momentum_score = 0.0
            
            # Combined momentum score
            combined_score = (rsi_score * 0.4 + volume_score * 0.3 + momentum_score * 0.3)
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            self.logger.error(f"[SCALPING] Error calculating momentum: {e}")
            return 0.5
    
    def calculate_dynamic_position_size(self, confidence: float, account_balance: float, current_price: float) -> float:
        """Calculate position size using modified Kelly criterion with validation"""
        try:
            # Validate inputs
            if confidence <= 0 or account_balance <= 0 or current_price <= 0:
                return 0.0
            
            # Kelly fraction calculation with safety bounds
            win_rate = max(min(self.win_rate, 0.9), 0.5)  # Bound between 50% and 90%
            avg_win = max(self.avg_profit, 0.005)  # Minimum 0.5% profit assumption
            avg_loss = 0.003  # Conservative 0.3% loss assumption
            
            if win_rate > 0 and avg_win > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(min(kelly_fraction, 0.05), 0.005)  # Cap between 0.5% and 5%
            else:
                kelly_fraction = 0.01  # Default 1%
            
            # Adjust by confidence
            adjusted_fraction = kelly_fraction * max(min(confidence, 1.0), 0.1)
            
            # Calculate position size with safety check
            position_value = account_balance * adjusted_fraction
            position_size = position_value / current_price
            
            return max(position_size, 0.0)
            
        except Exception as e:
            self.logger.error(f"[SCALPING] Error calculating position size: {e}")
            return account_balance * 0.01 / current_price  # Fallback 1%
    
    def analyze_market_microstructure(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Analyze order book for scalping opportunities"""
        try:
            if not order_book or "levels" not in order_book:
                return {"spread": 0.0, "depth": 0.0, "imbalance": 0.0}
            
            levels = order_book["levels"]
            if len(levels) < 2:
                return {"spread": 0.0, "depth": 0.0, "imbalance": 0.0}
            
            # Calculate bid-ask spread
            best_bid = float(levels[0][0]["px"])
            best_ask = float(levels[1][0]["px"])
            spread = (best_ask - best_bid) / best_bid
            
            # Calculate market depth
            bid_depth = sum(float(level["sz"]) for level in levels[0][:5])  # Top 5 bid levels
            ask_depth = sum(float(level["sz"]) for level in levels[1][:5])  # Top 5 ask levels
            total_depth = bid_depth + ask_depth
            
            # Calculate order book imbalance
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
            
            return {
                "spread": spread,
                "depth": total_depth,
                "imbalance": imbalance
            }
            
        except Exception as e:
            self.logger.error(f"[SCALPING] Error analyzing microstructure: {e}")
            return {"spread": 0.0, "depth": 0.0, "imbalance": 0.0}
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced scalping signal"""
        try:
            if not market_data or "price" not in market_data:
                return {}
            
            current_price = float(market_data["price"])
            price_history = market_data.get("price_history", [current_price])
            volume_history = market_data.get("volume_history", [1.0])
            order_book = market_data.get("order_book", {})
            
            # Calculate adaptive spread
            adaptive_spread = self.calculate_adaptive_spread(price_history)
            
            # Calculate momentum score
            momentum_score = self.calculate_momentum_score(price_history, volume_history)
            
            # Analyze market microstructure
            microstructure = self.analyze_market_microstructure(order_book)
            
            # Generate signal only if conditions are favorable
            signal_strength = 0.0
            
            # Check spread conditions
            if microstructure["spread"] < adaptive_spread and microstructure["spread"] > self.min_spread:
                signal_strength += 0.3
            
            # Check momentum conditions
            if momentum_score > self.momentum_threshold:
                signal_strength += 0.4
            
            # Check market depth
            if microstructure["depth"] > 10.0:  # Sufficient liquidity
                signal_strength += 0.2
            
            # Check order book imbalance for direction
            direction = "buy" if microstructure["imbalance"] > 0.1 else "sell" if microstructure["imbalance"] < -0.1 else "neutral"
            
            # Only generate signal if strength is sufficient
            if signal_strength >= 0.7 and direction != "neutral":
                confidence = min(signal_strength * momentum_score, 1.0)
                
                # Calculate dynamic targets
                profit_target = adaptive_spread * self.profit_multiplier
                stop_loss = adaptive_spread * 0.8  # Tighter stop loss
                
                signal = {
                    "action": direction,
                    "confidence": confidence,
                    "signal_strength": signal_strength,
                    "entry_price": current_price,
                    "profit_target": profit_target,
                    "stop_loss": stop_loss,
                    "max_holding_time": self.max_holding_time,
                    "adaptive_spread": adaptive_spread,
                    "momentum_score": momentum_score,
                    "microstructure": microstructure,
                    "timestamp": datetime.now()
                }
                
                self.signal_history.append(signal)
                self.logger.info(f"[SCALPING] Generated {direction} signal: confidence={confidence:.3f}, spread={adaptive_spread:.4f}")
                
                return signal
            
            return {}
            
        except Exception as e:
            self.logger.error(f"[SCALPING] Error generating signal: {e}")
            return {}
    
    def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        try:
            if not trade_result:
                return
            
            self.trade_count += 1
            pnl = trade_result.get("pnl", 0.0)
            
            if pnl > 0:
                self.win_rate = (self.win_rate * (self.trade_count - 1) + 1) / self.trade_count
                self.avg_profit = (self.avg_profit * (self.trade_count - 1) + abs(pnl)) / self.trade_count
            else:
                self.win_rate = (self.win_rate * (self.trade_count - 1)) / self.trade_count
            
            self.logger.info(f"[SCALPING] Updated metrics: Win rate={self.win_rate:.3f}, Avg profit={self.avg_profit:.4f}")
            
        except Exception as e:
            self.logger.error(f"[SCALPING] Error updating metrics: {e}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get current strategy statistics"""
        return {
            "name": self.name,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "avg_profit": self.avg_profit,
            "signal_count": len(self.signal_history),
            "min_spread": self.min_spread,
            "max_spread": self.max_spread,
            "momentum_threshold": self.momentum_threshold
        }
    
    def run(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Main strategy execution method with enhanced validation"""
        try:
            # Validate inputs
            if not self.validate_data(data) or not self.validate_params(params):
                return {}
            
            if not self.is_enabled():
                return {}
            
            # Generate signal
            signal = self.generate_signal(data)
            
            if signal:
                # Update last signal time
                self.last_signal_time = datetime.now()
                signal["strategy"] = "scalping"
                self.logger.info(f"[SCALPING] Generated signal for {data.get('symbol', 'UNKNOWN')}: {signal['action']} @ {signal['confidence']:.3f} confidence")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[SCALPING] Error in strategy run: {e}")
            return {}

    def __str__(self):
        return f"Enhanced Scalping Strategy (spread: {self.min_spread:.4f}-{self.max_spread:.4f})" 
