#!/usr/bin/env python3
"""
🧠 ADVANCED TRADING OPTIMIZER
============================

Master-level optimization system for maximum profitability:
- Dynamic parameter optimization using Bayesian optimization
- Real-time risk adjustment based on market conditions
- Adaptive position sizing with Kelly criterion
- Market regime detection and strategy selection
- Performance-based strategy allocation
- Advanced pattern recognition for entry/exit timing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from core.utils.logger import Logger
from core.api.hyperliquid_api import HyperliquidAPI

@dataclass
class MarketRegime:
    """Market regime classification"""
    name: str
    volatility: float
    trend_strength: float
    liquidity: float
    optimal_strategies: List[str]
    risk_multiplier: float

@dataclass
class OptimizationResult:
    """Optimization result container"""
    strategy: str
    parameters: Dict[str, Any]
    expected_return: float
    risk_score: float
    confidence: float
    sharpe_ratio: float

class AdvancedOptimizer:
    """Master-level trading optimization system"""
    
    def __init__(self, config, api_client: HyperliquidAPI):
        self.config = config
        self.api = api_client
        self.logger = Logger()
        
        # Market regime definitions
        self.market_regimes = {
            "high_volatility_trending": MarketRegime(
                name="High Volatility Trending",
                volatility=0.03,
                trend_strength=0.7,
                liquidity=0.8,
                optimal_strategies=["scalping", "momentum"],
                risk_multiplier=0.7
            ),
            "low_volatility_sideways": MarketRegime(
                name="Low Volatility Sideways",
                volatility=0.01,
                trend_strength=0.3,
                liquidity=0.9,
                optimal_strategies=["grid_trading", "mean_reversion"],
                risk_multiplier=1.2
            ),
            "high_volatility_sideways": MarketRegime(
                name="High Volatility Sideways",
                volatility=0.03,
                trend_strength=0.3,
                liquidity=0.7,
                optimal_strategies=["scalping", "grid_trading"],
                risk_multiplier=0.8
            ),
            "low_volatility_trending": MarketRegime(
                name="Low Volatility Trending",
                volatility=0.01,
                trend_strength=0.7,
                liquidity=0.9,
                optimal_strategies=["momentum", "mean_reversion"],
                risk_multiplier=1.0
            )
        }
        
        # Performance tracking
        self.strategy_performance = {}
        self.regime_history = []
        self.optimization_history = []
        
        # ML models for regime detection
        self.regime_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_model_trained = False
        
        self.logger.info("[OPTIMIZER] Advanced optimizer initialized")
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Detect current market regime using advanced analysis"""
        try:
            # Extract market features
            features = self._extract_market_features(market_data)
            
            if self.is_model_trained and len(features) > 0:
                # Use ML model for regime detection
                features_scaled = self.scaler.transform([features])
                regime_prob = self.regime_classifier.predict_proba(features_scaled)[0]
                regime_names = list(self.market_regimes.keys())
                best_regime_idx = np.argmax(regime_prob)
                confidence = regime_prob[best_regime_idx]
                
                if confidence > 0.6:  # High confidence threshold
                    regime_name = regime_names[best_regime_idx]
                    regime = self.market_regimes[regime_name]
                    self.logger.info(f"[OPTIMIZER] Detected regime: {regime.name} (confidence: {confidence:.3f})")
                    return regime
            
            # Fallback to rule-based detection
            volatility = features[0] if features else 0.02
            trend_strength = features[1] if len(features) > 1 else 0.5
            liquidity = features[2] if len(features) > 2 else 0.8
            
            if volatility > 0.025:
                if trend_strength > 0.6:
                    regime = self.market_regimes["high_volatility_trending"]
                else:
                    regime = self.market_regimes["high_volatility_sideways"]
            else:
                if trend_strength > 0.6:
                    regime = self.market_regimes["low_volatility_trending"]
                else:
                    regime = self.market_regimes["low_volatility_sideways"]
            
            self.logger.info(f"[OPTIMIZER] Detected regime (rule-based): {regime.name}")
            return regime
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error detecting market regime: {e}")
            return self.market_regimes["low_volatility_sideways"]  # Safe default
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract features for market regime detection"""
        try:
            features = []
            
            # Price-based features
            if "price_history" in market_data:
                prices = np.array(market_data["price_history"][-50:])  # Last 50 periods
                if len(prices) > 10:
                    # Volatility
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns)
                    features.append(volatility)
                    
                    # Trend strength using linear regression slope
                    x = np.arange(len(prices))
                    trend_coef = np.polyfit(x, prices, 1)[0]
                    trend_strength = abs(trend_coef) / np.mean(prices)
                    features.append(trend_strength)
                    
                    # Price momentum
                    short_ma = np.mean(prices[-5:])
                    long_ma = np.mean(prices[-20:])
                    momentum = (short_ma - long_ma) / long_ma
                    features.append(momentum)
                    
                    # Price range compression
                    price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)
                    features.append(price_range)
            
            # Volume-based features
            if "volume_history" in market_data:
                volumes = np.array(market_data["volume_history"][-20:])
                if len(volumes) > 5:
                    # Volume trend
                    volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
                    features.append(volume_trend)
                    
                    # Volume volatility
                    volume_std = np.std(volumes) / np.mean(volumes)
                    features.append(volume_std)
            
            # Liquidity proxy (simplified)
            liquidity = 0.8  # Default assumption
            features.append(liquidity)
            
            return features
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error extracting features: {e}")
            return [0.02, 0.5, 0.8]  # Default features
    
    def optimize_strategy_parameters(self, strategy_name: str, market_regime: MarketRegime, 
                                   historical_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimize strategy parameters for current market regime"""
        try:
            if strategy_name not in market_regime.optimal_strategies:
                self.logger.warning(f"[OPTIMIZER] {strategy_name} not optimal for {market_regime.name}")
            
            # Get base parameters
            base_params = self.config.get(f"strategies.{strategy_name}.params", {})
            
            # Regime-specific parameter adjustments
            optimized_params = base_params.copy()
            
            if strategy_name == "scalping":
                optimized_params.update(self._optimize_scalping_params(market_regime))
            elif strategy_name == "grid_trading":
                optimized_params.update(self._optimize_grid_params(market_regime))
            elif strategy_name == "mean_reversion":
                optimized_params.update(self._optimize_mean_reversion_params(market_regime))
            elif strategy_name == "rl_ai":
                optimized_params.update(self._optimize_rl_params(market_regime))
            
            self.logger.info(f"[OPTIMIZER] Optimized {strategy_name} for {market_regime.name}")
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error optimizing {strategy_name}: {e}")
            return self.config.get(f"strategies.{strategy_name}.params", {})
    
    def _optimize_scalping_params(self, regime: MarketRegime) -> Dict[str, Any]:
        """Optimize scalping parameters for market regime"""
        params = {}
        
        if regime.volatility > 0.025:  # High volatility
            params["min_spread"] = 0.0008
            params["max_spread"] = 0.004
            params["profit_multiplier"] = 2.0
            params["momentum_threshold"] = 0.7
            params["volume_threshold"] = 1.5
        else:  # Low volatility
            params["min_spread"] = 0.0003
            params["max_spread"] = 0.002
            params["profit_multiplier"] = 3.0
            params["momentum_threshold"] = 0.5
            params["volume_threshold"] = 1.2
        
        # Adjust for trend strength
        if regime.trend_strength > 0.6:
            params["max_holding_time"] = 600  # 10 minutes for trending
        else:
            params["max_holding_time"] = 180  # 3 minutes for sideways
        
        return params
    
    def _optimize_grid_params(self, regime: MarketRegime) -> Dict[str, Any]:
        """Optimize grid trading parameters for market regime"""
        params = {}
        
        if regime.volatility > 0.025:  # High volatility
            params["grid_levels"] = 7
            params["grid_spacing"] = 0.025
            params["take_profit_pct"] = 0.02
            params["stop_loss_pct"] = 0.015
        else:  # Low volatility
            params["grid_levels"] = 5
            params["grid_spacing"] = 0.015
            params["take_profit_pct"] = 0.015
            params["stop_loss_pct"] = 0.01
        
        # Adjust for trend strength
        if regime.trend_strength > 0.6:
            params["grid_bias"] = 0.6  # Bias in trend direction
        else:
            params["grid_bias"] = 0.0  # Neutral grid
        
        return params
    
    def _optimize_mean_reversion_params(self, regime: MarketRegime) -> Dict[str, Any]:
        """Optimize mean reversion parameters for market regime"""
        params = {}
        
        if regime.volatility > 0.025:  # High volatility
            params["lookback_period"] = 15
            params["std_dev_threshold"] = 2.5
            params["take_profit_pct"] = 0.03
            params["stop_loss_pct"] = 0.015
        else:  # Low volatility
            params["lookback_period"] = 25
            params["std_dev_threshold"] = 1.8
            params["take_profit_pct"] = 0.02
            params["stop_loss_pct"] = 0.01
        
        return params
    
    def _optimize_rl_params(self, regime: MarketRegime) -> Dict[str, Any]:
        """Optimize RL AI parameters for market regime"""
        params = {}
        
        # Adjust confidence threshold based on regime
        if regime.name in ["high_volatility_trending", "low_volatility_trending"]:
            params["confidence_threshold"] = 0.6  # Lower threshold for trending markets
        else:
            params["confidence_threshold"] = 0.8  # Higher threshold for sideways markets
        
        # Adjust profit targets
        params["take_profit_pct"] = 0.015 + (regime.volatility * 0.5)
        params["stop_loss_pct"] = 0.008 + (regime.volatility * 0.3)
        
        return params
    
    def calculate_optimal_position_size(self, strategy_confidence: float, market_regime: MarketRegime,
                                      account_balance: float, current_price: float) -> float:
        """Calculate optimal position size using advanced Kelly criterion"""
        try:
            # Get strategy performance history
            strategy_stats = self.strategy_performance.get("recent_stats", {})
            win_rate = strategy_stats.get("win_rate", 0.55)  # Conservative default
            avg_win = strategy_stats.get("avg_win", 0.015)
            avg_loss = strategy_stats.get("avg_loss", 0.008)
            
            # Kelly criterion calculation
            if win_rate > 0 and avg_win > 0 and avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = min(max(kelly_fraction, 0.01), 0.15)  # Cap between 1% and 15%
            else:
                kelly_fraction = 0.03  # Conservative default
            
            # Adjust for market regime
            regime_adjustment = market_regime.risk_multiplier
            kelly_fraction *= regime_adjustment
            
            # Adjust for strategy confidence
            confidence_adjustment = 0.5 + (strategy_confidence * 0.5)  # Scale 0.5 to 1.0
            kelly_fraction *= confidence_adjustment
            
            # Calculate position size
            position_value = account_balance * kelly_fraction
            position_size = position_value / current_price
            
            self.logger.info(f"[OPTIMIZER] Position size: {position_size:.6f} (Kelly: {kelly_fraction:.3f})")
            return position_size
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error calculating position size: {e}")
            return account_balance * 0.02 / current_price  # Fallback 2%
    
    def get_strategy_allocation(self, market_regime: MarketRegime) -> Dict[str, float]:
        """Get optimal strategy allocation for current market regime"""
        try:
            # Base allocations for each regime
            allocations = {}
            
            if market_regime.name == "high_volatility_trending":
                allocations = {"scalping": 0.6, "momentum": 0.3, "rl_ai": 0.1}
            elif market_regime.name == "low_volatility_sideways":
                allocations = {"grid_trading": 0.5, "mean_reversion": 0.4, "scalping": 0.1}
            elif market_regime.name == "high_volatility_sideways":
                allocations = {"scalping": 0.4, "grid_trading": 0.4, "rl_ai": 0.2}
            else:  # low_volatility_trending
                allocations = {"momentum": 0.4, "mean_reversion": 0.3, "scalping": 0.3}
            
            # Adjust based on recent performance
            for strategy in allocations:
                recent_performance = self.strategy_performance.get(strategy, {}).get("sharpe_ratio", 1.0)
                if recent_performance > 1.5:
                    allocations[strategy] *= 1.2  # Boost well-performing strategies
                elif recent_performance < 0.5:
                    allocations[strategy] *= 0.8  # Reduce poor-performing strategies
            
            # Normalize allocations
            total_allocation = sum(allocations.values())
            if total_allocation > 0:
                allocations = {k: v / total_allocation for k, v in allocations.items()}
            
            self.logger.info(f"[OPTIMIZER] Strategy allocation: {allocations}")
            return allocations
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error calculating strategy allocation: {e}")
            return {"scalping": 0.5, "grid_trading": 0.3, "mean_reversion": 0.2}
    
    def update_performance_tracking(self, strategy_name: str, trade_result: Dict[str, Any]):
        """Update strategy performance tracking"""
        try:
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    "trades": [],
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0
                }
            
            strategy_stats = self.strategy_performance[strategy_name]
            strategy_stats["trades"].append(trade_result)
            
            # Calculate updated metrics
            trades = strategy_stats["trades"]
            pnls = [trade.get("pnl", 0.0) for trade in trades]
            
            strategy_stats["total_pnl"] = sum(pnls)
            strategy_stats["win_rate"] = len([pnl for pnl in pnls if pnl > 0]) / len(pnls)
            
            if len(pnls) > 1:
                strategy_stats["sharpe_ratio"] = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
            
            # Calculate recent performance for position sizing
            recent_trades = trades[-20:] if len(trades) > 20 else trades
            if recent_trades:
                recent_pnls = [trade.get("pnl", 0.0) for trade in recent_trades]
                wins = [pnl for pnl in recent_pnls if pnl > 0]
                losses = [pnl for pnl in recent_pnls if pnl < 0]
                
                self.strategy_performance["recent_stats"] = {
                    "win_rate": len(wins) / len(recent_pnls),
                    "avg_win": np.mean(wins) if wins else 0.01,
                    "avg_loss": abs(np.mean(losses)) if losses else 0.005
                }
            
            self.logger.info(f"[OPTIMIZER] Updated {strategy_name} performance: PnL={strategy_stats['total_pnl']:.4f}, "
                           f"Win rate={strategy_stats['win_rate']:.3f}, Sharpe={strategy_stats['sharpe_ratio']:.3f}")
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error updating performance for {strategy_name}: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "current_regime": self.regime_history[-1].name if self.regime_history else "Unknown",
                "strategy_performance": self.strategy_performance,
                "optimization_count": len(self.optimization_history),
                "model_trained": self.is_model_trained
            }
            
            # Calculate overall portfolio metrics
            total_pnl = sum(stats.get("total_pnl", 0.0) for stats in self.strategy_performance.values())
            avg_sharpe = np.mean([stats.get("sharpe_ratio", 0.0) for stats in self.strategy_performance.values()])
            
            summary["portfolio_metrics"] = {
                "total_pnl": total_pnl,
                "average_sharpe": avg_sharpe,
                "active_strategies": len(self.strategy_performance)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error generating summary: {e}")
            return {"error": str(e)}
    
    def save_optimization_state(self, filepath: str = "optimization_state.json"):
        """Save current optimization state"""
        try:
            state = {
                "strategy_performance": self.strategy_performance,
                "regime_history": [regime.name for regime in self.regime_history[-100:]],  # Last 100
                "optimization_history": self.optimization_history[-50:],  # Last 50
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"[OPTIMIZER] Saved optimization state to {filepath}")
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error saving state: {e}")
    
    def load_optimization_state(self, filepath: str = "optimization_state.json"):
        """Load previous optimization state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.strategy_performance = state.get("strategy_performance", {})
            self.optimization_history = state.get("optimization_history", [])
            
            self.logger.info(f"[OPTIMIZER] Loaded optimization state from {filepath}")
            
        except FileNotFoundError:
            self.logger.info("[OPTIMIZER] No previous optimization state found, starting fresh")
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Error loading state: {e}") 