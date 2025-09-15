#!/usr/bin/env python3
"""
💰 PROFIT MAXIMIZATION ENGINE
============================

Master-level profit optimization system that automatically:
- Optimizes position sizing for maximum Kelly criterion efficiency
- Dynamically adjusts strategies based on market conditions
- Implements advanced risk-adjusted position management
- Maximizes profit rotation efficiency
- Applies machine learning for pattern recognition
- Optimizes timing for entries and exits
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

from core.utils.logger import Logger

@dataclass
class ProfitOpportunity:
    """Represents a profit opportunity"""
    symbol: str
    strategy: str
    confidence: float
    expected_profit: float
    risk_score: float
    optimal_size: float
    entry_price: float
    target_price: float
    stop_loss: float
    time_horizon: int

class ProfitMaximizer:
    """Advanced profit maximization engine"""
    
    def __init__(self, config, api_client):
        self.config = config
        self.api = api_client
        self.logger = Logger()
        
        # Profit optimization parameters
        self.profit_target_base = 0.012  # 1.2% base target (increased)
        self.max_profit_target = 0.035   # 3.5% maximum target (increased)
        self.dynamic_sizing_factor = 1.8  # Amplify successful strategies more
        self.risk_reward_min_ratio = 2.5  # Minimum 2.5:1 risk/reward (improved)
        
        # Performance tracking
        self.strategy_performance = {}
        self.profit_history = []
        self.optimization_cycles = 0
        
        # Advanced features
        self.use_compound_sizing = True
        self.use_momentum_scaling = True
        self.use_volatility_targeting = True
        
        self.logger.info("🚀 [PROFIT_MAX] Advanced profit maximization engine initialized")
    
    def calculate_optimal_profit_targets(self, symbol: str, market_data: Dict[str, Any], 
                                       strategy_confidence: float) -> Tuple[float, float]:
        """Calculate optimal profit target and stop loss for maximum profitability"""
        try:
            current_price = float(market_data.get("price", 0))
            if current_price <= 0:
                return 0.0, 0.0
            
            # Calculate volatility-adjusted targets
            price_history = market_data.get("price_history", [current_price])
            if len(price_history) > 10:
                prices = np.array(price_history[-20:])
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
            else:
                volatility = 0.025  # Default 2.5% (increased for more aggressive targets)
            
            # Base profit target adjusted for volatility and confidence
            volatility_multiplier = min(max(float(volatility * 60), 0.6), 3.5)  # Scale 0.6x to 3.5x
            confidence_multiplier = 0.6 + (strategy_confidence * 1.8)    # Scale 0.6x to 2.4x
            
            profit_target_pct = self.profit_target_base * volatility_multiplier * confidence_multiplier
            profit_target_pct = min(profit_target_pct, self.max_profit_target)
            
            # Calculate stop loss for optimal risk/reward ratio
            stop_loss_pct = profit_target_pct / self.risk_reward_min_ratio
            
            # Convert to absolute prices
            profit_target_price = current_price * (1 + profit_target_pct)
            stop_loss_price = current_price * (1 - stop_loss_pct)
            
            self.logger.info(f"💎 [PROFIT_MAX] {symbol} targets: Profit={profit_target_pct:.3f}%, "
                           f"Stop={stop_loss_pct:.3f}%, R/R={self.risk_reward_min_ratio:.1f}")
            
            return profit_target_price, stop_loss_price
            
        except Exception as e:
            self.logger.error(f"❌ [PROFIT_MAX] Error calculating targets for {symbol}: {e}")
            return 0.0, 0.0
    
    def calculate_compound_position_size(self, base_size: float, recent_performance: Dict[str, Any]) -> float:
        """Calculate position size with compound growth factor"""
        try:
            if not self.use_compound_sizing:
                return base_size
            
            # Get recent win rate and average profit
            win_rate = recent_performance.get("win_rate", 0.5)
            avg_profit = recent_performance.get("avg_profit", 0.005)
            trade_count = recent_performance.get("trade_count", 0)
            
            # Compound factor based on recent success
            if trade_count >= 3:  # Reduced minimum for faster adaptation
                if win_rate > 0.65 and avg_profit > 0.012:  # Strong performance
                    compound_factor = 1.0 + min((win_rate - 0.5) * 2.5, 0.75)  # Up to 75% boost
                elif win_rate < 0.35 or avg_profit < 0.003:  # Poor performance
                    compound_factor = 0.6  # Reduce size by 40%
                else:
                    compound_factor = 1.0
            else:
                compound_factor = 0.9  # Slightly conservative for new strategies
            
            optimized_size = base_size * compound_factor
            
            self.logger.info(f"📈 [PROFIT_MAX] Compound sizing: {base_size:.6f} -> {optimized_size:.6f} "
                           f"(factor: {compound_factor:.2f})")
            
            return optimized_size
            
        except Exception as e:
            self.logger.error(f"❌ [PROFIT_MAX] Error in compound sizing: {e}")
            return base_size
    
    def apply_momentum_scaling(self, position_size: float, market_data: Dict[str, Any]) -> float:
        """Apply momentum-based position scaling"""
        try:
            if not self.use_momentum_scaling:
                return position_size
            
            price_history = market_data.get("price_history", [])
            volume_history = market_data.get("volume_history", [])
            
            if len(price_history) < 5 or len(volume_history) < 5:
                return position_size
            
            # Calculate price momentum
            prices = np.array(price_history[-8:])
            short_ma = float(np.mean(prices[-2:]))
            long_ma = float(np.mean(prices[-5:]))
            price_momentum = abs(short_ma - long_ma) / long_ma
            
            # Calculate volume momentum
            volumes = np.array(volume_history[-8:])
            recent_vol = np.mean(volumes[-2:])
            avg_vol = np.mean(volumes[-5:])
            volume_momentum = recent_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Combined momentum score
            momentum_score = (price_momentum * 120 + volume_momentum) / 2
            momentum_multiplier = min(max(momentum_score, 0.7), 1.5)  # Scale 0.7x to 1.5x
            
            scaled_size = position_size * momentum_multiplier
            
            self.logger.info(f"⚡ [PROFIT_MAX] Momentum scaling: {momentum_multiplier:.2f}x")
            return scaled_size
            
        except Exception as e:
            self.logger.error(f"❌ [PROFIT_MAX] Error in momentum scaling: {e}")
            return position_size
    
    def optimize_profit_rotation_timing(self, open_positions: Dict[str, Any]) -> Dict[str, str]:
        """Optimize profit rotation timing for maximum efficiency"""
        try:
            recommendations = {}
            
            for symbol, position in open_positions.items():
                unrealized_pnl_pct = position.get("unrealized_pnl_pct", 0.0)
                entry_time = position.get("entry_time", datetime.now())
                holding_time = (datetime.now() - entry_time).total_seconds()
                
                # Get market data if available
                try:
                    market_data = self.api.get_market_data(symbol)
                    if market_data and "price_history" in market_data:
                        price_history = market_data["price_history"]
                        if len(price_history) > 5:
                            prices = np.array(price_history[-8:])
                            volatility = np.std(np.diff(prices) / prices[:-1])
                        else:
                            volatility = 0.025
                    else:
                        volatility = 0.025
                except:
                    volatility = 0.025
                
                # Dynamic profit target based on volatility and time held
                base_target = 0.012  # 1.2% increased
                volatility_adjustment = volatility * 30  # Scale volatility
                time_decay = max(0.4, 1 - (holding_time / 2400))  # Decay over 40 minutes
                
                dynamic_target = base_target + volatility_adjustment * time_decay
                
                # Recommendation logic
                if unrealized_pnl_pct >= dynamic_target:
                    recommendations[symbol] = "CLOSE_NOW"
                elif unrealized_pnl_pct >= dynamic_target * 0.75:
                    recommendations[symbol] = "MONITOR_CLOSELY"
                elif unrealized_pnl_pct <= -dynamic_target * 0.4:
                    recommendations[symbol] = "CONSIDER_STOP"
                else:
                    recommendations[symbol] = "HOLD"
                
                self.logger.info(f"🎯 [PROFIT_MAX] {symbol}: PnL={unrealized_pnl_pct:.3f}%, "
                              f"Target={dynamic_target:.3f}%, Action={recommendations[symbol]}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ [PROFIT_MAX] Error optimizing rotation timing: {e}")
            return {}
    
    def calculate_kelly_optimal_size(self, win_rate: float, avg_win: float, avg_loss: float, 
                                   available_capital: float) -> float:
        """Calculate optimal position size using Kelly criterion"""
        try:
            if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            kelly_fraction = min(max(kelly_fraction, 0.005), 0.15)  # Cap between 0.5% and 15%
            
            optimal_size = available_capital * kelly_fraction
            
            self.logger.info(f"🧮 [PROFIT_MAX] Kelly optimal: {kelly_fraction:.3f} of capital = ${optimal_size:.2f}")
            
            return optimal_size
            
        except Exception as e:
            self.logger.error(f"❌ [PROFIT_MAX] Error in Kelly calculation: {e}")
            return 0.0
    
    def get_advanced_profit_metrics(self) -> Dict[str, Any]:
        """Get advanced profit metrics and performance analysis"""
        try:
            if not self.profit_history:
                return {"status": "No trades recorded yet"}
            
            profits = np.array(self.profit_history)
            
            # Basic metrics
            total_profit = np.sum(profits)
            avg_profit = np.mean(profits)
            win_rate = len(profits[profits > 0]) / len(profits)
            
            # Advanced metrics
            sharpe_ratio = avg_profit / np.std(profits) if np.std(profits) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(profits)
            profit_factor = abs(np.sum(profits[profits > 0]) / np.sum(profits[profits < 0])) if np.sum(profits[profits < 0]) != 0 else float('inf')
            
            # Recent performance (last 10 trades)
            recent_profits = profits[-10:] if len(profits) >= 10 else profits
            recent_win_rate = len(recent_profits[recent_profits > 0]) / len(recent_profits) if len(recent_profits) > 0 else 0
            
            return {
                "total_profit": total_profit,
                "average_profit": avg_profit,
                "win_rate": win_rate,
                "recent_win_rate": recent_win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor,
                "total_trades": len(profits),
                "optimization_cycles": self.optimization_cycles,
                "strategy_performance": self.strategy_performance
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PROFIT_MAX] Error generating metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_max_drawdown(self, profits: np.ndarray) -> float:
        """Calculate maximum drawdown from profit series"""
        try:
            cumulative = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = np.min(drawdown)
            return max_drawdown
        except:
            return 0.0
    
    def record_trade_result(self, symbol: str, strategy: str, profit_pct: float):
        """Record trade result for performance tracking"""
        try:
            self.profit_history.append(profit_pct)
            
            # Update strategy performance
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    "trades": [],
                    "win_rate": 0.0,
                    "avg_profit": 0.0
                }
            
            self.strategy_performance[strategy]["trades"].append(profit_pct)
            trades = self.strategy_performance[strategy]["trades"]
            
            # Calculate updated metrics
            wins = [t for t in trades if t > 0]
            self.strategy_performance[strategy]["win_rate"] = len(wins) / len(trades)
            self.strategy_performance[strategy]["avg_profit"] = np.mean(trades)
            
            self.logger.info(f"📊 [PROFIT_MAX] Recorded {strategy} trade on {symbol}: {profit_pct:.3f}%")
            
        except Exception as e:
            self.logger.error(f"❌ [PROFIT_MAX] Error recording trade: {e}")
    
    def suggest_next_optimization(self) -> Dict[str, Any]:
        """Suggest next optimization based on performance analysis"""
        try:
            if len(self.profit_history) < 5:
                return {"suggestion": "Continue trading to gather performance data"}
            
            metrics = self.get_advanced_profit_metrics()
            suggestions = []
            
            # Analyze performance and suggest improvements
            if metrics["win_rate"] < 0.5:
                suggestions.append("Consider tightening entry criteria - win rate below 50%")
            
            if metrics["sharpe_ratio"] < 1.0:
                suggestions.append("Improve risk-adjusted returns - consider position sizing optimization")
            
            if metrics["profit_factor"] < 1.5:
                suggestions.append("Enhance profit factor by improving exit timing")
            
            if abs(metrics["max_drawdown"]) > 0.1:  # 10%
                suggestions.append("Reduce maximum drawdown with better risk management")
            
            return {
                "suggestions": suggestions,
                "current_metrics": metrics,
                "optimization_priority": "high" if len(suggestions) > 2 else "medium"
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PROFIT_MAX] Error suggesting optimization: {e}")
            return {"error": str(e)}
