#!/usr/bin/env python3
"""
🎯 ENHANCED STRATEGY MANAGER
===========================

Advanced strategy management with:
- Dynamic strategy selection based on market conditions
- Performance tracking and comparison
- Strategy health monitoring
- Automatic strategy rotation
- Risk-aware strategy allocation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.engines.market_regime import MarketRegime
from core.utils.logger import Logger
from core.strategies.base_strategy import TradingStrategy

# Import concrete strategy implementations
from core.strategies.scalping import Scalping
from core.strategies.mean_reversion import MeanReversion
from core.strategies.grid_trading import GridTradingStrategy as GridTrading
from core.strategies.rl_ai import RL_AI_Strategy

class StrategyManager:
    def __init__(self, config):
        self.logger = Logger()
        self.config = config
        
        # Initialize strategy_performance first before strategies
        self.strategy_performance = {}
        
        # Initialize strategies with error handling
        self.strategies = {}
        self._initialize_strategies()
        
        self.market_regime_analyzer = MarketRegime()
        self.last_regime_analysis = None
        self.strategy_rotation_enabled = True
        self.min_strategy_runtime = 300  # 5 minutes minimum per strategy
        
        self.logger.info(f"[STRATEGY_MANAGER] Initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self):
        """Initialize all strategies with proper error handling"""
        strategy_classes = {
            "scalping": Scalping,
            "mean_reversion": MeanReversion,
            "grid_trading": GridTrading,
            "rl_ai": RL_AI_Strategy,
        }
        
        for name, strategy_class in strategy_classes.items():
            try:
                strategy = strategy_class(self.config)
                self.strategies[name] = strategy
                
                # Initialize performance tracking
                self.strategy_performance[name] = {
                    "total_signals": 0,
                    "successful_signals": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "last_used": None,
                    "avg_confidence": 0.0,
                    "health_score": 1.0,
                    "enabled": True
                }
                
                self.logger.info(f"[STRATEGY_MANAGER] Successfully initialized {name} strategy")
                
            except Exception as e:
                self.logger.error(f"[STRATEGY_MANAGER] Failed to initialize {name} strategy: {e}")
    
    def get_strategy(self, name: str) -> Optional[TradingStrategy]:
        """Returns an instance of a registered strategy by name"""
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, TradingStrategy]:
        """Get all available strategies"""
        return self.strategies.copy()
    
    def get_enabled_strategies(self) -> Dict[str, TradingStrategy]:
        """Get only enabled strategies"""
        return {name: strategy for name, strategy in self.strategies.items() 
                if strategy.is_enabled() and self.strategy_performance[name]["enabled"]}
    
    def update_strategy_performance(self, strategy_name: str, signal_result: Dict[str, Any]):
        """Update performance metrics for a strategy"""
        try:
            if strategy_name not in self.strategy_performance:
                return
            
            perf = self.strategy_performance[strategy_name]
            perf["total_signals"] += 1
            perf["last_used"] = datetime.now()
            
            # Update confidence tracking
            confidence = signal_result.get("confidence", 0.0)
            if perf["total_signals"] == 1:
                perf["avg_confidence"] = confidence
            else:
                perf["avg_confidence"] = (perf["avg_confidence"] * (perf["total_signals"] - 1) + confidence) / perf["total_signals"]
            
            # Update PnL if available
            pnl = signal_result.get("pnl", 0.0)
            if pnl != 0.0:
                perf["total_pnl"] += pnl
                if pnl > 0:
                    perf["successful_signals"] += 1
                
                # Update win rate
                perf["win_rate"] = perf["successful_signals"] / perf["total_signals"]
            
            # Calculate health score
            self._calculate_health_score(strategy_name)
            
        except Exception as e:
            self.logger.error(f"[STRATEGY_MANAGER] Error updating performance for {strategy_name}: {e}")
    
    def _calculate_health_score(self, strategy_name: str):
        """Calculate health score for a strategy"""
        try:
            perf = self.strategy_performance[strategy_name]
            
            # Base score components
            win_rate_score = perf["win_rate"]
            confidence_score = perf["avg_confidence"]
            usage_score = min(perf["total_signals"] / 100.0, 1.0)  # More usage = more reliable
            
            # PnL score (normalized)
            pnl_score = 0.5  # Neutral
            if perf["total_pnl"] > 0:
                pnl_score = min(0.5 + (perf["total_pnl"] / 1000.0), 1.0)  # Cap at 1.0
            elif perf["total_pnl"] < 0:
                pnl_score = max(0.5 + (perf["total_pnl"] / 1000.0), 0.0)  # Floor at 0.0
            
            # Weighted health score
            health_score = (
                win_rate_score * 0.3 +
                confidence_score * 0.2 +
                usage_score * 0.2 +
                pnl_score * 0.3
            )
            
            perf["health_score"] = max(0.0, min(1.0, health_score))
            
        except Exception as e:
            self.logger.error(f"[STRATEGY_MANAGER] Error calculating health score for {strategy_name}: {e}")
    
    def select_adaptive_strategy(self, market_data: Dict[str, Any]) -> Optional[TradingStrategy]:
        """Select strategy based on market regime and strategy performance"""
        try:
            # Analyze current market regime
            regime = self.market_regime_analyzer.analyze_regime(market_data)
            self.last_regime_analysis = {
                "regime": regime,
                "timestamp": datetime.now(),
                "market_data": market_data
            }
            
            self.logger.info(f"[STRATEGY_MANAGER] Detected market regime: {regime}")
            
            # Get regime-appropriate strategies
            regime_strategies = self._get_strategies_for_regime(regime)
            
            if not regime_strategies:
                self.logger.warning(f"[STRATEGY_MANAGER] No strategies available for regime: {regime}")
                return None
            
            # Rank strategies by health score
            strategy_rankings = []
            for strategy_name in regime_strategies:
                if strategy_name in self.strategy_performance:
                    health_score = self.strategy_performance[strategy_name]["health_score"]
                    strategy_rankings.append((strategy_name, health_score))
            
            # Sort by health score (descending)
            strategy_rankings.sort(key=lambda x: x[1], reverse=True)
            
            # Select best strategy
            if strategy_rankings:
                best_strategy_name = strategy_rankings[0][0]
                best_strategy = self.strategies.get(best_strategy_name)
                
                if best_strategy and best_strategy.is_enabled():
                    self.logger.info(f"[STRATEGY_MANAGER] Selected {best_strategy_name} strategy (health: {strategy_rankings[0][1]:.2f})")
                    return best_strategy
            
            # Fallback to first available strategy
            for strategy_name in regime_strategies:
                strategy = self.strategies.get(strategy_name)
                if strategy and strategy.is_enabled():
                    self.logger.info(f"[STRATEGY_MANAGER] Fallback to {strategy_name} strategy")
                    return strategy
            
            return None
            
        except Exception as e:
            self.logger.error(f"[STRATEGY_MANAGER] Error in adaptive strategy selection: {e}")
            return None
    
    def _get_strategies_for_regime(self, regime: str) -> List[str]:
        """Get appropriate strategies for market regime"""
        regime_mapping = {
            "trending": ["mean_reversion", "rl_ai"],  # Mean reversion works well in trending markets
            "ranging": ["grid_trading", "scalping"],  # Grid and scalping for sideways markets
            "volatile": ["scalping", "rl_ai"],       # Quick strategies for volatile markets
            "calm": ["grid_trading", "mean_reversion", "rl_ai"],  # Any strategy works in calm markets
            "bullish": ["scalping", "grid_trading"],  # Momentum strategies
            "bearish": ["mean_reversion", "rl_ai"],   # Counter-trend strategies
        }
        
        return regime_mapping.get(regime, ["scalping", "mean_reversion"])  # Default strategies
    
    def run_selected_strategy(self, strategy_name: str, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific strategy by name"""
        try:
            strategy = self.get_strategy(strategy_name)
            if not strategy:
                self.logger.error(f"[STRATEGY_MANAGER] Strategy '{strategy_name}' not found")
                return {}
            
            if not strategy.is_enabled():
                self.logger.warning(f"[STRATEGY_MANAGER] Strategy '{strategy_name}' is disabled")
                return {}
            
            self.logger.info(f"[STRATEGY_MANAGER] Running strategy: {strategy_name}")
            
            # Run the strategy
            signal = strategy.run(data, params)
            
            # Update performance tracking
            if signal:
                self.update_strategy_performance(strategy_name, signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[STRATEGY_MANAGER] Error running strategy '{strategy_name}': {e}")
            return {}
    
    def run_best_strategy(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the best strategy based on current market conditions"""
        try:
            best_strategy = self.select_adaptive_strategy(data)
            if not best_strategy:
                return {}
            
            strategy_name = best_strategy.name
            return self.run_selected_strategy(strategy_name, data, params)
            
        except Exception as e:
            self.logger.error(f"[STRATEGY_MANAGER] Error running best strategy: {e}")
            return {}
    
    def enable_strategy(self, strategy_name: str):
        """Enable a specific strategy"""
        try:
            if strategy_name in self.strategies:
                self.strategies[strategy_name].enable()
                self.strategy_performance[strategy_name]["enabled"] = True
                self.logger.info(f"[STRATEGY_MANAGER] Enabled strategy: {strategy_name}")
            else:
                self.logger.error(f"[STRATEGY_MANAGER] Strategy '{strategy_name}' not found")
        except Exception as e:
            self.logger.error(f"[STRATEGY_MANAGER] Error enabling strategy '{strategy_name}': {e}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a specific strategy"""
        try:
            if strategy_name in self.strategies:
                self.strategies[strategy_name].disable()
                self.strategy_performance[strategy_name]["enabled"] = False
                self.logger.info(f"[STRATEGY_MANAGER] Disabled strategy: {strategy_name}")
            else:
                self.logger.error(f"[STRATEGY_MANAGER] Strategy '{strategy_name}' not found")
        except Exception as e:
            self.logger.error(f"[STRATEGY_MANAGER] Error disabling strategy '{strategy_name}': {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""
        return {
            "strategies": self.strategy_performance.copy(),
            "total_strategies": len(self.strategies),
            "enabled_strategies": len(self.get_enabled_strategies()),
            "last_regime": self.last_regime_analysis,
            "strategy_rotation_enabled": self.strategy_rotation_enabled
        }
    
    def reset_performance(self, strategy_name: Optional[str] = None):
        """Reset performance metrics for a strategy or all strategies"""
        try:
            if strategy_name:
                if strategy_name in self.strategy_performance:
                    self.strategy_performance[strategy_name] = {
                        "total_signals": 0,
                        "successful_signals": 0,
                        "total_pnl": 0.0,
                        "win_rate": 0.0,
                        "last_used": None,
                        "avg_confidence": 0.0,
                        "health_score": 1.0,
                        "enabled": True
                    }
                    if strategy_name in self.strategies:
                        self.strategies[strategy_name].reset_performance()
                    self.logger.info(f"[STRATEGY_MANAGER] Reset performance for {strategy_name}")
            else:
                for name in self.strategy_performance:
                    self.reset_performance(name)
                self.logger.info("[STRATEGY_MANAGER] Reset performance for all strategies")
                
        except Exception as e:
            self.logger.error(f"[STRATEGY_MANAGER] Error resetting performance: {e}")
    
    def __str__(self):
        enabled_count = len(self.get_enabled_strategies())
        return f"Strategy Manager ({enabled_count}/{len(self.strategies)} enabled)"

# This file marks the strategies directory as a Python package.


