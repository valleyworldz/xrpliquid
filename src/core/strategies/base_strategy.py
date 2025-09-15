from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

class TradingStrategy(ABC):
    """
    🏗️ BASE TRADING STRATEGY
    ======================
    
    Abstract base class for all trading strategies.
    Defines the interface and common functionality.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.name = self.__class__.__name__.lower()
        self.enabled = True
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "sharpe_ratio": 0.0
        }
        self.last_signal_time = None
        self.active_positions = {}
    
    @abstractmethod
    def run(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the trading strategy.
        
        Args:
            data: Market data including price, volume, etc.
            params: Strategy-specific parameters
            
        Returns:
            Trading signal dictionary or empty dict if no signal
        """
        pass
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data has required fields"""
        required_fields = ["symbol", "price"]
        return all(field in data for field in required_fields)
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate strategy parameters"""
        if not isinstance(params, dict):
            return False
        return True
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        if not trade_result:
            return
            
        self.performance_metrics["total_trades"] += 1
        pnl = trade_result.get("pnl", 0.0)
        self.performance_metrics["total_pnl"] += pnl
        
        if pnl > 0:
            self.performance_metrics["winning_trades"] += 1
        
        # Update win rate
        total_trades = self.performance_metrics["total_trades"]
        winning_trades = self.performance_metrics["winning_trades"]
        self.performance_metrics["win_rate"] = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Update average trade PnL
        self.performance_metrics["avg_trade_pnl"] = self.performance_metrics["total_pnl"] / total_trades if total_trades > 0 else 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def reset_performance(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "sharpe_ratio": 0.0
        }
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self.enabled
    
    def enable(self):
        """Enable the strategy"""
        self.enabled = True
    
    def disable(self):
        """Disable the strategy"""
        self.enabled = False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "last_signal_time": self.last_signal_time,
            "active_positions": len(self.active_positions),
            "performance": self.get_performance_metrics()
        }
    
    def __str__(self):
        return f"{self.name.title()} Strategy"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(enabled={self.enabled})"

# Alias for backward compatibility
BaseStrategy = TradingStrategy
