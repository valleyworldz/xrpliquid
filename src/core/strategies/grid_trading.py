#!/usr/bin/env python3
"""
🏗️ ENHANCED GRID TRADING STRATEGY
===============================

Advanced grid trading strategy with:
- Dynamic grid adjustment based on volatility
- Risk-managed position sizing
- Automatic grid rebalancing
- Profit target and stop-loss integration
- Market condition adaptation
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from core.strategies.base_strategy import TradingStrategy
from core.utils.logger import Logger

class GridTradingStrategy(TradingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.logger = Logger()
        
        # Grid parameters with safe defaults
        self.grid_levels = self._get_config("strategies.grid_trading.grid_levels", 5)
        self.grid_spacing_pct = self._get_config("strategies.grid_trading.grid_spacing", 0.01)  # 1%
        self.base_order_size = self._get_config("strategies.grid_trading.order_quantity", 0.005)
        self.max_grid_positions = self._get_config("strategies.grid_trading.max_grid_positions", 3)
        self.profit_per_grid = self._get_config("strategies.grid_trading.profit_per_grid", 0.015)  # 1.5%
        self.rebalance_frequency = self._get_config("strategies.grid_trading.rebalance_frequency", 600)  # 10 minutes
        self.volatility_adjustment = self._get_config("strategies.grid_trading.volatility_adjustment", True)
        self.stop_loss_pct = self._get_config("strategies.grid_trading.stop_loss", 0.05)  # 5%
        
        # Grid state tracking
        self.active_grids = {}  # {price_level: {"side": "buy/sell", "order_id": str, "quantity": float}}
        self.grid_center_price = 0.0
        self.grid_upper_bound = 0.0
        self.grid_lower_bound = 0.0
        self.last_rebalance_time = None
        self.total_grid_investment = 0.0
        
        # Performance tracking
        self.grid_profits = []
        self.completed_grids = 0
        self.grid_efficiency = 0.0
        
        self.logger.info(f"[GRID_TRADING] Strategy initialized with {self.grid_levels} levels, {self.grid_spacing_pct:.2%} spacing")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Safely get config value with fallback"""
        try:
            if self.config and hasattr(self.config, 'get'):
                return self.config.get(key, default)
            return default
        except Exception:
            return default
    
    def calculate_dynamic_grid_spacing(self, price_history: List[float]) -> float:
        """Calculate dynamic grid spacing based on recent volatility"""
        try:
            if not price_history or len(price_history) < 10:
                return self.grid_spacing_pct
            
            # Calculate recent volatility
            recent_prices = np.array(price_history[-20:], dtype=float)  # Last 20 data points
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns)
            
            # Adjust grid spacing based on volatility
            if self.volatility_adjustment:
                # Higher volatility = wider grid spacing
                volatility_multiplier = max(min(volatility * 10, 2.0), 0.5)  # 0.5x to 2.0x multiplier
                dynamic_spacing = self.grid_spacing_pct * volatility_multiplier
            else:
                dynamic_spacing = self.grid_spacing_pct
            
            return max(min(dynamic_spacing, 0.05), 0.001)  # Cap between 0.1% and 5%
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error calculating dynamic spacing: {e}")
            return self.grid_spacing_pct
    
    def calculate_grid_bounds(self, current_price: float, price_history: List[float]) -> Tuple[float, float]:
        """Calculate upper and lower bounds for the grid"""
        try:
            dynamic_spacing = self.calculate_dynamic_grid_spacing(price_history)
            
            # Calculate total grid range
            total_range = dynamic_spacing * self.grid_levels
            
            # Set bounds around current price
            upper_bound = current_price * (1 + total_range / 2)
            lower_bound = current_price * (1 - total_range / 2)
            
            return upper_bound, lower_bound
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error calculating grid bounds: {e}")
            return current_price * 1.05, current_price * 0.95  # Default 5% range
    
    def generate_grid_levels(self, current_price: float, price_history: List[float]) -> List[Dict[str, Any]]:
        """Generate grid levels with buy and sell orders"""
        try:
            upper_bound, lower_bound = self.calculate_grid_bounds(current_price, price_history)
            dynamic_spacing = self.calculate_dynamic_grid_spacing(price_history)
            
            grid_levels = []
            
            # Calculate grid price levels
            price_step = current_price * dynamic_spacing
            
            # Generate buy orders below current price
            for i in range(1, self.grid_levels // 2 + 1):
                buy_price = current_price - (price_step * i)
                if buy_price >= lower_bound:
                    grid_levels.append({
                        "price": buy_price,
                    "side": "buy",
                        "quantity": self.base_order_size,
                        "level": -i,  # Negative for buy levels
                        "target_price": buy_price * (1 + self.profit_per_grid)
                    })
            
            # Generate sell orders above current price
            for i in range(1, self.grid_levels // 2 + 1):
                sell_price = current_price + (price_step * i)
                if sell_price <= upper_bound:
                    grid_levels.append({
                        "price": sell_price,
                        "side": "sell",
                        "quantity": self.base_order_size,
                        "level": i,  # Positive for sell levels
                        "target_price": sell_price * (1 - self.profit_per_grid)
                    })
            
            return grid_levels
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error generating grid levels: {e}")
            return []
    
    def check_grid_rebalance_needed(self, current_price: float) -> bool:
        """Check if grid needs rebalancing"""
        try:
            # Time-based rebalancing
            if (self.last_rebalance_time and 
                (datetime.now() - self.last_rebalance_time).total_seconds() < self.rebalance_frequency):
                return False
            
            # Price-based rebalancing (if price moves outside grid bounds)
            if self.grid_upper_bound > 0 and self.grid_lower_bound > 0:
                if current_price > self.grid_upper_bound or current_price < self.grid_lower_bound:
                    return True
            
            # No active grids
            if not self.active_grids:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error checking rebalance: {e}")
            return True
    
    def calculate_grid_performance(self) -> Dict[str, float]:
        """Calculate grid trading performance metrics"""
        try:
            total_profit = sum(self.grid_profits)
            avg_profit_per_grid = np.mean(self.grid_profits) if self.grid_profits else 0.0
            
            # Calculate efficiency (profits vs investment)
            efficiency = total_profit / self.total_grid_investment if self.total_grid_investment > 0 else 0.0
            
            return {
                "total_profit": total_profit,
                "avg_profit_per_grid": avg_profit_per_grid,
                "completed_grids": self.completed_grids,
                "grid_efficiency": efficiency,
                "active_grid_count": len(self.active_grids)
            }
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error calculating performance: {e}")
            return {}
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate grid trading signals"""
        try:
            # Validate input data
            if not self.validate_data(market_data):
                return {}
            
            current_price = float(market_data["price"])
            symbol = market_data.get("symbol", "UNKNOWN")
            price_history = market_data.get("price_history", [current_price])
            
            # Check if rebalancing is needed
            if self.check_grid_rebalance_needed(current_price):
                return self.rebalance_grid(current_price, price_history, symbol)
            
            # Check for grid order executions and generate new orders
            return self.check_grid_executions(current_price, symbol)
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error generating signal: {e}")
            return {}
    
    def rebalance_grid(self, current_price: float, price_history: List[float], symbol: str) -> Dict[str, Any]:
        """Rebalance the entire grid"""
        try:
            self.logger.info(f"[GRID_TRADING] Rebalancing grid for {symbol} at price {current_price}")
            
            # Clear old grid state
            self.active_grids.clear()
            
            # Update grid bounds
            self.grid_upper_bound, self.grid_lower_bound = self.calculate_grid_bounds(current_price, price_history)
            self.grid_center_price = current_price
            self.last_rebalance_time = datetime.now()
            
            # Generate new grid levels
            grid_levels = self.generate_grid_levels(current_price, price_history)
            
            if not grid_levels:
                return {}
            
            # Create rebalance signal with multiple orders
            signal = {
                "action": "rebalance_grid",
                "confidence": 0.8,
                "grid_levels": grid_levels,
                "current_price": current_price,
                "grid_bounds": {
                    "upper": self.grid_upper_bound,
                    "lower": self.grid_lower_bound
                },
                "strategy": "grid_trading",
                "timestamp": datetime.now()
            }
            
            # Track total investment
            self.total_grid_investment = sum(level["price"] * level["quantity"] for level in grid_levels if level["side"] == "buy")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error rebalancing grid: {e}")
            return {}
    
    def check_grid_executions(self, current_price: float, symbol: str) -> Dict[str, Any]:
        """Check for grid order executions and generate profit-taking orders"""
        try:
            # This would normally check with the exchange for filled orders
            # For now, simulate based on price movements
            
            executed_levels = []
            
            for price_level, grid_info in self.active_grids.items():
                # Check if grid level was hit
                if (grid_info["side"] == "buy" and current_price <= price_level) or \
                   (grid_info["side"] == "sell" and current_price >= price_level):
                    executed_levels.append((price_level, grid_info))
            
            if executed_levels:
                # Generate profit-taking signal for the executed level
                level_price, level_info = executed_levels[0]  # Take first execution
                
                signal = {
                    "action": "grid_profit_take",
                    "confidence": 0.9,
                    "executed_level": {
                        "price": level_price,
                        "side": level_info["side"],
                        "quantity": level_info["quantity"],
                        "target_price": level_info.get("target_price", current_price)
                    },
                    "current_price": current_price,
                    "strategy": "grid_trading",
                    "timestamp": datetime.now()
                }
                
                # Update performance tracking
                profit = abs(current_price - level_price) * level_info["quantity"]
                self.grid_profits.append(profit)
                self.completed_grids += 1
                
                # Remove executed level from active grids
                del self.active_grids[level_price]
                
                return signal
            
            return {}
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error checking executions: {e}")
            return {}
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get current strategy statistics"""
        base_stats = super().get_strategy_info()
        grid_stats = {
            "grid_levels": self.grid_levels,
            "grid_spacing_pct": self.grid_spacing_pct,
            "active_grid_count": len(self.active_grids),
            "grid_center_price": self.grid_center_price,
            "grid_bounds": {
                "upper": self.grid_upper_bound,
                "lower": self.grid_lower_bound
            },
            "total_grid_investment": self.total_grid_investment,
            "completed_grids": self.completed_grids,
            "grid_efficiency": self.calculate_grid_performance().get("grid_efficiency", 0.0),
            "last_rebalance_time": self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
        }
        return {**base_stats, **grid_stats}
    
    def run(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Main strategy execution method"""
        try:
            # Validate inputs
            if not self.validate_data(data) or not self.validate_params(params):
                return {}
            
            if not self.is_enabled():
                return {}
            
            # Generate signal
            signal = self.generate_signal(data)
            
            if signal and signal.get("action"):
                self.last_signal_time = datetime.now()
                self.logger.info(f"[GRID_TRADING] Generated signal for {data.get('symbol', 'UNKNOWN')}: {signal['action']}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"[GRID_TRADING] Error in strategy run: {e}")
            return {}
    
    def __str__(self):
        return f"Enhanced Grid Trading Strategy ({self.grid_levels} levels, {self.grid_spacing_pct:.2%} spacing)"


