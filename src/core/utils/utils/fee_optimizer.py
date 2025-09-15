#!/usr/bin/env python3
"""
Fee Optimization Module
Optimizes trading to minimize fees and maximize profits
"""

import time
import logging
from typing import Dict, List, Optional

class FeeOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.daily_fees = 0.0
        self.daily_trades = 0
        self.batch_orders = []
        self.logger = logging.getLogger(__name__)
        
    def should_place_order(self, order_value: float, expected_profit: float) -> bool:
        """Determine if order should be placed based on fee optimization"""
        
        # Check if we've exceeded daily fee budget
        if self.daily_fees >= self.config.get('fee_budget', 10):
            self.logger.warning("Daily fee budget exceeded, skipping order")
            return False
        
        # Check minimum profit after fees
        min_profit = self.config.get('min_profit_after_fees', 0.1)
        if expected_profit < min_profit:
            self.logger.info(f"Expected profit ${expected_profit:.4f} below minimum ${min_profit}, skipping")
            return False
        
        # Check if we should batch this order
        if len(self.batch_orders) < self.config.get('batch_order_size', 3):
            self.batch_orders.append({
                'value': order_value,
                'expected_profit': expected_profit,
                'timestamp': time.time()
            })
            return False  # Don't place immediately, wait for batch
        
        return True
    
    def get_optimal_order_type(self, is_buy: bool, market_data: Dict) -> str:
        """Determine optimal order type (maker vs taker)"""
        
        spread = market_data.get('spread_percent', 0)
        volume = market_data.get('volume_24h', 0)
        
        # Prefer maker orders for better fees
        if self.config.get('prefer_maker_orders', True):
            return 'maker'
        
        # Use taker only for urgent orders or tight spreads
        if spread < 0.01 or volume < 1000000:
            return 'taker'
        
        return 'maker'
    
    def calculate_position_size(self, available_balance: float, risk_per_trade: float) -> float:
        """Calculate optimal position size considering fees"""
        
        min_size = self.config.get('min_position_size', 500)
        max_size = self.config.get('max_position_size', 2000)
        
        # Kelly criterion for position sizing
        win_rate = 0.6  # Assume 60% win rate
        avg_win = 0.02  # 2% average win
        avg_loss = 0.01  # 1% average loss
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0.1, min(0.5, kelly_fraction))  # Cap between 10-50%
        
        position_size = available_balance * kelly_fraction * risk_per_trade
        
        # Ensure minimum size
        position_size = max(min_size, position_size)
        
        # Cap at maximum size
        position_size = min(max_size, position_size)
        
        return position_size
    
    def update_fee_tracking(self, fee: float):
        """Update fee tracking"""
        self.daily_fees += fee
        self.daily_trades += 1
        
        self.logger.info(f"Daily fees: ${self.daily_fees:.4f}, trades: {self.daily_trades}")
    
    def reset_daily_tracking(self):
        """Reset daily fee and trade tracking"""
        self.daily_fees = 0.0
        self.daily_trades = 0
        self.batch_orders = []
        self.logger.info("Reset daily fee tracking")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            'daily_fees': self.daily_fees,
            'daily_trades': self.daily_trades,
            'avg_fee_per_trade': self.daily_fees / max(1, self.daily_trades),
            'fee_budget_remaining': max(0, self.config.get('fee_budget', 10) - self.daily_fees)
        }
