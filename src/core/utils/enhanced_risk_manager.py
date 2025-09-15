#!/usr/bin/env python3
"""
Enhanced Risk Management Module
Implements advanced risk management strategies
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class EnhancedRiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.trade_history = []
        self.logger = logging.getLogger(__name__)
        
    def can_open_position(self, coin: str, position_size: float, current_positions: List) -> bool:
        """Check if we can open a new position"""
        
        # Check daily loss limit
        if self.daily_pnl <= -self.config.get('max_daily_loss', 50):
            self.logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        # Check concurrent positions limit
        max_positions = self.config.get('max_concurrent_positions', 2)
        if len(current_positions) >= max_positions:
            self.logger.info(f"Maximum concurrent positions reached: {len(current_positions)}")
            return False
        
        # Check position size limits
        min_size = self.config.get('min_position_size', 500)
        max_size = self.config.get('max_position_size', 2000)
        
        if position_size < min_size:
            self.logger.warning(f"Position size ${position_size:.2f} below minimum ${min_size}")
            return False
        
        if position_size > max_size:
            self.logger.warning(f"Position size ${position_size:.2f} above maximum ${max_size}")
            return False
        
        # Check correlation with existing positions
        if self.config.get('correlation_filter', True):
            if self._has_high_correlation(coin, current_positions):
                self.logger.info(f"High correlation detected for {coin}, skipping")
                return False
        
        return True
    
    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """Calculate stop loss price"""
        
        stop_loss_percent = self.config.get('stop_loss_percent', 0.5) / 100
        
        if direction.lower() == 'long':
            return entry_price * (1 - stop_loss_percent)
        else:
            return entry_price * (1 + stop_loss_percent)
    
    def calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """Calculate take profit price"""
        
        profit_target = self.config.get('profit_target_percent', 1.0) / 100
        risk_reward = self.config.get('risk_reward_ratio', 2.0)
        
        # Use risk/reward ratio to calculate profit target
        stop_loss_percent = self.config.get('stop_loss_percent', 0.5) / 100
        profit_target = stop_loss_percent * risk_reward
        
        if direction.lower() == 'long':
            return entry_price * (1 + profit_target)
        else:
            return entry_price * (1 - profit_target)
    
    def should_close_position(self, position: Dict, current_price: float) -> bool:
        """Determine if position should be closed"""
        
        entry_price = position.get('entry_price', 0)
        direction = position.get('direction', 'long')
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        
        # Check stop loss
        if direction.lower() == 'long' and current_price <= stop_loss:
            self.logger.info(f"Stop loss triggered at ${current_price:.4f}")
            return True
        
        if direction.lower() == 'short' and current_price >= stop_loss:
            self.logger.info(f"Stop loss triggered at ${current_price:.4f}")
            return True
        
        # Check take profit
        if direction.lower() == 'long' and current_price >= take_profit:
            self.logger.info(f"Take profit triggered at ${current_price:.4f}")
            return True
        
        if direction.lower() == 'short' and current_price <= take_profit:
            self.logger.info(f"Take profit triggered at ${current_price:.4f}")
            return True
        
        return False
    
    def update_performance(self, pnl: float, trade_data: Dict):
        """Update performance tracking"""
        
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.current_balance += pnl
        
        # Update peak balance and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Store trade data
        trade_data['timestamp'] = time.time()
        trade_data['daily_pnl'] = self.daily_pnl
        trade_data['daily_trades'] = self.daily_trades
        self.trade_history.append(trade_data)
        
        self.logger.info(f"Daily PnL: ${self.daily_pnl:.4f}, Trades: {self.daily_trades}, Max DD: {self.max_drawdown:.2f}%")
    
    def reset_daily_tracking(self):
        """Reset daily performance tracking"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.logger.info("Reset daily performance tracking")
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'max_drawdown': self.max_drawdown,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'win_rate': self._calculate_win_rate()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
        return (winning_trades / len(self.trade_history)) * 100
    
    def _has_high_correlation(self, coin: str, positions: List) -> bool:
        """Check if coin has high correlation with existing positions"""
        # Simplified correlation check - in practice, use actual correlation data
        high_correlation_pairs = [
            ['BTC', 'ETH'],
            ['SOL', 'AVAX'],
            ['BNB', 'CAKE']
        ]
        
        for position in positions:
            position_coin = position.get('coin', '')
            for pair in high_correlation_pairs:
                if coin in pair and position_coin in pair:
                    return True
        
        return False
