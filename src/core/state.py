#!/usr/bin/env python3
"""
Runtime State Management
=======================

This module manages the runtime state of the trading bot, separate from configuration.
This prevents state drift and makes testing trivial.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime, date


@dataclass
class RuntimeState:
    """Runtime state for the trading bot - separate from configuration"""
    
    # Position tracking
    position_size: int = 0
    entry_price: Optional[float] = None
    entry_time: Optional[float] = None
    is_long: bool = True
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_oid: Optional[str] = None
    sl_oid: Optional[str] = None
    
    # Trading state
    tp_sl_active: bool = False
    trailing_active: bool = False
    breakeven_shifted: bool = False
    
    # Performance tracking
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    consecutive_losses: int = 0
    total_trades: int = 0
    
    # Risk management
    daily_trades: int = 0
    last_trade_time: Optional[float] = None
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # Market state
    current_price: Optional[float] = None
    current_funding_rate: Optional[float] = None
    last_price_update: Optional[float] = None
    
    # Active triggers
    active_triggers: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # System state
    is_running: bool = False
    last_heartbeat: Optional[float] = None
    error_count: int = 0
    last_error_time: Optional[float] = None
    
    def reset_daily_counters(self):
        """Reset daily counters - called at midnight"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        
    def update_trade_result(self, pnl: float, is_win: bool):
        """Update state after a trade completes"""
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.total_trades += 1
        self.daily_trades += 1
        
        if is_win:
            self.win_count += 1
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.consecutive_losses += 1
            
        # Update win rate
        if self.total_trades > 0:
            self.win_rate = self.win_count / self.total_trades
            
        # Update drawdown
        if self.total_pnl < 0:
            self.current_drawdown = abs(self.total_pnl)
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            self.current_drawdown = 0.0
            
    def has_position(self) -> bool:
        """Check if bot has an active position"""
        return self.position_size != 0
        
    def can_trade(self) -> bool:
        """Check if bot is in a state where it can place new trades"""
        return (
            self.is_running and 
            not self.has_position() and
            self.last_trade_time is None or 
            (time.time() - self.last_trade_time) > 300  # 5 min cooldown
        )
        
    def get_position_info(self) -> Dict[str, Any]:
        """Get current position information"""
        if not self.has_position():
            return {}
            
        return {
            'size': self.position_size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'is_long': self.is_long,
            'tp_price': self.tp_price,
            'sl_price': self.sl_price,
            'tp_oid': self.tp_oid,
            'sl_oid': self.sl_oid,
            'tp_sl_active': self.tp_sl_active,
            'trailing_active': self.trailing_active,
            'breakeven_shifted': self.breakeven_shifted
        }
        
    def clear_position(self):
        """Clear position state after closing"""
        self.position_size = 0
        self.entry_price = None
        self.entry_time = None
        self.tp_price = None
        self.sl_price = None
        self.tp_oid = None
        self.sl_oid = None
        self.tp_sl_active = False
        self.trailing_active = False
        self.breakeven_shifted = False
        self.active_triggers.clear()
        
    def set_position(self, size: int, entry_price: float, is_long: bool):
        """Set position state when entering a trade"""
        self.position_size = size
        self.entry_price = entry_price
        self.entry_time = time.time()
        self.is_long = is_long
        self.last_trade_time = time.time()
        
    def set_triggers(self, tp_oid: str, sl_oid: str, tp_price: float, sl_price: float):
        """Set TP/SL trigger state"""
        self.tp_oid = tp_oid
        self.sl_oid = sl_oid
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.tp_sl_active = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging/monitoring"""
        return {
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'is_long': self.is_long,
            'tp_sl_active': self.tp_sl_active,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'is_running': self.is_running,
            'error_count': self.error_count
        } 