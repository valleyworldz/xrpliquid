#!/usr/bin/env python3
"""
Trading Bot Configuration
=========================
Centralized configuration management with type safety.
All magic numbers and thresholds are defined here.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class RiskDecision(Enum):
    """Risk assessment decisions"""
    OK = "ok"
    COOL_DOWN = "cool_down"
    LIMIT_HIT = "limit_hit"
    BAD_RR = "bad_rr"
    BAD_FUNDING = "bad_funding"
    BAD_ATR = "bad_atr"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"

@dataclass
class TradingSettings:
    """Immutable trading configuration"""
    
    # Position sizing
    min_xrp: float = 10.0
    min_notional: float = 10.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 100.0
    
    # Signal thresholds
    confidence_threshold: float = 0.7
    volume_threshold: float = 0.8
    min_signal_strength: float = 0.7
    
    # Time constraints
    min_hold_time_minutes: int = 15
    max_hold_time_hours: int = 4
    post_trade_cooldown_minutes: int = 5
    
    # Risk limits
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_drawdown_pct: float = 0.10    # 10% max drawdown
    max_consecutive_losses: int = 3
    max_daily_trades: int = 50
    
    # Funding rate
    funding_rate_threshold: float = 0.0001
    funding_rate_buffer: float = 0.0001
    
    # Stop loss and take profit
    stop_loss_pct: float = 0.025      # 2.5% stop loss
    profit_target_pct: float = 0.035  # 3.5% profit target
    min_tp_distance_pct: float = 0.015
    min_sl_distance_pct: float = 0.008
    min_trigger_distance_pct: float = 0.0005
    
    # ATR settings
    atr_period: int = 14
    atr_multiplier_sl: float = 2.5
    atr_multiplier_tp: float = 5.0
    atr_period_low_vol: int = 21
    atr_period_high_vol: int = 7
    volatility_threshold: float = 0.005
    
    # Trailing stops
    min_trailing_distance_pct: float = 0.005
    default_trailing_distance_pct: float = 0.015
    trailing_atr_multiplier: float = 1.5
    trailing_activation_threshold: float = 0.5
    
    # Fees and buffers
    fee_buffer: float = 0.0006
    spread_buffer: float = 0.0001
    
    # Technical indicators
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    momentum_period: int = 5
    volatility_period: int = 20
    
    # Risk-reward
    min_rr_ratio: float = 2.0
    min_atr_pct: float = 0.001
    max_atr_pct: float = 0.1
    max_fee_to_reward_ratio: float = 0.1
    
    # Performance tracking
    min_trades_for_adaptation: int = 10
    adaptation_interval_seconds: int = 3600
    win_rate_threshold: float = 0.4
    profit_threshold_pct: float = 0.001
    
    # Compound sizing
    compound_enabled: bool = True
    compound_factor: float = 1.1
    max_compound_size: float = 500.0
    
    # Emergency settings
    emergency_stop_loss_pct: float = 0.005
    emergency_min_trade_interval: int = 300
    emergency_base_position_size: float = 5.0
    
    # Breakeven and trailing
    breakeven_activation_pct: float = 0.01
    breakeven_buffer_pct: float = 0.005
    fallback_trailing_distance_pct: float = 0.015
    fallback_profit_threshold_pct: float = 0.02
    
    # Margin requirements
    min_collateral: float = 50.0
    margin_ratio_threshold: float = 2.0
    
    # Trend analysis
    trend_strength_threshold: float = 0.0005
    min_bars_between_trades: int = 3
    
    # TP/SL tiers (for advanced strategies)
    tp_tiers: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.03])
    tp_sizes: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            errors.append("risk_per_trade must be between 0 and 0.1")
        
        if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > 0.5:
            errors.append("max_daily_loss_pct must be between 0 and 0.5")
        
        if self.min_rr_ratio < 1:
            errors.append("min_rr_ratio must be >= 1")
        
        if self.atr_multiplier_sl <= 0:
            errors.append("atr_multiplier_sl must be > 0")
        
        if self.atr_multiplier_tp <= 0:
            errors.append("atr_multiplier_tp must be > 0")
        
        return errors

@dataclass
class RuntimeState:
    """Mutable runtime state - separate from configuration"""
    
    # Position tracking
    current_position_size: int = 0
    current_entry_price: float = 0.0
    current_entry_time: float = 0.0
    is_long: bool = False
    
    # TP/SL state
    tp_sl_active: bool = False
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_oid: Optional[str] = None
    sl_oid: Optional[str] = None
    active_triggers: dict = field(default_factory=dict)
    
    # Performance tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    win_rate: float = 0.0
    
    # Risk tracking
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    
    # Timing
    last_trade_time: float = 0.0
    last_bar_time: float = 0.0
    
    # Compound sizing
    compound_factor: float = 1.0
    base_position_size: float = 10.0
    
    # Cached data
    last_price: float = 0.0
    last_funding_rate: Optional[float] = None
    last_atr: float = 0.0
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
    
    def update_equity(self, new_equity: float):
        """Update equity and track drawdown"""
        self.current_equity = new_equity
        
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - new_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def record_trade(self, pnl: float, is_win: bool):
        """Record trade result"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.total_trades += 1
        
        if is_win:
            self.total_wins += 1
            self.consecutive_losses = 0
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
        
        self.win_rate = self.total_wins / self.total_trades if self.total_trades > 0 else 0.0
        self.last_trade_time = 0.0  # Will be set by caller

# Default configurations
DEFAULT_SETTINGS = TradingSettings()
CONSERVATIVE_SETTINGS = TradingSettings(
    risk_per_trade=0.01,
    max_daily_loss_pct=0.03,
    max_consecutive_losses=2,
    min_rr_ratio=2.5,
    emergency_base_position_size=2.0
)
AGGRESSIVE_SETTINGS = TradingSettings(
    risk_per_trade=0.03,
    max_daily_loss_pct=0.08,
    max_consecutive_losses=4,
    min_rr_ratio=1.8,
    emergency_base_position_size=10.0
) 