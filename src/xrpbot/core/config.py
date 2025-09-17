#!/usr/bin/env python3
"""
Configuration Management
=======================
Centralized configuration for the XRP trading bot.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class RiskDecision(Enum):
    """Risk assessment decisions"""
    OK = "ok"
    BAD_RR = "bad_rr"
    BAD_ATR = "bad_atr"
    BAD_SIZE = "bad_size"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    COOLDOWN_ACTIVE = "cooldown_active"
    DAILY_LIMIT_HIT = "daily_limit_hit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    FUNDING_RISK = "funding_risk"
    HOLD_TIME_EXCEEDED = "hold_time_exceeded"

@dataclass
class BotConfig:
    """Main bot configuration with all trading parameters"""
    
    # Core trading parameters
    min_xrp: float = 10.0
    min_notional: float = 10.0
    risk_per_trade: float = 0.02
    confidence_threshold: float = 0.02
    volume_threshold: float = 0.8
    min_hold_time_minutes: int = 15
    max_hold_time_hours: int = 4
    funding_rate_threshold: float = 0.0001
    funding_rate_buffer: float = 0.0001
    funding_close_buffer: float = 0.0002  # Double threshold for closing positions
    stop_loss_pct: float = 0.025
    profit_target_pct: float = 0.035
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.10
    max_consecutive_losses: int = 3
    post_trade_cooldown_minutes: int = 5
    
    # ATR and technical indicators
    atr_period: int = 14
    atr_multiplier_sl: float = 2.5
    atr_multiplier_tp: float = 5.0
    atr_period_low_vol: int = 21
    atr_period_high_vol: int = 7
    volatility_threshold: float = 0.005
    
    # Trailing stop parameters
    min_trailing_distance_pct: float = 0.005
    default_trailing_distance_pct: float = 0.015
    trailing_atr_multiplier: float = 1.5
    trailing_activation_threshold: float = 0.5
    
    # Fee and distance parameters
    fee_buffer: float = 0.0006
    min_tp_distance_pct: float = 0.015
    min_sl_distance_pct: float = 0.008
    min_trigger_distance_pct: float = 0.0005
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Risk management
    min_collateral: float = 50.0
    min_profit_threshold_pct: float = 0.02
    min_bars_between_trades: int = 3
    trend_strength_threshold: float = 0.0005
    
    # Fallback and buffer parameters
    fallback_profit_threshold_pct: float = 0.02
    breakeven_buffer_pct: float = 0.005
    fallback_trailing_distance_pct: float = 0.015
    
    # GROK TP/SL parameters
    grok_tp_tiers: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.03])
    grok_tp_sizes: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    grok_breakeven_activation: float = 0.01
    grok_trailing_activation: float = 0.005
    grok_atr_multiplier: float = 2.0
    
    # Compound position sizing
    grok_compound_enabled: bool = True
    grok_base_position_size: float = 5.0
    grok_compound_factor: float = 1.1
    grok_max_position_size: float = 10.0
    grok_risk_per_trade: float = 0.02
    
    # Performance tracking
    grok_performance_tracking: bool = True
    grok_adaptation_interval: int = 3600
    grok_min_trades_for_adaptation: int = 10
    grok_win_rate_threshold: float = 0.4
    grok_profit_threshold: float = 0.001
    
    # Emergency parameters
    emergency_stop_loss_pct: float = 0.005
    emergency_min_trade_interval: int = 300
    emergency_max_daily_trades: int = 50
    emergency_max_consecutive_losses: int = 3
    emergency_min_signal_strength: float = 0.7
    emergency_max_daily_loss: float = 0.1
    emergency_base_position_size: float = 5.0
    emergency_max_position_size: float = 10.0
    
    # Magic numbers
    magic_buffer_pct: float = 0.005
    magic_trailing_fallback_pct: float = 0.015
    magic_fee_buffer: float = 0.0006
    magic_min_tp_distance_pct: float = 0.015
    magic_min_sl_distance_pct: float = 0.008
    
    # API and connection settings
    api_timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_prometheus: bool = True
    prometheus_port: int = 8000
    
    # Email alerts
    email_alerts_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load configuration from environment variables"""
        return cls(
            min_xrp=safe_float(os.getenv("MIN_XRP", "10.0")),
            risk_per_trade=safe_float(os.getenv("RISK_PER_TRADE", "0.02")),
            confidence_threshold=safe_float(os.getenv("CONFIDENCE_THRESHOLD", "0.02")),
            max_daily_loss_pct=safe_float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05")),
            max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8000")),
            email_alerts_enabled=os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true",
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if self.risk_per_trade <= 0 or self.risk_per_trade > 1:
            errors.append("risk_per_trade must be between 0 and 1")
        
        if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > 1:
            errors.append("max_daily_loss_pct must be between 0 and 1")
        
        if self.max_consecutive_losses <= 0:
            errors.append("max_consecutive_losses must be positive")
        
        if self.min_xrp <= 0:
            errors.append("min_xrp must be positive")
        
        if self.atr_period <= 0:
            errors.append("atr_period must be positive")
        
        return errors

@dataclass
class RuntimeState:
    """Runtime state tracking"""
    current_bar: int = 0
    last_trade_time: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
    
    def update_trade_result(self, pnl: float):
        """Update trade statistics"""
        self.total_trades += 1
        self.daily_trades += 1
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.total_wins += 1
            self.consecutive_losses = 0
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
    
    @property
    def win_rate(self) -> float:
        """Calculate current win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades

# Default configuration instance
DEFAULT_CONFIG = BotConfig() 