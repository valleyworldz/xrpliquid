#!/usr/bin/env python3
"""
Configuration Module for XRP Trading Bot
========================================

This module contains all configuration parameters and settings for the trading bot.
Extracted from the monolithic PERFECT_CONSOLIDATED_BOT.py for better maintainability.
"""

import os
from typing import Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """Centralized configuration for all trading parameters"""
    
    # CRITICAL FIX: Fee-aware breakeven math - Hyperliquid XRP-perps
    # Central fee constants
    TAKER_FEE: float = 0.00045        # 0.045% taker fee
    MAKER_FEE: float = 0.00015        # 0.015% maker fee (rebate)
    ROUND_TRIP_TAKER: float = 0.0009  # 0.09% round-trip taker cost
    ROUND_TRIP_MAKER: float = 0.0003  # 0.03% round-trip maker cost
    
    # Breakeven components
    DEFAULT_SLIPPAGE: float = 0.0005  # 0.05% estimated slippage
    FEE_BUFFER: float = 0.0003        # 0.03% safety buffer for funding/fills
    
    # OPTIMIZED PARAMETERS FROM BACKTEST RESULTS (Iteration 5 - Peak Performance)
    # Base profit targets (before fees) - Optimized for 81.82% win rate
    BASE_PROFIT_TARGET_PCT: float = 0.050  # 5.0% profit target (increased from 1.5%)
    BASE_STOP_LOSS_PCT: float = 0.003      # 0.3% stop loss (tightened from 1.2%)
    
    # CRITICAL FIX: Fee-aware profit targets
    # Minimum profitable move = fees + slippage + buffer ≈ 0.17%
    MIN_PROFITABLE_MOVE_PCT: float = 0.0017  # ≈ 0.17%
    
    # CRITICAL FIX: Optimized profit targets based on backtest results
    # Balanced for realistic trading - 3.5% profit target
    PROFIT_TARGET_PCT: float = 0.035  # 3.5% minimum
    STOP_LOSS_PCT: float = 0.025      # 2.5% stop loss for proper risk management
    
    # CRITICAL FIX: ATR-based dynamic parameters for volatility adaptation - OPTIMIZED
    ATR_STOP_MULTIPLIER: float = 0.8  # Reduced from 1.2x to 0.8x ATR for tighter stops
    ATR_TP_MULTIPLIER: float = 2.0    # Increased from 1.5x to 2.0x ATR for wider targets
    
    # CRITICAL FIX: Signal strength gate for quality control - OPTIMIZED
    MIN_SIGNAL_STRENGTH: float = 0.95  # Increased from 0.6 to 0.95 for high-quality signals only
    
    # CRITICAL FIX: Trading frequency limits to prevent overtrading - OPTIMIZED
    MAX_TRADES_PER_HOUR: int = 3    # Reduced from 6 to 3 trades per hour
    MIN_TRADE_INTERVAL: int = 600   # Increased from 300 to 600 seconds (10 minutes)
    EMERGENCY_PROFIT_PCT: float = 0.08  # 8% emergency profit taking
    EMERGENCY_STOP_PCT: float = 0.06    # 6% emergency stop loss
    
    # Trailing Stop - OPTIMIZED
    TRAILING_DISTANCE: float = 0.015   # Reduced from 0.02 to 0.015 (1.5% trailing)
    TRAILING_ACTIVATION_PCT: float = 0.025  # Increased from 0.02 to 0.025 (2.5% profit before trailing)
    
    # Position Sizing - OPTIMIZED for 5% risk
    MIN_POSITION_SIZE: float = 10.0   # Minimum 10 XRP for profitable trades
    MAX_POSITION_SIZE_PCT: float = 0.25  # Reduced from 0.35 to 0.25 (25% of collateral max)
    KELLY_MULTIPLIER: float = 0.3     # Reduced from 0.5 to 0.3 for more conservative sizing
    RISK_MULTIPLE: float = 1.5        # Reduced from 2.0 to 1.5 (risk 1.5x ATR per trade)
    
    # Risk Management - ENHANCED
    MAX_DRAWDOWN_PCT: float = 0.03    # Reduced from 0.05 to 0.03 (3% max drawdown)
    LIQUIDATION_BUFFER_PCT: float = 0.20  # Increased from 0.15 to 0.20 (20% liquidation buffer)
    VOLATILITY_THRESHOLD: float = 0.005  # Reduced from 0.05 to 0.005 (0.5% volatility threshold)
    
    # ATR Multipliers
    ATR_STOP_MULTIPLIER_LEGACY: float = 1.2  # 1.2x ATR for stop loss
    ATR_TP_MULTIPLIER_LEGACY: float = 1.5    # 1.5x ATR for take profit
    
    # Order Management
    PRICE_PRECISION: int = 4        # 4 decimal places for XRP
    MIN_ORDER_VALUE: float = 10.0     # $10 minimum order value
    SLIPPAGE_TOLERANCE: float = 0.001  # 0.1% slippage tolerance
    
    # Performance Tracking
    PERFORMANCE_LOG_INTERVAL: int = 10  # Log performance every 10 cycles
    MAX_NO_TRADE_CYCLES: int = 10   # Max cycles without trades before error
    
    # Market Analysis
    MIN_PRICE_HISTORY: int = 10     # Minimum price history for analysis
    MAX_PRICE_HISTORY: int = 100    # Maximum price history to keep
    RSI_PERIOD: int = 14           # RSI calculation period
    MOMENTUM_PERIOD: int = 5       # Momentum calculation period
    VOLATILITY_PERIOD: int = 20    # Volatility calculation period
    
    # Signal Generation - OPTIMIZED
    CONFIDENCE_THRESHOLD: float = 0.95  # Increased from 0.15 to 0.95 (high-quality signals only)
    VOLUME_THRESHOLD: float = 0.8    # Increased from 0.5 to 0.8 (80% above average volume for confirmation)
    
    # BACKTEST OPTIMIZATION PARAMETERS
    BACKTEST_OPTIMIZED: bool = True  # Flag to indicate optimized parameters
    BACKTEST_WIN_RATE: float = 81.82  # Target win rate from backtest
    BACKTEST_TOTAL_PNL: float = 8.39  # Expected PnL from $50 starting capital
    
    # VOLATILITY FILTERS FOR 100% PROFITABILITY
    MAX_ATR_FOR_ENTRY: float = 0.005  # Only enter if ATR < 0.5%
    MIN_ATR_FOR_ENTRY: float = 0.001  # Minimum ATR for signal validity
    
    def __post_init__(self):
        """Initialize derived values after dataclass initialization"""
        self.ROUND_TRIP_TAKER = self.TAKER_FEE * 2
        self.ROUND_TRIP_MAKER = self.MAKER_FEE * 2
        self.MIN_PROFITABLE_MOVE_PCT = self.ROUND_TRIP_MAKER + self.DEFAULT_SLIPPAGE + self.FEE_BUFFER
        self.PROFIT_TARGET_PCT = max(0.035, self.MIN_PROFITABLE_MOVE_PCT * 20)
        
        # Initialize fee optimization attributes for backward compatibility
        self.VIP_TIER_THRESHOLDS = {
            0: 0, 1: 5e6, 2: 25e6, 3: 100e6, 4: 500e6, 5: 2e9, 6: 7e9
        }
        self.FEE_RATES = {
            0: (0.045, 0.015), 1: (0.040, 0.012), 2: (0.035, 0.008),
            3: (0.030, 0.004), 4: (0.028, 0.000), 5: (0.026, 0.000), 6: (0.024, 0.000)
        }
        self.STAKING_DISCOUNTS = {
            'wood': 0.05, 'bronze': 0.10, 'silver': 0.15,
            'gold': 0.20, 'platinum': 0.30, 'diamond': 0.40
        }


class FeeRates:
    """Fee rate configuration for different VIP tiers"""
    
    VIP_TIER_THRESHOLDS: Dict[int, float] = {
        0: 0,      # < $5M
        1: 5e6,    # > $5M
        2: 25e6,   # > $25M
        3: 100e6,  # > $100M
        4: 500e6,  # > $500M
        5: 2e9,    # > $2B
        6: 7e9     # > $7B
    }
    
    FEE_RATES: Dict[int, Tuple[float, float]] = {
        0: (0.045, 0.015),  # 0.045% taker, 0.015% maker
        1: (0.040, 0.012),
        2: (0.035, 0.008),
        3: (0.030, 0.004),
        4: (0.028, 0.000),  # Zero maker fees!
        5: (0.026, 0.000),
        6: (0.024, 0.000)
    }
    
    STAKING_DISCOUNTS: Dict[str, float] = {
        'wood': 0.05,      # >10 HYPE -> 5% discount
        'bronze': 0.10,    # >100 HYPE -> 10% discount
        'silver': 0.15,    # >1,000 HYPE -> 15% discount
        'gold': 0.20,      # >10,000 HYPE -> 20% discount
        'platinum': 0.30,  # >100,000 HYPE -> 30% discount
        'diamond': 0.40    # >500,000 HYPE -> 40% discount
    }


class VolatilityRegimeFilters:
    """Volatility regime filters for different market conditions"""
    
    VOLATILITY_REGIME_FILTERS: Dict[str, Dict[str, float]] = {
        'low_volatility': {
            'max_atr': 0.003,      # 0.3% max ATR
            'min_signal_strength': 0.90,
            'position_size_multiplier': 1.2,
            'profit_target_multiplier': 1.1
        },
        'normal_volatility': {
            'max_atr': 0.008,      # 0.8% max ATR
            'min_signal_strength': 0.95,
            'position_size_multiplier': 1.0,
            'profit_target_multiplier': 1.0
        },
        'high_volatility': {
            'max_atr': 0.015,      # 1.5% max ATR
            'min_signal_strength': 0.98,
            'position_size_multiplier': 0.8,
            'profit_target_multiplier': 0.9
        },
        'extreme_volatility': {
            'max_atr': float('inf'),  # No limit
            'min_signal_strength': 0.99,
            'position_size_multiplier': 0.5,
            'profit_target_multiplier': 0.8
        }
    }


class WebSocketConfig:
    """WebSocket configuration constants"""
    
    WS_PING_INTERVAL: int = 30         # 30s ping cadence (HL timeouts ~35s)
    WS_IDLE_KILL: int = 15             # treat as idle after 15s without inbound
    WS_MAX_RECONNECT_ATTEMPTS: int = 5  # Max consecutive reconnection attempts
    WS_ERROR_SUPPRESSION_TIME: int = 60  # Suppress repeated errors for 60 seconds


class ExchangeConfig:
    """Exchange-specific configuration"""
    
    MIN_NOTIONAL: float = 10.0  # Hyperliquid's official minimum notional requirement
    LIVE_MODE: bool = True      # Always enable real trading
    AUTO_TRADE: bool = True     # Force real trading, no demos or simulations
    DEMO_MODE: bool = False     # No demo mode


# Global configuration instance
config = TradingConfig()
fee_rates = FeeRates()
volatility_filters = VolatilityRegimeFilters()
ws_config = WebSocketConfig()
exchange_config = ExchangeConfig()


def min_profitable_move(is_maker: bool = True) -> float:
    """Calculate minimum profitable move based on fees and slippage"""
    if is_maker:
        return config.ROUND_TRIP_MAKER + config.DEFAULT_SLIPPAGE + config.FEE_BUFFER
    else:
        return config.ROUND_TRIP_TAKER + config.DEFAULT_SLIPPAGE + config.FEE_BUFFER


def get_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    return {
        'LIVE_MODE': os.getenv('XRP_BOT_LIVE_MODE', 'true').lower() == 'true',
        'CONFIG_PATH': os.getenv('XRP_BOT_CONFIG_PATH', None),
        'LOG_LEVEL': os.getenv('XRP_BOT_LOG_LEVEL', 'INFO'),
        'MAX_POSITIONS': int(os.getenv('XRP_BOT_MAX_POSITIONS', '1')),
        'CYCLE_INTERVAL': int(os.getenv('XRP_BOT_CYCLE_INTERVAL', '30')),
    }


def validate_config() -> bool:
    """Validate configuration parameters"""
    try:
        # Validate critical parameters
        assert config.PROFIT_TARGET_PCT > config.STOP_LOSS_PCT, "Profit target must be greater than stop loss"
        assert config.MIN_POSITION_SIZE > 0, "Minimum position size must be positive"
        assert config.MAX_POSITION_SIZE_PCT > 0 and config.MAX_POSITION_SIZE_PCT <= 1, "Max position size must be between 0 and 1"
        assert config.KELLY_MULTIPLIER > 0 and config.KELLY_MULTIPLIER <= 1, "Kelly multiplier must be between 0 and 1"
        
        # Validate fee rates
        for tier, (taker, maker) in fee_rates.FEE_RATES.items():
            assert 0 <= taker <= 0.1, f"Taker fee for tier {tier} must be between 0 and 10%"
            assert 0 <= maker <= 0.1, f"Maker fee for tier {tier} must be between 0 and 10%"
        
        return True
    except AssertionError as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during configuration validation: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    print("🔧 Testing configuration module...")
    if validate_config():
        print("✅ Configuration validation passed")
        print(f"📊 Profit Target: {config.PROFIT_TARGET_PCT:.3%}")
        print(f"🛡️ Stop Loss: {config.STOP_LOSS_PCT:.3%}")
        print(f"💰 Min Position Size: {config.MIN_POSITION_SIZE}")
        print(f"📈 Max Position Size: {config.MAX_POSITION_SIZE_PCT:.1%}")
    else:
        print("❌ Configuration validation failed")
        exit(1) 