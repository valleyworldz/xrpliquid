#!/usr/bin/env python3
"""
XRP Trading Bot Package
======================
A modular, high-performance XRP trading bot with advanced risk management.
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "Advanced XRP trading bot with modular architecture"

from .core.config import BotConfig, RuntimeState, RiskDecision, DEFAULT_CONFIG
from .core.utils import (
    align_price_to_tick,
    calculate_atr,
    calculate_rsi,
    calculate_macd,
    normalize_l2_snapshot,
    calculate_spread,
    calculate_mid_price
)

__all__ = [
    "BotConfig",
    "RuntimeState", 
    "RiskDecision",
    "DEFAULT_CONFIG",
    "align_price_to_tick",
    "calculate_atr",
    "calculate_rsi",
    "calculate_macd",
    "normalize_l2_snapshot",
    "calculate_spread",
    "calculate_mid_price"
] 