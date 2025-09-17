"""
Price Utilities Module
======================
Low-level price manipulation and technical indicators.
All functions are pure and stateless for easy testing.
"""

from src.core.utils.decimal_boundary_guard import safe_decimal
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from typing import Union, List, Optional
import numpy as np


def align_price_to_tick(price: Union[float, Decimal], tick_size: Union[float, Decimal], direction: str) -> Decimal:
    """
    Align price to tick size with direction-aware rounding.
    
    Args:
        price: Price to align
        tick_size: Exchange tick size
        direction: "up"/"tp"/"buy" or "down"/"sl"/"sell"
    
    Returns:
        Decimal: Aligned price (for precision)
    
    Raises:
        ValueError: If direction is invalid
    """
    if tick_size <= 0:
        return safe_decimal(str(price))
    
    if direction not in ("up", "tp", "buy", "down", "sl", "sell"):
        raise ValueError(f"Direction must be one of: up/tp/buy/down/sl/sell, got: {direction}")
    
    p = safe_decimal(str(price))
    t = safe_decimal(str(tick_size))
    
    if direction in ("up", "tp", "buy"):
        # For buys/TP orders, round up to ensure we don't get a worse price
        return (p/t).to_integral_value(rounding=ROUND_CEILING) * t
    else:  # down, sl, sell
        # For sells/SL orders, round down to ensure we don't get a worse price
        return (p/t).to_integral_value(rounding=ROUND_FLOOR) * t


def calculate_atr(prices: List[float], period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) using close-only variant.
    
    Args:
        prices: List of closing prices
        period: ATR period (default 14)
    
    Returns:
        float: ATR value (always positive)
    """
    if len(prices) < period + 1:
        # CRITICAL: Clamp to at least tick_size * 2 to prevent invalid orders
        tick_size = 0.0001  # XRP tick size
        min_atr = tick_size * 2  # At least 2 ticks
        min_atr_pct = 0.001  # 0.1% of current price
        return max(prices[-1] * min_atr_pct, min_atr) if prices else min_atr
    
    # FIXED: Use proper close-only ATR variant
    # If you only have closes, treat TR as abs(close[i] - close[i-1])
    true_ranges = []
    for i in range(1, len(prices)):
        prev_close = prices[i-1]
        curr_close = prices[i]
        true_range = abs(curr_close - prev_close)  # Simple close-to-close range
        true_ranges.append(true_range)
    
    # Calculate ATR as simple moving average of true ranges
    if len(true_ranges) >= period:
        atr = sum(true_ranges[-period:]) / period
    else:
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0.001
    
    # Use percentage-based minimum instead of fixed dollar amount
    min_atr_pct = 0.001  # 0.1% of current price
    min_atr = prices[-1] * min_atr_pct if prices else 0.001
    return max(atr, min_atr)


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: List of closing prices
        period: RSI period (default 14)
    
    Returns:
        float: RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI
    
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    
    rs = gains / (losses or 1e-9)
    return 100 - (100 / (1 + rs))


def calculate_momentum(prices: List[float], period: int = 5) -> float:
    """
    Calculate momentum (rate of change).
    
    Args:
        prices: List of closing prices
        period: Momentum period (default 5)
    
    Returns:
        float: Momentum value (can be negative)
    """
    if len(prices) < period + 1:
        return 0.0
    
    return (prices[-1] - prices[-period-1]) / prices[-period-1]


def calculate_volatility(prices: List[float], period: int = 20) -> float:
    """
    Calculate volatility as standard deviation / mean.
    
    Args:
        prices: List of closing prices
        period: Volatility period (default 20)
    
    Returns:
        float: Volatility value (always positive)
    """
    if len(prices) < period:
        return 0.0
    
    recent = prices[-period:]
    mean_price = sum(recent) / len(recent)
    variance = sum((p - mean_price) ** 2 for p in recent) / len(recent)
    return (variance ** 0.5) / mean_price


def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        dict: MACD values {'macd': float, 'signal': float, 'histogram': float}
    """
    if len(prices) < max(fast, slow, signal):
        return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def ema_series(data: List[float], period: int) -> List[float]:
        """Calculate EMA series."""
        ema_values = []
        multiplier = 2.0 / (period + 1)
        
        # First EMA is SMA
        ema_values.append(sum(data[:period]) / period)
        
        # Calculate EMA for remaining values
        for i in range(period, len(data)):
            ema = (data[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    # Calculate fast and slow EMAs
    fast_ema = ema_series(prices, fast)
    slow_ema = ema_series(prices, slow)
    
    # Calculate MACD line
    macd_line = []
    for i in range(len(slow_ema)):
        macd_line.append(fast_ema[i + (slow - fast)] - slow_ema[i])
    
    # Calculate signal line (EMA of MACD)
    signal_line = ema_series(macd_line, signal)
    
    # Calculate histogram
    histogram = []
    for i in range(len(signal_line)):
        histogram.append(macd_line[i + (signal - 1)] - signal_line[i])
    
    return {
        'macd': macd_line[-1] if macd_line else 0.0,
        'signal': signal_line[-1] if signal_line else 0.0,
        'histogram': histogram[-1] if histogram else 0.0
    } 