#!/usr/bin/env python3
"""
Utility Functions
================

Common utility functions for price normalization, tick alignment, and calculations.
Addresses mixed units and magic thresholds issues.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from decimal import Decimal, ROUND_HALF_UP


def pct(value: float) -> float:
    """Convert decimal to percentage (e.g., 0.05 -> 5.0)"""
    return value * 100.0


def bp(value: float) -> float:
    """Convert decimal to basis points (e.g., 0.0001 -> 1.0)"""
    return value * 10000.0


def normalize_percentage(value: float) -> float:
    """Normalize percentage to decimal (e.g., 5.0 -> 0.05)"""
    return value / 100.0


def normalize_basis_points(value: float) -> float:
    """Normalize basis points to decimal (e.g., 1.0 -> 0.0001)"""
    return value / 10000.0


def align_price_to_tick(price: float, tick_size: float, direction: str = "neutral") -> float:
    """Align price to tick size with specified direction"""
    if tick_size <= 0:
        return price
        
    # Use Decimal for precise rounding
    decimal_price = Decimal(str(price))
    decimal_tick = Decimal(str(tick_size))
    
    if direction == "up":
        # Round up to next tick
        aligned = (decimal_price / decimal_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * decimal_tick
    elif direction == "down":
        # Round down to previous tick
        aligned = (decimal_price / decimal_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * decimal_tick
    else:
        # Neutral rounding
        aligned = (decimal_price / decimal_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * decimal_tick
        
    return float(aligned)


def calculate_atr(prices: List[float], period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(prices) < period + 1:
        return 0.0
        
    true_ranges = []
    for i in range(1, len(prices)):
        high = prices[i]
        low = prices[i]
        prev_close = prices[i-1]
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
        
    if len(true_ranges) < period:
        return np.mean(true_ranges) if true_ranges else 0.0
        
    return np.mean(true_ranges[-period:])


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI
        
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow:
        return 0.0, 0.0, 0.0
        
    prices_array = np.array(prices)
    
    # Calculate EMAs
    ema_fast = calculate_ema(prices_array, fast)
    ema_slow = calculate_ema(prices_array, slow)
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    macd_values = []
    for i in range(len(prices)):
        if i >= slow - 1:
            macd_values.append(ema_fast[i] - ema_slow[i])
        else:
            macd_values.append(0.0)
            
    signal_line = calculate_ema(np.array(macd_values), signal)
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_momentum(prices: List[float], period: int = 5) -> float:
    """Calculate momentum (rate of change)"""
    if len(prices) < period + 1:
        return 0.0
        
    current_price = prices[-1]
    past_price = prices[-period - 1]
    
    if past_price == 0:
        return 0.0
        
    return (current_price - past_price) / past_price


def calculate_volatility(prices: List[float], period: int = 20) -> float:
    """Calculate price volatility (standard deviation of returns)"""
    if len(prices) < period + 1:
        return 0.0
        
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
            
    if len(returns) < period:
        return np.std(returns) if returns else 0.0
        
    return np.std(returns[-period:])


def validate_price_range(price: float, min_price: float, max_price: float) -> bool:
    """Validate price is within acceptable range"""
    return min_price <= price <= max_price


def calculate_spread(bid: float, ask: float) -> float:
    """Calculate bid-ask spread as percentage"""
    if bid <= 0 or ask <= 0:
        return 0.0
        
    return (ask - bid) / bid


def normalize_l2_snapshot(raw_data: dict, depth: int = 0) -> List[Tuple[float, float]]:
    """Normalize L2 order book snapshot"""
    if not raw_data or 'levels' not in raw_data:
        return []
        
    levels = raw_data['levels']
    if not levels:
        return []
        
    normalized = []
    for level in levels[:depth] if depth > 0 else levels:
        if len(level) >= 2:
            price = float(level[0])
            size = float(level[1])
            normalized.append((price, size))
            
    return normalized


def calculate_mid_price(bid: float, ask: float) -> float:
    """Calculate mid price from bid and ask"""
    if bid <= 0 or ask <= 0:
        return 0.0
        
    return (bid + ask) / 2


def calculate_weighted_average_price(orders: List[Tuple[float, float]]) -> float:
    """Calculate Volume Weighted Average Price (VWAP)"""
    if not orders:
        return 0.0
        
    total_volume = sum(size for _, size in orders)
    if total_volume == 0:
        return 0.0
        
    weighted_sum = sum(price * size for price, size in orders)
    return weighted_sum / total_volume


def format_price(price: float, tick_size: float) -> str:
    """Format price to appropriate decimal places based on tick size"""
    if tick_size >= 1:
        return f"{price:.0f}"
    elif tick_size >= 0.1:
        return f"{price:.1f}"
    elif tick_size >= 0.01:
        return f"{price:.2f}"
    elif tick_size >= 0.001:
        return f"{price:.3f}"
    elif tick_size >= 0.0001:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
        
    return (new_value - old_value) / old_value


def calculate_ema(prices: np.ndarray, period: int) -> float:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return np.mean(prices) if len(prices) > 0 else 0.0
        
    alpha = 2.0 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
        
    return ema


def calculate_sma(prices: List[float], period: int) -> float:
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return np.mean(prices) if prices else 0.0
        
    return np.mean(prices[-period:])


def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return 0.0, 0.0, 0.0
        
    sma = calculate_sma(prices, period)
    volatility = calculate_volatility(prices, period)
    
    upper_band = sma + (std_dev * volatility * sma)
    lower_band = sma - (std_dev * volatility * sma)
    
    return upper_band, sma, lower_band


def detect_support_resistance(prices: List[float], window: int = 20) -> Tuple[float, float]:
    """Detect support and resistance levels"""
    if len(prices) < window:
        return min(prices) if prices else 0.0, max(prices) if prices else 0.0
        
    recent_prices = prices[-window:]
    support = min(recent_prices)
    resistance = max(recent_prices)
    
    return support, resistance


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    
    return {
        '0.0': low,
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
        '1.0': high
    } 