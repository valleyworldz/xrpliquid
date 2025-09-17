#!/usr/bin/env python3
"""
Utility Functions
================
Common utility functions for price normalization, tick alignment, and calculations.
"""

from src.core.utils.decimal_boundary_guard import safe_decimal
from src.core.utils.decimal_boundary_guard import safe_float
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING, ROUND_FLOOR
import math

def pct(value: float) -> float:
    """Convert to percentage (multiply by 100)"""
    return value * 100

def bp(value: float) -> float:
    """Convert to basis points (multiply by 10000)"""
    return value * 10000

def normalize_percentage(value: float) -> float:
    """Normalize percentage value to decimal (divide by 100)"""
    return value / 100

def normalize_basis_points(value: float) -> float:
    """Normalize basis points to decimal (divide by 10000)"""
    return value / 10000

def align_price_to_tick(price: float, tick_size: float, direction: str = "neutral") -> float:
    """Align price to tick size with direction-aware rounding"""
    try:
        if tick_size <= 0:
            return price
        
        # Use Decimal for precise tick alignment
        p = safe_decimal(str(price))
        t = safe_decimal(str(tick_size))
        
        # Explicit rounding based on direction
        if direction in ("up", "tp", "buy"):
            # For buys/TP orders, round up to ensure we don't get a worse price
            return safe_float((p/t).to_integral_value(rounding=ROUND_CEILING) * t)
        elif direction in ("down", "sl", "sell"):
            # For sells/SL orders, round down to ensure we don't get a worse price
            return safe_float((p/t).to_integral_value(rounding=ROUND_FLOOR) * t)
        else:
            # For neutral, use bankers' rounding (default)
            return safe_float((p/t).to_integral_value() * t)
            
    except Exception:
        return round(price, 4)

def calculate_atr(prices: List[float], period: int = 14) -> float:
    """Calculate Average True Range (ATR) with close-only variant"""
    try:
        if len(prices) < period + 1:
            # Use percentage-based minimum instead of fixed dollar amount
            min_atr_pct = 0.001  # 0.1% of current price
            return max(prices[-1] * min_atr_pct, 0.001) if prices else 0.001
        
        # Use proper close-only ATR variant
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
        
    except Exception:
        return 0.001

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI)"""
    try:
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if not enough data
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    except Exception:
        return 50.0

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        macd_values = []
        for i in range(len(prices) - slow + 1):
            ema_f = calculate_ema(prices[i:i+slow], fast)
            ema_s = calculate_ema(prices[i:i+slow], slow)
            macd_values.append(ema_f - ema_s)
        
        signal_line = calculate_ema(macd_values, signal) if len(macd_values) >= signal else macd_line
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    except Exception:
        return 0.0, 0.0, 0.0

def calculate_ema(prices: List[float], period: int) -> float:
    """Calculate Exponential Moving Average (EMA)"""
    try:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
        
    except Exception:
        return prices[-1] if prices else 0.0

def calculate_momentum(prices: List[float], period: int = 5) -> float:
    """Calculate price momentum"""
    try:
        if len(prices) < period:
            return 0.0
        
        return (prices[-1] - prices[-period]) / prices[-period]
        
    except Exception:
        return 0.0

def calculate_volatility(prices: List[float], period: int = 20) -> float:
    """Calculate price volatility (standard deviation)"""
    try:
        if len(prices) < period:
            return 0.0
        
        recent_prices = prices[-period:]
        mean_price = sum(recent_prices) / len(recent_prices)
        variance = sum((p - mean_price) ** 2 for p in recent_prices) / len(recent_prices)
        return math.sqrt(variance)
        
    except Exception:
        return 0.0

def validate_price_range(price: float, min_price: float, max_price: float) -> bool:
    """Validate if price is within acceptable range"""
    return min_price <= price <= max_price

def calculate_spread(bid: float, ask: float) -> float:
    """Calculate bid-ask spread as percentage"""
    try:
        if bid <= 0 or ask <= 0:
            return 0.0
        return (ask - bid) / bid
    except Exception:
        return 0.0

def normalize_l2_snapshot(raw_data: Dict[str, Any], depth: int = 0) -> List[Tuple[float, float]]:
    """Normalize L2 orderbook snapshot"""
    try:
        if not raw_data or "levels" not in raw_data:
            return []
        
        levels = raw_data["levels"]
        if not levels:
            return []
        
        # Extract bids and asks
        bids = []
        asks = []
        
        for level in levels:
            if "bids" in level:
                for bid in level["bids"]:
                    if len(bid) >= 2:
                        price = safe_float(bid[0])
                        size = safe_float(bid[1])
                        bids.append((price, size))
            
            if "asks" in level:
                for ask in level["asks"]:
                    if len(ask) >= 2:
                        price = safe_float(ask[0])
                        size = safe_float(ask[1])
                        asks.append((price, size))
        
        # Sort bids (descending) and asks (ascending)
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        # Limit depth if specified
        if depth > 0:
            bids = bids[:depth]
            asks = asks[:depth]
        
        return {"bids": bids, "asks": asks}
        
    except Exception:
        return {"bids": [], "asks": []}

def calculate_mid_price(bid: float, ask: float) -> float:
    """Calculate mid price from bid and ask"""
    try:
        return (bid + ask) / 2
    except Exception:
        return 0.0

def calculate_weighted_average_price(orders: List[Tuple[float, float]]) -> float:
    """Calculate Volume Weighted Average Price (VWAP)"""
    try:
        if not orders:
            return 0.0
        
        total_volume = sum(size for _, size in orders)
        if total_volume == 0:
            return 0.0
        
        weighted_sum = sum(price * size for price, size in orders)
        return weighted_sum / total_volume
        
    except Exception:
        return 0.0

def format_price(price: float, tick_size: float) -> str:
    """Format price according to tick size"""
    try:
        if tick_size <= 0:
            return f"{price:.4f}"
        
        # Determine decimal places based on tick size
        decimal_places = 0
        tick = tick_size
        while tick < 1 and tick > 0:
            tick *= 10
            decimal_places += 1
        
        return f"{price:.{decimal_places}f}"
        
    except Exception:
        return f"{price:.4f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    try:
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    except Exception:
        return 0.0

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between minimum and maximum"""
    return max(min_val, min(value, max_val))

def is_within_tolerance(value1: float, value2: float, tolerance: float = 0.001) -> bool:
    """Check if two values are within tolerance"""
    return abs(value1 - value2) <= tolerance

def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to nearest tick size"""
    try:
        if tick_size <= 0:
            return price
        
        return round(price / tick_size) * tick_size
    except Exception:
        return price 