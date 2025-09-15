"""
Volatility Analysis for Sweep Engine

Calculate volatility ratios for adaptive sweep behavior.
"""

import math
from typing import List, Sequence


def realized_vol_ratio(returns: Sequence[float], lookback: int, baseline_lookback: int) -> float:
    """
    Calculate volatility ratio: recent vol / baseline vol.
    
    Args:
        returns: Sequence of log returns or percentage returns
        lookback: Period for recent volatility calculation
        baseline_lookback: Period for baseline volatility calculation
    
    Returns:
        Volatility ratio (e.g., 2.0 means 2x recent volatility vs baseline)
    """
    if len(returns) < max(lookback, baseline_lookback):
        return 1.0
    
    def standard_deviation(values: Sequence[float]) -> float:
        """Calculate sample standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    # Calculate recent volatility
    recent_returns = returns[-lookback:] if len(returns) >= lookback else returns
    recent_vol = standard_deviation(recent_returns)
    
    # Calculate baseline volatility
    baseline_returns = returns[-baseline_lookback:] if len(returns) >= baseline_lookback else returns
    baseline_vol = standard_deviation(baseline_returns)
    
    # Return ratio, defaulting to 1.0 if baseline is zero
    if baseline_vol <= 0:
        return 1.0 if recent_vol <= 0 else 2.0  # High vol if recent > 0 but no baseline
    
    return recent_vol / baseline_vol


def simple_vol_ratio_from_prices(prices: List[float], lookback: int = 24, baseline: int = 48) -> float:
    """
    Calculate volatility ratio directly from price series.
    
    Args:
        prices: List of prices (most recent last)
        lookback: Periods for recent volatility
        baseline: Periods for baseline volatility
    
    Returns:
        Volatility ratio
    """
    if len(prices) < 2:
        return 1.0
    
    # Calculate log returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i] > 0 and prices[i-1] > 0:
            returns.append(math.log(prices[i] / prices[i-1]))
    
    if len(returns) < 2:
        return 1.0
    
    return realized_vol_ratio(returns, lookback, baseline)


def atr_based_vol_score(prices: List[float], atr_periods: int = 14, baseline_periods: int = 30) -> float:
    """
    Calculate volatility score using Average True Range (ATR).
    
    Args:
        prices: List of prices (OHLC assumed to be close prices)
        atr_periods: Periods for recent ATR
        baseline_periods: Periods for baseline ATR
    
    Returns:
        ATR-based volatility ratio
    """
    if len(prices) < max(atr_periods, baseline_periods):
        return 1.0
    
    def calculate_atr(price_series: List[float], periods: int) -> float:
        """Simple ATR approximation using price differences"""
        if len(price_series) < periods + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, min(len(price_series), periods + 1)):
            # For close prices, true range is just the absolute price change
            tr = abs(price_series[i] - price_series[i-1])
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    
    # Calculate recent and baseline ATR
    recent_atr = calculate_atr(prices, atr_periods)
    baseline_atr = calculate_atr(prices, baseline_periods)
    
    if baseline_atr <= 0:
        return 1.0 if recent_atr <= 0 else 2.0
    
    return recent_atr / baseline_atr


def get_vol_multiplier(vol_ratio: float, threshold: float = 2.0, multiplier: float = 1.5) -> float:
    """
    Get volatility multiplier for accumulator caps.
    
    Args:
        vol_ratio: Current volatility ratio
        threshold: Threshold for high volatility
        multiplier: Multiplier to apply when above threshold
    
    Returns:
        Multiplier (1.0 for normal vol, higher for elevated vol)
    """
    return multiplier if vol_ratio >= threshold else 1.0
