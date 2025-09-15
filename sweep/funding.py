"""
Funding Rate Utilities

Calculate funding impact and blackout windows for sweep timing.
"""

import time
from typing import Tuple


def equity_funding_impact_bps(equity: float, position_notional: float, next_hour_funding_rate: float) -> float:
    """
    Calculate the funding impact in basis points relative to equity.
    
    Args:
        equity: Total account equity (USDC)
        position_notional: Absolute position notional value (USDC)
        next_hour_funding_rate: Hourly funding rate (e.g., 0.0001 for 1bps)
    
    Returns:
        Impact in basis points (e.g., 20.0 for 20bps impact)
    """
    if equity <= 0:
        return 0.0
    
    funding_fee = abs(position_notional) * abs(next_hour_funding_rate)
    return 10000.0 * (funding_fee / equity)


def in_funding_blackout(now_ts: float, blackout_min: int, impact_bps: float, 
                       impact_guard_bps: float, hi_blackout_min: int) -> Tuple[bool, str]:
    """
    Check if we're in a funding blackout window.
    
    Args:
        now_ts: Current timestamp
        blackout_min: Base blackout minutes before funding
        impact_bps: Calculated funding impact in bps
        impact_guard_bps: Threshold for extended blackout
        hi_blackout_min: Extended blackout minutes for high impact
    
    Returns:
        (is_blackout, reason)
    """
    # Calculate minutes until next funding (top of hour)
    utc_seconds = int(now_ts) % 3600
    minutes_to_funding = (3600 - utc_seconds) / 60.0
    
    # Determine effective blackout period
    effective_blackout = blackout_min
    reason = "funding_blackout"
    
    if impact_bps >= impact_guard_bps:
        effective_blackout = max(blackout_min, hi_blackout_min)
        reason = "funding_impact_guard"
    
    is_blackout = minutes_to_funding <= effective_blackout
    return is_blackout, reason if is_blackout else ""


def get_funding_schedule_info(now_ts: float) -> dict:
    """
    Get information about the current funding schedule.
    
    Returns:
        Dict with funding timing information
    """
    utc_seconds = int(now_ts) % 3600
    minutes_to_funding = (3600 - utc_seconds) / 60.0
    seconds_to_funding = 3600 - utc_seconds
    
    return {
        "minutes_to_funding": minutes_to_funding,
        "seconds_to_funding": seconds_to_funding,
        "utc_minute_of_hour": utc_seconds // 60,
        "is_funding_hour": True,  # Hyperliquid has hourly funding
    }
