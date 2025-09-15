"""
Main Sweep Engine

Core logic for perp→spot profit sweeping with safety guards and observability.
"""

import time
import random
import logging
from typing import Optional, Tuple, Dict, Any

from .metrics import (
    SWEEP_SUCCESS_TOTAL, SWEEP_FAIL_TOTAL, SWEEP_SKIPPED_TOTAL,
    SWEEP_LAST_AMOUNT, SWEEP_POSTBUF_BPS, SWEEP_EQUITY, SWEEP_WITHDRAWABLE,
    SWEEP_PENDING, SWEEP_CD_REMAINING, SWEEP_AMOUNT_HISTOGRAM, SWEEP_POST_BUFFER_HISTOGRAM
)
from .funding import equity_funding_impact_bps, in_funding_blackout
from .transfer import transfer_perp_to_spot, validate_transfer_params, format_transfer_amount
from .volatility import get_vol_multiplier


logger = logging.getLogger(__name__)


def parse_account_numbers(user_state: dict) -> Tuple[float, float]:
    """
    Extract withdrawable and equity from user_state.
    
    Args:
        user_state: Raw user state from Hyperliquid API
    
    Returns:
        (withdrawable, equity) tuple
    """
    withdrawable = float(user_state.get("withdrawable", 0) or 0.0)
    
    # Try multiple paths for equity/account value
    equity = 0.0
    if "marginSummary" in user_state:
        margin_summary = user_state.get("marginSummary", {})
        equity = float(margin_summary.get("accountValue", 0) or 0.0)
    elif "crossMarginSummary" in user_state:
        cross_summary = user_state.get("crossMarginSummary", {})
        equity = float(cross_summary.get("accountValue", 0) or 0.0)
    else:
        equity = float(user_state.get("accountValue", withdrawable) or withdrawable)
    
    return withdrawable, equity


def extract_liq_px(user_state: dict, coin: str = "XRP") -> Optional[float]:
    """
    Extract liquidation price for a specific coin.
    
    Args:
        user_state: Raw user state from API
        coin: Coin symbol to find
    
    Returns:
        Liquidation price or None if not found
    """
    for asset_position in user_state.get("assetPositions", []):
        position = asset_position.get("position", {})
        if position.get("coin") == coin:
            liq_px = position.get("liquidationPx")
            if liq_px is not None:
                return float(liq_px)
    return None


def liq_buffer_bps(mark_px: float, liq_px: Optional[float], is_long: bool) -> Optional[float]:
    """
    Calculate liquidation buffer in basis points.
    
    Args:
        mark_px: Current mark price
        liq_px: Liquidation price
        is_long: True for long position
    
    Returns:
        Buffer in basis points or None if cannot calculate
    """
    if mark_px <= 0 or liq_px is None or liq_px <= 0:
        return None
    
    if is_long:
        # For longs: distance = (liq_px - mark_px) / mark_px (negative when close to liquidation)
        distance = (liq_px - mark_px) / mark_px
    else:
        # For shorts: distance = (mark_px - liq_px) / mark_px (negative when close to liquidation)  
        distance = (mark_px - liq_px) / mark_px
    
    return abs(distance) * 10000.0  # Convert to basis points


def project_post_buffer_bps(pre_bps: float, equity: float, sweep_amt: float) -> float:
    """
    Project post-sweep liquidation buffer.
    
    Args:
        pre_bps: Pre-sweep buffer in bps
        equity: Current equity
        sweep_amt: Amount to sweep
    
    Returns:
        Projected post-sweep buffer in bps
    """
    if equity <= 0:
        return 0.0
    
    # Simple linear approximation: buffer scales with remaining equity
    remaining_equity_ratio = max(0.0, (equity - sweep_amt) / equity)
    return max(0.0, pre_bps * remaining_equity_ratio)


def cooldown_ok(now_ts: float, state, cfg) -> bool:
    """
    Check if cooldown period has passed.
    
    Args:
        now_ts: Current timestamp
        state: SweepState instance
        cfg: SweepCfg instance
    
    Returns:
        True if cooldown has passed
    """
    # Add jitter to prevent synchronized sweeps
    jitter = random.randint(-cfg.jitter_s, cfg.jitter_s)
    elapsed = now_ts - state.last_sweep_ts
    required_cooldown = cfg.cooldown_s + jitter
    
    remaining = max(0, required_cooldown - elapsed)
    SWEEP_CD_REMAINING.set(remaining)
    
    return remaining <= 0


def maybe_sweep_to_spot(
    exchange,
    state,
    cfg,
    *,
    user_state: dict,
    pos: Optional[dict],
    mark_px: Optional[float],
    vol_ratio: float,
    next_hour_funding_rate: float,
    position_notional: float,
    coin: str = "XRP",
    force_sweep: bool = False,
) -> Dict[str, Any]:
    """
    Main sweep decision and execution function.
    
    Args:
        exchange: Hyperliquid Exchange client
        state: SweepState instance  
        cfg: SweepCfg instance
        user_state: Raw user state from API
        pos: Position dict {size, is_long, ...} or None
        mark_px: Current mark price
        vol_ratio: Volatility ratio for adaptive behavior
        next_hour_funding_rate: Estimated next funding rate
        position_notional: Position notional value
        coin: Coin symbol
        force_sweep: Override safety checks
    
    Returns:
        Dict describing the decision and outcome
    """
    now = time.time()
    
    # Parse basic account info
    withdrawable, equity = parse_account_numbers(user_state)
    SWEEP_EQUITY.set(equity)
    SWEEP_WITHDRAWABLE.set(withdrawable)
    
    # Check if sweep is enabled
    if not cfg.enabled and not force_sweep:
        SWEEP_SKIPPED_TOTAL.labels(reason="disabled").inc()
        return {"action": "skip", "reason": "disabled"}
    
    # Calculate funding impact
    impact_bps = equity_funding_impact_bps(equity, position_notional, next_hour_funding_rate or 0.0)
    
    # Check funding blackout window
    if not force_sweep:
        blackout, blackout_reason = in_funding_blackout(
            now, cfg.funding_blackout_min, impact_bps, 
            cfg.funding_impact_guard_bps, cfg.funding_blackout_hi_min
        )
        if blackout:
            SWEEP_SKIPPED_TOTAL.labels(reason=blackout_reason).inc()
            return {"action": "skip", "reason": blackout_reason}
    
    # Check cooldown
    if not force_sweep and not cooldown_ok(now, state, cfg):
        SWEEP_SKIPPED_TOTAL.labels(reason="cooldown").inc()
        return {"action": "skip", "reason": "cooldown"}
    
    # Update accumulator
    vol_multiplier = get_vol_multiplier(vol_ratio, cfg.vol_high_threshold, cfg.vol_multiplier_high)
    if cfg.accumulator_enabled:
        pending = state.update_accumulator(
            withdrawable, equity, vol_multiplier,
            cfg.max_pending_cap_usd, cfg.max_pending_pct_equity
        )
    else:
        pending = 0.0
    
    SWEEP_PENDING.set(pending)
    
    # Calculate adaptive trigger
    trigger = max(cfg.min_sweep_usdc, round(cfg.equity_trigger_pct * equity, 2))
    
    # Determine if in position
    in_pos = bool(pos and float(pos.get("size", 0)) != 0)
    is_long = bool(pos.get("is_long", True)) if in_pos else True
    
    # Mode: flat (no position) or in-position
    if not in_pos or force_sweep:
        return _handle_flat_mode(
            exchange, state, cfg, withdrawable, equity, trigger, pending, force_sweep, now
        )
    else:
        return _handle_inpos_mode(
            exchange, state, cfg, user_state, withdrawable, equity, trigger, pending,
            mark_px, is_long, coin, force_sweep, now
        )


def _handle_flat_mode(exchange, state, cfg, withdrawable, equity, trigger, pending, force_sweep, now):
    """Handle sweep in flat (no position) mode"""
    headroom = max(0.0, withdrawable - cfg.min_reserve_usdc)
    
    if headroom <= 0:
        SWEEP_SKIPPED_TOTAL.labels(reason="no_headroom_flat").inc()
        return {"action": "skip", "reason": "no_headroom_flat"}
    
    # Calculate sweep amount
    cap_amt = withdrawable * cfg.flat_sweep_pct
    amount = min(headroom, cap_amt)
    
    # Allow flushing pending accumulator when flat
    if cfg.accumulator_enabled and pending >= trigger:
        amount = max(amount, min(pending, headroom))
    
    # Check minimum amount
    if not force_sweep and amount < trigger:
        SWEEP_SKIPPED_TOTAL.labels(reason="too_small").inc()
        return {"action": "skip", "reason": "too_small", "amount": amount, "trigger": trigger}
    
    if amount <= 0:
        SWEEP_SKIPPED_TOTAL.labels(reason="no_headroom_flat").inc()
        return {"action": "skip", "reason": "no_headroom_flat"}
    
    return _execute_sweep(exchange, state, cfg, amount, 0.0, "flat", now)


def _handle_inpos_mode(exchange, state, cfg, user_state, withdrawable, equity, trigger, pending,
                      mark_px, is_long, coin, force_sweep, now):
    """Handle sweep in in-position mode with safety checks"""
    
    # Extract liquidation price and calculate buffer
    liq_px = extract_liq_px(user_state, coin)
    if mark_px is None or mark_px <= 0:
        SWEEP_SKIPPED_TOTAL.labels(reason="no_mark_price").inc()
        return {"action": "skip", "reason": "no_mark_price"}
    
    pre_buffer = liq_buffer_bps(mark_px, liq_px, is_long)
    if pre_buffer is None or pre_buffer < cfg.inpos_min_buffer_bps:
        SWEEP_SKIPPED_TOTAL.labels(reason="low_pre_buffer").inc()
        return {
            "action": "skip", 
            "reason": "low_pre_buffer",
            "pre_buffer": pre_buffer,
            "required": cfg.inpos_min_buffer_bps
        }
    
    # Calculate sweep amount with position constraints
    headroom = max(0.0, withdrawable - cfg.min_reserve_usdc)
    if headroom <= 0:
        SWEEP_SKIPPED_TOTAL.labels(reason="no_headroom_inpos").inc()
        return {"action": "skip", "reason": "no_headroom_inpos"}
    
    cap_amt = withdrawable * cfg.inpos_max_sweep_pct
    base_amt = max(trigger, pending) if cfg.accumulator_enabled else trigger
    amount = min(headroom, cap_amt, base_amt)
    
    # Check minimum amount
    if not force_sweep and amount < trigger:
        SWEEP_SKIPPED_TOTAL.labels(reason="too_small").inc()
        return {"action": "skip", "reason": "too_small", "amount": amount, "trigger": trigger}
    
    # Project post-sweep buffer
    post_buffer = project_post_buffer_bps(pre_buffer, equity, amount)
    if not force_sweep and post_buffer < cfg.inpos_post_floor_bps:
        SWEEP_SKIPPED_TOTAL.labels(reason="post_floor_violation").inc()
        return {
            "action": "skip",
            "reason": "post_floor_violation", 
            "pre_buffer": pre_buffer,
            "post_buffer": post_buffer,
            "required_post": cfg.inpos_post_floor_bps
        }
    
    return _execute_sweep(exchange, state, cfg, amount, post_buffer, "inpos", now)


def _execute_sweep(exchange, state, cfg, amount, post_buffer, mode, now):
    """Execute the actual sweep transfer"""
    
    # Format amount and validate
    amount = format_transfer_amount(amount)
    validation_error = validate_transfer_params(amount, cfg.min_sweep_usdc)
    if validation_error:
        SWEEP_SKIPPED_TOTAL.labels(reason="validation_failed").inc()
        return {"action": "skip", "reason": "validation_failed", "error": validation_error}
    
    # De-duplication check
    cents = int(round(amount * 100))
    bucket = int(now) // 5
    if not state.de_dupe_check(cents, bucket, cfg.dedupe_window_s):
        SWEEP_SKIPPED_TOTAL.labels(reason="dedupe").inc()
        return {"action": "skip", "reason": "dedupe"}
    
    # Execute transfer
    logger.info(f"Executing sweep: ${amount:.2f} USDC perp→spot ({mode} mode)")
    
    result = transfer_perp_to_spot(exchange, amount, cfg.chain)
    
    if result["success"]:
        # Success metrics and state updates
        SWEEP_SUCCESS_TOTAL.inc()
        SWEEP_LAST_AMOUNT.set(amount)
        SWEEP_POSTBUF_BPS.set(post_buffer)
        SWEEP_AMOUNT_HISTOGRAM.observe(amount)
        SWEEP_POST_BUFFER_HISTOGRAM.observe(post_buffer)
        
        # Update state
        state.mark_sweep_complete(now, int(now * 1000))
        if cfg.accumulator_enabled:
            state.reset_accumulator(amount)
        state.save()
        
        logger.info(f"✅ Sweep successful: ${amount:.2f} USDC transferred")
        
        return {
            "action": "sweep",
            "amount": amount,
            "mode": mode,
            "post_buffer": post_buffer,
            "response": result["response"]
        }
    else:
        # Failure metrics
        SWEEP_FAIL_TOTAL.inc()
        
        logger.warning(f"❌ Sweep failed: {result['error']}")
        
        return {
            "action": "error",
            "amount": amount,
            "mode": mode,
            "error": result["error"],
            "response": result["response"]
        }
