"""
DEPRECATED: Legacy sweep_engine.py - Use new sweep/ package instead.

This file is kept for compatibility during migration.
All new development should use the modular sweep/ package.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import warnings
import logging

# Import from new package
try:
    from sweep import SweepCfg, SweepState, maybe_sweep_to_spot
    from sweep import (
        SWEEP_SUCCESS_TOTAL, SWEEP_FAIL_TOTAL, SWEEP_SKIPPED_TOTAL,
        SWEEP_LAST_AMOUNT, SWEEP_POSTBUF_BPS, SWEEP_EQUITY, 
        SWEEP_WITHDRAWABLE, SWEEP_PENDING, SWEEP_CD_REMAINING
    )
    _NEW_PACKAGE_AVAILABLE = True
    
    # Legacy compatibility class
    class SweepEngine:
        """Legacy compatibility wrapper for new sweep package"""
        
        def __init__(self, cfg=None, state=None):
            warnings.warn(
                "SweepEngine class is deprecated. Use sweep.maybe_sweep_to_spot function directly.",
                DeprecationWarning,
                stacklevel=2
            )
            self.cfg = cfg or SweepCfg()
            self.state = state or SweepState(self.cfg.accumulator_file)
            self.state.load()
        
        def maybe_sweep_to_spot(self, exchange, user_state, pos, mark_px, vol_score, 
                               est_funding_rate_hourly, force=False, pre_buffer_bps=None):
            """Legacy method wrapper"""
            # Convert legacy parameters to new format
            position_notional = 0.0
            if pos:
                size = safe_float(pos.get("size", 0))
                entry_px = safe_float(pos.get("entry_px", mark_px or 0))
                position_notional = abs(size * entry_px)
            
            return maybe_sweep_to_spot(
                exchange=exchange,
                state=self.state,
                cfg=self.cfg,
                user_state=user_state,
                pos=pos,
                mark_px=mark_px,
                vol_ratio=vol_score,
                next_hour_funding_rate=est_funding_rate_hourly,
                position_notional=position_notional,
                force_sweep=force
            )
    
except ImportError:
    _NEW_PACKAGE_AVAILABLE = False
    warnings.warn("New sweep package not available, sweep functionality disabled")
    
    # Stub implementations to prevent import errors
    class SweepCfg:
        def __init__(self):
            self.enabled = False
    
    class SweepState:
        def __init__(self, path):
            pass
        def load(self):
            pass
    
    class SweepEngine:
        def __init__(self, cfg=None, state=None):
            logging.warning("Sweep engine disabled - new package not available")
        
        def maybe_sweep_to_spot(self, *args, **kwargs):
            return {"action": "skip", "reason": "disabled"}
    
    def maybe_sweep_to_spot(*args, **kwargs):
        return {"action": "skip", "reason": "disabled"}

# Re-export for compatibility
__all__ = ["SweepCfg", "SweepState", "SweepEngine", "maybe_sweep_to_spot"]

if _NEW_PACKAGE_AVAILABLE:
    __all__.extend([
        "SWEEP_SUCCESS_TOTAL", "SWEEP_FAIL_TOTAL", "SWEEP_SKIPPED_TOTAL",
        "SWEEP_LAST_AMOUNT", "SWEEP_POSTBUF_BPS", "SWEEP_EQUITY",
        "SWEEP_WITHDRAWABLE", "SWEEP_PENDING", "SWEEP_CD_REMAINING"
    ])
