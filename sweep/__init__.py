"""
Perp→Spot Profit Sweeping Engine v1.0

Automatically moves realized USDC profits from Perps to Spot on Hyperliquid
with safety guards, cooldowns, and observability.
"""

from .config import SweepCfg
from .state import SweepState
from .engine import maybe_sweep_to_spot
from .metrics import (
    SWEEP_SUCCESS_TOTAL,
    SWEEP_FAIL_TOTAL,
    SWEEP_SKIPPED_TOTAL,
    SWEEP_LAST_AMOUNT,
    SWEEP_POSTBUF_BPS,
    SWEEP_EQUITY,
    SWEEP_WITHDRAWABLE,
    SWEEP_PENDING,
    SWEEP_CD_REMAINING,
)

__version__ = "1.0.0"
__all__ = [
    "SweepCfg",
    "SweepState", 
    "maybe_sweep_to_spot",
    "SWEEP_SUCCESS_TOTAL",
    "SWEEP_FAIL_TOTAL",
    "SWEEP_SKIPPED_TOTAL",
    "SWEEP_LAST_AMOUNT",
    "SWEEP_POSTBUF_BPS",
    "SWEEP_EQUITY",
    "SWEEP_WITHDRAWABLE",
    "SWEEP_PENDING",
    "SWEEP_CD_REMAINING",
]
