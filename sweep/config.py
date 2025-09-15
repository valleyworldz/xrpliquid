"""
Sweep Engine Configuration

Environment-driven configuration with safe defaults for perp→spot sweeping.
All thresholds and timeouts are tunable via environment variables.
"""

from dataclasses import dataclass
import os


@dataclass
class SweepCfg:
    """Configuration for the sweep engine with ENV overrides"""
    
    enabled: bool = os.getenv("SWEEP_ENABLED", "false").lower() == "true"

    # Thresholds
    min_reserve_usdc: float = float(os.getenv("SWEEP_MIN_RESERVE_USDC", "150"))
    min_sweep_usdc: float = float(os.getenv("SWEEP_MIN_SWEEP_USDC", "20"))
    equity_trigger_pct: float = float(os.getenv("SWEEP_EQUITY_TRIGGER_PCT", "0.05"))

    # Modes
    flat_sweep_pct: float = float(os.getenv("SWEEP_FLAT_SWEEP_PCT", "0.95"))
    inpos_max_sweep_pct: float = float(os.getenv("SWEEP_INPOS_MAX_SWEEP_PCT", "0.33"))
    inpos_min_buffer_bps: float = float(os.getenv("SWEEP_INPOS_MIN_BUFFER_BPS", "3000"))
    inpos_post_floor_bps: float = float(os.getenv("SWEEP_INPOS_POST_FLOOR_BPS", "2000"))

    # Freshness & cadence
    max_staleness_s: int = int(os.getenv("SWEEP_MAX_STALENESS_S", "60"))
    cooldown_s: int = int(os.getenv("SWEEP_COOLDOWN_S", "1800"))  # 30m
    jitter_s: int = int(os.getenv("SWEEP_JITTER_S", "120"))     # ±2m

    # Funding blackout
    funding_window_aware: bool = os.getenv("SWEEP_FUNDING_AWARE", "true").lower() == "true"
    funding_blackout_min: int = int(os.getenv("SWEEP_FUNDING_BLACKOUT_MIN", "10"))
    funding_impact_guard_bps: float = float(os.getenv("SWEEP_FUNDING_IMPACT_GUARD_BPS", "20"))
    funding_blackout_hi_min: int = int(os.getenv("SWEEP_FUNDING_BLACKOUT_HI_MIN", "15"))

    # Accumulator
    accumulator_enabled: bool = os.getenv("SWEEP_ACCUM_ENABLED", "true").lower() == "true"
    accumulator_file: str = os.getenv("SWEEP_ACCUM_FILE", "sweep.state.json")
    max_pending_cap_usd: float = float(os.getenv("SWEEP_MAX_PENDING_CAP_USD", "200"))
    max_pending_pct_equity: float = float(os.getenv("SWEEP_MAX_PENDING_PCT_EQUITY", "0.05"))

    # Volatility
    vol_periods: int = int(os.getenv("SWEEP_VOL_PERIODS", "48"))      # e.g., 48 x 30m = 1 day
    vol_high_threshold: float = float(os.getenv("SWEEP_VOL_HIGH", "2.0"))
    vol_multiplier_high: float = float(os.getenv("SWEEP_VOL_MULTIPLIER_HIGH", "1.5"))
    
    # De-dupe
    dedupe_window_s: int = int(os.getenv("SWEEP_DEDUPE_WINDOW_S", "5"))

    # Chain
    chain: str = os.getenv("HL_CHAIN", "Mainnet")  # or "Testnet"
