"""
Sweep Engine State Management

Persistent state for cooldowns, accumulator, and de-duplication.
"""

import json
import time
import os
import threading
from typing import Dict, Any


class SweepState:
    """Thread-safe persistent state for sweep engine"""
    
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.last_sweep_ts = 0.0
        self.pending_accum = 0.0
        self.last_withdrawable = 0.0
        self.last_nonce = 0
        self.recent_keys: Dict[str, float] = {}  # de-dup map: key -> ts

    def load(self) -> None:
        """Load state from file"""
        with self._lock:
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.last_sweep_ts = float(data.get("last_sweep_ts", 0.0))
                    self.pending_accum = float(data.get("pending_accum", 0.0))
                    self.last_withdrawable = float(data.get("last_withdrawable", 0.0))
                    self.last_nonce = int(data.get("last_nonce", 0))
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                # Reset to defaults on any error
                pass

    def save(self) -> None:
        """Atomically save state to file"""
        with self._lock:
            try:
                data = {
                    "last_sweep_ts": self.last_sweep_ts,
                    "pending_accum": self.pending_accum,
                    "last_withdrawable": self.last_withdrawable,
                    "last_nonce": self.last_nonce,
                }
                # Atomic write using temp file
                tmp_path = self.path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, separators=(",", ":"))
                os.replace(tmp_path, self.path)
            except Exception:
                # Don't fail the sweep on state save errors
                pass

    def cooldown_remaining(self, cooldown_s: int, jitter_s: int) -> float:
        """Calculate remaining cooldown time in seconds"""
        now = time.time()
        # Note: jitter is applied at call site; here just remaining baseline
        elapsed = now - self.last_sweep_ts
        return max(0.0, cooldown_s - elapsed)

    def de_dupe_check(self, amount_cents: int, nonce_bucket_5s: int, window_s: int = 5) -> bool:
        """
        Check if this sweep is a duplicate within the window.
        Returns True if NOT a duplicate (safe to proceed).
        """
        now = time.time()
        key = f"{amount_cents}:{nonce_bucket_5s}"
        
        # Clean expired entries
        expired_keys = [k for k, ts in self.recent_keys.items() if now - ts > window_s]
        for k in expired_keys:
            self.recent_keys.pop(k, None)
        
        # Check if duplicate
        if key in self.recent_keys:
            return False  # Duplicate found
        
        # Mark this key as seen
        self.recent_keys[key] = now
        return True  # Not a duplicate

    def update_accumulator(self, withdrawable: float, equity: float, vol_multiplier: float, 
                          max_cap_usd: float, max_pct_equity: float) -> float:
        """
        Update the accumulator with new withdrawable amount.
        Returns the current pending amount.
        """
        with self._lock:
            # Calculate delta from last withdrawable
            delta = max(0.0, withdrawable - self.last_withdrawable)
            
            # Calculate cap with volatility multiplier
            cap = min(max_cap_usd, max_pct_equity * max(equity, 0.0)) * vol_multiplier
            
            # Update pending (capped)
            self.pending_accum = min(self.pending_accum + delta, cap)
            self.last_withdrawable = withdrawable
            
            return self.pending_accum

    def reset_accumulator(self, amount_swept: float) -> None:
        """Reset accumulator after a successful sweep"""
        with self._lock:
            self.pending_accum = max(0.0, self.pending_accum - amount_swept)

    def mark_sweep_complete(self, timestamp: float, nonce: int) -> None:
        """Mark a sweep as completed"""
        with self._lock:
            self.last_sweep_ts = timestamp
            self.last_nonce = nonce
