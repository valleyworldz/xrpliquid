from __future__ import annotations
import json
import os
import time
import threading
import logging

log = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_STATE = {"until_ts": 0.0}
_STATE_PATH = os.environ.get("COOLDOWN_STATE_PATH", "cooldown_state.json")


def load_cooldown_state() -> None:
    """Load a persisted cooldown window, if any."""
    with _STATE_LOCK:
        try:
            if os.path.exists(_STATE_PATH):
                with open(_STATE_PATH, "r") as f:
                    data = json.load(f)
                _STATE["until_ts"] = float(data.get("until_ts", 0.0))
                log.info("↩️ cooldown restored (until_ts=%s)", _STATE["until_ts"])
        except Exception as e:
            log.warning("cooldown_state: load failed: %s", e)


def set_cooldown(seconds: int = 30) -> None:
    """Arm a cooldown for N seconds from now and persist it."""
    with _STATE_LOCK:
        _STATE["until_ts"] = time.time() + max(0, int(seconds))
        try:
            with open(_STATE_PATH, "w") as f:
                json.dump(_STATE, f)
        except Exception as e:
            log.debug("cooldown_state: persist skipped (%s)", e)
        log.info("🧊 cooldown armed for %ss (until_ts=%s)", seconds, _STATE["until_ts"])


def cooldown_active() -> bool:
    """Return True if the cooldown window is still active."""
    with _STATE_LOCK:
        return time.time() < _STATE["until_ts"]


