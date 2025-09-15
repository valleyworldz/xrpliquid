from __future__ import annotations
import threading
import logging

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PEAK = 0.0
_LAST_WARNED_STEP = -1  # 0,1,2... for each 50 bps deeper drawdown


def _dd_bps(peak: float, av: float) -> float:
    if peak <= 0:
        return 0.0
    return (peak - av) / peak * 10_000.0  # basis points


def update_peak_and_maybe_warn(account_value: float) -> tuple[float, float]:
    """
    Track session peak and WARN every additional ~50 bps of drawdown.
    Returns (peak_value, drawdown_bps).
    """
    global _PEAK, _LAST_WARNED_STEP
    if account_value <= 0:
        return (_PEAK, 0.0)

    with _LOCK:
        if account_value > _PEAK:
            _PEAK = account_value
            _LAST_WARNED_STEP = -1
            return (_PEAK, 0.0)

        dd = _dd_bps(_PEAK, account_value)
        step = int(dd // 50.0)
        if step > _LAST_WARNED_STEP:
            _LAST_WARNED_STEP = step
            log.warning("📉 drawdown %.0f bps from peak (peak=%.4f, av=%.4f)", dd, _PEAK, account_value)
        return (_PEAK, dd)


