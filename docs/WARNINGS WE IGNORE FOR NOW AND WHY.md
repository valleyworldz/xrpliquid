# Warnings We Ignore For Now — And Why

This bot intentionally treats several advanced/optional modules as non-critical. When they are not installed, the bot logs a benign warning and switches to safe fallbacks. Below lists each warning, whether we need the module, pros/cons, and how to enable it later.

## 1) technical_indicators.indicators (optional TA helpers)
- What: Extra vectorized technical indicators (RSI/EMA/MACD/ATR) for speed/consistency.
- Why missing: Optional package not installed.
- Need it? No. Fallback uses pandas/NumPy; P&L unchanged.
- Pros: faster backtests on very large datasets; more TA variants.
- Cons: extra dependency to maintain.

## 2) enhanced_api.components (optional API wrappers)
- What: Higher-level API helpers (advanced rate-limiting, batching/TWAP, retries).
- Why missing: Optional module not installed.
- Need it? No. Fallback clients have safe backoff and batching where needed.
- Pros: better throughput/latency at scale.
- Cons: more complexity; potential drift with SDK.

## 3) structlog (structured logging)
- What: JSON structured logs for observability.
- Why missing: Not installed.
- Need it? No. Standard logging is sufficient.
- Pros: machine-readable logs; easier correlation.
- Cons: extra dependency; minor overhead.

## 4) FastAPI / Uvicorn (metrics/health server)
- What: Optional /healthz and /metrics endpoints.
- Why missing: Not installed.
- Need it? No. Useful for prod/Kubernetes only.
- Pros: live health and Prometheus metrics.
- Cons: overhead for local use.

## 5) aiohttp (async HTTP client)
- What: Faster, non-blocking downloads.
- Why missing: Not installed.
- Need it? No. requests fallback is fine.
- Pros: faster fetch; concurrency.
- Cons: extra dependency.

## 6) ML backends: PyTorch / scikit-learn
- What: Model-based probabilities for ensemble and Kelly.
- Why missing: If not installed, rules-only analyzer is used.
- Need it? Optional. Bot is safe without ML.
- Pros: higher edge when trained/validated.
- Cons: heavier deps; needs data/validation.

## 7) metrics module (Prometheus shim)
- What: Helper to export metrics.
- Why missing: Not installed or omitted.
- Need it? No. Only for production monitoring.
- Pros: easy dashboard/alerts.
- Cons: extra moving part.

---

### Why it’s safe to ignore
- Robust fallbacks for every optional module (pandas/NumPy/logging/requests).
- No impact on P&L correctness: core risk/TP-SL/sizing logic is unchanged.
- Enable only if you need faster downloads, structured logs, or ML sizing.

### Suggested enablement order (if needed)
1) aiohttp  2) scikit-learn  3) structlog  4) FastAPI/uvicorn  5) PyTorch

### Quick install bundle
```bash
pip install aiohttp scikit-learn structlog fastapi uvicorn
# Optional CPU-only Torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

These warnings are informational. The bot is production-capable with fallbacks and can be upgraded incrementally as needs grow.
