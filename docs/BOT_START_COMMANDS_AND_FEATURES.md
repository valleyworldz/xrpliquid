# Bot Start Commands and Features

This guide shows practical start commands and explains every major flag so users can quickly run the bot in different modes (smoke test, backtest, micro-live, bull/bear modes) and understand each feature.

---

## Quick Start Commands

### 1) Smoke test (sanity checks)
```bash
python newbotcode.py --smoke-test
```
- Verifies DNS, HTTP, API probe, and market helpers. No trading or data writes.

### 2) Backtest (daily bars)
```bash
python newbotcode.py --backtest xrp_2025.csv --initial-capital=100
```
- Runs a simple daily-bar backtest using the provided CSV.
- Summary metrics logged at the end (Sharpe, MaxDD, WinRate, FinalEquity).

### 3) Backtest (hourly micro, $30, Kelly cap, shorter DD lock)
```bash
python newbotcode.py \
  --backtest xrp_2025.csv --initial-capital=30 --hourly \
  --kelly-cap 0.25 --drawdown-lock-sec 1200 \
  --ta-conf-min 0.40 --pa-conf-min 0.40 --fee-threshold-multi 3.0
```
- Hourly simulation more closely matches live intraday behavior.
- Kelly sizing capped at 25%; DD lock set to 20 minutes.
- Ensemble gates tougher in chop; fee+funding threshold stricter for micro.

### 4) Backtest with auto-download (hourly)
```bash
python newbotcode.py \
  --backtest xrp_2025.csv --initial-capital=30 --hourly --download \
  --start-date 2023-01-01 --end-date 2025-01-01
```
- Auto-downloads data if the CSV does not exist.

### 5) Live (micro, bullish bias)
```bash
python newbotcode.py \
  --micro-live --bull-long-only \
  --fee-threshold-multi 3.0 --kelly-cap 0.18 --drawdown-lock-sec 1200
```
- Micro-optimizations on, long-only bias in bull regimes, conservative Kelly cap.

### 6) Live (micro, chop/range)
```bash
python newbotcode.py \
  --micro-live --ensemble-conflict-gate 0.40 \
  --fee-threshold-multi 3.0 --kelly-cap 0.15 --drawdown-lock-sec 1200
```
- Tighter ensemble gate reduces churn; neutral regime (no forced long/short only).

### 7) Live (micro, bearish bias)
```bash
python newbotcode.py \
  --micro-live --bear-short-only \
  --fee-threshold-multi 3.0 --kelly-cap 0.12 --drawdown-lock-sec 1500
```
- Short-only bias in bear regimes, smaller Kelly cap, slightly longer DD lock.

### 8) Sandbox TP/SL validation (one-off)
```bash
python newbotcode.py --sandbox-tpsl
```
- Places a tiny TP/SL pair to validate native trigger paths and fees.

### 9) Clear active triggers
```bash
python newbotcode.py --clear-triggers
```
- One-off cleanup; cancels active triggers before starting.

---

## Flag Reference (most useful)

- `--smoke-test`
  - Run connectivity and helper checks, then exit.

- `--backtest <CSV>`
  - Path to CSV with OHLCV columns. Enables offline simulation.

- `--initial-capital <float>`
  - Sets starting equity for backtests.

- `--hourly`
  - Uses hourly data for intraday realism (requires matching hourly CSV or `--download`).

- `--download`
  - Auto-downloads data if the specified backtest CSV is missing.

- `--start-date YYYY-MM-DD` / `--end-date YYYY-MM-DD`
  - Date range for downloads.

- `--use-live-equity`
  - In backtest, seeds initial capital from live withdrawable/free collateral if available.

- `--micro-live`
  - Enables micro account optimizations in live mode:
    - Maker preference, fee+funding threshold bump, tighter SL/TP defaults (SL≈1.6×ATR, TP≈3.2×ATR), stricter ensemble gate (≈0.40), bull-long-only by default unless `--bear-short-only` is set.

- `--bull-long-only` / `--bear-short-only`
  - Force long-only in bull regimes or short-only in bear regimes.

- `--fee-threshold-multi <float>`
  - After-fee economics guard multiplier for entry. Micro suggests 2.5–3.0.

- `--atr-sl-multi <float>` / `--atr-tp-multi <float>`
  - Override ATR multipliers for SL/TP in live mode.

- `--ta-conf-min <float>` / `--pa-conf-min <float>`
  - Minimum confidence thresholds for TA vs PatternAnalyzer in backtest ensemble logic.

- `--ensemble-conflict-gate <float>`
  - If TA and PA disagree and both are below this, HOLD (blocks churn in chop).

- `--kelly-cap <float>`
  - Caps Kelly fraction for sizing (safety in micro). Typical: 0.10–0.25.

- `--drawdown-lock-sec <int>`
  - Fixed lock duration after DD breach; overrides adaptive default when provided.

- `--suppress-cancels`
  - Only for specific exchange bug scenarios. Not recommended by default.

- `--sandbox-tpsl`
  - One-off TP/SL placement for native trigger validation.

- `--clear-triggers`
  - Cancels active triggers on startup.

---

## Drawdown Locking (adaptive by default)
- Default base lock: 1200s (20 min).
- Adaptive tiers by DD depth:
  - DD < 5% → 900s (15 min)
  - 5% ≤ DD < 10% → 1500s (25 min)
  - DD ≥ 10% → 3000s (50 min)
- Early unlock: if DD recovers below 50% of the threshold after ≥300s elapsed.
- Override anytime with `--drawdown-lock-sec`.

---

## Sizing and Risk (micro-focused)
- Kelly-capped sizing (default cap 0.25) adjusts to estimated edge; conservative by design for $30–$100 wallets.
- ATR-scaled sizing reduces size in high-volatility conditions.
- Drawdown throttle halves size above configured DD thresholds.

---

## Fees and Execution
- Dynamic maker/taker fees are applied by tier from exchange metadata.
- Fee-aware TP/SL alignment preserves intended RR after taker fees.
- Maker-first preference; fallback to taker only when needed.

---

## Tips
- Start with `--smoke-test` before any live run.
- For micro accounts, keep `--kelly-cap` between 0.10 and 0.20 and use a 1200s DD lock.
- Use `--hourly` backtests for more realistic sizing/TP/SL behavior.

---

## Examples Cheat Sheet
- Conservative micro live (neutral):
```bash
python newbotcode.py --micro-live --ensemble-conflict-gate 0.40 --fee-threshold-multi 3.0 --kelly-cap 0.15 --drawdown-lock-sec 1200
```
- Aggressive bull bias:
```bash
python newbotcode.py --micro-live --bull-long-only --fee-threshold-multi 3.0 --kelly-cap 0.20 --drawdown-lock-sec 900
```
- Defensive bear bias:
```bash
python newbotcode.py --micro-live --bear-short-only --fee-threshold-multi 3.0 --kelly-cap 0.12 --drawdown-lock-sec 1500
```

---

If you need additional examples or preset scripts, we can add starter `.bat` or shell scripts to wrap these commands.
