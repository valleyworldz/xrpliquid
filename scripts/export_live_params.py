#!/usr/bin/env python3
"""
Export optimized live parameters from real_backtest_summary.json
into optimized_params_live.json for use in live trading config.
"""

import json
import os
import sys
from typing import Any, Dict


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    base = os.getcwd()
    src = os.path.join(base, "real_backtest_summary.json")
    if not os.path.exists(src):
        print("real_backtest_summary.json not found. Run working_real_backtester.py first.")
        return 1

    data = load_summary(src)
    profiles = data.get("profiles", {})
    hours = int(data.get("hours", 168) or 168)
    ts = int(data.get("timestamp", 0) or 0)

    out: Dict[str, Any] = {
        "symbol": "XRP",
        "hours": hours,
        "timestamp": ts,
        "profiles": {},
        "execution": {
            "funding_max_long": 0.0005,
            "funding_max_short": 0.0005,
            "fee_buffer": 0.002,
            "impact_buffer": 0.001
        }
    }

    for key, info in profiles.items():
        per_symbol = info.get("per_symbol", {})
        xrp = per_symbol.get("XRP", {})
        sel = xrp.get("selected_params", {})
        params = sel.get("params", {})
        stop = sel.get("stop", None)
        tp_mult = sel.get("tp_mult", None)
        out["profiles"][key] = {
            "params": params,
            "stop": stop,
            "tp_mult": tp_mult,
            "meta": {
                "return_pct": info.get("summary", {}).get("return", 0.0),
                "win_rate_pct": info.get("summary", {}).get("win_rate", 0.0),
                "sharpe": info.get("summary", {}).get("sharpe", 0.0),
                "max_dd_pct": info.get("summary", {}).get("drawdown", 0.0),
                "overall_score": info.get("summary", {}).get("overall_score", 0.0)
            }
        }

    dst = os.path.join(base, "optimized_params_live.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


