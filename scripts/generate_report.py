#!/usr/bin/env python3
"""
Generate summary CSV and optional plots from real_backtest_summary.json.

Outputs:
- real_backtest_report.csv
- real_backtest_report.png (if matplotlib is available)
"""

import json
import math
import os
import sys
from typing import Any, Dict


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_cagr(return_pct: float, hours: int) -> float:
    """Approximate annualized CAGR given a total return percentage over 'hours'."""
    try:
        r = return_pct / 100.0
        if hours <= 0:
            return 0.0
        annual_hours = 24 * 365
        cagr = (1.0 + r) ** (annual_hours / float(hours)) - 1.0
        # Guard against extreme small-hours instability
        if math.isfinite(cagr):
            return cagr * 100.0
        return 0.0
    except Exception:
        return 0.0


def write_csv(rows, out_path: str) -> None:
    import csv
    headers = [
        "profile",
        "overall_score",
        "return_pct",
        "cagr_pct",
        "win_rate_pct",
        "sharpe",
        "max_drawdown_pct",
        "total_trades",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def maybe_plot(rows, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        profiles = [r["profile"] for r in rows]
        returns = [r["return_pct"] for r in rows]
        scores = [r["overall_score"] for r in rows]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        axes[0].bar(profiles, returns, color="#4E79A7")
        axes[0].set_title("Return (%) by Profile")
        axes[0].set_ylabel("Return %")
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(profiles, scores, color="#F28E2B")
        axes[1].set_title("Overall Score by Profile")
        axes[1].set_ylabel("Score (0-100)")
        axes[1].grid(axis="y", alpha=0.3)

        for ax in axes:
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
                tick.set_ha("right")

        fig.suptitle("Real-Data Backtest Summary", fontsize=14)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        # Plotting is optional; skip on any failure
        pass


def main() -> int:
    summary_path = os.path.join(os.getcwd(), "real_backtest_summary.json")
    if not os.path.exists(summary_path):
        print("real_backtest_summary.json not found. Run working_real_backtester.py first.")
        return 1

    data = load_summary(summary_path)
    hours = int(data.get("hours", 168) or 168)
    profiles = data.get("profiles", {})

    rows = []
    for key, info in profiles.items():
        s = info.get("summary", {})
        profile_name = s.get("name", key)
        ret = float(s.get("return", 0.0))
        cagr = compute_cagr(ret, hours)
        row = {
            "profile": profile_name,
            "overall_score": round(float(s.get("overall_score", 0.0)), 3),
            "return_pct": round(ret, 4),
            "cagr_pct": round(cagr, 4),
            "win_rate_pct": round(float(s.get("win_rate", 0.0)), 2),
            "sharpe": round(float(s.get("sharpe", 0.0)), 3),
            "max_drawdown_pct": round(float(s.get("drawdown", 0.0)), 4),
            "total_trades": int(s.get("total_trades", 0)),
        }
        rows.append(row)

    csv_path = os.path.join(os.getcwd(), "real_backtest_report.csv")
    write_csv(rows, csv_path)
    png_path = os.path.join(os.getcwd(), "real_backtest_report.png")
    maybe_plot(rows, png_path)

    print(f"Wrote {csv_path}")
    if os.path.exists(png_path):
        print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


