#!/usr/bin/env python3
"""
Real-Data Optimizer
Tunes profile parameters (leverage, risk pct, stop type, mode) using ONLY real Hyperliquid data.
Depends on working_real_backtester.py (real-data backtester already created in this workspace).
"""

import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple
import working_real_backtester as wrb


@dataclass
class Candidate:
    leverage: float
    position_risk_pct: float
    stop_loss_type: str
    trading_mode: str


def score_results(results: List[wrb.RealBacktestResult]) -> float:
    if not results:
        return 0.0
    valid = [r for r in results if r.total_trades > 0]
    if not valid:
        return 0.0
    avg_return = sum(r.total_return_percent for r in valid) / len(valid)
    avg_win_rate = sum(r.win_rate for r in valid) / len(valid)
    avg_sharpe = sum(r.sharpe_ratio for r in valid) / len(valid)
    avg_drawdown = sum(r.max_drawdown_percent for r in valid) / len(valid)
    total_trades = sum(r.total_trades for r in valid)

    # Composite score (0-100) weighted toward risk control and consistency
    return_score = min(100, max(0, (avg_return + 2) * 20))
    consistency_score = max(0, min(100, avg_win_rate))
    risk_score = max(0, min(100, 100 - avg_drawdown * 4))
    sharpe_score = min(100, max(0, (avg_sharpe + 0.5) * 40))
    activity_score = min(100, total_trades * 3)

    overall = (
        return_score * 0.30 +
        consistency_score * 0.25 +
        risk_score * 0.25 +
        sharpe_score * 0.15 +
        activity_score * 0.05
    )
    return overall


def run_optimizer():
    print('🧠 REAL-DATA OPTIMIZER')
    print('=' * 60)
    tester = wrb.RealStrategyTester()
    # Broaden symbol coverage for more robust scoring
    symbols = ['BTC', 'ETH', 'SOL', 'ATOM', 'AVAX', 'BNB']

    grids: Dict[str, List[Candidate]] = {
        'day_trader': [
            Candidate(l, r, s, 'scalping')
            for l in [2, 3, 4, 5]
            for r in [0.5, 1.0, 1.5, 2.0]
            for s in ['tight', 'normal']
        ],
        'swing_trader': [
            Candidate(l, r, s, 'swing')
            for l in [3, 4, 5, 6]
            for r in [1.0, 2.0, 3.0]
            for s in ['normal', 'tight']
        ],
        'hodl_king': [
            Candidate(l, r, s, 'position')
            for l in [1, 2, 3]
            for r in [0.5, 1.0, 1.5]
            for s in ['wide', 'normal']
        ],
        'degen_mode': [
            Candidate(l, r, 'tight', 'scalping')
            for l in [10, 12, 15, 20]
            for r in [3.0, 5.0, 8.0]
        ],
    }

    best: Dict[str, Tuple[float, Candidate, Dict[str, wrb.RealBacktestResult]]] = {}

    for key, grid in grids.items():
        print(f"\n🔬 Optimizing {wrb.TRADING_PROFILES[key]['name']} ({len(grid)} candidates)...")
        best_score = -1.0
        best_cand = None
        best_results: Dict[str, wrb.RealBacktestResult] = {}

        # Coarse pass on 24h window for speed
        for cand in grid:
            cfg = wrb.StartupConfig(
                leverage=cand.leverage,
                risk_profile='balanced',
                trading_mode=cand.trading_mode,
                position_risk_pct=cand.position_risk_pct,
                stop_loss_type=cand.stop_loss_type,
            )
            per_symbol: Dict[str, wrb.RealBacktestResult] = {}
            for sym in symbols:
                res = tester.test_strategy_with_hours(sym, cfg, 24)
                per_symbol[sym] = res
            sc = score_results(list(per_symbol.values()))
            if sc > best_score:
                best_score, best_cand, best_results = sc, cand, per_symbol

        # Fine pass: refine around the best candidate
        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        if best_cand is not None:
            fine_grid: List[Candidate] = []
            levs = sorted({clamp(best_cand.leverage + d, 1, 20) for d in (-1, 0, 1)})
            risks = sorted({clamp(best_cand.position_risk_pct + d, 0.5, 8.0) for d in (-0.5, 0, 0.5)})
            stops = ['tight', 'normal', 'wide'] if best_cand.stop_loss_type in ('tight','normal') else ['normal','wide','tight']
            modes = [best_cand.trading_mode]
            for l in levs:
                for r in risks:
                    for s in stops:
                        for m in modes:
                            fine_grid.append(Candidate(l, r, s, m))

            # Fine pass on 72h window (adds robustness)
            for cand in fine_grid:
                cfg = wrb.StartupConfig(
                    leverage=cand.leverage,
                    risk_profile='balanced',
                    trading_mode=cand.trading_mode,
                    position_risk_pct=cand.position_risk_pct,
                    stop_loss_type=cand.stop_loss_type,
                )
                per_symbol: Dict[str, wrb.RealBacktestResult] = {}
                for sym in symbols:
                    res = tester.test_strategy_with_hours(sym, cfg, 72)
                    per_symbol[sym] = res
                sc = score_results(list(per_symbol.values()))
                if sc > best_score:
                    best_score, best_cand, best_results = sc, cand, per_symbol

        # Validation pass on 168h window for the best candidate
        if best_cand is not None:
            cfg = wrb.StartupConfig(
                leverage=best_cand.leverage,
                risk_profile='balanced',
                trading_mode=best_cand.trading_mode,
                position_risk_pct=best_cand.position_risk_pct,
                stop_loss_type=best_cand.stop_loss_type,
            )
            per_symbol: Dict[str, wrb.RealBacktestResult] = {}
            for sym in symbols:
                res = tester.test_strategy_with_hours(sym, cfg, 168)
                per_symbol[sym] = res
            sc = score_results(list(per_symbol.values()))
            if sc > best_score:
                best_score, best_cand, best_results = sc, cand, per_symbol

        best[key] = (best_score, best_cand, best_results)
        print(f"   ✅ Best score: {best_score:.1f} with {best_cand}")

    print('\n🏁 OPTIMIZATION SUMMARY (REAL DATA)')
    print('=' * 60)
    for key, (score, cand, results) in best.items():
        name = wrb.TRADING_PROFILES[key]['name']
        valids = [r for r in results.values() if r.total_trades > 0]
        avg_ret = (sum(r.total_return_percent for r in valids) / len(valids)) if valids else 0.0
        print(
            f"{name}: {score:.1f}/100 | leverage={cand.leverage}x, risk={cand.position_risk_pct}%, "
            f"stop={cand.stop_loss_type}, mode={cand.trading_mode} | avgRet={avg_ret:+.2f}% across "
            f"{len(valids)}/{len(results)} symbols"
        )

    print('\n📋 SUGGESTED UPDATED PROFILES:')
    for key, (score, cand, _) in best.items():
        print(
            f"{key}: leverage={cand.leverage}, risk={cand.position_risk_pct}, "
            f"stop='{cand.stop_loss_type}', mode='{cand.trading_mode}' | score={score:.1f}"
        )


if __name__ == '__main__':
    run_optimizer()


