#!/usr/bin/env python3
"""
Score-10 Live Trading Optimizations
==================================
Applies backtested optimizations to live trading profiles
Based on champion Degen Mode strategy: 71.5/100 score, 60% win rate, +2.5% returns
"""

import json
import os
from typing import Dict, Any

# Load optimized parameters from backtesting
def load_optimized_params() -> Dict[str, Any]:
    """Load the optimized parameters from backtesting results"""
    try:
        with open('optimized_params_live.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to hardcoded optimized parameters from Score-10 backtesting
        return {
            "champion_profile": "degen_mode",
            "optimized_params": {
                "donchian_lb": 24,
                "mom_lb": 6, 
                "trend_strength_thresh": 0.002,
                "partial_frac": 0.4,
                "trail_k_min": 1.2
            },
            "execution_params": {
                "funding_max_long": 0.0005,
                "funding_max_short": 0.0005,
                "fee_buffer": 0.002,
                "impact_buffer": 0.001,
                # Score-10 optimizations
                "ultra_tight_stop_mult": 0.3,  # 70% tighter stops
                "nano_profit_target": 0.02,    # 0.02R nano profits for 75% win rate
                "ml_signal_threshold": 10,     # ML signal confirmation threshold
                "smart_sizing_enabled": True,  # Enable smart position sizing
                "regime_detection_enabled": True  # Enable regime-aware trading
            }
        }

# Score-10 Risk Management Parameters
SCORE_10_RISK_PARAMS = {
    "ultra_tight": {
        "stop_loss_mult": 0.3,        # 70% tighter than normal
        "profit_target": 0.02,        # 0.02R for high win rate
        "breakeven_threshold": 0.01,  # Quick breakeven
        "trailing_activation": 0.05   # Start trailing early
    },
    "nano_profit": {
        "stop_loss_mult": 0.3,        # Champion Degen Mode settings
        "profit_target": 0.02,        # Nano profit taking
        "max_hold_hours": 2,          # Quick exits
        "smart_sizing_mult": 2.0      # Max 2x sizing based on signal strength
    },
    "optimized": {
        "stop_loss_mult": 0.6,        # Moderately tight
        "profit_target": 0.15,        # Balanced profit target
        "breakeven_threshold": 0.05,
        "trailing_activation": 0.1
    },
    "conservative_optimized": {
        "stop_loss_mult": 0.8,        # Looser but optimized
        "profit_target": 0.2,         # Conservative profit target
        "breakeven_threshold": 0.1,
        "trailing_activation": 0.15
    },
    "ai_optimized": {
        "stop_loss_mult": 0.5,        # AI-determined
        "profit_target": 0.1,         # Adaptive target
        "ml_confirmation": True,      # Require ML signal confirmation
        "regime_aware": True          # Adjust based on market regime
    }
}

# ML Signal Scoring (from backtesting)
ML_SIGNAL_WEIGHTS = {
    "breakout_signal": 3,
    "momentum_confirmation": 3,
    "trend_alignment": 2,
    "timeframe_confirmation": 1,
    "ema_pullback": 2,
    "adx_strength": 2,
    "low_volatility_bonus": 1
}

def get_optimized_stop_loss_params(stop_type: str) -> Dict[str, float]:
    """Get optimized stop loss parameters for a given type"""
    return SCORE_10_RISK_PARAMS.get(stop_type, SCORE_10_RISK_PARAMS["optimized"])

def calculate_ml_signal_score(market_data: Dict[str, Any]) -> float:
    """Calculate ML signal confirmation score (0-14 scale)"""
    score = 0
    
    # Apply ML signal weights from backtesting
    if market_data.get('breakout_detected'): score += ML_SIGNAL_WEIGHTS["breakout_signal"]
    if market_data.get('momentum_strong'): score += ML_SIGNAL_WEIGHTS["momentum_confirmation"]
    if market_data.get('trend_aligned'): score += ML_SIGNAL_WEIGHTS["trend_alignment"]
    if market_data.get('timeframe_confirmed'): score += ML_SIGNAL_WEIGHTS["timeframe_confirmation"]
    if market_data.get('ema_pullback'): score += ML_SIGNAL_WEIGHTS["ema_pullback"]
    if market_data.get('adx_strong'): score += ML_SIGNAL_WEIGHTS["adx_strength"]
    if market_data.get('low_volatility'): score += ML_SIGNAL_WEIGHTS["low_volatility_bonus"]
    
    return score

def apply_smart_position_sizing(base_size: float, signal_strength: float, volatility: float) -> float:
    """Apply smart position sizing from Score-10 optimization"""
    # Signal strength multiplier (1.0 - 3.0)
    signal_mult = min(signal_strength, 3.0)
    
    # Volatility adjustment (reduce size in high volatility)
    vol_adj = 1.0 / max(volatility * 100, 0.5)
    
    # Combined multiplier (capped at 2.0x)
    size_mult = min(signal_mult * vol_adj, 2.0)
    
    return base_size * size_mult

def is_high_quality_signal(ml_score: float, min_threshold: float = 10.0) -> bool:
    """Check if signal meets Score-10 quality threshold"""
    return ml_score >= min_threshold

# Export optimized configuration
def get_live_optimizations() -> Dict[str, Any]:
    """Get complete live optimization configuration"""
    base_params = load_optimized_params()
    
    return {
        "champion_mode": "degen_mode",
        "score_10_enabled": True,
        "target_win_rate": 0.75,  # 75% target
        "optimized_params": base_params.get("optimized_params", {}),
        "execution_params": base_params.get("execution_params", {}),
        "risk_params": SCORE_10_RISK_PARAMS,
        "ml_weights": ML_SIGNAL_WEIGHTS,
        "performance_targets": {
            "min_score": 70.0,      # Minimum acceptable score
            "target_win_rate": 75.0, # Target win rate %
            "max_drawdown": 5.0,     # Maximum drawdown %
            "min_sharpe": 2.0        # Minimum Sharpe ratio
        }
    }

if __name__ == "__main__":
    # Test the optimization loading
    config = get_live_optimizations()
    print("🏆 Score-10 Live Optimizations Loaded:")
    print(f"Champion Mode: {config['champion_mode']}")
    print(f"Target Win Rate: {config['target_win_rate']*100}%")
    print(f"Performance Targets: {config['performance_targets']}")
