#!/usr/bin/env python3
"""
Apply Score-10 Optimizations to Live Bot
========================================
Applies champion backtesting results to live trading configuration
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def apply_optimizations():
    """Apply Score-10 optimizations to live bot configuration"""
    
    print("🏆 APPLYING SCORE-10 OPTIMIZATIONS TO LIVE BOT")
    print("=" * 60)
    
    # Load the optimized parameters
    try:
        with open('optimized_params_live.json', 'r') as f:
            optimized_data = json.load(f)
        print("✅ Loaded optimized parameters from backtesting")
    except FileNotFoundError:
        print("❌ optimized_params_live.json not found")
        return False
    
    # Extract champion performance
    champion_profile = None
    best_score = 0
    
    for profile_name, profile_data in optimized_data.get('profiles', {}).items():
        score = profile_data.get('meta', {}).get('overall_score', 0)
        if score > best_score:
            best_score = score
            champion_profile = profile_name
    
    if not champion_profile:
        print("❌ No champion profile found")
        return False
    
    champion_data = optimized_data['profiles'][champion_profile]
    print(f"🏆 Champion Profile: {champion_profile}")
    print(f"📊 Score: {best_score:.1f}/100")
    print(f"💰 Return: {champion_data['meta']['return_pct']:.2f}%")
    print(f"🎯 Win Rate: {champion_data['meta']['win_rate_pct']:.1f}%")
    print(f"🛡️ Max DD: {champion_data['meta']['max_dd_pct']:.2f}%")
    print()
    
    # Create live runtime configuration
    live_config = {
        "score_10_mode": True,
        "champion_profile": champion_profile,
        "champion_score": best_score,
        "optimized_trading_params": champion_data['params'],
        "execution_settings": {
            "ultra_tight_stops": True,
            "nano_profit_taking": True,
            "ml_signal_confirmation": True,
            "smart_position_sizing": True,
            "regime_detection": True,
            "funding_aware_entries": True
        },
        "risk_management": {
            "stop_loss_multiplier": 0.3,    # 70% tighter stops
            "profit_target_ratio": 0.02,    # 0.02R nano profits
            "ml_threshold": 10,             # Require strong ML signals
            "max_position_multiplier": 2.0, # Max 2x smart sizing
            "daily_loss_limit": 0.05,      # 5% daily loss limit
            "weekly_loss_limit": 0.15      # 15% weekly loss limit
        },
        "performance_targets": {
            "target_win_rate": 75.0,       # 75% win rate target
            "min_score": 70.0,             # Minimum 70/100 score
            "max_drawdown": 5.0,           # Max 5% drawdown
            "min_monthly_return": 5.0      # Target 5%+ monthly returns
        },
        "timestamp": optimized_data.get('timestamp'),
        "symbol": optimized_data.get('symbol', 'XRP'),
        "backtest_hours": optimized_data.get('hours', 720)
    }
    
    # Save live runtime overrides
    with open('live_runtime_overrides.json', 'w') as f:
        json.dump(live_config, f, indent=2)
    print("✅ Created live_runtime_overrides.json")
    
    # Update last config to champion
    last_config = {
        "profile": champion_profile,
        "symbol": live_config["symbol"],
        "score_10_optimized": True,
        "champion_score": best_score,
        "optimizations_applied": True,
        "timestamp": live_config["timestamp"]
    }
    
    with open('last_config.json', 'w') as f:
        json.dump(last_config, f, indent=2)
    print("✅ Updated last_config.json with champion profile")
    
    # Create environment variables file for easy deployment
    env_vars = f"""# Score-10 Optimization Environment Variables
SCORE_10_MODE=1
CHAMPION_PROFILE={champion_profile}
ULTRA_TIGHT_STOPS=1
NANO_PROFIT_TAKING=1
ML_SIGNAL_CONFIRMATION=1
SMART_POSITION_SIZING=1
REGIME_DETECTION=1
TARGET_WIN_RATE=75
MIN_SCORE=70
"""
    
    with open('score10_env.txt', 'w') as f:
        f.write(env_vars)
    print("✅ Created score10_env.txt with optimization flags")
    
    print()
    print("🎉 SCORE-10 OPTIMIZATIONS SUCCESSFULLY APPLIED!")
    print("=" * 60)
    print("📋 DEPLOYMENT SUMMARY:")
    print(f"   🏆 Champion: {champion_profile} (Score: {best_score:.1f}/100)")
    print(f"   🎯 Target Win Rate: 75%")
    print(f"   💰 Expected Returns: Based on +{champion_data['meta']['return_pct']:.2f}% (30d)")
    print(f"   🛡️ Risk Control: {champion_data['meta']['max_dd_pct']:.2f}% max drawdown")
    print()
    print("📁 FILES CREATED/UPDATED:")
    print("   ✅ live_runtime_overrides.json - Main configuration")
    print("   ✅ last_config.json - Quick start config")
    print("   ✅ score10_env.txt - Environment variables")
    print()
    print("🚀 Ready for live deployment with Score-10 optimizations!")
    
    return True

if __name__ == "__main__":
    success = apply_optimizations()
    if success:
        print("\n🎯 To start the optimized bot, run:")
        print("   python scripts/launch_bot.py")
        print("   (The bot will automatically load Score-10 optimizations)")
    else:
        print("\n❌ Failed to apply optimizations")
        sys.exit(1)
