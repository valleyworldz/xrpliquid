#!/usr/bin/env python3
"""
Force Trade Execution Script
Bypasses ALL remaining restrictions to force trade execution
"""

import os
import json
import time

def force_trade_execution():
    """Force trade execution by updating critical parameters"""
    print("🚨 FORCE TRADE EXECUTION - BYPASSING ALL RESTRICTIONS")
    print("=" * 60)

    # Set ultra-aggressive environment variables
    os.environ["BOT_CONFIDENCE_THRESHOLD"] = "0.001"  # Ultra-low threshold
    os.environ["V8_MICROSTRUCTURE_SPREAD_CAP"] = "0.005"  # Increased spread tolerance
    os.environ["V8_MICROSTRUCTURE_IMBALANCE_GATE"] = "0.25"  # Increased imbalance tolerance
    os.environ["BOT_AGGRESSIVE_MODE"] = "true"
    os.environ["EMERGENCY_MICROSTRUCTURE_BYPASS"] = "true"  # Force bypass
    os.environ["V8_POSITION_LOSS_THRESHOLD"] = "0.10"  # Increased loss tolerance
    os.environ["BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD"] = "true"
    os.environ["BOT_MIN_PNL_THRESHOLD"] = "0.0001"  # Ultra-low PnL threshold
    os.environ["BOT_BYPASS_INTERACTIVE"] = "true"
    os.environ["BOT_FEE_THRESHOLD_MULTIPLIER"] = "0.1"  # Ultra-low fee threshold
    
    # NEW: Bypass RSI and other technical filters
    os.environ["BOT_DISABLE_RSI_FILTER"] = "true"
    os.environ["BOT_DISABLE_TECHNICAL_FILTERS"] = "true"
    os.environ["BOT_DISABLE_MOMENTUM_FILTER"] = "true"
    os.environ["BOT_DISABLE_TREND_FILTER"] = "true"
    os.environ["BOT_DISABLE_VOLATILITY_FILTER"] = "true"
    
    # Force override internal thresholds
    os.environ["BOT_FORCE_CONFIDENCE_THRESHOLD"] = "0.001"
    os.environ["BOT_OVERRIDE_INTERNAL_THRESHOLDS"] = "true"
    os.environ["BOT_DISABLE_DYNAMIC_THRESHOLD"] = "true"

    print("✅ Environment variables set for ultra-aggressive trading")

    # Update ML engine state with ultra-low confidence threshold
    try:
        with open('ml_engine_state.json', 'r') as f:
            config = json.load(f)

        config['current_params']['confidence_threshold'] = 0.001
        config['current_params']['position_size_multiplier'] = 2.0
        config['current_params']['risk_multiplier'] = 1.5

        with open('ml_engine_state.json', 'w') as f:
            json.dump(config, f, indent=2)

        print("✅ ML engine updated with ultra-low confidence threshold (0.001)")

    except Exception as e:
        print(f"⚠️ ML engine update failed: {e}")

    # Create ultra-aggressive startup script
    startup_script = """@echo off
echo ============================================================
echo ULTRA-AGGRESSIVE FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Setting ULTRA-AGGRESSIVE environment variables...
set BOT_CONFIDENCE_THRESHOLD=0.001
set V8_MICROSTRUCTURE_SPREAD_CAP=0.005
set V8_MICROSTRUCTURE_IMBALANCE_GATE=0.25
set EMERGENCY_MICROSTRUCTURE_BYPASS=true
set V8_POSITION_LOSS_THRESHOLD=0.10
set BOT_AGGRESSIVE_MODE=true
set BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD=true
set BOT_MIN_PNL_THRESHOLD=0.0001
set BOT_BYPASS_INTERACTIVE=true
set BOT_FEE_THRESHOLD_MULTIPLIER=0.1

set BOT_DISABLE_RSI_FILTER=true
set BOT_DISABLE_TECHNICAL_FILTERS=true
set BOT_DISABLE_MOMENTUM_FILTER=true
set BOT_DISABLE_TREND_FILTER=true
set BOT_DISABLE_VOLATILITY_FILTER=true

set BOT_FORCE_CONFIDENCE_THRESHOLD=0.001
set BOT_OVERRIDE_INTERNAL_THRESHOLDS=true
set BOT_DISABLE_DYNAMIC_THRESHOLD=true

echo.
echo ULTRA-AGGRESSIVE environment variables set
echo.

echo Starting bot with FORCE TRADE EXECUTION parameters...
python newbotcode.py --fee_threshold_multi 0.1 --confidence_threshold 0.001 --aggressive_mode true

pause
"""

    with open('start_force_trades.bat', 'w') as f:
        f.write(startup_script)

    print("✅ Created start_force_trades.bat with ultra-aggressive parameters")

    print("\n🎯 FORCE TRADE EXECUTION READY!")
    print("=" * 40)
    print("1. Run: .\\start_force_trades.bat")
    print("2. Bot will execute trades with confidence > 0.001")
    print("3. ALL restrictions bypassed for maximum trade frequency")
    print("4. RSI, technical, and momentum filters disabled")
    print("5. Internal threshold overrides enabled")
    print("6. Monitor trades_log.csv for execution")

    return True

if __name__ == "__main__":
    force_trade_execution()
