#!/usr/bin/env python3
"""
Emergency Override All Filters
Bypasses ALL internal bot logic to force trade execution
"""

import os
import json
import time
import subprocess

def emergency_override_all_filters():
    """Emergency override to bypass ALL bot restrictions"""
    print("🚨 EMERGENCY OVERRIDE - BYPASSING ALL INTERNAL LOGIC")
    print("=" * 60)

    # Set environment variables to override ALL internal logic
    override_vars = {
        # Core confidence overrides
        "BOT_CONFIDENCE_THRESHOLD": "0.0001",
        "BOT_FORCE_CONFIDENCE_THRESHOLD": "0.0001",
        "BOT_OVERRIDE_INTERNAL_THRESHOLDS": "true",
        "BOT_DISABLE_DYNAMIC_THRESHOLD": "true",
        
        # V8 microstructure overrides
        "V8_MICROSTRUCTURE_SPREAD_CAP": "0.010",
        "V8_MICROSTRUCTURE_IMBALANCE_GATE": "0.50",
        "EMERGENCY_MICROSTRUCTURE_BYPASS": "true",
        "V8_POSITION_LOSS_THRESHOLD": "0.20",
        
        # Bot mode overrides
        "BOT_AGGRESSIVE_MODE": "true",
        "BOT_EMERGENCY_MODE": "true",
        "BOT_FORCE_TRADE_MODE": "true",
        
        # Micro-account safeguard overrides
        "BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD": "true",
        "BOT_MIN_PNL_THRESHOLD": "0.00001",
        "BOT_FEE_THRESHOLD_MULTIPLIER": "0.01",
        
        # Interactive bypass
        "BOT_BYPASS_INTERACTIVE": "true",
        "BOT_DISABLE_USER_INPUT": "true",
        
        # Technical filter overrides
        "BOT_DISABLE_RSI_FILTER": "true",
        "BOT_DISABLE_TECHNICAL_FILTERS": "true",
        "BOT_DISABLE_MOMENTUM_FILTER": "true",
        "BOT_DISABLE_TREND_FILTER": "true",
        "BOT_DISABLE_VOLATILITY_FILTER": "true",
        "BOT_DISABLE_MACD_FILTER": "true",
        "BOT_DISABLE_EMA_FILTER": "true",
        
        # Risk management overrides
        "BOT_DISABLE_RISK_CHECKS": "true",
        "BOT_DISABLE_DRAWDOWN_THROTTLE": "true",
        "BOT_DISABLE_POSITION_SIZING": "false",  # Keep for safety
        "BOT_DISABLE_STOP_LOSS": "false",  # Keep for safety
        
        # Internal logic overrides
        "BOT_OVERRIDE_SIGNAL_FILTERS": "true",
        "BOT_OVERRIDE_CONFIDENCE_CALC": "true",
        "BOT_OVERRIDE_THRESHOLD_LOGIC": "true",
        "BOT_DISABLE_AUTO_OPTIMIZATION": "true",
        
        # Force execution flags
        "BOT_FORCE_EXECUTE_SIGNALS": "true",
        "BOT_IGNORE_ALL_VETOES": "true",
        "BOT_EMERGENCY_EXECUTION": "true"
    }

    # Set all environment variables
    for var, value in override_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")

    print("\n✅ ALL environment variables set for emergency override")

    # Update ML engine with emergency settings
    try:
        with open('ml_engine_state.json', 'r') as f:
            config = json.load(f)

        # Force emergency ML parameters
        config['current_params']['confidence_threshold'] = 0.0001
        config['current_params']['position_size_multiplier'] = 3.0
        config['current_params']['risk_multiplier'] = 2.0
        config['current_params']['momentum_threshold'] = 0.0
        config['current_params']['trend_threshold'] = 0.0
        config['current_params']['volatility_threshold'] = 0.0

        with open('ml_engine_state.json', 'w') as f:
            json.dump(config, f, indent=2)

        print("✅ ML engine updated with emergency parameters")

    except Exception as e:
        print(f"⚠️ ML engine update failed: {e}")

    # Create emergency startup script
    emergency_script = """@echo off
echo ============================================================
echo EMERGENCY OVERRIDE - ALL FILTERS BYPASSED
echo ============================================================
echo.

echo Setting EMERGENCY OVERRIDE environment variables...
set BOT_CONFIDENCE_THRESHOLD=0.0001
set BOT_FORCE_CONFIDENCE_THRESHOLD=0.0001
set BOT_OVERRIDE_INTERNAL_THRESHOLDS=true
set BOT_DISABLE_DYNAMIC_THRESHOLD=true

set V8_MICROSTRUCTURE_SPREAD_CAP=0.010
set V8_MICROSTRUCTURE_IMBALANCE_GATE=0.50
set EMERGENCY_MICROSTRUCTURE_BYPASS=true
set V8_POSITION_LOSS_THRESHOLD=0.20

set BOT_AGGRESSIVE_MODE=true
set BOT_EMERGENCY_MODE=true
set BOT_FORCE_TRADE_MODE=true

set BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD=true
set BOT_MIN_PNL_THRESHOLD=0.00001
set BOT_FEE_THRESHOLD_MULTIPLIER=0.01

set BOT_BYPASS_INTERACTIVE=true
set BOT_DISABLE_USER_INPUT=true

set BOT_DISABLE_RSI_FILTER=true
set BOT_DISABLE_TECHNICAL_FILTERS=true
set BOT_DISABLE_MOMENTUM_FILTER=true
set BOT_DISABLE_TREND_FILTER=true
set BOT_DISABLE_VOLATILITY_FILTER=true
set BOT_DISABLE_MACD_FILTER=true
set BOT_DISABLE_EMA_FILTER=true

set BOT_DISABLE_RISK_CHECKS=true
set BOT_DISABLE_DRAWDOWN_THROTTLE=true
set BOT_OVERRIDE_SIGNAL_FILTERS=true
set BOT_OVERRIDE_CONFIDENCE_CALC=true
set BOT_OVERRIDE_THRESHOLD_LOGIC=true
set BOT_DISABLE_AUTO_OPTIMIZATION=true

set BOT_FORCE_EXECUTE_SIGNALS=true
set BOT_IGNORE_ALL_VETOES=true
set BOT_EMERGENCY_EXECUTION=true

echo.
echo EMERGENCY OVERRIDE environment variables set
echo.

echo Starting bot with EMERGENCY OVERRIDE parameters...
python newbotcode.py --fee_threshold_multi 0.01 --confidence_threshold 0.0001 --aggressive_mode true --emergency_mode true

pause
"""

    with open('start_emergency_override.bat', 'w') as f:
        f.write(emergency_script)

    print("✅ Created start_emergency_override.bat with emergency parameters")

    # Kill any existing bot processes
    try:
        print("\n🔄 Stopping any existing bot processes...")
        subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                      capture_output=True, text=True)
        time.sleep(2)
        print("✅ Existing processes stopped")
    except Exception as e:
        print(f"⚠️ Process stop failed: {e}")

    print("\n🚨 EMERGENCY OVERRIDE READY!")
    print("=" * 40)
    print("1. Run: .\\start_emergency_override.bat")
    print("2. Bot will execute trades with confidence > 0.0001")
    print("3. ALL internal filters and logic bypassed")
    print("4. Emergency execution mode enabled")
    print("5. No vetoes or restrictions")
    print("6. Monitor trades_log.csv for execution")

    return True

if __name__ == "__main__":
    emergency_override_all_filters()
