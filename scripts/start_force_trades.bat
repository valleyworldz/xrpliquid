@echo off
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

echo.
echo ULTRA-AGGRESSIVE environment variables set
echo.

echo Starting bot with FORCE TRADE EXECUTION parameters...
python newbotcode.py --fee_threshold_multi 0.1 --confidence_threshold 0.001 --aggressive_mode true

pause
