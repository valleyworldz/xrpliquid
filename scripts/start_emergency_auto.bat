@echo off
echo ============================================================
echo EMERGENCY AUTO-START BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Setting emergency environment variables...
set BOT_BYPASS_INTERACTIVE=true
set BOT_AUTO_START=true
set BOT_FORCE_TRADING=true
set BOT_CONFIDENCE_THRESHOLD=0.0001
set BOT_FEE_THRESHOLD_MULTI=0.01
set BOT_DISABLE_RSI_VETO=true
set BOT_DISABLE_MOMENTUM_VETO=true
set BOT_DISABLE_MICROSTRUCTURE_VETO=true
set BOT_EMERGENCY_MODE=true
set BOT_MAX_TRADE_FREQUENCY=true

echo.
echo Starting bot with EMERGENCY AUTO-EXECUTION...
echo 1 | python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
