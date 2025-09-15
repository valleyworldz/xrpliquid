@echo off
echo ============================================================
echo TARGETED PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting TARGETED PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee_threshold_multi 0.01 --confidence_threshold 0.0001 --aggressive_mode true

pause
