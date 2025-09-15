@echo off
echo ============================================================
echo CAREFULLY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting CAREFULLY PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee_threshold_multi 0.01 --confidence_threshold 0.0001 --aggressive_mode true

pause
