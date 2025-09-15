@echo off
echo ============================================================
echo SURGICALLY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting SURGICALLY PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
