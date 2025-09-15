@echo off
echo ============================================================
echo COMPREHENSIVELY PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting COMPREHENSIVELY PATCHED bot with FORCE EXECUTION...
python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
