@echo off
echo ============================================================
echo SIMPLY DATA FIX PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting SIMPLY DATA FIX PATCHED bot...
echo Bot will use 0.0001 threshold and execute ALL trades
echo Data structure issues fixed - no more crashes

python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
