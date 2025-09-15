@echo off
echo ============================================================
echo SAFELY CONFIDENCE PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting SAFELY CONFIDENCE PATCHED bot...
echo Bot will use 0.0001 threshold and execute ALL trades
echo Interactive menus will work normally - select option 1 manually

python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
