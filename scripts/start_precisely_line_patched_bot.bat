@echo off
echo ============================================================
echo PRECISELY LINE PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting PRECISELY LINE PATCHED bot...
echo Bot will use 0.0001 threshold and execute ALL trades
python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
