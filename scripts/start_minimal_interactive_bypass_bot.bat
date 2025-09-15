@echo off
echo ============================================================
echo MINIMAL INTERACTIVE BYPASS PATCHED BOT - FORCE TRADE EXECUTION
echo ============================================================
echo.

echo Starting MINIMAL INTERACTIVE BYPASS PATCHED bot...
echo Bot will skip the main interactive menu and start trading automatically
echo Bot will use 0.0001 threshold and execute ALL trades

REM Use echo to automatically send "1" to the bot's input
echo 1 | python newbotcode.py --fee-threshold-multi 0.01 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

pause
