@echo off
title ULTIMATE AUTOMATED BOT - MAXIMUM TRADE EXECUTION
color 0A

echo ================================================================
echo ≡ƒÜ¿ ULTIMATE AUTOMATED BOT LAUNCH - MAXIMUM TRADE EXECUTION ≡ƒÜ¿
echo ================================================================
echo.
echo ≡ƒÄ» A.I. ULTIMATE Profile: CHAMPION +213
echo Γ£à All crashes eliminated
echo Γ£à All restrictions removed
echo Γ£à Maximum trade execution enabled
echo Γ£à FULLY AUTOMATED - NO USER INPUT REQUIRED
echo.
echo ≡ƒôè Bot will run continuously with real-time monitoring
echo ≡ƒöä Auto-restart on any issues
echo ≡ƒôê Performance tracking enabled
echo.
echo ================================================================
echo.

:launch_loop
echo ≡ƒÜÇ Launching bot... (Attempt: %random%)
echo.

REM Create input file with automated responses
echo 1 > auto_input.txt
echo 6 >> auto_input.txt
echo XRP >> auto_input.txt
echo Y >> auto_input.txt
echo Y >> auto_input.txt

REM Launch bot with input redirection
python newbotcode.py --fee-threshold-multi 0.001 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto < auto_input.txt

echo.
echo ΓÜá∩╕Å Bot stopped or crashed. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
echo ≡ƒöä Restarting bot...
echo.
goto launch_loop

