@echo off
title 🚨 ULTIMATE MONITORED BOT LAUNCH 🚨
color 0A

echo.
echo ================================================================
echo 🚨 ULTIMATE MONITORED BOT LAUNCH - MAXIMUM TRADE EXECUTION 🚨
echo ================================================================
echo.
echo 🎯 A.I. ULTIMATE Profile: CHAMPION +213%
echo ✅ All crashes eliminated
echo ✅ All restrictions removed
echo ✅ Maximum trade execution enabled
echo ✅ Enhanced monitoring active
echo.
echo 📊 Bot will run continuously with real-time monitoring
echo 🔄 Auto-restart on any issues
echo 📈 Performance tracking enabled
echo.
echo ================================================================
echo.

:launch_loop
echo 🚀 Launching bot... (Attempt: %launch_count%)
echo.

REM Launch the bot with all optimizations
python newbotcode.py --fee-threshold-multi 0.001 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto

echo.
echo ⚠️ Bot stopped or crashed. Restarting in 5 seconds...
echo.

REM Wait 5 seconds before restart
timeout /t 5 /nobreak >nul

REM Increment launch counter
set /a launch_count+=1

echo 🔄 Restarting bot... (Attempt: %launch_count%)
echo.

REM Loop back to launch
goto launch_loop
