@echo off
title VERBOSE LOGGING BOT - MAXIMUM VISIBILITY
color 0B

echo ================================================================
echo ≡ƒÜ¿ VERBOSE LOGGING BOT - MAXIMUM BOT VISIBILITY ≡ƒÜ¿
echo ================================================================
echo.
echo ≡ƒÄ» A.I. ULTIMATE Profile: CHAMPION +213
echo Γ£à All crashes eliminated
echo Γ£à All restrictions removed
echo Γ£à Maximum trade execution enabled
echo Γ£à VERBOSE LOGGING ENABLED - SEE EVERYTHING
echo.
echo ≡ƒôè Real-time verbose output capture
echo ≡ƒöä All bot activity visible
echo ≡ƒôê Maximum transparency
echo.
echo ================================================================
echo.

echo ≡ƒÜÇ Creating automated input file...
echo 1 > verbose_input.txt
echo 6 >> verbose_input.txt
echo XRP >> verbose_input.txt
echo Y >> verbose_input.txt
echo Y >> verbose_input.txt
echo ✅ Input file created

echo.
echo ≡ƒÜÇ Launching bot with VERBOSE LOGGING...
echo 🔍 ALL OUTPUT WILL BE DISPLAYED IN REAL-TIME
echo.

REM Launch bot with valid arguments and show ALL output
python newbotcode.py --fee-threshold-multi 0.001 --disable-rsi-veto --disable-momentum-veto --disable-microstructure-veto < verbose_input.txt

echo.
echo ΓÜá∩╕Å Bot stopped. Press any key to exit...
pause >nul
