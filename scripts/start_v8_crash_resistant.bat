@echo off
echo ============================================================
echo 🚨 V8 EMERGENCY FIXES - CRASH RESISTANT STARTUP
echo ============================================================
echo.

echo 🔧 Setting V8 Emergency Fixes environment variables...
set V8_MICROSTRUCTURE_SPREAD_CAP=0.0025
set V8_MICROSTRUCTURE_IMBALANCE_GATE=0.15
set EMERGENCY_MICROSTRUCTURE_BYPASS=false
set V8_POSITION_LOSS_THRESHOLD=0.05
set BOT_CONFIDENCE_THRESHOLD=0.005
set BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD=true
set BOT_MIN_PNL_THRESHOLD=0.001
set BOT_BYPASS_INTERACTIVE=true

echo.
echo ✅ Environment variables set:
echo    V8_MICROSTRUCTURE_SPREAD_CAP=%V8_MICROSTRUCTURE_SPREAD_CAP%
echo    V8_MICROSTRUCTURE_IMBALANCE_GATE=%V8_MICROSTRUCTURE_IMBALANCE_GATE%
echo    EMERGENCY_MICROSTRUCTURE_BYPASS=%EMERGENCY_MICROSTRUCTURE_BYPASS%
echo    V8_POSITION_LOSS_THRESHOLD=%V8_POSITION_LOSS_THRESHOLD%
echo    BOT_CONFIDENCE_THRESHOLD=%BOT_CONFIDENCE_THRESHOLD%
echo    BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD=%BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD%
echo    BOT_MIN_PNL_THRESHOLD=%BOT_MIN_PNL_THRESHOLD%
echo    BOT_BYPASS_INTERACTIVE=%BOT_BYPASS_INTERACTIVE%

echo.
echo 🚀 Starting XRP Bot with V8 optimizations...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found or not in PATH
    echo Please ensure Python is installed and accessible
    pause
    exit /b 1
)

REM Check if newbotcode.py exists
if not exist "newbotcode.py" (
    echo ❌ ERROR: newbotcode.py not found in current directory
    echo Please ensure you're running this script from the correct directory
    pause
    exit /b 1
)

echo ✅ Pre-flight checks passed
echo.

REM Start the bot with error handling
echo 🎯 Launching bot with command: python newbotcode.py --fee_threshold_multi 0.5
echo.

REM Run the bot and capture any errors
python newbotcode.py --fee_threshold_multi 0.5 2>&1

REM Check exit code
if errorlevel 1 (
    echo.
    echo ❌ BOT CRASHED with exit code %errorlevel%
    echo.
    echo 🔍 TROUBLESHOOTING STEPS:
    echo 1. Check if all required files exist
    echo 2. Verify Python dependencies are installed
    echo 3. Check bot logs for specific error messages
    echo 4. Ensure environment variables are properly set
    echo.
    echo 📋 CRASH ANALYSIS:
    echo - Exit Code: %errorlevel%
    echo - Time: %time%
    echo - Date: %date%
    echo.
    pause
    exit /b %errorlevel%
) else (
    echo.
    echo ✅ Bot exited normally
    echo.
)

echo.
echo 🏁 Startup script completed
pause
