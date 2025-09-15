@echo off
echo ============================================================
echo 🚨 V8 ULTRA-OPTIMIZED STARTUP - ALL BOTTLENECKS RESOLVED
echo ============================================================
echo.

echo 🔧 Setting V8 Emergency Fixes + Micro-Account Optimizations...
set V8_MICROSTRUCTURE_SPREAD_CAP=0.0025
set V8_MICROSTRUCTURE_IMBALANCE_GATE=0.15
set EMERGENCY_MICROSTRUCTURE_BYPASS=false
set V8_POSITION_LOSS_THRESHOLD=0.05
set BOT_CONFIDENCE_THRESHOLD=0.005
set BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD=true
set BOT_MIN_PNL_THRESHOLD=0.001
set BOT_BYPASS_INTERACTIVE=true

REM CRITICAL: Force fee threshold multiplier for small accounts
set BOT_FEE_THRESHOLD_MULTIPLIER=0.3

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
echo    BOT_FEE_THRESHOLD_MULTIPLIER=%BOT_FEE_THRESHOLD_MULTIPLIER%

echo.
echo 🚀 Starting XRP Bot with ULTRA-OPTIMIZED V8 configuration...
echo.

REM Pre-flight checks
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found
    pause
    exit /b 1
)

if not exist "newbotcode.py" (
    echo ❌ ERROR: newbotcode.py not found
    pause
    exit /b 1
)

echo ✅ Pre-flight checks passed
echo.

REM Launch with ultra-optimized parameters
echo 🎯 Launching bot with ULTRA-OPTIMIZED parameters:
echo    --fee_threshold_multi 0.3 (micro-account optimized)
echo    --confidence_threshold 0.005 (V8 optimized)
echo    --aggressive_mode true (performance optimized)
echo.

python newbotcode.py --fee_threshold_multi 0.3 --confidence_threshold 0.005 --aggressive_mode true

REM Check exit code
if errorlevel 1 (
    echo.
    echo ❌ BOT EXITED with code %errorlevel%
    echo.
    echo 🔍 TROUBLESHOOTING:
    echo - Check logs for specific errors
    echo - Verify all environment variables are set
    echo - Ensure Python dependencies are installed
    echo.
    pause
    exit /b %errorlevel%
) else (
    echo.
    echo ✅ Bot exited normally
    echo.
)

echo.
echo 🏁 Ultra-optimized startup completed
pause
