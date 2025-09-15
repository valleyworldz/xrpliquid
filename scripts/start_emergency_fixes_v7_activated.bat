@echo off
echo ============================================================
echo 🚨 EMERGENCY FIXES V7 - CRITICAL ATTRIBUTE & ASYNC FIXES
echo ============================================================
echo.
echo 🔧 FIXES APPLIED:
echo ✅ Added missing BotConfig.macd_threshold attribute
echo ✅ Added missing BotConfig.ema_threshold attribute  
echo ✅ Fixed async monitor_resources "coroutine never awaited" warning
echo ✅ Signal filter optimization should now work properly
echo.
echo 🚀 LAUNCHING BOT WITH V7 FIXES...
echo.

REM Set environment variables for Emergency Fixes V7
set BOT_ENV=prod
set BOT_SYMBOL=XRP
set BOT_MARKET=perp
set BOT_QUOTE=USDT
set BOT_OPTIMIZE=false
set BOT_BYPASS_INTERACTIVE=true
set BOT_AGGRESSIVE_MODE=true
set BOT_MACD_THRESHOLD=0.000025
set BOT_RSI_RANGE=20-80
set BOT_ATR_THRESHOLD=0.0005
set BOT_CONFIDENCE_THRESHOLD=0.015

REM Launch the bot with Emergency Fixes V7
python newbotcode.py

echo.
echo ✅ Emergency Fixes V7 deployment complete
pause
