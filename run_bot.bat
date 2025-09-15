@echo off
echo 🚀 XRP Trading Bot Launcher
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found!
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Try to find and run the bot
if exist "src\core\main_bot.py" (
    echo ✅ Found bot file: src\core\main_bot.py
    echo 🚀 Starting bot...
    echo ================================================
    python src\core\main_bot.py
) else if exist "main_bot.py" (
    echo ✅ Found bot file: main_bot.py
    echo 🚀 Starting bot...
    echo ================================================
    python main_bot.py
) else if exist "newbotcode.py" (
    echo ✅ Found bot file: newbotcode.py
    echo 🚀 Starting bot...
    echo ================================================
    python newbotcode.py
) else if exist "bot.py" (
    echo ✅ Found bot file: bot.py
    echo 🚀 Starting bot...
    echo ================================================
    python bot.py
) else (
    echo ❌ ERROR: Could not find main bot file!
    echo 📁 Looking for:
    echo    • src\core\main_bot.py
    echo    • main_bot.py
    echo    • newbotcode.py
    echo    • bot.py
    echo.
    echo 🔧 Please ensure the bot file exists in one of these locations.
    pause
    exit /b 1
)

echo.
echo 🛑 Bot stopped
pause
