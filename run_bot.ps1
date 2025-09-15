# 🚀 XRP Trading Bot Launcher (PowerShell)
Write-Host "🚀 XRP Trading Bot Launcher" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python or add it to your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Try to find and run the bot
$botFiles = @(
    "src\core\main_bot.py",
    "main_bot.py", 
    "newbotcode.py",
    "bot.py",
    "trading_bot.py"
)

$botFound = $false
foreach ($botFile in $botFiles) {
    if (Test-Path $botFile) {
        Write-Host "✅ Found bot file: $botFile" -ForegroundColor Green
        Write-Host "🚀 Starting bot..." -ForegroundColor Green
        Write-Host "================================================" -ForegroundColor Green
        
        try {
            python $botFile
            $botFound = $true
            break
        } catch {
            Write-Host "❌ Error running bot: $_" -ForegroundColor Red
            $botFound = $true
            break
        }
    }
}

if (-not $botFound) {
    Write-Host "❌ ERROR: Could not find main bot file!" -ForegroundColor Red
    Write-Host "📁 Looking for:" -ForegroundColor Yellow
    foreach ($botFile in $botFiles) {
        Write-Host "   • $botFile" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "🔧 Please ensure the bot file exists in one of these locations." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🛑 Bot stopped" -ForegroundColor Yellow
Read-Host "Press Enter to exit"
