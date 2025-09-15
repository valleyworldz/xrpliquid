# PowerShell script to start the bot with bypass
Write-Host "STARTING A.I. ULTIMATE CHAMPION BOT (POWERSHELL)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# Set environment variables
$env:BOT_BYPASS_INTERACTIVE = "true"
$env:BOT_SYMBOL = "XRP"
$env:BOT_MARKET = "perp"
$env:BOT_QUOTE = "USDT"
$env:BOT_OPTIMIZE = "false"

Write-Host "Environment variables set" -ForegroundColor Green
Write-Host "Target: +213.6% annual returns" -ForegroundColor Yellow
Write-Host "Leverage: 8.0x" -ForegroundColor Yellow
Write-Host "Risk: Quantum Master (4.0%)" -ForegroundColor Yellow
Write-Host "Mode: Quantum Adaptive" -ForegroundColor Yellow
Write-Host "Stops: Quantum Optimal" -ForegroundColor Yellow
Write-Host "Guardian: Quantum-Adaptive (Enhanced)" -ForegroundColor Yellow

Write-Host "`nStarting bot..." -ForegroundColor Green

# Start the bot
python newbotcode.py
