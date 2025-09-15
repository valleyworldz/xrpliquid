#!/bin/bash
# ENHANCED LAUNCHER FOR HYPERLIQUID TRADING BOT
# Runs startup wizard first, then launches the enhanced GUI

echo "🚀 HyperLiquid Enhanced Trading Bot Launcher"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "enhanced_auto_bot.py" ]; then
    echo "❌ Error: enhanced_auto_bot.py not found"
    echo "Please run this script from the hyperliquid_test directory"
    exit 1
fi

# Check Python and tkinter
echo "🔍 Checking dependencies..."

if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "📦 Installing tkinter..."
    sudo apt-get update
    sudo apt-get install -y python3-tk
fi

echo "✅ All dependencies ready"

# Check for existing configuration
if [ -f "secure_creds.env" ] && [ -f "bot_config.json" ]; then
    echo "📋 Existing configuration found"
    echo "🎯 Choose an option:"
    echo "1) Launch with existing configuration"
    echo "2) Run setup wizard to change settings"
    echo "3) Exit"
    
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo "🚀 Launching with existing configuration..."
            python3 trading_bot_gui.py
            ;;
        2)
            echo "⚙️ Running setup wizard..."
            python3 startup_wizard.py
            ;;
        3)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Launching setup wizard..."
            python3 startup_wizard.py
            ;;
    esac
else
    echo "🆕 No configuration found - running setup wizard..."
    python3 startup_wizard.py
fi

echo "👋 Launcher finished"

