#!/bin/bash
# GUI Launcher Script for HyperLiquid Trading Bot
# This script ensures all dependencies are available and launches the GUI

echo "🚀 HyperLiquid Trading Bot GUI Launcher"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "ultimate_auto_bot.py" ]; then
    echo "❌ Error: ultimate_auto_bot.py not found"
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

# Check if GUI file exists
if [ ! -f "trading_bot_gui.py" ]; then
    echo "❌ Error: trading_bot_gui.py not found"
    exit 1
fi

echo "✅ All dependencies ready"
echo "🎯 Launching GUI..."

# Launch the GUI
python3 trading_bot_gui.py

echo "👋 GUI closed"

