#!/usr/bin/env python3
"""
🚀 GUI Launcher for Multi-Asset Trading Bot
===========================================
Simple launcher script with error handling and setup
"""

import sys
import os

def check_requirements():
    """Check if required packages are available"""
    missing = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    
    # Check tkinter
    try:
        import tkinter
        print("✅ GUI framework (tkinter) available")
    except ImportError:
        missing.append("tkinter")
    
    # Check if trading bot is available
    try:
        import newbotcode
        print("✅ Trading bot code available")
    except ImportError:
        print("⚠️ Trading bot not found (will use fallback mode)")
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Install missing packages and try again")
        return False
    
    return True

def main():
    """Launch the trading bot GUI"""
    print("🚀 MULTI-ASSET TRADING BOT GUI LAUNCHER")
    print("=" * 50)
    print()
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        print("\n❌ Requirements not met. Please install missing packages.")
        input("Press Enter to exit...")
        return
    
    print("✅ All requirements met!")
    print()
    
    # Launch GUI
    try:
        print("🎮 Launching GUI...")
        from trading_gui import TradingBotGUI
        
        app = TradingBotGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n👋 GUI closed by user")
    except Exception as e:
        print(f"\n❌ GUI Error: {e}")
        print("💡 Check that all dependencies are installed")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
