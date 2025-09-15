#!/usr/bin/env python3
"""
🚀 LAUNCH TRADING SYSTEM
========================
Simple launcher for the HyperLiquid trading system
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the trading system"""
    print("🚀 LAUNCHING HYPERLIQUID TRADING SYSTEM")
    print("=" * 45)
    
    # Check if main system exists in root
    root_system = "ULTIMATE_100_PERCENT_PERFECT_SYSTEM.py"
    optimized_system = "optimized_scripts/trading_systems/ULTIMATE_100_PERCENT_PERFECT_SYSTEM.py"
    
    if os.path.exists(root_system):
        print(f"✅ Found system in root: {root_system}")
        system_path = root_system
    elif os.path.exists(optimized_system):
        print(f"✅ Found system in optimized: {optimized_system}")
        system_path = optimized_system
    else:
        print("❌ No trading system found!")
        print("Available options:")
        
        # List available systems
        if os.path.exists("optimized_scripts/trading_systems/"):
            systems = os.listdir("optimized_scripts/trading_systems/")
            for system in systems:
                print(f"   - optimized_scripts/trading_systems/{system}")
        
        if os.path.exists("backup_scripts/"):
            backups = [f for f in os.listdir("backup_scripts/") if f.endswith('.py')]
            for backup in backups:
                print(f"   - backup_scripts/{backup}")
        
        return
    
    print(f"🎯 Launching: {system_path}")
    print("⏳ Starting autonomous trading...")
    print("-" * 45)
    
    try:
        # Launch the system
        subprocess.run([sys.executable, system_path], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Trading system stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching system: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 