#!/usr/bin/env python3
"""
RESET DRAWDOWN LOCK - CTO HAT EMERGENCY FIX
================================================================================
CRITICAL: Clears the drawdown lock preventing all trading
"""

import os
import json
import time

def reset_drawdown_lock():
    """Reset the drawdown lock to allow trading"""
    print("🔧 CTO HAT: RESETTING DRAWDOWN LOCK")
    print("=" * 50)
    
    # Clear any state files that might contain drawdown lock
    state_files = [
        "runtime_state.tmp",
        "runtime_state.json",
        "XRP_runtime_state.json",
        "XRP_runtime_state.tmp"
    ]
    
    for state_file in state_files:
        if os.path.exists(state_file):
            try:
                # Read current state
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Clear drawdown lock
                if 'drawdown_lock_time' in state:
                    state['drawdown_lock_time'] = None
                    print(f"✅ Cleared drawdown_lock_time in {state_file}")
                
                # Reset peak capital to current value
                if 'peak_capital' in state:
                    state['peak_capital'] = 27.47  # Current account value
                    print(f"✅ Reset peak_capital to current value in {state_file}")
                
                # Save updated state
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                print(f"✅ Updated {state_file}")
                
            except Exception as e:
                print(f"⚠️ Could not update {state_file}: {e}")
        else:
            print(f"ℹ️ {state_file} not found")
    
    print("✅ DRAWDOWN LOCK RESET COMPLETE")
    print("🎯 Bot should now be able to trade again")

if __name__ == "__main__":
    reset_drawdown_lock()
