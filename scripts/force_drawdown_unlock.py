#!/usr/bin/env python3
"""
CTO HAT: FORCE DRAWDOWN UNLOCK
Aggressively clear drawdown lock from bot memory and state files
"""

import json
import os
import time
from datetime import datetime

def force_drawdown_unlock():
    """Force unlock drawdown by clearing all possible state files and memory"""
    
    print("🔧 CTO HAT: FORCE DRAWDOWN UNLOCK")
    print("=" * 60)
    print("🚨 AGGRESSIVE DRAWDOWN LOCK CLEARING")
    print("=" * 60)
    
    # List of all possible state files
    state_files = [
        "runtime_state.tmp",
        "runtime_state.json", 
        "XRP_runtime_state.json",
        "XRP_runtime_state.tmp",
        "ml_engine_state.json",
        "cooldown_state.json"
    ]
    
    cleared_files = []
    
    for state_file in state_files:
        if os.path.exists(state_file):
            try:
                # Read current state
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Clear drawdown lock from state
                if 'drawdown_lock_time' in state:
                    state['drawdown_lock_time'] = None
                    cleared_files.append(state_file)
                
                # Clear any other lock-related fields
                lock_fields = [
                    'dd_peak', 'peak_capital', 'current_capital',
                    'trading_paused_for_day', 'cooldown_until'
                ]
                
                for field in lock_fields:
                    if field in state:
                        if field == 'trading_paused_for_day':
                            state[field] = False
                        elif field == 'cooldown_until':
                            state[field] = 0
                        else:
                            state[field] = None
                
                # Write back cleared state
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)
                
                print(f"✅ Cleared drawdown lock from {state_file}")
                
            except Exception as e:
                print(f"⚠️ Could not clear {state_file}: {e}")
        else:
            print(f"ℹ️ {state_file} not found")
    
    # Create a new clean state file
    clean_state = {
        "drawdown_lock_time": None,
        "dd_peak": None,
        "peak_capital": 0.0,
        "current_capital": 0.0,
        "trading_paused_for_day": False,
        "cooldown_until": 0,
        "last_reset": time.time(),
        "force_unlock": True,
        "unlock_timestamp": datetime.now().isoformat()
    }
    
    # Write clean state to multiple locations
    for state_file in ["runtime_state.json", "XRP_runtime_state.json"]:
        try:
            with open(state_file, 'w') as f:
                json.dump(clean_state, f, indent=2)
            print(f"✅ Created clean state file: {state_file}")
        except Exception as e:
            print(f"⚠️ Could not create {state_file}: {e}")
    
    print("=" * 60)
    print("✅ FORCE DRAWDOWN UNLOCK COMPLETE")
    print(f"📊 Cleared {len(cleared_files)} state files")
    print("🎯 Bot should now be able to trade immediately")
    print("=" * 60)

if __name__ == "__main__":
    force_drawdown_unlock()
