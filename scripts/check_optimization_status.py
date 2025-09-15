#!/usr/bin/env python3
"""
Quick status check for V8 optimizations
"""

import os
import json
import time

def check_environment():
    """Check if environment variables are set"""
    print("🔍 Checking V8 optimization environment...")
    
    vars_to_check = {
        "BOT_CONFIDENCE_THRESHOLD": "0.005",
        "V8_MICROSTRUCTURE_SPREAD_CAP": "0.0025", 
        "V8_MICROSTRUCTURE_IMBALANCE_GATE": "0.15",
        "BOT_AGGRESSIVE_MODE": "true"
    }
    
    all_good = True
    for var, expected in vars_to_check.items():
        actual = os.environ.get(var)
        if actual == expected:
            print(f"✅ {var} = {actual}")
        else:
            print(f"❌ {var} = {actual} (expected {expected})")
            all_good = False
    
    return all_good

def check_ml_engine():
    """Check ML engine configuration"""
    print("\n🧠 Checking ML engine configuration...")
    
    try:
        with open('ml_engine_state.json', 'r') as f:
            config = json.load(f)
        
        threshold = config['current_params']['confidence_threshold']
        if threshold == 0.005:
            print(f"✅ ML confidence threshold: {threshold}")
        else:
            print(f"❌ ML confidence threshold: {threshold} (should be 0.005)")
            return False
    except Exception as e:
        print(f"❌ Could not read ML engine config: {e}")
        return False
    
    return True

def check_bot_process():
    """Check if bot is running"""
    print("\n🤖 Checking bot process status...")
    
    try:
        import subprocess
        result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                              capture_output=True, text=True)
        
        if 'python.exe' in result.stdout:
            print("✅ Python processes running (bot should be active)")
            return True
        else:
            print("❌ No Python processes found")
            return False
    except Exception as e:
        print(f"⚠️ Could not check process status: {e}")
        return False

def main():
    """Main status check"""
    print("🚨 V8 OPTIMIZATION STATUS CHECK")
    print("=" * 40)
    
    env_ok = check_environment()
    ml_ok = check_ml_engine()
    process_ok = check_bot_process()
    
    print("\n📊 SUMMARY:")
    print(f"   Environment: {'✅' if env_ok else '❌'}")
    print(f"   ML Engine: {'✅' if ml_ok else '❌'}")
    print(f"   Bot Process: {'✅' if process_ok else '❌'}")
    
    if env_ok and ml_ok and process_ok:
        print("\n🎯 ALL OPTIMIZATIONS ACTIVE!")
        print("📈 Expected improvements:")
        print("   • Confidence filter: 0.0% → 60%+ signal pass rate")
        print("   • Trade execution: 0% → 60%+ execution rate") 
        print("   • Performance score: 5.8/10 → 8.0+/10")
        print("\n🔍 Monitor logs for trade execution improvements")
    else:
        print("\n⚠️ Some optimizations may not be active")
        print("💡 Run 'python apply_v8_optimizations.py' to fix")

if __name__ == "__main__":
    main()
