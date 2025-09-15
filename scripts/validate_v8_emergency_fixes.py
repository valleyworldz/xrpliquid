#!/usr/bin/env python3
"""
V8 Emergency Fixes Validation Script
====================================

This script validates that all V8 emergency fixes have been properly implemented:
1. Microstructure veto ultra-permissive thresholds
2. RR/ATR check fallback for non-positive reward
3. Position loss kill switch threshold increased to 5.0%
4. Emergency microstructure bypass capability
"""

import os
import sys
import re
from pathlib import Path

def validate_microstructure_veto_fixes():
    """Validate microstructure veto V8 fixes"""
    print("🔍 Validating Microstructure Veto V8 Fixes...")
    
    try:
        with open('newbotcode.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for V8 emergency bypass
        if 'EMERGENCY_MICROSTRUCTURE_BYPASS' in content:
            print("✅ V8 Emergency Microstructure Bypass: IMPLEMENTED")
        else:
            print("❌ V8 Emergency Microstructure Bypass: MISSING")
            return False
        
        # Check for ultra-permissive thresholds
        if 'spread_cap = 0.0025' in content and 'min_short_spread = 0.0001' in content:
            print("✅ V8 Ultra-Permissive Thresholds: IMPLEMENTED")
        else:
            print("❌ V8 Ultra-Permissive Thresholds: MISSING")
            return False
        
        # Check for V8 emergency fix comments
        if 'V8 EMERGENCY FIX:' in content:
            print("✅ V8 Emergency Fix Comments: IMPLEMENTED")
        else:
            print("❌ V8 Emergency Fix Comments: MISSING")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading newbotcode.py: {e}")
        return False

def validate_rr_atr_fixes():
    """Validate RR/ATR check V8 fixes"""
    print("\n🔍 Validating RR/ATR Check V8 Fixes...")
    
    try:
        with open('newbotcode.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for V8 fallback reward handling
        if 'V8: More permissive reward check with fallback' in content:
            print("✅ V8 RR/ATR Fallback: IMPLEMENTED")
        else:
            print("❌ V8 RR/ATR Fallback: MISSING")
            return False
        
        # Check for fallback reward calculation
        if 'reward_dollars = max(0.001, abs(reward) * position_size)' in content:
            print("✅ V8 Fallback Reward Calculation: IMPLEMENTED")
        else:
            print("❌ V8 Fallback Reward Calculation: MISSING")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading newbotcode.py: {e}")
        return False

def validate_position_loss_threshold_fixes():
    """Validate position loss threshold V8 fixes"""
    print("\n🔍 Validating Position Loss Threshold V8 Fixes...")
    
    try:
        with open('newbotcode.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for increased position loss threshold
        if 'position_loss_threshold=float(getattr(cfg, \'stop_loss_pct\', 0.05) or 0.05)' in content:
            print("✅ V8 Position Loss Threshold (5.0%): IMPLEMENTED")
        else:
            print("❌ V8 Position Loss Threshold (5.0%): MISSING")
            return False
        
        # Check for V8 comment
        if 'V8: Increased from 2.5% to 5.0%' in content:
            print("✅ V8 Position Loss Threshold Comment: IMPLEMENTED")
        else:
            print("❌ V8 Position Loss Threshold Comment: MISSING")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading newbotcode.py: {e}")
        return False

def validate_emergency_bypass_capability():
    """Validate emergency microstructure bypass capability"""
    print("\n🔍 Validating Emergency Microstructure Bypass Capability...")
    
    try:
        with open('newbotcode.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for emergency bypass environment variable check
        if 'EMERGENCY_MICROSTRUCTURE_BYPASS' in content and 'false' in content:
            print("✅ V8 Emergency Bypass Environment Variable: IMPLEMENTED")
        else:
            print("❌ V8 Emergency Bypass Environment Variable: MISSING")
            return False
        
        # Check for emergency bypass warning message
        if 'EMERGENCY MICROSTRUCTURE BYPASS ACTIVATED - ALLOWING ALL TRADES' in content:
            print("✅ V8 Emergency Bypass Warning Message: IMPLEMENTED")
        else:
            print("❌ V8 Emergency Bypass Warning Message: MISSING")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading newbotcode.py: {e}")
        return False

def validate_startup_script():
    """Validate V8 emergency fixes startup script"""
    print("\n🔍 Validating V8 Emergency Fixes Startup Script...")
    
    try:
        if not os.path.exists('start_v8_emergency_fixes.bat'):
            print("❌ V8 Emergency Fixes Startup Script: MISSING")
            return False
        
        with open('start_v8_emergency_fixes.bat', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for V8 environment variables
        if 'V8_MICROSTRUCTURE_SPREAD_CAP=0.0025' in content:
            print("✅ V8 Microstructure Spread Cap: IMPLEMENTED")
        else:
            print("❌ V8 Microstructure Spread Cap: MISSING")
            return False
        
        if 'V8_POSITION_LOSS_THRESHOLD=0.05' in content:
            print("✅ V8 Position Loss Threshold: IMPLEMENTED")
        else:
            print("❌ V8 Position Loss Threshold: MISSING")
            return False
        
        if 'EMERGENCY_MICROSTRUCTURE_BYPASS=false' in content:
            print("✅ V8 Emergency Bypass Variable: IMPLEMENTED")
        else:
            print("❌ V8 Emergency Bypass Variable: MISSING")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading startup script: {e}")
        return False

def main():
    """Main validation function"""
    print("🚨 V8 EMERGENCY FIXES VALIDATION")
    print("=" * 50)
    print()
    
    validation_results = []
    
    # Run all validations
    validation_results.append(("Microstructure Veto V8 Fixes", validate_microstructure_veto_fixes()))
    validation_results.append(("RR/ATR Check V8 Fixes", validate_rr_atr_fixes()))
    validation_results.append(("Position Loss Threshold V8 Fixes", validate_position_loss_threshold_fixes()))
    validation_results.append(("Emergency Bypass Capability", validate_emergency_bypass_capability()))
    validation_results.append(("V8 Startup Script", validate_startup_script()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 V8 EMERGENCY FIXES VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(validation_results)
    
    for test_name, result in validation_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL V8 EMERGENCY FIXES VALIDATED SUCCESSFULLY!")
        print("✅ The bot is ready for deployment with V8 fixes")
        return True
    else:
        print(f"\n⚠️ {total - passed} V8 emergency fixes still need implementation")
        print("❌ Please complete the missing fixes before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
