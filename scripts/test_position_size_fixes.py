#!/usr/bin/env python3
"""
Test script to verify position size fixes are in place
"""

def test_position_size_fixes():
    """Test that the position size fixes are in the code"""
    print("🔧 Testing position size fixes...")
    
    try:
        # Read the bot code and check for our fixes
        with open('newbotcode.py', 'r') as f:
            content = f.read()
        
        # Test 1: Check for ATR scaling fix
        if 'raw_size = int(raw_size * scaling_factor)' in content:
            print("✅ Test 1: ATR scaling variable fix found")
        else:
            print("❌ Test 1: ATR scaling variable fix NOT found")
            return False
        
        # Test 2: Check for minimum order value fix
        if 'floor_min_order_value = 4' in content:
            print("✅ Test 2: Minimum order value fix found")
        else:
            print("❌ Test 2: Minimum order value fix NOT found")
            return False
        
        # Test 3: Check for raw_size minimum enforcement
        if 'if raw_size < floor_min_order_value:' in content:
            print("✅ Test 3: Raw size minimum enforcement found")
        else:
            print("❌ Test 3: Raw size minimum enforcement NOT found")
            return False
        
        # Test 4: Check for dynamic threshold configuration
        if 'dynamic_threshold_enabled: bool = True' in content:
            print("✅ Test 4: Dynamic threshold configuration found")
        else:
            print("❌ Test 4: Dynamic threshold configuration NOT found")
            return False
        
        # Test 5: Check for ATR scaling configuration
        if 'atr_scaled_position_enabled: bool = True' in content:
            print("✅ Test 5: ATR scaling configuration found")
        else:
            print("❌ Test 5: ATR scaling configuration NOT found")
            return False
        
        print("✅ All position size fixes are in place!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_position_size_fixes()
    if success:
        print("🎉 Position size fixes are correctly implemented!")
    else:
        print("❌ Position size fixes need attention!")
        exit(1) 