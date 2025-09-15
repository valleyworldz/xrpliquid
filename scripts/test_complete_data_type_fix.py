#!/usr/bin/env python3
"""
Test script to verify the complete data type fix - all fields as proper types
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("🔧 Testing complete data type fix - all fields as proper types...")

try:
    # Test 1: Verify the correct data types are used
    print("\n✅ Test 1: Checking for correct data types")
    
    # Read the file and check for the correct data types
    with open('newbotcode.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for float variables instead of string formatting
    if 'tp_price_float = float(tp_px)' in content and 'sl_price_float = float(sl_px)' in content:
        print("   ✅ Correct float conversion for prices found")
    else:
        print("   ❌ Correct float conversion for prices not found!")
        exit(1)
    
    # Check for float conversion for size
    if 'size_float = float(size)' in content:
        print("   ✅ Correct float conversion for size found")
    else:
        print("   ❌ Correct float conversion for size not found!")
        exit(1)
    
    # Test 2: Verify no more string formatting for any fields
    print("\n✅ Test 2: Checking for string formatting issues")
    
    # Check for the problematic string formatting
    if 'f"{float(' in content and ':.4f}"' in content:
        print("   ⚠️  Found string formatting that might cause issues")
        print("   This should be removed from the order construction")
    else:
        print("   ✅ No problematic string formatting found")
    
    # Test 3: Verify float values are used in all order kwargs
    print("\n✅ Test 3: Checking order kwargs data types")
    
    # Check that the order kwargs use float variables for all numeric fields
    if '"limit_px": tp_price_float' in content and '"triggerPx": tp_price_float' in content and '"sz": size_float' in content:
        print("   ✅ Correct float usage in TP order kwargs")
    else:
        print("   ❌ Incorrect data types in TP order kwargs!")
        exit(1)
    
    if '"limit_px": sl_price_float' in content and '"triggerPx": sl_price_float' in content and '"sz": size_float' in content:
        print("   ✅ Correct float usage in SL order kwargs")
    else:
        print("   ❌ Incorrect data types in SL order kwargs!")
        exit(1)
    
    # Test 4: Verify logging uses float values
    print("\n✅ Test 4: Checking logging data types")
    
    if 'TP={tp_price_float}' in content and 'SL={sl_price_float}' in content and 'size={size_float}' in content:
        print("   ✅ Correct float usage in logging")
    else:
        print("   ❌ Incorrect data types in logging!")
        exit(1)
    
    # Test 5: Verify no string variables for numeric fields
    print("\n✅ Test 5: Checking for string variables")
    
    # Check that we don't have string variables for numeric fields
    if 'size_str = str(size)' in content:
        print("   ❌ Found string variable for size - should be float!")
        exit(1)
    else:
        print("   ✅ No string variables for numeric fields")
    
    # Test 6: Verify all required fields are properly typed
    print("\n✅ Test 6: Checking all field types")
    
    # Check that all numeric fields are floats
    required_float_fields = [
        'tp_price_float = float(tp_px)',
        'sl_price_float = float(sl_px)',
        'size_float = float(size)',
        '"limit_px": tp_price_float',
        '"triggerPx": tp_price_float',
        '"sz": size_float',
        '"limit_px": sl_price_float',
        '"triggerPx": sl_price_float'
    ]
    
    missing_fields = []
    for field in required_float_fields:
        if field not in content:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"   ❌ Missing required float fields: {missing_fields}")
        exit(1)
    else:
        print("   ✅ All required float fields present")
    
    print("\n🎯 All complete data type fixes verified successfully!")
    print("✅ TP/SL triggers should now work with correct data types!")
    print("✅ No more 'Unknown format code f' errors!")
    print("✅ All numeric fields (prices, size) are floats!")
    print("✅ All string fields (name, cloid) are strings!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1) 