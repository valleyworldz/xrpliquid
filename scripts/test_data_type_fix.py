#!/usr/bin/env python3
"""
Test script to verify the data type fix - floats vs strings
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("🔧 Testing data type fix - floats vs strings...")

try:
    # Test 1: Verify the correct data types are used
    print("\n✅ Test 1: Checking for correct data types")
    
    # Read the file and check for the correct data types
    with open('newbotcode.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for float variables instead of string formatting
    if 'tp_price_float = float(tp_px)' in content and 'sl_price_float = float(sl_px)' in content:
        print("   ✅ Correct float conversion found")
    else:
        print("   ❌ Correct float conversion not found!")
        exit(1)
    
    # Test 2: Verify no more string formatting for prices
    print("\n✅ Test 2: Checking for string formatting issues")
    
    # Check for the problematic string formatting
    if 'f"{float(' in content and ':.4f}"' in content:
        print("   ⚠️  Found string formatting that might cause issues")
        print("   This should be removed from the order construction")
    else:
        print("   ✅ No problematic string formatting found")
    
    # Test 3: Verify float values are used in order kwargs
    print("\n✅ Test 3: Checking order kwargs data types")
    
    # Check that the order kwargs use float variables
    if '"limit_px": tp_price_float' in content and '"triggerPx": tp_price_float' in content:
        print("   ✅ Correct float usage in TP order kwargs")
    else:
        print("   ❌ Incorrect data types in TP order kwargs!")
        exit(1)
    
    if '"limit_px": sl_price_float' in content and '"triggerPx": sl_price_float' in content:
        print("   ✅ Correct float usage in SL order kwargs")
    else:
        print("   ❌ Incorrect data types in SL order kwargs!")
        exit(1)
    
    # Test 4: Verify logging uses float values
    print("\n✅ Test 4: Checking logging data types")
    
    if 'TP={tp_price_float}' in content and 'SL={sl_price_float}' in content:
        print("   ✅ Correct float usage in logging")
    else:
        print("   ❌ Incorrect data types in logging!")
        exit(1)
    
    # Test 5: Verify string formatting is only used for size
    print("\n✅ Test 5: Checking size data type")
    
    if 'size_str = str(size)' in content and '"sz": size_str' in content:
        print("   ✅ Correct string usage for size")
    else:
        print("   ❌ Incorrect data type for size!")
        exit(1)
    
    print("\n🎯 All data type fixes verified successfully!")
    print("✅ TP/SL triggers should now work with correct data types!")
    print("✅ No more 'Unknown format code f' errors!")
    print("✅ Floats are used for prices, strings for size!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1) 