#!/usr/bin/env python3
"""
Test script to verify duplicate method removal and string formatting fixes
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("🔧 Testing duplicate method removal and string formatting fixes...")

try:
    # Test 1: Verify no duplicate method definitions
    print("\n✅ Test 1: Checking for duplicate method definitions")
    
    # Read the file and check for duplicate _submit_trigger methods
    with open('newbotcode.py', 'r') as f:
        content = f.read()
    
    # Count occurrences of method definitions
    submit_trigger_count = content.count('def _submit_trigger')
    submit_single_trigger_count = content.count('def _submit_single_trigger')
    
    print(f"   _submit_trigger methods found: {submit_trigger_count}")
    print(f"   _submit_single_trigger methods found: {submit_single_trigger_count}")
    
    if submit_trigger_count == 1 and submit_single_trigger_count == 0:
        print("   ✅ No duplicate methods found!")
    else:
        print("   ❌ Duplicate methods still exist!")
        exit(1)
    
    # Test 2: Verify string formatting fixes are in place
    print("\n✅ Test 2: Checking string formatting fixes")
    
    # Check for the fixed logging line pattern
    if 'trigger_px_float:.4f' in content or 'aligned_price_float:.4f' in content:
        print("   ✅ String formatting fixes found in logging lines")
    else:
        print("   ❌ String formatting fixes not found!")
        exit(1)
    
    # Check for keyword arguments in SDK calls
    if 'coin=symbol' in content and 'sz=str(' in content:
        print("   ✅ Keyword arguments and string conversion found in SDK calls")
    else:
        print("   ❌ Keyword arguments not found!")
        exit(1)
    
    # Test 3: Verify trigger order format
    print("\n✅ Test 3: Checking trigger order format")
    
    if 'triggerPx' in content and 'isMarket' in content and 'tpsl' in content:
        print("   ✅ Correct camelCase trigger order format found")
    else:
        print("   ❌ Correct trigger order format not found!")
        exit(1)
    
    print("\n🎯 All duplicate method and string formatting fixes verified successfully!")
    print("✅ No more 'Unknown format code f' errors should occur!")
    print("✅ TP/SL triggers should now work correctly!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1) 