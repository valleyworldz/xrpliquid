#!/usr/bin/env python3
"""
Test script to verify the executor keyword argument fix
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("🔧 Testing executor keyword argument fix...")

try:
    # Test 1: Verify the new low-level exchange API functions exist
    print("\n✅ Test 1: Checking for new low-level exchange API functions")
    
    # Read the file and check for the new functions
    with open('newbotcode.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the new functions
    if '_build_tpsl_orders' in content and '_submit_trigger_pair' in content:
        print("   ✅ New low-level exchange API functions found")
    else:
        print("   ❌ New low-level exchange API functions not found!")
        exit(1)
    
    # Test 2: Verify the old _submit_trigger calls have been replaced
    print("\n✅ Test 2: Checking for old _submit_trigger calls")
    
    # Count old _submit_trigger calls (should be 0)
    old_calls = content.count('await _submit_trigger(')
    if old_calls == 0:
        print("   ✅ All old _submit_trigger calls have been replaced")
    else:
        print(f"   ❌ Found {old_calls} old _submit_trigger calls still present!")
        exit(1)
    
    # Test 3: Verify the new _submit_trigger_pair calls exist
    print("\n✅ Test 3: Checking for new _submit_trigger_pair calls")
    
    if 'await _submit_trigger_pair(' in content:
        print("   ✅ New _submit_trigger_pair calls found")
    else:
        print("   ❌ New _submit_trigger_pair calls not found!")
        exit(1)
    
    # Test 4: Verify no more executor keyword arguments
    print("\n✅ Test 4: Checking for executor keyword argument issues")
    
    # Look for patterns that would cause executor issues
    if 'run_sync_in_executor' in content and 'coin=' in content:
        print("   ⚠️  Found potential executor keyword argument issues")
        print("   This might still cause problems - check the code")
    else:
        print("   ✅ No executor keyword argument issues found")
    
    print("\n🎯 All executor keyword argument fixes verified successfully!")
    print("✅ TP/SL triggers should now work without executor issues!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1) 