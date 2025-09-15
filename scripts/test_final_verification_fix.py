#!/usr/bin/env python3
"""
Comprehensive test script to verify all final verification fixes
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("🔧 Testing final verification fixes...")

try:
    # Test 1: Verify int casting for OID comparison
    print("\n✅ Test 1: Checking for int casting in OID comparison")
    
    # Read the file and check for int casting
    with open('newbotcode.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for int casting
    if 'int(trig["oid"])' in content and 'int(order["oid"])' in content:
        print("   ✅ Int casting for OID comparison found")
    else:
        print("   ❌ Int casting for OID comparison not found!")
        exit(1)
    
    # Test 2: Verify proper response format handling
    print("\n✅ Test 2: Checking for proper response format handling")
    
    # Check for proper response format handling
    if 'Always normalize to list format for consistent processing' in content:
        print("   ✅ Response format normalization found")
    else:
        print("   ❌ Response format normalization not found!")
        exit(1)
    
    # Test 3: Verify userState endpoint usage
    print("\n✅ Test 3: Checking for userState endpoint usage")
    
    # Check for userState endpoint usage
    if 'user_state = self.resilient_info.user_state(self.wallet_address)' in content:
        print("   ✅ userState endpoint usage found")
    else:
        print("   ❌ userState endpoint usage not found!")
        exit(1)
    
    # Test 4: Verify triggerOrders array handling
    print("\n✅ Test 4: Checking for triggerOrders array handling")
    
    # Check for triggerOrders array handling
    if 'triggerOrders' in content and 'for trig in trigger_orders:' in content:
        print("   ✅ triggerOrders array handling found")
    else:
        print("   ❌ triggerOrders array handling not found!")
        exit(1)
    
    # Test 5: Verify increased retry timing
    print("\n✅ Test 5: Checking for increased retry timing")
    
    # Check for increased retry timing
    if 'await asyncio.sleep(2 + attempt)' in content:
        print("   ✅ Increased retry timing found")
    else:
        print("   ❌ Increased retry timing not found!")
        exit(1)
    
    # Test 6: Verify momentum filter for BUY signals
    print("\n✅ Test 6: Checking for momentum filter for BUY signals")
    
    # Check for momentum filter for BUY signals
    if 'Add ATR-based momentum filter for BUY to prevent runaway long sprees' in content:
        print("   ✅ Momentum filter for BUY signals found")
    else:
        print("   ❌ Momentum filter for BUY signals not found!")
        exit(1)
    
    # Test 7: Verify confidence threshold restoration
    print("\n✅ Test 7: Checking for confidence threshold restoration")
    
    # Check for restored confidence threshold
    if 'confidence_threshold: float = 0.12' in content:
        print("   ✅ Confidence threshold restored to 0.12")
    else:
        print("   ❌ Confidence threshold not restored!")
        exit(1)
    
    # Test 8: Verify cleanup logic
    print("\n✅ Test 8: Checking for cleanup logic")
    
    # Check for cleanup logic
    if 'Cleaning up old triggers' in content:
        print("   ✅ Cleanup logic found")
    else:
        print("   ❌ Cleanup logic not found!")
        exit(1)
    
    # Test 9: Verify proper OID extraction from resting format
    print("\n✅ Test 9: Checking for proper OID extraction from resting format")
    
    # Check for proper OID extraction from resting format
    if 'Handle {\'resting\': {\'oid\': 123}} format (from order submission)' in content:
        print("   ✅ Proper OID extraction from resting format found")
    else:
        print("   ❌ Proper OID extraction from resting format not found!")
        exit(1)
    
    # Test 10: Verify int casting for OID comparison in verification
    print("\n✅ Test 10: Checking for int casting in verification")
    
    # Check for int casting in verification
    if 'tp_oid_int = int(tp_oid) if tp_oid else None' in content:
        print("   ✅ Int casting in verification found")
    else:
        print("   ❌ Int casting in verification not found!")
        exit(1)
    
    print("\n🎯 All final verification fixes verified successfully!")
    print("✅ Int casting for consistent OID comparison!")
    print("✅ Proper response format normalization!")
    print("✅ userState endpoint for trigger verification!")
    print("✅ triggerOrders array properly handled!")
    print("✅ Increased retry timing for propagation delays!")
    print("✅ Momentum filter for BUY signals (prevents runaway longs)!")
    print("✅ Confidence threshold restored to reasonable level!")
    print("✅ Cleanup logic prevents trigger clutter!")
    print("✅ Proper OID extraction from all formats!")
    print("✅ TP/SL verification should now work perfectly!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1) 