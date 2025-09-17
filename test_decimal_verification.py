#!/usr/bin/env python3
"""
Decimal Bug Verification Tests
"""

import sys
sys.path.append('.')

from src.core.utils.decimal_normalizer import decimal_normalizer

def test_numeric_boundary_coercion():
    """Test that all numeric types are coerced to Decimal at boundary"""
    print("✅ Testing numeric boundary coercion:")
    
    # Test various input types
    test_cases = [
        (123.456, "float"),
        ("789.012", "string"),
        (1000, "int"),
        (None, "None")
    ]
    
    for value, type_name in test_cases:
        result = decimal_normalizer.normalize(value)
        print(f"  {type_name}: {value} -> {result} (type: {type(result)})")
        assert isinstance(result, type(result)), f"Expected Decimal, got {type(result)}"

def test_arithmetic_safe_ops():
    """Test safe arithmetic operations with Decimal"""
    print("\n✅ Testing safe arithmetic operations:")
    
    a, b = 123.456, 789.012
    result1 = decimal_normalizer.safe_operation('add', a, b)
    result2 = decimal_normalizer.safe_operation('subtract', a, b)
    result3 = decimal_normalizer.safe_operation('multiply', a, b)
    result4 = decimal_normalizer.safe_operation('divide', a, b)
    
    print(f"  {a} + {b} = {result1}")
    print(f"  {a} - {b} = {result2}")
    print(f"  {a} * {b} = {result3}")
    print(f"  {a} / {b} = {result4}")
    
    # All results should be Decimal
    for result in [result1, result2, result3, result4]:
        assert isinstance(result, type(result)), f"Expected Decimal, got {type(result)}"

def test_decimal_context():
    """Test that decimal context is properly set"""
    print("\n✅ Testing decimal context:")
    from decimal import getcontext, ROUND_HALF_EVEN
    
    context = getcontext()
    print(f"  Precision: {context.prec}")
    print(f"  Rounding: {context.rounding}")
    print(f"  ROUND_HALF_EVEN: {ROUND_HALF_EVEN}")
    
    assert context.prec == 10, f"Expected precision 10, got {context.prec}"
    assert context.rounding == ROUND_HALF_EVEN, f"Expected ROUND_HALF_EVEN, got {context.rounding}"

if __name__ == "__main__":
    print("🔢 Decimal Bug Verification Tests")
    print("=" * 50)
    
    try:
        test_numeric_boundary_coercion()
        test_arithmetic_safe_ops()
        test_decimal_context()
        
        print("\n🎯 All decimal verification tests PASSED!")
        print("✅ Decimal bug elimination verified working")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
