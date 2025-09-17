"""
Decimal Boundary Guard - Fast patch for decimal/float mixing elimination
"""

import logging
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from typing import Any, Union, Dict, List
import functools

# Set import-time context (single source of truth)
C = getcontext()
C.prec = 10
C.rounding = ROUND_HALF_EVEN

logger = logging.getLogger(__name__)
logger.info("🔢 DECIMAL_BOUNDARY_GUARD_ACTIVE: context=ROUND_HALF_EVEN, precision=10")

def decimal_boundary_guard(func):
    """
    Decorator that ensures all numeric inputs are converted to Decimal
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert all numeric args to Decimal
        decimal_args = []
        for arg in args:
            if isinstance(arg, (int, float, str)):
                try:
                    decimal_args.append(Decimal(str(arg)))
                except:
                    decimal_args.append(arg)
            else:
                decimal_args.append(arg)
        
        # Convert all numeric kwargs to Decimal
        decimal_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (int, float, str)):
                try:
                    decimal_kwargs[key] = Decimal(str(value))
                except:
                    decimal_kwargs[key] = value
            else:
                decimal_kwargs[key] = value
        
        return func(*decimal_args, **decimal_kwargs)
    
    return wrapper

def coerce_to_decimal(value: Any) -> Decimal:
    """
    Coerce any value to Decimal safely
    """
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except:
            return Decimal('0')
    else:
        return Decimal('0')

def coerce_dict_to_decimal(data: Dict[str, Any], decimal_fields: List[str]) -> Dict[str, Any]:
    """
    Coerce specific fields in a dictionary to Decimal
    """
    result = data.copy()
    for field in decimal_fields:
        if field in result:
            result[field] = coerce_to_decimal(result[field])
    return result

def coerce_price_data_to_decimal(price_data: Any) -> Decimal:
    """
    Coerce price data to Decimal, handling various formats
    """
    if isinstance(price_data, dict):
        # Handle dictionary with price fields
        for key in ['price', 'close', 'c', 'value']:
            if key in price_data:
                return coerce_to_decimal(price_data[key])
        return Decimal('0')
    elif isinstance(price_data, (list, tuple)):
        # Handle list/tuple of prices
        if len(price_data) > 0:
            return coerce_to_decimal(price_data[0])
        return Decimal('0')
    else:
        return coerce_to_decimal(price_data)

def coerce_position_data_to_decimal(position_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce position data to use Decimal values
    """
    decimal_fields = ['size', 'szi', 'entry_price', 'entryPx', 'unrealizedPnl', 'markPx', 'positionValue']
    return coerce_dict_to_decimal(position_data, decimal_fields)

def coerce_account_data_to_decimal(account_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce account data to use Decimal values
    """
    decimal_fields = ['accountValue', 'account_value', 'freeCollateral', 'withdrawable', 'totalMarginUsed']
    return coerce_dict_to_decimal(account_data, decimal_fields)

def coerce_order_data_to_decimal(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce order data to use Decimal values
    """
    decimal_fields = ['size', 'price', 'take_profit', 'stop_loss', 'quantity', 'amount']
    return coerce_dict_to_decimal(order_data, decimal_fields)

# Global coercion functions for easy import
def safe_float(value: Any) -> Decimal:
    """
    Safe replacement for float() that returns Decimal
    """
    return coerce_to_decimal(value)

def safe_float_operation(operation: str, a: Any, b: Any = None) -> Decimal:
    """
    Safe replacement for float arithmetic operations
    """
    decimal_a = coerce_to_decimal(a)
    
    if b is not None:
        decimal_b = coerce_to_decimal(b)
        
        if operation == '+':
            return decimal_a + decimal_b
        elif operation == '-':
            return decimal_a - decimal_b
        elif operation == '*':
            return decimal_a * decimal_b
        elif operation == '/':
            return decimal_a / decimal_b if decimal_b != 0 else Decimal('0')
        elif operation == '//':
            return decimal_a // decimal_b if decimal_b != 0 else Decimal('0')
        elif operation == '%':
            return decimal_a % decimal_b if decimal_b != 0 else Decimal('0')
        elif operation == '**':
            return decimal_a ** decimal_b
        else:
            logger.error(f"❌ Unknown operation: {operation}")
            return Decimal('0')
    else:
        # Unary operations
        if operation == 'abs':
            return abs(decimal_a)
        elif operation == 'neg':
            return -decimal_a
        else:
            logger.error(f"❌ Unknown unary operation: {operation}")
            return decimal_a

# Demo function
def demo_decimal_boundary_guard():
    """Demo the decimal boundary guard"""
    print("🔢 Decimal Boundary Guard Demo")
    print("=" * 50)
    
    # Test various conversions
    test_values = [
        123.456,
        "789.012",
        42,
        {"price": 1.234, "close": 1.235},
        [1.1, 1.2, 1.3],
        None
    ]
    
    for value in test_values:
        try:
            result = safe_float(value)
            print(f"✅ {value} -> {result} ({type(result).__name__})")
        except Exception as e:
            print(f"❌ {value} -> Error: {e}")
    
    # Test arithmetic operations
    print(f"\n🔢 Arithmetic Operations:")
    print(f"✅ 1.5 + 2.3 = {safe_float_operation('+', 1.5, 2.3)}")
    print(f"✅ 5.0 - 1.2 = {safe_float_operation('-', 5.0, 1.2)}")
    print(f"✅ 2.5 * 3.0 = {safe_float_operation('*', 2.5, 3.0)}")
    print(f"✅ 10.0 / 2.0 = {safe_float_operation('/', 10.0, 2.0)}")
    
    # Test position data conversion
    position_data = {
        'size': 100.5,
        'entry_price': 1.234,
        'unrealizedPnl': -5.67,
        'coin': 'XRP'
    }
    
    converted = coerce_position_data_to_decimal(position_data)
    print(f"\n🔢 Position Data Conversion:")
    for key, value in converted.items():
        print(f"✅ {key}: {value} ({type(value).__name__})")
    
    print(f"\n✅ Decimal Boundary Guard Demo Complete")

if __name__ == "__main__":
    demo_decimal_boundary_guard()
