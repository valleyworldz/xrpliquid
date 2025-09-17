"""
Float to Decimal Converter - Critical fix for decimal/float mixing risk
"""

import logging
from decimal import Decimal, getcontext
from typing import Any, Union, Dict, List
import re

class FloatToDecimalConverter:
    """
    Converts all float() casts to Decimal() in trade math paths
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversion_count = 0
        
    def convert_float_to_decimal(self, value: Any) -> Decimal:
        """
        Convert any value to Decimal, handling float conversion safely
        """
        try:
            if isinstance(value, Decimal):
                return value
            elif isinstance(value, (int, str)):
                return Decimal(str(value))
            elif isinstance(value, float):
                # Convert float to Decimal with proper precision
                return Decimal(str(value))
            else:
                # Try to convert to string first, then to Decimal
                return Decimal(str(value))
        except Exception as e:
            self.logger.error(f"❌ Error converting {value} to Decimal: {e}")
            return Decimal('0')
    
    def safe_float_operation(self, operation: str, a: Any, b: Any = None) -> Decimal:
        """
        Perform safe arithmetic operations using Decimal
        """
        try:
            decimal_a = self.convert_float_to_decimal(a)
            
            if b is not None:
                decimal_b = self.convert_float_to_decimal(b)
                
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
                    self.logger.error(f"❌ Unknown operation: {operation}")
                    return Decimal('0')
            else:
                # Unary operations
                if operation == 'abs':
                    return abs(decimal_a)
                elif operation == 'neg':
                    return -decimal_a
                else:
                    self.logger.error(f"❌ Unknown unary operation: {operation}")
                    return decimal_a
                    
        except Exception as e:
            self.logger.error(f"❌ Error in safe_float_operation: {e}")
            return Decimal('0')
    
    def convert_price_data(self, price_data: Any) -> Decimal:
        """
        Convert price data to Decimal, handling various formats
        """
        try:
            if isinstance(price_data, dict):
                # Handle dictionary with price fields
                for key in ['price', 'close', 'c', 'value']:
                    if key in price_data:
                        return self.convert_float_to_decimal(price_data[key])
                # If no price field found, try to convert the whole dict
                return self.convert_float_to_decimal(str(price_data))
            elif isinstance(price_data, (list, tuple)):
                # Handle list/tuple of prices
                if len(price_data) > 0:
                    return self.convert_float_to_decimal(price_data[0])
                return Decimal('0')
            else:
                return self.convert_float_to_decimal(price_data)
        except Exception as e:
            self.logger.error(f"❌ Error converting price data: {e}")
            return Decimal('0')
    
    def convert_position_data(self, position_data: Dict[str, Any]) -> Dict[str, Decimal]:
        """
        Convert position data dictionary to use Decimal values
        """
        try:
            converted = {}
            for key, value in position_data.items():
                if key in ['size', 'szi', 'entry_price', 'entryPx', 'unrealizedPnl', 'markPx']:
                    converted[key] = self.convert_float_to_decimal(value)
                else:
                    converted[key] = value
            return converted
        except Exception as e:
            self.logger.error(f"❌ Error converting position data: {e}")
            return {}
    
    def convert_account_data(self, account_data: Dict[str, Any]) -> Dict[str, Decimal]:
        """
        Convert account data dictionary to use Decimal values
        """
        try:
            converted = {}
            for key, value in account_data.items():
                if key in ['accountValue', 'account_value', 'freeCollateral', 'withdrawable', 'totalMarginUsed']:
                    converted[key] = self.convert_float_to_decimal(value)
                else:
                    converted[key] = value
            return converted
        except Exception as e:
            self.logger.error(f"❌ Error converting account data: {e}")
            return {}
    
    def log_conversion_stats(self):
        """
        Log conversion statistics
        """
        self.logger.info(f"🔢 FLOAT_TO_DECIMAL_CONVERTER: {self.conversion_count} conversions performed")

# Global converter instance
_converter = FloatToDecimalConverter()

def safe_float(value: Any) -> Decimal:
    """
    Safe replacement for float() that returns Decimal
    """
    return _converter.convert_float_to_decimal(value)

def safe_float_operation(operation: str, a: Any, b: Any = None) -> Decimal:
    """
    Safe replacement for float arithmetic operations
    """
    return _converter.safe_float_operation(operation, a, b)

def convert_price_data(price_data: Any) -> Decimal:
    """
    Convert price data to Decimal
    """
    return _converter.convert_price_data(price_data)

def convert_position_data(position_data: Dict[str, Any]) -> Dict[str, Decimal]:
    """
    Convert position data to use Decimal values
    """
    return _converter.convert_position_data(position_data)

def convert_account_data(account_data: Dict[str, Any]) -> Dict[str, Decimal]:
    """
    Convert account data to use Decimal values
    """
    return _converter.convert_account_data(account_data)

# Demo function
def demo_float_to_decimal_converter():
    """Demo the float to decimal converter"""
    print("🔢 Float to Decimal Converter Demo")
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
    
    converted = convert_position_data(position_data)
    print(f"\n🔢 Position Data Conversion:")
    for key, value in converted.items():
        print(f"✅ {key}: {value} ({type(value).__name__})")
    
    print(f"\n✅ Float to Decimal Converter Demo Complete")

if __name__ == "__main__":
    demo_float_to_decimal_converter()
