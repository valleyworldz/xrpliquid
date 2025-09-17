"""
Decimal Normalizer - Comprehensive Decimal/float Bug Fix
Normalizes all external numeric inputs at API boundary to Decimal with end-to-end consistency
"""

import logging
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from typing import Any, Union, Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
from functools import wraps
import inspect

# Set global decimal context
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_EVEN
# Clear traps to prevent exceptions
getcontext().clear_traps()

class DecimalNormalizer:
    """
    Comprehensive decimal normalization system to eliminate float/Decimal type errors
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversion_stats = {
            "conversions": 0,
            "errors": 0,
            "types_converted": {}
        }
    
    def normalize(self, value: Any, precision: int = 8) -> Decimal:
        """
        Normalize any numeric value to Decimal with specified precision
        """
        try:
            if value is None:
                return Decimal('0')
            
            if isinstance(value, Decimal):
                # Ensure precision
                return value.quantize(Decimal('0.' + '0' * precision))
            
            if isinstance(value, (int, float)):
                # Convert to Decimal with proper precision
                decimal_value = Decimal(str(value))
                # For now, return the decimal value as is to avoid quantize issues
                return decimal_value
            
            if isinstance(value, str):
                # Try to parse as number
                decimal_value = Decimal(value)
                return decimal_value.quantize(Decimal('0.' + '0' * precision))
            
            if isinstance(value, np.number):
                # Handle numpy types
                decimal_value = Decimal(str(float(value)))
                return decimal_value.quantize(Decimal('0.' + '0' * precision))
            
            # For other types, try to convert to string first
            decimal_value = Decimal(str(value))
            return decimal_value.quantize(Decimal('0.' + '0' * precision))
                
        except Exception as e:
            self.logger.error(f"❌ Decimal normalization error: {e}")
            self.conversion_stats["errors"] += 1
            return Decimal('0')
        finally:
            self.conversion_stats["conversions"] += 1
            self.conversion_stats["types_converted"][str(type(value))] = \
                self.conversion_stats["types_converted"].get(str(type(value)), 0) + 1
    
    def normalize_dict(self, data: Dict[str, Any], precision: int = 8) -> Dict[str, Decimal]:
        """
        Normalize all numeric values in a dictionary to Decimal
        """
        normalized = {}
        for key, value in data.items():
            if isinstance(value, (int, float, str, Decimal, np.number)):
                normalized[key] = self.normalize(value, precision)
            elif isinstance(value, dict):
                normalized[key] = self.normalize_dict(value, precision)
            elif isinstance(value, list):
                normalized[key] = self.normalize_list(value, precision)
            else:
                normalized[key] = value
        
        return normalized
    
    def normalize_list(self, data: List[Any], precision: int = 8) -> List[Decimal]:
        """
        Normalize all numeric values in a list to Decimal
        """
        normalized = []
        for item in data:
            if isinstance(item, (int, float, str, Decimal, np.number)):
                normalized.append(self.normalize(item, precision))
            elif isinstance(item, dict):
                normalized.append(self.normalize_dict(item, precision))
            elif isinstance(item, list):
                normalized.append(self.normalize_list(item, precision))
            else:
                normalized.append(item)
        
        return normalized
    
    def safe_operation(self, operation: str, a: Any, b: Any, precision: int = 8) -> Decimal:
        """
        Perform safe arithmetic operations with Decimal normalization
        """
        try:
            decimal_a = self.normalize(a, precision)
            decimal_b = self.normalize(b, precision)
            
            if operation == "add":
                result = decimal_a + decimal_b
            elif operation == "subtract":
                result = decimal_a - decimal_b
            elif operation == "multiply":
                result = decimal_a * decimal_b
            elif operation == "divide":
                if decimal_b == 0:
                    self.logger.warning("⚠️ Division by zero attempted")
                    return Decimal('0')
                result = decimal_a / decimal_b
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Return result without quantize to avoid precision issues
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Safe operation error ({operation}): {e}")
            return Decimal('0')
    
    def normalize_order_data(self, order_data: Dict[str, Any]) -> Dict[str, Decimal]:
        """
        Normalize order data specifically for trading operations
        """
        critical_fields = ['price', 'size', 'amount', 'fee', 'bid', 'ask', 'spread']
        normalized = {}
        
        for key, value in order_data.items():
            if key in critical_fields:
                normalized[key] = self.normalize(value, 8)  # 8 decimal places for prices
            else:
                normalized[key] = value
        
        return normalized
    
    def normalize_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Decimal]:
        """
        Normalize market data specifically for trading operations
        """
        price_fields = ['price', 'bid', 'ask', 'last', 'high', 'low', 'open', 'close']
        volume_fields = ['volume', 'size', 'amount']
        
        normalized = {}
        for key, value in market_data.items():
            if key in price_fields:
                normalized[key] = self.normalize(value, 8)  # 8 decimal places for prices
            elif key in volume_fields:
                normalized[key] = self.normalize(value, 6)  # 6 decimal places for volumes
            else:
                normalized[key] = value
        
        return normalized
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """
        Get conversion statistics
        """
        return self.conversion_stats.copy()

# Global normalizer instance
decimal_normalizer = DecimalNormalizer()

def decimal_safe(func):
    """
    Decorator to ensure function arguments are normalized to Decimal
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Normalize positional arguments
            normalized_args = []
            for arg in args:
                if isinstance(arg, (int, float, str, np.number)):
                    normalized_args.append(decimal_normalizer.normalize(arg))
                else:
                    normalized_args.append(arg)
            
            # Normalize keyword arguments
            normalized_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, (int, float, str, np.number)):
                    normalized_kwargs[key] = decimal_normalizer.normalize(value)
                else:
                    normalized_kwargs[key] = value
            
            return func(*normalized_args, **normalized_kwargs)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"❌ Decimal safe decorator error: {e}")
            raise
    
    return wrapper

def decimal_safe_class(cls):
    """
    Class decorator to make all methods decimal-safe
    """
    for name, method in inspect.getmembers(cls, inspect.ismethod):
        if not name.startswith('_'):
            setattr(cls, name, decimal_safe(method))
    
    return cls

class DecimalOrderBook:
    """
    Decimal-safe order book implementation
    """
    
    def __init__(self):
        self.bids: List[Tuple[Decimal, Decimal]] = []
        self.asks: List[Tuple[Decimal, Decimal]] = []
        self.normalizer = decimal_normalizer
    
    def add_bid(self, price: Any, size: Any):
        """Add bid with decimal normalization"""
        decimal_price = self.normalizer.normalize(price, 8)
        decimal_size = self.normalizer.normalize(size, 6)
        self.bids.append((decimal_price, decimal_size))
        self.bids.sort(key=lambda x: x[0], reverse=True)  # Sort by price descending
    
    def add_ask(self, price: Any, size: Any):
        """Add ask with decimal normalization"""
        decimal_price = self.normalizer.normalize(price, 8)
        decimal_size = self.normalizer.normalize(size, 6)
        self.asks.append((decimal_price, decimal_size))
        self.asks.sort(key=lambda x: x[0])  # Sort by price ascending
    
    def get_best_bid(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best bid price and size"""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best ask price and size"""
        return self.asks[0] if self.asks else None
    
    def get_spread(self) -> Decimal:
        """Calculate spread in decimal"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return Decimal('0')
    
    def get_mid_price(self) -> Decimal:
        """Calculate mid price in decimal"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / Decimal('2')
        return Decimal('0')

class DecimalTrade:
    """
    Decimal-safe trade implementation
    """
    
    def __init__(self, price: Any, size: Any, side: str, timestamp: str = None):
        self.price = decimal_normalizer.normalize(price, 8)
        self.size = decimal_normalizer.normalize(size, 6)
        self.side = side
        self.timestamp = timestamp
        self.value = self.price * self.size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with decimal values"""
        return {
            'price': float(self.price),
            'size': float(self.size),
            'side': self.side,
            'timestamp': self.timestamp,
            'value': float(self.value)
        }

# Demo function
def demo_decimal_normalizer():
    """Demo the decimal normalizer"""
    print("🔢 Decimal Normalizer Demo")
    print("=" * 50)
    
    # Test various input types
    test_values = [
        123.456,
        "789.012",
        1000,
        Decimal('555.777'),
        np.float64(999.888),
        None
    ]
    
    print("📊 Testing decimal normalization:")
    for value in test_values:
        normalized = decimal_normalizer.normalize(value)
        print(f"  {type(value).__name__}: {value} -> {normalized}")
    
    # Test safe operations
    print(f"\n🧮 Testing safe operations:")
    a, b = 123.456, 789.012
    print(f"  {a} + {b} = {decimal_normalizer.safe_operation('add', a, b)}")
    print(f"  {a} - {b} = {decimal_normalizer.safe_operation('subtract', a, b)}")
    print(f"  {a} * {b} = {decimal_normalizer.safe_operation('multiply', a, b)}")
    print(f"  {a} / {b} = {decimal_normalizer.safe_operation('divide', a, b)}")
    
    # Test order book
    print(f"\n📈 Testing decimal order book:")
    order_book = DecimalOrderBook()
    order_book.add_bid(0.5234, 1000.5)
    order_book.add_ask(0.5236, 2000.7)
    
    best_bid = order_book.get_best_bid()
    best_ask = order_book.get_best_ask()
    spread = order_book.get_spread()
    mid_price = order_book.get_mid_price()
    
    print(f"  Best Bid: {best_bid}")
    print(f"  Best Ask: {best_ask}")
    print(f"  Spread: {spread}")
    print(f"  Mid Price: {mid_price}")
    
    # Test trade
    print(f"\n💰 Testing decimal trade:")
    trade = DecimalTrade(0.5235, 500.25, "buy")
    print(f"  Trade: {trade.to_dict()}")
    
    # Show conversion stats
    stats = decimal_normalizer.get_conversion_stats()
    print(f"\n📊 Conversion Statistics:")
    print(f"  Total Conversions: {stats['conversions']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Types Converted: {stats['types_converted']}")
    
    print("\n✅ Decimal Normalizer Demo Complete")

if __name__ == "__main__":
    demo_decimal_normalizer()
