"""
Unit tests for decimal normalizer to verify decimal bug elimination
"""

import pytest
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
import numpy as np
from src.core.utils.decimal_normalizer import (
    DecimalNormalizer, 
    decimal_normalizer, 
    decimal_safe, 
    DecimalOrderBook, 
    DecimalTrade
)

class TestDecimalNormalizer:
    """Test decimal normalizer functionality"""
    
    def test_numeric_boundary_coercion(self):
        """Test that all numeric types are coerced to Decimal at boundary"""
        normalizer = DecimalNormalizer()
        
        # Test various input types
        test_cases = [
            (123.456, Decimal('123.45600000')),
            ("789.012", Decimal('789.01200000')),
            (1000, Decimal('1000.00000000')),
            (Decimal('555.777'), Decimal('555.77700000')),
            (np.float64(999.888), Decimal('999.88800000')),
            (None, Decimal('0.00000000'))
        ]
        
        for input_val, expected in test_cases:
            result = normalizer.normalize(input_val)
            assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"
            assert result == expected, f"Expected {expected}, got {result}"
    
    def test_arithmetic_safe_ops(self):
        """Test safe arithmetic operations with Decimal"""
        normalizer = DecimalNormalizer()
        
        # Test safe operations
        a, b = 123.456, 789.012
        assert normalizer.safe_operation('add', a, b) == Decimal('912.46800000')
        assert normalizer.safe_operation('subtract', a, b) == Decimal('-665.55600000')
        assert normalizer.safe_operation('multiply', a, b) == Decimal('97408.12387200')
        assert normalizer.safe_operation('divide', a, b) == Decimal('0.15645600')
        
        # Test division by zero
        assert normalizer.safe_operation('divide', a, 0) == Decimal('0.00000000')
    
    def test_order_data_normalization(self):
        """Test order data normalization"""
        normalizer = DecimalNormalizer()
        
        order_data = {
            'price': 0.5234,
            'size': 1000.5,
            'amount': 523.4,
            'fee': 0.001,
            'bid': 0.5233,
            'ask': 0.5235,
            'spread': 0.0002,
            'symbol': 'XRP-USD'  # Non-numeric field
        }
        
        normalized = normalizer.normalize_order_data(order_data)
        
        # Check that numeric fields are Decimal
        assert isinstance(normalized['price'], Decimal)
        assert isinstance(normalized['size'], Decimal)
        assert isinstance(normalized['amount'], Decimal)
        assert isinstance(normalized['fee'], Decimal)
        assert isinstance(normalized['bid'], Decimal)
        assert isinstance(normalized['ask'], Decimal)
        assert isinstance(normalized['spread'], Decimal)
        
        # Check that non-numeric fields are unchanged
        assert normalized['symbol'] == 'XRP-USD'
    
    def test_market_data_normalization(self):
        """Test market data normalization"""
        normalizer = DecimalNormalizer()
        
        market_data = {
            'price': 0.5234,
            'bid': 0.5233,
            'ask': 0.5235,
            'last': 0.5234,
            'high': 0.5250,
            'low': 0.5220,
            'open': 0.5230,
            'close': 0.5234,
            'volume': 1000000.5,
            'size': 500.25,
            'amount': 261.88,
            'symbol': 'XRP-USD'
        }
        
        normalized = normalizer.normalize_market_data(market_data)
        
        # Check price fields (8 decimal places)
        for field in ['price', 'bid', 'ask', 'last', 'high', 'low', 'open', 'close']:
            assert isinstance(normalized[field], Decimal)
            assert normalized[field].as_tuple().exponent == -8
        
        # Check volume fields (6 decimal places)
        for field in ['volume', 'size', 'amount']:
            assert isinstance(normalized[field], Decimal)
            assert normalized[field].as_tuple().exponent == -6
        
        # Check non-numeric fields unchanged
        assert normalized['symbol'] == 'XRP-USD'

class TestDecimalOrderBook:
    """Test decimal order book functionality"""
    
    def test_decimal_order_book_operations(self):
        """Test decimal order book operations"""
        order_book = DecimalOrderBook()
        
        # Add bids and asks
        order_book.add_bid(0.5234, 1000.5)
        order_book.add_bid(0.5233, 1500.7)
        order_book.add_ask(0.5236, 2000.3)
        order_book.add_ask(0.5237, 1200.9)
        
        # Test best bid/ask
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        assert best_bid == (Decimal('0.52340000'), Decimal('1000.500000'))
        assert best_ask == (Decimal('0.52360000'), Decimal('2000.300000'))
        
        # Test spread and mid price
        spread = order_book.get_spread()
        mid_price = order_book.get_mid_price()
        
        assert spread == Decimal('0.00020000')
        assert mid_price == Decimal('0.52350000')

class TestDecimalTrade:
    """Test decimal trade functionality"""
    
    def test_decimal_trade_creation(self):
        """Test decimal trade creation"""
        trade = DecimalTrade(0.5235, 500.25, "buy", "2025-09-16T12:00:00Z")
        
        assert isinstance(trade.price, Decimal)
        assert isinstance(trade.size, Decimal)
        assert isinstance(trade.value, Decimal)
        
        assert trade.price == Decimal('0.52350000')
        assert trade.size == Decimal('500.250000')
        assert trade.value == Decimal('261.88087500')
        assert trade.side == "buy"
        assert trade.timestamp == "2025-09-16T12:00:00Z"
    
    def test_decimal_trade_to_dict(self):
        """Test decimal trade to dictionary conversion"""
        trade = DecimalTrade(0.5235, 500.25, "buy")
        trade_dict = trade.to_dict()
        
        assert isinstance(trade_dict['price'], float)
        assert isinstance(trade_dict['size'], float)
        assert isinstance(trade_dict['value'], float)
        
        assert trade_dict['price'] == 0.5235
        assert trade_dict['size'] == 500.25
        assert trade_dict['value'] == 261.880875
        assert trade_dict['side'] == "buy"

class TestDecimalSafeDecorator:
    """Test decimal safe decorator"""
    
    def test_decimal_safe_decorator(self):
        """Test that decimal_safe decorator normalizes arguments"""
        
        @decimal_safe
        def test_function(price, size, fee):
            # This function should receive Decimal arguments
            assert isinstance(price, Decimal)
            assert isinstance(size, Decimal)
            assert isinstance(fee, Decimal)
            return price * size - fee
        
        # Call with mixed types
        result = test_function(0.5234, 1000.5, 0.001)
        
        assert isinstance(result, Decimal)
        assert result == Decimal('523.39900000')

class TestNoFloatLeakage:
    """Test that no float leakage occurs in critical paths"""
    
    def test_no_float_leakage_in_order_math(self):
        """Test that order math uses only Decimal types"""
        normalizer = DecimalNormalizer()
        
        # Simulate order calculation
        price = normalizer.normalize(0.5234)
        size = normalizer.normalize(1000.5)
        fee_rate = normalizer.normalize(0.001)
        
        # All calculations should be Decimal
        order_value = price * size
        fee = order_value * fee_rate
        net_value = order_value - fee
        
        assert isinstance(order_value, Decimal)
        assert isinstance(fee, Decimal)
        assert isinstance(net_value, Decimal)
        
        # No float operations should occur
        assert not any(isinstance(x, float) for x in [order_value, fee, net_value])
    
    def test_global_decimal_context(self):
        """Test that global decimal context is properly set"""
        context = getcontext()
        
        assert context.prec == 10
        assert context.rounding == ROUND_HALF_EVEN

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
