"""
🏛️ HYPERLIQUID INVARIANTS ENFORCEMENT
=====================================
Pre-validation of orders against Hyperliquid exchange constraints
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP

logger = logging.getLogger(__name__)

@dataclass
class HyperliquidInvariants:
    """Hyperliquid exchange invariants and constraints"""
    
    # Tick sizes for major assets
    TICK_SIZES = {
        'XRP': Decimal('0.0001'),
        'BTC': Decimal('0.01'),
        'ETH': Decimal('0.01'),
        'SOL': Decimal('0.001'),
        'ARB': Decimal('0.0001'),
        'OP': Decimal('0.0001'),
    }
    
    # Minimum notional values
    MIN_NOTIONAL = {
        'XRP': Decimal('1.0'),    # $1 minimum
        'BTC': Decimal('10.0'),   # $10 minimum
        'ETH': Decimal('10.0'),   # $10 minimum
        'SOL': Decimal('5.0'),    # $5 minimum
        'ARB': Decimal('1.0'),    # $1 minimum
        'OP': Decimal('1.0'),     # $1 minimum
    }
    
    # Maximum position sizes
    MAX_POSITION_SIZE = {
        'XRP': Decimal('1000000.0'),  # $1M max
        'BTC': Decimal('5000000.0'),  # $5M max
        'ETH': Decimal('5000000.0'),  # $5M max
        'SOL': Decimal('2000000.0'),  # $2M max
        'ARB': Decimal('1000000.0'),  # $1M max
        'OP': Decimal('1000000.0'),   # $1M max
    }
    
    # Funding cycle constraints
    FUNDING_CYCLE_HOURS = 1  # Hyperliquid standard: 1-hour funding cycles
    FUNDING_WINDOW_MINUTES = 5  # 5-minute window before funding
    
    # Order type constraints
    SUPPORTED_ORDER_TYPES = ['limit', 'market', 'stop_limit', 'scale', 'twap']
    POST_ONLY_ORDER_TYPES = ['limit']
    
    # Leverage constraints
    MAX_LEVERAGE = Decimal('20.0')  # 20x maximum leverage
    MIN_LEVERAGE = Decimal('1.0')   # 1x minimum leverage
    
    # Margin constraints
    MIN_MARGIN_RATIO = Decimal('0.05')  # 5% minimum margin ratio
    MAINTENANCE_MARGIN_RATIO = Decimal('0.03')  # 3% maintenance margin

class HyperliquidValidator:
    """Validates orders against Hyperliquid invariants"""
    
    def __init__(self):
        self.invariants = HyperliquidInvariants()
        logger.info("🏛️ [HL_INVARIANTS] Hyperliquid validator initialized")
    
    def validate_tick_size(self, symbol: str, price: float) -> Tuple[bool, Optional[str]]:
        """Validate price against tick size constraints"""
        if symbol not in self.invariants.TICK_SIZES:
            return False, f"Unknown symbol: {symbol}"
        
        tick_size = self.invariants.TICK_SIZES[symbol]
        price_decimal = Decimal(str(price))
        
        # Check if price is aligned with tick size
        remainder = price_decimal % tick_size
        if remainder != 0:
            # Round to nearest valid tick
            rounded_price = price_decimal.quantize(tick_size, rounding=ROUND_DOWN)
            return False, f"Price {price} not aligned with tick size {tick_size}. Use {rounded_price}"
        
        return True, None
    
    def validate_min_notional(self, symbol: str, quantity: float, price: float) -> Tuple[bool, Optional[str]]:
        """Validate minimum notional value"""
        if symbol not in self.invariants.MIN_NOTIONAL:
            return False, f"Unknown symbol: {symbol}"
        
        notional = quantity * price
        min_notional = float(self.invariants.MIN_NOTIONAL[symbol])
        
        if notional < min_notional:
            return False, f"Notional ${notional:.2f} below minimum ${min_notional:.2f} for {symbol}"
        
        return True, None
    
    def validate_max_position_size(self, symbol: str, quantity: float, price: float) -> Tuple[bool, Optional[str]]:
        """Validate maximum position size"""
        if symbol not in self.invariants.MAX_POSITION_SIZE:
            return False, f"Unknown symbol: {symbol}"
        
        notional = quantity * price
        max_notional = float(self.invariants.MAX_POSITION_SIZE[symbol])
        
        if notional > max_notional:
            return False, f"Notional ${notional:.2f} exceeds maximum ${max_notional:.2f} for {symbol}"
        
        return True, None
    
    def validate_leverage(self, leverage: float) -> Tuple[bool, Optional[str]]:
        """Validate leverage constraints"""
        leverage_decimal = Decimal(str(leverage))
        
        if leverage_decimal < self.invariants.MIN_LEVERAGE:
            return False, f"Leverage {leverage} below minimum {self.invariants.MIN_LEVERAGE}"
        
        if leverage_decimal > self.invariants.MAX_LEVERAGE:
            return False, f"Leverage {leverage} exceeds maximum {self.invariants.MAX_LEVERAGE}"
        
        return True, None
    
    def validate_reduce_only(self, order: Dict[str, Any], current_position: float) -> Tuple[bool, Optional[str]]:
        """Validate reduce-only flag logic"""
        if not order.get('reduce_only', False):
            return True, None
        
        side = order.get('side', '').lower()
        quantity = float(order.get('quantity', 0))
        
        if side == 'buy' and current_position >= 0:
            return False, "Reduce-only buy order with no short position"
        
        if side == 'sell' and current_position <= 0:
            return False, "Reduce-only sell order with no long position"
        
        return True, None
    
    def validate_funding_window(self, timestamp: int) -> Tuple[bool, Optional[str]]:
        """Validate if order is within funding window"""
        from datetime import datetime, timedelta
        
        order_time = datetime.fromtimestamp(timestamp)
        funding_cycle_minutes = self.invariants.FUNDING_CYCLE_HOURS * 60
        
        # Calculate next funding time
        hours_since_epoch = order_time.hour
        next_funding_hour = ((hours_since_epoch // self.invariants.FUNDING_CYCLE_HOURS) + 1) * self.invariants.FUNDING_CYCLE_HOURS
        next_funding_time = order_time.replace(hour=next_funding_hour % 24, minute=0, second=0, microsecond=0)
        
        if next_funding_hour >= 24:
            next_funding_time += timedelta(days=1)
        
        time_to_funding = (next_funding_time - order_time).total_seconds() / 60
        
        if time_to_funding <= self.invariants.FUNDING_WINDOW_MINUTES:
            return False, f"Order too close to funding time: {time_to_funding:.1f} minutes"
        
        return True, None
    
    def validate_order_type(self, order_type: str, post_only: bool = False) -> Tuple[bool, Optional[str]]:
        """Validate order type constraints"""
        if order_type not in self.invariants.SUPPORTED_ORDER_TYPES:
            return False, f"Unsupported order type: {order_type}"
        
        if post_only and order_type not in self.invariants.POST_ONLY_ORDER_TYPES:
            return False, f"Post-only not supported for order type: {order_type}"
        
        return True, None
    
    def validate_margin_requirements(self, symbol: str, quantity: float, price: float, 
                                   leverage: float, current_margin: float) -> Tuple[bool, Optional[str]]:
        """Validate margin requirements"""
        notional = quantity * price
        required_margin = notional / leverage
        
        if required_margin > current_margin:
            return False, f"Insufficient margin: required ${required_margin:.2f}, available ${current_margin:.2f}"
        
        # Check minimum margin ratio
        margin_ratio = required_margin / notional
        if margin_ratio < float(self.invariants.MIN_MARGIN_RATIO):
            return False, f"Margin ratio {margin_ratio:.2%} below minimum {self.invariants.MIN_MARGIN_RATIO:.2%}"
        
        return True, None
    
    def validate_order(self, order: Dict[str, Any], current_position: float = 0.0, 
                      current_margin: float = 0.0) -> Tuple[bool, Optional[str]]:
        """Comprehensive order validation"""
        symbol = order.get('symbol', '')
        price = float(order.get('price', 0))
        quantity = float(order.get('quantity', 0))
        leverage = float(order.get('leverage', 1.0))
        order_type = order.get('order_type', 'limit')
        post_only = order.get('post_only', False)
        timestamp = order.get('timestamp', 0)
        
        # Validate tick size
        valid, error = self.validate_tick_size(symbol, price)
        if not valid:
            return False, f"Tick size validation failed: {error}"
        
        # Validate minimum notional
        valid, error = self.validate_min_notional(symbol, quantity, price)
        if not valid:
            return False, f"Min notional validation failed: {error}"
        
        # Validate maximum position size
        valid, error = self.validate_max_position_size(symbol, quantity, price)
        if not valid:
            return False, f"Max position size validation failed: {error}"
        
        # Validate leverage
        valid, error = self.validate_leverage(leverage)
        if not valid:
            return False, f"Leverage validation failed: {error}"
        
        # Validate reduce-only logic
        valid, error = self.validate_reduce_only(order, current_position)
        if not valid:
            return False, f"Reduce-only validation failed: {error}"
        
        # Validate funding window
        if timestamp > 0:
            valid, error = self.validate_funding_window(timestamp)
            if not valid:
                return False, f"Funding window validation failed: {error}"
        
        # Validate order type
        valid, error = self.validate_order_type(order_type, post_only)
        if not valid:
            return False, f"Order type validation failed: {error}"
        
        # Validate margin requirements
        valid, error = self.validate_margin_requirements(symbol, quantity, price, leverage, current_margin)
        if not valid:
            return False, f"Margin validation failed: {error}"
        
        logger.info(f"🏛️ [HL_INVARIANTS] Order validation passed for {symbol} {quantity}@{price}")
        return True, None

# Global validator instance
hl_validator = HyperliquidValidator()

def validate_hyperliquid_order(order: Dict[str, Any], current_position: float = 0.0, 
                              current_margin: float = 0.0) -> Tuple[bool, Optional[str]]:
    """Convenience function for order validation"""
    return hl_validator.validate_order(order, current_position, current_margin)

# Doctests for validation functions
if __name__ == "__main__":
    import doctest
    
    def test_tick_size_validation():
        """Test tick size validation"""
        validator = HyperliquidValidator()
        
        # Valid tick size
        valid, error = validator.validate_tick_size('XRP', 0.5000)
        assert valid, f"Valid tick size failed: {error}"
        
        # Invalid tick size
        valid, error = validator.validate_tick_size('XRP', 0.5001)
        assert not valid, "Invalid tick size should fail"
        assert "0.5001" in error, f"Error message should contain price: {error}"
        
        print("✅ Tick size validation tests passed")
    
    def test_min_notional_validation():
        """Test minimum notional validation"""
        validator = HyperliquidValidator()
        
        # Valid notional
        valid, error = validator.validate_min_notional('XRP', 100, 0.5)
        assert valid, f"Valid notional failed: {error}"
        
        # Invalid notional
        valid, error = validator.validate_min_notional('XRP', 1, 0.5)
        assert not valid, "Invalid notional should fail"
        assert "below minimum" in error, f"Error message should indicate minimum: {error}"
        
        print("✅ Min notional validation tests passed")
    
    def test_comprehensive_validation():
        """Test comprehensive order validation"""
        validator = HyperliquidValidator()
        
        # Valid order
        order = {
            'symbol': 'XRP',
            'price': 0.5000,
            'quantity': 100,
            'leverage': 2.0,
            'order_type': 'limit',
            'post_only': True,
            'timestamp': 1640995200
        }
        
        valid, error = validator.validate_order(order, current_position=0.0, current_margin=1000.0)
        assert valid, f"Valid order failed: {error}"
        
        # Invalid order (wrong tick size)
        order['price'] = 0.5001
        valid, error = validator.validate_order(order, current_position=0.0, current_margin=1000.0)
        assert not valid, "Invalid order should fail"
        assert "Tick size validation failed" in error, f"Error should indicate tick size: {error}"
        
        print("✅ Comprehensive validation tests passed")
    
    # Run tests
    test_tick_size_validation()
    test_min_notional_validation()
    test_comprehensive_validation()
    
    print("🎉 All Hyperliquid invariants tests passed!")
