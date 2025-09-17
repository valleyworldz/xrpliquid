"""
Type Coercion Layer for Decimal vs Float
Ensures reliable order placement by handling type mismatches.
"""

from src.core.utils.decimal_boundary_guard import safe_decimal
from src.core.utils.decimal_boundary_guard import safe_float
import decimal
from typing import Union, Any, Dict, List
import logging
from decimal import Decimal, InvalidOperation
import numpy as np

logger = logging.getLogger(__name__)

class TypeCoercionError(Exception):
    """Exception raised when type coercion fails."""
    pass

class TypeCoercionLayer:
    """Handles type coercion between Decimal and float for order placement."""
    
    def __init__(self, precision: int = 8):
        self.precision = precision
        self.decimal_context = decimal.getcontext()
        self.decimal_context.prec = precision
        
        # Common type mappings
        self.type_mappings = {
            'price': Decimal,
            'quantity': Decimal,
            'amount': Decimal,
            'fee': Decimal,
            'slippage': float,
            'percentage': float,
            'ratio': float
        }
    
    def coerce_to_decimal(self, value: Any, field_name: str = None) -> Decimal:
        """Coerce value to Decimal with proper precision."""
        
        if isinstance(value, Decimal):
            return value
        
        if isinstance(value, (int, float)):
            try:
                # Convert to string first to avoid floating point precision issues
                if isinstance(value, float):
                    # Handle special float values
                    if np.isnan(value) or np.isinf(value):
                        raise TypeCoercionError(f"Invalid float value: {value}")
                    
                    # Convert to string with sufficient precision
                    value_str = f"{value:.{self.precision}f}"
                else:
                    value_str = str(value)
                
                return safe_decimal(value_str)
            
            except (InvalidOperation, ValueError) as e:
                raise TypeCoercionError(f"Failed to coerce {value} to Decimal: {e}")
        
        if isinstance(value, str):
            try:
                return safe_decimal(value)
            except (InvalidOperation, ValueError) as e:
                raise TypeCoercionError(f"Failed to coerce string '{value}' to Decimal: {e}")
        
        raise TypeCoercionError(f"Cannot coerce {type(value)} to Decimal: {value}")
    
    def coerce_to_float(self, value: Any, field_name: str = None) -> float:
        """Coerce value to float with validation."""
        
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                raise TypeCoercionError(f"Invalid float value: {value}")
            return value
        
        if isinstance(value, Decimal):
            try:
                return safe_float(value)
            except (OverflowError, ValueError) as e:
                raise TypeCoercionError(f"Failed to coerce Decimal {value} to float: {e}")
        
        if isinstance(value, (int, str)):
            try:
                return safe_float(value)
            except (ValueError, OverflowError) as e:
                raise TypeCoercionError(f"Failed to coerce {value} to float: {e}")
        
        raise TypeCoercionError(f"Cannot coerce {type(value)} to float: {value}")
    
    def coerce_order_data(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce order data to appropriate types."""
        
        coerced_data = {}
        
        for key, value in order_data.items():
            try:
                # Determine target type based on field name
                if key in ['price', 'quantity', 'amount', 'fee']:
                    coerced_data[key] = self.coerce_to_decimal(value, key)
                elif key in ['slippage', 'percentage', 'ratio', 'confidence']:
                    coerced_data[key] = self.coerce_to_float(value, key)
                else:
                    # Keep original type for other fields
                    coerced_data[key] = value
                
            except TypeCoercionError as e:
                logger.error(f"Type coercion failed for {key}: {e}")
                raise
        
        return coerced_data
    
    def validate_decimal_precision(self, value: Decimal, max_precision: int = None) -> bool:
        """Validate Decimal precision."""
        
        if max_precision is None:
            max_precision = self.precision
        
        # Count decimal places
        decimal_places = len(str(value).split('.')[-1]) if '.' in str(value) else 0
        
        return decimal_places <= max_precision
    
    def round_decimal(self, value: Decimal, precision: int = None) -> Decimal:
        """Round Decimal to specified precision."""
        
        if precision is None:
            precision = self.precision
        
        return value.quantize(safe_decimal('0.1') ** precision)
    
    def safe_decimal_operation(self, operation: str, a: Union[Decimal, float], 
                             b: Union[Decimal, float]) -> Decimal:
        """Perform safe Decimal operations."""
        
        # Coerce inputs to Decimal
        decimal_a = self.coerce_to_decimal(a)
        decimal_b = self.coerce_to_decimal(b)
        
        try:
            if operation == 'add':
                result = decimal_a + decimal_b
            elif operation == 'subtract':
                result = decimal_a - decimal_b
            elif operation == 'multiply':
                result = decimal_a * decimal_b
            elif operation == 'divide':
                if decimal_b == 0:
                    raise TypeCoercionError("Division by zero")
                result = decimal_a / decimal_b
            else:
                raise TypeCoercionError(f"Unknown operation: {operation}")
            
            return self.round_decimal(result)
            
        except (InvalidOperation, OverflowError) as e:
            raise TypeCoercionError(f"Decimal operation failed: {e}")
    
    def coerce_price_levels(self, price_levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Coerce price levels for order book data."""
        
        coerced_levels = []
        
        for level in price_levels:
            try:
                coerced_level = {
                    'price': self.coerce_to_decimal(level['price']),
                    'quantity': self.coerce_to_decimal(level['quantity']),
                    'timestamp': level.get('timestamp'),
                    'side': level.get('side')
                }
                coerced_levels.append(coerced_level)
                
            except (KeyError, TypeCoercionError) as e:
                logger.error(f"Failed to coerce price level: {e}")
                raise
        
        return coerced_levels
    
    def coerce_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce trade execution data."""
        
        try:
            coerced_trade = {
                'trade_id': trade_data['trade_id'],
                'symbol': trade_data['symbol'],
                'side': trade_data['side'],
                'quantity': self.coerce_to_decimal(trade_data['quantity']),
                'price': self.coerce_to_decimal(trade_data['price']),
                'timestamp': trade_data['timestamp'],
                'fee': self.coerce_to_decimal(trade_data.get('fee', 0)),
                'slippage': self.coerce_to_float(trade_data.get('slippage', 0.0))
            }
            
            return coerced_trade
            
        except (KeyError, TypeCoercionError) as e:
            logger.error(f"Failed to coerce trade data: {e}")
            raise
    
    def get_type_coercion_stats(self) -> Dict[str, Any]:
        """Get statistics about type coercion operations."""
        
        return {
            'precision': self.precision,
            'decimal_context_prec': self.decimal_context.prec,
            'type_mappings': self.type_mappings,
            'supported_operations': ['add', 'subtract', 'multiply', 'divide']
        }

class OrderTypeCoercion:
    """Specialized type coercion for order placement."""
    
    def __init__(self, coercion_layer: TypeCoercionLayer):
        self.coercion = coercion_layer
    
    def prepare_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare order data with proper type coercion."""
        
        # Validate required fields
        required_fields = ['symbol', 'side', 'quantity', 'price']
        for field in required_fields:
            if field not in order_data:
                raise TypeCoercionError(f"Missing required field: {field}")
        
        # Coerce order data
        coerced_order = self.coercion.coerce_order_data(order_data)
        
        # Additional validation
        if coerced_order['quantity'] <= 0:
            raise TypeCoercionError("Quantity must be positive")
        
        if coerced_order['price'] <= 0:
            raise TypeCoercionError("Price must be positive")
        
        # Validate precision
        if not self.coercion.validate_decimal_precision(coerced_order['price']):
            logger.warning(f"Price precision exceeds limit: {coerced_order['price']}")
        
        if not self.coercion.validate_decimal_precision(coerced_order['quantity']):
            logger.warning(f"Quantity precision exceeds limit: {coerced_order['quantity']}")
        
        return coerced_order
    
    def prepare_take_profit_stop_loss(self, tp_sl_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare TP/SL orders with type coercion."""
        
        try:
            coerced_tp_sl = {
                'symbol': tp_sl_data['symbol'],
                'side': tp_sl_data['side'],
                'quantity': self.coercion.coerce_to_decimal(tp_sl_data['quantity']),
                'take_profit_price': self.coercion.coerce_to_decimal(tp_sl_data['take_profit_price']),
                'stop_loss_price': self.coercion.coerce_to_decimal(tp_sl_data['stop_loss_price']),
                'current_price': self.coercion.coerce_to_decimal(tp_sl_data['current_price'])
            }
            
            # Validate TP/SL logic
            if coerced_tp_sl['side'] == 'buy':
                if coerced_tp_sl['take_profit_price'] <= coerced_tp_sl['current_price']:
                    raise TypeCoercionError("Take profit price must be above current price for buy orders")
                if coerced_tp_sl['stop_loss_price'] >= coerced_tp_sl['current_price']:
                    raise TypeCoercionError("Stop loss price must be below current price for buy orders")
            else:  # sell
                if coerced_tp_sl['take_profit_price'] >= coerced_tp_sl['current_price']:
                    raise TypeCoercionError("Take profit price must be below current price for sell orders")
                if coerced_tp_sl['stop_loss_price'] <= coerced_tp_sl['current_price']:
                    raise TypeCoercionError("Stop loss price must be above current price for sell orders")
            
            return coerced_tp_sl
            
        except (KeyError, TypeCoercionError) as e:
            logger.error(f"Failed to prepare TP/SL order: {e}")
            raise

def main():
    """Demonstrate type coercion layer."""
    
    # Initialize type coercion layer
    coercion_layer = TypeCoercionLayer(precision=8)
    order_coercion = OrderTypeCoercion(coercion_layer)
    
    print("🧪 Testing Type Coercion Layer")
    print("=" * 50)
    
    # Test basic type coercion
    test_values = [
        (123.456789, 'price'),
        ('0.00012345', 'quantity'),
        (safe_decimal('0.001'), 'fee'),
        (np.float64(0.5), 'ratio')
    ]
    
    for value, field_name in test_values:
        try:
            decimal_result = coercion_layer.coerce_to_decimal(value, field_name)
            float_result = coercion_layer.coerce_to_float(value, field_name)
            print(f"✅ {field_name}: {value} -> Decimal: {decimal_result}, Float: {float_result}")
        except TypeCoercionError as e:
            print(f"❌ {field_name}: {value} -> Error: {e}")
    
    # Test order preparation
    test_order = {
        'symbol': 'XRP',
        'side': 'buy',
        'quantity': 123.456789,
        'price': '0.52',
        'fee': 0.001
    }
    
    try:
        prepared_order = order_coercion.prepare_order(test_order)
        print(f"\n✅ Order prepared: {prepared_order}")
    except TypeCoercionError as e:
        print(f"\n❌ Order preparation failed: {e}")
    
    # Test TP/SL preparation
    test_tp_sl = {
        'symbol': 'XRP',
        'side': 'buy',
        'quantity': 1000,
        'take_profit_price': 0.55,
        'stop_loss_price': 0.50,
        'current_price': 0.52
    }
    
    try:
        prepared_tp_sl = order_coercion.prepare_take_profit_stop_loss(test_tp_sl)
        print(f"✅ TP/SL prepared: {prepared_tp_sl}")
    except TypeCoercionError as e:
        print(f"❌ TP/SL preparation failed: {e}")
    
    # Test decimal operations
    try:
        result = coercion_layer.safe_decimal_operation('multiply', 0.52, 1000)
        print(f"✅ Decimal operation: 0.52 * 1000 = {result}")
    except TypeCoercionError as e:
        print(f"❌ Decimal operation failed: {e}")
    
    print(f"\n📊 Type Coercion Stats: {coercion_layer.get_type_coercion_stats()}")
    
    return 0

if __name__ == "__main__":
    exit(main())
