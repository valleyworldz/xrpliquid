"""
UltimateBacktestBuilder-2025: Decimal Consistency Engine
Ensures all financial data is converted to Decimal at API ingest boundaries
"""

from decimal import Decimal, getcontext, ROUND_HALF_UP
import logging
from typing import Any, Dict, List, Union

# Set high precision for financial calculations
getcontext().prec = 18
getcontext().rounding = ROUND_HALF_UP

logger = logging.getLogger(__name__)

def to_decimal(val: Any) -> Decimal:
    """
    Convert any value to Decimal with proper error handling
    CRITICAL: This is the single source of truth for Decimal conversion
    """
    if val is None:
        return Decimal('0')
    
    if isinstance(val, Decimal):
        return val
    
    try:
        # Convert to string first to avoid float precision issues
        return Decimal(str(val))
    except (ValueError, TypeError, Exception) as e:
        logger.warning(f"⚠️ Decimal conversion failed for {val} ({type(val)}): {e}")
        return Decimal('0')

def decimal_safe_dict(data: Dict[str, Any], decimal_keys: List[str]) -> Dict[str, Any]:
    """
    Convert specific keys in a dictionary to Decimal
    Used for API response processing
    """
    result = data.copy()
    for key in decimal_keys:
        if key in result:
            result[key] = to_decimal(result[key])
    return result

def decimal_safe_position(position_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert position data to Decimal-safe format
    CRITICAL: All position calculations must use this
    """
    decimal_keys = [
        'size', 'entryPx', 'positionValue', 'unrealizedPnl', 
        'marginUsed', 'liquidationPx', 'markPx', 'notional'
    ]
    return decimal_safe_dict(position_data, decimal_keys)

def decimal_safe_account(account_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert account data to Decimal-safe format
    CRITICAL: All account calculations must use this
    """
    decimal_keys = [
        'accountValue', 'totalMarginUsed', 'totalNtlPos', 
        'totalRawUsd', 'crossMarginSummary', 'crossMaintenanceMarginUsed'
    ]
    return decimal_safe_dict(account_data, decimal_keys)

def decimal_safe_trade(trade_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert trade data to Decimal-safe format
    CRITICAL: All trade calculations must use this
    """
    decimal_keys = [
        'px', 'sz', 'side', 'time', 'startPosition', 'dir', 
        'closedPnl', 'hash', 'oid', 'fee', 'liquidationMarkPx'
    ]
    return decimal_safe_dict(trade_data, decimal_keys)

def decimal_safe_market_data(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert market data to Decimal-safe format
    CRITICAL: All market calculations must use this
    """
    decimal_keys = [
        'price', 'size', 'timestamp', 'bid', 'ask', 'last', 
        'volume', 'fundingRate', 'openInterest'
    ]
    return decimal_safe_dict(market_data, decimal_keys)

def validate_decimal_precision(value: Decimal, expected_precision: int = 8) -> Decimal:
    """
    Validate and quantize Decimal to expected precision
    """
    try:
        return value.quantize(Decimal('0.' + '0' * expected_precision))
    except Exception as e:
        logger.warning(f"⚠️ Decimal quantization failed: {e}")
        return value

def decimal_arithmetic_safe(a: Any, b: Any, operation: str) -> Decimal:
    """
    Perform safe Decimal arithmetic operations
    """
    try:
        dec_a = to_decimal(a)
        dec_b = to_decimal(b)
        
        if operation == 'add':
            return dec_a + dec_b
        elif operation == 'sub':
            return dec_a - dec_b
        elif operation == 'mul':
            return dec_a * dec_b
        elif operation == 'div':
            if dec_b == 0:
                return Decimal('0')
            return dec_a / dec_b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        logger.error(f"❌ Decimal arithmetic failed: {a} {operation} {b} = {e}")
        return Decimal('0')

# Critical financial calculation helpers
def calculate_position_value(size: Any, entry_price: Any) -> Decimal:
    """Calculate position value with Decimal precision"""
    return decimal_arithmetic_safe(size, entry_price, 'mul')

def calculate_pnl(current_price: Any, entry_price: Any, size: Any, side: str) -> Decimal:
    """Calculate PnL with Decimal precision"""
    price_diff = decimal_arithmetic_safe(current_price, entry_price, 'sub')
    if side.lower() == 'long':
        return decimal_arithmetic_safe(price_diff, size, 'mul')
    else:  # short
        return decimal_arithmetic_safe(decimal_arithmetic_safe(price_diff, -1, 'mul'), size, 'mul')

def calculate_margin_ratio(position_value: Any, account_value: Any) -> Decimal:
    """Calculate margin ratio with Decimal precision"""
    if to_decimal(account_value) == 0:
        return Decimal('0')
    return decimal_arithmetic_safe(position_value, account_value, 'div')

def calculate_leverage(position_value: Any, margin_used: Any) -> Decimal:
    """Calculate leverage with Decimal precision"""
    if to_decimal(margin_used) == 0:
        return Decimal('0')
    return decimal_arithmetic_safe(position_value, margin_used, 'div')

# Logging helper for Decimal operations
def log_decimal_operation(operation: str, inputs: Dict[str, Any], result: Decimal):
    """Log Decimal operations for debugging"""
    logger.debug(f"🔢 Decimal {operation}: {inputs} = {result}")

# Initialize Decimal context
def initialize_decimal_context():
    """Initialize Decimal context for financial calculations"""
    getcontext().prec = 18
    getcontext().rounding = ROUND_HALF_UP
    logger.info("✅ Decimal context initialized for financial precision")

# Auto-initialize
initialize_decimal_context()
