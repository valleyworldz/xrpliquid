
"""
Decimal Guard - Comprehensive decimal/float safety for financial calculations
"""

from decimal import Decimal, getcontext, ROUND_HALF_EVEN, InvalidOperation
from typing import Union, Any
import logging

# Set global decimal context
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_EVEN
getcontext().traps[InvalidOperation] = 0

logger = logging.getLogger(__name__)

def safe_float(value: Any) -> Decimal:
    """
    Safely convert any value to Decimal
    """
    try:
        if value is None:
            return Decimal('0')
        elif isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, str):
            # Handle empty strings
            if not value.strip():
                return Decimal('0')
            return Decimal(value)
        else:
            # Try to convert to string first
            return Decimal(str(value))
    except Exception as e:
        logger.error(f"Error converting {value} to Decimal: {e}")
        return Decimal('0')

def safe_decimal(value: Any) -> Decimal:
    """
    Alias for safe_float for consistency
    """
    return safe_float(value)

def safe_arithmetic(a: Any, b: Any, operation: str) -> Decimal:
    """
    Safely perform arithmetic operations between mixed types
    """
    try:
        decimal_a = safe_float(a)
        decimal_b = safe_float(b)
        
        if operation == '+':
            return decimal_a + decimal_b
        elif operation == '-':
            return decimal_a - decimal_b
        elif operation == '*':
            return decimal_a * decimal_b
        elif operation == '/':
            if decimal_b == 0:
                logger.warning("Division by zero detected, returning 0")
                return Decimal('0')
            return decimal_a / decimal_b
        else:
            logger.error(f"Unknown operation: {operation}")
            return Decimal('0')
    except Exception as e:
        logger.error(f"Error in safe arithmetic {a} {operation} {b}: {e}")
        return Decimal('0')

def enforce_decimal_precision(value: Decimal, precision: int = 10) -> Decimal:
    """
    Enforce decimal precision
    """
    try:
        return value.quantize(Decimal('0.' + '0' * precision))
    except Exception as e:
        logger.error(f"Error enforcing precision for {value}: {e}")
        return Decimal('0')

# Global decimal context enforcement
def enforce_global_decimal_context():
    """
    Enforce global decimal context for all financial calculations
    """
    getcontext().prec = 10
    getcontext().rounding = ROUND_HALF_EVEN
    getcontext().traps[InvalidOperation] = 0
    logger.info("Global decimal context enforced")

# Auto-enforce on import
enforce_global_decimal_context()
