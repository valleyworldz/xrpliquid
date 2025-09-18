"""
Central Numeric Helpers - Consistent Type Boundaries
Ensures proper Decimal/float discipline throughout the trading system
"""

from decimal import Decimal, getcontext
import logging

# Set high precision for financial calculations
getcontext().prec = 18
getcontext().rounding = 'ROUND_HALF_EVEN'

def D(x) -> Decimal:
    """
    Convert any value to Decimal safely.
    Use for money/price/size/ratios that multiply money.
    """
    if isinstance(x, Decimal):
        return x
    if x is None:
        return Decimal('0.0')
    try:
        return Decimal(str(x))
    except Exception as e:
        logging.warning(f"⚠️ Decimal conversion failed for {x} ({type(x)}): {e}")
        return Decimal('0.0')

def F(x) -> float:
    """
    Convert any value to float safely.
    Use for indicator arrays/internal ML.
    """
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    try:
        return float(x)
    except Exception as e:
        logging.warning(f"⚠️ Float conversion failed for {x} ({type(x)}): {e}")
        return 0.0

def safe_money_math(operation, *args):
    """
    Perform money-related arithmetic with Decimal precision.
    All arguments are converted to Decimal before operation.
    """
    try:
        decimal_args = [D(arg) for arg in args]
        if operation == 'add':
            return sum(decimal_args)
        elif operation == 'sub':
            return decimal_args[0] - decimal_args[1]
        elif operation == 'mul':
            result = decimal_args[0]
            for arg in decimal_args[1:]:
                result *= arg
            return result
        elif operation == 'div':
            if decimal_args[1] == 0:
                return Decimal('0.0')
            return decimal_args[0] / decimal_args[1]
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        logging.error(f"❌ Money math operation failed: {e}")
        return Decimal('0.0')

def safe_indicator_math(operation, *args):
    """
    Perform indicator-related arithmetic with float precision.
    All arguments are converted to float before operation.
    """
    try:
        float_args = [F(arg) for arg in args]
        if operation == 'add':
            return sum(float_args)
        elif operation == 'sub':
            return float_args[0] - float_args[1]
        elif operation == 'mul':
            result = float_args[0]
            for arg in float_args[1:]:
                result *= arg
            return result
        elif operation == 'div':
            if float_args[1] == 0:
                return 0.0
            return float_args[0] / float_args[1]
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as e:
        logging.error(f"❌ Indicator math operation failed: {e}")
        return 0.0

def convert_at_boundary(value, target_type='decimal'):
    """
    Convert value at type boundaries.
    target_type: 'decimal' or 'float'
    """
    if target_type == 'decimal':
        return D(value)
    elif target_type == 'float':
        return F(value)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

# Initialize Decimal context
getcontext().prec = 18
getcontext().rounding = 'ROUND_HALF_EVEN'
