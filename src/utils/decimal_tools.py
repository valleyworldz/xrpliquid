"""
Decimal Tools - Comprehensive Decimal/float handling utilities
Ensures full Decimal discipline throughout the trading system
"""

from decimal import Decimal, getcontext, InvalidOperation, FloatOperation
import logging

# Set high precision for financial calculations
getcontext().prec = 28

def D(x):
    """
    Force float or int to Decimal safely.
    This is the core utility for maintaining Decimal discipline.
    
    Args:
        x: Any numeric type (int, float, str, Decimal)
        
    Returns:
        Decimal: Safe Decimal representation
        
    Examples:
        >>> D(3.14)
        Decimal('3.14')
        >>> D("3.14")
        Decimal('3.14')
        >>> D(Decimal("3.14"))
        Decimal('3.14')
    """
    if isinstance(x, Decimal):
        return x
    elif isinstance(x, (int, float)):
        return Decimal(str(x))
    elif isinstance(x, str):
        try:
            return Decimal(x)
        except:
            return Decimal('0')
    else:
        return Decimal('0')

def safe_decimal_operation(operation, *args):
    """
    Safely perform arithmetic operations with Decimal discipline.
    
    Args:
        operation: Function to perform (add, sub, mul, div, etc.)
        *args: Arguments to convert to Decimal and pass to operation
        
    Returns:
        Decimal: Result of the operation
    """
    try:
        decimal_args = [D(arg) for arg in args]
        return operation(*decimal_args)
    except Exception as e:
        logging.warning(f"Decimal operation failed: {e}, returning 0")
        return Decimal('0')

def decimal_add(*args):
    """Add multiple values with Decimal discipline."""
    result = D(0)
    for arg in args:
        result += D(arg)
    return result

def decimal_sub(a, b):
    """Subtract b from a with Decimal discipline."""
    return D(a) - D(b)

def decimal_mul(*args):
    """Multiply multiple values with Decimal discipline."""
    result = D(1)
    for arg in args:
        result *= D(arg)
    return result

def decimal_div(a, b):
    """Divide a by b with Decimal discipline."""
    try:
        return D(a) / D(b)
    except:
        return Decimal('0')

def decimal_percentage(value, percentage):
    """
    Calculate percentage of value with Decimal discipline.
    
    Args:
        value: Base value
        percentage: Percentage as decimal (0.02 for 2%)
        
    Returns:
        Decimal: Percentage of value
    """
    return D(value) * D(percentage)

def decimal_percentage_add(value, percentage):
    """
    Add percentage to value with Decimal discipline.
    
    Args:
        value: Base value
        percentage: Percentage as decimal (0.02 for 2%)
        
    Returns:
        Decimal: Value + percentage
    """
    return D(value) * (D(1) + D(percentage))

def decimal_percentage_sub(value, percentage):
    """
    Subtract percentage from value with Decimal discipline.
    
    Args:
        value: Base value
        percentage: Percentage as decimal (0.02 for 2%)
        
    Returns:
        Decimal: Value - percentage
    """
    return D(value) * (D(1) - D(percentage))

def decimal_abs(value):
    """Absolute value with Decimal discipline."""
    return abs(D(value))

def decimal_max(*args):
    """Maximum value with Decimal discipline."""
    if not args:
        return Decimal('0')
    return max(D(arg) for arg in args)

def decimal_min(*args):
    """Minimum value with Decimal discipline."""
    if not args:
        return Decimal('0')
    return min(D(arg) for arg in args)

def decimal_round(value, precision=4):
    """Round Decimal to specified precision."""
    return D(value).quantize(Decimal('0.' + '0' * precision))

def ensure_decimal_context():
    """Ensure proper Decimal context for financial calculations."""
    getcontext().prec = 28
    getcontext().rounding = 'ROUND_HALF_EVEN'
    # Disable specific traps instead of all exceptions
    getcontext().traps[InvalidOperation] = False
    getcontext().traps[FloatOperation] = False

# Initialize Decimal context
ensure_decimal_context()
