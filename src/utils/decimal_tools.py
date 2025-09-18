"""
Decimal Tools - Comprehensive Decimal/float handling utilities
Ensures full Decimal discipline throughout the trading system
"""

from decimal import Decimal, getcontext, InvalidOperation, FloatOperation, ROUND_HALF_EVEN
import logging
import functools

# Set high precision for financial calculations
getcontext().prec = 18
getcontext().rounding = ROUND_HALF_EVEN

def D(x):
    """
    CRITICAL: Safe Decimal wrapper - single source of truth for all conversions
    This is the ONLY way to convert external data to Decimal
    
    Args:
        x: Any numeric type (int, float, str, Decimal, None)
        
    Returns:
        Decimal: Safe Decimal representation
        
    Examples:
        >>> D(3.14)
        Decimal('3.14')
        >>> D("3.14")
        Decimal('3.14')
        >>> D(Decimal("3.14"))
        Decimal('3.14')
        >>> D(None)
        Decimal('0.0')
    """
    if isinstance(x, Decimal):
        return x
    if x is None:
        return Decimal('0.0')
    try:
        return Decimal(str(x))
    except Exception as e:
        logging.warning(f"⚠️ Decimal conversion failed for {x} ({type(x)}): {e}")
        return Decimal('0.0') # Fallback to Decimal zero on conversion error

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

def decimal_power(base, exponent):
    """Raise a Decimal to a power."""
    return D(base) ** D(exponent)

def decimal_sqrt(value):
    """Calculate square root of a Decimal."""
    return D(value).sqrt()

def decimal_abs(value):
    """Get absolute value of a Decimal."""
    return abs(D(value))

def decimal_sum(values):
    """Sum a list of values as Decimals."""
    return sum(D(v) for v in values)

def decimal_avg(values):
    """Calculate average of values as Decimals."""
    if not values:
        return D(0)
    return decimal_sum(values) / D(len(values))

def ensure_decimal_context():
    """Ensure proper Decimal context for financial calculations."""
    getcontext().prec = 18
    getcontext().rounding = ROUND_HALF_EVEN
    # Disable specific traps instead of all exceptions
    getcontext().traps[InvalidOperation] = False
    getcontext().traps[FloatOperation] = False

# Initialize Decimal context
ensure_decimal_context()

# Type enforcement decorator for debugging
def ensure_decimal(func):
    """
    Decorator to ensure function results are Decimal-safe.
    Used for debugging and catching type leaks.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # Convert float results to Decimal for consistency
        if isinstance(result, float):
            return D(result)
        return result
    return wrapper

# Market data normalization functions
def normalize_price(price):
    """Normalize price data to Decimal."""
    return D(price)

def normalize_atr(atr_value):
    """Normalize ATR data to Decimal."""
    return D(atr_value)

def normalize_balance(balance):
    """Normalize account balance to Decimal."""
    return D(balance)

def normalize_position_size(size):
    """Normalize position size to Decimal."""
    return D(size)

# Financial calculation helpers with Decimal discipline
def calculate_risk_amount(portfolio_value, risk_percentage):
    """Calculate risk amount with Decimal precision."""
    return D(portfolio_value) * D(risk_percentage)

def calculate_position_value(size, price):
    """Calculate position value with Decimal precision."""
    return D(size) * D(price)

def calculate_pnl(current_price, entry_price, size, side):
    """Calculate PnL with Decimal precision."""
    price_diff = D(current_price) - D(entry_price)
    if side.lower() == 'long':
        return price_diff * D(size)
    else:  # short
        return -price_diff * D(size)

def calculate_leverage(position_value, margin_used):
    """Calculate leverage with Decimal precision."""
    if D(margin_used) == 0:
        return D(0)
    return D(position_value) / D(margin_used)

def calculate_margin_ratio(position_value, account_value):
    """Calculate margin ratio with Decimal precision."""
    if D(account_value) == 0:
        return D(0)
    return D(position_value) / D(account_value)
