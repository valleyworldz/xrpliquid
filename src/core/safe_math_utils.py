
# Safe Mathematical Operations
def safe_division(numerator, denominator, default=0.0):
    """Safe division with zero check"""
    try:
        if denominator == 0 or abs(denominator) < 1e-10:
            return default
        return numerator / denominator
    except Exception:
        return default

def safe_calculate_pnl_percentage(current_price, entry_price, default=0.0):
    """Safe P&L percentage calculation"""
    try:
        if entry_price == 0 or abs(entry_price) < 1e-10:
            return default
        return ((current_price - entry_price) / entry_price) * 100
    except Exception:
        return default

def safe_round_quantity(quantity, lot_size=0.001):
    """Safe quantity rounding"""
    try:
        if quantity <= 0:
            return 0.0
        return round(quantity / lot_size) * lot_size
    except Exception:
        return 0.0

def safe_calculate_margin_requirement(size, price, leverage=10):
    """Safe margin requirement calculation"""
    try:
        if size <= 0 or price <= 0 or leverage <= 0:
            return 0.0
        return (size * price) / leverage
    except Exception:
        return 0.0
