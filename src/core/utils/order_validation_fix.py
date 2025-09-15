import math

def validate_and_fix_order(asset, size, price, order_type="market"):
    """Validate and fix order parameters"""
    
    # Asset-specific tick sizes (from Hyperliquid docs)
    tick_sizes = {
        "BTC": 0.5,
        "ETH": 0.05,
        "SOL": 0.01,
        "DOGE": 0.0001,
        "AVAX": 0.01,
        "DYDX": 0.001,
        "APE": 0.001,
        "OP": 0.001,
        "LTC": 0.01,
        "ARB": 0.001,
        "INJ": 0.001,
        "CRV": 0.001,
        "LDO": 0.001,
        "LINK": 0.001,
        "STX": 0.001,
        "CFX": 0.001,
        "SNX": 0.001,
        "WLD": 0.001,
        "YGG": 0.001,
        "TRX": 0.0001,
        "UNI": 0.001,
        "SEI": 0.001
    }
    
    # Asset-specific min sizes
    min_sizes = {
        "BTC": 0.001,
        "ETH": 0.01,
        "SOL": 0.1,
        "DOGE": 1.0,
        "AVAX": 0.1,
        "DYDX": 0.1,
        "APE": 0.1,
        "OP": 0.1,
        "LTC": 0.01,
        "ARB": 0.1,
        "INJ": 0.1,
        "CRV": 0.1,
        "LDO": 0.1,
        "LINK": 0.1,
        "STX": 0.1,
        "CFX": 0.1,
        "SNX": 0.1,
        "WLD": 0.1,
        "YGG": 0.1,
        "TRX": 1.0,
        "UNI": 0.1,
        "SEI": 0.1
    }
    
    # Get tick size for asset
    tick_size = tick_sizes.get(asset, 0.001)
    min_size = min_sizes.get(asset, 0.1)
    
    # Fix price to tick size
    if order_type == "limit":
        price = round(price / tick_size) * tick_size
    
    # Fix size to minimum
    if size < min_size:
        size = min_size
    
    # Round size to appropriate precision
    if asset in ["BTC", "ETH"]:
        size = round(size, 3)
    elif asset in ["SOL", "AVAX", "LTC"]:
        size = round(size, 1)
    else:
        size = round(size, 1)
    
    return {
        "asset": asset,
        "size": size,
        "price": price,
        "tick_size": tick_size,
        "min_size": min_size,
        "valid": True
    }

def calculate_safe_order_size(asset, available_margin, price, max_risk_pct=0.1):
    """Calculate safe order size based on available margin"""
    
    # Conservative position sizing
    max_position_value = available_margin * max_risk_pct
    
    # Account for leverage and fees
    effective_margin = max_position_value * 0.8  # 20% buffer
    
    # Calculate size
    size = effective_margin / price
    
    # Validate and fix size
    result = validate_and_fix_order(asset, size, price)
    
    return result["size"]

def emergency_margin_recovery_strategy(positions, available_margin, target_margin):
    """Enhanced emergency margin recovery"""
    
    recovery_plan = []
    
    # Sort positions by P&L (worst first)
    sorted_positions = sorted(positions, key=lambda x: x.get('pnl', 0))
    
    for pos in sorted_positions:
        if available_margin >= target_margin:
            break
            
        asset = pos.get('asset')
        size = pos.get('size', 0)
        value = pos.get('value', 0)
        pnl = pos.get('pnl', 0)
        
        # Calculate how much to close
        close_pct = min(0.5, (target_margin - available_margin) / value)
        close_size = size * close_pct
        
        if close_size > 0:
            recovery_plan.append({
                "asset": asset,
                "action": "partial_close",
                "size": close_size,
                "reason": f"Emergency margin recovery - P&L: {pnl}"
            })
            
            available_margin += value * close_pct
    
    return recovery_plan
