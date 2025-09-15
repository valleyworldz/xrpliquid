import math
import time
import logging

class EnhancedOrderManager:
    def __init__(self, config):
        self.config = config
        self.max_retries = config.get('order_management', {}).get('max_retry_attempts', 2)
        self.retry_delay = config.get('order_management', {}).get('retry_delay', 1)
        self.dust_threshold = config.get('trading', {}).get('dust_position_threshold', 0.001)
        
    def validate_and_fix_order(self, asset, size, price, order_type="market"):
        """Enhanced order validation with dust handling"""
        
        # Asset-specific tick sizes
        tick_sizes = {
            "BTC": 0.5, "ETH": 0.05, "SOL": 0.01, "DOGE": 0.0001, "AVAX": 0.01,
            "DYDX": 0.001, "APE": 0.001, "OP": 0.001, "LTC": 0.01, "ARB": 0.001,
            "INJ": 0.001, "CRV": 0.001, "LDO": 0.001, "LINK": 0.001, "STX": 0.001,
            "CFX": 0.001, "SNX": 0.001, "WLD": 0.001, "YGG": 0.001, "TRX": 0.0001,
            "UNI": 0.001, "SEI": 0.001
        }
        
        # Asset-specific min sizes (reduced for low margin)
        min_sizes = {
            "BTC": 0.0001, "ETH": 0.001, "SOL": 0.01, "DOGE": 0.1, "AVAX": 0.01,
            "DYDX": 0.01, "APE": 0.01, "OP": 0.01, "LTC": 0.001, "ARB": 0.01,
            "INJ": 0.01, "CRV": 0.01, "LDO": 0.01, "LINK": 0.01, "STX": 0.01,
            "CFX": 0.01, "SNX": 0.01, "WLD": 0.01, "YGG": 0.01, "TRX": 0.1,
            "UNI": 0.01, "SEI": 0.01
        }
        
        tick_size = tick_sizes.get(asset, 0.001)
        min_size = min_sizes.get(asset, 0.01)
        
        # Check if this is a dust position
        if size < self.dust_threshold:
            return {
                "asset": asset,
                "size": size,
                "price": price,
                "valid": False,
                "reason": "dust_position",
                "action": "ignore"
            }
        
        # Fix price to tick size
        if order_type == "limit":
            price = round(price / tick_size) * tick_size
        
        # Fix size to minimum
        if size < min_size:
            size = min_size
        
        # Round size appropriately
        if asset in ["BTC", "ETH"]:
            size = round(size, 4)
        elif asset in ["SOL", "AVAX", "LTC"]:
            size = round(size, 2)
        else:
            size = round(size, 2)
        
        return {
            "asset": asset,
            "size": size,
            "price": price,
            "tick_size": tick_size,
            "min_size": min_size,
            "valid": True
        }
    
    def calculate_safe_order_size(self, asset, available_margin, price, max_risk_pct=0.05):
        """Calculate very conservative order size"""
        
        # Very conservative sizing for low margin
        max_position_value = available_margin * max_risk_pct
        
        # Apply additional safety buffer
        effective_margin = max_position_value * 0.5  # 50% buffer
        
        # Calculate size
        size = effective_margin / price
        
        # Validate and fix size
        result = self.validate_and_fix_order(asset, size, price)
        
        if not result["valid"]:
            return 0
        
        return result["size"]
    
    def emergency_margin_recovery(self, positions, account_info, max_attempts=2):
        """Enhanced emergency margin recovery with limits"""
        
        recovery_plan = []
        total_margin = account_info.get('total_margin', 0)
        free_margin = account_info.get('free_margin', 0)
        
        # Calculate target margin
        target_free_margin = total_margin * 0.1  # 10% minimum
        margin_needed = target_free_margin - free_margin
        
        if margin_needed <= 0:
            return recovery_plan
        
        # Sort positions by P&L (worst first)
        sorted_positions = sorted(positions, key=lambda x: x.get('pnl', 0))
        
        attempts = 0
        for pos in sorted_positions:
            if margin_needed <= 0 or attempts >= max_attempts:
                break
                
            asset = pos.get('asset')
            size = pos.get('size', 0)
            value = pos.get('value', 0)
            pnl = pos.get('pnl', 0)
            
            # Skip dust positions
            if size < self.dust_threshold:
                continue
            
            # Calculate close amount (more aggressive)
            close_pct = min(0.75, margin_needed / value)
            close_size = size * close_pct
            
            if close_size > 0:
                recovery_plan.append({
                    "asset": asset,
                    "action": "partial_close",
                    "size": close_size,
                    "reason": f"Emergency margin recovery - P&L: {pnl:.2f}"
                })
                
                margin_needed -= value * close_pct
                attempts += 1
        
        return recovery_plan 