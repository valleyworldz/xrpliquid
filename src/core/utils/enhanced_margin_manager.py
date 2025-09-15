import time
import logging

class EnhancedMarginManager:
    def __init__(self, config):
        self.config = config
        self.min_margin_ratio = config.get('risk_management', {}).get('margin_management', {}).get('min_margin_ratio', 0.2)
        self.emergency_margin_ratio = config.get('risk_management', {}).get('margin_management', {}).get('emergency_margin_ratio', 0.1)
        self.force_close_threshold = config.get('risk_management', {}).get('margin_management', {}).get('force_close_threshold', 0.05)
        
    def check_margin_health(self, account_info):
        """Enhanced margin health check"""
        
        total_margin = account_info.get('total_margin', 0)
        used_margin = account_info.get('used_margin', 0)
        free_margin = account_info.get('free_margin', 0)
        
        if total_margin <= 0:
            return {"status": "critical", "message": "No margin available"}
        
        margin_ratio = free_margin / total_margin
        
        if margin_ratio <= self.emergency_margin_ratio:
            return {
                "status": "emergency",
                "message": f"Emergency margin level: {margin_ratio:.2%}",
                "action": "force_close_positions"
            }
        elif margin_ratio <= self.min_margin_ratio:
            return {
                "status": "warning",
                "message": f"Low margin level: {margin_ratio:.2%}",
                "action": "reduce_position_sizes"
            }
        else:
            return {
                "status": "healthy",
                "message": f"Margin healthy: {margin_ratio:.2%}",
                "action": "normal_trading"
            }
    
    def calculate_safe_position_size(self, asset, price, available_margin, risk_pct=0.1):
        """Calculate safe position size"""
        
        # Conservative sizing
        max_position_value = available_margin * risk_pct
        
        # Apply additional safety buffer
        safe_position_value = max_position_value * 0.8
        
        # Calculate size
        size = safe_position_value / price
        
        # Ensure minimum viable size
        min_size = self.get_min_size(asset)
        if size < min_size:
            size = min_size
            
        return size
    
    def get_min_size(self, asset):
        """Get minimum order size for asset"""
        min_sizes = {
            "BTC": 0.001,
            "ETH": 0.01,
            "SOL": 0.1,
            "DOGE": 1.0,
            "AVAX": 0.1
        }
        return min_sizes.get(asset, 0.1)
    
    def emergency_margin_recovery(self, positions, account_info):
        """Enhanced emergency margin recovery"""
        
        recovery_actions = []
        total_margin = account_info.get('total_margin', 0)
        free_margin = account_info.get('free_margin', 0)
        
        # Calculate how much margin we need to free up
        target_free_margin = total_margin * self.min_margin_ratio
        margin_needed = target_free_margin - free_margin
        
        if margin_needed <= 0:
            return recovery_actions
        
        # Sort positions by P&L (close losing positions first)
        sorted_positions = sorted(positions, key=lambda x: x.get('pnl', 0))
        
        for pos in sorted_positions:
            if margin_needed <= 0:
                break
                
            asset = pos.get('asset')
            size = pos.get('size', 0)
            value = pos.get('value', 0)
            pnl = pos.get('pnl', 0)
            
            # Calculate close amount
            close_pct = min(0.5, margin_needed / value)
            close_size = size * close_pct
            
            if close_size > 0:
                recovery_actions.append({
                    "asset": asset,
                    "action": "partial_close",
                    "size": close_size,
                    "reason": f"Emergency margin recovery - P&L: {pnl:.2f}"
                })
                
                margin_needed -= value * close_pct
        
        return recovery_actions
