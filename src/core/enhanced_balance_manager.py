
# Enhanced Balance Calculation System
class EnhancedBalanceManager:
    def __init__(self, api):
        self.api = api
        self.safety_buffer = 0.9
        self.leverage = 10
        
    def calculate_available_balance(self, account_value, open_positions):
        """Calculate available balance with proper margin requirements"""
        try:
            total_margin = 0
            for pos in open_positions.values():
                size = abs(pos['size'])
                entry_price = pos['entry_price']
                margin_required = size * entry_price / self.leverage
                total_margin += margin_required
            
            available_balance = (account_value * self.safety_buffer) - total_margin
            available_balance = max(available_balance, account_value * 0.1)
            
            return available_balance
        except Exception as e:
            print(f"Balance calculation error: {e}")
            return account_value * 0.5
    
    def validate_order_size(self, order_size, available_balance, symbol_price):
        """Validate order size against available balance"""
        try:
            required_margin = order_size * symbol_price / self.leverage
            return required_margin <= available_balance
        except Exception as e:
            print(f"Order validation error: {e}")
            return False
    
    def get_account_summary(self):
        """Get comprehensive account summary"""
        try:
            user_state = self.api.get_user_state()
            if not user_state:
                return None
            
            margin_summary = user_state.get("marginSummary", {})
            account_value = float(margin_summary.get("accountValue", "0"))
            
            positions = {}
            asset_positions = user_state.get("assetPositions", [])
            
            for pos in asset_positions:
                symbol = pos.get("position", {}).get("coin", "")
                size = float(pos.get("position", {}).get("szi", "0"))
                
                if abs(size) > 0.001:
                    positions[symbol] = {
                        'size': size,
                        'entry_price': float(pos.get("position", {}).get("entryPx", "0"))
                    }
            
            available_balance = self.calculate_available_balance(account_value, positions)
            
            return {
                'account_value': account_value,
                'available_balance': available_balance,
                'open_positions': len(positions),
                'positions': positions,
                'margin_used': account_value - available_balance
            }
        except Exception as e:
            print(f"Account summary error: {e}")
            return None
