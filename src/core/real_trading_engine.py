
# Real Trading Engine
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Optional

class RealTradingEngine:
    def __init__(self, api, config):
        self.api = api
        self.config = config
        self.trade_history = []
        self.active_orders = {}
        self.is_trading_enabled = True
        
    async def execute_real_trade(self, symbol, side, size, order_type="market"):
        """Execute real trade with proper validation"""
        try:
            if not self.is_trading_enabled:
                print("Trading is disabled")
                return False
            
            if not self.validate_order_parameters(symbol, side, size):
                return False
            
            order = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'type': order_type
            }
            
            result = await self.api.place_order(order)
            
            if result:
                trade_record = {
                    'timestamp': time.time(),
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'type': order_type,
                    'status': 'executed'
                }
                self.trade_history.append(trade_record)
                
                await self.save_trade_log(trade_record)
                
                print(f"Real trade executed: {symbol} {side} {size}")
                return True
            else:
                print(f"Trade execution failed: {symbol}")
                return False
                
        except Exception as e:
            print(f"Trade execution error: {e}")
            return False
    
    def validate_order_parameters(self, symbol, side, size):
        """Validate order parameters"""
        try:
            if not symbol or not side or size <= 0:
                return False
            if side not in ['buy', 'sell']:
                return False
            if size < 0.001:
                return False
            return True
        except Exception:
            return False
    
    async def save_trade_log(self, trade_record):
        """Save trade to log file"""
        try:
            os.makedirs('trade_logs', exist_ok=True)
            
            date_str = datetime.now().strftime('%Y%m%d')
            log_file = f'trade_logs/trades_{date_str}.json'
            
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except FileNotFoundError:
                logs = []
            
            logs.append(trade_record)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Trade log save error: {e}")
    
    def get_trade_history(self):
        """Get trade history"""
        return self.trade_history
    
    def enable_trading(self):
        """Enable trading"""
        self.is_trading_enabled = True
        print("Trading enabled")
    
    def disable_trading(self):
        """Disable trading"""
        self.is_trading_enabled = False
        print("Trading disabled")
