
# Enhanced Position Management System
import asyncio
import time
from typing import Dict, List, Optional

class EnhancedPositionManager:
    def __init__(self, api, config):
        self.api = api
        self.config = config
        self.positions = {}
        self.max_positions = 3
        self.position_rotation_enabled = True
        self.min_position_size = 0.001
        self.max_position_size = 1.0
        
    async def get_current_positions(self):
        """Get current positions from API"""
        try:
            user_state = self.api.get_user_state()
            if not user_state:
                return {}
            
            positions = {}
            asset_positions = user_state.get("assetPositions", [])
            
            for pos in asset_positions:
                symbol = pos.get("position", {}).get("coin", "")
                size = float(pos.get("position", {}).get("szi", "0"))
                
                if abs(size) > self.min_position_size:
                    positions[symbol] = {
                        'size': size,
                        'entry_price': float(pos.get("position", {}).get("entryPx", "0")),
                        'timestamp': time.time()
                    }
            
            return positions
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}
    
    async def manage_positions(self):
        """Enhanced position management with rotation"""
        try:
            current_positions = await self.get_current_positions()
            
            if len(current_positions) >= self.max_positions and self.position_rotation_enabled:
                await self.rotate_positions(current_positions)
            
            await self.validate_position_sizes(current_positions)
            return True
        except Exception as e:
            print(f"Position management error: {e}")
            return False
    
    async def rotate_positions(self, positions):
        """Close oldest positions when new opportunities arise"""
        try:
            sorted_positions = sorted(positions.items(), key=lambda x: x[1]['timestamp'])
            positions_to_close = len(positions) - self.max_positions + 1
            
            for i in range(positions_to_close):
                if i < len(sorted_positions):
                    symbol, position = sorted_positions[i]
                    await self.close_position(symbol, position)
        except Exception as e:
            print(f"Position rotation error: {e}")
    
    async def close_position(self, symbol, position):
        """Close position with proper reduce_only order"""
        try:
            if abs(position['size']) < self.min_position_size:
                return True
            
            order = {
                'symbol': symbol,
                'side': 'sell' if position['size'] > 0 else 'buy',
                'size': abs(position['size']),
                'reduce_only': True,
                'type': 'market'
            }
            
            result = await self.api.place_order(order)
            if result:
                print(f"Position closed: {symbol}")
                return True
            else:
                print(f"Failed to close position: {symbol}")
                return False
        except Exception as e:
            print(f"Error closing position {symbol}: {e}")
            return False
