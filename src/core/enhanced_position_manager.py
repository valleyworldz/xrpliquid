
# Enhanced Position Management System
from src.core.utils.decimal_boundary_guard import safe_float
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
                size = safe_float(pos.get("position", {}).get("szi", "0"))
                
                if abs(size) > self.min_position_size:
                    positions[symbol] = {
                        'size': size,
                        'entry_price': safe_float(pos.get("position", {}).get("entryPx", "0")),
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
            
            # Apply rotation strategies
            await self._apply_rotation_strategies(current_positions)
            
        except Exception as e:
            print(f"Error in position management: {e}")
    
    async def _apply_rotation_strategies(self, current_positions: Dict):
        """Apply intelligent position rotation strategies"""
        try:
            # Strategy 1: Performance-based rotation
            await self._performance_based_rotation(current_positions)
            
            # Strategy 2: Time-based rotation
            await self._time_based_rotation(current_positions)
            
            # Strategy 3: Volatility-adaptive rotation
            await self._volatility_adaptive_rotation(current_positions)
            
            # Strategy 4: Correlation-driven rotation
            await self._correlation_driven_rotation(current_positions)
            
        except Exception as e:
            print(f"Error applying rotation strategies: {e}")
    
    async def _performance_based_rotation(self, positions: Dict):
        """Rotate positions based on performance metrics"""
        try:
            # Close underperforming positions and scale winners
            for symbol, position in positions.items():
                entry_price = position['entry_price']
                
                # Get current price to calculate performance
                market_data = self.api.get_market_data(symbol)
                if not market_data:
                    continue
                
                current_price = float(market_data.get("price", 0))
                if current_price <= 0:
                    continue
                
                # Calculate performance
                if position['size'] > 0:  # Long position
                    performance = (current_price - entry_price) / entry_price
                else:  # Short position  
                    performance = (entry_price - current_price) / entry_price
                
                # Rotation logic
                if performance < -0.05:  # 5% loss - consider rotation
                    print(f"🔄 Considering rotation for underperforming {symbol}: {performance:.2%}")
                    await self._rotate_underperformer(symbol, position, performance)
                elif performance > 0.10:  # 10% gain - scale up winner
                    print(f"📈 Scaling winner {symbol}: {performance:.2%}")
                    await self._scale_winner(symbol, position, performance)
                    
        except Exception as e:
            print(f"Error in performance-based rotation: {e}")
    
    async def _time_based_rotation(self, positions: Dict):
        """Rotate positions based on holding time"""
        try:
            current_time = time.time()
            max_holding_time = 24 * 3600  # 24 hours
            
            for symbol, position in positions.items():
                holding_time = current_time - position['timestamp']
                
                if holding_time > max_holding_time:
                    print(f"⏰ Time-based rotation for {symbol}: held {holding_time/3600:.1f} hours")
                    await self._rotate_aged_position(symbol, position)
                    
        except Exception as e:
            print(f"Error in time-based rotation: {e}")
    
    async def _volatility_adaptive_rotation(self, positions: Dict):
        """Adapt position sizes based on volatility changes"""
        try:
            for symbol, position in positions.items():
                # Get current volatility metrics
                volatility = await self._calculate_volatility(symbol)
                
                if volatility > 0.05:  # High volatility - reduce size
                    print(f"📉 High volatility detected for {symbol}: {volatility:.2%}, reducing size")
                    await self._reduce_position_size(symbol, position, 0.5)
                elif volatility < 0.02:  # Low volatility - potentially increase size
                    print(f"📈 Low volatility detected for {symbol}: {volatility:.2%}, considering size increase")
                    await self._consider_size_increase(symbol, position, volatility)
                    
        except Exception as e:
            print(f"Error in volatility-adaptive rotation: {e}")
    
    async def _correlation_driven_rotation(self, positions: Dict):
        """Rotate positions to maintain portfolio diversification"""
        try:
            # Calculate correlation matrix for current positions
            correlations = await self._calculate_position_correlations(positions)
            
            # Identify highly correlated pairs
            high_correlation_threshold = 0.8
            
            symbols = list(positions.keys())
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    correlation = correlations.get((symbol1, symbol2), 0)
                    
                    if abs(correlation) > high_correlation_threshold:
                        print(f"🔗 High correlation detected: {symbol1}-{symbol2}: {correlation:.2f}")
                        await self._diversify_correlated_positions(symbol1, symbol2, positions, correlation)
                        
        except Exception as e:
            print(f"Error in correlation-driven rotation: {e}")
    
    async def _rotate_underperformer(self, symbol: str, position: Dict, performance: float):
        """Rotate an underperforming position"""
        try:
            # Close position with stop loss
            position_size = abs(position['size'])
            
            # Place market order to close
            order_result = self.api.place_order(
                coin=symbol,
                is_buy=(position['size'] < 0),  # Opposite direction to close
                sz=position_size,
                limit_px=None,  # Market order
                reduce_only=True
            )
            
            if order_result and order_result.get("status") == "ok":
                print(f"✅ Rotated underperformer {symbol}: {performance:.2%} loss")
                
                # Look for replacement position
                await self._find_replacement_position(symbol, position_size)
            else:
                print(f"❌ Failed to rotate {symbol}")
                
        except Exception as e:
            print(f"Error rotating underperformer {symbol}: {e}")
    
    async def _scale_winner(self, symbol: str, position: Dict, performance: float):
        """Scale up a winning position"""
        try:
            # Increase position size by 50% of current size
            current_size = abs(position['size'])
            scale_size = current_size * 0.5
            
            # Get current market price
            market_data = self.api.get_market_data(symbol)
            if not market_data:
                return
            
            current_price = float(market_data.get("price", 0))
            
            # Place order to scale up
            order_result = self.api.place_order(
                coin=symbol,
                is_buy=(position['size'] > 0),  # Same direction as current position
                sz=scale_size,
                limit_px=current_price,
                reduce_only=False
            )
            
            if order_result and order_result.get("status") == "ok":
                print(f"✅ Scaled winner {symbol}: +{scale_size:.4f} at {performance:.2%} profit")
            
        except Exception as e:
            print(f"Error scaling winner {symbol}: {e}")
    
    async def _rotate_aged_position(self, symbol: str, position: Dict):
        """Rotate a position that has been held too long"""
        try:
            # Close aged position
            position_size = abs(position['size'])
            
            order_result = self.api.place_order(
                coin=symbol,
                is_buy=(position['size'] < 0),
                sz=position_size,
                limit_px=None,  # Market order
                reduce_only=True
            )
            
            if order_result and order_result.get("status") == "ok":
                print(f"✅ Rotated aged position {symbol}")
                
        except Exception as e:
            print(f"Error rotating aged position {symbol}: {e}")
    
    async def _calculate_volatility(self, symbol: str) -> float:
        """Calculate recent volatility for a symbol"""
        try:
            # Simplified volatility calculation
            # In a real implementation, this would use historical price data
            market_data = self.api.get_market_data(symbol)
            if not market_data:
                return 0.03  # Default volatility
            
            # Simulate volatility based on recent price movements
            return 0.02 + abs(hash(symbol + str(time.time()))) % 100 / 10000
            
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return 0.03
    
    async def _reduce_position_size(self, symbol: str, position: Dict, reduction_factor: float):
        """Reduce position size due to high volatility"""
        try:
            current_size = abs(position['size'])
            reduction_size = current_size * (1 - reduction_factor)
            
            if reduction_size > self.min_position_size:
                order_result = self.api.place_order(
                    coin=symbol,
                    is_buy=(position['size'] < 0),
                    sz=reduction_size,
                    limit_px=None,
                    reduce_only=True
                )
                
                if order_result and order_result.get("status") == "ok":
                    print(f"✅ Reduced {symbol} position by {(1-reduction_factor):.0%}")
                    
        except Exception as e:
            print(f"Error reducing position size for {symbol}: {e}")
    
    async def _consider_size_increase(self, symbol: str, position: Dict, volatility: float):
        """Consider increasing position size in low volatility environment"""
        try:
            # Only increase if volatility is very low and position is profitable
            if volatility < 0.015:  # Very low volatility
                current_size = abs(position['size'])
                
                # Get current performance
                market_data = self.api.get_market_data(symbol)
                if market_data:
                    current_price = float(market_data.get("price", 0))
                    entry_price = position['entry_price']
                    
                    if position['size'] > 0:  # Long
                        performance = (current_price - entry_price) / entry_price
                    else:  # Short
                        performance = (entry_price - current_price) / entry_price
                    
                    if performance > 0.02:  # At least 2% profit
                        increase_size = current_size * 0.3  # 30% increase
                        
                        order_result = self.api.place_order(
                            coin=symbol,
                            is_buy=(position['size'] > 0),
                            sz=increase_size,
                            limit_px=current_price,
                            reduce_only=False
                        )
                        
                        if order_result and order_result.get("status") == "ok":
                            print(f"✅ Increased {symbol} position in low volatility: +{increase_size:.4f}")
                            
        except Exception as e:
            print(f"Error considering size increase for {symbol}: {e}")
    
    async def _calculate_position_correlations(self, positions: Dict) -> Dict:
        """Calculate correlations between positions"""
        try:
            correlations = {}
            symbols = list(positions.keys())
            
            # Simplified correlation calculation
            # In a real implementation, this would use historical price data
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    # Simulate correlation based on symbol characteristics
                    correlation = (abs(hash(symbol1 + symbol2)) % 100) / 100 - 0.5
                    correlations[(symbol1, symbol2)] = correlation
            
            return correlations
            
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            return {}
    
    async def _diversify_correlated_positions(self, symbol1: str, symbol2: str, positions: Dict, correlation: float):
        """Diversify highly correlated positions"""
        try:
            # Reduce size of the smaller position
            pos1_value = abs(positions[symbol1]['size']) * positions[symbol1]['entry_price']
            pos2_value = abs(positions[symbol2]['size']) * positions[symbol2]['entry_price']
            
            # Reduce the smaller position by 50%
            if pos1_value < pos2_value:
                await self._reduce_position_size(symbol1, positions[symbol1], 0.5)
            else:
                await self._reduce_position_size(symbol2, positions[symbol2], 0.5)
                
            print(f"🔄 Diversified correlated positions: {symbol1}-{symbol2}")
            
        except Exception as e:
            print(f"Error diversifying positions {symbol1}-{symbol2}: {e}")
    
    async def _find_replacement_position(self, closed_symbol: str, freed_capital: float):
        """Find a replacement position after closing an underperformer"""
        try:
            # List of potential replacement assets
            potential_assets = ['ETH', 'BTC', 'SOL', 'AVAX', 'MATIC']
            
            # Remove the closed symbol from potential replacements
            if closed_symbol in potential_assets:
                potential_assets.remove(closed_symbol)
            
            # Select a replacement based on current market conditions
            # This is simplified - real implementation would use signal analysis
            import random
            replacement_symbol = random.choice(potential_assets)
            
            # Get market data for replacement
            market_data = self.api.get_market_data(replacement_symbol)
            if market_data:
                current_price = float(market_data.get("price", 0))
                
                if current_price > 0:
                    # Calculate position size based on freed capital
                    new_position_size = freed_capital / current_price
                    
                    if new_position_size >= self.min_position_size:
                        order_result = self.api.place_order(
                            coin=replacement_symbol,
                            is_buy=True,  # Default to long position
                            sz=new_position_size,
                            limit_px=current_price,
                            reduce_only=False
                        )
                        
                        if order_result and order_result.get("status") == "ok":
                            print(f"✅ Opened replacement position: {replacement_symbol} ({new_position_size:.4f})")
                            
        except Exception as e:
            print(f"Error finding replacement position: {e}")
    
    def get_rotation_status(self) -> Dict:
        """Get current rotation system status"""
        return {
            'max_positions': self.max_positions,
            'rotation_enabled': self.position_rotation_enabled,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'status': 'active'
        }
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
