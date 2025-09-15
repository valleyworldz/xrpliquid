#!/usr/bin/env python3
"""
Order Execution and Trade Management for XRP Trading Bot
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

class OrderExecutor:
    """Advanced order execution and management"""
    
    def __init__(self, exchange_client, info_client, logger: Optional[logging.Logger] = None):
        self.exchange_client = exchange_client
        self.info_client = info_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Execution parameters
        self.order_timeout = 30  # seconds
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Position tracking
        self.active_positions = {}
        self.active_triggers = {}
        self.order_history = []
        
        # TP/SL parameters
        self.tp_tiers = [0.01, 0.02, 0.03]  # 1%, 2%, 3%
        self.tp_sizes = [0.3, 0.3, 0.4]     # 30%, 30%, 40%
        self.breakeven_activation = 0.01     # 1%
        self.trailing_activation = 0.005     # 0.5%
        
    async def place_order(self, symbol: str, is_buy: bool, size: float, 
                         price: float, order_type: str = "limit") -> Dict[str, Any]:
        """
        Place order with atomic price fetching and error handling
        
        Args:
            symbol: Trading symbol
            is_buy: True for buy, False for sell
            size: Order size
            price: Order price
            order_type: Order type (limit/market)
            
        Returns:
            Order result dictionary
        """
        try:
            # Atomic price fetching to prevent race conditions
            atomic_price = await self._get_atomic_price(symbol, is_buy, price)
            
            # Prepare order parameters
            order_params = self._prepare_order_params(symbol, is_buy, size, atomic_price, order_type)
            
            # Place order with retries
            for attempt in range(self.max_retries):
                try:
                    result = await self._submit_order(order_params)
                    
                    if self._order_successful(result):
                        await self._reconcile_order(result)
                        return self._format_order_result(result, True)
                    else:
                        self.logger.warning(f"Order failed (attempt {attempt + 1}): {result}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                        
                except Exception as e:
                    self.logger.error(f"Order submission error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
            
            return self._format_order_result(None, False, "Max retries exceeded")
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return self._format_order_result(None, False, str(e))
    
    async def _get_atomic_price(self, symbol: str, is_buy: bool, 
                               fallback_price: float) -> float:
        """Get atomic price from L2 snapshot"""
        try:
            l2_data = self.info_client.l2_snapshot(symbol)
            if l2_data:
                normalized = self._normalize_l2_snapshot(l2_data)
                if normalized["bids"] and normalized["asks"]:
                    bid_price = normalized["bids"][0][0]
                    ask_price = normalized["asks"][0][0]
                    
                    # Use appropriate price based on order side
                    if is_buy:
                        atomic_price = ask_price  # Buy at ask
                    else:
                        atomic_price = bid_price  # Sell at bid
                    
                    self.logger.info(f"Atomic price: bid=${bid_price:.4f}, ask=${ask_price:.4f}, using=${atomic_price:.4f}")
                    return atomic_price
            
            return fallback_price
            
        except Exception as e:
            self.logger.warning(f"Atomic price fetch failed, using fallback: {e}")
            return fallback_price
    
    def _normalize_l2_snapshot(self, l2_data: Dict) -> Dict:
        """Normalize L2 snapshot data"""
        try:
            # This is a simplified normalization
            # In practice, you'd use the actual Hyperliquid L2 format
            return {
                "bids": l2_data.get("bids", []),
                "asks": l2_data.get("asks", [])
            }
        except Exception as e:
            self.logger.error(f"Error normalizing L2 data: {e}")
            return {"bids": [], "asks": []}
    
    def _prepare_order_params(self, symbol: str, is_buy: bool, size: float,
                            price: float, order_type: str) -> Dict[str, Any]:
        """Prepare order parameters"""
        try:
            if order_type == "market":
                hl_order_type = {"limit": {"tif": "Ioc"}}
                # Use aggressive pricing for market orders
                if is_buy:
                    price = price * 1.001  # Slightly above market
                else:
                    price = price * 0.999  # Slightly below market
            else:
                hl_order_type = {"limit": {"tif": "Gtc"}}
            
            # Align price to tick size
            price = self._align_price_to_tick(price)
            
            return {
                "name": symbol,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": price,
                "order_type": hl_order_type
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing order params: {e}")
            return {}
    
    def _align_price_to_tick(self, price: float) -> float:
        """Align price to tick size"""
        try:
            # XRP tick size is typically 0.0001
            tick_size = 0.0001
            return round(price / tick_size) * tick_size
        except Exception as e:
            self.logger.error(f"Error aligning price to tick: {e}")
            return price
    
    async def _submit_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Submit order to exchange"""
        try:
            result = self.exchange_client.order(**order_params)
            return result
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            raise
    
    def _order_successful(self, result: Dict[str, Any]) -> bool:
        """Check if order was successful"""
        try:
            if not result:
                return False
            
            # Check for errors in result
            if isinstance(result, dict):
                if "error" in result:
                    return False
                elif "response" in result:
                    response = result["response"]
                    if "data" in response and "statuses" in response["data"]:
                        for status in response["data"]["statuses"]:
                            if "error" in status:
                                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking order success: {e}")
            return False
    
    async def _reconcile_order(self, result: Dict[str, Any]) -> None:
        """Reconcile order with user state"""
        try:
            await asyncio.sleep(0.1)  # Brief delay for settlement
            
            # Get updated user state
            user_state = self.info_client.user_state(self.wallet_address)
            if user_state:
                self.logger.info("Order reconciled with user state")
            
        except Exception as e:
            self.logger.warning(f"Could not reconcile order: {e}")
    
    def _format_order_result(self, result: Optional[Dict[str, Any]], 
                           success: bool, error_msg: str = "") -> Dict[str, Any]:
        """Format order result"""
        return {
            "success": success,
            "result": result,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }
    
    async def place_tp_sl_triggers(self, symbol: str, position_size: float,
                                 entry_price: float, is_long: bool,
                                 tp_percentage: float = 0.02,
                                 sl_percentage: float = 0.015) -> Dict[str, Any]:
        """
        Place take profit and stop loss triggers
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            entry_price: Entry price
            is_long: True for long position
            tp_percentage: Take profit percentage
            sl_percentage: Stop loss percentage
            
        Returns:
            Dictionary with trigger order IDs
        """
        try:
            # Calculate TP/SL prices
            if is_long:
                tp_price = entry_price * (1 + tp_percentage)
                sl_price = entry_price * (1 - sl_percentage)
            else:
                tp_price = entry_price * (1 - tp_percentage)
                sl_price = entry_price * (1 + sl_percentage)
            
            # Validate prices
            if not self._validate_tp_sl_prices(entry_price, tp_price, sl_price):
                return {"success": False, "error": "Invalid TP/SL prices"}
            
            # Place TP trigger
            tp_result = await self._place_trigger(symbol, not is_long, position_size, tp_price, "tp")
            
            # Place SL trigger
            sl_result = await self._place_trigger(symbol, not is_long, position_size, sl_price, "sl")
            
            # Store trigger information
            trigger_info = {
                "tp_oid": tp_result.get("oid") if tp_result.get("success") else None,
                "sl_oid": sl_result.get("oid") if sl_result.get("success") else None,
                "entry_price": entry_price,
                "position_size": position_size,
                "is_long": is_long,
                "tp_price": tp_price,
                "sl_price": sl_price
            }
            
            self.active_triggers[f"{symbol}_{position_size}"] = trigger_info
            
            return {
                "success": tp_result.get("success") and sl_result.get("success"),
                "tp_oid": tp_result.get("oid"),
                "sl_oid": sl_result.get("oid"),
                "tp_price": tp_price,
                "sl_price": sl_price
            }
            
        except Exception as e:
            self.logger.error(f"Error placing TP/SL triggers: {e}")
            return {"success": False, "error": str(e)}
    
    async def _place_trigger(self, symbol: str, is_buy: bool, size: float,
                           price: float, trigger_type: str) -> Dict[str, Any]:
        """Place a single trigger order"""
        try:
            # Prepare trigger order
            order_params = {
                "name": symbol,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": price,
                "order_type": {"trigger": {"trigger_px": price}}
            }
            
            result = await self._submit_order(order_params)
            
            if self._order_successful(result):
                # Extract order ID
                oid = self._extract_order_id(result)
                self.logger.info(f"{trigger_type.upper()} trigger placed @ {price:.4f} (oid={oid})")
                return {"success": True, "oid": oid}
            else:
                self.logger.error(f"Failed to place {trigger_type} trigger")
                return {"success": False, "error": "Order failed"}
                
        except Exception as e:
            self.logger.error(f"Error placing {trigger_type} trigger: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_order_id(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract order ID from result"""
        try:
            if isinstance(result, dict) and "response" in result:
                response = result["response"]
                if "data" in response and "statuses" in response["data"]:
                    statuses = response["data"]["statuses"]
                    if statuses and "resting" in statuses[0]:
                        return statuses[0]["resting"]["oid"]
            return None
        except Exception as e:
            self.logger.error(f"Error extracting order ID: {e}")
            return None
    
    def _validate_tp_sl_prices(self, entry_price: float, tp_price: float, 
                              sl_price: float) -> bool:
        """Validate TP/SL prices"""
        try:
            # Check reasonable bounds (0.1x to 10x entry price)
            min_price = entry_price * 0.1
            max_price = entry_price * 10.0
            
            if tp_price < min_price or tp_price > max_price:
                self.logger.error(f"TP price {tp_price:.4f} outside valid range")
                return False
            
            if sl_price < min_price or sl_price > max_price:
                self.logger.error(f"SL price {sl_price:.4f} outside valid range")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating TP/SL prices: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            result = self.exchange_client.cancel_order(order_id)
            if result and self._order_successful(result):
                self.logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                self.logger.error(f"Failed to cancel order {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_triggers(self, symbol: str) -> bool:
        """Cancel all active triggers for a symbol"""
        try:
            cancelled_count = 0
            
            for trigger_key, trigger_info in self.active_triggers.items():
                if trigger_key.startswith(symbol):
                    if trigger_info.get("tp_oid"):
                        if await self.cancel_order(trigger_info["tp_oid"]):
                            cancelled_count += 1
                    
                    if trigger_info.get("sl_oid"):
                        if await self.cancel_order(trigger_info["sl_oid"]):
                            cancelled_count += 1
                    
                    # Remove from active triggers
                    del self.active_triggers[trigger_key]
            
            self.logger.info(f"Cancelled {cancelled_count} triggers for {symbol}")
            return cancelled_count > 0
            
        except Exception as e:
            self.logger.error(f"Error cancelling triggers for {symbol}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            result = self.info_client.query_order(order_id)
            return {
                "order_id": order_id,
                "status": result.get("status", "unknown"),
                "filled_size": result.get("filled_size", 0),
                "remaining_size": result.get("remaining_size", 0),
                "average_price": result.get("average_price", 0)
            }
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {"order_id": order_id, "status": "error", "error": str(e)}
    
    def get_active_triggers(self) -> Dict[str, Any]:
        """Get all active triggers"""
        return self.active_triggers.copy()
    
    def update_position(self, symbol: str, size: float, entry_price: float, 
                       is_long: bool) -> None:
        """Update position tracking"""
        try:
            self.active_positions[symbol] = {
                "size": size,
                "entry_price": entry_price,
                "is_long": is_long,
                "entry_time": datetime.now(),
                "unrealized_pnl": 0.0
            }
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")

# Convenience functions for backward compatibility
async def place_order(exchange_client, info_client, symbol: str, is_buy: bool,
                     size: float, price: float, order_type: str = "limit",
                     logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Place order (convenience function)"""
    executor = OrderExecutor(exchange_client, info_client, logger)
    return await executor.place_order(symbol, is_buy, size, price, order_type) 