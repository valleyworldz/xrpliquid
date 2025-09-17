#!/usr/bin/env python3
"""
💰 SPOT TRADING EXECUTION HANDLER
=================================

Dedicated handler for spot trading operations.
Handles simple buy/sell operations without leverage or margin.

Features:
- Simple balance management
- No leverage calculations
- Direct spot order execution
- Position tracking
"""

from src.core.utils.decimal_boundary_guard import safe_float
import time
from typing import Dict, Any, Optional
from core.api.hyperliquid_api import HyperliquidAPI
from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager

class SpotExecutor:
    """
    Spot Trading Execution Handler
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = Logger()
        self.api = HyperliquidAPI()
        self.active_positions = {}
        self.position_history = []
        
    def get_balance(self, token: str) -> float:
        """
        Get spot balance for a token.
        
        Args:
            token: Token symbol
            
        Returns:
            float: Available balance
        """
        try:
            user_state = self.api.get_user_state()
            if user_state and "assetPositions" in user_state:
                for position in user_state["assetPositions"]:
                    if position.get("coin") == token:
                        return safe_float(position.get("szi", "0"))
            return 0.0
        except Exception as e:
            self.logger.error(f"[SPOT] Error getting balance for {token}: {e}")
            return 0.0
    
    def place_spot_order(self, token: str, side: str, quantity: float, price: Optional[float] = None,
                        order_type: str = "market", take_profit_pct: float = 0.02,
                        stop_loss_pct: float = 0.01) -> Dict[str, Any]:
        """
        Place a spot trading order.
        
        Args:
            token: Token symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Order price (None for market orders)
            order_type: "market" or "limit"
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Dict containing order response
        """
        try:
            self.logger.info(f"[SPOT] Placing {order_type} order: {side} {quantity} {token}")
            
            # Check balance for sell orders
            if side == "sell":
                balance = self.get_balance(token)
                if balance < quantity:
                    self.logger.error(f"[SPOT] Insufficient balance: {balance} < {quantity}")
                    return {"status": "failed", "error": "Insufficient balance"}
            
            # Validate order
            asset_id, coin_name = self.api.resolve_symbol_to_asset_id(token)
            if asset_id is None:
                self.logger.error(f"Could not resolve symbol {token} to asset ID")
                return {"status": "failed", "error": f"Invalid symbol: {token}"}
            try:
                quantity, price = self.api.validate_and_round_order(asset_id, quantity, price)
            except Exception as e:
                self.logger.error(f"Order validation failed for {token}: {e}")
                return {"status": "failed", "error": str(e)}
            
            # Prepare order
            order = {
                "symbol": token,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "reduce_only": False
            }
            
            if price is not None and order_type == "limit":
                order["price"] = price
            
            # Place order
            response = self.api.place_order(order)
            
            if response and response.get('success'):
                self.logger.info(f"Order placed successfully. Order ID: {response.get('order_id', 'Unknown')}")
                if response.get('filled_immediately'):
                    self.logger.info(f"Order filled immediately. Quantity: {response.get('quantity')} Price: ${response.get('price')}")
                else:
                    self.logger.info(f"Order status: {response.get('status', 'resting')}")
                
                # Track position if it's a buy order
                if side == "buy":
                    entry_price = price if price is not None else response.get("price", 0)
                    position_info = {
                        "token": token,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "entry_time": time.time(),
                        "order_id": response.get("order_id"),
                        "take_profit_pct": take_profit_pct,
                        "stop_loss_pct": stop_loss_pct,
                        "status": "open"
                    }
                    
                    position_key = f"{token}_{side}_{response.get('order_id')}"
                    self.active_positions[position_key] = position_info
                    
                    # Place TP/SL orders if specified
                    if take_profit_pct > 0 or stop_loss_pct > 0:
                        self._place_spot_tp_sl(token, quantity, position_info)
                
                return {"status": "success", "response": response}
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                self.logger.error(f"Order failed: {error_msg}")
                return {"status": "failed", "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"[SPOT] Error placing order: {e}")
            return {"status": "error", "error": str(e)}
    
    def _place_spot_tp_sl(self, token: str, quantity: float, position_info: Dict[str, Any]) -> bool:
        """
        Place take-profit and stop-loss orders for spot position.
        
        Args:
            token: Token symbol
            quantity: Position quantity
            position_info: Position information
            
        Returns:
            bool: True if orders placed successfully
        """
        try:
            entry_price = position_info["entry_price"]
            tp_pct = position_info["take_profit_pct"]
            sl_pct = position_info["stop_loss_pct"]
            
            # Calculate TP/SL prices
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            
            # Place take-profit order
            if tp_pct > 0:
                tp_order = {
                    "symbol": token,
                    "side": "sell",
                    "quantity": quantity,
                    "price": tp_price,
                    "order_type": "limit",
                    "reduce_only": True
                }
                tp_response = self.api.place_order(tp_order)
                if tp_response:
                    self.logger.info(f"[SPOT] TP order placed: {tp_price:.4f}")
            
            # Place stop-loss order
            if sl_pct > 0:
                sl_order = {
                    "symbol": token,
                    "side": "sell",
                    "quantity": quantity,
                    "price": sl_price,
                    "order_type": "limit",
                    "reduce_only": True
                }
                sl_response = self.api.place_order(sl_order)
                if sl_response:
                    self.logger.info(f"[SPOT] SL order placed: {sl_price:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[SPOT] Error placing TP/SL orders: {e}")
            return False
    
    def close_spot_position(self, token: str, quantity: Optional[float] = None) -> bool:
        """
        Close a spot position.
        
        Args:
            token: Token symbol
            quantity: Quantity to sell (None for all)
            
        Returns:
            bool: True if position closed successfully
        """
        try:
            # Get current balance
            balance = self.get_balance(token)
            if balance <= 0:
                self.logger.warning(f"[SPOT] No {token} balance to close")
                return False
            
            # Use specified quantity or all balance
            sell_quantity = quantity if quantity is not None else balance
            
            # Place sell order
            response = self.api.place_order(
                symbol=token,
                side="sell",
                quantity=sell_quantity,
                order_type="market",
                reduce_only=True
            )
            
            # Log the full response for debugging
            self.logger.info(f"[SPOT] Close order response for {token}: {response}")
            
            if response and response.get("success") == True:
                # Try to extract order_id with fallback
                try:
                    order_id = response.get("order_id") or response.get("orderId") or response.get("id")
                    if order_id:
                        self.logger.info(f"[SPOT] Successfully closed {token} position with order ID: {order_id}")
                    else:
                        self.logger.warning(f"[SPOT] Close successful but no order ID found in response: {response}")
                except KeyError as e:
                    self.logger.error(f"[SPOT] KeyError accessing order ID in response: {e}")
                    self.logger.error(f"[SPOT] Full response: {response}")
                    # Still return True if the order was successful
                
                return True
            else:
                self.logger.error(f"[SPOT] Failed to close position: {response}")
                
                # Try market order fallback
                self.logger.warning(f"[SPOT] Retrying {token} close as MARKET order")
                return self._close_spot_position_market_fallback(token, sell_quantity)
                
        except Exception as e:
            self.logger.error(f"[SPOT] Error closing position: {e}")
            # Try market order fallback on any exception
            return self._close_spot_position_market_fallback(token, sell_quantity)
    
    def _close_spot_position_market_fallback(self, token: str, quantity: float) -> bool:
        """Fallback method to close spot position with market order"""
        try:
            self.logger.info(f"[SPOT] Attempting market order fallback for {token}")
            
            # Place market order with reduce_only
            response = self.api.place_order(
                symbol=token,
                side="sell",
                quantity=quantity,
                order_type="market",
                reduce_only=True
            )
            
            # Log the full response for debugging
            self.logger.info(f"[SPOT] Market fallback response for {token}: {response}")
            
            if response and response.get("success") == True:
                self.logger.info(f"[SPOT] Market fallback successful for {token}")
                return True
            else:
                self.logger.error(f"[SPOT] Market fallback failed for {token}: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"[SPOT] Error in market fallback for {token}: {e}")
            return False
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value in USD.
        
        Returns:
            float: Total portfolio value
        """
        try:
            total_value = 0.0
            user_state = self.api.get_user_state()
            
            if user_state and "assetPositions" in user_state:
                for position in user_state["assetPositions"]:
                    token = position.get("coin")
                    quantity = safe_float(position.get("szi", "0"))
                    
                    if quantity > 0:
                        # Get current price
                        market_data = self.api.get_market_data(token)
                        if market_data and "price" in market_data:
                            price = market_data["price"]
                            value = quantity * price
                            total_value += value
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"[SPOT] Error calculating portfolio value: {e}")
            return 0.0
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get spot trading position summary.
        
        Returns:
            Dict containing position summary
        """
        try:
            portfolio_value = self.get_portfolio_value()
            positions = {}
            
            user_state = self.api.get_user_state()
            if user_state and "assetPositions" in user_state:
                for position in user_state["assetPositions"]:
                    token = position.get("coin")
                    quantity = safe_float(position.get("szi", "0"))
                    
                    if quantity > 0:
                        market_data = self.api.get_market_data(token)
                        price = market_data.get("price", 0) if market_data else 0
                        value = quantity * price
                        
                        positions[token] = {
                            "quantity": quantity,
                            "price": price,
                            "value": value,
                            "weight": value / portfolio_value if portfolio_value > 0 else 0
                        }
            
            return {
                "portfolio_value": portfolio_value,
                "positions": positions,
                "active_positions": len(self.active_positions),
                "position_history": len(self.position_history)
            }
            
        except Exception as e:
            self.logger.error(f"[SPOT] Error getting position summary: {e}")
            return {}
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all active spot orders.
        
        Returns:
            bool: True if orders cancelled successfully
        """
        try:
            open_orders = self.api.get_open_orders()
            cancelled_count = 0
            
            for order in open_orders:
                response = self.api.cancel_order(order["oid"], order["coin"])
                if response and response.get("status") == "ok":
                    cancelled_count += 1
            
            self.logger.info(f"[SPOT] Cancelled {cancelled_count} orders")
            return True
            
        except Exception as e:
            self.logger.error(f"[SPOT] Error cancelling orders: {e}")
            return False 