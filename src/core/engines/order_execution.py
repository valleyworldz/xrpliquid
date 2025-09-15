#!/usr/bin/env python3
"""
🎯 ENHANCED ORDER EXECUTION ENGINE
==================================

Advanced order execution with take-profit and stop-loss support.
Handles both spot and perpetual trading with proper exit management.

Features:
- Take-profit and stop-loss orders
- OCO (one-cancels-other) order support
- Position tracking and management
- Exit signal processing
"""

import time
from typing import Dict, Any, Optional, Tuple
from core.api.hyperliquid_api import HyperliquidAPI
from core.engines.token_metadata import TokenMetadata
from decimal import Decimal, ROUND_DOWN
from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager

class OrderExecutionEngine:
    """
    Enhanced Order Execution Engine with TP/SL Support
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = Logger()
        self.hyperliquid_api = HyperliquidAPI()
        self.token_metadata = TokenMetadata()
        self.active_positions = {}  # Track active positions with TP/SL
        self.position_history = []

    def _format_price(self, price, px_decimals):
        """
        Formats the price to the correct decimal precision based on tick_size.
        """
        if px_decimals is None or px_decimals == 0:
            self.hyperliquid_api.logger.warning(f"Tick size is invalid ({px_decimals}). Returning original price.")
            return price
        # Use quantize with ROUND_DOWN to ensure price is at or below the desired limit price
        return float(Decimal(str(price)).quantize(Decimal(str(1 / (10 ** px_decimals))), rounding=ROUND_DOWN))

    def _format_quantity(self, quantity, sz_decimals):
        """
        Format quantity to match exchange requirements.
        """
        if sz_decimals is None or sz_decimals == 0:
            self.hyperliquid_api.logger.warning(f"Min size is invalid ({sz_decimals}). Returning original quantity.")
            return quantity
        
        min_size = 1 / (10 ** sz_decimals)
        
        # Ensure quantity is at least min_size
        if quantity < min_size:
            self.hyperliquid_api.logger.warning(f"[SKIP] Quantity ({quantity}) too small, below min size ({min_size}).")
            return 0 # Or raise an error, depending on desired behavior
        
        # Round quantity to the nearest multiple of min_size
        return float(Decimal(str(round(quantity / min_size) * min_size)).quantize(Decimal(str(min_size))))

    def place_entry_order(self, token: str, side: str, size: float, entry_price: float,
                         take_profit_pct: float = 0.02, stop_loss_pct: float = 0.01,
                         order_type: str = "limit") -> Dict[str, Any]:
        """
        Place entry order with attached take-profit and stop-loss orders.
        
        Args:
            token: Token symbol
            side: "buy" or "sell"
            size: Position size
            entry_price: Entry price
            take_profit_pct: Take profit percentage (e.g., 0.02 for 2%)
            stop_loss_pct: Stop loss percentage (e.g., 0.01 for 1%)
            order_type: "market" or "limit"
            
        Returns:
            Dict containing order response and position tracking info
        """
        try:
            self.logger.info(f"[ORDER] Placing entry order: {side} {size} {token} @ {entry_price}")
            
            # Calculate TP/SL prices
            if side == "buy":
                tp_price = entry_price * (1 + take_profit_pct)
                sl_price = entry_price * (1 - stop_loss_pct)
            else:  # sell (short)
                tp_price = entry_price * (1 - take_profit_pct)
                sl_price = entry_price * (1 + stop_loss_pct)
            
            # Prepare entry order
            entry_order = {
                "symbol": token,
                "side": side,
                "quantity": size,
                "price": entry_price,
                "order_type": order_type,
                "reduce_only": False
            }
            
            # Place entry order
            entry_response = self.hyperliquid_api.place_order(entry_order)
            
            if entry_response and entry_response.get('success'):
                self.logger.info(f"Entry order placed successfully. Order ID: {entry_response.get('order_id', 'Unknown')}")
                if entry_response.get('filled_immediately'):
                    self.logger.info(f"Order filled immediately. Quantity: {entry_response.get('quantity')} Price: ${entry_response.get('price')}")
                else:
                    self.logger.info(f"Order status: {entry_response.get('status', 'resting')}")
                
                # Track position with TP/SL
                position_info = {
                    "token": token,
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "entry_time": time.time(),
                    "order_id": entry_response.get("order_id"),
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "take_profit_pct": take_profit_pct,
                    "stop_loss_pct": stop_loss_pct,
                    "status": "open"
                }
                
                # Store position tracking
                position_key = f"{token}_{side}_{entry_response.get('order_id')}"
                self.active_positions[position_key] = position_info
                
                # Place TP/SL orders as OCO
                self._place_tp_sl_orders(token, side, size, tp_price, sl_price, entry_response.get("order_id"))
                
                return {
                    "status": "success",
                    "entry_order": entry_response,
                    "position_info": position_info,
                    "tp_price": tp_price,
                    "sl_price": sl_price
                }
            else:
                error_msg = entry_response.get('error', 'Unknown error') if entry_response else 'No response'
                self.logger.error(f"Entry order failed: {error_msg}")
                return {"status": "failed", "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"[ORDER] Error placing entry order: {e}")
            return {"status": "error", "error": str(e)}
    
    def _place_tp_sl_orders(self, token: str, side: str, size: float, tp_price: float, 
                           sl_price: float, parent_order_id: str) -> bool:
        """
        Place take-profit and stop-loss orders as OCO (one-cancels-other).
        
        Args:
            token: Token symbol
            side: Original position side
            size: Position size
            tp_price: Take profit price
            sl_price: Stop loss price
            parent_order_id: ID of the parent entry order
            
        Returns:
            bool: True if orders placed successfully
        """
        try:
            # Determine exit side (opposite of entry)
            exit_side = "sell" if side == "buy" else "buy"
            
            # Place take-profit order
            tp_order = {
                "symbol": token,
                "side": exit_side,
                "quantity": size,
                "price": tp_price,
                "order_type": "limit",
                "reduce_only": True,
                "oco_group": f"oco_{parent_order_id}",
                "oco_id": f"tp_{parent_order_id}"
            }
            
            tp_response = self.hyperliquid_api.place_order(tp_order)
            
            # Place stop-loss order
            sl_order = {
                "symbol": token,
                "side": exit_side,
                "quantity": size,
                "price": sl_price,
                "order_type": "market",  # Market order for immediate execution
                "reduce_only": True,
                "oco_group": f"oco_{parent_order_id}",
                "oco_id": f"sl_{parent_order_id}"
            }
            
            sl_response = self.hyperliquid_api.place_order(sl_order)
            
            if tp_response and sl_response:
                self.logger.info(f"[ORDER] TP/SL orders placed. TP: {tp_price:.4f}, SL: {sl_price:.4f}")
                return True
            else:
                self.logger.error(f"[ORDER] Failed to place TP/SL orders. TP: {tp_response}, SL: {sl_response}")
                return False
                
        except Exception as e:
            self.logger.error(f"[ORDER] Error placing TP/SL orders: {e}")
            return False
    
    def process_exit_signal(self, token: str, signal_type: str = "manual") -> bool:
        """
        Process exit signal for a token.
        
        Args:
            token: Token symbol to exit
            signal_type: Type of exit signal ("manual", "tp", "sl", "time")
            
        Returns:
            bool: True if exit successful
        """
        try:
            # Find active positions for this token
            positions_to_close = []
            for key, position in self.active_positions.items():
                if position["token"] == token and position["status"] == "open":
                    positions_to_close.append((key, position))
            
            if not positions_to_close:
                self.logger.warning(f"[ORDER] No active positions found for {token}")
                return False
            
            success_count = 0
            for key, position in positions_to_close:
                # Close position with market order
                exit_order = {
                    "symbol": token,
                    "side": "sell" if position["side"] == "buy" else "buy",
                    "quantity": position["size"],
                    "price": 0,  # Market order
                    "order_type": "market",
                    "reduce_only": True
                }
                
                response = self.hyperliquid_api.place_order(exit_order)
                
                if response and response.get("status") == "ok":
                    # Update position status
                    position["status"] = "closed"
                    position["exit_time"] = time.time()
                    position["exit_signal"] = signal_type
                    
                    # Calculate PnL
                    if position["side"] == "buy":
                        pnl = (response.get("price", position["entry_price"]) - position["entry_price"]) * position["size"]
                    else:
                        pnl = (position["entry_price"] - response.get("price", position["entry_price"])) * position["size"]
                    
                    position["pnl"] = pnl
                    
                    # Move to history
                    self.position_history.append(position)
                    del self.active_positions[key]
                    
                    success_count += 1
                    self.logger.info(f"[ORDER] Position closed for {token}. PnL: {pnl:.4f}")
                else:
                    self.logger.error(f"[ORDER] Failed to close position for {token}: {response}")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"[ORDER] Error processing exit signal: {e}")
            return False
    
    def check_tp_sl_triggers(self, current_prices: Dict[str, float]) -> None:
        """
        Check if any active positions have triggered TP/SL conditions.
        
        Args:
            current_prices: Dict of current prices by token
        """
        try:
            for key, position in list(self.active_positions.items()):
                if position["status"] != "open":
                    continue
                
                token = position["token"]
                current_price = current_prices.get(token)
                
                if not current_price:
                    continue
                
                # Check TP/SL triggers
                if position["side"] == "buy":
                    if current_price >= position["tp_price"]:
                        self.logger.info(f"[ORDER] TP triggered for {token} @ {current_price}")
                        self.process_exit_signal(token, "tp")
                    elif current_price <= position["sl_price"]:
                        self.logger.info(f"[ORDER] SL triggered for {token} @ {current_price}")
                        self.process_exit_signal(token, "sl")
                else:  # sell (short)
                    if current_price <= position["tp_price"]:
                        self.logger.info(f"[ORDER] TP triggered for {token} @ {current_price}")
                        self.process_exit_signal(token, "tp")
                    elif current_price >= position["sl_price"]:
                        self.logger.info(f"[ORDER] SL triggered for {token} @ {current_price}")
                        self.process_exit_signal(token, "sl")
                        
        except Exception as e:
            self.logger.error(f"[ORDER] Error checking TP/SL triggers: {e}")
    
    def get_active_positions(self) -> Dict[str, Any]:
        """Get all active positions with TP/SL info"""
        return self.active_positions
    
    def get_position_history(self) -> list:
        """Get position history with PnL data"""
        return self.position_history
    
    def calculate_total_pnl(self) -> float:
        """Calculate total PnL from all closed positions"""
        total_pnl = 0.0
        for position in self.position_history:
            total_pnl += position.get("pnl", 0.0)
        return total_pnl
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary"""
        active_count = len([p for p in self.active_positions.values() if p["status"] == "open"])
        closed_count = len(self.position_history)
        total_pnl = self.calculate_total_pnl()
        
        return {
            "active_positions": active_count,
            "closed_positions": closed_count,
            "total_pnl": total_pnl,
            "positions": self.active_positions,
            "history": self.position_history
        }
    
    def cancel_all_orders(self) -> bool:
        """Cancel all active orders"""
        try:
            # Cancel TP/SL orders
            for key, position in self.active_positions.items():
                if "order_id" in position:
                    self.hyperliquid_api.cancel_order(position["order_id"], position["token"])
            
            self.logger.info("[ORDER] All orders cancelled")
            return True
            
        except Exception as e:
            self.logger.error(f"[ORDER] Error cancelling orders: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all active positions"""
        try:
            tokens_to_close = set()
            for position in self.active_positions.values():
                if position["status"] == "open":
                    tokens_to_close.add(position["token"])
            
            success = True
            for token in tokens_to_close:
                if not self.process_exit_signal(token, "emergency"):
                    success = False
            
            self.logger.info(f"[ORDER] All positions closed. Success: {success}")
            return success
            
        except Exception as e:
            self.logger.error(f"[ORDER] Error closing all positions: {e}")
            return False

    def execute_order(self, strategy_signal, risk_params):
        """
        Executes a trading order based on strategy signals and risk parameters.

        This method handles the complexities of order placement, including:
        - Determining optimal order type (market, limit, stop).
        - Calculating precise order size based on risk management rules.
        - Routing orders to the Hyperliquid API.
        - Handling order acknowledgments and potential errors.

        Args:
            strategy_signal (dict): Contains details of the trade signal (e.g., symbol, side, quantity, price, reduceOnly, cloid).
            risk_params (dict): Parameters from the risk management layer (e.g., max position size, stop-loss).

        Returns:
            dict: A dictionary containing order execution details (e.g., order_id, status, filled_price).
        """
        self.hyperliquid_api.logger.info(f"Executing order based on signal: {strategy_signal} and risk params: {risk_params}")

        # Extract necessary order details from strategy_signal
        symbol = strategy_signal.get("symbol")
        side = strategy_signal.get("side")
        quantity = strategy_signal.get("quantity")
        price = strategy_signal.get("price")
        order_type = strategy_signal.get("order_type", "limit") # Default to limit order
        reduce_only = strategy_signal.get("reduce_only", False) # Explicitly set reduceOnly
        cloid = strategy_signal.get("cloid", None) # Optional client order ID

        if not all([symbol, side, quantity, price]):
            self.hyperliquid_api.logger.error("Error: Missing required order parameters.")
            return {"status": "failed", "message": "Missing required order parameters"}

        # 1. Confirm Token Metadata Access & Validate Price Format
        asset_details = self.token_metadata.fetch_asset_details(symbol)
        if not asset_details:
            self.hyperliquid_api.logger.error(f"Could not fetch asset details for {symbol}. Cannot place order.")
            return {"status": "failed", "message": f"Could not fetch asset details for {symbol}"}

        tick_size = asset_details.get("tick_size")
        min_size = asset_details.get("minimum_order")

        # Log raw price and quantity before formatting
        self.hyperliquid_api.logger.info(f"[DEBUG] {symbol} signal -> raw price: {price}, raw qty: {quantity}, tick: {tick_size}, minSize: {min_size}")

        # Format price and quantity
        formatted_price = self._format_price(price, tick_size)
        formatted_quantity = self._format_quantity(quantity, min_size)

        if formatted_quantity == 0:
            self.hyperliquid_api.logger.warning(f"[SKIP] Formatted quantity is zero or too small for {symbol}. Cannot place order.")
            return {"status": "skipped", "message": "Formatted quantity is zero or too small"}

        # 4. Handle Insufficient Balance or Filters Gracefully
        user_state = self.hyperliquid_api.get_user_state()
        if user_state and "marginSummary" in user_state:
            available_usd = float(user_state["marginSummary"]["available"].replace(",","")) # Remove commas and convert to float
            order_cost = formatted_quantity * formatted_price
            if order_cost > available_usd * 0.98: # Use 98% of available balance as a buffer
                self.hyperliquid_api.logger.warning(f"[SKIP] Order cost ({order_cost:.4f}) too large for available balance ({available_usd:.4f}).")
                return {"status": "skipped", "message": "Order too large for available balance"}
        else:
            self.hyperliquid_api.logger.warning("Could not fetch user state for balance check. Proceeding without balance validation.")

        # Prepare order details for Hyperliquid API
        asset_id, coin_name = self.hyperliquid_api.resolve_symbol_to_asset_id(symbol)
        if asset_id is None:
            self.logger.error(f"Could not resolve symbol {symbol} to asset ID")
            return {"status": "failed", "message": f"Invalid symbol: {symbol}"}
        try:
            formatted_quantity, formatted_price = self.hyperliquid_api.validate_and_round_order(asset_id, quantity, price)
        except Exception as e:
            self.logger.error(f"Order validation failed for {symbol}: {e}")
            return {"status": "failed", "message": str(e)}
        order_details = {
            "symbol": symbol,
            "side": side,
            "quantity": formatted_quantity,
            "price": formatted_price,
            "order_type": order_type,
            "reduce_only": reduce_only,
            "cloid": cloid
        }
        try:
            self.logger.info(f"[ORDER REQUEST] Placing {side.upper()} {formatted_quantity} {symbol} at {formatted_price}")
            response = self.hyperliquid_api.place_order(order_details)
            if response and response.get('success'):
                self.logger.info(f"Order placed successfully. Order ID: {response.get('order_id', 'Unknown')}")
                if response.get('filled_immediately'):
                    self.logger.info(f"Order filled immediately. Quantity: {response.get('quantity')} Price: ${response.get('price')}")
                else:
                    self.logger.info(f"Order status: {response.get('status', 'resting')}")
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                self.logger.error(f"Order failed: {error_msg}")
            return response
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"status": "failed", "message": str(e)}

    def cancel_order(self, order_id, symbol):
        """
        Cancels an open order.

        Args:
            order_id (str): The ID of the order to cancel.
            symbol (str): The symbol of the order to cancel.

        Returns:
            dict: A dictionary containing cancellation details.
        """
        self.hyperliquid_api.logger.info(f"Cancelling order {order_id} for {symbol}")
        try:
            response = self.hyperliquid_api.cancel_order(order_id, symbol)
            self.hyperliquid_api.logger.info(f"Order cancellation response: {response}")
            return response
        except Exception as e:
            self.hyperliquid_api.logger.error(f"Error cancelling order: {e}")
            return {"status": "failed", "message": str(e)}



