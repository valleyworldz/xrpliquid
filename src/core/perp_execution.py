#!/usr/bin/env python3
"""
🚀 PERPETUAL TRADING EXECUTION HANDLER
======================================

Dedicated handler for perpetual trading operations.
Handles leverage, margin, funding rates, and liquidation management.

Features:
- Leverage management
- Margin calculations
- Funding rate handling
- Liquidation protection
- Advanced TP/SL with OCO
"""

import time
from typing import Dict, Any, Optional
from core.api.hyperliquid_api import HyperliquidAPI
from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager

class PerpExecutor:
    """
    Perpetual Trading Execution Handler
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = Logger()
        self.api = HyperliquidAPI()
        self.active_positions = {}
        self.position_history = []
        
    def get_margin_info(self, token: str) -> Dict[str, Any]:
        """
        Get margin information for a token.
        
        Args:
            token: Token symbol
            
        Returns:
            Dict containing margin info
        """
        try:
            user_state = self.api.get_user_state()
            if user_state and "marginSummary" in user_state:
                margin_info = user_state["marginSummary"]
                return {
                    "account_value": float(margin_info.get("accountValue", "0")),
                    "total_margin_used": float(margin_info.get("totalMarginUsed", "0")),
                    "total_n_unrealized_pnl": float(margin_info.get("totalNUnrealizedPnl", "0")),
                    "total_margin_used_pct": float(margin_info.get("totalMarginUsedPct", "0"))
                }
            return {}
        except Exception as e:
            self.logger.error(f"[PERP] Error getting margin info: {e}")
            return {}
    
    def get_position_info(self, token: str) -> Dict[str, Any]:
        """
        Get current position information for a token.
        
        Args:
            token: Token symbol
            
        Returns:
            Dict containing position info
        """
        try:
            user_state = self.api.get_user_state()
            if user_state and "assetPositions" in user_state:
                for position in user_state["assetPositions"]:
                    if position.get("coin") == token:
                        return {
                            "size": float(position.get("szi", "0")),
                            "entry_price": float(position.get("entryPx", "0")),
                            "unrealized_pnl": float(position.get("unrealizedPnl", "0")),
                            "leverage": float(position.get("leverage", "1")),
                            "margin_used": float(position.get("marginUsed", "0"))
                        }
            return {"size": 0, "entry_price": 0, "unrealized_pnl": 0, "leverage": 1, "margin_used": 0}
        except Exception as e:
            self.logger.error(f"[PERP] Error getting position info for {token}: {e}")
            return {"size": 0, "entry_price": 0, "unrealized_pnl": 0, "leverage": 1, "margin_used": 0}
    
    def calculate_position_size(self, token: str, usd_amount: float, leverage: float = 1.0) -> float:
        """
        Calculate position size based on USD amount and leverage.
        
        Args:
            token: Token symbol
            usd_amount: USD amount to invest
            leverage: Leverage multiplier
            
        Returns:
            float: Position size in token units
        """
        try:
            # Get current price
            market_data = self.api.get_market_data(token)
            if not market_data or "price" not in market_data:
                self.logger.error(f"[PERP] Could not get price for {token}")
                return 0.0
            
            current_price = market_data["price"]
            
            # Calculate position size
            position_size = (usd_amount * leverage) / current_price
            
            # Get token metadata for size formatting
            token_meta = self.api.get_token_metadata(token)
            if token_meta and "szDecimals" in token_meta:
                sz_decimals = token_meta["szDecimals"]
                min_size = 1 / (10 ** sz_decimals)
                
                # Ensure minimum size
                if position_size < min_size:
                    self.logger.warning(f"[PERP] Position size {position_size} below minimum {min_size}")
                    return 0.0
                
                # Round to valid size
                position_size = round(position_size / min_size) * min_size
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"[PERP] Error calculating position size: {e}")
            return 0.0
    
    def place_perp_order(self, token: str, side: str, size: float, price: Optional[float] = None,
                        order_type: str = "market", leverage: float = 1.0,
                        take_profit_pct: float = 0.02, stop_loss_pct: float = 0.01) -> Dict[str, Any]:
        """
        Place a perpetual trading order with leverage.
        
        Args:
            token: Token symbol
            side: "buy" or "sell"
            size: Position size
            price: Order price (None for market orders)
            order_type: "market" or "limit"
            leverage: Leverage multiplier
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Dict containing order response
        """
        try:
            self.logger.info(f"[PERP] Placing {order_type} order: {side} {size} {token} @ {leverage}x")
            
            # Check margin requirements
            margin_info = self.get_margin_info(token)
            if margin_info:
                margin_used_pct = margin_info.get("total_margin_used_pct", 0)
                if margin_used_pct > 0.8:  # 80% margin usage warning
                    self.logger.warning(f"[PERP] High margin usage: {margin_used_pct:.2%}")
            
            # Validate and round order
            asset_id, coin_name = self.api.resolve_symbol_to_asset_id(token)
            if asset_id is None:
                self.logger.error(f"Could not resolve symbol {token} to asset ID")
                return {"status": "failed", "error": f"Invalid symbol: {token}"}
            try:
                size, price = self.api.validate_and_round_order(asset_id, size, price)
            except Exception as e:
                self.logger.error(f"Order validation failed for {token}: {e}")
                return {"status": "failed", "error": str(e)}
            
            # Prepare order
            order = {
                "symbol": token,
                "side": side,
                "quantity": size,
                "order_type": order_type,
                "reduce_only": False,
                "leverage": leverage
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
                
                # Track position
                entry_price = price if price is not None else response.get("price", 0)
                position_info = {
                    "token": token,
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "entry_time": time.time(),
                    "order_id": response.get("order_id"),
                    "leverage": leverage,
                    "take_profit_pct": take_profit_pct,
                    "stop_loss_pct": stop_loss_pct,
                    "status": "open"
                }
                
                position_key = f"{token}_{side}_{response.get('order_id')}"
                self.active_positions[position_key] = position_info
                
                # Place TP/SL orders with OCO
                if take_profit_pct > 0 or stop_loss_pct > 0:
                    self._place_perp_tp_sl(token, side, size, entry_price, take_profit_pct, stop_loss_pct, response.get("order_id"))
                
                return {"status": "success", "response": response, "position_info": position_info}
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                self.logger.error(f"Order failed: {error_msg}")
                return {"status": "failed", "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"[PERP] Error placing order: {e}")
            return {"status": "error", "error": str(e)}
    
    def _place_perp_tp_sl(self, token: str, side: str, size: float, entry_price: float,
                          take_profit_pct: float, stop_loss_pct: float, parent_order_id: str) -> bool:
        """
        Place take-profit and stop-loss orders for perpetual position with OCO.
        
        Args:
            token: Token symbol
            side: Original position side
            size: Position size
            entry_price: Entry price
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            parent_order_id: Parent order ID
            
        Returns:
            bool: True if orders placed successfully
        """
        try:
            # Calculate TP/SL prices
            if side == "buy":
                tp_price = entry_price * (1 + take_profit_pct)
                sl_price = entry_price * (1 - stop_loss_pct)
            else:  # sell (short)
                tp_price = entry_price * (1 - take_profit_pct)
                sl_price = entry_price * (1 + stop_loss_pct)
            
            # Determine exit side
            exit_side = "sell" if side == "buy" else "buy"
            
            # Place take-profit order
            if take_profit_pct > 0:
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
                tp_response = self.api.place_order(tp_order)
                if tp_response:
                    self.logger.info(f"[PERP] TP order placed: {tp_price:.4f}")
            
            # Place stop-loss order
            if stop_loss_pct > 0:
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
                sl_response = self.api.place_order(sl_order)
                if sl_response:
                    self.logger.info(f"[PERP] SL order placed: {sl_price:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[PERP] Error placing TP/SL orders: {e}")
            return False
    
    def close_perp_position(self, token: str, size: Optional[float] = None) -> bool:
        """
        Close a perpetual position.
        
        Args:
            token: Token symbol
            size: Size to close (None for all)
            
        Returns:
            bool: True if position closed successfully
        """
        try:
            # Get current position
            position_info = self.get_position_info(token)
            current_size = position_info["size"]
            
            if current_size == 0:
                self.logger.warning(f"[PERP] No position to close for {token}")
                return False
            
            # Use specified size or all
            close_size = size if size is not None else current_size
            
            # Determine side (opposite of current position)
            side = "sell" if current_size > 0 else "buy"
            
            # Place closing order
            response = self.api.place_order(
                symbol=token,
                side=side,
                quantity=abs(close_size),
                order_type="market",
                reduce_only=True
            )
            
            # Log the full response for debugging
            self.logger.info(f"[PERP] Close order response for {token}: {response}")
            
            if response and response.get("success") == True:
                # Try to extract order_id with fallback
                try:
                    order_id = response.get("order_id") or response.get("orderId") or response.get("id")
                    if order_id:
                        self.logger.info(f"[PERP] Successfully closed {token} position with order ID: {order_id}")
                    else:
                        self.logger.warning(f"[PERP] Close successful but no order ID found in response: {response}")
                except KeyError as e:
                    self.logger.error(f"[PERP] KeyError accessing order ID in response: {e}")
                    self.logger.error(f"[PERP] Full response: {response}")
                    # Still return True if the order was successful
                
                return True
            else:
                self.logger.error(f"[PERP] Failed to close position: {response}")
                
                # Try market order fallback
                self.logger.warning(f"[PERP] Retrying {token} close as MARKET order")
                return self._close_perp_position_market_fallback(token, abs(close_size), side)
                
        except Exception as e:
            self.logger.error(f"[PERP] Error closing position: {e}")
            # Try market order fallback on any exception
            return self._close_perp_position_market_fallback(token, abs(close_size), side)
    
    def _close_perp_position_market_fallback(self, token: str, size: float, side: str) -> bool:
        """Fallback method to close perpetual position with market order"""
        try:
            self.logger.info(f"[PERP] Attempting market order fallback for {token}")
            
            # Place market order with reduce_only
            response = self.api.place_order(
                symbol=token,
                side=side,
                quantity=size,
                order_type="market",
                reduce_only=True
            )
            
            # Log the full response for debugging
            self.logger.info(f"[PERP] Market fallback response for {token}: {response}")
            
            if response and response.get("success") == True:
                self.logger.info(f"[PERP] Market fallback successful for {token}")
                return True
            else:
                self.logger.error(f"[PERP] Market fallback failed for {token}: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"[PERP] Error in market fallback for {token}: {e}")
            return False
    
    def get_funding_rate(self, token: str) -> float:
        """
        Get current funding rate for a token.
        
        Args:
            token: Token symbol
            
        Returns:
            float: Current funding rate
        """
        try:
            market_data = self.api.get_market_data(token)
            if market_data and "fundingRate" in market_data:
                return float(market_data["fundingRate"])
            return 0.0
        except Exception as e:
            self.logger.error(f"[PERP] Error getting funding rate for {token}: {e}")
            return 0.0
    
    def check_liquidation_risk(self, token: str) -> Dict[str, Any]:
        """
        Check liquidation risk for current positions.
        
        Args:
            token: Token symbol
            
        Returns:
            Dict containing liquidation risk info
        """
        try:
            margin_info = self.get_margin_info(token)
            position_info = self.get_position_info(token)
            
            risk_level = "low"
            margin_used_pct = margin_info.get("total_margin_used_pct", 0)
            
            if margin_used_pct > 0.9:
                risk_level = "critical"
            elif margin_used_pct > 0.8:
                risk_level = "high"
            elif margin_used_pct > 0.6:
                risk_level = "medium"
            
            return {
                "risk_level": risk_level,
                "margin_used_pct": margin_used_pct,
                "account_value": margin_info.get("account_value", 0),
                "unrealized_pnl": position_info.get("unrealized_pnl", 0),
                "position_size": position_info.get("size", 0)
            }
            
        except Exception as e:
            self.logger.error(f"[PERP] Error checking liquidation risk: {e}")
            return {"risk_level": "unknown", "margin_used_pct": 0}
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get perpetual trading position summary.
        
        Returns:
            Dict containing position summary
        """
        try:
            margin_info = self.get_margin_info("")  # Get overall margin info
            positions = {}
            total_unrealized_pnl = 0.0
            
            user_state = self.api.get_user_state()
            if user_state and "assetPositions" in user_state:
                for position in user_state["assetPositions"]:
                    token = position.get("coin")
                    size = float(position.get("szi", "0"))
                    
                    if size != 0:
                        unrealized_pnl = float(position.get("unrealizedPnl", "0"))
                        total_unrealized_pnl += unrealized_pnl
                        
                        positions[token] = {
                            "size": size,
                            "entry_price": float(position.get("entryPx", "0")),
                            "unrealized_pnl": unrealized_pnl,
                            "leverage": float(position.get("leverage", "1")),
                            "margin_used": float(position.get("marginUsed", "0"))
                        }
            
            return {
                "account_value": margin_info.get("account_value", 0),
                "total_margin_used": margin_info.get("total_margin_used", 0),
                "margin_used_pct": margin_info.get("total_margin_used_pct", 0),
                "total_unrealized_pnl": total_unrealized_pnl,
                "positions": positions,
                "active_positions": len(self.active_positions),
                "position_history": len(self.position_history)
            }
            
        except Exception as e:
            self.logger.error(f"[PERP] Error getting position summary: {e}")
            return {}
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all active perpetual orders.
        
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
            
            self.logger.info(f"[PERP] Cancelled {cancelled_count} orders")
            return True
            
        except Exception as e:
            self.logger.error(f"[PERP] Error cancelling orders: {e}")
            return False 