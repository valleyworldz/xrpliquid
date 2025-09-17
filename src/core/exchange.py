#!/usr/bin/env python3
"""
Exchange Adapter Layer
=====================

This module provides a clean abstraction over the Hyperliquid SDK,
handling quirks like cancel signatures, open_orders formats, and
providing both sync and async interfaces.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils.constants import MAINNET_API_URL
except ImportError:
    raise RuntimeError("Hyperliquid SDK not installed")

from src.core.config import TradingConfig


@dataclass
class OrderResult:
    """Standardized order result"""
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    filled_size: int = 0
    avg_fill_price: Optional[float] = None


@dataclass
class PositionInfo:
    """Standardized position information"""
    size: int
    entry_price: float
    is_long: bool
    unrealized_pnl: float
    margin_used: float


class HyperliquidClient:
    """Synchronous Hyperliquid client with error handling"""
    
    def __init__(self, config: TradingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.info = Info(MAINNET_API_URL)
        self.exchange = None  # Will be initialized with credentials
        self._last_request_time = 0
        self._min_request_interval = 0.25  # 4 req/s max
        
    def initialize(self, private_key: str):
        """Initialize exchange client with credentials"""
        try:
            self.exchange = Exchange(MAINNET_API_URL, private_key)
            self.logger.info("✅ Hyperliquid exchange client initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize exchange client: {e}")
            raise
            
    def _rate_limit(self):
        """Simple rate limiting"""
        wait_time = self._min_request_interval - (time.time() - self._last_request_time)
        if wait_time > 0:
            time.sleep(wait_time)
        self._last_request_time = time.time()
        
    def get_current_price(self, symbol: str = "XRP") -> Optional[float]:
        """Get current market price"""
        try:
            self._rate_limit()
            meta = self.info.meta()
            
            # Find XRP asset
            xrp_asset = None
            for asset in meta['universe']:
                if asset['name'] == symbol:
                    xrp_asset = asset
                    break
                    
            if not xrp_asset:
                self.logger.error(f"❌ Asset {symbol} not found in universe")
                return None
                
            # Get L2 snapshot
            l2_snapshot = self.info.l2_snapshot(xrp_asset['name'])
            
            if not l2_snapshot or 'levels' not in l2_snapshot:
                self.logger.error(f"❌ Invalid L2 snapshot for {symbol}")
                return None
                
            # Calculate mid price
            best_bid = safe_float(l2_snapshot['levels'][0][0]) if l2_snapshot['levels'] else None
            best_ask = safe_float(l2_snapshot['levels'][0][1]) if l2_snapshot['levels'] else None
            
            if best_bid and best_ask:
                mid_price = (best_bid + best_ask) / 2
                return mid_price
            else:
                self.logger.error(f"❌ No valid bid/ask for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting current price: {e}")
            return None
            
    def get_account_status(self) -> Optional[Dict[str, Any]]:
        """Get account status and balance"""
        try:
            self._rate_limit()
            if not self.exchange:
                self.logger.error("❌ Exchange client not initialized")
                return None
                
            account_info = self.exchange.account_info()
            
            if not account_info or 'marginSummary' not in account_info:
                self.logger.error("❌ Invalid account info response")
                return None
                
            margin_summary = account_info['marginSummary']
            
            return {
                'free_collateral': safe_float(margin_summary.get('accountValue', 0)),
                'used_margin': safe_float(margin_summary.get('totalMarginUsed', 0)),
                'available_margin': safe_float(margin_summary.get('totalNtlPos', 0)),
                'account_value': safe_float(margin_summary.get('accountValue', 0))
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting account status: {e}")
            return None
            
    def get_positions(self, symbol: str = "XRP") -> List[PositionInfo]:
        """Get current positions"""
        try:
            self._rate_limit()
            if not self.exchange:
                return []
                
            account_info = self.exchange.account_info()
            
            if not account_info or 'assetPositions' not in account_info:
                return []
                
            positions = []
            for pos in account_info['assetPositions']:
                if pos['coin'] == symbol and safe_float(pos['position']['szi']) != 0:
                    size = int(safe_float(pos['position']['szi']))
                    entry_price = safe_float(pos['position']['entryPx'])
                    is_long = size > 0
                    
                    # Calculate unrealized PnL
                    unrealized_pnl = safe_float(pos.get('unrealizedPnl', 0))
                    margin_used = safe_float(pos.get('marginUsed', 0))
                    
                    positions.append(PositionInfo(
                        size=size,
                        entry_price=entry_price,
                        is_long=is_long,
                        unrealized_pnl=unrealized_pnl,
                        margin_used=margin_used
                    ))
                    
            return positions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting positions: {e}")
            return []
            
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get open orders with standardized format"""
        try:
            self._rate_limit()
            if not self.exchange:
                return []
                
            # Handle different SDK response formats
            response = self.exchange.open_orders()
            
            # Some SDK versions return {"orders": [...]}
            if isinstance(response, dict) and 'orders' in response:
                orders = response['orders']
            elif isinstance(response, list):
                orders = response
            else:
                self.logger.error(f"❌ Unexpected open_orders response format: {type(response)}")
                return []
                
            return orders
            
        except Exception as e:
            self.logger.error(f"❌ Error getting open orders: {e}")
            return []
            
    def place_order(self, symbol: str, is_buy: bool, size: int, price: float, 
                   order_type: str = "limit") -> OrderResult:
        """Place an order with standardized result"""
        try:
            if not self.exchange:
                return OrderResult(success=False, error="Exchange client not initialized")
                
            # Prepare order parameters
            order_params = {
                'coin': symbol,
                'is_buy': is_buy,
                'sz': str(size),
                'limit_px': str(price) if order_type == "limit" else "0",
                'reduce_only': False
            }
            
            # Submit order
            result = self.exchange.order(order_params)
            
            if result and 'status' in result and result['status'] == 'ok':
                order_id = result.get('response', {}).get('data', {}).get('oid')
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    filled_size=0,  # Market orders might be filled immediately
                    avg_fill_price=price if order_type == "market" else None
                )
            else:
                error_msg = result.get('response', {}).get('error', 'Unknown error') if result else 'No response'
                return OrderResult(success=False, error=error_msg)
                
        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            return OrderResult(success=False, error=str(e))
            
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not self.exchange:
                return False
                
            result = self.exchange.cancel(order_id)
            
            if result and 'status' in result and result['status'] == 'ok':
                return True
            else:
                self.logger.warning(f"⚠️ Failed to cancel order {order_id}: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error canceling order {order_id}: {e}")
            return False
            
    def get_funding_rate(self, symbol: str = "XRP") -> Optional[float]:
        """Get current funding rate"""
        try:
            self._rate_limit()
            meta = self.info.meta()
            
            # Find XRP asset
            xrp_asset = None
            for asset in meta['universe']:
                if asset['name'] == symbol:
                    xrp_asset = asset
                    break
                    
            if not xrp_asset:
                return None
                
            # Get funding rate
            funding_info = self.info.funding_history(xrp_asset['name'])
            
            if funding_info and len(funding_info) > 0:
                return safe_float(funding_info[0].get('funding', 0))
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting funding rate: {e}")
            return None


class AsyncHyperliquidClient:
    """Async wrapper around HyperliquidClient"""
    
    def __init__(self, sync_client: HyperliquidClient):
        self.sync_client = sync_client
        self.logger = sync_client.logger
        
    async def get_current_price(self, symbol: str = "XRP") -> Optional[float]:
        """Async wrapper for get_current_price"""
        return await asyncio.to_thread(self.sync_client.get_current_price, symbol)
        
    async def get_account_status(self) -> Optional[Dict[str, Any]]:
        """Async wrapper for get_account_status"""
        return await asyncio.to_thread(self.sync_client.get_account_status)
        
    async def get_positions(self, symbol: str = "XRP") -> List[PositionInfo]:
        """Async wrapper for get_positions"""
        return await asyncio.to_thread(self.sync_client.get_positions, symbol)
        
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Async wrapper for get_open_orders"""
        return await asyncio.to_thread(self.sync_client.get_open_orders)
        
    async def place_order(self, symbol: str, is_buy: bool, size: int, price: float, 
                         order_type: str = "limit") -> OrderResult:
        """Async wrapper for place_order"""
        return await asyncio.to_thread(
            self.sync_client.place_order, symbol, is_buy, size, price, order_type
        )
        
    async def cancel_order(self, order_id: str) -> bool:
        """Async wrapper for cancel_order"""
        return await asyncio.to_thread(self.sync_client.cancel_order, order_id)
        
    async def get_funding_rate(self, symbol: str = "XRP") -> Optional[float]:
        """Async wrapper for get_funding_rate"""
        return await asyncio.to_thread(self.sync_client.get_funding_rate, symbol) 