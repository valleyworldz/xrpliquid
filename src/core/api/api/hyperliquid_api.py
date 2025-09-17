from src.core.utils.decimal_boundary_guard import safe_float
import requests
import json
import time
from eth_account import Account
from hyperliquid_sdk.utils.signing import sign_l1_action, sign_user_signed_action
from hyperliquid_sdk.utils.signing import float_to_wire
from hyperliquid_sdk.info import Info
from hyperliquid_sdk.exchange import Exchange
from core.utils.credential_manager import CredentialManager
from core.utils.logger import Logger
from core.utils.meta_manager import MetaManager
import traceback
from typing import Dict, List, Optional, Any
from decimal import Decimal, ROUND_DOWN
import logging
from hyperliquid_sdk.utils.signing import OrderType
from core.utils.config_manager import ConfigManager
from core.utils.emergency_handler import EmergencyHandler
import math

class HyperliquidAPI:
    def __init__(self, *, testnet: bool = False, logger=None):
        # Set base URL based on testnet flag
        if testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
        else:
            self.base_url = "https://api.hyperliquid.xyz"

        self.info_client = Info(base_url=self.base_url)
        self.logger = logger or Logger()
        self.credential_manager = CredentialManager()

        # Load private key and wallet address securely
        self.private_key = self.credential_manager.get_credential("HYPERLIQUID_PRIVATE_KEY")
        if not self.private_key:
            self.logger.error("HYPERLIQUID_PRIVATE_KEY not found in environment variables or .env file.")
            raise ValueError("Hyperliquid private key is required for authenticated actions.")

        self.wallet = Account.from_key(self.private_key)
        self.wallet_address = self.wallet.address
        self.exchange_client = Exchange(self.wallet, base_url=self.base_url)

        # Initialize MetaManager for proper order validation and load immediately
        self.meta_manager = MetaManager(self)
        if not self.meta_manager.load():
            self.logger.error("Failed to load meta data - order validation may be incorrect")
        else:
            self.logger.info(f"Meta data loaded successfully for {len(self.meta_manager._mapping)} assets")
        
        self.logger.info(f"HyperliquidAPI initialized for wallet: {self.wallet_address} (testnet: {testnet})")
        
        # Initialize cache for asset mappings
        self._asset_mappings_cache = None
        self._asset_mappings_cache_time = 0

        # Rate limiting attributes (1200 requests per minute = 20 requests per second)
        self.request_timestamps = []
        self.rate_limit_interval = 60 # seconds
        self.max_requests_per_interval = 800  # Reduced from 1200 to be more conservative
        self.max_retries = 3  # Reduced from 5 to prevent long delays
        self.initial_backoff_delay = 0.5 # Reduced from 1 to 0.5 seconds

        # Cache for asset mappings
        self._asset_cache = None
        self._asset_cache_time = 0
        self._cache_duration = 300  # 5 minutes

        # Track halted tokens
        self._halted_tokens = set()
        self._halted_tokens_expiry = {}  # Track when to retry halted tokens

    def _get_asset_mappings(self):
        """Get asset mappings with caching"""
        current_time = time.time()
        if (self._asset_cache is None or 
            current_time - self._asset_cache_time > self._cache_duration):
            
            try:
                # Get meta info for asset mappings
                meta_response = self.info_client.meta()
                if meta_response and "universe" in meta_response:
                    self._asset_cache = {
                        "coin_to_asset": {},
                        "asset_to_coin": {}
                    }
                    
                    for i, asset_info in enumerate(meta_response["universe"]):
                        coin_name = asset_info.get("name", "")
                        self._asset_cache["coin_to_asset"][coin_name] = i
                        self._asset_cache["asset_to_coin"][i] = coin_name
                    
                    self._asset_cache_time = current_time
                    self.logger.info(f"Cached {len(self._asset_cache['coin_to_asset'])} asset mappings")
                else:
                    self.logger.error("Failed to get meta info for asset mappings")
                    return None
            except Exception as e:
                self.logger.error(f"Error getting asset mappings: {e}")
                return None
        
        return self._asset_cache

    def _rate_limit_check(self):
        current_time = time.time()
        # Remove timestamps older than the rate limit interval
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < self.rate_limit_interval]

        if len(self.request_timestamps) >= self.max_requests_per_interval:
            # Calculate time to wait until a slot opens up
            time_to_wait = self.rate_limit_interval - (current_time - self.request_timestamps[0])
            if time_to_wait > 0:
                self.logger.warning(f"Rate limit hit. Waiting for {time_to_wait:.2f} seconds.")
                time.sleep(min(time_to_wait, 5.0))  # Cap wait time at 5 seconds
                # After waiting, re-check (in case multiple waits are needed)
                self._rate_limit_check()
        self.request_timestamps.append(time.time())

    def _make_request(self, method, endpoint, params=None, json_data=None, headers=None):
        url = f"{self.base_url}{endpoint}"
        retries = 0
        while retries < self.max_retries:
            self._rate_limit_check() # Apply rate limiting before each request
            try:
                response = requests.request(method, url, params=params, json=json_data, headers=headers)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 429: # Too Many Requests
                    self.logger.warning(f"Rate limit (429) hit. Retrying in {self.initial_backoff_delay * (2 ** retries):.2f} seconds.")
                    time.sleep(self.initial_backoff_delay * (2 ** retries))
                    retries += 1
                elif 500 <= response.status_code < 600: # Server errors
                    self.logger.warning(f"Server error ({response.status_code}). Retrying in {self.initial_backoff_delay * (2 ** retries):.2f} seconds.")
                    time.sleep(self.initial_backoff_delay * (2 ** retries))
                    retries += 1
                else:
                    self.logger.error(f"HTTP error occurred: {http_err} - {response.text}")
                    raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
                self.logger.warning(f"Connection/Timeout error occurred: {conn_err}. Retrying in {self.initial_backoff_delay * (2 ** retries):.2f} seconds.")
                time.sleep(self.initial_backoff_delay * (2 ** retries))
                retries += 1
            except requests.exceptions.RequestException as req_err:
                self.logger.error(f"An unexpected error occurred: {req_err}")
                raise
        self.logger.error(f"Failed to complete request after {self.max_retries} retries: {method} {endpoint}")
        raise requests.exceptions.RequestException("Max retries exceeded.")

    def get_market_data(self, symbol):
        """Get market data for a symbol using the allMids endpoint"""
        self.logger.info(f"Fetching market data for {symbol} from Hyperliquid API...")
        try:
            # Normalize symbol (remove -USD suffix if present)
            normalized_symbol = symbol.replace('-USD', '').replace('USD', '')
            
            # Get mids for all coins using the info client
            mids_response = self.info_client.all_mids()
            
            # Handle different response formats
            if isinstance(mids_response, dict):
                mids = mids_response
            elif hasattr(mids_response, 'get') and callable(mids_response.get):
                mids = mids_response
            else:
                self.logger.error(f"Unexpected mids response format: {type(mids_response)}")
                return None
            
            # Check if normalized symbol exists in mids
            if normalized_symbol in mids:
                price = safe_float(mids[normalized_symbol])
                if price > 0:
                    self.logger.info(f"Retrieved price for {normalized_symbol}: ${price}")
                    return {
                        'symbol': normalized_symbol,
                        'price': price,
                        'timestamp': time.time()
                    }
                else:
                    self.logger.warning(f"Invalid price for {normalized_symbol}: {price}")
                    return None
            else:
                self.logger.warning(f"Symbol {normalized_symbol} not found in mids response. Available: {list(mids.keys())[:10]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def get_user_state(self):
        """Get user state including positions and balances"""
        try:
            # Get user state using the info client
            user_state = self.info_client.user_state(self.wallet_address)
            if user_state:
                self.logger.info(f"Retrieved user state for {self.wallet_address}")
                return user_state
            else:
                self.logger.warning("No user state returned")
                return None
        except Exception as e:
            self.logger.error(f"Error getting user state: {e}")
            return None

    def _get_asset_info(self, asset_id):
        """Get asset information including tick size and lot size"""
        try:
            meta_response = self.info_client.meta()
            if meta_response and "universe" in meta_response:
                for i, asset_info in enumerate(meta_response["universe"]):
                    if i == asset_id:
                        return asset_info
            return None
        except Exception as e:
            self.logger.error(f"Error getting asset info: {e}")
            return None

    def _round_to_tick_size(self, price, asset_id):
        """Round price to valid tick size for the asset"""
        try:
            asset_info = self._get_asset_info(asset_id)
            if asset_info and "tickSize" in asset_info:
                tick_size = safe_float(asset_info["tickSize"])
                # Round to nearest tick size
                rounded_price = round(price / tick_size) * tick_size
                return rounded_price
            return price
        except Exception as e:
            self.logger.error(f"Error rounding to tick size: {e}")
            return price

    def _round_to_lot_size(self, quantity, asset_id):
        """Round quantity to valid lot size for the asset"""
        try:
            asset_info = self._get_asset_info(asset_id)
            if asset_info and "lotSize" in asset_info:
                lot_size = safe_float(asset_info["lotSize"])
                # Round to nearest lot size
                rounded_quantity = round(quantity / lot_size) * lot_size
                # Ensure minimum lot size
                if rounded_quantity < lot_size:
                    rounded_quantity = lot_size
                return rounded_quantity
            return quantity
        except Exception as e:
            self.logger.error(f"Error rounding to lot size: {e}")
            return quantity

    def _round_to_valid_precision(self, value, is_price=False):
        """Round value to valid precision for Hyperliquid API"""
        try:
            # For prices, we need to ensure they fit within the wire format requirements
            # For quantities, we need to round to the appropriate lot size
            if is_price:
                # Round to 4 decimal places for prices to avoid wire format issues
                return round(value, 4)
            else:
                # Round to 6 decimal places for quantities
                return round(value, 6)
        except Exception as e:
            self.logger.error(f"Error rounding value {value}: {e}")
            return value

    def resolve_symbol_to_asset_id(self, symbol):
        """Resolve symbol to asset ID and coin name"""
        try:
            self.logger.info(f"[DEBUG] Resolving symbol: {symbol} (type: {type(symbol)})")
            
            asset_mappings = self._get_asset_mappings()
            if not asset_mappings:
                self.logger.error(f"[DEBUG] No asset mappings available for {symbol}")
                return (None, None)
            
            self.logger.info(f"[DEBUG] Asset mappings: {len(asset_mappings['coin_to_asset'])} coins available")
            
            if isinstance(symbol, int):
                # If symbol is already an asset ID, find the coin name
                asset_id = symbol
                coin_name = None
                for coin, aid in asset_mappings["coin_to_asset"].items():
                    if aid == asset_id:
                        coin_name = coin
                        break
                result = (asset_id, coin_name)
                self.logger.info(f"[DEBUG] Integer symbol {symbol} -> {result}")
                return result
            
            # Handle symbol normalization (remove -USD suffix)
            normalized_symbol = symbol
            if isinstance(symbol, str):
                # Remove common suffixes
                normalized_symbol = symbol.replace('-USD', '').replace('USD', '').replace('-USDC', '').replace('USDC', '')
                self.logger.info(f"[DEBUG] Normalized symbol: {symbol} -> {normalized_symbol}")
            
            # Try the normalized symbol first
            if normalized_symbol in asset_mappings["coin_to_asset"]:
                asset_id = asset_mappings["coin_to_asset"][normalized_symbol]
                result = (asset_id, normalized_symbol)
                self.logger.info(f"[DEBUG] Normalized symbol {normalized_symbol} -> {result}")
                return result
            
            # Try the original symbol as fallback
            elif symbol in asset_mappings["coin_to_asset"]:
                asset_id = asset_mappings["coin_to_asset"][symbol]
                result = (asset_id, symbol)
                self.logger.info(f"[DEBUG] Original symbol {symbol} -> {result}")
                return result
            else:
                # Log available symbols for debugging
                available_symbols = list(asset_mappings["coin_to_asset"].keys())
                self.logger.error(f"[DEBUG] Symbol {symbol} (normalized: {normalized_symbol}) not found in mappings")
                self.logger.error(f"[DEBUG] Available symbols: {available_symbols[:20]}...")  # Show first 20
                return (None, None)
                
        except Exception as e:
            self.logger.error(f"[DEBUG] Error resolving symbol {symbol}: {e}")
            return (None, None)

    def get_asset_metadata(self, asset_id):
        """Get asset metadata including tick size and lot size from meta response"""
        try:
            if not hasattr(self, '_asset_metadata_cache'):
                self._asset_metadata_cache = {}
            
            if asset_id in self._asset_metadata_cache:
                return self._asset_metadata_cache[asset_id]
            
            # Get asset metadata from info client meta response
            meta_response = self.info_client.meta()
            if meta_response and 'universe' in meta_response:
                for i, asset_info in enumerate(meta_response['universe']):
                    if i == asset_id:
                        # Extract szDecimals for lot size calculation
                        sz_decimals = asset_info.get('szDecimals', 0)
                        px_decimals = asset_info.get('pxDecimals', 0)
                        
                        # Calculate lot size (minimum order size)
                        lot_size = 1.0 / (10 ** sz_decimals)
                        
                        # Calculate tick size (minimum price increment)
                        tick_size = 1.0 / (10 ** px_decimals)
                        
                        metadata = {
                            'tick_size': tick_size,
                            'lot_size': lot_size,
                            'sz_decimals': sz_decimals,
                            'px_decimals': px_decimals,
                            'name': asset_info.get('name', ''),
                            'base_currency': asset_info.get('baseCurrency', ''),
                            'quote_currency': asset_info.get('quoteCurrency', '')
                        }
                        
                        self._asset_metadata_cache[asset_id] = metadata
                        return metadata
            
            # Fallback metadata if not found
            return {
                'tick_size': 0.01,
                'lot_size': 0.01,
                'sz_decimals': 2,
                'px_decimals': 2
            }
            
        except Exception as e:
            self.logger.error(f"Error getting asset metadata for {asset_id}: {e}")
            # Return conservative fallback
            return {
                'tick_size': 0.01,
                'lot_size': 0.01,
                'sz_decimals': 2,
                'px_decimals': 2
            }

    def validate_and_round_order(self, asset_id: int, quantity: float, price: float):
        asset = self.get_asset_metadata(asset_id)
        tick = asset["tick_size"]
        lot  = asset["lot_size"]
        
        # ADAPTIVE MINIMUM VALUE: Adjust based on available margin
        try:
            user_state = self.get_user_state()
            if user_state and "marginSummary" in user_state:
                margin_summary = user_state["marginSummary"]
                account_value = safe_float(margin_summary.get("accountValue", "0"))
                total_margin_used = safe_float(margin_summary.get("totalMarginUsed", "0"))
                available_margin = account_value - total_margin_used
                
                # Adjust minimum order value based on available margin
                if available_margin < 10.0:
                    min_val = max(5.0, available_margin * 0.7)  # Use 70% of available margin, minimum $5
                    self.logger.info(f"🔧 Adaptive minimum: ${min_val:.2f} (available: ${available_margin:.2f})")
                else:
                    min_val = 10.0  # Standard $10 minimum
            else:
                min_val = 6.0  # Conservative fallback when can't check margin
                
        except Exception as e:
            min_val = 6.0  # Emergency fallback
            self.logger.warning(f"Could not check margin, using ${min_val:.2f} minimum: {e}")
            
        px_decimals = asset["px_decimals"]
        sz_decimals = asset["sz_decimals"]

        # CRITICAL FIX: Ensure lot size is never zero or negative
        if lot <= 0:
            lot = 0.01  # Default minimum lot size
            self.logger.warning(f"[VALIDATION] Invalid lot size for asset {asset_id}, using default: {lot}")

        # CRITICAL FIX: Ensure tick size is never zero or negative
        if tick <= 0:
            tick = 0.01  # Default minimum tick size
            self.logger.warning(f"[VALIDATION] Invalid tick size for asset {asset_id}, using default: {tick}")

        # ENHANCED PRICE ROUNDING: Use proper tick size compliance
        # Calculate the exact number of ticks, then round to nearest integer
        tick_count = price / tick
        rounded_tick_count = round(tick_count)  # Round to nearest tick
        
        # Ensure at least 1 tick
        if rounded_tick_count < 1:
            rounded_tick_count = 1
            
        # Calculate the rounded price
        rounded_price = rounded_tick_count * tick
        
        # Apply decimal precision with respect to px_decimals
        rounded_price = round(rounded_price, px_decimals)
        
        # CRITICAL: Final tick size validation to ensure compliance
        final_tick_count = rounded_price / tick
        if abs(final_tick_count - round(final_tick_count)) > 1e-10:
            # Force exact tick compliance
            rounded_price = round(final_tick_count) * tick
            rounded_price = round(rounded_price, px_decimals)
        
        # ENHANCED QUANTITY ROUNDING: Use proper lot size compliance
        if quantity <= 0:
            # If quantity is zero or negative, use minimum lot size
            rounded_qty = lot
            self.logger.warning(f"[VALIDATION] Zero/negative quantity for asset {asset_id}, using minimum lot: {lot}")
        else:
            # Calculate the exact number of lots, then round up to ensure minimum
            lot_count = quantity / lot
            rounded_lot_count = math.ceil(lot_count)  # Always round up to ensure minimum
            
            # Calculate the rounded quantity
            rounded_qty = rounded_lot_count * lot
            
            # Apply decimal precision with respect to sz_decimals
            rounded_qty = round(rounded_qty, sz_decimals)
            
            # CRITICAL: Final lot size validation to ensure compliance
            final_lot_count = rounded_qty / lot
            if abs(final_lot_count - round(final_lot_count)) > 1e-10:
                # Force exact lot compliance
                rounded_qty = round(final_lot_count) * lot
                rounded_qty = round(rounded_qty, sz_decimals)

        # Enhanced validation
        if rounded_price <= 0:
            self.logger.error(f"[VALIDATION FAILED] Asset {asset_id}: price {price} -> {rounded_price} (tick={tick})")
            raise Exception(f"Price rounded to {rounded_price} <= 0 (tick={tick}) for asset={asset_id}")
        
        if rounded_qty <= 0:
            self.logger.error(f"[VALIDATION FAILED] Asset {asset_id}: qty {quantity} -> {rounded_qty} (lot={lot})")
            raise Exception(f"Quantity rounded to {rounded_qty} <= 0 (lot={lot}) for asset={asset_id}")
        
        order_value = rounded_price * rounded_qty
        
        # MARGIN-AWARE ORDER SIZING
        try:
            user_state = self.get_user_state()
            if user_state and "marginSummary" in user_state:
                margin_summary = user_state["marginSummary"]
                account_value = safe_float(margin_summary.get("accountValue", "0"))
                total_margin_used = safe_float(margin_summary.get("totalMarginUsed", "0"))
                available_margin = account_value - total_margin_used
                
                # Ensure order doesn't exceed 80% of available margin
                max_order_value = available_margin * 0.8
                
                if order_value > max_order_value and max_order_value > min_val:
                    # Reduce order size to fit available margin
                    adjusted_qty = max_order_value / rounded_price
                    adjusted_lot_count = math.floor(adjusted_qty / lot)
                    rounded_qty = adjusted_lot_count * lot
                    rounded_qty = round(rounded_qty, sz_decimals)
                    order_value = rounded_price * rounded_qty
                    
                    self.logger.info(f"🔧 Margin-adjusted order: ${order_value:.2f} (was ${rounded_price * (adjusted_qty):.2f})")
                    
        except Exception as e:
            self.logger.warning(f"Could not check margin for order sizing: {e}")
        
        if order_value < min_val:
            self.logger.warning(f"[VALIDATION] Order value ${order_value:.2f} < ${min_val:.2f} for asset {asset_id}")
            # Calculate minimum quantity needed for minimum order value
            min_qty_for_value = min_val / rounded_price
            
            # Round up to nearest lot size to ensure we meet minimum value
            min_lot_count = math.ceil(min_qty_for_value / lot)
            rounded_qty = min_lot_count * lot
            rounded_qty = round(rounded_qty, sz_decimals)
            
            self.logger.info(f"[VALIDATION] Adjusted quantity to meet minimum value: {rounded_qty}")
        
        # Final validation log
        final_order_value = rounded_price * rounded_qty
        self.logger.info(f"[VALIDATION SUCCESS] Asset {asset_id}: ${final_order_value:.2f} (qty={rounded_qty}, price=${rounded_price:.{px_decimals}f})")
        
        return rounded_qty, rounded_price

    def place_order(self, symbol, side, quantity, price, order_type='limit', time_in_force='Gtc', reduce_only=False):
        """Place an order with smart fallback and error handling"""
        try:
            # Check if token is halted
            if self._is_token_halted(symbol):
                self.logger.warning(f"⚠️ Skipping {symbol} - trading is currently halted")
                return {'success': False, 'error': 'Trading halted', 'skip_token': True}
            
            # Resolve symbol to asset ID and coin name
            resolution_result = self.resolve_symbol_to_asset_id(symbol)
            if not isinstance(resolution_result, tuple) or len(resolution_result) != 2:
                return {'success': False, 'error': f'Invalid symbol: {symbol}'}
            
            asset_id, coin_name = resolution_result
            if asset_id is None or coin_name is None:
                return {'success': False, 'error': f'Could not resolve symbol: {symbol}'}
            
            # Validate and round order parameters
            try:
                validated_q, validated_p = self.validate_and_round_order(asset_id, quantity, price)
            except Exception as e:
                return {'success': False, 'error': f'Order validation failed: {e}'}
            
            # Execute order with smart fallback
            return self._place_order_with_smart_fallback(symbol, side, validated_q, validated_p, order_type, time_in_force, reduce_only)
            
        except Exception as e:
            self.logger.error(f"[ORDER PLACEMENT ERROR] {e}")
            return {'success': False, 'error': str(e)}

    def _place_order_with_smart_fallback(self, symbol, side, quantity, price, order_type='limit', time_in_force='Gtc', reduce_only=False):
        """Execute order with intelligent fallback for minimum order requirements"""
        try:
            # Fix for symbol resolution - ensure we always get a tuple
            resolution_result = self.resolve_symbol_to_asset_id(symbol)
            if not isinstance(resolution_result, tuple) or len(resolution_result) != 2:
                self.logger.error(f"[ORDER ERROR] Invalid resolution result for {symbol}: {resolution_result}")
                return {'success': False, 'error': f"Invalid symbol resolution for: {symbol}"}
            
            asset_id, coin_name = resolution_result
            if asset_id is None:
                return {'success': False, 'error': f"Could not resolve symbol: {symbol}"}
            
            self.logger.info(f"[ORDER PROCESSING] Resolved symbol: {symbol} -> Asset ID: {asset_id}, Coin: {coin_name}")
            
            # Initialize order_value for all strategies
            order_value = 0.0
            
            # Strategy 1: Try original order first
            try:
                q, p = self.validate_and_round_order(asset_id, quantity, price)
                order_value = q * p
                
                # Check if order meets minimum value requirement
                if order_value >= 10.0:
                    self.logger.info(f"🎯 Executing original order: {symbol} {side} ${order_value:.2f}")
                    return self._execute_order(coin_name, side, q, p, order_type, time_in_force, reduce_only, symbol)
                
            except Exception as e:
                self.logger.error(f"[ORDER VALIDATION ERROR] {e}")
                # Continue to fallback strategies
            
            # Strategy 2: Bump up order size to meet minimum
            try:
                min_order_value = 10.0
                bumped_quantity = min_order_value / price
                bumped_q, bumped_p = self.validate_and_round_order(asset_id, bumped_quantity, price)
                bumped_value = bumped_q * bumped_p
                
                if bumped_value >= min_order_value:
                    self.logger.info(f"📈 Bumping order size for {symbol}: ${order_value:.2f} → ${bumped_value:.2f}")
                    result = self._execute_order(coin_name, side, bumped_q, bumped_p, order_type, time_in_force, reduce_only, symbol)
                    if result.get('success'):
                        self.logger.info(f"✅ Bumped order executed successfully: {symbol}")
                        return result
                
            except Exception as e:
                self.logger.warning(f"❌ Bump strategy failed for {symbol}: {e}")
            
            # Strategy 3: Switch to cheaper token with better liquidity
            self.logger.info(f"🔄 Switching to cheaper token for {symbol} (order value: ${order_value:.2f})")
            return self._switch_to_cheaper_token(symbol, side, order_type, time_in_force, reduce_only)
            
        except Exception as e:
            self.logger.error(f"[ORDER EXCEPTION] Error placing order: {e}")
            return {'success': False, 'error': str(e), 'message': f"Exception: {e}"}

    def _execute_order(self, coin_name, side, q, p, order_type, time_in_force, reduce_only, symbol):
        """Execute the actual order with proper price validation and market order fallback"""
        try:
            # PRICE VALIDATION: Ensure order price is within 95% of reference price
            current_price = self._get_current_market_price(coin_name)
            if current_price and current_price > 0:
                price_deviation = abs(p - current_price) / current_price
                if price_deviation > 0.95:  # More than 95% away from reference price
                    self.logger.warning(f"⚠️ Price deviation too high: {price_deviation:.2%} for {symbol}")
                    # Use market order instead of limit order
                    order_type = 'market'
                    p = current_price  # Use current market price
                    self.logger.info(f"🔄 Switching to market order for {symbol} at ${p:.4f}")
            
            # MARKET ORDER OPTIMIZATION: Use market orders for better execution
            if order_type.lower() == 'market':
                # Use market order with IOC (Immediate or Cancel)
                market_order_type: OrderType = {"limit": {"tif": "Ioc"}}
                # Use current market price for market orders
                if current_price and current_price > 0:
                    p = current_price
            else:
                tif_value = time_in_force if time_in_force in ['Alo', 'Ioc', 'Gtc'] else 'Gtc'
            
            # CRITICAL FIX: Round price to proper tick size to avoid "Price must be divisible by tick size" error
            try:
                # Get asset ID for tick size calculation
                resolution_result = self.resolve_symbol_to_asset_id(symbol)
                if isinstance(resolution_result, tuple) and len(resolution_result) == 2:
                    asset_id, _ = resolution_result
                    if asset_id is not None:
                        # Round price to tick size using existing validation method
                        _, rounded_price = self.validate_and_round_order(asset_id, q, p)
                        p = rounded_price
                        self.logger.info(f"🔧 Rounded price for {symbol}: ${p:.6f} (tick size compliant)")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not round price for {symbol}: {e}")
                # Continue with original price
            
            # All orders are limit orders with appropriate time in force
            if order_type.lower() == 'market':
                order_type_dict = market_order_type
            else:
                order_type_dict: OrderType = {"limit": {"tif": tif_value}}
            
            self.logger.info(f"[ORDER PREPARED] Using coin: {coin_name}, is_buy: {side.lower() == 'buy'}, sz: {q}, price: {p}, type: {order_type}")
            
            response = self.exchange_client.order(
                coin_name,
                side.lower() == 'buy',
                q,
                p,
                order_type_dict,
                reduce_only
            )
            
            self.logger.info(f"[ORDER RESPONSE] Exchange response: {response}")
            
            if response and response.get('status') == 'ok':
                order_data = response.get('response', {}).get('data', {})
                statuses = order_data.get('statuses', [])
                if statuses:
                    status = statuses[0]
                    if 'filled' in status:
                        fill_data = status['filled']
                        order_id = self._extract_order_id(fill_data) or fill_data.get('oid')
                        filled_quantity = self._extract_quantity(fill_data) or safe_float(fill_data.get('totalSz', 0))
                        avg_price = self._extract_price(fill_data) or safe_float(fill_data.get('avgPx', 0))
                        self.logger.info(f"[ORDER FILLED] Order ID: {order_id} - Filled: {filled_quantity} {symbol} @ ${avg_price}")
                        return {'success': True, 'order_id': order_id, 'status': 'filled', 'quantity': filled_quantity, 'price': avg_price, 'filled_immediately': True}
                    elif 'resting' in status:
                        order_id = self._extract_order_id(status['resting']) or status['resting'].get('oid')
                        self.logger.info(f"[ORDER RESTING] Order ID: {order_id} - {symbol} {side} {q} @ ${p} (resting)")
                        return {'success': True, 'order_id': order_id, 'status': 'resting', 'filled_immediately': False}
                    elif 'error' in status:
                        error_msg = status['error']
                        self.logger.error(f"[ORDER FAILED] Exchange returned error: {error_msg}")
                        
                        # Handle specific price validation errors
                        if '95% away from the reference price' in error_msg:
                            self.logger.warning(f"🔄 Price validation failed for {symbol}, retrying with market order")
                            return self._retry_with_market_order(coin_name, side, q, reduce_only, symbol)
                        
                        # Handle tick size errors
                        if 'Price must be divisible by tick size' in error_msg:
                            self.logger.warning(f"🔄 Tick size error for {symbol}, retrying with properly rounded price")
                            return self._retry_with_rounded_price(coin_name, side, q, reduce_only, symbol)
                        
                        # Handle liquidity issues
                        if 'Order could not immediately match against any resting orders' in error_msg:
                            self.logger.warning(f"🔄 Liquidity issue for {symbol}, retrying with smaller size and market order")
                            return self._retry_with_liquidity_fix(coin_name, side, q, reduce_only, symbol)
                        
                        # Handle trading halted errors
                        if 'Trading is halted' in error_msg:
                            self.logger.warning(f"⚠️ Trading halted for {symbol}, skipping this token")
                            self._mark_token_halted(symbol, duration_minutes=30)
                            return {'success': False, 'error': 'Trading halted', 'skip_token': True}
                        
                        # Handle minimum order value errors
                        if 'Order must have minimum value of $10' in error_msg:
                            self.logger.warning(f"🔄 Minimum order value error for {symbol}, retrying with larger size")
                            return self._retry_with_minimum_value(coin_name, side, q, reduce_only, symbol)
                        
                        # Handle invalid order size errors
                        if 'Order has invalid size' in error_msg:
                            self.logger.warning(f"🔄 Invalid order size error for {symbol}, retrying with adjusted size")
                            return self._retry_with_adjusted_size(coin_name, side, q, reduce_only, symbol)
                        
                        return {'success': False, 'error': error_msg}
                    else:
                        self.logger.warning(f"[ORDER UNKNOWN] Unknown order status: {status}")
                        return {'success': False, 'error': f"Unknown order status: {status}"}
                else:
                    self.logger.error("[ORDER FAILED] No statuses in response")
                    return {'success': False, 'error': "No order statuses in response"}
            else:
                self.logger.error(f"[ORDER FAILED] Exchange returned error status: {response}")
                return {'success': False, 'error': f"Exchange error: {response}"}
                
        except Exception as e:
            self.logger.error(f"[ORDER EXECUTION ERROR] {e}")
            return {'success': False, 'error': str(e)}

    def _retry_with_market_order(self, coin_name, side, q, reduce_only, symbol):
        """Retry order execution with market order when price validation fails"""
        try:
            current_price = self._get_current_market_price(coin_name)
            if not current_price or current_price <= 0:
                return {'success': False, 'error': 'Unable to get current market price'}
            
            self.logger.info(f"🔄 Retrying with market order: {symbol} {side} {q} @ market price")
            
            # Use market order with IOC (Immediate or Cancel)
            order_type_dict: OrderType = {"limit": {"tif": "Ioc"}}
            
            response = self.exchange_client.order(
                coin_name,
                side.lower() == 'buy',
                q,
                current_price,  # Use current market price
                order_type_dict,
                reduce_only
            )
            
            if response and response.get('status') == 'ok':
                order_data = response.get('response', {}).get('data', {})
                statuses = order_data.get('statuses', [])
                if statuses:
                    status = statuses[0]
                    if 'filled' in status:
                        fill_data = status['filled']
                        order_id = self._extract_order_id(fill_data) or fill_data.get('oid')
                        filled_quantity = self._extract_quantity(fill_data) or safe_float(fill_data.get('totalSz', 0))
                        avg_price = self._extract_price(fill_data) or safe_float(fill_data.get('avgPx', 0))
                        self.logger.info(f"[MARKET ORDER FILLED] Order ID: {order_id} - Filled: {filled_quantity} {symbol} @ ${avg_price}")
                        return {'success': True, 'order_id': order_id, 'status': 'filled', 'quantity': filled_quantity, 'price': avg_price, 'filled_immediately': True}
                    elif 'error' in status:
                        error_msg = status['error']
                        self.logger.error(f"[MARKET ORDER FAILED] {error_msg}")
                        return {'success': False, 'error': error_msg}
            
            return {'success': False, 'error': 'Market order retry failed'}
            
        except Exception as e:
            self.logger.error(f"[MARKET ORDER RETRY ERROR] {e}")
            return {'success': False, 'error': str(e)}

    def _retry_with_rounded_price(self, coin_name, side, q, reduce_only, symbol):
        """Retry order execution with properly rounded price"""
        try:
            # Get current market price
            current_price = self._get_current_market_price(coin_name)
            if not current_price or current_price <= 0:
                return {'success': False, 'error': 'Unable to get current market price'}
            
            # Get asset ID and round price properly
            resolution_result = self.resolve_symbol_to_asset_id(symbol)
            if not isinstance(resolution_result, tuple) or len(resolution_result) != 2:
                return {'success': False, 'error': 'Invalid symbol resolution'}
            
            asset_id, _ = resolution_result
            if asset_id is None:
                return {'success': False, 'error': 'Could not resolve asset ID'}
            
            # Round price to tick size
            _, rounded_price = self.validate_and_round_order(asset_id, q, current_price)
            
            self.logger.info(f"🔄 Retrying with rounded price: {symbol} {side} {q} @ ${rounded_price:.6f}")
            
            # Use IOC for immediate execution
            order_type_dict: OrderType = {"limit": {"tif": "Ioc"}}
            
            response = self.exchange_client.order(
                coin_name,
                side.lower() == 'buy',
                q,
                rounded_price,
                order_type_dict,
                reduce_only
            )
            
            if response and response.get('status') == 'ok':
                order_data = response.get('response', {}).get('data', {})
                statuses = order_data.get('statuses', [])
                if statuses:
                    status = statuses[0]
                    if 'filled' in status:
                        fill_data = status['filled']
                        order_id = self._extract_order_id(fill_data) or fill_data.get('oid')
                        filled_quantity = self._extract_quantity(fill_data) or safe_float(fill_data.get('totalSz', 0))
                        avg_price = self._extract_price(fill_data) or safe_float(fill_data.get('avgPx', 0))
                        self.logger.info(f"[ROUNDED PRICE FILLED] Order ID: {order_id} - Filled: {filled_quantity} {symbol} @ ${avg_price}")
                        return {'success': True, 'order_id': order_id, 'status': 'filled', 'quantity': filled_quantity, 'price': avg_price, 'filled_immediately': True}
                    elif 'error' in status:
                        error_msg = status['error']
                        self.logger.error(f"[ROUNDED PRICE FAILED] {error_msg}")
                        return {'success': False, 'error': error_msg}
            
            return {'success': False, 'error': 'Rounded price retry failed'}
            
        except Exception as e:
            self.logger.error(f"[ROUNDED PRICE RETRY ERROR] {e}")
            return {'success': False, 'error': str(e)}

    def _retry_with_liquidity_fix(self, coin_name, side, q, reduce_only, symbol):
        """Retry order execution with liquidity fixes (smaller size + market order)"""
        try:
            current_price = self._get_current_market_price(coin_name)
            if not current_price or current_price <= 0:
                return {'success': False, 'error': 'Unable to get current market price'}
            
            # Reduce order size to improve liquidity
            reduced_q = q * 0.5  # Reduce by 50%
            
            # Round the reduced quantity to valid lot size
            resolution_result = self.resolve_symbol_to_asset_id(symbol)
            if isinstance(resolution_result, tuple) and len(resolution_result) == 2:
                asset_id, _ = resolution_result
                if asset_id is not None:
                    reduced_q = self._round_to_lot_size(reduced_q, asset_id)
                    # Also round the price to tick size
                    rounded_price = self._round_to_tick_size(current_price, asset_id)
                else:
                    rounded_price = current_price
            else:
                rounded_price = current_price
            
            self.logger.info(f"🔄 Retrying with liquidity fix: {symbol} {side} {reduced_q} (reduced from {q}) @ ${rounded_price:.6f}")
            
            # Use market order with IOC for better liquidity
            order_type_dict: OrderType = {"limit": {"tif": "Ioc"}}
            
            response = self.exchange_client.order(
                coin_name,
                side.lower() == 'buy',
                reduced_q,
                rounded_price,  # Use rounded price
                order_type_dict,
                reduce_only
            )
            
            if response and response.get('status') == 'ok':
                order_data = response.get('response', {}).get('data', {})
                statuses = order_data.get('statuses', [])
                if statuses:
                    status = statuses[0]
                    if 'filled' in status:
                        fill_data = status['filled']
                        order_id = self._extract_order_id(fill_data) or fill_data.get('oid')
                        filled_quantity = self._extract_quantity(fill_data) or safe_float(fill_data.get('totalSz', 0))
                        avg_price = self._extract_price(fill_data) or safe_float(fill_data.get('avgPx', 0))
                        self.logger.info(f"[LIQUIDITY FIX FILLED] Order ID: {order_id} - Filled: {filled_quantity} {symbol} @ ${avg_price}")
                        return {'success': True, 'order_id': order_id, 'status': 'filled', 'quantity': filled_quantity, 'price': avg_price, 'filled_immediately': True}
                    elif 'error' in status:
                        error_msg = status['error']
                        self.logger.error(f"[LIQUIDITY FIX FAILED] {error_msg}")
                        
                        # Handle trading halted errors
                        if 'Trading is halted' in error_msg:
                            self.logger.warning(f"⚠️ Trading halted for {symbol}, skipping this token")
                            self._mark_token_halted(symbol, duration_minutes=30)
                            return {'success': False, 'error': 'Trading halted', 'skip_token': True}
                        
                        # Handle minimum order value errors
                        if 'Order must have minimum value of $10' in error_msg:
                            self.logger.warning(f"🔄 Minimum order value error for {symbol}, retrying with larger size")
                            return self._retry_with_minimum_value(coin_name, side, q, reduce_only, symbol)
                        
                        return {'success': False, 'error': error_msg}
            
            return {'success': False, 'error': 'Liquidity fix retry failed'}
            
        except Exception as e:
            self.logger.error(f"[LIQUIDITY FIX RETRY ERROR] {e}")
            return {'success': False, 'error': str(e)}

    def _get_current_market_price(self, coin_name):
        """Get current market price for a coin"""
        try:
            # Try to get price from market data
            market_data = self.get_market_data(coin_name)
            if market_data and 'price' in market_data:
                return safe_float(market_data['price'])
            
            # Fallback: try to get from user state if we have positions
            user_state = self.get_user_state()
            if user_state and 'assetPositions' in user_state:
                for position in user_state['assetPositions']:
                    if position.get('position', {}).get('coin') == coin_name:
                        mark_px = position['position'].get('markPx')
                        if mark_px:
                            return safe_float(mark_px)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not get current price for {coin_name}: {e}")
            return None

    def _switch_to_cheaper_token(self, original_symbol, side, order_type, time_in_force, reduce_only):
        """Switch to a cheaper token that can meet minimum order requirements"""
        try:
            # Get all available trading pairs
            asset_mappings = self._get_asset_mappings()
            if not asset_mappings:
                return {'success': False, 'error': 'No asset mappings available'}
            
            all_symbols = list(asset_mappings['coin_to_asset'].keys())
            
            # Get current prices for all symbols
            symbol_prices = {}
            for symbol in all_symbols:
                if symbol != original_symbol:  # Skip the original symbol
                    try:
                        market_data = self.get_market_data(symbol)
                        if market_data and 'price' in market_data:
                            symbol_prices[symbol] = safe_float(market_data['price'])
                    except:
                        continue
            
            if not symbol_prices:
                self.logger.warning(f"❌ No alternative symbols available for {original_symbol}")
                return {'success': False, 'error': 'No alternative symbols available'}
            
            # Sort symbols by price (cheapest first) and calculate minimum order sizes
            affordable_symbols = []
            for symbol, price in symbol_prices.items():
                if price > 0:
                    min_size = 10.0 / price  # Minimum size to meet $10 requirement
                    min_value = min_size * price
                    
                    affordable_symbols.append({
                        'symbol': symbol,
                        'price': price,
                        'min_size': min_size,
                        'min_value': min_value,
                        'liquidity_score': self._get_liquidity_score(symbol)
                    })
            
            if not affordable_symbols:
                self.logger.warning(f"❌ No affordable symbols found for minimum order requirements")
                return {'success': False, 'error': 'No affordable symbols found'}
            
            # Sort by liquidity score (higher is better) and price (lower is better)
            affordable_symbols.sort(key=lambda x: (-x['liquidity_score'], x['price']))
            
            # Try the best alternative symbol
            best_symbol = affordable_symbols[0]
            self.logger.info(f"🔄 Switching from {original_symbol} to {best_symbol['symbol']} "
                           f"(price: ${best_symbol['price']:.4f}, min_size: {best_symbol['min_size']:.4f})")
            
            # Execute order with the new symbol
            result = self._execute_order(
                best_symbol['symbol'], 
                side, 
                best_symbol['min_size'], 
                best_symbol['price'], 
                order_type, 
                time_in_force, 
                reduce_only, 
                best_symbol['symbol']
            )
            
            if result.get('success'):
                self.logger.info(f"✅ Successfully switched to {best_symbol['symbol']} and executed {side}")
                return result
            else:
                # Try next best symbol
                if len(affordable_symbols) > 1:
                    second_best = affordable_symbols[1]
                    self.logger.info(f"🔄 Trying second best: {second_best['symbol']}")
                    
                    result = self._execute_order(
                        second_best['symbol'], 
                        side, 
                        second_best['min_size'], 
                        second_best['price'], 
                        order_type, 
                        time_in_force, 
                        reduce_only, 
                        second_best['symbol']
                    )
                    
                    if result.get('success'):
                        self.logger.info(f"✅ Successfully switched to {second_best['symbol']} and executed {side}")
                        return result
            
            self.logger.warning(f"❌ Could not find suitable alternative for {original_symbol}")
            return {'success': False, 'error': 'Could not find suitable alternative'}
            
        except Exception as e:
            self.logger.error(f"❌ Token switching failed: {e}")
            return {'success': False, 'error': f'Token switching failed: {e}'}

    def _get_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for a symbol (higher is better)"""
        try:
            # Simple liquidity scoring based on symbol characteristics
            liquidity_scores = {
                'BTC': 1.0,    # Highest liquidity
                'ETH': 0.95,   # Very high liquidity
                'SOL': 0.85,   # High liquidity
                'BNB': 0.80,   # Good liquidity
                'INJ': 0.75,   # Moderate liquidity
                'SUI': 0.70,   # Moderate liquidity
                'DOGE': 0.65,  # Lower liquidity
            }
            
            return liquidity_scores.get(symbol, 0.5)  # Default score for unknown symbols
            
        except Exception as e:
            self.logger.error(f"❌ Liquidity scoring failed: {e}")
            return 0.5  # Default score

    def cancel_order(self, order_id, symbol):
        """Cancel an order by order ID with improved error handling"""
        self.logger.info(f"Cancelling order: {order_id} for symbol: {symbol}")
        try:
            # First check if order still exists
            if not self.verify_order_on_exchange(order_id, symbol):
                self.logger.info(f"[CANCEL SKIPPED] Order {order_id} not found - likely filled or already cancelled")
                return {"status": "ok", "message": "Order not found - likely filled or already cancelled"}

            # Get asset mappings
            asset_mappings = self._get_asset_mappings()
            if not asset_mappings:
                return {"status": "failed", "message": "Asset mappings not available"}

            # Resolve symbol to asset ID
            asset_id = None
            if isinstance(symbol, int):
                asset_id = symbol
            elif symbol in asset_mappings["coin_to_asset"]:
                asset_id = asset_mappings["coin_to_asset"][symbol]
            else:
                return {"status": "failed", "message": f"Unknown symbol: {symbol}"}

            # Ensure asset_id is not None
            if asset_id is None:
                return {"status": "failed", "message": f"Could not resolve asset ID for symbol: {symbol}"}

            # Cancel order using exchange client
            response = self.exchange_client.cancel(str(asset_id), int(order_id))
            
            if response and response.get("status") == "ok":
                self.logger.info(f"[CANCEL SUCCESS] Order {order_id} cancelled successfully")
                return {"status": "ok", "response": response}
            else:
                # Handle specific error codes
                error_msg = response.get("error", str(response)) if response else "No response"
                
                # Common Hyperliquid cancel error codes
                if str(error_msg) in ['1', '2', '3', '4', '5', '6', '7']:
                    error_meanings = {
                        '1': "Order not found",
                        '2': "Order already filled", 
                        '3': "Order already cancelled",
                        '4': "Invalid order ID",
                        '5': "Order expired",
                        '6': "Insufficient permissions",
                        '7': "Order being processed"
                    }
                    meaning = error_meanings.get(str(error_msg), "Unknown error")
                    self.logger.info(f"[CANCEL INFO] Order {order_id}: {meaning} (code: {error_msg})")
                    
                    # These are actually success cases - order is no longer active
                    if str(error_msg) in ['1', '2', '3', '5']:
                        return {"status": "ok", "message": f"Order inactive: {meaning}"}
                
                self.logger.warning(f"[CANCEL FAILED] Order {order_id}: {error_msg}")
                return {"status": "failed", "message": error_msg}
                
        except Exception as e:
            self.logger.error(f"[CANCEL EXCEPTION] Error cancelling order {order_id}: {e}")
            return {"status": "failed", "message": str(e)}

    def get_open_orders(self, symbol=None):
        """Get open orders for a symbol or all symbols"""
        try:
            # Get open orders using info client
            open_orders = self.info_client.open_orders(self.wallet_address)
            
            if symbol and open_orders:
                # Filter by symbol if specified
                asset_mappings = self._get_asset_mappings()
                if asset_mappings and symbol in asset_mappings["coin_to_asset"]:
                    asset_id = asset_mappings["coin_to_asset"][symbol]
                    filtered_orders = [order for order in open_orders if order.get("asset") == asset_id]
                    return filtered_orders
            
            return open_orders
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []

    def get_positions(self):
        """Get current positions"""
        try:
            user_state = self.get_user_state()
            if user_state and "assetPositions" in user_state:
                positions = []
                for position in user_state["assetPositions"]:
                    size = safe_float(position.get("szi", "0"))
                    if size != 0:  # Only include non-zero positions
                        positions.append({
                            "coin": position.get("coin", "Unknown"),
                            "size": size,
                            "entry_price": safe_float(position.get("entryPx", "0")),
                            "unrealized_pnl": safe_float(position.get("unrealizedPnl", "0"))
                        })
                return positions
            return []
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    def get_open_positions(self):
        """Get current open positions - alias for get_positions for backward compatibility"""
        try:
            user_state = self.get_user_state()
            if user_state and "assetPositions" in user_state:
                positions = {}
                for asset_position in user_state["assetPositions"]:
                    if "position" in asset_position:
                        position = asset_position["position"]
                        size = safe_float(position.get("szi", "0"))
                        if size != 0:  # Only include non-zero positions
                            coin = position.get("coin", "Unknown")
                            positions[coin] = {
                                "coin": coin,
                                "szi": position.get("szi", "0"),
                                "size": size,
                                "entry_price": safe_float(position.get("entryPx", "0")),
                                "entryPx": position.get("entryPx", "0"),
                                "unrealized_pnl": safe_float(position.get("unrealizedPnl", "0")),
                                "unrealizedPnl": position.get("unrealizedPnl", "0"),
                                "leverage": position.get("leverage", {}),
                                "marginUsed": position.get("marginUsed", "0"),
                                "positionValue": position.get("positionValue", "0"),
                                "liquidationPx": position.get("liquidationPx"),
                                "returnOnEquity": position.get("returnOnEquity", "0")
                            }
                return positions
            return {}
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return {}

    def get_balance(self):
        """Get account balance"""
        try:
            user_state = self.get_user_state()
            if user_state and "marginSummary" in user_state:
                margin_summary = user_state["marginSummary"]
                account_value = safe_float(margin_summary.get("accountValue", "0"))
                total_margin_used = safe_float(margin_summary.get("totalMarginUsed", "0"))
                available_margin = account_value - total_margin_used
                
                return {
                    "account_value": account_value,
                    "total_margin_used": total_margin_used,
                    "available": available_margin,  # Add available margin field
                    "total_ntl_pos": safe_float(margin_summary.get("totalNtlPos", "0")),
                    "total_raw_usd": safe_float(margin_summary.get("totalRawUsd", "0"))
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return None

    def _extract(self, resp: Dict, *keys) -> Any:
        """Extract value from response with fallback keys to prevent KeyErrors"""
        for k in keys:
            if k in resp:
                return resp[k]
        self.logger.error(f"Unexpected response shape - missing keys {keys}: {resp}")
        raise RuntimeError(f"API response missing expected keys: {keys}")

    def _extract_order_id(self, resp: Dict) -> Optional[str]:
        """Extract order ID from response with multiple fallback keys"""
        # Hyperliquid uses 'oid' as the standard field - check this first
        if "oid" in resp:
            return str(resp["oid"])
        
        # Fallback to other possible fields
        for key in ["orderId", "order_id", "id"]:
            if key in resp and resp[key] is not None:
                return str(resp[key])
        
        # Don't log warning for valid Hyperliquid responses that just use 'oid'
        return None

    def _extract_price(self, resp: Dict) -> Optional[float]:
        """Extract price from response with multiple fallback keys"""
        try:
            price = self._extract(resp, "price", "avgPx", "avg_price", "filled_price")
            return safe_float(price) if price is not None else None
        except (RuntimeError, ValueError, TypeError):
            self.logger.warning(f"Could not extract price from response: {resp}")
            return None

    def _extract_quantity(self, resp: Dict) -> Optional[float]:
        """Extract quantity from response with multiple fallback keys"""
        try:
            qty = self._extract(resp, "quantity", "totalSz", "total_size", "filled_size")
            return safe_float(qty) if qty is not None else None
        except (RuntimeError, ValueError, TypeError):
            self.logger.warning(f"Could not extract quantity from response: {resp}")
            return None

    def verify_order_on_exchange(self, order_id, symbol):
        """Verify that an order exists on the exchange"""
        try:
            if not order_id:
                return False
                
            # Get open orders for this symbol
            open_orders = self.get_open_orders(symbol)
            
            # Check if our order is in the open orders
            for order in open_orders:
                if order.get('oid') == order_id:
                    self.logger.info(f"[ORDER VERIFIED] Order {order_id} confirmed on exchange for {symbol}")
                    return True
            
            # If not in open orders, check if it was filled
            user_state = self.get_user_state()
            if user_state and 'assetPositions' in user_state:
                for position_data in user_state['assetPositions']:
                    if 'position' in position_data:
                        position = position_data['position']
                        if position.get('coin') == symbol:
                            self.logger.info(f"[ORDER VERIFIED] Order {order_id} likely filled for {symbol}")
                            return True
            
            self.logger.warning(f"[ORDER NOT FOUND] Order {order_id} not found on exchange for {symbol}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error verifying order {order_id}: {e}")
            return False

    def _is_token_halted(self, symbol: str) -> bool:
        """Check if a token is currently halted from trading"""
        current_time = time.time()
        
        # Check if token is in halted list and hasn't expired
        if symbol in self._halted_tokens:
            expiry_time = self._halted_tokens_expiry.get(symbol, 0)
            if current_time < expiry_time:
                return True
            else:
                # Remove expired halt
                self._halted_tokens.remove(symbol)
                if symbol in self._halted_tokens_expiry:
                    del self._halted_tokens_expiry[symbol]
        
        return False
    
    def _mark_token_halted(self, symbol: str, duration_minutes: int = 30):
        """Mark a token as halted for a specified duration"""
        current_time = time.time()
        expiry_time = current_time + (duration_minutes * 60)
        
        self._halted_tokens.add(symbol)
        self._halted_tokens_expiry[symbol] = expiry_time
        
        self.logger.warning(f"⚠️ Marked {symbol} as halted for {duration_minutes} minutes")

    def _retry_with_minimum_value(self, coin_name, side, q, reduce_only, symbol):
        """Retry order execution with larger size to meet minimum order value"""
        try:
            current_price = self._get_current_market_price(coin_name)
            if not current_price or current_price <= 0:
                return {'success': False, 'error': 'Unable to get current market price'}
            
            # Increase order size to meet minimum order value
            increased_q = q * 2  # Double the quantity
            
            # Round the increased quantity to valid lot size
            resolution_result = self.resolve_symbol_to_asset_id(symbol)
            if isinstance(resolution_result, tuple) and len(resolution_result) == 2:
                asset_id, _ = resolution_result
                if asset_id is not None:
                    increased_q = self._round_to_lot_size(increased_q, asset_id)
                    # Also round the price to tick size
                    rounded_price = self._round_to_tick_size(current_price, asset_id)
                else:
                    rounded_price = current_price
            else:
                rounded_price = current_price
            
            self.logger.info(f"🔄 Retrying with larger size: {symbol} {side} {increased_q} (increased from {q}) @ ${rounded_price:.6f}")
            
            # Use market order with IOC for immediate execution
            order_type_dict: OrderType = {"limit": {"tif": "Ioc"}}
            
            response = self.exchange_client.order(
                coin_name,
                side.lower() == 'buy',
                increased_q,
                rounded_price,
                order_type_dict,
                reduce_only
            )
            
            if response and response.get('status') == 'ok':
                order_data = response.get('response', {}).get('data', {})
                statuses = order_data.get('statuses', [])
                if statuses:
                    status = statuses[0]
                    if 'filled' in status:
                        fill_data = status['filled']
                        order_id = self._extract_order_id(fill_data) or fill_data.get('oid')
                        filled_quantity = self._extract_quantity(fill_data) or safe_float(fill_data.get('totalSz', 0))
                        avg_price = self._extract_price(fill_data) or safe_float(fill_data.get('avgPx', 0))
                        self.logger.info(f"[LARGER SIZE FILLED] Order ID: {order_id} - Filled: {filled_quantity} {symbol} @ ${avg_price}")
                        return {'success': True, 'order_id': order_id, 'status': 'filled', 'quantity': filled_quantity, 'price': avg_price, 'filled_immediately': True}
                    elif 'error' in status:
                        error_msg = status['error']
                        self.logger.error(f"[LARGER SIZE FAILED] {error_msg}")
                        return {'success': False, 'error': error_msg}
            
            return {'success': False, 'error': 'Larger size retry failed'}
            
        except Exception as e:
            self.logger.error(f"[LARGER SIZE RETRY ERROR] {e}")
            return {'success': False, 'error': str(e)}

    def _retry_with_adjusted_size(self, coin_name, side, q, reduce_only, symbol):
        """Retry order execution with properly adjusted size to fix invalid size errors"""
        try:
            current_price = self._get_current_market_price(coin_name)
            if not current_price or current_price <= 0:
                return {'success': False, 'error': 'Unable to get current market price'}
            
            # Get asset metadata for proper size adjustment
            resolution_result = self.resolve_symbol_to_asset_id(symbol)
            if not isinstance(resolution_result, tuple) or len(resolution_result) != 2:
                return {'success': False, 'error': 'Invalid symbol resolution'}
            
            asset_id, _ = resolution_result
            if asset_id is None:
                return {'success': False, 'error': 'Could not resolve asset ID'}
            
            # Get asset metadata
            asset_metadata = self.get_asset_metadata(asset_id)
            lot_size = asset_metadata.get('lot_size', 0.01)
            
            # CRITICAL FIX: Ensure quantity is at least the minimum lot size
            if q <= 0 or q < lot_size:
                adjusted_q = lot_size
                self.logger.warning(f"[SIZE ADJUSTMENT] Quantity {q} below minimum lot {lot_size}, using minimum")
            else:
                # Round to nearest lot size using ceil to ensure we don't go below minimum
                adjusted_q = math.ceil(q / lot_size) * lot_size
                adjusted_q = round(adjusted_q, 6)  # Apply precision
            
            # Also round the price to tick size
            rounded_price = self._round_to_tick_size(current_price, asset_id)
            
            self.logger.info(f"🔄 Retrying with adjusted size: {symbol} {side} {adjusted_q} (adjusted from {q}) @ ${rounded_price:.6f}")
            
            # Use market order with IOC for immediate execution
            order_type_dict: OrderType = {"limit": {"tif": "Ioc"}}
            
            response = self.exchange_client.order(
                coin_name,
                side.lower() == 'buy',
                adjusted_q,
                rounded_price,
                order_type_dict,
                reduce_only
            )
            
            if response and response.get('status') == 'ok':
                order_data = response.get('response', {}).get('data', {})
                statuses = order_data.get('statuses', [])
                if statuses:
                    status = statuses[0]
                    if 'filled' in status:
                        fill_data = status['filled']
                        order_id = self._extract_order_id(fill_data) or fill_data.get('oid')
                        filled_quantity = self._extract_quantity(fill_data) or safe_float(fill_data.get('totalSz', 0))
                        avg_price = self._extract_price(fill_data) or safe_float(fill_data.get('avgPx', 0))
                        self.logger.info(f"[ADJUSTED SIZE FILLED] Order ID: {order_id} - Filled: {filled_quantity} {symbol} @ ${avg_price}")
                        return {'success': True, 'order_id': order_id, 'status': 'filled', 'quantity': filled_quantity, 'price': avg_price, 'filled_immediately': True}
                    elif 'error' in status:
                        error_msg = status['error']
                        self.logger.error(f"[ADJUSTED SIZE FAILED] {error_msg}")
                        
                        # Handle trading halted errors
                        if 'Trading is halted' in error_msg:
                            self.logger.warning(f"⚠️ Trading halted for {symbol}, skipping this token")
                            self._mark_token_halted(symbol, duration_minutes=30)
                            return {'success': False, 'error': 'Trading halted', 'skip_token': True}
                        
                        return {'success': False, 'error': error_msg}
            
            return {'success': False, 'error': 'Adjusted size retry failed'}
            
        except Exception as e:
            self.logger.error(f"[ADJUSTED SIZE RETRY ERROR] {e}")
            return {'success': False, 'error': str(e)}

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol - wrapper for get_market_data"""
        try:
            market_data = self.get_market_data(symbol)
            if market_data and 'price' in market_data:
                return safe_float(market_data['price'])
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None


