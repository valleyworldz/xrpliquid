
#!/usr/bin/env python3
"""
Fixed Hyperliquid API - Works with Encrypted Credentials
=======================================================
API that properly handles encrypted credentials
"""

from src.core.utils.decimal_boundary_guard import safe_float
import requests
import json
import time
from eth_account import Account
from hyperliquid_sdk.utils.signing import sign_l1_action, sign_user_signed_action
from hyperliquid_sdk.utils.signing import float_to_wire
from hyperliquid_sdk.info import Info
from hyperliquid_sdk.exchange import Exchange
from core.enhanced_credential_manager import EnhancedCredentialManager
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

class FixedHyperliquidAPI:
    def __init__(self, *, testnet: bool = False, logger=None):
        # Set base URL based on testnet flag
        if testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
        else:
            self.base_url = "https://api.hyperliquid.xyz"

        self.info_client = Info(base_url=self.base_url)
        self.logger = logger or Logger()
        self.credential_manager = EnhancedCredentialManager()

        # Load private key and wallet address securely
        self.private_key = self.credential_manager.get_credential("HYPERLIQUID_PRIVATE_KEY")
        if not self.private_key:
            self.logger.error("Failed to load private key from encrypted credentials.")
            raise ValueError("Hyperliquid private key is required for authenticated actions.")

        self.wallet = Account.from_key(self.private_key)
        self.wallet_address = self.wallet.address
        self.exchange_client = Exchange(self.wallet, base_url=self.base_url)

        # Initialize MetaManager for proper order validation
        self.meta_manager = MetaManager(self)
        if not self.meta_manager.load():
            self.logger.error("Failed to load meta data - order validation may be incorrect")
        else:
            self.logger.info(f"Meta data loaded successfully for {len(self.meta_manager._mapping)} assets")
        
        self.logger.info(f"FixedHyperliquidAPI initialized for wallet: {self.wallet_address} (testnet: {testnet})")
        
        # Initialize cache for asset mappings
        self._asset_mappings_cache = None
        self._asset_mappings_cache_time = 0

        # Rate limiting attributes
        self.request_timestamps = []
        self.rate_limit_interval = 60
        self.max_requests_per_interval = 800
        self.max_retries = 3
        self.initial_backoff_delay = 0.5

        # Cache for asset mappings
        self._asset_cache = None
        self._asset_cache_time = 0
        self._cache_duration = 300

        # Track halted tokens
        self._halted_tokens = set()
        self._halted_tokens_expiry = {}

    def get_user_state(self):
        """Get user state including positions and balances"""
        try:
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

    def get_market_data(self, symbol):
        """Get market data for a symbol"""
        try:
            normalized_symbol = symbol.replace('-USD', '').replace('USD', '')
            mids_response = self.info_client.all_mids()
            
            if isinstance(mids_response, dict):
                mids = mids_response
            elif hasattr(mids_response, 'get') and callable(mids_response.get):
                mids = mids_response
            else:
                self.logger.error(f"Unexpected mids response format: {type(mids_response)}")
                return None
            
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
                self.logger.warning(f"Symbol {normalized_symbol} not found in mids response")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def place_order(self, order_data):
        """Place order with proper validation"""
        try:
            symbol = order_data.get('symbol')
            side = order_data.get('side')
            size = order_data.get('size')
            order_type = order_data.get('type', 'market')
            reduce_only = order_data.get('reduce_only', False)
            
            if not all([symbol, side, size]):
                self.logger.error("Missing required order parameters")
                return False
            
            # Get current price for market orders
            if order_type == 'market':
                market_data = self.get_market_data(symbol)
                if not market_data:
                    self.logger.error(f"Could not get market data for {symbol}")
                    return False
                price = market_data['price']
            else:
                price = order_data.get('price', 0)
            
            # Place order using exchange client
            try:
                result = self.exchange_client.order(
                    coin=symbol,
                    is_buy=side == 'buy',
                    sz=size,
                    limit_px=price,
                    reduce_only=reduce_only
                )
                
                if result:
                    self.logger.info(f"Order placed successfully: {symbol} {side} {size}")
                    return True
                else:
                    self.logger.error(f"Order placement failed: {symbol}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Order placement error: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Order placement error: {e}")
            return False

    def get_ticker(self, symbol):
        """Get ticker data for symbol"""
        try:
            market_data = self.get_market_data(symbol)
            if market_data:
                return {
                    'symbol': market_data['symbol'],
                    'last': market_data['price'],
                    'timestamp': market_data['timestamp']
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
