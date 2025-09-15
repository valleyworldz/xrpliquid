#!/usr/bin/env python3
"""
MetaManager: Centralized tick and lot size management for Hyperliquid
"""

import logging
from typing import Dict, Optional, Any
from decimal import Decimal, ROUND_DOWN

class MetaManager:
    """
    Fetches and caches per-asset lot-size & tick-size from Hyperliquid's
    Meta/Info API. Provides rounding helpers for order size/price.
    """

    def __init__(self, api):
        self.api = api
        self.logger = logging.getLogger(self.__class__.__name__)
        self._mapping: Dict[str, Dict[str, float]] = {}

    def load(self) -> bool:
        """
        Fetch the on-chain asset metadata exactly once and build our
        internal symbol → {'lot', 'tick'} map.
        """
        try:
            # 1) Try the Hyperliquid SDK Info client first
            try:
                raw = self.api.info_client.meta()  # SDK method for metadata
            except Exception as e:
                self.logger.warning("Info client .meta() failed, trying fallback: %s", e)
                raw = self._fallback_raw_meta()

            # 2) Parse the raw response
            assets = raw.get("universe") or raw.get("assets") or raw.get("data") or []
            if not isinstance(assets, list):
                self.logger.error("Unexpected meta response shape, no 'universe' list: %s", raw)
                return False

            for i, asset in enumerate(assets):
                # Extract symbol from asset info
                symbol = self._extract_symbol(asset, i)
                if not symbol:
                    continue
                
                # Extract lot and tick sizes
                lot = self._extract_lot_size(asset)
                tick = self._extract_tick_size(asset)
                
                if lot > 0 and tick > 0:
                    self._mapping[symbol] = {
                        "lot": lot, 
                        "tick": tick,
                        "asset_id": i,
                        "sz_decimals": asset.get('szDecimals', 2),
                        "px_decimals": asset.get('pxDecimals', 2)
                    }
                    self.logger.debug(f"Mapped {symbol}: lot={lot}, tick={tick}")
                else:
                    self.logger.debug("Skipping asset with incomplete meta: %s", asset)

            if not self._mapping:
                self.logger.error("MetaManager: no valid assets loaded")
                return False
                
            self.logger.info("MetaManager loaded %d assets", len(self._mapping))
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading meta data: {e}")
            return False

    def _extract_symbol(self, asset: Dict[str, Any], asset_id: int) -> Optional[str]:
        """Extract symbol from asset info with fallbacks"""
        # Try direct symbol field first
        symbol = asset.get("name") or asset.get("symbol") or asset.get("ticker")
        if symbol:
            return symbol
        
        # Try to get from asset mappings
        try:
            if hasattr(self.api, '_get_asset_mappings'):
                asset_mappings = self.api._get_asset_mappings()
                if asset_mappings and "asset_to_coin" in asset_mappings:
                    return asset_mappings["asset_to_coin"].get(asset_id)
        except Exception:
            pass
        
        # Fallback: use common symbol mapping
        common_symbols = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "MATIC", "LINK", "UNI", "AAVE", "COMP"]
        if asset_id < len(common_symbols):
            return common_symbols[asset_id]
        
        return None

    def _extract_lot_size(self, asset_info: Dict[str, Any]) -> float:
        """Extract lot size from asset info"""
        try:
            # szDecimals indicates the number of decimal places for size
            sz_decimals = asset_info.get('szDecimals', 2)
            lot_size = 1.0 / (10 ** sz_decimals)
            return lot_size
        except Exception as e:
            self.logger.error(f"Error extracting lot size: {e}")
            return 0.01

    def _extract_tick_size(self, asset_info: Dict[str, Any]) -> float:
        """Extract tick size from asset info"""
        try:
            # pxDecimals indicates the number of decimal places for price
            px_decimals = asset_info.get('pxDecimals', 2)
            tick_size = 1.0 / (10 ** px_decimals)
            return tick_size
        except Exception as e:
            self.logger.error(f"Error extracting tick size: {e}")
            return 0.01

    def _fallback_raw_meta(self) -> Dict[str, Any]:
        """
        As a last resort, hit the raw /meta REST endpoint on the base_url.
        """
        try:
            import requests
            url = f"{self.api.base_url}/meta"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.error("Fallback meta fetch failed: %s", e)
            return {}

    def get_lot(self, symbol: str) -> float:
        """Get lot size for symbol"""
        if symbol in self._mapping:
            return self._mapping[symbol]["lot"]
        else:
            self.logger.warning(f"No lot size found for {symbol}, using default 0.01")
            return 0.01

    def get_tick(self, symbol: str) -> float:
        """Get tick size for symbol"""
        if symbol in self._mapping:
            return self._mapping[symbol]["tick"]
        else:
            self.logger.warning(f"No tick size found for {symbol}, using default 0.01")
            return 0.01

    def get_asset_id(self, symbol: str) -> int | None:
        """Get the asset ID for a symbol, or None if not found"""
        asset_id = self._mapping.get(symbol, {}).get('asset_id')
        if asset_id is not None:
            try:
                return int(asset_id)
            except Exception:
                return None
        return None

    def get_decimals(self, symbol: str) -> tuple[float, float]:
        """Get size and price decimals for symbol"""
        if symbol in self._mapping:
            return (
                self._mapping[symbol]["sz_decimals"],
                self._mapping[symbol]["px_decimals"]
            )
        return (2.0, 2.0)  # Default fallback

    # DEPRECATED: Use HyperliquidAPI.validate_and_round_order for all live order validation and rounding.
    def round_size(self, symbol: str, raw_size: float) -> float:
        """
        DEPRECATED: Use HyperliquidAPI.validate_and_round_order for all live order validation and rounding.
        """
        raise NotImplementedError("round_size is deprecated. Use HyperliquidAPI.validate_and_round_order.")

    # DEPRECATED: Use HyperliquidAPI.validate_and_round_order for all live order validation and rounding.
    def round_price(self, symbol: str, raw_price: float) -> float:
        """
        DEPRECATED: Use HyperliquidAPI.validate_and_round_order for all live order validation and rounding.
        """
        raise NotImplementedError("round_price is deprecated. Use HyperliquidAPI.validate_and_round_order.")

    # DEPRECATED: Use HyperliquidAPI.validate_and_round_order for all live order validation and rounding.
    def validate_order(self, symbol: str, size: float, price: float) -> tuple[float, float]:
        """
        DEPRECATED: Use HyperliquidAPI.validate_and_round_order for all live order validation and rounding.
        """
        raise NotImplementedError("validate_order is deprecated. Use HyperliquidAPI.validate_and_round_order.")

    # DEPRECATED: Use HyperliquidAPI.validate_and_round_order for all live order validation and rounding.
    def validate_for_wire_format(self, symbol: str, size: float, price: float) -> tuple[float, float]:
        """
        DEPRECATED: Use HyperliquidAPI.validate_and_round_order for all live order validation and rounding.
        """
        raise NotImplementedError("validate_for_wire_format is deprecated. Use HyperliquidAPI.validate_and_round_order.")

    def get_all_symbols(self) -> list[str]:
        """Get all available symbols"""
        return list(self._mapping.keys())

    def print_mapping(self):
        """Print the current symbol mapping for debugging"""
        print("Symbol Mapping:")
        print("=" * 80)
        for symbol, info in self._mapping.items():
            print(f"{symbol:>8}: lot={info['lot']:<8} tick={info['tick']:<8} asset_id={info['asset_id']:<3} sz_dec={info['sz_decimals']:<2} px_dec={info['px_decimals']:<2}")
        print("=" * 80) 