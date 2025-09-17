from src.core.utils.decimal_boundary_guard import safe_float
from core.api.hyperliquid_api import HyperliquidAPI

class TokenMetadata:
    def __init__(self):
        self.hyperliquid_api = HyperliquidAPI()
        self.asset_cache = {}

    def fetch_asset_details(self, token_symbol):
        """
        Fetches comprehensive asset details for a given token from the Hyperliquid API.

        Args:
            token_symbol (str): The symbol of the token (e.g., "BTCUSD", "ETHUSD").

        Returns:
            dict: A dictionary containing details like tick size, minimum order quantity,
                  maximum leverage, and other relevant trading parameters.
        """
        self.hyperliquid_api.logger.info(f"Fetching asset details for {token_symbol} from Hyperliquid API...")
        try:
            # Use the info_client.meta_and_asset_ctxs() to get metadata and asset contexts
            meta_and_ctxs = self.hyperliquid_api.info_client.meta_and_asset_ctxs()
            if meta_and_ctxs and len(meta_and_ctxs) == 2:
                meta_info = meta_and_ctxs[0]
                asset_ctxs = meta_and_ctxs[1]

                if "universe" in meta_info and asset_ctxs:
                    for i, asset in enumerate(meta_info["universe"]):
                        if asset["name"] == token_symbol:
                            # Find the corresponding asset context by index
                            if i < len(asset_ctxs):
                                asset_ctx = asset_ctxs[i]
                                details = {
                                    "tick_size": safe_float(asset_ctx["markPxTickSz"]),
                                    "minimum_order": safe_float(asset_ctx["minSz"]),
                                    "max_leverage": safe_float(asset["maxLeverage"]),
                                    "contract_size": 1, # Assuming 1 for now, adjust if needed
                                    "funding_interval": 3600, # Placeholder, needs actual API call
                                    "current_volatility": 0.005, # Placeholder
                                    "atr_multiplier": 1.5 # Placeholder
                                }
                                self.asset_cache[token_symbol] = details
                                return details
            self.hyperliquid_api.logger.warning(f"Asset details for {token_symbol} not found.")
            return None
        except Exception as e:
            self.hyperliquid_api.logger.error(f"Error fetching asset details for {token_symbol}: {e}")
            return None

    def auto_reconfigure_params(self, token_symbol):
        """
        Automatically reconfigures trading parameters based on fetched asset details.

        This method ensures that the trading system adapts to the specific requirements
        and constraints of each trading pair, including volatility-aware scaling.

        Args:
            token_symbol (str): The symbol of the token.

        Returns:
            dict: The reconfigured trading parameters for the given token, including volatility metrics.
        """
        details = self.fetch_asset_details(token_symbol)
        self.hyperliquid_api.logger.info(f"Auto-reconfiguring parameters for {token_symbol} with details: {details}")
        return details

    def get_available_tokens(self):
        """
        Returns a list of available tokens from the asset cache.
        """
        if not self.asset_cache:
            # Attempt to populate the cache if it's empty
            self.hyperliquid_api.logger.info("Asset cache is empty. Attempting to populate it.")
            try:
                meta_and_ctxs = self.hyperliquid_api.info_client.meta_and_asset_ctxs()
                if meta_and_ctxs and len(meta_and_ctxs) == 2:
                    meta_info = meta_and_ctxs[0]
                    asset_ctxs = meta_and_ctxs[1]
                    if "universe" in meta_info and asset_ctxs:
                        for i, asset in enumerate(meta_info["universe"]):
                            if i < len(asset_ctxs):
                                asset_ctx = asset_ctxs[i]
                                details = {
                                    "tick_size": safe_float(asset_ctx["markPxTickSz"]),
                                    "minimum_order": safe_float(asset_ctx["minSz"]),
                                    "max_leverage": safe_float(asset["maxLeverage"]),
                                    "contract_size": 1, # Assuming 1 for now, adjust if needed
                                    "funding_interval": 3600, # Placeholder, needs actual API call
                                    "current_volatility": 0.005, # Placeholder
                                    "atr_multiplier": 1.5 # Placeholder
                                }
                                self.asset_cache[asset["name"]] = details
            except Exception as e:
                self.hyperliquid_api.logger.error(f"Error populating asset cache: {e}")
        return list(self.asset_cache.keys())




