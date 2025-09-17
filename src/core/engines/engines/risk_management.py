from src.core.utils.decimal_boundary_guard import safe_float
from core.api.hyperliquid_api import HyperliquidAPI
from core.utils.logger import Logger
from core.utils.emergency_handler import EmergencyHandler
from core.utils.config_manager import ConfigManager
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class RiskManagement:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.hyperliquid_api = HyperliquidAPI()
        self.logger = Logger()
        self.emergency_handler_instance = EmergencyHandler()

        # Load risk parameters from config
        self.max_drawdown = config.get("risk.max_drawdown", 0.05)
        self.max_exposure_per_asset = config.get("risk.max_exposure_per_asset", 0.2)
        self.max_total_exposure = config.get("risk.max_total_exposure", 0.5)
        self.liquidation_buffer = config.get("risk.liquidation_buffer", 0.02)
        
        # Dynamic leverage parameters
        self.base_leverage = config.get("risk.base_leverage", 10)
        self.min_leverage = config.get("risk.min_leverage", 2)
        self.max_leverage = config.get("risk.max_leverage", 20)
        self.drawdown_limit = config.get("risk.drawdown_limit", 0.05)
        
        # Drawdown tracking
        self.initial_equity = self._get_current_equity()
        self.peak_equity = self.initial_equity
        self.token_peaks = {}  # Track peak equity per token
        self.token_drawdowns = {}  # Track drawdown per token
        self.cooldown_tokens = {}  # Tokens in drawdown cooldown
        
        # Historical data for volatility calculation
        self.price_history = {}  # Store recent prices for volatility calculation

    def _get_current_equity(self):
        """
        Fetches the current total equity from the exchange.
        """
        try:
            user_state = self.hyperliquid_api.get_user_state()
            if user_state and "marginSummary" in user_state and "equity" in user_state["marginSummary"]:
                total_equity = safe_float(user_state["marginSummary"]["equity"])
                self.logger.info(f"Current equity: {total_equity}")
                return total_equity
            else:
                self.logger.warning("Could not retrieve current equity from user state. User state: %s", user_state)
                return 0.0
        except Exception as e:
            self.logger.error(f"Error getting current equity: {e}")
            return 0.0

    def compute_dynamic_leverage(self, token: str) -> float:
        """
        Compute dynamic leverage based on volatility and market conditions.
        
        Args:
            token (str): Token symbol
            
        Returns:
            float: Recommended leverage (between min_leverage and max_leverage)
        """
        try:
            # Get historical returns for volatility calculation
            returns = self._get_historical_returns(token, minutes=60)
            
            if not returns or len(returns) < 10:
                self.logger.warning(f"[RISK] Insufficient data for {token} leverage calculation")
                return self.base_leverage
            
            # Calculate volatility
            volatility = np.std(returns) * np.sqrt(len(returns))
            
            # Adjust leverage inversely to volatility
            # Higher volatility = lower leverage
            volatility_factor = 1.0 / max(volatility * 100, 0.1)  # Scale volatility
            
            # Calculate dynamic leverage
            dynamic_leverage = self.base_leverage * volatility_factor
            
            # Apply bounds
            leverage = np.clip(dynamic_leverage, self.min_leverage, self.max_leverage)
            
            self.logger.info(f"[RISK] {token} - Volatility: {volatility:.4f}, Leverage: {leverage:.2f}")
            
            return safe_float(leverage)
            
        except Exception as e:
            self.logger.error(f"[RISK] Error computing dynamic leverage for {token}: {e}")
            return self.base_leverage
    
    def _get_historical_returns(self, token: str, minutes: int = 60) -> list:
        """
        Get historical price returns for volatility calculation.

        Args:
            token (str): Token symbol
            minutes (int): Lookback period in minutes

        Returns:
            list: List of price returns
        """
        try:
            # In production, fetch actual historical data from API
            # For now, use synthetic data or cached prices
            
            if token not in self.price_history:
                # Initialize with current price
                market_data = self.hyperliquid_api.get_market_data(token)
                if market_data and "price" in market_data:
                    self.price_history[token] = [market_data["price"]]
                else:
                    return []
            
            # Generate synthetic returns for demonstration
            # In production, replace with actual historical data
            current_price = self.price_history[token][-1]
            returns = []
            
            for i in range(minutes):
                # Generate synthetic price movement
                price_change = np.random.normal(0, current_price * 0.001)  # 0.1% volatility
                new_price = current_price + price_change
                return_val = (new_price - current_price) / current_price
                returns.append(return_val)
                current_price = new_price
            
            return returns
            
        except Exception as e:
            self.logger.error(f"[RISK] Error getting historical returns for {token}: {e}")
            return []
    
    def check_drawdown_cooldown(self, token: str) -> bool:
        """
        Check if a token is in drawdown cooldown (should skip trading).

        Args:
            token (str): Token symbol

        Returns:
            bool: True if token should be skipped due to drawdown
        """
        try:
            # Get current position value for this token
            positions = self.hyperliquid_api.get_positions()
            current_value = 0.0
            
            for position in positions:
                if position.get("coin") == token:
                    size = safe_float(position.get("sz", "0"))
                    price = self._get_token_price(token)
                    current_value = size * price
                    break
            
            # Update peak if current value is higher
            if token not in self.token_peaks or current_value > self.token_peaks[token]:
                self.token_peaks[token] = current_value
            
            # Calculate drawdown
            if self.token_peaks[token] > 0:
                drawdown = (self.token_peaks[token] - current_value) / self.token_peaks[token]
                self.token_drawdowns[token] = drawdown
                
                # Check if drawdown exceeds limit
                if drawdown > self.drawdown_limit:
                    if token not in self.cooldown_tokens:
                        self.cooldown_tokens[token] = datetime.now()
                        self.logger.warning(f"[RISK] {token} entered drawdown cooldown: {drawdown:.2%}")
                    
                    # Check if cooldown period has passed (e.g., 1 hour)
                    cooldown_start = self.cooldown_tokens[token]
                    if datetime.now() - cooldown_start > timedelta(hours=1):
                        # Reset cooldown if drawdown has improved
                        if drawdown < self.drawdown_limit * 0.5:  # 50% improvement
                            del self.cooldown_tokens[token]
                            self.logger.info(f"[RISK] {token} drawdown cooldown cleared")
                            return False
                        else:
                            return True  # Still in cooldown
                    else:
                        return True  # In cooldown period
                else:
                    # No drawdown, clear cooldown if exists
                    if token in self.cooldown_tokens:
                        del self.cooldown_tokens[token]
                        self.logger.info(f"[RISK] {token} drawdown cooldown cleared")
            
            return False
            
        except Exception as e:
            self.logger.error(f"[RISK] Error checking drawdown cooldown for {token}: {e}")
            return False
    
    def _get_token_price(self, token: str) -> float:
        """Get current price for a token"""
        try:
            market_data = self.hyperliquid_api.get_market_data(token)
            if market_data and "price" in market_data:
                return market_data["price"]
            else:
                self.logger.warning(f"[RISK] No price data for {token}")
                return 0.0
        except Exception as e:
            self.logger.error(f"[RISK] Error getting price for {token}: {e}")
            return 0.0

    def calculate_adaptive_position_size(self, token: str, entry_price: float, stop_loss_price: float, 
                                       risk_per_trade_percent: float = 0.01) -> float:
        """
        Calculate position size with dynamic leverage and risk adaptation.
        
        Args:
            token (str): Token symbol
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price
            risk_per_trade_percent (float): Risk per trade percentage
            
        Returns:
            float: Calculated position size
        """
        try:
            # Check drawdown cooldown first
            if self.check_drawdown_cooldown(token):
                self.logger.info(f"[RISK] Skipping {token} due to drawdown cooldown")
                return 0.0

            # Get current equity
            current_equity = self._get_current_equity()
            if current_equity <= 0:
                self.logger.error("[RISK] Cannot calculate position size: equity <= 0")
                return 0.0

            # Calculate dynamic leverage
            leverage = self.compute_dynamic_leverage(token)
            
            # Calculate base position size
            dollar_risk_per_unit = abs(entry_price - stop_loss_price)
            if dollar_risk_per_unit == 0:
                self.logger.error("[RISK] Entry and stop loss prices are equal")
                return 0.0
            
            total_risk_amount = current_equity * risk_per_trade_percent
            base_position_size = total_risk_amount / dollar_risk_per_unit
            
            # Apply leverage adjustment
            leveraged_position_size = base_position_size * leverage
            
            # Apply exposure limits
            max_position_value = current_equity * self.max_exposure_per_asset
            max_position_size = max_position_value / entry_price
            
            final_position_size = min(leveraged_position_size, max_position_size)
            
            self.logger.info(f"[RISK] {token} - Leverage: {leverage:.2f}, Size: {final_position_size:.4f}")
            
            return final_position_size
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating adaptive position size for {token}: {e}")
            return 0.0

    def monitor_portfolio_risk(self) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk monitoring.
        
        Returns:
            dict: Risk metrics and status
        """
        try:
            if self.emergency_handler_instance.is_kill_switch_active():
                self.logger.warning("[RISK] Risk monitoring suspended: Kill switch active")
                return {"status": "suspended", "reason": "kill_switch_active"}
            
            current_equity = self._get_current_equity()
            if current_equity <= 0:
                self.logger.error("[RISK] Cannot monitor risk: equity <= 0")
                return {"status": "error", "reason": "no_equity"}
            
            # Update peak equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            # Calculate overall drawdown
            overall_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            # Get positions and calculate exposure
            positions = self.hyperliquid_api.get_positions()
            total_exposure = 0.0
            token_exposures = {}
            
            for position in positions:
                token = position.get("coin", "")
                size = safe_float(position.get("sz", "0"))
                price = self._get_token_price(token)
                value = abs(size * price)
                
                total_exposure += value
                token_exposures[token] = value
            
            # Calculate exposure percentages
            exposure_percentage = total_exposure / current_equity if current_equity > 0 else 0.0
            
            # Risk assessment
            risk_status = "normal"
            risk_alerts = []
            
            if overall_drawdown > self.max_drawdown:
                risk_status = "critical"
                risk_alerts.append(f"Overall drawdown {overall_drawdown:.2%} exceeds limit {self.max_drawdown:.2%}")
            
            if exposure_percentage > self.max_total_exposure:
                risk_status = "warning"
                risk_alerts.append(f"Total exposure {exposure_percentage:.2%} exceeds limit {self.max_total_exposure:.2%}")
            
            # Check individual token exposures
            for token, exposure in token_exposures.items():
                token_exposure_pct = exposure / current_equity if current_equity > 0 else 0.0
                if token_exposure_pct > self.max_exposure_per_asset:
                    risk_alerts.append(f"{token} exposure {token_exposure_pct:.2%} exceeds limit {self.max_exposure_per_asset:.2%}")
            
            # Log risk summary
            self.logger.info(f"[RISK] Portfolio risk - Drawdown: {overall_drawdown:.2%}, Exposure: {exposure_percentage:.2%}, Status: {risk_status}")
            
            # Trigger emergency if critical
            if risk_status == "critical":
                reason = f"Critical risk detected: {', '.join(risk_alerts)}"
                self.logger.critical(f"[RISK] {reason}")
                self.emergency_handler_instance.handle_emergency(reason)
            
            return {
                "status": risk_status,
                "current_equity": current_equity,
                "peak_equity": self.peak_equity,
                "overall_drawdown": overall_drawdown,
                "total_exposure": total_exposure,
                "exposure_percentage": exposure_percentage,
                "token_exposures": token_exposures,
                "cooldown_tokens": list(self.cooldown_tokens.keys()),
                "alerts": risk_alerts
            }
            
        except Exception as e:
            self.logger.error(f"[RISK] Error in portfolio risk monitoring: {e}")
            return {"status": "error", "reason": str(e)}
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary.
        
        Returns:
            dict: Complete risk summary
        """
        try:
            risk_metrics = self.monitor_portfolio_risk()
            
            # Add additional risk metrics
            summary = {
                "risk_metrics": risk_metrics,
                "risk_parameters": {
                    "max_drawdown": self.max_drawdown,
                    "max_exposure_per_asset": self.max_exposure_per_asset,
                    "max_total_exposure": self.max_total_exposure,
                    "base_leverage": self.base_leverage,
                    "min_leverage": self.min_leverage,
                    "max_leverage": self.max_leverage,
                    "drawdown_limit": self.drawdown_limit
                },
                "token_peaks": self.token_peaks,
                "token_drawdowns": self.token_drawdowns,
                "cooldown_tokens": self.cooldown_tokens
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"[RISK] Error getting risk summary: {e}")
            return {"error": str(e)}

    def monitor_risk(self, current_pnl):
        """
        Legacy method for backward compatibility.
        """
        self.monitor_portfolio_risk()

    def calculate_position_size(self, symbol, entry_price, stop_loss_price, risk_per_trade_percent=0.01):
        """
        Legacy method for backward compatibility.
        """
        return self.calculate_adaptive_position_size(symbol, entry_price, stop_loss_price, risk_per_trade_percent)

    def check_liquidation_risk(self, symbol, current_price, position_size, entry_price, leverage):
        """
        Checks if current positions are approaching liquidation price.
        This is a simplified check and needs to be integrated with actual margin calculations from Hyperliquid.

        Args:
            symbol (str): Trading pair symbol.
            current_price (float): Current market price.
            position_size (float): Current position size.
            entry_price (float): Average entry price of the position.
            leverage (float): Leverage used for the position.

        Returns:
            bool: True if liquidation risk is high, False otherwise.
        """
        # This is a highly simplified model. Actual liquidation price calculation is complex and depends on
        # initial margin, maintenance margin, and total account equity/collateral.
        # The Hyperliquid API provides user state which includes margin details.
        # For a robust solution, you would query user_state and calculate based on that.

        # Placeholder for actual liquidation price calculation
        # Example: For a long position, liquidation price is typically entry_price * (1 - (initial_margin_rate / leverage))
        # For a short position, liquidation price is typically entry_price * (1 + (initial_margin_rate / leverage))

        # For demonstration, let's assume a simple buffer check
        # You would get the actual liquidation price from Hyperliquid API or calculate it precisely.
        # For now, we'll just check if current price is within a certain buffer of a hypothetical liquidation point.

        # This part needs to be replaced with actual Hyperliquid margin logic
        self.logger.warning("Liquidation risk check is a simplified placeholder. Needs integration with Hyperliquid margin data.")
        return False

    def emergency_handler(self, reason):
        """
        Executes predefined emergency procedures.
        """
        self.logger.critical(f"Risk Management Emergency: {reason}", send_alert=True)
        self.cancel_all_orders()
        self.close_all_positions()
        self.emergency_handler_instance.handle_emergency(reason) # Trigger system-wide emergency

    def cancel_all_orders(self):
        """
        Cancels all open orders on the exchange.
        """
        self.logger.info("Cancelling all open orders...")
        try:
            open_orders = self.hyperliquid_api.get_open_orders()
            for order in open_orders:
                self.hyperliquid_api.cancel_order(order["oid"], order["coin"])
            self.logger.info("All open orders cancelled.")
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")

    def close_all_positions(self):
        """
        Closes all open positions on the exchange by placing market orders.
        """
        self.logger.info("Closing all open positions...")
        try:
            positions = self.hyperliquid_api.get_positions()
            for position in positions:
                if safe_float(position["szi"]) != 0: # Check if there's an open position
                    symbol = position["coin"]
                    current_price_data = self.hyperliquid_api.get_market_data(symbol)
                    if not current_price_data or "price" not in current_price_data:
                        self.logger.error(f"Could not get current price for {symbol} to close position.")
                        continue

                    current_price = current_price_data["price"]
                    side = "sell" if safe_float(position["szi"]) > 0 else "buy" # Opposite side to close
                    quantity = abs(safe_float(position["szi"]))

                    order_details = {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": current_price, # Market order, price is indicative
                        "order_type": "market",
                        "reduce_only": True # Ensure this closes position
                    }
                    self.hyperliquid_api.place_order(order_details)
                    self.logger.info(f"Placed market order to close {quantity} of {symbol} ({side}).")
            self.logger.info("All open positions closed.")
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")







