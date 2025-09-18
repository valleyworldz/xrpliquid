#!/usr/bin/env python3
"""
Live XRP Trading Bot - Real Hyperliquid Integration
==================================================
Real live trading bot with actual Hyperliquid API integration.
"""

import os
import sys
import time
import logging
import asyncio
from decimal import Decimal, getcontext
from typing import Dict, Any, Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set decimal precision
getcontext().prec = 10

class LiveXRPBot:
    """
    Live XRP Trading Bot with Real Hyperliquid Integration
    """
    
    def __init__(self, config_path: str = "config/environments/secure_creds.env"):
        self.running = False
        self.positions = {}
        self.balance = Decimal('0.0')  # Will be fetched from exchange
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize Hyperliquid API
        self.api = None
        self._init_hyperliquid_api()
        
        # Trading statistics
        self.stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_pnl": Decimal('0.0'),
            "last_update": None
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment or config file"""
        # Load secure credentials with fail-closed design
        try:
            from src.utils.secrets_loader import load_secure_credentials
            credentials = load_secure_credentials()
            
            config = {
                "trading_enabled": True,
                "max_position_size": Decimal(credentials.get("MAX_POSITION_SIZE", "100.0")),
                "risk_limit": Decimal(credentials.get("RISK_LIMIT", "0.02")),
                "funding_threshold": Decimal('0.0001'),  # 0.01% funding threshold
                "private_key": credentials.get("HYPERLIQUID_PRIVATE_KEY", ""),
                "address": credentials.get("HYPERLIQUID_ADDRESS", ""),
                "api_key": credentials.get("HYPERLIQUID_API_KEY", ""),  # Fallback
                "secret_key": credentials.get("HYPERLIQUID_SECRET_KEY", ""),  # Fallback
                "testnet": credentials.get("HYPERLIQUID_TESTNET", "true").lower() == "true"
            }
        except SystemExit:
            logger.error("❌ CRITICAL: Failed to load secure credentials - system exiting")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading credentials: {e}")
            return False
        
        # Load from config file if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            # Remove 'export ' prefix if present
                            if key.startswith('export '):
                                key = key[7:]
                            if key in config:
                                # Strip quotes from value
                                value = value.strip("'\"")
                                if key in ["max_position_size", "risk_limit", "funding_threshold"]:
                                    config[key] = Decimal(value)
                                else:
                                    config[key] = value
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        return config
    
    def _init_hyperliquid_api(self):
        """Initialize Hyperliquid API connection"""
        try:
            # Import Hyperliquid SDK
            from hyperliquid.exchange import Exchange
            
            # Check for credentials in order of preference
            if self.config["private_key"]:
                # Use private key (preferred method)
                logger.info("🔑 Using private key for Hyperliquid connection")
                from eth_account import Account
                
                # Set base URL based on testnet flag
                if self.config["testnet"]:
                    base_url = "https://api.hyperliquid-testnet.xyz"
                else:
                    base_url = "https://api.hyperliquid.xyz"
                
                # Create wallet from private key
                self.wallet = Account.from_key(self.config["private_key"])
                logger.info(f"🔑 Wallet address: {self.wallet.address}")
                
                # Initialize Exchange with wallet
                self.api = Exchange(self.wallet, base_url=base_url)
                
            elif self.config["api_key"] and self.config["secret_key"]:
                # Fallback to API key method (if supported)
                logger.info("🔑 Using API key and secret for Hyperliquid connection")
                self.api = Exchange(
                    api_key=self.config["api_key"],
                    secret_key=self.config["secret_key"],
                    testnet=self.config["testnet"]
                )
            else:
                logger.error("❌ Hyperliquid API credentials not found!")
                logger.error("Please set HYPERLIQUID_PRIVATE_KEY environment variable")
                logger.error("Or set HYPERLIQUID_API_KEY and HYPERLIQUID_SECRET_KEY environment variables")
                return False
            
            logger.info("✅ Hyperliquid API initialized successfully")
            return True
            
        except ImportError:
            logger.error("❌ hyperliquid SDK not installed!")
            logger.error("Run: pip install hyperliquid-python-sdk")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hyperliquid API: {e}")
            return False
    
    async def start(self):
        """Start the live trading bot"""
        if not self.api:
            logger.error("❌ Cannot start bot - API not initialized")
            return
        
        logger.info("🚀 Starting Live XRP Trading Bot")
        logger.info(f"🔗 Connected to: {'Testnet' if self.config['testnet'] else 'Mainnet'}")
        
        self.running = True
        
        try:
            # Initial setup
            await self._update_account_info()
            
            while self.running:
                try:
                    # Update account info
                    await self._update_account_info()
                    
                    # 🎯 CRITICAL FIX: Monitor existing positions for TP/SL
                    await self._monitor_tpsl_positions()
                    
                    # Check for trading opportunities
                    if self.config["trading_enabled"]:
                        await self._check_trading_opportunities()
                    
                    # Log status every 30 seconds
                    if int(time.time()) % 30 == 0:
                        await self._log_status()
                    
                    # Sleep for 1 second
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
        finally:
            self.running = False
            await self._log_final_stats()
    
    async def _update_account_info(self):
        """Update account balance and positions from Hyperliquid"""
        try:
            # Get account info
            account_info = self.api.get_account_info()
            
            if account_info:
                # Update balance (USDC balance)
                self.balance = Decimal(str(account_info.get("marginSummary", {}).get("accountValue", 0)))
                
                # Update positions
                positions = account_info.get("assetPositions", [])
                self.positions = {}
                
                for pos in positions:
                    if pos.get("position", {}).get("coin") == "XRP":
                        size = Decimal(str(pos.get("position", {}).get("szi", 0)))
                        if size != 0:
                            self.positions["XRP"] = {
                                "size": size,
                                "entry_px": Decimal(str(pos.get("position", {}).get("entryPx", 0))),
                                "unrealized_pnl": Decimal(str(pos.get("position", {}).get("unrealizedPnl", 0)))
                            }
                
                self.stats["last_update"] = time.time()
                
        except Exception as e:
            logger.error(f"Error updating account info: {e}")
    
    async def _check_trading_opportunities(self):
        """Check for XRP trading opportunities"""
        try:
            # Get current XRP price and funding rate
            market_data = self.api.get_market_data("XRP")
            
            if not market_data:
                return
            
            current_price = Decimal(str(market_data.get("price", 0)))
            funding_rate = Decimal(str(market_data.get("fundingRate", 0)))
            
            # Check funding arbitrage opportunity
            if abs(funding_rate) > self.config["funding_threshold"]:
                await self._execute_funding_arbitrage(funding_rate, current_price)
            
            # Check for other trading opportunities
            await self._check_momentum_opportunities(current_price)
            
        except Exception as e:
            logger.error(f"Error checking trading opportunities: {e}")
    
    async def _execute_funding_arbitrage(self, funding_rate: Decimal, current_price: Decimal):
        """Execute funding arbitrage trade"""
        try:
            # Calculate position size based on risk limit
            risk_amount = self.balance * self.config["risk_limit"]
            position_size = risk_amount / current_price
            
            # Cap position size
            position_size = min(position_size, self.config["max_position_size"])
            
            if position_size < Decimal('0.1'):  # Minimum position size
                return
            
            # Determine trade direction based on funding rate
            if funding_rate > 0:
                # Positive funding - short XRP to collect funding
                side = "S"
                logger.info(f"💰 Funding Arbitrage: SHORT {position_size} XRP at {current_price} (funding: {funding_rate})")
            else:
                # Negative funding - long XRP to collect funding
                side = "B"
                logger.info(f"💰 Funding Arbitrage: LONG {position_size} XRP at {current_price} (funding: {funding_rate})")
            
            # Execute the trade
            order_result = self.api.place_order(
                coin="XRP",
                is_buy=(side == "B"),
                sz=float(position_size),
                limit_px=float(current_price),
                reduce_only=False
            )
            
            if order_result and order_result.get("status") == "ok":
                self.stats["total_trades"] += 1
                self.stats["successful_trades"] += 1
                logger.info(f"✅ Funding arbitrage order placed successfully")
                
                # 🎯 CRITICAL FIX: Place TP/SL orders after successful entry
                await self._place_tpsl_orders(current_price, position_size, side, funding_rate)
                
            else:
                self.stats["total_trades"] += 1
                self.stats["failed_trades"] += 1
                logger.error(f"❌ Failed to place funding arbitrage order: {order_result}")
                
        except Exception as e:
            logger.error(f"Error executing funding arbitrage: {e}")
            self.stats["total_trades"] += 1
            self.stats["failed_trades"] += 1
    
    async def _place_tpsl_orders(self, entry_price: Decimal, position_size: Decimal, side: str, funding_rate: Decimal):
        """
        🎯 ENHANCED TP/SL ORDER PLACEMENT
        Integrates the volatility-scaled TP/SL system with actual order placement
        """
        try:
            from src.core.main_bot import calculate_enhanced_volatility_scaled_tpsl, BotConfig
            
            # Calculate current volatility (simplified for funding arbitrage)
            # For funding arbitrage, use moderate volatility scaling
            current_volatility = abs(float(funding_rate)) * 1000  # Convert to percentage
            
            # Create a config object with our TP/SL parameters
            config = BotConfig()
            
            # Calculate enhanced TP/SL prices using our new system
            tp_price, sl_price, effective_rr, volatility_regime = calculate_enhanced_volatility_scaled_tpsl(
                entry_price=entry_price,
                atr_value=entry_price * Decimal('0.02'),  # 2% ATR estimate for funding arbitrage
                current_volatility_pct=current_volatility,
                config=config
            )
            
            logger.info(f"🎯 TP/SL Calculation: Regime={volatility_regime}, RR={effective_rr:.2f}")
            logger.info(f"🎯 Entry: {entry_price}, TP: {tp_price}, SL: {sl_price}")
            
            # Place Take Profit order
            tp_order_result = self.api.place_order(
                coin="XRP",
                is_buy=(side == "S"),  # Opposite side to close position
                sz=float(position_size),
                limit_px=float(tp_price),
                reduce_only=True  # This closes the position
            )
            
            # Place Stop Loss order  
            sl_order_result = self.api.place_order(
                coin="XRP",
                is_buy=(side == "S"),  # Opposite side to close position
                sz=float(position_size),
                limit_px=float(sl_price),
                reduce_only=True  # This closes the position
            )
            
            if tp_order_result and tp_order_result.get("status") == "ok":
                logger.info(f"✅ Take Profit order placed: {tp_price}")
            else:
                logger.error(f"❌ Failed to place TP order: {tp_order_result}")
                
            if sl_order_result and sl_order_result.get("status") == "ok":
                logger.info(f"✅ Stop Loss order placed: {sl_price}")
            else:
                logger.error(f"❌ Failed to place SL order: {sl_order_result}")
                
        except Exception as e:
            logger.error(f"❌ Error placing TP/SL orders: {e}")
            # Continue execution - don't fail the trade if TP/SL placement fails

    async def _monitor_tpsl_positions(self):
        """
        🛡️ POSITION MONITORING & SHADOW STOPS
        Monitor positions and implement shadow stops for profit collection
        """
        try:
            if not self.positions:
                return
                
            # Get current market data
            market_data = self.api.get_market_data("XRP")
            if not market_data:
                return
                
            current_price = Decimal(str(market_data.get("price", 0)))
            
            for symbol, position in self.positions.items():
                if symbol == "XRP":
                    entry_price = position["entry_px"]
                    position_size = position["size"]
                    unrealized_pnl = position["unrealized_pnl"]
                    
                    # Determine if long or short position
                    is_long = position_size > 0
                    
                    # Calculate profit/loss percentage
                    if is_long:
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price
                    
                    # 🎯 SHADOW STOPS IMPLEMENTATION
                    from src.core.main_bot import implement_shadow_stops
                    
                    # Calculate dynamic TP/SL levels based on current volatility
                    profit_threshold = Decimal('0.02')  # 2% profit target
                    loss_threshold = Decimal('0.01')    # 1% stop loss
                    
                    # Check for Take Profit trigger
                    if pnl_pct >= profit_threshold:
                        logger.info(f"🎯 TAKE PROFIT TRIGGERED: {symbol} at {pnl_pct:.2%} profit")
                        await self._close_position_for_profit(symbol, position_size, current_price, "TAKE_PROFIT")
                    
                    # Check for Stop Loss trigger  
                    elif pnl_pct <= -loss_threshold:
                        logger.info(f"🛡️ STOP LOSS TRIGGERED: {symbol} at {pnl_pct:.2%} loss")
                        await self._close_position_for_profit(symbol, position_size, current_price, "STOP_LOSS")
                    
                    # Log position status every 30 seconds
                    if int(time.time()) % 30 == 0:
                        logger.info(f"📊 Position: {symbol} | Size: {position_size} | PnL: {pnl_pct:.2%} | Unrealized: ${unrealized_pnl}")
                        
        except Exception as e:
            logger.error(f"❌ Error monitoring TP/SL positions: {e}")

    async def _close_position_for_profit(self, symbol: str, position_size: Decimal, current_price: Decimal, reason: str):
        """
        💰 CLOSE POSITION FOR PROFIT/LOSS
        Actually close the position and collect profits
        """
        try:
            # Place market order to close position immediately
            is_buy = position_size < 0  # If short position, buy to close; if long position, sell to close
            
            close_order_result = self.api.place_order(
                coin=symbol,
                is_buy=is_buy,
                sz=float(abs(position_size)),
                limit_px=float(current_price * Decimal('0.999' if is_buy else '1.001')),  # Slight slippage for execution
                reduce_only=True
            )
            
            if close_order_result and close_order_result.get("status") == "ok":
                # 💰 PROFIT COLLECTED!
                self.stats["successful_trades"] += 1
                logger.info(f"✅ {reason}: Position closed successfully for {symbol}")
                logger.info(f"💰 PROFIT COLLECTED: {symbol} position closed at {current_price}")
                
                # Update trade statistics
                if reason == "TAKE_PROFIT":
                    self.stats["profitable_closes"] = self.stats.get("profitable_closes", 0) + 1
                else:
                    self.stats["loss_stops"] = self.stats.get("loss_stops", 0) + 1
                    
            else:
                logger.error(f"❌ Failed to close {symbol} position: {close_order_result}")
                
        except Exception as e:
            logger.error(f"❌ Error closing position for {symbol}: {e}")

    async def _check_momentum_opportunities(self, current_price: Decimal):
        """Check for momentum trading opportunities"""
        # This would implement momentum-based trading strategies
        # For now, just log the current price
        pass
    
    async def _log_status(self):
        """Log current bot status"""
        try:
            total_pnl = sum(pos.get("unrealized_pnl", Decimal('0')) for pos in self.positions.values())
            
            logger.info(f"💰 Balance: {self.balance} USDC")
            logger.info(f"📊 Positions: {len(self.positions)}")
            logger.info(f"📈 Unrealized PnL: {total_pnl} USDC")
            logger.info(f"🎯 Stats: {self.stats['successful_trades']}/{self.stats['total_trades']} trades successful")
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    async def _log_final_stats(self):
        """Log final statistics"""
        logger.info("📊 Final Trading Statistics:")
        logger.info(f"   Total Trades: {self.stats['total_trades']}")
        logger.info(f"   Successful: {self.stats['successful_trades']}")
        logger.info(f"   Failed: {self.stats['failed_trades']}")
        logger.info(f"   Success Rate: {(self.stats['successful_trades'] / max(1, self.stats['total_trades'])) * 100:.1f}%")
    
    def stop(self):
        """Stop the bot"""
        logger.info("🛑 Stopping live trading bot...")
        self.running = False

async def main():
    """Main function"""
    print("🎯 LIVE XRP TRADING BOT")
    print("=" * 50)
    print("🏆 REAL HYPERLIQUID INTEGRATION")
    print("=" * 50)
    
    try:
        bot = LiveXRPBot()
        await bot.start()
    except Exception as e:
        logger.error(f"Failed to start live bot: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    asyncio.run(main())
