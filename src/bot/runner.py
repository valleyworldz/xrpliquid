#!/usr/bin/env python3
"""
Main Bot Runner - Async Execution Loop and Component Coordination
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from .config import BotConfig
from .signals import SignalGenerator
from .risk import RiskManager
from .execution import OrderExecutor
from .utils.logging import setup_logging
from .utils.metrics import MetricsManager

class BotRunner:
    """Main bot runner with async execution loop"""
    
    def __init__(self, config: Optional[BotConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or BotConfig()
        self.logger = logger or setup_logging()
        
        # Core components
        self.signal_generator = SignalGenerator(self.logger)
        self.risk_manager = RiskManager(self.logger)
        self.metrics = MetricsManager(enable_prometheus=self.config.get('ENABLE_METRICS', True))
        
        # Execution components (will be set during initialization)
        self.order_executor = None
        self.exchange_client = None
        self.info_client = None
        
        # Runtime state
        self.running = False
        self.cycle_count = 0
        self.last_cycle_time = 0
        self.start_time = None
        
        # Performance tracking
        self.cycle_times = []
        self.error_count = 0
        self.success_count = 0
        
    async def initialize(self, exchange_client, info_client, wallet_address: str):
        """Initialize bot with exchange clients"""
        try:
            self.exchange_client = exchange_client
            self.info_client = info_client
            self.wallet_address = wallet_address
            
            # Initialize order executor
            self.order_executor = OrderExecutor(exchange_client, info_client, self.logger)
            
            # Set start time
            self.start_time = time.time()
            
            self.logger.info("Bot runner initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing bot runner: {e}")
            return False
    
    async def start(self):
        """Start the bot execution loop"""
        try:
            self.running = True
            self.logger.info("🚀 Starting bot execution loop...")
            
            # Main execution loop
            while self.running:
                cycle_start = time.time()
                
                try:
                    await self._execute_trading_cycle()
                    self.success_count += 1
                    
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"Error in trading cycle: {e}")
                    
                    # Record error in metrics
                    self.metrics.record_risk_check_failure("cycle_error")
                
                # Update cycle tracking
                cycle_duration = time.time() - cycle_start
                self.cycle_times.append(cycle_duration)
                self.cycle_count += 1
                
                # Keep only last 100 cycle times
                if len(self.cycle_times) > 100:
                    self.cycle_times = self.cycle_times[-100:]
                
                # Record metrics
                self.metrics.record_cycle_duration(cycle_duration)
                self.metrics.update_uptime(time.time() - self.start_time)
                
                # Wait for next cycle
                cycle_interval = self.config.get('CYCLE_INTERVAL', 2)
                await asyncio.sleep(cycle_interval)
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Bot stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Bot crashed: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _execute_trading_cycle(self):
        """Execute a single trading cycle"""
        try:
            # Get market data
            current_price = await self._get_current_price()
            if not current_price:
                self.logger.warning("Could not get current price")
                return
            
            # Get account status
            account_data = await self._get_account_status()
            if not account_data:
                self.logger.warning("Could not get account status")
                return
            
            # Extract account information
            free_collateral = float(account_data.get("withdrawable", 0))
            margin_ratio = float(account_data.get("marginRatio", 0))
            
            # Check margin safety
            if not self.risk_manager.check_margin_ratio(margin_ratio, free_collateral):
                self.logger.warning("Margin ratio check failed")
                return
            
            # Get price history for signal analysis
            price_history = await self._get_price_history()
            if len(price_history) < 50:
                self.logger.warning("Insufficient price history for analysis")
                return
            
            # Generate trading signals
            signal = self.signal_generator.analyze_market_signals(
                price_history, current_price=current_price
            )
            
            # Record signal metrics
            if signal and signal.get('signal'):
                self.metrics.record_signal(
                    signal['signal'], 
                    signal.get('confidence', 0)
                )
            
            # Check if we should trade
            if not self._should_trade(signal, free_collateral):
                return
            
            # Execute trade
            await self._execute_trade(signal, current_price, free_collateral)
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            raise
    
    async def _get_current_price(self) -> Optional[float]:
        """Get current XRP price"""
        try:
            # Try to get price from info client
            if hasattr(self.info_client, 'l2_snapshot'):
                l2_data = self.info_client.l2_snapshot("XRP")
                if l2_data:
                    normalized = self.order_executor._normalize_l2_snapshot(l2_data)
                    if normalized["bids"] and normalized["asks"]:
                        bid = normalized["bids"][0][0]
                        ask = normalized["asks"][0][0]
                        return (bid + ask) / 2
            
            # Fallback to other methods
            if hasattr(self.info_client, 'get_current_price'):
                return self.info_client.get_current_price("XRP")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    async def _get_account_status(self) -> Optional[Dict[str, Any]]:
        """Get account status"""
        try:
            if hasattr(self.info_client, 'user_state'):
                return self.info_client.user_state(self.wallet_address)
            return None
        except Exception as e:
            self.logger.error(f"Error getting account status: {e}")
            return None
    
    async def _get_price_history(self) -> List[float]:
        """Get price history for analysis"""
        try:
            # This would typically come from a price feed or database
            # For now, return a placeholder
            return [0.5] * 100  # Placeholder
        except Exception as e:
            self.logger.error(f"Error getting price history: {e}")
            return []
    
    def _should_trade(self, signal: Dict[str, Any], free_collateral: float) -> bool:
        """Determine if we should execute a trade"""
        try:
            # Check if we have a valid signal
            if not signal or not signal.get('signal'):
                return False
            
            # Check signal confidence
            confidence = signal.get('confidence', 0)
            if confidence < self.config.get('CONFIDENCE_THRESHOLD', 0.95):
                return False
            
            # Check risk limits
            trade_value = free_collateral * 0.1  # Estimate trade value
            if not self.risk_manager.check_risk_limits(trade_value, confidence):
                return False
            
            # Check if we have sufficient collateral
            if free_collateral < self.config.get('MIN_ORDER_VALUE', 10.0):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking if should trade: {e}")
            return False
    
    async def _execute_trade(self, signal: Dict[str, Any], current_price: float, 
                           free_collateral: float):
        """Execute a trade based on signal"""
        try:
            signal_type = signal['signal']
            confidence = signal.get('confidence', 0)
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                current_price, free_collateral, confidence, signal_type
            )
            
            if position_size <= 0:
                self.logger.warning("Position size too small to trade")
                return
            
            # Determine order side
            is_buy = signal_type == 'LONG'
            
            # Place order
            order_result = await self.order_executor.place_order(
                "XRP", is_buy, position_size, current_price, "limit"
            )
            
            if order_result.get('success'):
                self.logger.info(f"✅ Trade executed: {signal_type} {position_size} XRP @ ${current_price:.4f}")
                
                # Record trade metrics
                trade_value = position_size * current_price
                self.metrics.record_trade(
                    "BUY" if is_buy else "SELL",
                    "SUCCESS",
                    trade_value,
                    position_size
                )
                
                # Place TP/SL triggers
                await self._place_tp_sl_triggers(position_size, current_price, is_buy)
                
                # Update position tracking
                self.order_executor.update_position("XRP", position_size, current_price, is_buy)
                
            else:
                self.logger.error(f"❌ Trade failed: {order_result.get('error', 'Unknown error')}")
                
                # Record failed trade
                trade_value = position_size * current_price
                self.metrics.record_trade(
                    "BUY" if is_buy else "SELL",
                    "FAILED",
                    trade_value,
                    position_size
                )
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    async def _place_tp_sl_triggers(self, position_size: float, entry_price: float, is_long: bool):
        """Place take profit and stop loss triggers"""
        try:
            tp_percentage = self.config.get('PROFIT_TARGET_PCT', 0.035)
            sl_percentage = self.config.get('STOP_LOSS_PCT', 0.025)
            
            trigger_result = await self.order_executor.place_tp_sl_triggers(
                "XRP", position_size, entry_price, is_long, tp_percentage, sl_percentage
            )
            
            if trigger_result.get('success'):
                self.logger.info("✅ TP/SL triggers placed successfully")
            else:
                self.logger.warning(f"⚠️ Failed to place TP/SL triggers: {trigger_result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error placing TP/SL triggers: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            self.running = False
            
            # Cancel all active orders
            if self.order_executor:
                await self.order_executor.cancel_all_triggers("XRP")
            
            # Log final statistics
            runtime = time.time() - self.start_time if self.start_time else 0
            avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0
            
            self.logger.info(f"📊 Bot Statistics:")
            self.logger.info(f"   Runtime: {runtime:.1f} seconds")
            self.logger.info(f"   Cycles: {self.cycle_count}")
            self.logger.info(f"   Success Rate: {self.success_count / max(self.cycle_count, 1) * 100:.1f}%")
            self.logger.info(f"   Avg Cycle Time: {avg_cycle_time:.3f} seconds")
            self.logger.info(f"   Errors: {self.error_count}")
            
            self.logger.info("🛑 Bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        try:
            runtime = time.time() - self.start_time if self.start_time else 0
            avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0
            
            return {
                "running": self.running,
                "runtime": runtime,
                "cycle_count": self.cycle_count,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "avg_cycle_time": avg_cycle_time,
                "active_positions": self.order_executor.active_positions if self.order_executor else {},
                "active_triggers": self.order_executor.active_triggers if self.order_executor else {},
                "risk_summary": self.risk_manager.get_risk_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

# Convenience function for backward compatibility
async def run_bot(exchange_client, info_client, wallet_address: str,
                 config: Optional[BotConfig] = None,
                 logger: Optional[logging.Logger] = None):
    """Run bot (convenience function)"""
    runner = BotRunner(config, logger)
    if await runner.initialize(exchange_client, info_client, wallet_address):
        await runner.start()
    else:
        raise RuntimeError("Failed to initialize bot runner") 