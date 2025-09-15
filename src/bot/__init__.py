#!/usr/bin/env python3
"""
XRP Trading Bot - Modular Architecture
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from .config import BotConfig
from .runner import BotRunner
from .utils.logging import setup_logging
from .utils.metrics import MetricsManager

class XRPTradingBot:
    """Main XRP Trading Bot with modular architecture"""
    
    def __init__(self, config_file: Optional[str] = None, 
                 log_level: str = "INFO", enable_metrics: bool = True):
        """
        Initialize the XRP Trading Bot
        
        Args:
            config_file: Optional configuration file path
            log_level: Logging level
            enable_metrics: Enable Prometheus metrics
        """
        # Initialize configuration
        self.config = BotConfig(config_file)
        
        # Setup logging
        self.logger = setup_logging(log_level)
        
        # Setup metrics
        self.metrics = MetricsManager(enable_prometheus=enable_metrics)
        
        # Initialize bot runner
        self.runner = BotRunner(self.config, self.logger)
        
        # Exchange clients (will be set during setup)
        self.exchange_client = None
        self.info_client = None
        self.wallet_address = None
        
        # Runtime state
        self.initialized = False
        self.running = False
        
        self.logger.info("XRP Trading Bot initialized with modular architecture")
    
    def setup_api_clients(self, exchange_client, info_client, wallet_address: str):
        """Setup API clients"""
        try:
            self.exchange_client = exchange_client
            self.info_client = info_client
            self.wallet_address = wallet_address
            
            self.logger.info("API clients configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up API clients: {e}")
            return False
    
    async def initialize(self):
        """Initialize the bot"""
        try:
            if not self.exchange_client or not self.info_client:
                raise RuntimeError("API clients not configured. Call setup_api_clients() first.")
            
            # Initialize bot runner
            success = await self.runner.initialize(
                self.exchange_client, 
                self.info_client, 
                self.wallet_address
            )
            
            if success:
                self.initialized = True
                self.logger.info("✅ Bot initialized successfully")
                return True
            else:
                self.logger.error("❌ Failed to initialize bot")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing bot: {e}")
            return False
    
    async def start(self):
        """Start the bot"""
        try:
            if not self.initialized:
                if not await self.initialize():
                    raise RuntimeError("Bot initialization failed")
            
            self.running = True
            self.logger.info("🚀 Starting XRP Trading Bot...")
            
            # Start the bot runner
            await self.runner.start()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Bot stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Bot crashed: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the bot"""
        try:
            self.running = False
            if self.runner:
                await self.runner.shutdown()
            self.logger.info("🛑 Bot stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        try:
            status = {
                "initialized": self.initialized,
                "running": self.running,
                "config": self.config.to_dict(),
                "metrics": self.metrics.get_metrics_summary()
            }
            
            if self.runner:
                status.update(self.runner.get_status())
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.to_dict()
    
    def update_config(self, key: str, value: Any):
        """Update configuration value"""
        try:
            self.config.set(key, value)
            self.logger.info(f"Configuration updated: {key} = {value}")
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
    
    async def run_async(self):
        """Async entry point for backward compatibility"""
        await self.start()
    
    def run(self):
        """Synchronous entry point for backward compatibility"""
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Bot crashed: {e}")
            raise

# Convenience functions for backward compatibility
def create_bot(config_file: Optional[str] = None, 
               log_level: str = "INFO", 
               enable_metrics: bool = True) -> XRPTradingBot:
    """Create a new bot instance"""
    return XRPTradingBot(config_file, log_level, enable_metrics)

async def run_bot_async(exchange_client, info_client, wallet_address: str,
                       config_file: Optional[str] = None,
                       log_level: str = "INFO",
                       enable_metrics: bool = True):
    """Run bot asynchronously"""
    bot = XRPTradingBot(config_file, log_level, enable_metrics)
    bot.setup_api_clients(exchange_client, info_client, wallet_address)
    await bot.start()

def run_bot_sync(exchange_client, info_client, wallet_address: str,
                config_file: Optional[str] = None,
                log_level: str = "INFO",
                enable_metrics: bool = True):
    """Run bot synchronously"""
    bot = XRPTradingBot(config_file, log_level, enable_metrics)
    bot.setup_api_clients(exchange_client, info_client, wallet_address)
    bot.run()
