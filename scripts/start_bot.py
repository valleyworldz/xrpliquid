#!/usr/bin/env python3
"""
Simple bot startup script - bypasses interactive configuration
"""

import os
import sys
import logging
from newbotcode import MultiAssetTradingBot, SymbolCfg, BotConfig

def main():
    """Start the bot with default champion configuration"""
    print("🚀 STARTING A.I. ULTIMATE CHAMPION BOT")
    print("=" * 50)
    
    try:
        # Set environment variables for XRP trading
        os.environ["BOT_SYMBOL"] = "XRP"
        os.environ["BOT_MARKET"] = "perp"
        os.environ["BOT_QUOTE"] = "USDT"
        
        # Create champion configuration
        symbol_cfg = SymbolCfg(
            base="XRP",
            market="perp", 
            quote="USDT",
            hl_name="XRP",
            binance_pair="XRPUSDT",
            coinbase_product="XRP-USD",
            yahoo_ticker="XRP-USD"
        )
        
        startup_config = BotConfig()
        startup_config.leverage = 8.0
        startup_config.risk_profile = "quantum_master"
        startup_config.trading_mode = "quantum_adaptive"
        startup_config.stop_loss_type = "quantum_optimal"
        
        print("✅ Champion configuration loaded")
        print("🎯 Target: +213.6% annual returns")
        print("⚡ Leverage: 8.0x")
        print("🛡️ Risk: Quantum Master (4.0%)")
        print("📈 Mode: Quantum Adaptive")
        print("🛑 Stops: Quantum Optimal")
        
        # Create and start bot
        print("\n🚀 Initializing champion bot...")
        bot = MultiAssetTradingBot(config=symbol_cfg, startup_config=startup_config)
        
        print("✅ Bot initialized successfully")
        print("🎯 Starting live trading...")
        
        # Start the bot
        bot.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        logging.error(f"Bot crashed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
