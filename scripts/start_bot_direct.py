#!/usr/bin/env python3
"""
Direct bot startup script - bypasses interactive configuration
"""

import os
import sys
import logging

# Handle sympy import gracefully
try:
    import sympy as sp
except ImportError:
    print("⚠️ SymPy not available, using fallback calculations")
    sp = None

from newbotcode import MultiAssetTradingBot, SymbolCfg, BotConfig

def main():
    """Start the bot with champion configuration directly"""
    print("🚀 STARTING A.I. ULTIMATE CHAMPION BOT (DIRECT)")
    print("=" * 60)
    
    try:
        # Set environment variables for XRP trading
        os.environ["BOT_SYMBOL"] = "XRP"
        os.environ["BOT_MARKET"] = "perp"
        os.environ["BOT_QUOTE"] = "USDT"
        os.environ["BOT_OPTIMIZE"] = "false"  # Disable optimization to avoid dual startup
        
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
        print("🚀 Guardian: Quantum-Adaptive (Enhanced)")
        
        # Create and start bot
        print("\n🚀 Initializing champion bot...")
        bot = MultiAssetTradingBot(config=symbol_cfg, startup_config=startup_config)
        
        print("✅ Bot initialized successfully")
        print("🎯 Starting live trading with quantum guardian...")
        
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
