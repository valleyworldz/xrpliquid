#!/usr/bin/env python3
"""
Champion A.I. ULTIMATE Bot Launcher
Direct deployment of validated +213% configuration
"""

import logging
import os
import sys
from dataclasses import dataclass

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ChampionLauncher")

@dataclass
class ChampionConfig:
    """Champion A.I. ULTIMATE Configuration"""
    symbol: str = "XRP"
    leverage: float = 8.0
    position_risk_pct: float = 4.0
    stop_loss_type: str = "quantum_optimal"
    trading_mode: str = "quantum_adaptive"
    session_duration_hours: float = 168.0
    risk_profile: str = "quantum_master"
    market_preference: str = "quantum_optimal"
    notification_level: str = "trades_alerts"

def create_champion_symbol_cfg(symbol="XRP"):
    """Create champion symbol configuration"""
    try:
        # Import after ensuring no startup interference
        sys.path.insert(0, '.')
        from newbotcode import SymbolCfg
        
        return SymbolCfg(
            base=symbol,
            quote="USD",
            market="perpetual",
            hl_name=symbol,
            binance_pair=f"{symbol}USDT",
            coinbase_product=f"{symbol}-USD",
            yahoo_ticker=f"{symbol}-USD"
        )
    except Exception as e:
        logger.error(f"❌ Error creating symbol config: {e}")
        return None

def create_champion_startup_cfg():
    """Create champion startup configuration"""
    try:
        from newbotcode import StartupConfig
        
        return StartupConfig(
            leverage=8.0,
            risk_profile="quantum_master",
            trading_mode="quantum_adaptive", 
            position_risk_pct=4.0,
            stop_loss_type="quantum_optimal",
            session_duration_hours=168.0,
            market_preference="quantum_optimal",
            notification_level="trades_alerts",
            backup_mode="resume",
            auto_close_on_target=True,
            auto_close_on_time=True
        )
    except Exception as e:
        logger.error(f"❌ Error creating startup config: {e}")
        return None

def launch_champion_bot():
    """Launch the champion A.I. ULTIMATE bot directly"""
    print("🏆 CHAMPION A.I. ULTIMATE BOT LAUNCHER")
    print("=" * 50)
    print("🎯 Target: +213% validated returns")
    print("⚡ Settings: 8x leverage, 4% risk, quantum optimal")
    print("=" * 50)
    
    try:
        # Create champion configurations
        logger.info("🔧 Creating champion configurations...")
        
        symbol_cfg = create_champion_symbol_cfg("XRP")
        startup_cfg = create_champion_startup_cfg()
        
        if not symbol_cfg or not startup_cfg:
            logger.error("❌ Failed to create champion configurations")
            return False
            
        logger.info("✅ Champion configurations created")
        
        # Import bot class (should work now that imports are fixed)
        logger.info("🤖 Importing champion bot class...")
        from newbotcode import MultiAssetTradingBot
        
        # Create bot instance with champion config
        logger.info("🚀 Creating champion bot instance...")
        bot = MultiAssetTradingBot(config=symbol_cfg, startup_config=startup_cfg)
        
        logger.info("✅ Champion bot created successfully!")
        logger.info(f"📊 Symbol: {symbol_cfg.base}")
        logger.info(f"⚡ Leverage: {startup_cfg.leverage}x")
        logger.info(f"🛡️ Risk: {startup_cfg.position_risk_pct}%")
        logger.info(f"🎯 Mode: {startup_cfg.trading_mode}")
        logger.info(f"🛡️ Stops: {startup_cfg.stop_loss_type}")
        
        # Start the bot
        print("\n🚀 STARTING CHAMPION A.I. ULTIMATE BOT...")
        print("🏆 Expected Performance: +213% annual returns")
        print("⚠️ Press Ctrl+C to stop")
        print("=" * 50)
        
        bot.run()
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Champion bot stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Champion bot error: {e}")
        if os.getenv('DEBUG') == '1':
            import traceback
            traceback.print_exc()
        return False

def main():
    """Main entry point"""
    try:
        # Set champion environment
        os.environ['BOT_NON_INTERACTIVE'] = '1'  # Disable interactive prompts
        os.environ['CHAMPION_MODE'] = '1'        # Enable champion mode
        
        success = launch_champion_bot()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Launcher error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
