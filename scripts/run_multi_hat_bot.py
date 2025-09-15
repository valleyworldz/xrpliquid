#!/usr/bin/env python3
"""
MULTI-HAT TRADING BOT - COMPLETE INTEGRATION
============================================
Complete integration script that runs all trading hats simultaneously
with comprehensive confirmation and monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_hat_bot import MultiHatTradingBot

def setup_logging():
    """Setup comprehensive logging"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/multi_hat_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger
    logger = logging.getLogger("MultiHatTradingBot")
    logger.info("🚀 Multi-Hat Trading Bot Starting...")
    
    return logger

async def demonstrate_hat_capabilities(bot, logger):
    """Demonstrate all hat capabilities"""
    logger.info("🎩 Demonstrating Multi-Hat Capabilities...")
    
    # 1. Get comprehensive status report
    logger.info("📊 Getting comprehensive status report...")
    status_report = await bot.get_hat_status_report()
    
    logger.info(f"System Overview: {status_report['system_overview']}")
    logger.info(f"Total Hats: {status_report['system_overview']['total_hats']}")
    logger.info(f"Active Hats: {status_report['system_overview']['active_hats']}")
    
    # 2. Execute multiple trading cycles
    logger.info("🎯 Executing trading cycles...")
    
    market_scenarios = [
        {
            "symbol": "XRP",
            "price": 0.65,
            "volume": 1000000,
            "market_conditions": "bullish",
            "volatility": 0.05,
            "timestamp": time.time()
        },
        {
            "symbol": "XRP", 
            "price": 0.63,
            "volume": 1500000,
            "market_conditions": "bearish",
            "volatility": 0.08,
            "timestamp": time.time()
        },
        {
            "symbol": "XRP",
            "price": 0.64,
            "volume": 800000,
            "market_conditions": "sideways",
            "volatility": 0.03,
            "timestamp": time.time()
        }
    ]
    
    for i, scenario in enumerate(market_scenarios, 1):
        logger.info(f"🎯 Trading Cycle {i}: {scenario['market_conditions'].upper()} market")
        
        result = await bot.execute_trading_cycle(scenario)
        
        if result.get("status") == "success":
            decision = result.get("decision")
            if decision:
                logger.info(f"   Decision: {decision.decision_type} from {decision.hat_name}")
                logger.info(f"   Confidence: {decision.confidence:.2f}")
                logger.info(f"   Priority: {decision.priority.name}")
            else:
                logger.info("   No decision made")
        else:
            logger.warning(f"   Trading cycle failed: {result.get('error', 'Unknown error')}")
        
        # Wait between cycles
        await asyncio.sleep(2)
    
    # 3. Continuous monitoring demonstration
    logger.info("🔄 Starting continuous monitoring demonstration...")
    
    for i in range(5):
        # Get confirmation status
        confirmation = await bot.confirm_all_hats_active()
        all_active = all(confirmation.values())
        
        logger.info(f"   Monitoring Check {i+1}: {'ALL HATS ACTIVE' if all_active else 'SOME HATS INACTIVE'}")
        
        # Show individual hat status
        for hat_name, status in confirmation.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"     {status_icon} {hat_name}: {'ACTIVE' if status else 'INACTIVE'}")
        
        await asyncio.sleep(3)
    
    # 4. Performance summary
    logger.info("📈 Performance Summary:")
    system_summary = bot.get_system_summary()
    logger.info(f"   Uptime: {system_summary['uptime_formatted']}")
    logger.info(f"   Total Hats: {system_summary['total_hats']}")
    logger.info(f"   Active Hats: {system_summary['active_hats']}")
    logger.info(f"   System Running: {system_summary['system_running']}")

async def main():
    """Main function"""
    # Setup logging
    logger = setup_logging()
    
    try:
        # Create and start the multi-hat trading bot
        logger.info("🎩 Creating Multi-Hat Trading Bot...")
        bot = MultiHatTradingBot(logger)
        
        # Start the bot
        logger.info("🚀 Starting Multi-Hat Trading Bot...")
        success = await bot.start_bot()
        
        if success:
            logger.info("🎉 Multi-Hat Trading Bot started successfully!")
            
            # Wait for system to stabilize
            logger.info("⏳ Waiting for system to stabilize...")
            await asyncio.sleep(5)
            
            # Demonstrate capabilities
            await demonstrate_hat_capabilities(bot, logger)
            
            # Keep running for extended demonstration
            logger.info("🔄 Running extended demonstration...")
            logger.info("   Press Ctrl+C to stop")
            
            try:
                # Run for 2 minutes with periodic status updates
                start_time = time.time()
                while time.time() - start_time < 120:  # 2 minutes
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                    # Get latest confirmation
                    latest_confirmation = bot.confirmation_system.get_latest_confirmation()
                    if latest_confirmation:
                        overall_status = latest_confirmation["overall_status"]
                        logger.info(f"🔄 Status Check: {overall_status}")
                    
                    # Show system summary
                    summary = bot.get_system_summary()
                    logger.info(f"   Uptime: {summary['uptime_formatted']}, Active Hats: {summary['active_hats']}/{summary['total_hats']}")
                    
            except KeyboardInterrupt:
                logger.info("⏸️ Extended demonstration interrupted by user")
                
        else:
            logger.error("❌ Failed to start Multi-Hat Trading Bot")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏸️ Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        # Shutdown the bot
        if 'bot' in locals():
            logger.info("🛑 Shutting down Multi-Hat Trading Bot...")
            await bot.shutdown()
        
        logger.info("🎉 Multi-Hat Trading Bot shutdown complete")
    
    return 0

if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
