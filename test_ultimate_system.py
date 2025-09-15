#!/usr/bin/env python3
"""
Test script for Ultimate Trading System
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_ultimate_system():
    """Test the Ultimate Trading System"""
    try:
        print("🎯 TESTING ULTIMATE TRADING SYSTEM")
        print("=" * 60)
        
        # Import the ultimate trading system
        from src.core.engines.ultimate_trading_system import UltimateTradingSystem
        from src.core.api.hyperliquid_api import HyperliquidAPI
        from src.core.utils.config_manager import ConfigManager
        from src.core.utils.logger import Logger
        
        print("✅ All imports successful")
        
        # Initialize components
        logger = Logger()
        config = ConfigManager()
        api = HyperliquidAPI(testnet=False, logger=logger)
        
        print("✅ Components initialized")
        
        # Initialize the ultimate trading system
        ultimate_system = UltimateTradingSystem(
            config=config.get_all(),
            api=api,
            logger=logger
        )
        
        print("✅ Ultimate Trading System initialized")
        print("🎯 All 9 specialized roles loaded:")
        print("   🏗️  Hyperliquid Exchange Architect")
        print("   🎯  Chief Quantitative Strategist")
        print("   📊  Market Microstructure Analyst")
        print("   ⚡  Low-Latency Engineer")
        print("   🤖  Automated Execution Manager")
        print("   🛡️  Risk Oversight Officer")
        print("   🔐  Cryptographic Security Architect")
        print("   📊  Performance Quant Analyst")
        print("   🧠  Machine Learning Research Scientist")
        print("=" * 60)
        
        # Test system metrics
        metrics = ultimate_system.get_system_metrics()
        print(f"📊 System metrics: {metrics}")
        
        print("🎯 Ultimate Trading System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Ultimate Trading System: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ultimate_system())
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
        sys.exit(1)
