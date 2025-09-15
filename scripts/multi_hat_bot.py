#!/usr/bin/env python3
"""
MULTI-HAT TRADING BOT INTEGRATION
==================================
Main integration file that activates all trading hats simultaneously
with comprehensive confirmation and monitoring systems.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any

# Import hat architecture
from hat_architecture import HatCoordinator, HatConfig, DecisionPriority
from hat_confirmation_system import HatConfirmationSystem

# Import all hat implementations
from strategy_hats import (
    ChiefQuantitativeStrategist,
    MarketMicrostructureAnalyst, 
    MacroCryptoEconomist
)

from technical_hats import (
    SmartContractEngineer,
    LowLatencyEngineer,
    APIIntegrationSpecialist
)

from operational_hats import (
    HFTOperator,
    AutomatedExecutionManager,
    RiskOversightOfficer
)

class MultiHatTradingBot:
    """Main trading bot that coordinates all specialized hats"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.coordinator = HatCoordinator(logger)
        self.confirmation_system = HatConfirmationSystem(logger)
        self.hats_initialized = False
        self.system_running = False
        self.startup_time = None
        
        # Initialize all hats
        self._initialize_all_hats()
    
    def _initialize_all_hats(self):
        """Initialize all trading hats with proper configurations"""
        self.logger.info("🎩 Initializing Multi-Hat Trading Bot...")
        
        # Strategy and Research Hats
        quant_strategist = ChiefQuantitativeStrategist(self.logger)
        quant_config = HatConfig(
            name="ChiefQuantitativeStrategist",
            priority=DecisionPriority.MEDIUM,
            dependencies=[]
        )
        self.coordinator.register_hat(quant_strategist, quant_config)
        
        microstructure_analyst = MarketMicrostructureAnalyst(self.logger)
        microstructure_config = HatConfig(
            name="MarketMicrostructureAnalyst", 
            priority=DecisionPriority.HIGH,
            dependencies=[]
        )
        self.coordinator.register_hat(microstructure_analyst, microstructure_config)
        
        macro_economist = MacroCryptoEconomist(self.logger)
        macro_config = HatConfig(
            name="MacroCryptoEconomist",
            priority=DecisionPriority.MEDIUM,
            dependencies=[]
        )
        self.coordinator.register_hat(macro_economist, macro_config)
        
        # Technical Development Hats
        smart_contract_engineer = SmartContractEngineer(self.logger)
        smart_contract_config = HatConfig(
            name="SmartContractEngineer",
            priority=DecisionPriority.CRITICAL,
            dependencies=[]
        )
        self.coordinator.register_hat(smart_contract_engineer, smart_contract_config)
        
        low_latency_engineer = LowLatencyEngineer(self.logger)
        low_latency_config = HatConfig(
            name="LowLatencyEngineer",
            priority=DecisionPriority.HIGH,
            dependencies=[]
        )
        self.coordinator.register_hat(low_latency_engineer, low_latency_config)
        
        api_integration_specialist = APIIntegrationSpecialist(self.logger)
        api_config = HatConfig(
            name="APIIntegrationSpecialist",
            priority=DecisionPriority.HIGH,
            dependencies=[]
        )
        self.coordinator.register_hat(api_integration_specialist, api_config)
        
        # Operational and Execution Hats
        hft_operator = HFTOperator(self.logger)
        hft_config = HatConfig(
            name="HFTOperator",
            priority=DecisionPriority.HIGH,
            dependencies=[]
        )
        self.coordinator.register_hat(hft_operator, hft_config)
        
        execution_manager = AutomatedExecutionManager(self.logger)
        execution_config = HatConfig(
            name="AutomatedExecutionManager",
            priority=DecisionPriority.HIGH,
            dependencies=[]
        )
        self.coordinator.register_hat(execution_manager, execution_config)
        
        risk_officer = RiskOversightOfficer(self.logger)
        risk_config = HatConfig(
            name="RiskOversightOfficer",
            priority=DecisionPriority.CRITICAL,
            dependencies=[]
        )
        self.coordinator.register_hat(risk_officer, risk_config)
        
        self.logger.info(f"🎩 Registered {len(self.coordinator.hats)} trading hats")
    
    async def start_bot(self) -> bool:
        """Start the multi-hat trading bot"""
        try:
            self.logger.info("🚀 Starting Multi-Hat Trading Bot...")
            self.startup_time = time.time()
            
            # Initialize all hats
            self.logger.info("🔧 Initializing all trading hats...")
            initialization_success = await self.coordinator.initialize_all_hats()
            
            if not initialization_success:
                self.logger.error("❌ Failed to initialize all hats")
                return False
            
            # Comprehensive confirmation using confirmation system
            self.logger.info("✅ Performing comprehensive hat confirmation...")
            confirmation_results = await self.confirmation_system.confirm_all_hats_activated(self.coordinator)
            
            if confirmation_results["overall_status"] not in ["FULLY_OPERATIONAL", "MOSTLY_OPERATIONAL"]:
                self.logger.error(f"❌ Hat confirmation failed: {confirmation_results['overall_status']}")
                return False
            
            self.hats_initialized = True
            self.logger.info("🎉 ALL HATS CONFIRMED ACTIVE AND FUNCTIONING!")
            
            # Start coordination loop
            self.system_running = True
            await self.coordinator.start_coordination_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start multi-hat trading bot: {e}")
            return False
    
    async def confirm_all_hats_active(self) -> Dict[str, bool]:
        """Confirm all hats are active and functioning"""
        if not self.hats_initialized:
            self.logger.warning("⚠️ Hats not yet initialized")
            return {}
        
        return await self.coordinator.confirm_all_hats_active()
    
    async def get_hat_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report for all hats"""
        if not self.hats_initialized:
            return {"error": "Hats not initialized"}
        
        system_status = self.coordinator.get_system_status()
        
        # Add detailed hat information
        detailed_status = {
            "system_overview": system_status,
            "hat_details": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        # Get detailed status for each hat
        for hat_name, hat in self.coordinator.hats.items():
            detailed_status["hat_details"][hat_name] = {
                "status": hat.get_status(),
                "decisions_made": len(hat.decisions_made),
                "performance_metrics": hat.performance_metrics,
                "last_health_check": hat.last_health_check
            }
        
        # Generate recommendations
        detailed_status["recommendations"] = self._generate_recommendations(system_status)
        
        return detailed_status
    
    def _generate_recommendations(self, system_status: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on system status"""
        recommendations = []
        
        active_hats = system_status.get("active_hats", 0)
        total_hats = system_status.get("total_hats", 0)
        
        if active_hats == total_hats:
            recommendations.append("✅ All hats are active and functioning optimally")
        else:
            recommendations.append(f"⚠️ {total_hats - active_hats} hats are not active")
        
        total_decisions = system_status.get("total_decisions", 0)
        if total_decisions > 100:
            recommendations.append("📊 High decision volume - system is actively trading")
        elif total_decisions > 10:
            recommendations.append("📈 Moderate decision volume - system is operational")
        else:
            recommendations.append("⏳ Low decision volume - system may be in startup phase")
        
        return recommendations
    
    async def execute_trading_cycle(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete trading cycle with all hats"""
        if not self.hats_initialized:
            return {"error": "Hats not initialized"}
        
        try:
            # Coordinate decision from all hats
            decision = await self.coordinator.coordinate_decision(market_context)
            
            if decision:
                return {
                    "decision": decision,
                    "timestamp": time.time(),
                    "status": "success",
                    "hats_participated": len(self.coordinator.hats)
                }
            else:
                return {
                    "decision": None,
                    "timestamp": time.time(),
                    "status": "no_decision",
                    "hats_participated": 0
                }
                
        except Exception as e:
            self.logger.error(f"❌ Trading cycle execution error: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "status": "error"
            }
    
    async def shutdown(self):
        """Gracefully shutdown the multi-hat trading bot"""
        self.logger.info("🛑 Shutting down Multi-Hat Trading Bot...")
        self.system_running = False
        
        if self.hats_initialized:
            await self.coordinator.shutdown()
        
        self.logger.info("🎉 Multi-Hat Trading Bot shutdown complete")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system summary information"""
        uptime = time.time() - self.startup_time if self.startup_time else 0
        
        return {
            "system_running": self.system_running,
            "hats_initialized": self.hats_initialized,
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "total_hats": len(self.coordinator.hats),
            "active_hats": sum(1 for hat in self.coordinator.hats.values() if hat.status.value == "active"),
            "startup_time": self.startup_time
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

async def main():
    """Main function to demonstrate multi-hat trading bot"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("MultiHatTradingBot")
    
    # Create and start the bot
    bot = MultiHatTradingBot(logger)
    
    try:
        # Start the bot
        success = await bot.start_bot()
        
        if success:
            logger.info("🎉 Multi-Hat Trading Bot started successfully!")
            
            # Wait a bit for the system to stabilize
            await asyncio.sleep(5)
            
            # Get status report
            status_report = await bot.get_hat_status_report()
            logger.info(f"📊 System Status: {status_report['system_overview']}")
            
            # Execute a sample trading cycle
            market_context = {
                "symbol": "XRP",
                "price": 0.65,
                "volume": 1000000,
                "timestamp": time.time()
            }
            
            trading_result = await bot.execute_trading_cycle(market_context)
            logger.info(f"🎯 Trading Cycle Result: {trading_result}")
            
            # Keep running for demonstration
            logger.info("🔄 Bot running... Press Ctrl+C to stop")
            while bot.system_running:
                await asyncio.sleep(10)
                
                # Periodic status check
                confirmation = await bot.confirm_all_hats_active()
                all_active = all(confirmation.values())
                logger.info(f"🎩 Hat Status: {'ALL ACTIVE' if all_active else 'SOME INACTIVE'}")
                
        else:
            logger.error("❌ Failed to start Multi-Hat Trading Bot")
            
    except KeyboardInterrupt:
        logger.info("⏸️ Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
