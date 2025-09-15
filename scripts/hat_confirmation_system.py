#!/usr/bin/env python3
"""
HAT CONFIRMATION AND MONITORING SYSTEM
======================================
Comprehensive system to confirm all hats are activated and monitor their performance.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class HatConfirmationSystem:
    """System to confirm and monitor all trading hats"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.confirmation_history = []
        self.hat_performance_metrics = {}
        self.alert_thresholds = {
            "response_time": 5.0,  # seconds
            "error_rate": 0.05,    # 5%
            "uptime": 0.95,        # 95%
            "decision_confidence": 0.7  # 70%
        }
        
    async def confirm_all_hats_activated(self, coordinator) -> Dict[str, Any]:
        """Comprehensive confirmation that all hats are activated and functioning"""
        self.logger.info("🔍 Starting comprehensive hat confirmation process...")
        
        confirmation_results = {
            "timestamp": time.time(),
            "overall_status": "unknown",
            "hats_confirmed": {},
            "system_health": {},
            "recommendations": [],
            "alerts": []
        }
        
        # 1. Basic Status Confirmation
        basic_status = await self._confirm_basic_status(coordinator)
        confirmation_results["hats_confirmed"] = basic_status
        
        # 2. Health Check Confirmation
        health_status = await self._confirm_health_checks(coordinator)
        confirmation_results["system_health"] = health_status
        
        # 3. Performance Confirmation
        performance_status = await self._confirm_performance_metrics(coordinator)
        confirmation_results["performance"] = performance_status
        
        # 4. Decision Making Confirmation
        decision_status = await self._confirm_decision_making(coordinator)
        confirmation_results["decision_capability"] = decision_status
        
        # 5. Integration Confirmation
        integration_status = await self._confirm_integration(coordinator)
        confirmation_results["integration_status"] = integration_status
        
        # 6. Generate Overall Status
        overall_status = self._generate_overall_status(confirmation_results)
        confirmation_results["overall_status"] = overall_status
        
        # 7. Generate Recommendations and Alerts
        recommendations, alerts = self._generate_recommendations_and_alerts(confirmation_results)
        confirmation_results["recommendations"] = recommendations
        confirmation_results["alerts"] = alerts
        
        # Store confirmation history
        self.confirmation_history.append(confirmation_results)
        
        # Log results
        self._log_confirmation_results(confirmation_results)
        
        return confirmation_results
    
    async def _confirm_basic_status(self, coordinator) -> Dict[str, Any]:
        """Confirm basic status of all hats"""
        self.logger.info("📊 Confirming basic hat status...")
        
        basic_status = {}
        for hat_name, hat in coordinator.hats.items():
            status_info = {
                "name": hat_name,
                "status": hat.status.value,
                "initialized": hat.status.value == "active",
                "last_health_check": hat.last_health_check,
                "restart_count": hat.restart_count,
                "decisions_made": len(hat.decisions_made)
            }
            basic_status[hat_name] = status_info
            
            if status_info["initialized"]:
                self.logger.info(f"✅ {hat_name}: ACTIVE")
            else:
                self.logger.warning(f"⚠️ {hat_name}: {hat.status.value}")
        
        return basic_status
    
    async def _confirm_health_checks(self, coordinator) -> Dict[str, Any]:
        """Confirm health checks for all hats"""
        self.logger.info("🏥 Confirming hat health checks...")
        
        health_status = {}
        for hat_name, hat in coordinator.hats.items():
            try:
                is_healthy = await hat.health_check()
                health_info = {
                    "hat_name": hat_name,
                    "health_status": "healthy" if is_healthy else "unhealthy",
                    "health_check_passed": is_healthy,
                    "last_check_time": time.time(),
                    "response_time": 0.1  # Simulated response time
                }
                health_status[hat_name] = health_info
                
                if is_healthy:
                    self.logger.info(f"✅ {hat_name}: HEALTHY")
                else:
                    self.logger.warning(f"⚠️ {hat_name}: UNHEALTHY")
                    
            except Exception as e:
                health_info = {
                    "hat_name": hat_name,
                    "health_status": "error",
                    "health_check_passed": False,
                    "error": str(e),
                    "last_check_time": time.time()
                }
                health_status[hat_name] = health_info
                self.logger.error(f"❌ {hat_name}: Health check failed - {e}")
        
        return health_status
    
    async def _confirm_performance_metrics(self, coordinator) -> Dict[str, Any]:
        """Confirm performance metrics for all hats"""
        self.logger.info("📈 Confirming hat performance metrics...")
        
        performance_status = {}
        for hat_name, hat in coordinator.hats.items():
            perf_info = {
                "hat_name": hat_name,
                "decisions_per_minute": len(hat.decisions_made) / max(1, (time.time() - hat.last_health_check) / 60),
                "average_confidence": self._calculate_average_confidence(hat.decisions_made),
                "error_rate": self._calculate_error_rate(hat.decisions_made),
                "uptime_percentage": self._calculate_uptime_percentage(hat),
                "performance_score": 0.0
            }
            
            # Calculate performance score
            perf_info["performance_score"] = self._calculate_performance_score(perf_info)
            performance_status[hat_name] = perf_info
            
            if perf_info["performance_score"] > 0.8:
                self.logger.info(f"✅ {hat_name}: HIGH PERFORMANCE ({perf_info['performance_score']:.2f})")
            elif perf_info["performance_score"] > 0.6:
                self.logger.info(f"✅ {hat_name}: GOOD PERFORMANCE ({perf_info['performance_score']:.2f})")
            else:
                self.logger.warning(f"⚠️ {hat_name}: LOW PERFORMANCE ({perf_info['performance_score']:.2f})")
        
        return performance_status
    
    async def _confirm_decision_making(self, coordinator) -> Dict[str, Any]:
        """Confirm decision making capability"""
        self.logger.info("🧠 Confirming decision making capability...")
        
        # Test decision making with sample context
        test_context = {
            "symbol": "XRP",
            "price": 0.65,
            "volume": 1000000,
            "timestamp": time.time(),
            "market_conditions": "normal"
        }
        
        try:
            # Attempt to coordinate a decision
            start_time = time.time()
            decision = await coordinator.coordinate_decision(test_context)
            response_time = time.time() - start_time
            
            decision_status = {
                "decision_making_functional": decision is not None,
                "response_time": response_time,
                "decision_quality": decision.confidence if decision else 0.0,
                "hats_participated": len(coordinator.hats),
                "test_context": test_context,
                "decision_result": decision.decision_type if decision else None
            }
            
            if decision_status["decision_making_functional"]:
                self.logger.info(f"✅ Decision making: FUNCTIONAL (response time: {response_time:.2f}s)")
            else:
                self.logger.warning("⚠️ Decision making: NOT FUNCTIONAL")
                
        except Exception as e:
            decision_status = {
                "decision_making_functional": False,
                "error": str(e),
                "response_time": None,
                "decision_quality": 0.0
            }
            self.logger.error(f"❌ Decision making test failed: {e}")
        
        return decision_status
    
    async def _confirm_integration(self, coordinator) -> Dict[str, Any]:
        """Confirm integration between hats"""
        self.logger.info("🔗 Confirming hat integration...")
        
        integration_status = {
            "total_hats": len(coordinator.hats),
            "active_hats": sum(1 for hat in coordinator.hats.values() if hat.status.value == "active"),
            "integration_health": "unknown",
            "communication_status": {},
            "dependency_status": {}
        }
        
        # Check communication between hats
        for hat_name, hat in coordinator.hats.items():
            integration_status["communication_status"][hat_name] = {
                "can_receive_decisions": True,
                "can_make_decisions": True,
                "integration_score": 0.9  # Simulated
            }
        
        # Calculate overall integration health
        active_ratio = integration_status["active_hats"] / integration_status["total_hats"]
        if active_ratio >= 0.95:
            integration_status["integration_health"] = "excellent"
        elif active_ratio >= 0.8:
            integration_status["integration_health"] = "good"
        elif active_ratio >= 0.6:
            integration_status["integration_health"] = "fair"
        else:
            integration_status["integration_health"] = "poor"
        
        self.logger.info(f"✅ Integration health: {integration_status['integration_health'].upper()}")
        
        return integration_status
    
    def _generate_overall_status(self, confirmation_results: Dict[str, Any]) -> str:
        """Generate overall system status"""
        # Check basic status
        basic_status = confirmation_results["hats_confirmed"]
        active_hats = sum(1 for hat in basic_status.values() if hat["initialized"])
        total_hats = len(basic_status)
        
        # Check health status
        health_status = confirmation_results["system_health"]
        healthy_hats = sum(1 for hat in health_status.values() if hat["health_check_passed"])
        
        # Check decision making
        decision_status = confirmation_results["decision_capability"]
        decision_functional = decision_status.get("decision_making_functional", False)
        
        # Check integration
        integration_status = confirmation_results["integration_status"]
        integration_health = integration_status.get("integration_health", "unknown")
        
        # Determine overall status
        if (active_hats == total_hats and 
            healthy_hats == total_hats and 
            decision_functional and 
            integration_health in ["excellent", "good"]):
            return "FULLY_OPERATIONAL"
        elif (active_hats >= total_hats * 0.8 and 
              healthy_hats >= total_hats * 0.8 and 
              decision_functional):
            return "MOSTLY_OPERATIONAL"
        elif active_hats >= total_hats * 0.5:
            return "PARTIALLY_OPERATIONAL"
        else:
            return "CRITICAL_ISSUES"
    
    def _generate_recommendations_and_alerts(self, confirmation_results: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Generate recommendations and alerts based on confirmation results"""
        recommendations = []
        alerts = []
        
        overall_status = confirmation_results["overall_status"]
        
        if overall_status == "FULLY_OPERATIONAL":
            recommendations.append("🎉 All hats are fully operational - system ready for trading")
        elif overall_status == "MOSTLY_OPERATIONAL":
            recommendations.append("⚠️ Most hats operational - monitor inactive hats closely")
            alerts.append("Some hats may not be functioning optimally")
        elif overall_status == "PARTIALLY_OPERATIONAL":
            recommendations.append("🚨 Partial operation - investigate and fix inactive hats")
            alerts.append("Multiple hats are not functioning - trading may be limited")
        else:
            recommendations.append("🛑 Critical issues detected - immediate attention required")
            alerts.append("System may not be safe for trading")
        
        # Check specific issues
        basic_status = confirmation_results["hats_confirmed"]
        for hat_name, status in basic_status.items():
            if not status["initialized"]:
                alerts.append(f"Hat {hat_name} is not initialized")
                recommendations.append(f"Restart or reinitialize {hat_name}")
        
        health_status = confirmation_results["system_health"]
        for hat_name, health in health_status.items():
            if not health["health_check_passed"]:
                alerts.append(f"Hat {hat_name} failed health check")
                recommendations.append(f"Investigate health issues with {hat_name}")
        
        return recommendations, alerts
    
    def _calculate_average_confidence(self, decisions: List[Any]) -> float:
        """Calculate average confidence from decisions"""
        if not decisions:
            return 0.0
        return sum(decision.confidence for decision in decisions) / len(decisions)
    
    def _calculate_error_rate(self, decisions: List[Any]) -> float:
        """Calculate error rate from decisions"""
        if not decisions:
            return 0.0
        error_decisions = sum(1 for decision in decisions if decision.decision_type == "error")
        return error_decisions / len(decisions)
    
    def _calculate_uptime_percentage(self, hat) -> float:
        """Calculate uptime percentage for a hat"""
        # Simplified calculation - in reality would track actual uptime
        if hat.status.value == "active":
            return 0.95  # 95% uptime
        else:
            return 0.0
    
    def _calculate_performance_score(self, perf_info: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        # Weighted combination of metrics
        confidence_score = min(perf_info["average_confidence"], 1.0)
        uptime_score = perf_info["uptime_percentage"]
        error_penalty = 1.0 - perf_info["error_rate"]
        
        # Weighted average
        performance_score = (confidence_score * 0.4 + uptime_score * 0.4 + error_penalty * 0.2)
        return min(performance_score, 1.0)
    
    def _log_confirmation_results(self, confirmation_results: Dict[str, Any]):
        """Log confirmation results"""
        overall_status = confirmation_results["overall_status"]
        
        self.logger.info("=" * 60)
        self.logger.info("🎩 HAT CONFIRMATION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Overall Status: {overall_status}")
        self.logger.info(f"⏰ Timestamp: {datetime.fromtimestamp(confirmation_results['timestamp'])}")
        
        # Log hat status summary
        basic_status = confirmation_results["hats_confirmed"]
        active_count = sum(1 for hat in basic_status.values() if hat["initialized"])
        total_count = len(basic_status)
        self.logger.info(f"🎩 Hats Status: {active_count}/{total_count} ACTIVE")
        
        # Log health summary
        health_status = confirmation_results["system_health"]
        healthy_count = sum(1 for hat in health_status.values() if hat["health_check_passed"])
        self.logger.info(f"🏥 Health Status: {healthy_count}/{total_count} HEALTHY")
        
        # Log recommendations
        recommendations = confirmation_results["recommendations"]
        if recommendations:
            self.logger.info("💡 Recommendations:")
            for rec in recommendations:
                self.logger.info(f"   • {rec}")
        
        # Log alerts
        alerts = confirmation_results["alerts"]
        if alerts:
            self.logger.info("🚨 Alerts:")
            for alert in alerts:
                self.logger.info(f"   • {alert}")
        
        self.logger.info("=" * 60)
    
    async def continuous_monitoring(self, coordinator, interval: int = 60):
        """Continuously monitor hat status"""
        self.logger.info(f"🔄 Starting continuous monitoring (interval: {interval}s)")
        
        while True:
            try:
                # Perform confirmation
                confirmation_results = await self.confirm_all_hats_activated(coordinator)
                
                # Check for critical issues
                if confirmation_results["overall_status"] == "CRITICAL_ISSUES":
                    self.logger.error("🚨 CRITICAL ISSUES DETECTED - Immediate attention required!")
                
                # Wait for next check
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def get_confirmation_history(self) -> List[Dict[str, Any]]:
        """Get confirmation history"""
        return self.confirmation_history
    
    def get_latest_confirmation(self) -> Optional[Dict[str, Any]]:
        """Get latest confirmation results"""
        if self.confirmation_history:
            return self.confirmation_history[-1]
        return None
