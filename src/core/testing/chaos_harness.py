"""
Chaos Testing Harness
Simulates real-world failure scenarios to test system resilience.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class ChaosScenario:
    """Represents a chaos testing scenario."""
    name: str
    description: str
    failure_type: str
    duration_seconds: int
    severity: str  # low, medium, high, critical
    expected_behavior: str
    recovery_time_seconds: int


class ChaosHarness:
    """Chaos testing harness for system resilience validation."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.incidents_dir = self.reports_dir / "incidents"
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        
        # Chaos scenarios
        self.scenarios = self._create_chaos_scenarios()
        
        # Test results
        self.test_results = []
    
    def _create_chaos_scenarios(self) -> List[ChaosScenario]:
        """Create comprehensive chaos testing scenarios."""
        
        scenarios = [
            # Network failures
            ChaosScenario(
                name="network_packet_loss",
                description="Simulate packet loss in network communication",
                failure_type="network",
                duration_seconds=30,
                severity="medium",
                expected_behavior="System should handle packet loss gracefully with retries",
                recovery_time_seconds=5
            ),
            
            ChaosScenario(
                name="websocket_disconnect",
                description="Simulate WebSocket connection drops",
                failure_type="network",
                duration_seconds=60,
                severity="high",
                expected_behavior="System should reconnect and resync state",
                recovery_time_seconds=10
            ),
            
            ChaosScenario(
                name="api_rate_limit",
                description="Simulate API rate limiting (429 responses)",
                failure_type="api",
                duration_seconds=120,
                severity="medium",
                expected_behavior="System should back off and retry with exponential backoff",
                recovery_time_seconds=15
            ),
            
            ChaosScenario(
                name="api_server_error",
                description="Simulate API server errors (5xx responses)",
                failure_type="api",
                duration_seconds=45,
                severity="high",
                expected_behavior="System should handle errors gracefully and retry",
                recovery_time_seconds=10
            ),
            
            ChaosScenario(
                name="stale_market_data",
                description="Simulate stale market data feeds",
                failure_type="data",
                duration_seconds=90,
                severity="high",
                expected_behavior="System should detect staleness and request fresh data",
                recovery_time_seconds=5
            ),
            
            ChaosScenario(
                name="partial_order_cancels",
                description="Simulate partial order cancellation failures",
                failure_type="execution",
                duration_seconds=60,
                severity="medium",
                expected_behavior="System should handle partial cancels and retry",
                recovery_time_seconds=8
            ),
            
            ChaosScenario(
                name="memory_pressure",
                description="Simulate high memory usage",
                failure_type="system",
                duration_seconds=180,
                severity="medium",
                expected_behavior="System should continue operating under memory pressure",
                recovery_time_seconds=20
            ),
            
            ChaosScenario(
                name="disk_space_full",
                description="Simulate disk space exhaustion",
                failure_type="system",
                duration_seconds=120,
                severity="critical",
                expected_behavior="System should handle disk full gracefully",
                recovery_time_seconds=30
            ),
            
            ChaosScenario(
                name="clock_skew",
                description="Simulate system clock skew",
                failure_type="system",
                duration_seconds=300,
                severity="high",
                expected_behavior="System should detect and handle clock skew",
                recovery_time_seconds=10
            ),
            
            ChaosScenario(
                name="database_connection_pool_exhaustion",
                description="Simulate database connection pool exhaustion",
                failure_type="database",
                duration_seconds=90,
                severity="high",
                expected_behavior="System should handle connection pool exhaustion",
                recovery_time_seconds=15
            )
        ]
        
        return scenarios
    
    async def run_chaos_test(self, 
                           scenario: ChaosScenario,
                           system_under_test: Any) -> Dict[str, Any]:
        """Run a single chaos test scenario."""
        
        test_start = datetime.now(timezone.utc)
        
        print(f"🔥 Starting chaos test: {scenario.name}")
        print(f"   Description: {scenario.description}")
        print(f"   Duration: {scenario.duration_seconds}s")
        print(f"   Severity: {scenario.severity}")
        
        # Record initial system state
        initial_state = await self._capture_system_state(system_under_test)
        
        # Inject failure
        failure_injected = await self._inject_failure(scenario, system_under_test)
        
        if not failure_injected:
            return {
                "scenario": scenario.name,
                "status": "failed",
                "error": "Failed to inject failure",
                "timestamp": test_start.isoformat()
            }
        
        # Monitor system behavior during failure
        behavior_log = []
        start_time = time.time()
        
        while time.time() - start_time < scenario.duration_seconds:
            current_state = await self._capture_system_state(system_under_test)
            behavior_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state": current_state
            })
            await asyncio.sleep(1)  # Monitor every second
        
        # Stop failure injection
        await self._stop_failure_injection(scenario, system_under_test)
        
        # Wait for recovery
        recovery_start = time.time()
        recovered = False
        
        while time.time() - recovery_start < scenario.recovery_time_seconds:
            current_state = await self._capture_system_state(system_under_test)
            if self._is_system_recovered(initial_state, current_state):
                recovered = True
                break
            await asyncio.sleep(1)
        
        # Record final state
        final_state = await self._capture_system_state(system_under_test)
        
        # Analyze results
        test_result = {
            "scenario": scenario.name,
            "status": "passed" if recovered else "failed",
            "timestamp": test_start.isoformat(),
            "duration_seconds": scenario.duration_seconds,
            "recovery_time_seconds": time.time() - recovery_start if recovered else None,
            "recovered": recovered,
            "initial_state": initial_state,
            "final_state": final_state,
            "behavior_log": behavior_log,
            "failure_injected": failure_injected,
            "expected_behavior": scenario.expected_behavior,
            "severity": scenario.severity
        }
        
        # Save test result
        self.test_results.append(test_result)
        
        # Generate incident report if test failed
        if not recovered:
            await self._generate_incident_report(test_result)
        
        print(f"✅ Chaos test completed: {scenario.name} - {'PASSED' if recovered else 'FAILED'}")
        
        return test_result
    
    async def _inject_failure(self, scenario: ChaosScenario, system: Any) -> bool:
        """Inject the specified failure into the system."""
        
        try:
            if scenario.failure_type == "network":
                if scenario.name == "network_packet_loss":
                    # Simulate packet loss by dropping some messages
                    await self._simulate_packet_loss(system, 0.1)  # 10% packet loss
                elif scenario.name == "websocket_disconnect":
                    # Simulate WebSocket disconnection
                    await self._simulate_websocket_disconnect(system)
            
            elif scenario.failure_type == "api":
                if scenario.name == "api_rate_limit":
                    # Simulate rate limiting
                    await self._simulate_rate_limiting(system)
                elif scenario.name == "api_server_error":
                    # Simulate server errors
                    await self._simulate_server_errors(system)
            
            elif scenario.failure_type == "data":
                if scenario.name == "stale_market_data":
                    # Simulate stale data
                    await self._simulate_stale_data(system)
            
            elif scenario.failure_type == "execution":
                if scenario.name == "partial_order_cancels":
                    # Simulate partial cancel failures
                    await self._simulate_partial_cancels(system)
            
            elif scenario.failure_type == "system":
                if scenario.name == "memory_pressure":
                    # Simulate memory pressure
                    await self._simulate_memory_pressure(system)
                elif scenario.name == "disk_space_full":
                    # Simulate disk space full
                    await self._simulate_disk_full(system)
                elif scenario.name == "clock_skew":
                    # Simulate clock skew
                    await self._simulate_clock_skew(system)
            
            elif scenario.failure_type == "database":
                if scenario.name == "database_connection_pool_exhaustion":
                    # Simulate connection pool exhaustion
                    await self._simulate_connection_pool_exhaustion(system)
            
            return True
            
        except Exception as e:
            print(f"Error injecting failure {scenario.name}: {e}")
            return False
    
    async def _stop_failure_injection(self, scenario: ChaosScenario, system: Any):
        """Stop the failure injection."""
        
        # Reset any simulated failures
        if hasattr(system, 'reset_chaos_state'):
            await system.reset_chaos_state()
    
    async def _capture_system_state(self, system: Any) -> Dict[str, Any]:
        """Capture current system state."""
        
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": "unknown"
        }
        
        try:
            # Capture basic system metrics
            if hasattr(system, 'get_health_status'):
                state["system_health"] = await system.get_health_status()
            
            if hasattr(system, 'get_active_orders'):
                state["active_orders"] = await system.get_active_orders()
            
            if hasattr(system, 'get_connection_status'):
                state["connection_status"] = await system.get_connection_status()
            
            if hasattr(system, 'get_performance_metrics'):
                state["performance_metrics"] = await system.get_performance_metrics()
            
        except Exception as e:
            state["error"] = str(e)
        
        return state
    
    def _is_system_recovered(self, initial_state: Dict[str, Any], current_state: Dict[str, Any]) -> bool:
        """Check if system has recovered to initial state."""
        
        # Simple recovery check - can be made more sophisticated
        if initial_state.get("system_health") == current_state.get("system_health"):
            return True
        
        # Check if critical metrics are back to normal
        if "error" in current_state:
            return False
        
        return True
    
    async def _simulate_packet_loss(self, system: Any, loss_rate: float):
        """Simulate packet loss."""
        if hasattr(system, 'set_packet_loss_rate'):
            await system.set_packet_loss_rate(loss_rate)
    
    async def _simulate_websocket_disconnect(self, system: Any):
        """Simulate WebSocket disconnection."""
        if hasattr(system, 'disconnect_websocket'):
            await system.disconnect_websocket()
    
    async def _simulate_rate_limiting(self, system: Any):
        """Simulate API rate limiting."""
        if hasattr(system, 'simulate_rate_limit'):
            await system.simulate_rate_limit()
    
    async def _simulate_server_errors(self, system: Any):
        """Simulate server errors."""
        if hasattr(system, 'simulate_server_errors'):
            await system.simulate_server_errors()
    
    async def _simulate_stale_data(self, system: Any):
        """Simulate stale market data."""
        if hasattr(system, 'simulate_stale_data'):
            await system.simulate_stale_data()
    
    async def _simulate_partial_cancels(self, system: Any):
        """Simulate partial order cancellation failures."""
        if hasattr(system, 'simulate_partial_cancels'):
            await system.simulate_partial_cancels()
    
    async def _simulate_memory_pressure(self, system: Any):
        """Simulate memory pressure."""
        if hasattr(system, 'simulate_memory_pressure'):
            await system.simulate_memory_pressure()
    
    async def _simulate_disk_full(self, system: Any):
        """Simulate disk space full."""
        if hasattr(system, 'simulate_disk_full'):
            await system.simulate_disk_full()
    
    async def _simulate_clock_skew(self, system: Any):
        """Simulate clock skew."""
        if hasattr(system, 'simulate_clock_skew'):
            await system.simulate_clock_skew()
    
    async def _simulate_connection_pool_exhaustion(self, system: Any):
        """Simulate database connection pool exhaustion."""
        if hasattr(system, 'simulate_connection_pool_exhaustion'):
            await system.simulate_connection_pool_exhaustion()
    
    async def _generate_incident_report(self, test_result: Dict[str, Any]):
        """Generate incident report for failed test."""
        
        incident_id = f"chaos_{test_result['scenario']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        incident_report = {
            "incident_id": incident_id,
            "timestamp": test_result["timestamp"],
            "severity": test_result["severity"],
            "scenario": test_result["scenario"],
            "status": "open",
            "description": f"Chaos test failure: {test_result['scenario']}",
            "impact": "System resilience test failed",
            "root_cause": "Chaos testing scenario",
            "resolution": "Investigate system resilience",
            "test_result": test_result
        }
        
        # Save incident report
        incident_file = self.incidents_dir / f"{incident_id}.json"
        with open(incident_file, 'w') as f:
            json.dump(incident_report, f, indent=2)
        
        print(f"📋 Incident report generated: {incident_file}")
    
    async def run_chaos_suite(self, system_under_test: Any) -> Dict[str, Any]:
        """Run complete chaos testing suite."""
        
        print("🔥 Starting Chaos Testing Suite")
        print(f"   Total scenarios: {len(self.scenarios)}")
        
        suite_start = datetime.now(timezone.utc)
        results = []
        
        for scenario in self.scenarios:
            try:
                result = await self.run_chaos_test(scenario, system_under_test)
                results.append(result)
                
                # Wait between tests
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"❌ Chaos test {scenario.name} failed with error: {e}")
                results.append({
                    "scenario": scenario.name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Generate suite summary
        suite_summary = {
            "suite_start": suite_start.isoformat(),
            "suite_end": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": len(self.scenarios),
            "passed": len([r for r in results if r.get("status") == "passed"]),
            "failed": len([r for r in results if r.get("status") == "failed"]),
            "errors": len([r for r in results if r.get("status") == "error"]),
            "results": results
        }
        
        # Save suite results
        suite_file = self.incidents_dir / f"chaos_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(suite_file, 'w') as f:
            json.dump(suite_summary, f, indent=2)
        
        print(f"✅ Chaos Testing Suite completed:")
        print(f"   Passed: {suite_summary['passed']}")
        print(f"   Failed: {suite_summary['failed']}")
        print(f"   Errors: {suite_summary['errors']}")
        
        return suite_summary


def main():
    """Test chaos harness functionality."""
    
    class MockSystem:
        """Mock system for testing."""
        
        async def get_health_status(self):
            return "healthy"
        
        async def get_active_orders(self):
            return []
        
        async def get_connection_status(self):
            return "connected"
        
        async def get_performance_metrics(self):
            return {"latency_ms": 50}
        
        async def reset_chaos_state(self):
            pass
        
        async def set_packet_loss_rate(self, rate):
            pass
        
        async def disconnect_websocket(self):
            pass
    
    async def run_test():
        harness = ChaosHarness()
        mock_system = MockSystem()
        
        # Run a single test
        scenario = harness.scenarios[0]  # network_packet_loss
        result = await harness.run_chaos_test(scenario, mock_system)
        
        print(f"✅ Chaos test result: {result['status']}")
        
        # Run full suite
        suite_result = await harness.run_chaos_suite(mock_system)
        print(f"✅ Chaos suite completed: {suite_result['passed']}/{suite_result['total_scenarios']} passed")
    
    # Run async test
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
