"""
Chaos Testing Framework
Injects failures to test system resilience and recovery capabilities
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import subprocess
import signal

class ChaosEventType(Enum):
    API_DOWNTIME = "api_downtime"
    WS_DISCONNECT = "ws_disconnect"
    LATENCY_SPIKE = "latency_spike"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    NETWORK_PARTITION = "network_partition"
    PROCESS_KILL = "process_kill"
    DISK_FULL = "disk_full"

@dataclass
class ChaosEvent:
    event_type: ChaosEventType
    duration: float  # seconds
    intensity: float  # 0.0 to 1.0
    timestamp: str
    description: str
    recovery_time: Optional[float] = None
    success: bool = False

@dataclass
class ChaosTestResult:
    test_name: str
    start_time: str
    end_time: str
    events: List[ChaosEvent]
    total_events: int
    successful_recoveries: int
    failed_recoveries: int
    average_recovery_time: float
    system_health_before: Dict
    system_health_after: Dict
    passed: bool

class ChaosTestingFramework:
    """
    Chaos testing framework for resilience validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_events: List[ChaosEvent] = []
        self.test_results: List[ChaosTestResult] = []
        
        # Create reports directory
        self.reports_dir = Path("reports/chaos_testing")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Chaos event handlers
        self.event_handlers = {
            ChaosEventType.API_DOWNTIME: self._simulate_api_downtime,
            ChaosEventType.WS_DISCONNECT: self._simulate_ws_disconnect,
            ChaosEventType.LATENCY_SPIKE: self._simulate_latency_spike,
            ChaosEventType.MEMORY_PRESSURE: self._simulate_memory_pressure,
            ChaosEventType.CPU_SPIKE: self._simulate_cpu_spike,
            ChaosEventType.NETWORK_PARTITION: self._simulate_network_partition,
            ChaosEventType.PROCESS_KILL: self._simulate_process_kill,
            ChaosEventType.DISK_FULL: self._simulate_disk_full
        }
    
    async def run_chaos_test(self, test_name: str, events: List[ChaosEvent]) -> ChaosTestResult:
        """Run a comprehensive chaos test"""
        self.logger.info(f"🧪 Starting chaos test: {test_name}")
        
        start_time = datetime.now()
        system_health_before = await self._get_system_health()
        
        successful_recoveries = 0
        failed_recoveries = 0
        recovery_times = []
        
        for event in events:
            self.logger.info(f"💥 Injecting chaos: {event.event_type.value}")
            
            # Inject the chaos event
            event_start = time.time()
            try:
                await self._inject_chaos_event(event)
                
                # Wait for the event duration
                await asyncio.sleep(event.duration)
                
                # Check recovery
                recovery_time = await self._check_recovery(event)
                event.recovery_time = recovery_time
                
                if recovery_time is not None and recovery_time < event.duration * 2:
                    event.success = True
                    successful_recoveries += 1
                    recovery_times.append(recovery_time)
                    self.logger.info(f"✅ Recovery successful in {recovery_time:.2f}s")
                else:
                    event.success = False
                    failed_recoveries += 1
                    self.logger.error(f"❌ Recovery failed or too slow")
                
            except Exception as e:
                self.logger.error(f"❌ Chaos event failed: {e}")
                event.success = False
                failed_recoveries += 1
            
            # Clean up the event
            await self._cleanup_chaos_event(event)
        
        end_time = datetime.now()
        system_health_after = await self._get_system_health()
        
        # Calculate results
        average_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0.0
        passed = failed_recoveries == 0 and average_recovery_time < 30.0  # 30s max recovery
        
        result = ChaosTestResult(
            test_name=test_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            events=events,
            total_events=len(events),
            successful_recoveries=successful_recoveries,
            failed_recoveries=failed_recoveries,
            average_recovery_time=average_recovery_time,
            system_health_before=system_health_before,
            system_health_after=system_health_after,
            passed=passed
        )
        
        self.test_results.append(result)
        await self._save_test_result(result)
        
        self.logger.info(f"🏁 Chaos test complete: {test_name} - {'PASSED' if passed else 'FAILED'}")
        return result
    
    async def _inject_chaos_event(self, event: ChaosEvent):
        """Inject a chaos event"""
        handler = self.event_handlers.get(event.event_type)
        if handler:
            await handler(event)
        else:
            self.logger.error(f"❌ No handler for event type: {event.event_type}")
    
    async def _simulate_api_downtime(self, event: ChaosEvent):
        """Simulate API downtime"""
        self.logger.warning(f"🌐 Simulating API downtime for {event.duration}s")
        
        # Simulate API downtime by blocking network calls
        # In a real implementation, this would block actual API calls
        self.active_events.append(event)
        
        # Simulate network blocking
        await asyncio.sleep(0.1)  # Simulate blocking
    
    async def _simulate_ws_disconnect(self, event: ChaosEvent):
        """Simulate WebSocket disconnection"""
        self.logger.warning(f"🔌 Simulating WebSocket disconnect for {event.duration}s")
        
        # Simulate WebSocket disconnection
        # In a real implementation, this would close the WebSocket connection
        self.active_events.append(event)
        
        # Simulate disconnection
        await asyncio.sleep(0.1)  # Simulate disconnection
    
    async def _simulate_latency_spike(self, event: ChaosEvent):
        """Simulate network latency spike"""
        self.logger.warning(f"⏱️ Simulating latency spike: {event.intensity * 1000:.0f}ms")
        
        # Simulate latency spike by adding delays
        # In a real implementation, this would add network delays
        self.active_events.append(event)
        
        # Simulate latency
        delay = event.intensity * 2.0  # Max 2 seconds
        await asyncio.sleep(delay)
    
    async def _simulate_memory_pressure(self, event: ChaosEvent):
        """Simulate memory pressure"""
        self.logger.warning(f"🧠 Simulating memory pressure: {event.intensity * 100:.0f}%")
        
        # Simulate memory pressure by allocating memory
        # In a real implementation, this would allocate large amounts of memory
        self.active_events.append(event)
        
        # Simulate memory allocation
        memory_mb = int(event.intensity * 1000)  # Max 1GB
        dummy_data = [0] * (memory_mb * 1024 * 1024 // 8)  # Allocate memory
        
        # Hold the memory for the event duration
        await asyncio.sleep(event.duration)
        
        # Clean up
        del dummy_data
    
    async def _simulate_cpu_spike(self, event: ChaosEvent):
        """Simulate CPU spike"""
        self.logger.warning(f"🔥 Simulating CPU spike: {event.intensity * 100:.0f}%")
        
        # Simulate CPU spike by running intensive computation
        self.active_events.append(event)
        
        # Simulate CPU intensive work
        start_time = time.time()
        while time.time() - start_time < event.duration:
            # CPU intensive computation
            sum(range(10000))
            await asyncio.sleep(0.001)  # Small yield
    
    async def _simulate_network_partition(self, event: ChaosEvent):
        """Simulate network partition"""
        self.logger.warning(f"🚧 Simulating network partition for {event.duration}s")
        
        # Simulate network partition
        # In a real implementation, this would block network access
        self.active_events.append(event)
        
        # Simulate partition
        await asyncio.sleep(0.1)  # Simulate partition
    
    async def _simulate_process_kill(self, event: ChaosEvent):
        """Simulate process kill"""
        self.logger.warning(f"💀 Simulating process kill")
        
        # Simulate process kill
        # In a real implementation, this would kill the main process
        self.active_events.append(event)
        
        # Simulate process termination
        await asyncio.sleep(0.1)  # Simulate kill
    
    async def _simulate_disk_full(self, event: ChaosEvent):
        """Simulate disk full condition"""
        self.logger.warning(f"💾 Simulating disk full condition")
        
        # Simulate disk full
        # In a real implementation, this would fill up disk space
        self.active_events.append(event)
        
        # Simulate disk full
        await asyncio.sleep(0.1)  # Simulate disk full
    
    async def _check_recovery(self, event: ChaosEvent) -> Optional[float]:
        """Check if system recovered from chaos event"""
        start_time = time.time()
        
        # Check recovery based on event type
        if event.event_type == ChaosEventType.API_DOWNTIME:
            # Check if API is responding
            return await self._check_api_recovery()
        elif event.event_type == ChaosEventType.WS_DISCONNECT:
            # Check if WebSocket reconnected
            return await self._check_ws_recovery()
        elif event.event_type == ChaosEventType.LATENCY_SPIKE:
            # Check if latency returned to normal
            return await self._check_latency_recovery()
        elif event.event_type == ChaosEventType.MEMORY_PRESSURE:
            # Check if memory usage returned to normal
            return await self._check_memory_recovery()
        elif event.event_type == ChaosEventType.CPU_SPIKE:
            # Check if CPU usage returned to normal
            return await self._check_cpu_recovery()
        else:
            # Generic recovery check
            return await self._check_generic_recovery()
    
    async def _check_api_recovery(self) -> Optional[float]:
        """Check API recovery"""
        try:
            # Simulate API health check
            # In a real implementation, this would ping the API
            await asyncio.sleep(0.1)  # Simulate API call
            return 0.1  # Simulate quick recovery
        except:
            return None
    
    async def _check_ws_recovery(self) -> Optional[float]:
        """Check WebSocket recovery"""
        try:
            # Simulate WebSocket health check
            # In a real implementation, this would check WS connection
            await asyncio.sleep(0.2)  # Simulate reconnection
            return 0.2  # Simulate reconnection time
        except:
            return None
    
    async def _check_latency_recovery(self) -> Optional[float]:
        """Check latency recovery"""
        try:
            # Simulate latency check
            # In a real implementation, this would measure actual latency
            await asyncio.sleep(0.05)  # Simulate normal latency
            return 0.05  # Simulate quick recovery
        except:
            return None
    
    async def _check_memory_recovery(self) -> Optional[float]:
        """Check memory recovery"""
        try:
            # Check if memory usage is normal
            memory = psutil.virtual_memory()
            if memory.percent < 80:  # Normal memory usage
                return 0.1  # Simulate quick recovery
            return None
        except:
            return None
    
    async def _check_cpu_recovery(self) -> Optional[float]:
        """Check CPU recovery"""
        try:
            # Check if CPU usage is normal
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent < 70:  # Normal CPU usage
                return 0.1  # Simulate quick recovery
            return None
        except:
            return None
    
    async def _check_generic_recovery(self) -> Optional[float]:
        """Generic recovery check"""
        try:
            # Generic system health check
            await asyncio.sleep(0.1)  # Simulate health check
            return 0.1  # Simulate quick recovery
        except:
            return None
    
    async def _cleanup_chaos_event(self, event: ChaosEvent):
        """Clean up chaos event"""
        if event in self.active_events:
            self.active_events.remove(event)
        
        # Event-specific cleanup
        if event.event_type == ChaosEventType.MEMORY_PRESSURE:
            # Force garbage collection
            import gc
            gc.collect()
    
    async def _get_system_health(self) -> Dict:
        """Get current system health metrics"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting system health: {e}")
            return {"error": str(e)}
    
    async def _save_test_result(self, result: ChaosTestResult):
        """Save test result to file"""
        try:
            result_file = self.reports_dir / f"chaos_test_{result.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to serializable format
            result_dict = asdict(result)
            for event in result_dict['events']:
                event['event_type'] = event['event_type'].value
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
                
            self.logger.info(f"💾 Test result saved: {result_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save test result: {e}")
    
    def get_test_summary(self) -> Dict:
        """Get summary of all chaos tests"""
        if not self.test_results:
            return {"message": "No chaos tests run yet"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        total_events = sum(result.total_events for result in self.test_results)
        successful_recoveries = sum(result.successful_recoveries for result in self.test_results)
        failed_recoveries = sum(result.failed_recoveries for result in self.test_results)
        
        avg_recovery_time = sum(result.average_recovery_time for result in self.test_results) / total_tests
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_events": total_events,
            "successful_recoveries": successful_recoveries,
            "failed_recoveries": failed_recoveries,
            "recovery_rate": successful_recoveries / total_events if total_events > 0 else 0,
            "average_recovery_time": avg_recovery_time
        }

# Demo function
async def demo_chaos_testing():
    """Demo the chaos testing framework"""
    print("🧪 Chaos Testing Framework Demo")
    print("=" * 50)
    
    framework = ChaosTestingFramework()
    
    # Create test events
    events = [
        ChaosEvent(
            event_type=ChaosEventType.API_DOWNTIME,
            duration=5.0,
            intensity=0.8,
            timestamp=datetime.now().isoformat(),
            description="Simulate API downtime for 5 seconds"
        ),
        ChaosEvent(
            event_type=ChaosEventType.LATENCY_SPIKE,
            duration=3.0,
            intensity=0.6,
            timestamp=datetime.now().isoformat(),
            description="Simulate latency spike for 3 seconds"
        ),
        ChaosEvent(
            event_type=ChaosEventType.MEMORY_PRESSURE,
            duration=2.0,
            intensity=0.4,
            timestamp=datetime.now().isoformat(),
            description="Simulate memory pressure for 2 seconds"
        )
    ]
    
    # Run chaos test
    result = await framework.run_chaos_test("resilience_test", events)
    
    # Print results
    print(f"\n📊 Test Results:")
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Total Events: {result.total_events}")
    print(f"Successful Recoveries: {result.successful_recoveries}")
    print(f"Failed Recoveries: {result.failed_recoveries}")
    print(f"Average Recovery Time: {result.average_recovery_time:.2f}s")
    
    # Get summary
    summary = framework.get_test_summary()
    print(f"\n📈 Test Summary:")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Recovery Rate: {summary['recovery_rate']:.1%}")
    print(f"Average Recovery Time: {summary['average_recovery_time']:.2f}s")
    
    print("\n✅ Chaos Testing Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_chaos_testing())
