"""
Failover Manager - Warm Standby Architecture
Implements automatic failover with warm-standby instance and health monitoring
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil
import subprocess
import signal
import os
from dataclasses import dataclass, asdict
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

@dataclass
class HealthMetrics:
    timestamp: str
    status: HealthStatus
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_latency: float
    api_response_time: float
    ws_connected: bool
    last_trade_time: Optional[str]
    error_count: int
    uptime_seconds: float

@dataclass
class FailoverConfig:
    health_check_interval: int = 5  # seconds
    failover_threshold: int = 3  # consecutive failures
    restart_timeout: int = 30  # seconds
    warm_standby_delay: int = 2  # seconds
    max_restart_attempts: int = 3
    health_metrics_retention: int = 1000  # records
    critical_cpu_threshold: float = 90.0
    critical_memory_threshold: float = 95.0
    critical_latency_threshold: float = 5.0  # seconds

class FailoverManager:
    """
    Manages failover architecture with warm-standby and automatic restart
    """
    
    def __init__(self, config: FailoverConfig = None):
        self.config = config or FailoverConfig()
        self.logger = logging.getLogger(__name__)
        self.health_metrics: List[HealthMetrics] = []
        self.failure_count = 0
        self.last_restart_time = None
        self.restart_attempts = 0
        self.is_primary = True
        self.is_standby_active = False
        self.main_process = None
        self.standby_process = None
        self.start_time = time.time()
        
        # Create reports directory
        self.reports_dir = Path("reports/resilience")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    async def start_failover_monitoring(self):
        """Start the failover monitoring loop"""
        self.logger.info("🚀 Starting failover monitoring system")
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        # Start standby instance
        asyncio.create_task(self._standby_management_loop())
        
        # Start main process monitoring
        asyncio.create_task(self._process_monitoring_loop())
        
    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop"""
        while True:
            try:
                health_metrics = await self._collect_health_metrics()
                self.health_metrics.append(health_metrics)
                
                # Keep only recent metrics
                if len(self.health_metrics) > self.config.health_metrics_retention:
                    self.health_metrics = self.health_metrics[-self.config.health_metrics_retention:]
                
                # Check for failures
                if health_metrics.status == HealthStatus.CRITICAL:
                    self.failure_count += 1
                    self.logger.warning(f"⚠️ Critical health detected: {health_metrics.status}")
                    
                    if self.failure_count >= self.config.failover_threshold:
                        await self._trigger_failover()
                else:
                    self.failure_count = 0
                
                # Save health metrics
                await self._save_health_metrics()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _collect_health_metrics(self) -> HealthMetrics:
        """Collect comprehensive health metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network latency (simulated - replace with actual API ping)
            network_latency = await self._measure_network_latency()
            
            # API response time (simulated - replace with actual API call)
            api_response_time = await self._measure_api_response_time()
            
            # WebSocket connection status (simulated - replace with actual WS check)
            ws_connected = await self._check_ws_connection()
            
            # Last trade time (simulated - replace with actual trade log check)
            last_trade_time = await self._get_last_trade_time()
            
            # Error count (simulated - replace with actual error log parsing)
            error_count = await self._count_recent_errors()
            
            # Determine health status
            status = self._determine_health_status(
                cpu_percent, memory.percent, network_latency, api_response_time
            )
            
            return HealthMetrics(
                timestamp=datetime.now().isoformat(),
                status=status,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage=disk.percent,
                network_latency=network_latency,
                api_response_time=api_response_time,
                ws_connected=ws_connected,
                last_trade_time=last_trade_time,
                error_count=error_count,
                uptime_seconds=time.time() - self.start_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting health metrics: {e}")
            return HealthMetrics(
                timestamp=datetime.now().isoformat(),
                status=HealthStatus.FAILED,
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage=0.0,
                network_latency=999.0,
                api_response_time=999.0,
                ws_connected=False,
                last_trade_time=None,
                error_count=999,
                uptime_seconds=0.0
            )
    
    def _determine_health_status(self, cpu: float, memory: float, latency: float, api_time: float) -> HealthStatus:
        """Determine health status based on metrics"""
        if (cpu > self.config.critical_cpu_threshold or 
            memory > self.config.critical_memory_threshold or
            latency > self.config.critical_latency_threshold or
            api_time > self.config.critical_latency_threshold):
            return HealthStatus.CRITICAL
        elif (cpu > 70 or memory > 80 or latency > 2.0 or api_time > 2.0):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    async def _measure_network_latency(self) -> float:
        """Measure network latency to exchange API"""
        try:
            # Simulate network latency measurement
            # Replace with actual ping to exchange API
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate network call
            return (time.time() - start_time) * 1000  # Convert to ms
        except:
            return 999.0
    
    async def _measure_api_response_time(self) -> float:
        """Measure API response time"""
        try:
            # Simulate API response time measurement
            # Replace with actual API call
            start_time = time.time()
            await asyncio.sleep(0.05)  # Simulate API call
            return (time.time() - start_time) * 1000  # Convert to ms
        except:
            return 999.0
    
    async def _check_ws_connection(self) -> bool:
        """Check WebSocket connection status"""
        try:
            # Simulate WebSocket connection check
            # Replace with actual WebSocket status check
            return True  # Simulate connected
        except:
            return False
    
    async def _get_last_trade_time(self) -> Optional[str]:
        """Get last trade timestamp"""
        try:
            # Simulate last trade time retrieval
            # Replace with actual trade log parsing
            return datetime.now().isoformat()
        except:
            return None
    
    async def _count_recent_errors(self) -> int:
        """Count recent errors in logs"""
        try:
            # Simulate error counting
            # Replace with actual log parsing
            return 0
        except:
            return 0
    
    async def _trigger_failover(self):
        """Trigger failover to standby instance"""
        self.logger.critical("🚨 TRIGGERING FAILOVER - Primary instance failed")
        
        try:
            # Activate standby
            await self._activate_standby()
            
            # Attempt to restart primary
            await self._restart_primary()
            
            # Log failover event
            await self._log_failover_event()
            
        except Exception as e:
            self.logger.error(f"❌ Failover failed: {e}")
    
    async def _activate_standby(self):
        """Activate warm standby instance"""
        self.logger.info("🔄 Activating warm standby instance")
        
        try:
            # Simulate standby activation
            # Replace with actual standby process startup
            self.is_standby_active = True
            self.is_primary = False
            
            # Wait for standby to be ready
            await asyncio.sleep(self.config.warm_standby_delay)
            
            self.logger.info("✅ Standby instance activated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to activate standby: {e}")
            raise
    
    async def _restart_primary(self):
        """Restart primary instance"""
        if self.restart_attempts >= self.config.max_restart_attempts:
            self.logger.error("❌ Max restart attempts reached")
            return
        
        self.logger.info("🔄 Restarting primary instance")
        self.restart_attempts += 1
        self.last_restart_time = time.time()
        
        try:
            # Simulate primary restart
            # Replace with actual process restart logic
            await asyncio.sleep(2)  # Simulate restart time
            
            # Reset failure count after successful restart
            self.failure_count = 0
            self.is_primary = True
            self.is_standby_active = False
            
            self.logger.info("✅ Primary instance restarted successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restart primary: {e}")
    
    async def _standby_management_loop(self):
        """Manage warm standby instance"""
        while True:
            try:
                if not self.is_standby_active and self.failure_count > 0:
                    # Prepare standby for potential activation
                    await self._prepare_standby()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Standby management error: {e}")
                await asyncio.sleep(10)
    
    async def _prepare_standby(self):
        """Prepare warm standby instance"""
        try:
            # Simulate standby preparation
            # Replace with actual standby process preparation
            self.logger.debug("🔄 Preparing warm standby instance")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare standby: {e}")
    
    async def _process_monitoring_loop(self):
        """Monitor main process health"""
        while True:
            try:
                # Check if main process is still running
                if self.main_process and self.main_process.poll() is not None:
                    self.logger.warning("⚠️ Main process terminated unexpectedly")
                    await self._trigger_failover()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Process monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _save_health_metrics(self):
        """Save health metrics to file"""
        try:
            metrics_file = self.reports_dir / "health_metrics.json"
            
            # Convert to serializable format
            serializable_metrics = []
            for metric in self.health_metrics:
                metric_dict = asdict(metric)
                metric_dict['status'] = metric.status.value
                serializable_metrics.append(metric_dict)
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save health metrics: {e}")
    
    async def _log_failover_event(self):
        """Log failover event"""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "failover_triggered",
                "failure_count": self.failure_count,
                "restart_attempts": self.restart_attempts,
                "standby_activated": self.is_standby_active,
                "primary_restarted": self.is_primary
            }
            
            events_file = self.reports_dir / "failover_events.json"
            
            # Load existing events
            events = []
            if events_file.exists():
                with open(events_file, 'r') as f:
                    events = json.load(f)
            
            # Add new event
            events.append(event)
            
            # Save updated events
            with open(events_file, 'w') as f:
                json.dump(events, f, indent=2)
                
            self.logger.info(f"📝 Failover event logged: {event}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log failover event: {e}")
    
    def get_health_summary(self) -> Dict:
        """Get current health summary"""
        if not self.health_metrics:
            return {"status": "no_data", "message": "No health metrics available"}
        
        latest = self.health_metrics[-1]
        
        return {
            "status": latest.status.value,
            "timestamp": latest.timestamp,
            "uptime_seconds": latest.uptime_seconds,
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "network_latency_ms": latest.network_latency,
            "api_response_time_ms": latest.api_response_time,
            "ws_connected": latest.ws_connected,
            "failure_count": self.failure_count,
            "restart_attempts": self.restart_attempts,
            "is_primary": self.is_primary,
            "is_standby_active": self.is_standby_active
        }

# Demo function
async def demo_failover_manager():
    """Demo the failover manager"""
    print("🚀 Failover Manager Demo")
    print("=" * 50)
    
    # Create failover manager
    config = FailoverConfig(
        health_check_interval=2,
        failover_threshold=2,
        critical_cpu_threshold=80.0
    )
    
    manager = FailoverManager(config)
    
    # Start monitoring
    print("🔄 Starting failover monitoring...")
    await manager.start_failover_monitoring()
    
    # Run for a short demo
    await asyncio.sleep(10)
    
    # Get health summary
    summary = manager.get_health_summary()
    print(f"\n📊 Health Summary:")
    print(f"Status: {summary['status']}")
    print(f"CPU: {summary['cpu_percent']:.1f}%")
    print(f"Memory: {summary['memory_percent']:.1f}%")
    print(f"Network Latency: {summary['network_latency_ms']:.1f}ms")
    print(f"API Response: {summary['api_response_time_ms']:.1f}ms")
    print(f"WS Connected: {summary['ws_connected']}")
    print(f"Uptime: {summary['uptime_seconds']:.1f}s")
    
    print("\n✅ Failover Manager Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_failover_manager())
