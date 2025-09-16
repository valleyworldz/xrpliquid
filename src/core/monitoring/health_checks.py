"""
Health Checks & Graceful Drain
Implements /healthz, /readyz endpoints and graceful shutdown procedures.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import aiohttp
from aiohttp import web


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Represents a health check component."""
    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    response_time_ms: float
    details: Dict[str, Any]


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.health_dir = self.reports_dir / "health"
        self.health_dir.mkdir(parents=True, exist_ok=True)
        
        # Health check components
        self.components = {}
        
        # Graceful drain state
        self.drain_mode = False
        self.drain_start_time = None
        self.drain_timeout_seconds = 300  # 5 minutes
        
        # Health check history
        self.health_history = []
    
    def register_component(self, 
                          name: str, 
                          check_function: callable,
                          critical: bool = True,
                          timeout_seconds: int = 30) -> None:
        """Register a health check component."""
        
        self.components[name] = {
            "check_function": check_function,
            "critical": critical,
            "timeout_seconds": timeout_seconds,
            "last_check": None,
            "last_status": HealthStatus.UNKNOWN
        }
    
    async def check_component(self, name: str) -> HealthCheck:
        """Check a single component's health."""
        
        if name not in self.components:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Component not registered",
                last_check=datetime.now(timezone.utc),
                response_time_ms=0.0,
                details={}
            )
        
        component = self.components[name]
        start_time = time.time()
        
        try:
            # Run health check with timeout
            result = await asyncio.wait_for(
                component["check_function"](),
                timeout=component["timeout_seconds"]
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Parse result
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
                message = result.get("message", "OK")
                details = result.get("details", {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Check failed"
                details = {}
            
            health_check = HealthCheck(
                name=name,
                status=status,
                message=message,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                details=details
            )
            
        except asyncio.TimeoutError:
            health_check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {component['timeout_seconds']}s",
                last_check=datetime.now(timezone.utc),
                response_time_ms=component["timeout_seconds"] * 1000,
                details={"timeout": True}
            )
            
        except Exception as e:
            health_check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                last_check=datetime.now(timezone.utc),
                response_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
        
        # Update component state
        component["last_check"] = health_check.last_check
        component["last_status"] = health_check.status
        
        return health_check
    
    async def check_all_components(self) -> Dict[str, HealthCheck]:
        """Check all registered components."""
        
        checks = {}
        
        # Run all checks concurrently
        tasks = []
        for name in self.components.keys():
            task = asyncio.create_task(self.check_component(name))
            tasks.append((name, task))
        
        # Collect results
        for name, task in tasks:
            try:
                checks[name] = await task
            except Exception as e:
                checks[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.now(timezone.utc),
                    response_time_ms=0.0,
                    details={"error": str(e)}
                )
        
        return checks
    
    def get_overall_health(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall system health."""
        
        if not checks:
            return HealthStatus.UNKNOWN
        
        # Check for any critical component failures
        critical_failures = []
        degraded_components = []
        
        for name, check in checks.items():
            component = self.components.get(name, {})
            if component.get("critical", True):
                if check.status == HealthStatus.UNHEALTHY:
                    critical_failures.append(name)
                elif check.status == HealthStatus.DEGRADED:
                    degraded_components.append(name)
        
        # Determine overall status
        if critical_failures:
            return HealthStatus.UNHEALTHY
        elif degraded_components:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    async def healthz_handler(self, request: web.Request) -> web.Response:
        """Handle /healthz endpoint."""
        
        checks = await self.check_all_components()
        overall_health = self.get_overall_health(checks)
        
        # Prepare response
        response_data = {
            "status": overall_health.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "last_check": check.last_check.isoformat(),
                    "response_time_ms": check.response_time_ms,
                    "details": check.details
                }
                for name, check in checks.items()
            }
        }
        
        # Set HTTP status code
        if overall_health == HealthStatus.HEALTHY:
            status_code = 200
        elif overall_health == HealthStatus.DEGRADED:
            status_code = 200  # Still operational
        else:
            status_code = 503  # Service Unavailable
        
        return web.json_response(response_data, status=status_code)
    
    async def readyz_handler(self, request: web.Request) -> web.Response:
        """Handle /readyz endpoint."""
        
        # Check if system is ready to accept traffic
        if self.drain_mode:
            return web.json_response({
                "status": "draining",
                "message": "System is draining and not ready for new requests",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, status=503)
        
        # Check critical components only
        critical_checks = {}
        for name, component in self.components.items():
            if component.get("critical", True):
                critical_checks[name] = await self.check_component(name)
        
        overall_health = self.get_overall_health(critical_checks)
        
        response_data = {
            "status": overall_health.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ready": overall_health in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        }
        
        status_code = 200 if response_data["ready"] else 503
        
        return web.json_response(response_data, status=status_code)
    
    async def start_graceful_drain(self) -> Dict[str, Any]:
        """Start graceful drain process."""
        
        if self.drain_mode:
            return {
                "status": "already_draining",
                "message": "System is already in drain mode",
                "drain_start_time": self.drain_start_time.isoformat() if self.drain_start_time else None
            }
        
        self.drain_mode = True
        self.drain_start_time = datetime.now(timezone.utc)
        
        drain_info = {
            "status": "draining_started",
            "drain_start_time": self.drain_start_time.isoformat(),
            "drain_timeout_seconds": self.drain_timeout_seconds,
            "actions": [
                "Stop accepting new orders",
                "Cancel pending orders",
                "Flatten positions",
                "Close connections",
                "Save state"
            ]
        }
        
        # Log drain start
        await self._log_drain_event("drain_started", drain_info)
        
        return drain_info
    
    async def stop_graceful_drain(self) -> Dict[str, Any]:
        """Stop graceful drain process."""
        
        if not self.drain_mode:
            return {
                "status": "not_draining",
                "message": "System is not in drain mode"
            }
        
        drain_duration = (datetime.now(timezone.utc) - self.drain_start_time).total_seconds()
        
        drain_info = {
            "status": "drain_completed",
            "drain_start_time": self.drain_start_time.isoformat(),
            "drain_end_time": datetime.now(timezone.utc).isoformat(),
            "drain_duration_seconds": drain_duration
        }
        
        # Reset drain state
        self.drain_mode = False
        self.drain_start_time = None
        
        # Log drain completion
        await self._log_drain_event("drain_completed", drain_info)
        
        return drain_info
    
    async def _log_drain_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log drain events."""
        
        log_entry = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": event_data
        }
        
        # Save to drain log
        drain_log_file = self.health_dir / "drain_events.jsonl"
        with open(drain_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    async def create_health_check_app(self) -> web.Application:
        """Create aiohttp application with health check endpoints."""
        
        app = web.Application()
        
        # Add health check endpoints
        app.router.add_get('/healthz', self.healthz_handler)
        app.router.add_get('/readyz', self.readyz_handler)
        
        # Add drain endpoints
        app.router.add_post('/drain/start', self._drain_start_handler)
        app.router.add_post('/drain/stop', self._drain_stop_handler)
        app.router.add_get('/drain/status', self._drain_status_handler)
        
        return app
    
    async def _drain_start_handler(self, request: web.Request) -> web.Response:
        """Handle drain start request."""
        result = await self.start_graceful_drain()
        return web.json_response(result)
    
    async def _drain_stop_handler(self, request: web.Request) -> web.Response:
        """Handle drain stop request."""
        result = await self.stop_graceful_drain()
        return web.json_response(result)
    
    async def _drain_status_handler(self, request: web.Request) -> web.Response:
        """Handle drain status request."""
        
        status_data = {
            "drain_mode": self.drain_mode,
            "drain_start_time": self.drain_start_time.isoformat() if self.drain_start_time else None,
            "drain_timeout_seconds": self.drain_timeout_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.drain_mode and self.drain_start_time:
            elapsed = (datetime.now(timezone.utc) - self.drain_start_time).total_seconds()
            status_data["elapsed_seconds"] = elapsed
            status_data["remaining_seconds"] = max(0, self.drain_timeout_seconds - elapsed)
        
        return web.json_response(status_data)
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "drain_mode": self.drain_mode,
            "drain_start_time": self.drain_start_time.isoformat() if self.drain_start_time else None,
            "registered_components": list(self.components.keys()),
            "component_status": {}
        }
        
        # Get current status of all components
        for name, component in self.components.items():
            report["component_status"][name] = {
                "critical": component.get("critical", True),
                "timeout_seconds": component.get("timeout_seconds", 30),
                "last_check": component.get("last_check").isoformat() if component.get("last_check") else None,
                "last_status": component.get("last_status").value if component.get("last_status") else "unknown"
            }
        
        return report


def main():
    """Test health checker functionality."""
    
    async def test_health_checks():
        checker = HealthChecker()
        
        # Register test components
        async def healthy_component():
            return {"status": "healthy", "message": "OK", "details": {}}
        
        async def degraded_component():
            return {"status": "degraded", "message": "High latency", "details": {"latency_ms": 1000}}
        
        async def unhealthy_component():
            return {"status": "unhealthy", "message": "Connection failed", "details": {"error": "timeout"}}
        
        checker.register_component("database", healthy_component, critical=True)
        checker.register_component("api", degraded_component, critical=False)
        checker.register_component("websocket", unhealthy_component, critical=True)
        
        # Test individual component checks
        db_check = await checker.check_component("database")
        print(f"✅ Database health: {db_check.status.value}")
        
        # Test all components
        all_checks = await checker.check_all_components()
        overall_health = checker.get_overall_health(all_checks)
        print(f"✅ Overall health: {overall_health.value}")
        
        # Test graceful drain
        drain_start = await checker.start_graceful_drain()
        print(f"✅ Drain started: {drain_start['status']}")
        
        drain_stop = await checker.stop_graceful_drain()
        print(f"✅ Drain stopped: {drain_stop['status']}")
        
        # Generate report
        report = checker.generate_health_report()
        print(f"✅ Health report generated: {len(report['registered_components'])} components")
    
    # Run async test
    asyncio.run(test_health_checks())


if __name__ == "__main__":
    main()
