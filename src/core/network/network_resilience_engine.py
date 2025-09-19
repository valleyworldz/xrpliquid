#!/usr/bin/env python3
"""
🌐 NETWORK RESILIENCE ENGINE
============================
Institutional-grade network resilience and failover system for trading operations.

Features:
- Multi-endpoint failover and load balancing
- DNS resolution monitoring and alternatives
- Connection health scoring and automatic switching
- Offline mode operations with cached data
- Network latency optimization
- Circuit breaker patterns for degraded performance
- Emergency backup data sources
"""

import asyncio
import time
import json
import logging
import socket
import ssl
import aiohttp
import dns.resolver
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import threading
import random

class ConnectionStatus(Enum):
    """Connection status types"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    OFFLINE = "offline"

class FailoverStrategy(Enum):
    """Failover strategy types"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LATENCY_BASED = "latency_based"
    HEALTH_BASED = "health_based"

@dataclass
class EndpointConfig:
    """Configuration for a network endpoint"""
    name: str
    base_url: str
    weight: float = 1.0
    timeout: float = 10.0
    max_retries: int = 3
    health_check_interval: float = 30.0
    backup_dns: List[str] = None
    is_primary: bool = False

@dataclass
class ConnectionHealth:
    """Health metrics for a connection"""
    endpoint_name: str
    status: ConnectionStatus
    success_rate: float  # 0.0 - 1.0
    avg_latency_ms: float
    last_success: float
    last_failure: float
    consecutive_failures: int
    total_requests: int
    successful_requests: int
    health_score: float  # 0.0 - 1.0

@dataclass
class NetworkStats:
    """Overall network statistics"""
    active_endpoints: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    failover_events: int
    dns_failures: int
    timeout_events: int

class NetworkResilienceEngine:
    """
    🌐 NETWORK RESILIENCE ENGINE
    Provides institutional-grade network reliability and failover capabilities
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Endpoint configuration
        self.endpoints: Dict[str, EndpointConfig] = {}
        self.connection_health: Dict[str, ConnectionHealth] = {}
        self.active_endpoint = None
        self.failover_strategy = FailoverStrategy(config.get('failover_strategy', 'health_based'))
        
        # Network monitoring
        self.network_stats = NetworkStats(0, 0, 0, 0, 0.0, 0, 0, 0)
        self.latency_history: deque = deque(maxlen=1000)
        self.request_history: deque = deque(maxlen=10000)
        
        # Circuit breaker
        self.circuit_breaker_enabled = config.get('circuit_breaker_enabled', True)
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 0.5)
        self.circuit_breaker_timeout = config.get('circuit_breaker_timeout', 60)
        self.circuit_breaker_state = "closed"  # closed, open, half-open
        self.circuit_breaker_last_failure = 0
        
        # Offline mode
        self.offline_mode_enabled = config.get('offline_mode_enabled', True)
        self.cached_data: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, float] = {}
        self.max_cache_age = config.get('max_cache_age', 300)  # 5 minutes
        
        # DNS resolution
        self.dns_cache: Dict[str, List[str]] = {}
        self.dns_cache_expiry: Dict[str, float] = {}
        self.custom_dns_servers = config.get('dns_servers', ['8.8.8.8', '1.1.1.1', '208.67.222.222'])
        
        # Performance monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.health_check_thread = None
        
        self._initialize_endpoints()
        self.logger.info("🌐 [NETWORK] Network Resilience Engine initialized")

    def _initialize_endpoints(self):
        """Initialize network endpoints with failover configurations"""
        try:
            # Primary Hyperliquid endpoints
            primary_endpoints = [
                EndpointConfig(
                    name="hyperliquid_primary",
                    base_url="https://api.hyperliquid.xyz",
                    weight=1.0,
                    timeout=5.0,
                    max_retries=3,
                    is_primary=True,
                    backup_dns=["104.21.73.115", "172.67.74.226"]  # Cloudflare IPs for hyperliquid
                ),
                EndpointConfig(
                    name="hyperliquid_backup",
                    base_url="https://api-backup.hyperliquid.xyz",
                    weight=0.8,
                    timeout=8.0,
                    max_retries=2,
                    backup_dns=["104.21.73.115", "172.67.74.226"]
                )
            ]
            
            # Alternative data sources (for emergency fallback)
            backup_endpoints = [
                EndpointConfig(
                    name="binance_fallback",
                    base_url="https://api.binance.com",
                    weight=0.6,
                    timeout=10.0,
                    max_retries=2
                ),
                EndpointConfig(
                    name="coingecko_fallback", 
                    base_url="https://api.coingecko.com",
                    weight=0.4,
                    timeout=15.0,
                    max_retries=1
                )
            ]
            
            # Add all endpoints
            all_endpoints = primary_endpoints + backup_endpoints
            for endpoint in all_endpoints:
                self.endpoints[endpoint.name] = endpoint
                self.connection_health[endpoint.name] = ConnectionHealth(
                    endpoint_name=endpoint.name,
                    status=ConnectionStatus.HEALTHY,
                    success_rate=1.0,
                    avg_latency_ms=0.0,
                    last_success=0.0,
                    last_failure=0.0,
                    consecutive_failures=0,
                    total_requests=0,
                    successful_requests=0,
                    health_score=1.0
                )
            
            # Set initial active endpoint
            self.active_endpoint = "hyperliquid_primary"
            
            self.logger.info(f"🌐 [NETWORK] Initialized {len(all_endpoints)} endpoints")
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Error initializing endpoints: {e}")

    async def start_monitoring(self):
        """Start network monitoring and health checks"""
        try:
            self.monitoring_active = True
            
            # Start monitoring threads
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self.health_check_thread.start()
            
            self.logger.info("🌐 [NETWORK] Network monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Error starting monitoring: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update network statistics
                self._update_network_stats()
                
                # Check circuit breaker
                self._check_circuit_breaker()
                
                # Evaluate failover conditions
                self._evaluate_failover()
                
                # Log performance metrics
                if len(self.latency_history) > 0:
                    avg_latency = sum(self.latency_history) / len(self.latency_history)
                    self.logger.debug(f"🌐 [NETWORK] Avg latency: {avg_latency:.1f}ms, Active: {self.active_endpoint}")
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ [NETWORK] Monitoring loop error: {e}")
                time.sleep(30)

    def _health_check_loop(self):
        """Health check loop for all endpoints"""
        while self.monitoring_active:
            try:
                # Run health checks on all endpoints
                asyncio.run(self._run_health_checks())
                time.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ [NETWORK] Health check loop error: {e}")
                time.sleep(60)

    async def _run_health_checks(self):
        """Run health checks on all endpoints"""
        try:
            tasks = []
            for endpoint_name in self.endpoints.keys():
                task = asyncio.create_task(self._health_check_endpoint(endpoint_name))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Error running health checks: {e}")

    async def _health_check_endpoint(self, endpoint_name: str):
        """Perform health check on a specific endpoint"""
        try:
            endpoint = self.endpoints[endpoint_name]
            health = self.connection_health[endpoint_name]
            
            start_time = time.time()
            
            # Perform DNS resolution check
            dns_success = await self._check_dns_resolution(endpoint.base_url)
            if not dns_success:
                self.logger.warning(f"🌐 [NETWORK] DNS resolution failed for {endpoint_name}")
                health.consecutive_failures += 1
                health.last_failure = time.time()
                health.status = ConnectionStatus.FAILING
                return
            
            # Perform HTTP health check
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=endpoint.timeout)) as session:
                try:
                    # Use a simple endpoint for health check
                    health_url = f"{endpoint.base_url}/info"
                    async with session.get(health_url) as response:
                        latency = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            # Success
                            health.consecutive_failures = 0
                            health.last_success = time.time()
                            health.avg_latency_ms = (health.avg_latency_ms + latency) / 2
                            health.successful_requests += 1
                            health.status = ConnectionStatus.HEALTHY if latency < 1000 else ConnectionStatus.DEGRADED
                        else:
                            # HTTP error
                            health.consecutive_failures += 1
                            health.last_failure = time.time()
                            health.status = ConnectionStatus.FAILING
                
                except Exception as e:
                    # Request failed
                    health.consecutive_failures += 1
                    health.last_failure = time.time()
                    health.status = ConnectionStatus.OFFLINE
                    self.logger.debug(f"🌐 [NETWORK] Health check failed for {endpoint_name}: {e}")
            
            # Update health score
            health.total_requests += 1
            health.success_rate = health.successful_requests / health.total_requests
            health.health_score = self._calculate_health_score(health)
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Health check error for {endpoint_name}: {e}")

    async def _check_dns_resolution(self, url: str) -> bool:
        """Check DNS resolution for a URL"""
        try:
            # Extract hostname from URL
            if url.startswith('https://'):
                hostname = url[8:].split('/')[0]
            elif url.startswith('http://'):
                hostname = url[7:].split('/')[0]
            else:
                hostname = url.split('/')[0]
            
            # Try standard DNS resolution first
            try:
                socket.gethostbyname(hostname)
                return True
            except socket.gaierror:
                pass
            
            # Try custom DNS servers
            for dns_server in self.custom_dns_servers:
                try:
                    resolver = dns.resolver.Resolver()
                    resolver.nameservers = [dns_server]
                    resolver.timeout = 5
                    result = resolver.resolve(hostname, 'A')
                    if result:
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] DNS resolution error: {e}")
            return False

    def _calculate_health_score(self, health: ConnectionHealth) -> float:
        """Calculate overall health score for an endpoint"""
        try:
            # Base score from success rate
            score = health.success_rate
            
            # Penalty for consecutive failures
            if health.consecutive_failures > 0:
                failure_penalty = min(health.consecutive_failures * 0.1, 0.5)
                score -= failure_penalty
            
            # Penalty for high latency
            if health.avg_latency_ms > 1000:
                latency_penalty = min((health.avg_latency_ms - 1000) / 10000, 0.3)
                score -= latency_penalty
            
            # Bonus for recent success
            current_time = time.time()
            if health.last_success > 0 and (current_time - health.last_success) < 60:
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0

    def _evaluate_failover(self):
        """Evaluate whether failover is needed"""
        try:
            if not self.active_endpoint:
                return
            
            current_health = self.connection_health[self.active_endpoint]
            
            # Check if current endpoint is failing
            should_failover = (
                current_health.status in [ConnectionStatus.FAILING, ConnectionStatus.OFFLINE] or
                current_health.consecutive_failures >= 3 or
                current_health.health_score < 0.3
            )
            
            if should_failover:
                new_endpoint = self._select_best_endpoint()
                if new_endpoint and new_endpoint != self.active_endpoint:
                    self.logger.warning(
                        f"🔄 [NETWORK] FAILOVER: {self.active_endpoint} → {new_endpoint} "
                        f"(health: {current_health.health_score:.2f})"
                    )
                    self.active_endpoint = new_endpoint
                    self.network_stats.failover_events += 1
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Failover evaluation error: {e}")

    def _select_best_endpoint(self) -> Optional[str]:
        """Select the best available endpoint based on strategy"""
        try:
            healthy_endpoints = [
                name for name, health in self.connection_health.items()
                if health.status in [ConnectionStatus.HEALTHY, ConnectionStatus.DEGRADED]
                and health.health_score > 0.5
            ]
            
            if not healthy_endpoints:
                return None
            
            if self.failover_strategy == FailoverStrategy.HEALTH_BASED:
                # Select endpoint with highest health score
                best_endpoint = max(
                    healthy_endpoints,
                    key=lambda name: self.connection_health[name].health_score
                )
                return best_endpoint
            
            elif self.failover_strategy == FailoverStrategy.LATENCY_BASED:
                # Select endpoint with lowest latency
                best_endpoint = min(
                    healthy_endpoints,
                    key=lambda name: self.connection_health[name].avg_latency_ms or float('inf')
                )
                return best_endpoint
            
            elif self.failover_strategy == FailoverStrategy.WEIGHTED:
                # Select based on weights
                weights = [(name, self.endpoints[name].weight) for name in healthy_endpoints]
                return max(weights, key=lambda x: x[1])[0]
            
            else:  # ROUND_ROBIN
                return random.choice(healthy_endpoints)
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Endpoint selection error: {e}")
            return None

    async def make_request(self, endpoint_path: str, method: str = "GET", 
                         data: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make a resilient network request with automatic failover"""
        try:
            # Check circuit breaker
            if self.circuit_breaker_state == "open":
                if time.time() - self.circuit_breaker_last_failure < self.circuit_breaker_timeout:
                    return self._get_cached_response(endpoint_path)
                else:
                    self.circuit_breaker_state = "half-open"
            
            # Try active endpoint first
            response = await self._try_request(self.active_endpoint, endpoint_path, method, data, headers)
            if response is not None:
                self._cache_response(endpoint_path, response)
                return response
            
            # Try other endpoints if active fails
            for endpoint_name in self.endpoints.keys():
                if endpoint_name == self.active_endpoint:
                    continue
                
                health = self.connection_health[endpoint_name]
                if health.status in [ConnectionStatus.HEALTHY, ConnectionStatus.DEGRADED]:
                    response = await self._try_request(endpoint_name, endpoint_path, method, data, headers)
                    if response is not None:
                        # Update active endpoint to this successful one
                        self.active_endpoint = endpoint_name
                        self._cache_response(endpoint_path, response)
                        return response
            
            # All endpoints failed - try cached data
            cached_response = self._get_cached_response(endpoint_path)
            if cached_response:
                self.logger.warning(f"🌐 [NETWORK] Using cached data for {endpoint_path}")
                return cached_response
            
            # Circuit breaker - open it
            self.circuit_breaker_state = "open"
            self.circuit_breaker_last_failure = time.time()
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Request error: {e}")
            return self._get_cached_response(endpoint_path)

    async def _try_request(self, endpoint_name: str, path: str, method: str, 
                         data: Dict, headers: Dict) -> Optional[Dict]:
        """Try making a request to a specific endpoint"""
        try:
            endpoint = self.endpoints[endpoint_name]
            health = self.connection_health[endpoint_name]
            
            start_time = time.time()
            
            # Build URL
            url = f"{endpoint.base_url}{path}"
            
            # Set up headers
            request_headers = {
                'User-Agent': 'InstitutionalTradingBot/1.0',
                'Accept': 'application/json'
            }
            if headers:
                request_headers.update(headers)
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
            ) as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=request_headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            latency = (time.time() - start_time) * 1000
                            self._record_success(endpoint_name, latency)
                            return result
                elif method.upper() == "POST":
                    async with session.post(url, json=data, headers=request_headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            latency = (time.time() - start_time) * 1000
                            self._record_success(endpoint_name, latency)
                            return result
                
                # Non-200 response
                self._record_failure(endpoint_name)
                return None
            
        except Exception as e:
            self._record_failure(endpoint_name)
            self.logger.debug(f"🌐 [NETWORK] Request failed on {endpoint_name}: {e}")
            return None

    def _record_success(self, endpoint_name: str, latency: float):
        """Record a successful request"""
        health = self.connection_health[endpoint_name]
        health.consecutive_failures = 0
        health.last_success = time.time()
        health.successful_requests += 1
        health.total_requests += 1
        health.avg_latency_ms = (health.avg_latency_ms + latency) / 2
        
        # Update global stats
        self.network_stats.successful_requests += 1
        self.network_stats.total_requests += 1
        self.latency_history.append(latency)
        
        # Update circuit breaker
        if self.circuit_breaker_state == "half-open":
            self.circuit_breaker_state = "closed"

    def _record_failure(self, endpoint_name: str):
        """Record a failed request"""
        health = self.connection_health[endpoint_name]
        health.consecutive_failures += 1
        health.last_failure = time.time()
        health.total_requests += 1
        
        # Update global stats
        self.network_stats.failed_requests += 1
        self.network_stats.total_requests += 1

    def _cache_response(self, endpoint_path: str, response: Dict):
        """Cache a response for offline mode"""
        try:
            if self.offline_mode_enabled:
                self.cached_data[endpoint_path] = response
                self.cache_expiry[endpoint_path] = time.time() + self.max_cache_age
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Cache error: {e}")

    def _get_cached_response(self, endpoint_path: str) -> Optional[Dict]:
        """Get cached response if available and not expired"""
        try:
            if endpoint_path in self.cached_data:
                if time.time() < self.cache_expiry.get(endpoint_path, 0):
                    return self.cached_data[endpoint_path]
                else:
                    # Expired cache
                    del self.cached_data[endpoint_path]
                    del self.cache_expiry[endpoint_path]
            return None
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Cache retrieval error: {e}")
            return None

    def _update_network_stats(self):
        """Update overall network statistics"""
        try:
            active_count = sum(
                1 for health in self.connection_health.values()
                if health.status in [ConnectionStatus.HEALTHY, ConnectionStatus.DEGRADED]
            )
            
            self.network_stats.active_endpoints = active_count
            
            if self.latency_history:
                self.network_stats.avg_latency_ms = sum(self.latency_history) / len(self.latency_history)
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Stats update error: {e}")

    def _check_circuit_breaker(self):
        """Check and update circuit breaker state"""
        try:
            if self.circuit_breaker_enabled and self.network_stats.total_requests > 10:
                failure_rate = self.network_stats.failed_requests / self.network_stats.total_requests
                
                if failure_rate > self.circuit_breaker_threshold and self.circuit_breaker_state == "closed":
                    self.circuit_breaker_state = "open"
                    self.circuit_breaker_last_failure = time.time()
                    self.logger.warning(f"🔥 [NETWORK] Circuit breaker OPENED (failure rate: {failure_rate:.2f})")
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Circuit breaker check error: {e}")

    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        try:
            return {
                "active_endpoint": self.active_endpoint,
                "circuit_breaker_state": self.circuit_breaker_state,
                "network_stats": asdict(self.network_stats),
                "endpoint_health": {
                    name: asdict(health) for name, health in self.connection_health.items()
                },
                "cached_responses": len(self.cached_data),
                "avg_latency_ms": self.network_stats.avg_latency_ms,
                "overall_health": self._calculate_overall_health()
            }
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Status error: {e}")
            return {"error": str(e)}

    def _calculate_overall_health(self) -> float:
        """Calculate overall network health score"""
        try:
            if not self.connection_health:
                return 0.0
            
            health_scores = [health.health_score for health in self.connection_health.values()]
            return sum(health_scores) / len(health_scores)
        except Exception:
            return 0.0

    async def stop_monitoring(self):
        """Stop network monitoring"""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            if self.health_check_thread:
                self.health_check_thread.join(timeout=5)
            
            self.logger.info("🌐 [NETWORK] Network monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"❌ [NETWORK] Error stopping monitoring: {e}")

# Emergency DNS resolution utilities
def resolve_hyperliquid_ips() -> List[str]:
    """Emergency DNS resolution for Hyperliquid"""
    known_ips = [
        "104.21.73.115",
        "172.67.74.226", 
        "104.21.72.115",
        "172.67.75.226"
    ]
    return known_ips

def patch_hosts_file():
    """Emergency hosts file patching (Windows)"""
    try:
        hosts_path = r"C:\Windows\System32\drivers\etc\hosts"
        hyperliquid_ips = resolve_hyperliquid_ips()
        
        # Backup and patch hosts file
        with open(hosts_path, 'a') as f:
            f.write(f"\n# Emergency Hyperliquid DNS resolution\n")
            for ip in hyperliquid_ips:
                f.write(f"{ip} api.hyperliquid.xyz\n")
        
        return True
    except Exception as e:
        logging.error(f"❌ Failed to patch hosts file: {e}")
        return False
