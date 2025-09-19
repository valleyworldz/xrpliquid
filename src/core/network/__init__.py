"""
🌐 NETWORK MODULE
================
Network resilience and connectivity management for institutional trading.

This module provides:
- Network failover and load balancing
- DNS resolution alternatives
- Circuit breaker patterns
- Offline mode operations
- Connection health monitoring
"""

from .network_resilience_engine import (
    NetworkResilienceEngine,
    EndpointConfig,
    ConnectionHealth,
    NetworkStats,
    ConnectionStatus,
    FailoverStrategy,
    resolve_hyperliquid_ips,
    patch_hosts_file
)

__all__ = [
    'NetworkResilienceEngine',
    'EndpointConfig', 
    'ConnectionHealth',
    'NetworkStats',
    'ConnectionStatus',
    'FailoverStrategy',
    'resolve_hyperliquid_ips',
    'patch_hosts_file'
]
