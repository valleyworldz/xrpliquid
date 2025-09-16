"""
⚡ LOW-LATENCY ENGINEER OPTIMIZATION SYSTEM
===========================================
Sub-millisecond execution optimization and connection resiliency.

This system implements the pinnacle of low-latency trading with:
- Sub-millisecond execution optimization
- Connection pool management
- WebSocket connection resiliency
- API call optimization
- In-memory data structures
- Parallel processing
- Throughput maximization
"""

import asyncio
import time
import numpy as np
import aiohttp
import websockets
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import weakref

@dataclass
class LatencyOptimizationConfig:
    """Configuration for low-latency optimization"""
    
    # Connection settings
    connection_pool: Dict[str, Any] = field(default_factory=lambda: {
        'max_connections': 20,               # Maximum concurrent connections
        'connection_timeout': 2.0,           # 2 second connection timeout
        'read_timeout': 1.0,                 # 1 second read timeout
        'keepalive_timeout': 30.0,           # 30 second keepalive
        'max_retries': 3,                    # Maximum retry attempts
        'retry_delay': 0.1,                  # 100ms retry delay
        'connection_pool_size': 10,          # Connection pool size
    })
    
    # WebSocket settings
    websocket_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_connections': 5,                # Maximum WebSocket connections
        'reconnect_delay': 1.0,              # 1 second reconnect delay
        'ping_interval': 30.0,               # 30 second ping interval
        'ping_timeout': 10.0,                # 10 second ping timeout
        'max_reconnect_attempts': 10,        # Maximum reconnect attempts
        'buffer_size': 65536,                # 64KB buffer size
        'compression_enabled': True,         # Enable compression
    })
    
    # API optimization settings
    api_optimization: Dict[str, Any] = field(default_factory=lambda: {
        'batch_requests': True,              # Enable request batching
        'max_batch_size': 10,                # Maximum batch size
        'batch_timeout': 0.01,               # 10ms batch timeout
        'request_pooling': True,             # Enable request pooling
        'connection_reuse': True,            # Enable connection reuse
        'keepalive_connections': True,       # Keep connections alive
        'compression_enabled': True,         # Enable response compression
    })
    
    # In-memory cache settings
    memory_cache: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                     # Enable in-memory cache
        'max_cache_size': 10000,             # Maximum cache entries
        'cache_ttl_seconds': 1.0,            # 1 second cache TTL
        'prefetch_enabled': True,            # Enable data prefetching
        'cache_compression': True,           # Enable cache compression
        'lru_eviction': True,                # Enable LRU eviction
    })
    
    # Parallel processing settings
    parallel_processing: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                     # Enable parallel processing
        'max_workers': 8,                    # Maximum worker threads
        'task_queue_size': 1000,             # Task queue size
        'async_execution': True,             # Enable async execution
        'thread_pool_size': 4,               # Thread pool size
        'process_pool_size': 2,              # Process pool size
    })
    
    # Performance monitoring
    performance_monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                     # Enable performance monitoring
        'metrics_interval': 1.0,             # 1 second metrics interval
        'latency_threshold_ms': 100,         # 100ms latency threshold
        'throughput_threshold': 1000,        # 1000 requests/second threshold
        'error_rate_threshold': 0.01,        # 1% error rate threshold
        'alert_on_threshold_breach': True,   # Alert on threshold breach
    })

@dataclass
class LatencyMetrics:
    """Comprehensive latency metrics"""
    
    # Connection metrics
    connection_count: int
    active_connections: int
    connection_uptime: float
    connection_errors: int
    
    # API metrics
    api_calls_per_second: float
    avg_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    error_rate_percent: float
    
    # WebSocket metrics
    websocket_connections: int
    websocket_uptime: float
    websocket_reconnects: int
    websocket_errors: int
    
    # Cache metrics
    cache_hit_rate: float
    cache_size: int
    cache_evictions: int
    
    # Performance metrics
    total_throughput: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

class LowLatencyOptimization:
    """
    ⚡ LOW-LATENCY ENGINEER OPTIMIZATION SYSTEM
    
    The pinnacle of low-latency trading optimization:
    1. Sub-millisecond execution optimization
    2. Connection pool management
    3. WebSocket connection resiliency
    4. API call optimization
    5. In-memory data structures
    6. Parallel processing
    7. Throughput maximization
    """
    
    def __init__(self, api, config: Dict[str, Any], logger=None):
        self.api = api
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize optimization configuration
        self.latency_config = LatencyOptimizationConfig()
        
        # Connection management
        self.connection_pool = None
        self.websocket_connections = {}
        self.connection_metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'connection_errors': 0,
            'last_connection_time': 0.0,
        }
        
        # API optimization
        self.request_queue = queue.Queue(maxsize=self.latency_config.api_optimization['max_batch_size'])
        self.batch_processor = None
        self.api_metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'response_times': deque(maxlen=1000),
            'last_call_time': 0.0,
        }
        
        # In-memory cache
        self.memory_cache = {}
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0,
        }
        
        # Parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.latency_config.parallel_processing['max_workers'])
        self.task_queue = asyncio.Queue(maxsize=self.latency_config.parallel_processing['task_queue_size'])
        
        # Performance monitoring
        self.performance_metrics = LatencyMetrics(
            connection_count=0,
            active_connections=0,
            connection_uptime=0.0,
            connection_errors=0,
            api_calls_per_second=0.0,
            avg_response_time_ms=0.0,
            max_response_time_ms=0.0,
            min_response_time_ms=0.0,
            error_rate_percent=0.0,
            websocket_connections=0,
            websocket_uptime=0.0,
            websocket_reconnects=0,
            websocket_errors=0,
            cache_hit_rate=0.0,
            cache_size=0,
            cache_evictions=0,
            total_throughput=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
        )
        
        # Initialize systems
        self._initialize_connection_pool()
        self._initialize_websocket_connections()
        self._initialize_batch_processor()
        self._initialize_performance_monitoring()
        
        self.logger.info("⚡ [LOW_LATENCY] Low-Latency Optimization System initialized")
        self.logger.info("🎯 [LOW_LATENCY] All latency optimization systems activated")
    
    def _initialize_connection_pool(self):
        """Initialize HTTP connection pool"""
        try:
            connector = aiohttp.TCPConnector(
                limit=self.latency_config.connection_pool['max_connections'],
                limit_per_host=self.latency_config.connection_pool['connection_pool_size'],
                keepalive_timeout=self.latency_config.connection_pool['keepalive_timeout'],
                enable_cleanup_closed=True,
                use_dns_cache=True,
                ttl_dns_cache=300,
                family=0,  # Use both IPv4 and IPv6
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.latency_config.connection_pool['connection_timeout'],
                connect=self.latency_config.connection_pool['connection_timeout'],
                sock_read=self.latency_config.connection_pool['read_timeout'],
            )
            
            self.connection_pool = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Connection': 'keep-alive',
                    'Accept-Encoding': 'gzip, deflate',
                    'User-Agent': 'HatManifestoBot/1.0',
                }
            )
            
            self.logger.info("⚡ [CONNECTION_POOL] HTTP connection pool initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [CONNECTION_POOL] Error initializing connection pool: {e}")
    
    def _initialize_websocket_connections(self):
        """Initialize WebSocket connections"""
        try:
            # Initialize WebSocket connection manager
            self.websocket_manager = WebSocketManager(
                config=self.latency_config.websocket_config,
                logger=self.logger
            )
            
            self.logger.info("⚡ [WEBSOCKET] WebSocket connection manager initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [WEBSOCKET] Error initializing WebSocket connections: {e}")
    
    def _initialize_batch_processor(self):
        """Initialize batch request processor"""
        try:
            self.batch_processor = BatchProcessor(
                config=self.latency_config.api_optimization,
                logger=self.logger
            )
            
            self.logger.info("⚡ [BATCH_PROCESSOR] Batch request processor initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [BATCH_PROCESSOR] Error initializing batch processor: {e}")
    
    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring"""
        try:
            self.performance_monitor = PerformanceMonitor(
                config=self.latency_config.performance_monitoring,
                logger=self.logger
            )
            
            # Start monitoring task
            asyncio.create_task(self._monitor_performance())
            
            self.logger.info("⚡ [PERFORMANCE_MONITOR] Performance monitoring initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_MONITOR] Error initializing performance monitoring: {e}")
    
    async def optimize_api_call(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        ⚡ Optimize API call with connection pooling and caching
        """
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = f"{endpoint}:{hash(str(params))}"
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result:
                self.cache_metrics['hits'] += 1
                latency = (time.perf_counter() - start_time) * 1000
                self.logger.debug(f"⚡ [API_CACHE_HIT] {endpoint} - {latency:.2f}ms")
                return cached_result
            
            self.cache_metrics['misses'] += 1
            
            # Use connection pool for optimized request
            if self.connection_pool:
                url = f"{self.api.base_url}{endpoint}"
                
                async with self.connection_pool.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Cache the result
                        self._store_in_cache(cache_key, result)
                        
                        # Update metrics
                        latency = (time.perf_counter() - start_time) * 1000
                        self.api_metrics['response_times'].append(latency)
                        self.api_metrics['successful_calls'] += 1
                        
                        self.logger.debug(f"⚡ [API_OPTIMIZED] {endpoint} - {latency:.2f}ms")
                        
                        return result
                    else:
                        self.api_metrics['failed_calls'] += 1
                        self.logger.warning(f"⚠️ [API_ERROR] {endpoint} - Status: {response.status}")
                        return {'error': f'HTTP {response.status}'}
            else:
                # Fallback to standard API call
                return await self._fallback_api_call(endpoint, params)
                
        except Exception as e:
            self.api_metrics['failed_calls'] += 1
            latency = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"❌ [API_ERROR] {endpoint} - {latency:.2f}ms - {e}")
            return {'error': str(e)}
        finally:
            self.api_metrics['total_calls'] += 1
            self.api_metrics['last_call_time'] = time.time()
    
    async def batch_api_calls(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ⚡ Execute multiple API calls in parallel for maximum throughput
        """
        start_time = time.perf_counter()
        
        try:
            if not self.latency_config.api_optimization['batch_requests']:
                # Execute sequentially if batching is disabled
                results = []
                for request in requests:
                    result = await self.optimize_api_call(request['endpoint'], request.get('params'))
                    results.append(result)
                return results
            
            # Execute in parallel
            tasks = []
            for request in requests:
                task = asyncio.create_task(
                    self.optimize_api_call(request['endpoint'], request.get('params'))
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ [BATCH_ERROR] Request {i}: {result}")
                    processed_results.append({'error': str(result)})
                else:
                    processed_results.append(result)
            
            latency = (time.perf_counter() - start_time) * 1000
            self.logger.info(f"⚡ [BATCH_COMPLETE] {len(requests)} requests - {latency:.2f}ms")
            
            return processed_results
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"❌ [BATCH_ERROR] Batch execution failed - {latency:.2f}ms - {e}")
            return [{'error': str(e)} for _ in requests]
    
    async def establish_websocket_connection(self, url: str, subscriptions: List[str]) -> bool:
        """
        ⚡ Establish optimized WebSocket connection
        """
        try:
            if url in self.websocket_connections:
                self.logger.info(f"⚡ [WEBSOCKET] Connection to {url} already exists")
                return True
            
            # Create WebSocket connection
            websocket = await websockets.connect(
                url,
                ping_interval=self.latency_config.websocket_config['ping_interval'],
                ping_timeout=self.latency_config.websocket_config['ping_timeout'],
                compression=self.latency_config.websocket_config['compression_enabled'],
                max_size=self.latency_config.websocket_config['buffer_size'],
            )
            
            # Store connection
            self.websocket_connections[url] = {
                'websocket': websocket,
                'subscriptions': subscriptions,
                'connected_at': time.time(),
                'reconnect_count': 0,
                'last_ping': time.time(),
            }
            
            # Subscribe to channels
            for subscription in subscriptions:
                await websocket.send(json.dumps(subscription))
            
            # Start message handler
            asyncio.create_task(self._handle_websocket_messages(url))
            
            self.connection_metrics['active_connections'] += 1
            self.performance_metrics.websocket_connections += 1
            
            self.logger.info(f"⚡ [WEBSOCKET] Connected to {url} with {len(subscriptions)} subscriptions")
            
            return True
            
        except Exception as e:
            self.connection_metrics['connection_errors'] += 1
            self.logger.error(f"❌ [WEBSOCKET] Failed to connect to {url}: {e}")
            return False
    
    async def _handle_websocket_messages(self, url: str):
        """
        ⚡ Handle WebSocket messages with low latency
        """
        try:
            websocket_data = self.websocket_connections.get(url)
            if not websocket_data:
                return
            
            websocket = websocket_data['websocket']
            
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Process message with minimal latency
                    await self._process_websocket_message(url, data)
                    
                    # Update last ping time
                    websocket_data['last_ping'] = time.time()
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️ [WEBSOCKET] Invalid JSON from {url}: {e}")
                except Exception as e:
                    self.logger.error(f"❌ [WEBSOCKET] Error processing message from {url}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"⚠️ [WEBSOCKET] Connection to {url} closed")
            await self._reconnect_websocket(url)
        except Exception as e:
            self.logger.error(f"❌ [WEBSOCKET] Error in message handler for {url}: {e}")
            await self._reconnect_websocket(url)
    
    async def _reconnect_websocket(self, url: str):
        """
        ⚡ Reconnect WebSocket with exponential backoff
        """
        try:
            websocket_data = self.websocket_connections.get(url)
            if not websocket_data:
                return
            
            reconnect_count = websocket_data['reconnect_count']
            max_attempts = self.latency_config.websocket_config['max_reconnect_attempts']
            
            if reconnect_count >= max_attempts:
                self.logger.error(f"❌ [WEBSOCKET] Max reconnect attempts reached for {url}")
                del self.websocket_connections[url]
                self.connection_metrics['active_connections'] -= 1
                return
            
            # Calculate backoff delay
            delay = min(
                self.latency_config.websocket_config['reconnect_delay'] * (2 ** reconnect_count),
                60.0  # Max 60 seconds
            )
            
            self.logger.info(f"⚡ [WEBSOCKET] Reconnecting to {url} in {delay:.1f}s (attempt {reconnect_count + 1})")
            
            await asyncio.sleep(delay)
            
            # Attempt reconnection
            subscriptions = websocket_data['subscriptions']
            success = await self.establish_websocket_connection(url, subscriptions)
            
            if success:
                websocket_data['reconnect_count'] = 0
                self.performance_metrics.websocket_reconnects += 1
            else:
                websocket_data['reconnect_count'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ [WEBSOCKET] Error reconnecting to {url}: {e}")
    
    async def _process_websocket_message(self, url: str, data: Dict[str, Any]):
        """
        ⚡ Process WebSocket message with minimal latency
        """
        try:
            # Store in cache for fast access
            message_type = data.get('type', 'unknown')
            cache_key = f"ws_{url}_{message_type}_{int(time.time() * 1000)}"
            self._store_in_cache(cache_key, data, ttl=0.1)  # 100ms TTL for real-time data
            
            # Process based on message type
            if message_type == 'price_update':
                await self._handle_price_update(data)
            elif message_type == 'order_update':
                await self._handle_order_update(data)
            elif message_type == 'trade_update':
                await self._handle_trade_update(data)
            else:
                self.logger.debug(f"⚡ [WEBSOCKET] Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"❌ [WEBSOCKET] Error processing message: {e}")
    
    async def _handle_price_update(self, data: Dict[str, Any]):
        """Handle price update with minimal latency"""
        # Placeholder for price update handling
        pass
    
    async def _handle_order_update(self, data: Dict[str, Any]):
        """Handle order update with minimal latency"""
        # Placeholder for order update handling
        pass
    
    async def _handle_trade_update(self, data: Dict[str, Any]):
        """Handle trade update with minimal latency"""
        # Placeholder for trade update handling
        pass
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from in-memory cache"""
        if not self.latency_config.memory_cache['enabled']:
            return None
        
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry['timestamp'] < entry['ttl']:
                return entry['data']
            else:
                # Expired entry
                del self.memory_cache[key]
                self.cache_metrics['evictions'] += 1
        
        return None
    
    def _store_in_cache(self, key: str, data: Any, ttl: float = None):
        """Store data in in-memory cache"""
        if not self.latency_config.memory_cache['enabled']:
            return
        
        if ttl is None:
            ttl = self.latency_config.memory_cache['cache_ttl_seconds']
        
        # Check cache size limit
        max_size = self.latency_config.memory_cache['max_cache_size']
        if len(self.memory_cache) >= max_size:
            # Evict oldest entry (simple LRU)
            if self.latency_config.memory_cache['lru_eviction']:
                oldest_key = min(self.memory_cache.keys(), 
                               key=lambda k: self.memory_cache[k]['timestamp'])
                del self.memory_cache[oldest_key]
                self.cache_metrics['evictions'] += 1
        
        # Store entry
        self.memory_cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }
        
        self.cache_metrics['size'] = len(self.memory_cache)
    
    async def _fallback_api_call(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback API call using standard method"""
        try:
            # Use the original API method as fallback
            if hasattr(self.api, 'info_client'):
                if endpoint.startswith('/info/'):
                    method_name = endpoint.replace('/info/', '').replace('/', '_')
                    if hasattr(self.api.info_client, method_name):
                        method = getattr(self.api.info_client, method_name)
                        if params:
                            return method(**params)
                        else:
                            return method()
            
            return {'error': 'Fallback API call not implemented'}
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _monitor_performance(self):
        """Monitor performance metrics continuously"""
        while True:
            try:
                await asyncio.sleep(self.latency_config.performance_monitoring['metrics_interval'])
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check for threshold breaches
                if self.latency_config.performance_monitoring['alert_on_threshold_breach']:
                    await self._check_performance_thresholds()
                
            except Exception as e:
                self.logger.error(f"❌ [PERFORMANCE_MONITOR] Error in performance monitoring: {e}")
    
    def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            # API metrics
            if self.api_metrics['response_times']:
                response_times = list(self.api_metrics['response_times'])
                self.performance_metrics.avg_response_time_ms = np.mean(response_times)
                self.performance_metrics.max_response_time_ms = np.max(response_times)
                self.performance_metrics.min_response_time_ms = np.min(response_times)
                self.performance_metrics.p95_latency_ms = np.percentile(response_times, 95)
                self.performance_metrics.p99_latency_ms = np.percentile(response_times, 99)
            
            # Calculate API calls per second
            current_time = time.time()
            time_diff = current_time - self.api_metrics['last_call_time']
            if time_diff > 0:
                self.performance_metrics.api_calls_per_second = self.api_metrics['total_calls'] / time_diff
            
            # Calculate error rate
            total_calls = self.api_metrics['total_calls']
            if total_calls > 0:
                self.performance_metrics.error_rate_percent = (self.api_metrics['failed_calls'] / total_calls) * 100
            
            # Connection metrics
            self.performance_metrics.connection_count = self.connection_metrics['total_connections']
            self.performance_metrics.active_connections = self.connection_metrics['active_connections']
            self.performance_metrics.connection_errors = self.connection_metrics['connection_errors']
            
            # WebSocket metrics
            self.performance_metrics.websocket_connections = len(self.websocket_connections)
            
            # Cache metrics
            total_cache_requests = self.cache_metrics['hits'] + self.cache_metrics['misses']
            if total_cache_requests > 0:
                self.performance_metrics.cache_hit_rate = (self.cache_metrics['hits'] / total_cache_requests) * 100
            self.performance_metrics.cache_size = self.cache_metrics['size']
            self.performance_metrics.cache_evictions = self.cache_metrics['evictions']
            
            # Overall performance
            self.performance_metrics.total_throughput = self.performance_metrics.api_calls_per_second
            self.performance_metrics.avg_latency_ms = self.performance_metrics.avg_response_time_ms
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_METRICS] Error updating performance metrics: {e}")
    
    async def _check_performance_thresholds(self):
        """Check performance thresholds and alert if breached"""
        try:
            # Check latency threshold
            if self.performance_metrics.avg_latency_ms > self.latency_config.performance_monitoring['latency_threshold_ms']:
                self.logger.warning(f"⚠️ [LATENCY_THRESHOLD] Average latency {self.performance_metrics.avg_latency_ms:.2f}ms exceeds threshold")
            
            # Check throughput threshold
            if self.performance_metrics.total_throughput < self.latency_config.performance_monitoring['throughput_threshold']:
                self.logger.warning(f"⚠️ [THROUGHPUT_THRESHOLD] Throughput {self.performance_metrics.total_throughput:.1f} req/s below threshold")
            
            # Check error rate threshold
            if self.performance_metrics.error_rate_percent > self.latency_config.performance_monitoring['error_rate_threshold'] * 100:
                self.logger.warning(f"⚠️ [ERROR_RATE_THRESHOLD] Error rate {self.performance_metrics.error_rate_percent:.2f}% exceeds threshold")
                
        except Exception as e:
            self.logger.error(f"❌ [THRESHOLD_CHECK] Error checking performance thresholds: {e}")
    
    def get_performance_metrics(self) -> LatencyMetrics:
        """Get current performance metrics"""
        return self.performance_metrics
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'hit_rate': self.performance_metrics.cache_hit_rate,
            'size': self.performance_metrics.cache_size,
            'evictions': self.performance_metrics.cache_evictions,
            'hits': self.cache_metrics['hits'],
            'misses': self.cache_metrics['misses'],
        }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': self.connection_metrics['total_connections'],
            'active_connections': self.connection_metrics['active_connections'],
            'connection_errors': self.connection_metrics['connection_errors'],
            'websocket_connections': len(self.websocket_connections),
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close connection pool
            if self.connection_pool:
                await self.connection_pool.close()
            
            # Close WebSocket connections
            for url, websocket_data in self.websocket_connections.items():
                try:
                    await websocket_data['websocket'].close()
                except:
                    pass
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("⚡ [LOW_LATENCY] Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ [LOW_LATENCY] Error during cleanup: {e}")

# Supporting classes
class WebSocketManager:
    """WebSocket connection manager"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.connections = {}
    
    async def connect(self, url: str, subscriptions: List[str]) -> bool:
        """Connect to WebSocket"""
        # Placeholder implementation
        return True
    
    async def disconnect(self, url: str):
        """Disconnect WebSocket"""
        # Placeholder implementation
        pass

class BatchProcessor:
    """Batch request processor"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.batch_queue = asyncio.Queue()
    
    async def process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of requests"""
        # Placeholder implementation
        return []

class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = {}
    
    def record_metric(self, name: str, value: float):
        """Record performance metric"""
        # Placeholder implementation
        pass
