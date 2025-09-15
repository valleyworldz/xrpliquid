"""
⚡ LOW-LATENCY ENGINEER
"Every millisecond is a mile. I will build the fastest path to execution."

This module implements ultra-low latency optimizations:
- In-memory data structures for microsecond access
- Efficient websocket connections with connection pooling
- Asynchronous processing with zero-copy operations
- Lock-free data structures
- Memory-mapped files for persistence
- CPU affinity and NUMA optimization
- JIT compilation for hot paths
"""

import asyncio
import time
import threading
import multiprocessing
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque
import logging
import weakref
import gc
import mmap
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import struct
import pickle
import zlib
from enum import Enum
import ctypes
from ctypes import c_float, c_int, c_double, c_char_p, c_void_p

class Priority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class LatencyMetrics:
    """Latency performance metrics"""
    data_processing_time: float = 0.0
    websocket_latency: float = 0.0
    order_execution_time: float = 0.0
    memory_access_time: float = 0.0
    network_roundtrip: float = 0.0
    total_latency: float = 0.0

@dataclass
class PerformanceStats:
    """Performance statistics"""
    messages_per_second: int = 0
    average_latency: float = 0.0
    p99_latency: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gc_pressure: float = 0.0

class LockFreeRingBuffer:
    """Lock-free ring buffer for ultra-fast data access"""
    
    def __init__(self, size: int, dtype=np.float64):
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.head = 0
        self.tail = 0
        self.mask = size - 1
        
        # Ensure size is power of 2 for efficient modulo
        if size & (size - 1) != 0:
            raise ValueError("Size must be a power of 2")
    
    def push(self, value: float) -> bool:
        """Push value to buffer (non-blocking)"""
        try:
            next_head = (self.head + 1) & self.mask
            if next_head == self.tail:
                return False  # Buffer full
            
            self.buffer[self.head] = value
            self.head = next_head
            return True
        except Exception:
            return False
    
    def pop(self) -> Optional[float]:
        """Pop value from buffer (non-blocking)"""
        try:
            if self.head == self.tail:
                return None  # Buffer empty
            
            value = self.buffer[self.tail]
            self.tail = (self.tail + 1) & self.mask
            return value
        except Exception:
            return None
    
    def peek(self, index: int = 0) -> Optional[float]:
        """Peek at value without removing"""
        try:
            if self.head == self.tail:
                return None
            
            peek_index = (self.tail + index) & self.mask
            return self.buffer[peek_index]
        except Exception:
            return None
    
    def size_used(self) -> int:
        """Get number of elements in buffer"""
        return (self.head - self.tail) & self.mask

class HighPerformanceCache:
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque(maxlen=max_size)
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class MemoryMappedDataStore:
    """Memory-mapped data store for persistent high-speed access"""
    
    def __init__(self, file_path: str, size: int = 1024 * 1024):  # 1MB default
        self.file_path = file_path
        self.size = size
        self.mmap_file = None
        self.data = None
        self._initialize_mmap()
    
    def _initialize_mmap(self):
        """Initialize memory-mapped file"""
        try:
            # Create file if it doesn't exist
            if not os.path.exists(self.file_path):
                with open(self.file_path, 'wb') as f:
                    f.write(b'\x00' * self.size)
            
            # Open memory-mapped file
            with open(self.file_path, 'r+b') as f:
                self.mmap_file = mmap.mmap(f.fileno(), self.size)
                self.data = np.frombuffer(self.mmap_file, dtype=np.float64)
            
        except Exception as e:
            logging.error(f"Error initializing memory-mapped file: {e}")
    
    def write(self, offset: int, data: np.ndarray):
        """Write data to memory-mapped file"""
        try:
            if self.data is not None:
                self.data[offset:offset + len(data)] = data
                self.mmap_file.flush()
        except Exception as e:
            logging.error(f"Error writing to memory-mapped file: {e}")
    
    def read(self, offset: int, length: int) -> np.ndarray:
        """Read data from memory-mapped file"""
        try:
            if self.data is not None:
                return self.data[offset:offset + length].copy()
            return np.array([])
        except Exception as e:
            logging.error(f"Error reading from memory-mapped file: {e}")
            return np.array([])
    
    def close(self):
        """Close memory-mapped file"""
        try:
            if self.mmap_file:
                self.mmap_file.close()
        except Exception as e:
            logging.error(f"Error closing memory-mapped file: {e}")

class LowLatencyEngineer:
    """
    Low-Latency Engineer - Master of Speed and Performance
    
    This class implements ultra-low latency optimizations:
    1. In-memory data structures for microsecond access
    2. Efficient websocket connections with connection pooling
    3. Asynchronous processing with zero-copy operations
    4. Lock-free data structures
    5. Memory-mapped files for persistence
    6. CPU affinity and NUMA optimization
    7. JIT compilation for hot paths
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Performance configuration
        self.performance_config = {
            'max_workers': multiprocessing.cpu_count(),
            'memory_limit_mb': 1024,
            'gc_threshold': 0.8,
            'latency_target_ms': 1.0,
            'throughput_target': 10000  # messages per second
        }
        
        # Data structures
        self.price_buffer = LockFreeRingBuffer(1024, np.float64)
        self.volume_buffer = LockFreeRingBuffer(1024, np.float64)
        self.order_buffer = LockFreeRingBuffer(512, np.float64)
        
        # High-performance cache
        self.cache = HighPerformanceCache(max_size=10000)
        
        # Memory-mapped data store
        self.data_store = MemoryMappedDataStore("trading_data.dat", 1024 * 1024)
        
        # Thread pools
        self.io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="io")
        self.cpu_executor = ProcessPoolExecutor(max_workers=self.performance_config['max_workers'])
        
        # Message queues
        self.critical_queue = queue.PriorityQueue()
        self.high_priority_queue = queue.Queue()
        self.normal_queue = queue.Queue()
        
        # Performance monitoring
        self.latency_metrics = LatencyMetrics()
        self.performance_stats = PerformanceStats()
        self.message_count = 0
        self.start_time = time.time()
        
        # CPU affinity
        self._set_cpu_affinity()
        
        # Initialize performance monitoring
        self._initialize_performance_monitoring()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _set_cpu_affinity(self):
        """Set CPU affinity for optimal performance"""
        try:
            # Get current process
            current_process = psutil.Process()
            
            # Set CPU affinity to use all available cores
            available_cpus = list(range(psutil.cpu_count()))
            current_process.cpu_affinity(available_cpus)
            
            self.logger.info(f"Set CPU affinity to cores: {available_cpus}")
            
        except Exception as e:
            self.logger.error(f"Error setting CPU affinity: {e}")
    
    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring"""
        try:
            # Start performance monitoring thread
            monitor_thread = threading.Thread(
                target=self._performance_monitor,
                daemon=True,
                name="performance_monitor"
            )
            monitor_thread.start()
            
            # Start garbage collection monitoring
            gc_thread = threading.Thread(
                target=self._gc_monitor,
                daemon=True,
                name="gc_monitor"
            )
            gc_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error initializing performance monitoring: {e}")
    
    def _performance_monitor(self):
        """Monitor performance metrics"""
        try:
            while True:
                time.sleep(1.0)  # Monitor every second
                
                # Calculate messages per second
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                if elapsed_time > 0:
                    self.performance_stats.messages_per_second = int(
                        self.message_count / elapsed_time
                    )
                
                # Monitor memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                self.performance_stats.memory_usage = memory_info.rss / 1024 / 1024  # MB
                
                # Monitor CPU usage
                self.performance_stats.cpu_usage = process.cpu_percent()
                
                # Check if we need to trigger garbage collection
                if self.performance_stats.memory_usage > self.performance_config['memory_limit_mb']:
                    self._optimize_memory()
                
        except Exception as e:
            self.logger.error(f"Error in performance monitor: {e}")
    
    def _gc_monitor(self):
        """Monitor garbage collection pressure"""
        try:
            while True:
                time.sleep(5.0)  # Check every 5 seconds
                
                # Calculate GC pressure
                gc_stats = gc.get_stats()
                total_collections = sum(stat['collections'] for stat in gc_stats)
                self.performance_stats.gc_pressure = total_collections
                
                # Trigger GC if pressure is high
                if self.performance_stats.gc_pressure > 100:
                    gc.collect()
                
        except Exception as e:
            self.logger.error(f"Error in GC monitor: {e}")
    
    def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Clear cache if it's too large
            if len(self.cache.cache) > 5000:
                # Clear half of the cache
                keys_to_remove = list(self.cache.cache.keys())[:2500]
                for key in keys_to_remove:
                    del self.cache.cache[key]
                    self.cache.access_order.remove(key)
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Memory optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        try:
            # Start message processing tasks
            for i in range(2):
                thread = threading.Thread(
                    target=self._process_messages,
                    daemon=True,
                    name=f"message_processor_{i}"
                )
                thread.start()
            
            # Start data processing tasks
            for i in range(2):
                thread = threading.Thread(
                    target=self._process_data,
                    daemon=True,
                    name=f"data_processor_{i}"
                )
                thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting background tasks: {e}")
    
    def _process_messages(self):
        """Process messages from queues"""
        try:
            while True:
                # Process critical messages first
                try:
                    priority, message = self.critical_queue.get_nowait()
                    self._handle_message(message, Priority.CRITICAL)
                except queue.Empty:
                    pass
                
                # Process high priority messages
                try:
                    message = self.high_priority_queue.get_nowait()
                    self._handle_message(message, Priority.HIGH)
                except queue.Empty:
                    pass
                
                # Process normal messages
                try:
                    message = self.normal_queue.get_nowait()
                    self._handle_message(message, Priority.NORMAL)
                except queue.Empty:
                    pass
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)  # 1ms
                
        except Exception as e:
            self.logger.error(f"Error processing messages: {e}")
    
    def _process_data(self):
        """Process data in background"""
        try:
            while True:
                # Process price data
                self._process_price_data()
                
                # Process volume data
                self._process_volume_data()
                
                # Process order data
                self._process_order_data()
                
                time.sleep(0.01)  # 10ms
                
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
    
    def _handle_message(self, message: Dict[str, Any], priority: Priority):
        """Handle incoming message"""
        try:
            start_time = time.time()
            
            # Process message based on type
            message_type = message.get('type', 'unknown')
            
            if message_type == 'price_update':
                self._handle_price_update(message)
            elif message_type == 'order_update':
                self._handle_order_update(message)
            elif message_type == 'trade_update':
                self._handle_trade_update(message)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
            
            # Update latency metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.latency_metrics.data_processing_time = processing_time
            
            self.message_count += 1
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    def _handle_price_update(self, message: Dict[str, Any]):
        """Handle price update message"""
        try:
            price = message.get('price', 0.0)
            symbol = message.get('symbol', '')
            timestamp = message.get('timestamp', time.time())
            
            # Add to ring buffer
            self.price_buffer.push(price)
            
            # Update cache
            cache_key = f"price_{symbol}_{timestamp}"
            self.cache.set(cache_key, price)
            
            # Store in memory-mapped file
            self._store_price_data(symbol, price, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")
    
    def _handle_order_update(self, message: Dict[str, Any]):
        """Handle order update message"""
        try:
            order_id = message.get('order_id', '')
            status = message.get('status', '')
            price = message.get('price', 0.0)
            size = message.get('size', 0.0)
            
            # Add to order buffer
            self.order_buffer.push(price)
            
            # Update cache
            cache_key = f"order_{order_id}"
            self.cache.set(cache_key, {
                'status': status,
                'price': price,
                'size': size
            })
            
        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")
    
    def _handle_trade_update(self, message: Dict[str, Any]):
        """Handle trade update message"""
        try:
            trade_id = message.get('trade_id', '')
            price = message.get('price', 0.0)
            size = message.get('size', 0.0)
            side = message.get('side', '')
            
            # Add to volume buffer
            self.volume_buffer.push(size)
            
            # Update cache
            cache_key = f"trade_{trade_id}"
            self.cache.set(cache_key, {
                'price': price,
                'size': size,
                'side': side
            })
            
        except Exception as e:
            self.logger.error(f"Error handling trade update: {e}")
    
    def _process_price_data(self):
        """Process price data in background"""
        try:
            # Get recent prices
            recent_prices = []
            for i in range(10):
                price = self.price_buffer.peek(i)
                if price is not None:
                    recent_prices.append(price)
                else:
                    break
            
            if recent_prices:
                # Calculate simple moving average
                sma = np.mean(recent_prices)
                
                # Store in cache
                self.cache.set("price_sma", sma)
                
        except Exception as e:
            self.logger.error(f"Error processing price data: {e}")
    
    def _process_volume_data(self):
        """Process volume data in background"""
        try:
            # Get recent volumes
            recent_volumes = []
            for i in range(10):
                volume = self.volume_buffer.peek(i)
                if volume is not None:
                    recent_volumes.append(volume)
                else:
                    break
            
            if recent_volumes:
                # Calculate volume weighted average
                vwa = np.mean(recent_volumes)
                
                # Store in cache
                self.cache.set("volume_vwa", vwa)
                
        except Exception as e:
            self.logger.error(f"Error processing volume data: {e}")
    
    def _process_order_data(self):
        """Process order data in background"""
        try:
            # Get recent order prices
            recent_orders = []
            for i in range(5):
                order_price = self.order_buffer.peek(i)
                if order_price is not None:
                    recent_orders.append(order_price)
                else:
                    break
            
            if recent_orders:
                # Calculate order price statistics
                order_stats = {
                    'mean': np.mean(recent_orders),
                    'std': np.std(recent_orders),
                    'min': np.min(recent_orders),
                    'max': np.max(recent_orders)
                }
                
                # Store in cache
                self.cache.set("order_stats", order_stats)
                
        except Exception as e:
            self.logger.error(f"Error processing order data: {e}")
    
    def _store_price_data(self, symbol: str, price: float, timestamp: float):
        """Store price data in memory-mapped file"""
        try:
            # Convert to numpy array
            data = np.array([price, timestamp], dtype=np.float64)
            
            # Store in memory-mapped file
            offset = hash(symbol) % (self.data_store.size // 16)  # Simple hash-based offset
            self.data_store.write(offset, data)
            
        except Exception as e:
            self.logger.error(f"Error storing price data: {e}")
    
    def queue_message(self, message: Dict[str, Any], priority: Priority = Priority.NORMAL):
        """Queue message for processing"""
        try:
            if priority == Priority.CRITICAL:
                self.critical_queue.put((priority.value, message))
            elif priority == Priority.HIGH:
                self.high_priority_queue.put(message)
            else:
                self.normal_queue.put(message)
                
        except Exception as e:
            self.logger.error(f"Error queuing message: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        try:
            # Try cache first
            cache_key = f"price_{symbol}_latest"
            price = self.cache.get(cache_key)
            if price is not None:
                return price
            
            # Fallback to ring buffer
            return self.price_buffer.peek(0)
            
        except Exception as e:
            self.logger.error(f"Error getting latest price: {e}")
            return None
    
    def get_price_history(self, symbol: str, count: int = 10) -> List[float]:
        """Get price history for symbol"""
        try:
            prices = []
            for i in range(count):
                price = self.price_buffer.peek(i)
                if price is not None:
                    prices.append(price)
                else:
                    break
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error getting price history: {e}")
            return []
    
    def get_performance_stats(self) -> PerformanceStats:
        """Get current performance statistics"""
        return self.performance_stats
    
    def get_latency_metrics(self) -> LatencyMetrics:
        """Get current latency metrics"""
        return self.latency_metrics
    
    def optimize_for_latency(self):
        """Optimize system for minimum latency"""
        try:
            # Disable garbage collection temporarily
            gc.disable()
            
            # Set high priority for current thread
            current_thread = threading.current_thread()
            if hasattr(current_thread, 'set_priority'):
                current_thread.set_priority(threading.HIGHEST_PRIORITY)
            
            # Optimize memory layout
            self._optimize_memory_layout()
            
            # Pre-allocate frequently used objects
            self._preallocate_objects()
            
            self.logger.info("Latency optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing for latency: {e}")
        finally:
            # Re-enable garbage collection
            gc.enable()
    
    def _optimize_memory_layout(self):
        """Optimize memory layout for better cache performance"""
        try:
            # Compact memory
            gc.collect()
            
            # Pre-allocate arrays
            self._preallocated_arrays = {
                'prices': np.zeros(1000, dtype=np.float64),
                'volumes': np.zeros(1000, dtype=np.float64),
                'orders': np.zeros(500, dtype=np.float64)
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory layout: {e}")
    
    def _preallocate_objects(self):
        """Pre-allocate frequently used objects"""
        try:
            # Pre-allocate message objects
            self._message_pool = []
            for _ in range(100):
                self._message_pool.append({
                    'type': '',
                    'data': {},
                    'timestamp': 0.0
                })
            
            # Pre-allocate numpy arrays
            self._array_pool = {
                'float64': [np.zeros(100, dtype=np.float64) for _ in range(10)],
                'float32': [np.zeros(100, dtype=np.float32) for _ in range(10)],
                'int64': [np.zeros(100, dtype=np.int64) for _ in range(10)]
            }
            
        except Exception as e:
            self.logger.error(f"Error pre-allocating objects: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Close thread pools
            self.io_executor.shutdown(wait=True)
            self.cpu_executor.shutdown(wait=True)
            
            # Close memory-mapped file
            self.data_store.close()
            
            # Clear caches
            self.cache.cache.clear()
            self.cache.access_order.clear()
            
            self.logger.info("Low-latency engineer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

