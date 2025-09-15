"""
🤖 AUTOMATED EXECUTION MANAGER
"The strategy is the brain; I am the steady hand that carries out its will."

This module implements robust automated execution management:
- Order lifecycle state machine
- Error handling and recovery
- Retry mechanisms with exponential backoff
- Order confirmation and tracking
- Position management
- Execution quality monitoring
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import json
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

class ExecutionPriority(Enum):
    """Execution priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    size: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    average_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class ExecutionResult:
    """Execution result data structure"""
    order_id: str
    success: bool
    filled_size: float = 0.0
    average_price: float = 0.0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    size: float
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: float = field(default_factory=time.time)

class OrderStateMachine:
    """State machine for order lifecycle management"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.state_transitions = {
            OrderStatus.PENDING: [OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.ERROR],
            OrderStatus.SUBMITTED: [OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.ERROR],
            OrderStatus.PARTIALLY_FILLED: [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.ERROR],
            OrderStatus.FILLED: [],  # Terminal state
            OrderStatus.CANCELLED: [],  # Terminal state
            OrderStatus.REJECTED: [],  # Terminal state
            OrderStatus.EXPIRED: [],  # Terminal state
            OrderStatus.ERROR: [OrderStatus.PENDING, OrderStatus.CANCELLED]  # Can retry or cancel
        }
    
    def can_transition(self, current_status: OrderStatus, new_status: OrderStatus) -> bool:
        """Check if transition from current to new status is valid"""
        return new_status in self.state_transitions.get(current_status, [])
    
    def transition(self, order: Order, new_status: OrderStatus, 
                   error_message: Optional[str] = None) -> bool:
        """Transition order to new status"""
        try:
            if not self.can_transition(order.status, new_status):
                self.logger.warning(f"Invalid transition from {order.status} to {new_status} for order {order.order_id}")
                return False
            
            old_status = order.status
            order.status = new_status
            order.updated_at = time.time()
            
            if error_message:
                order.error_message = error_message
            
            self.logger.info(f"Order {order.order_id} transitioned from {old_status} to {new_status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error transitioning order {order.order_id}: {e}")
            return False

class AutomatedExecutionManager:
    """
    Automated Execution Manager - Master of Order Execution
    
    This class manages the complete order execution lifecycle:
    1. Order lifecycle state machine
    2. Error handling and recovery
    3. Retry mechanisms with exponential backoff
    4. Order confirmation and tracking
    5. Position management
    6. Execution quality monitoring
    """
    
    def __init__(self, api_client, config: Dict[str, Any], logger=None):
        self.api = api_client
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Execution configuration
        self.execution_config = {
            'max_retries': 3,
            'retry_delay': 1.0,  # seconds
            'max_retry_delay': 30.0,  # seconds
            'order_timeout': 300.0,  # 5 minutes
            'position_update_interval': 1.0,  # seconds
            'execution_timeout': 30.0,  # seconds
            'batch_size': 10,
            'max_concurrent_orders': 5
        }
        
        # State management
        self.state_machine = OrderStateMachine(logger)
        self.active_orders: Dict[str, Order] = {}
        self.order_history: deque = deque(maxlen=10000)
        self.positions: Dict[str, Position] = {}
        
        # Execution tracking
        self.execution_queue = queue.PriorityQueue()
        self.execution_results: Dict[str, ExecutionResult] = {}
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'average_execution_time': 0.0,
            'total_volume_executed': 0.0,
            'total_slippage': 0.0
        }
        
        # Threading
        self.execution_thread = None
        self.position_thread = None
        self.running = False
        
        # Callbacks
        self.order_callbacks: Dict[str, List[Callable]] = {
            'on_order_filled': [],
            'on_order_cancelled': [],
            'on_order_rejected': [],
            'on_position_updated': []
        }
        
        # Initialize execution system
        self._initialize_execution_system()
    
    def _initialize_execution_system(self):
        """Initialize the execution system"""
        try:
            self.logger.info("Initializing automated execution manager...")
            
            # Start execution thread
            self.execution_thread = threading.Thread(
                target=self._execution_loop,
                daemon=True,
                name="execution_manager"
            )
            self.execution_thread.start()
            
            # Start position monitoring thread
            self.position_thread = threading.Thread(
                target=self._position_monitoring_loop,
                daemon=True,
                name="position_monitor"
            )
            self.position_thread.start()
            
            self.running = True
            self.logger.info("Automated execution manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing execution system: {e}")
    
    def _execution_loop(self):
        """Main execution loop"""
        try:
            while self.running:
                try:
                    # Get next order from queue
                    priority, order = self.execution_queue.get(timeout=1.0)
                    
                    # Execute order
                    asyncio.run(self._execute_order(order))
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in execution loop: {e}")
                    time.sleep(1.0)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in execution loop: {e}")
    
    def _position_monitoring_loop(self):
        """Position monitoring loop"""
        try:
            while self.running:
                try:
                    # Update positions
                    asyncio.run(self._update_positions())
                    
                    # Sleep for update interval
                    time.sleep(self.execution_config['position_update_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Error in position monitoring loop: {e}")
                    time.sleep(5.0)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in position monitoring loop: {e}")
    
    async def submit_order(self, symbol: str, side: str, order_type: OrderType,
                          size: float, price: Optional[float] = None,
                          priority: ExecutionPriority = ExecutionPriority.NORMAL,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit order for execution"""
        try:
            # Create order
            order_id = str(uuid.uuid4())
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                size=size,
                price=price,
                priority=priority,
                metadata=metadata or {}
            )
            
            # Add to active orders
            self.active_orders[order_id] = order
            
            # Add to execution queue
            self.execution_queue.put((priority.value, order))
            
            # Update metrics
            self.execution_metrics['total_orders'] += 1
            
            self.logger.info(f"Order {order_id} submitted for execution: {side} {size} {symbol} @ {price}")
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            raise
    
    async def _execute_order(self, order: Order):
        """Execute a single order"""
        try:
            start_time = time.time()
            
            # Check if order is expired
            if order.expires_at and time.time() > order.expires_at:
                self.state_machine.transition(order, OrderStatus.EXPIRED)
                self._handle_order_completion(order)
                return
            
            # Transition to submitted
            self.state_machine.transition(order, OrderStatus.SUBMITTED)
            
            # Execute order based on type
            if order.order_type == OrderType.MARKET:
                result = await self._execute_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                result = await self._execute_limit_order(order)
            elif order.order_type == OrderType.POST_ONLY:
                result = await self._execute_post_only_order(order)
            elif order.order_type == OrderType.IOC:
                result = await self._execute_ioc_order(order)
            elif order.order_type == OrderType.FOK:
                result = await self._execute_fok_order(order)
            else:
                raise ValueError(f"Unknown order type: {order.order_type}")
            
            # Update order with result
            if result.success:
                order.filled_size = result.filled_size
                order.average_price = result.average_price
                
                if result.filled_size >= order.size:
                    self.state_machine.transition(order, OrderStatus.FILLED)
                else:
                    self.state_machine.transition(order, OrderStatus.PARTIALLY_FILLED)
                
                # Update metrics
                self.execution_metrics['successful_orders'] += 1
                self.execution_metrics['total_volume_executed'] += result.filled_size
                
            else:
                # Handle failure
                if order.retry_count < order.max_retries:
                    order.retry_count += 1
                    self.state_machine.transition(order, OrderStatus.ERROR, result.error_message)
                    
                    # Schedule retry with exponential backoff
                    retry_delay = min(
                        self.execution_config['retry_delay'] * (2 ** order.retry_count),
                        self.execution_config['max_retry_delay']
                    )
                    
                    await asyncio.sleep(retry_delay)
                    self.execution_queue.put((order.priority.value, order))
                    
                else:
                    self.state_machine.transition(order, OrderStatus.REJECTED, result.error_message)
                    self.execution_metrics['failed_orders'] += 1
            
            # Update execution time
            execution_time = time.time() - start_time
            self.execution_metrics['average_execution_time'] = (
                (self.execution_metrics['average_execution_time'] * (self.execution_metrics['total_orders'] - 1) + execution_time) /
                self.execution_metrics['total_orders']
            )
            
            # Store execution result
            self.execution_results[order.order_id] = result
            
            # Handle order completion
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                self._handle_order_completion(order)
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.order_id}: {e}")
            self.state_machine.transition(order, OrderStatus.ERROR, str(e))
            self.execution_metrics['failed_orders'] += 1
    
    async def _execute_market_order(self, order: Order) -> ExecutionResult:
        """Execute market order"""
        try:
            # Place market order
            result = await self.api.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type='market',
                size=order.size
            )
            
            if result.get('success'):
                return ExecutionResult(
                    order_id=order.order_id,
                    success=True,
                    filled_size=order.size,
                    average_price=result.get('price', 0.0),
                    execution_time=time.time() - order.created_at
                )
            else:
                return ExecutionResult(
                    order_id=order.order_id,
                    success=False,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_limit_order(self, order: Order) -> ExecutionResult:
        """Execute limit order"""
        try:
            # Place limit order
            result = await self.api.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type='limit',
                size=order.size,
                price=order.price
            )
            
            if result.get('success'):
                # For limit orders, we need to monitor for fills
                return await self._monitor_limit_order_fills(order, result.get('order_id'))
            else:
                return ExecutionResult(
                    order_id=order.order_id,
                    success=False,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_post_only_order(self, order: Order) -> ExecutionResult:
        """Execute post-only order"""
        try:
            # Place post-only order
            result = await self.api.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type='post_only',
                size=order.size,
                price=order.price
            )
            
            if result.get('success'):
                return await self._monitor_limit_order_fills(order, result.get('order_id'))
            else:
                return ExecutionResult(
                    order_id=order.order_id,
                    success=False,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_ioc_order(self, order: Order) -> ExecutionResult:
        """Execute Immediate or Cancel order"""
        try:
            # Place IOC order
            result = await self.api.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type='ioc',
                size=order.size,
                price=order.price
            )
            
            if result.get('success'):
                # IOC orders are filled immediately or cancelled
                return ExecutionResult(
                    order_id=order.order_id,
                    success=True,
                    filled_size=result.get('filled_size', 0.0),
                    average_price=result.get('price', 0.0),
                    execution_time=time.time() - order.created_at
                )
            else:
                return ExecutionResult(
                    order_id=order.order_id,
                    success=False,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_fok_order(self, order: Order) -> ExecutionResult:
        """Execute Fill or Kill order"""
        try:
            # Place FOK order
            result = await self.api.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type='fok',
                size=order.size,
                price=order.price
            )
            
            if result.get('success'):
                # FOK orders are filled completely or cancelled
                return ExecutionResult(
                    order_id=order.order_id,
                    success=True,
                    filled_size=order.size,
                    average_price=result.get('price', 0.0),
                    execution_time=time.time() - order.created_at
                )
            else:
                return ExecutionResult(
                    order_id=order.order_id,
                    success=False,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                error_message=str(e)
            )
    
    async def _monitor_limit_order_fills(self, order: Order, exchange_order_id: str) -> ExecutionResult:
        """Monitor limit order for fills"""
        try:
            start_time = time.time()
            timeout = self.execution_config['execution_timeout']
            
            while time.time() - start_time < timeout:
                # Check order status
                order_status = await self.api.get_order_status(exchange_order_id)
                
                if order_status.get('status') == 'filled':
                    return ExecutionResult(
                        order_id=order.order_id,
                        success=True,
                        filled_size=order_status.get('filled_size', 0.0),
                        average_price=order_status.get('average_price', 0.0),
                        execution_time=time.time() - order.created_at
                    )
                elif order_status.get('status') == 'cancelled':
                    return ExecutionResult(
                        order_id=order.order_id,
                        success=False,
                        error_message='Order cancelled'
                    )
                elif order_status.get('status') == 'rejected':
                    return ExecutionResult(
                        order_id=order.order_id,
                        success=False,
                        error_message='Order rejected'
                    )
                
                # Wait before checking again
                await asyncio.sleep(0.1)
            
            # Timeout reached
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                error_message='Execution timeout'
            )
            
        except Exception as e:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                error_message=str(e)
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            order = self.active_orders[order_id]
            
            # Cancel order on exchange
            result = await self.api.cancel_order(order_id)
            
            if result.get('success'):
                self.state_machine.transition(order, OrderStatus.CANCELLED)
                self._handle_order_completion(order)
                return True
            else:
                self.logger.error(f"Failed to cancel order {order_id}: {result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def _update_positions(self):
        """Update position information"""
        try:
            # Get current positions from exchange
            positions = await self.api.get_positions()
            
            for position_data in positions:
                symbol = position_data.get('symbol')
                if not symbol:
                    continue
                
                # Update or create position
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position.size = position_data.get('size', 0.0)
                    position.average_price = position_data.get('average_price', 0.0)
                    position.unrealized_pnl = position_data.get('unrealized_pnl', 0.0)
                    position.realized_pnl = position_data.get('realized_pnl', 0.0)
                    position.last_updated = time.time()
                else:
                    position = Position(
                        symbol=symbol,
                        size=position_data.get('size', 0.0),
                        average_price=position_data.get('average_price', 0.0),
                        unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
                        realized_pnl=position_data.get('realized_pnl', 0.0)
                    )
                    self.positions[symbol] = position
                
                # Trigger position update callbacks
                self._trigger_callbacks('on_position_updated', position)
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _handle_order_completion(self, order: Order):
        """Handle order completion"""
        try:
            # Move to history
            self.order_history.append(order)
            
            # Remove from active orders
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            
            # Trigger appropriate callbacks
            if order.status == OrderStatus.FILLED:
                self._trigger_callbacks('on_order_filled', order)
            elif order.status == OrderStatus.CANCELLED:
                self._trigger_callbacks('on_order_cancelled', order)
            elif order.status == OrderStatus.REJECTED:
                self._trigger_callbacks('on_order_rejected', order)
            
            self.logger.info(f"Order {order.order_id} completed with status {order.status}")
            
        except Exception as e:
            self.logger.error(f"Error handling order completion: {e}")
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger event callbacks"""
        try:
            if event in self.order_callbacks:
                for callback in self.order_callbacks[event]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in callback for {event}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error triggering callbacks: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event in self.order_callbacks:
            self.order_callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback"""
        if event in self.order_callbacks and callback in self.order_callbacks[event]:
            self.order_callbacks[event].remove(callback)
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.active_orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.positions.copy()
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        return self.execution_metrics.copy()
    
    def get_execution_result(self, order_id: str) -> Optional[ExecutionResult]:
        """Get execution result for order"""
        return self.execution_results.get(order_id)
    
    def shutdown(self):
        """Shutdown execution manager"""
        try:
            self.running = False
            
            # Cancel all active orders
            for order in self.active_orders.values():
                asyncio.run(self.cancel_order(order.order_id))
            
            self.logger.info("Automated execution manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

