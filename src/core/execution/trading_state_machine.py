"""
🎯 TRADING STATE MACHINE
========================
Production-grade trading state machine with comprehensive observability.

States: IDLE→SIGNAL→PLACE→ACK→LIVE→FILL/REJECT→RECONCILE→ATTRIB
Emit structured JSON logs + Prometheus counters for full observability.
"""

import asyncio
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
import sys
from collections import deque

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger

class TradingState(Enum):
    """Trading state enumeration"""
    IDLE = "idle"
    SIGNAL = "signal"
    PLACE = "place"
    ACK = "ack"
    LIVE = "live"
    FILL = "fill"
    REJECT = "reject"
    RECONCILE = "reconcile"
    ATTRIB = "attrib"
    ERROR = "error"
    TIMEOUT = "timeout"

class SignalType(Enum):
    """Signal type enumeration"""
    BUY = "buy"
    SELL = "sell"
    FUNDING_ARB = "funding_arb"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    REBALANCE = "rebalance"
    EMERGENCY = "emergency"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    
    signal_id: str
    signal_type: SignalType
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    urgency: float  # 0-1, higher is more urgent
    confidence: float  # 0-1, signal confidence
    reason_code: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderExecution:
    """Order execution data structure"""
    
    order_id: str
    cloid: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_type: str
    time_in_force: str
    post_only: bool
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateTransition:
    """State transition data structure"""
    
    transition_id: str
    from_state: TradingState
    to_state: TradingState
    timestamp: float
    duration_ms: float
    signal_id: Optional[str] = None
    order_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    
    # Latency metrics
    p95_loop_ms: float = 0.0
    p99_loop_ms: float = 0.0
    avg_loop_ms: float = 0.0
    
    # Order metrics
    orders_placed: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    orders_rejected: int = 0
    
    # Fill metrics
    fills_total: int = 0
    fills_partial: int = 0
    fills_complete: int = 0
    
    # Performance metrics
    maker_ratio: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_fee_bps: float = 0.0
    total_fees_paid: float = 0.0
    total_rebates_received: float = 0.0
    
    # P&L metrics
    funding_pnl: float = 0.0
    net_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Error metrics
    errors_total: int = 0
    errors_timeout: int = 0
    errors_reject: int = 0
    errors_network: int = 0
    errors_risk: int = 0

class TradingStateMachine:
    """
    🎯 TRADING STATE MACHINE
    
    Production-grade trading state machine with comprehensive observability
    and structured logging for full system transparency.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or Logger()
        
        # State machine configuration
        self.state_config = {
            'max_signal_age_seconds': 60,      # Max age for signals
            'order_timeout_seconds': 30,       # Order timeout
            'reconcile_timeout_seconds': 10,   # Reconcile timeout
            'max_retries': 3,                  # Max retries per order
            'loop_interval_ms': 100,           # Main loop interval
        }
        
        # State machine state
        self.current_state = TradingState.IDLE
        self.current_signal: Optional[TradingSignal] = None
        self.current_order: Optional[OrderExecution] = None
        self.state_history: deque = deque(maxlen=1000)
        self.transition_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.loop_times: deque = deque(maxlen=1000)
        self.order_history: deque = deque(maxlen=1000)
        
        # Observability
        self.structured_logs: deque = deque(maxlen=10000)
        self.prometheus_counters: Dict[str, float] = {}
        
        # State machine flags
        self.is_running = False
        self.shutdown_requested = False
        
        self.logger.info("🎯 [STATE_MACHINE] Trading State Machine initialized")
        self.logger.info("🎯 [STATE_MACHINE] Comprehensive observability enabled")
    
    async def start_state_machine(self):
        """Start the trading state machine"""
        try:
            self.logger.info("🎯 [STATE_MACHINE] Starting trading state machine...")
            
            self.is_running = True
            self.current_state = TradingState.IDLE
            
            # Emit initial state
            await self._emit_state_transition(TradingState.IDLE, TradingState.IDLE, "startup")
            
            # Main state machine loop
            while self.is_running and not self.shutdown_requested:
                loop_start = time.time()
                
                try:
                    await self._state_machine_loop()
                except Exception as e:
                    await self._handle_state_error(e)
                
                # Track loop performance
                loop_duration = (time.time() - loop_start) * 1000
                self.loop_times.append(loop_duration)
                await self._update_performance_metrics()
                
                # Wait for next loop
                await asyncio.sleep(self.state_config['loop_interval_ms'] / 1000)
            
            self.logger.info("🎯 [STATE_MACHINE] Trading state machine stopped")
            
        except Exception as e:
            self.logger.error(f"❌ [STATE_MACHINE] Error in state machine: {e}")
            await self._handle_state_error(e)
    
    async def _state_machine_loop(self):
        """Main state machine loop"""
        try:
            # Execute current state
            if self.current_state == TradingState.IDLE:
                await self._handle_idle_state()
            elif self.current_state == TradingState.SIGNAL:
                await self._handle_signal_state()
            elif self.current_state == TradingState.PLACE:
                await self._handle_place_state()
            elif self.current_state == TradingState.ACK:
                await self._handle_ack_state()
            elif self.current_state == TradingState.LIVE:
                await self._handle_live_state()
            elif self.current_state == TradingState.FILL:
                await self._handle_fill_state()
            elif self.current_state == TradingState.REJECT:
                await self._handle_reject_state()
            elif self.current_state == TradingState.RECONCILE:
                await self._handle_reconcile_state()
            elif self.current_state == TradingState.ATTRIB:
                await self._handle_attrib_state()
            elif self.current_state == TradingState.ERROR:
                await self._handle_error_state()
            elif self.current_state == TradingState.TIMEOUT:
                await self._handle_timeout_state()
            
        except Exception as e:
            self.logger.error(f"❌ [STATE_LOOP] Error in state machine loop: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_idle_state(self):
        """Handle IDLE state - waiting for signals"""
        try:
            # Check for new signals
            signal = await self._get_next_signal()
            
            if signal:
                self.current_signal = signal
                await self._transition_to_state(TradingState.SIGNAL, signal_id=signal.signal_id)
            else:
                # Stay in IDLE state
                await self._emit_structured_log("idle_waiting", {
                    "state": self.current_state.value,
                    "timestamp": time.time(),
                })
                
        except Exception as e:
            self.logger.error(f"❌ [IDLE_STATE] Error in idle state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_signal_state(self):
        """Handle SIGNAL state - processing trading signal"""
        try:
            if not self.current_signal:
                await self._transition_to_state(TradingState.IDLE)
                return
            
            signal = self.current_signal
            
            # Validate signal
            if not await self._validate_signal(signal):
                await self._transition_to_state(TradingState.REJECT, 
                                              signal_id=signal.signal_id,
                                              error_message="Signal validation failed")
                return
            
            # Check signal age
            signal_age = time.time() - signal.timestamp
            if signal_age > self.state_config['max_signal_age_seconds']:
                await self._transition_to_state(TradingState.REJECT,
                                              signal_id=signal.signal_id,
                                              error_message="Signal too old")
                return
            
            # Emit signal processing log
            await self._emit_structured_log("signal_processing", {
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type.value,
                "symbol": signal.symbol,
                "side": signal.side,
                "quantity": signal.quantity,
                "price": signal.price,
                "urgency": signal.urgency,
                "confidence": signal.confidence,
                "signal_age_seconds": signal_age,
            })
            
            # Transition to PLACE state
            await self._transition_to_state(TradingState.PLACE, signal_id=signal.signal_id)
            
        except Exception as e:
            self.logger.error(f"❌ [SIGNAL_STATE] Error in signal state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_place_state(self):
        """Handle PLACE state - placing order"""
        try:
            if not self.current_signal:
                await self._transition_to_state(TradingState.IDLE)
                return
            
            signal = self.current_signal
            
            # Create order execution
            order = await self._create_order_execution(signal)
            self.current_order = order
            
            # Place order via API
            order_result = await self._place_order(order)
            
            if order_result['success']:
                order.order_id = order_result['order_id']
                await self._transition_to_state(TradingState.ACK, 
                                              signal_id=signal.signal_id,
                                              order_id=order.order_id)
            else:
                await self._transition_to_state(TradingState.REJECT,
                                              signal_id=signal.signal_id,
                                              order_id=order.order_id,
                                              error_message=order_result.get('error', 'Order placement failed'))
            
        except Exception as e:
            self.logger.error(f"❌ [PLACE_STATE] Error in place state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_ack_state(self):
        """Handle ACK state - waiting for order acknowledgment"""
        try:
            if not self.current_order:
                await self._transition_to_state(TradingState.IDLE)
                return
            
            order = self.current_order
            
            # Check for order acknowledgment
            ack_result = await self._check_order_acknowledgment(order)
            
            if ack_result['acknowledged']:
                await self._transition_to_state(TradingState.LIVE,
                                              signal_id=self.current_signal.signal_id if self.current_signal else None,
                                              order_id=order.order_id)
            elif ack_result['timeout']:
                await self._transition_to_state(TradingState.TIMEOUT,
                                              signal_id=self.current_signal.signal_id if self.current_signal else None,
                                              order_id=order.order_id)
            else:
                # Stay in ACK state, check again next loop
                await self._emit_structured_log("ack_waiting", {
                    "order_id": order.order_id,
                    "cloid": order.cloid,
                    "wait_time_seconds": time.time() - order.timestamp,
                })
                
        except Exception as e:
            self.logger.error(f"❌ [ACK_STATE] Error in ack state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_live_state(self):
        """Handle LIVE state - monitoring live order"""
        try:
            if not self.current_order:
                await self._transition_to_state(TradingState.IDLE)
                return
            
            order = self.current_order
            
            # Check order status
            order_status = await self._check_order_status(order)
            
            if order_status['status'] == 'filled':
                await self._transition_to_state(TradingState.FILL,
                                              signal_id=self.current_signal.signal_id if self.current_signal else None,
                                              order_id=order.order_id)
            elif order_status['status'] == 'rejected':
                await self._transition_to_state(TradingState.REJECT,
                                              signal_id=self.current_signal.signal_id if self.current_signal else None,
                                              order_id=order.order_id,
                                              error_message=order_status.get('error', 'Order rejected'))
            elif order_status['status'] == 'cancelled':
                await self._transition_to_state(TradingState.REJECT,
                                              signal_id=self.current_signal.signal_id if self.current_signal else None,
                                              order_id=order.order_id,
                                              error_message='Order cancelled')
            elif order_status['timeout']:
                await self._transition_to_state(TradingState.TIMEOUT,
                                              signal_id=self.current_signal.signal_id if self.current_signal else None,
                                              order_id=order.order_id)
            else:
                # Stay in LIVE state
                await self._emit_structured_log("live_monitoring", {
                    "order_id": order.order_id,
                    "status": order_status['status'],
                    "fill_quantity": order_status.get('fill_quantity', 0),
                    "remaining_quantity": order_status.get('remaining_quantity', order.quantity),
                })
                
        except Exception as e:
            self.logger.error(f"❌ [LIVE_STATE] Error in live state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_fill_state(self):
        """Handle FILL state - order filled"""
        try:
            if not self.current_order:
                await self._transition_to_state(TradingState.IDLE)
                return
            
            order = self.current_order
            
            # Emit fill log
            await self._emit_structured_log("order_filled", {
                "order_id": order.order_id,
                "cloid": order.cloid,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "fill_time_seconds": time.time() - order.timestamp,
            })
            
            # Update performance metrics
            self.performance_metrics.orders_filled += 1
            self.performance_metrics.fills_total += 1
            
            # Transition to RECONCILE state
            await self._transition_to_state(TradingState.RECONCILE,
                                          signal_id=self.current_signal.signal_id if self.current_signal else None,
                                          order_id=order.order_id)
            
        except Exception as e:
            self.logger.error(f"❌ [FILL_STATE] Error in fill state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_reject_state(self):
        """Handle REJECT state - order rejected"""
        try:
            if not self.current_order:
                await self._transition_to_state(TradingState.IDLE)
                return
            
            order = self.current_order
            
            # Emit reject log
            await self._emit_structured_log("order_rejected", {
                "order_id": order.order_id,
                "cloid": order.cloid,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "reject_time_seconds": time.time() - order.timestamp,
            })
            
            # Update performance metrics
            self.performance_metrics.orders_rejected += 1
            
            # Clear current order and signal
            self.current_order = None
            self.current_signal = None
            
            # Transition back to IDLE
            await self._transition_to_state(TradingState.IDLE)
            
        except Exception as e:
            self.logger.error(f"❌ [REJECT_STATE] Error in reject state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_reconcile_state(self):
        """Handle RECONCILE state - reconciling order"""
        try:
            if not self.current_order:
                await self._transition_to_state(TradingState.IDLE)
                return
            
            order = self.current_order
            
            # Reconcile order with exchange
            reconcile_result = await self._reconcile_order(order)
            
            if reconcile_result['success']:
                await self._emit_structured_log("order_reconciled", {
                    "order_id": order.order_id,
                    "cloid": order.cloid,
                    "reconcile_time_seconds": time.time() - order.timestamp,
                })
                
                # Transition to ATTRIB state
                await self._transition_to_state(TradingState.ATTRIB,
                                              signal_id=self.current_signal.signal_id if self.current_signal else None,
                                              order_id=order.order_id)
            else:
                await self._emit_structured_log("reconcile_failed", {
                    "order_id": order.order_id,
                    "error": reconcile_result.get('error', 'Reconcile failed'),
                })
                
                # Transition to ERROR state
                await self._transition_to_state(TradingState.ERROR,
                                              signal_id=self.current_signal.signal_id if self.current_signal else None,
                                              order_id=order.order_id,
                                              error_message=reconcile_result.get('error', 'Reconcile failed'))
            
        except Exception as e:
            self.logger.error(f"❌ [RECONCILE_STATE] Error in reconcile state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_attrib_state(self):
        """Handle ATTRIB state - attributing trade"""
        try:
            if not self.current_order:
                await self._transition_to_state(TradingState.IDLE)
                return
            
            order = self.current_order
            
            # Attribute trade to strategy/signal
            attribution_result = await self._attribute_trade(order, self.current_signal)
            
            # Emit attribution log
            await self._emit_structured_log("trade_attributed", {
                "order_id": order.order_id,
                "signal_id": self.current_signal.signal_id if self.current_signal else None,
                "attribution": attribution_result,
                "attrib_time_seconds": time.time() - order.timestamp,
            })
            
            # Clear current order and signal
            self.current_order = None
            self.current_signal = None
            
            # Transition back to IDLE
            await self._transition_to_state(TradingState.IDLE)
            
        except Exception as e:
            self.logger.error(f"❌ [ATTRIB_STATE] Error in attrib state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _handle_error_state(self):
        """Handle ERROR state - error recovery"""
        try:
            # Emit error log
            await self._emit_structured_log("state_error", {
                "current_state": self.current_state.value,
                "signal_id": self.current_signal.signal_id if self.current_signal else None,
                "order_id": self.current_order.order_id if self.current_order else None,
                "error_time": time.time(),
            })
            
            # Update error metrics
            self.performance_metrics.errors_total += 1
            
            # Clear current order and signal
            self.current_order = None
            self.current_signal = None
            
            # Transition back to IDLE
            await self._transition_to_state(TradingState.IDLE)
            
        except Exception as e:
            self.logger.error(f"❌ [ERROR_STATE] Error in error state: {e}")
            # Force transition to IDLE
            self.current_state = TradingState.IDLE
            self.current_order = None
            self.current_signal = None
    
    async def _handle_timeout_state(self):
        """Handle TIMEOUT state - timeout recovery"""
        try:
            # Emit timeout log
            await self._emit_structured_log("state_timeout", {
                "current_state": self.current_state.value,
                "signal_id": self.current_signal.signal_id if self.current_signal else None,
                "order_id": self.current_order.order_id if self.current_order else None,
                "timeout_time": time.time(),
            })
            
            # Update timeout metrics
            self.performance_metrics.errors_timeout += 1
            
            # Clear current order and signal
            self.current_order = None
            self.current_signal = None
            
            # Transition back to IDLE
            await self._transition_to_state(TradingState.IDLE)
            
        except Exception as e:
            self.logger.error(f"❌ [TIMEOUT_STATE] Error in timeout state: {e}")
            await self._transition_to_state(TradingState.ERROR, error_message=str(e))
    
    async def _transition_to_state(self, new_state: TradingState, **kwargs):
        """Transition to new state"""
        try:
            old_state = self.current_state
            transition_time = time.time()
            
            # Calculate transition duration
            duration_ms = 0.0
            if self.state_history:
                last_transition = self.state_history[-1]
                duration_ms = (transition_time - last_transition['timestamp']) * 1000
            
            # Create transition record
            transition = StateTransition(
                transition_id=str(uuid.uuid4()),
                from_state=old_state,
                to_state=new_state,
                timestamp=transition_time,
                duration_ms=duration_ms,
                signal_id=kwargs.get('signal_id'),
                order_id=kwargs.get('order_id'),
                error_message=kwargs.get('error_message'),
                metadata=kwargs.get('metadata', {}),
            )
            
            # Update state
            self.current_state = new_state
            
            # Store transition
            self.transition_history.append(transition)
            self.state_history.append({
                'state': new_state.value,
                'timestamp': transition_time,
                'transition_id': transition.transition_id,
            })
            
            # Emit transition log
            await self._emit_state_transition(old_state, new_state, **kwargs)
            
        except Exception as e:
            self.logger.error(f"❌ [TRANSITION] Error transitioning to state {new_state.value}: {e}")
    
    async def _emit_state_transition(self, from_state: TradingState, to_state: TradingState, **kwargs):
        """Emit state transition log"""
        try:
            log_data = {
                "event_type": "state_transition",
                "from_state": from_state.value,
                "to_state": to_state.value,
                "timestamp": time.time(),
                "signal_id": kwargs.get('signal_id'),
                "order_id": kwargs.get('order_id'),
                "error_message": kwargs.get('error_message'),
                "metadata": kwargs.get('metadata', {}),
            }
            
            await self._emit_structured_log("state_transition", log_data)
            
        except Exception as e:
            self.logger.error(f"❌ [EMIT_TRANSITION] Error emitting state transition: {e}")
    
    async def _emit_structured_log(self, event_type: str, data: Dict[str, Any]):
        """Emit structured JSON log"""
        try:
            log_entry = {
                "timestamp": time.time(),
                "event_type": event_type,
                "data": data,
            }
            
            # Store in structured logs
            self.structured_logs.append(log_entry)
            
            # Log to standard logger
            self.logger.info(f"📊 [STRUCTURED_LOG] {event_type}: {json.dumps(log_entry)}")
            
        except Exception as e:
            self.logger.error(f"❌ [STRUCTURED_LOG] Error emitting structured log: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if self.loop_times:
                # Calculate latency percentiles
                sorted_times = sorted(self.loop_times)
                n = len(sorted_times)
                
                self.performance_metrics.p95_loop_ms = sorted_times[int(n * 0.95)]
                self.performance_metrics.p99_loop_ms = sorted_times[int(n * 0.99)]
                self.performance_metrics.avg_loop_ms = sum(sorted_times) / n
            
            # Update Prometheus counters
            self.prometheus_counters.update({
                'p95_loop_ms': self.performance_metrics.p95_loop_ms,
                'p99_loop_ms': self.performance_metrics.p99_loop_ms,
                'avg_loop_ms': self.performance_metrics.avg_loop_ms,
                'orders_placed': self.performance_metrics.orders_placed,
                'orders_filled': self.performance_metrics.orders_filled,
                'orders_cancelled': self.performance_metrics.orders_cancelled,
                'orders_rejected': self.performance_metrics.orders_rejected,
                'fills_total': self.performance_metrics.fills_total,
                'maker_ratio': self.performance_metrics.maker_ratio,
                'avg_slippage_bps': self.performance_metrics.avg_slippage_bps,
                'avg_fee_bps': self.performance_metrics.avg_fee_bps,
                'fees_paid': self.performance_metrics.total_fees_paid,
                'funding_pnl': self.performance_metrics.funding_pnl,
                'net_pnl': self.performance_metrics.net_pnl,
            })
            
        except Exception as e:
            self.logger.error(f"❌ [UPDATE_METRICS] Error updating performance metrics: {e}")
    
    async def _handle_state_error(self, error: Exception):
        """Handle state machine error"""
        try:
            await self._emit_structured_log("state_machine_error", {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "current_state": self.current_state.value,
                "timestamp": time.time(),
            })
            
            # Update error metrics
            self.performance_metrics.errors_total += 1
            
            # Transition to ERROR state
            await self._transition_to_state(TradingState.ERROR, error_message=str(error))
            
        except Exception as e:
            self.logger.error(f"❌ [HANDLE_ERROR] Error handling state error: {e}")
    
    # Placeholder methods for integration with actual trading system
    async def _get_next_signal(self) -> Optional[TradingSignal]:
        """Get next trading signal (placeholder)"""
        return None
    
    async def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal (placeholder)"""
        return True
    
    async def _create_order_execution(self, signal: TradingSignal) -> OrderExecution:
        """Create order execution from signal (placeholder)"""
        return OrderExecution(
            order_id="",
            cloid=f"order_{int(time.time())}",
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
            price=signal.price,
            order_type="limit",
            time_in_force="GTC",
            post_only=True,
            timestamp=time.time(),
        )
    
    async def _place_order(self, order: OrderExecution) -> Dict[str, Any]:
        """Place order via API (placeholder)"""
        return {'success': True, 'order_id': f"order_{int(time.time())}"}
    
    async def _check_order_acknowledgment(self, order: OrderExecution) -> Dict[str, Any]:
        """Check order acknowledgment (placeholder)"""
        return {'acknowledged': True, 'timeout': False}
    
    async def _check_order_status(self, order: OrderExecution) -> Dict[str, Any]:
        """Check order status (placeholder)"""
        return {'status': 'live', 'timeout': False}
    
    async def _reconcile_order(self, order: OrderExecution) -> Dict[str, Any]:
        """Reconcile order with exchange (placeholder)"""
        return {'success': True}
    
    async def _attribute_trade(self, order: OrderExecution, signal: Optional[TradingSignal]) -> Dict[str, Any]:
        """Attribute trade to strategy (placeholder)"""
        return {'attributed': True, 'strategy': 'default'}
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        try:
            return {
                'current_state': self.current_state.value,
                'is_running': self.is_running,
                'shutdown_requested': self.shutdown_requested,
                'performance_metrics': self.performance_metrics.__dict__,
                'prometheus_counters': self.prometheus_counters,
                'recent_transitions': list(self.transition_history)[-10:],
                'recent_logs': list(self.structured_logs)[-10:],
                'state_config': self.state_config,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [STATE_SUMMARY] Error getting state summary: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown state machine"""
        try:
            self.logger.info("🎯 [STATE_MACHINE] Shutting down state machine...")
            
            self.shutdown_requested = True
            
            # Wait for current state to complete
            await asyncio.sleep(1)
            
            self.is_running = False
            
            self.logger.info("🎯 [STATE_MACHINE] State machine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ [SHUTDOWN] Error shutting down state machine: {e}")
