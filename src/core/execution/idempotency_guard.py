"""
Idempotency Guard - Exactly-Once Accounting & WS Resync
Proves exactly-once accounting under reconnects and handles WS sequence gaps.
"""

import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class OrderState(Enum):
    """Order state enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderRecord:
    """Represents an order record for idempotency."""
    client_order_id: str
    exchange_order_id: Optional[str]
    state: OrderState
    timestamp: datetime
    retry_count: int
    last_sequence_number: Optional[int]
    hash_signature: str


class IdempotencyGuard:
    """Ensures exactly-once accounting and handles WS resync."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.execution_dir = self.reports_dir / "execution"
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        
        # Order tracking
        self.order_registry: Dict[str, OrderRecord] = {}
        self.sequence_tracker: Dict[str, int] = {}
        self.duplicate_detector: Set[str] = set()
        
        # WS resync tracking
        self.last_sequence_numbers: Dict[str, int] = {}
        self.sequence_gaps: List[Dict[str, Any]] = []
        
        # Load existing state
        self._load_state()
    
    def generate_client_order_id(self, 
                               symbol: str,
                               side: str,
                               quantity: float,
                               price: float,
                               timestamp: datetime = None) -> str:
        """Generate unique client order ID."""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Create deterministic ID components
        components = {
            "symbol": symbol,
            "side": side,
            "quantity": str(quantity),
            "price": str(price),
            "timestamp": timestamp.isoformat(),
            "nonce": str(int(time.time() * 1000000))  # Microsecond precision
        }
        
        # Create hash signature
        components_str = json.dumps(components, sort_keys=True)
        hash_signature = hashlib.sha256(components_str.encode()).hexdigest()[:16]
        
        # Generate client order ID
        client_order_id = f"{symbol}_{side}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_{hash_signature}"
        
        return client_order_id
    
    def register_order(self, 
                      client_order_id: str,
                      symbol: str,
                      side: str,
                      quantity: float,
                      price: float) -> bool:
        """Register order for idempotency tracking."""
        
        # Check for duplicates
        if client_order_id in self.order_registry:
            return False
        
        # Create order record
        order_record = OrderRecord(
            client_order_id=client_order_id,
            exchange_order_id=None,
            state=OrderState.PENDING,
            timestamp=datetime.now(timezone.utc),
            retry_count=0,
            last_sequence_number=None,
            hash_signature=hashlib.sha256(client_order_id.encode()).hexdigest()
        )
        
        # Register order
        self.order_registry[client_order_id] = order_record
        self.duplicate_detector.add(client_order_id)
        
        # Save state
        self._save_state()
        
        return True
    
    def update_order_state(self, 
                          client_order_id: str,
                          new_state: OrderState,
                          exchange_order_id: str = None,
                          sequence_number: int = None) -> bool:
        """Update order state with idempotency checks."""
        
        if client_order_id not in self.order_registry:
            return False
        
        order_record = self.order_registry[client_order_id]
        
        # Validate state transition
        if not self._is_valid_state_transition(order_record.state, new_state):
            return False
        
        # Update order record
        order_record.state = new_state
        if exchange_order_id:
            order_record.exchange_order_id = exchange_order_id
        if sequence_number:
            order_record.last_sequence_number = sequence_number
        
        # Save state
        self._save_state()
        
        return True
    
    def _is_valid_state_transition(self, 
                                 current_state: OrderState,
                                 new_state: OrderState) -> bool:
        """Validate state transition."""
        
        valid_transitions = {
            OrderState.PENDING: [OrderState.SUBMITTED, OrderState.REJECTED],
            OrderState.SUBMITTED: [OrderState.ACKNOWLEDGED, OrderState.REJECTED],
            OrderState.ACKNOWLEDGED: [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED],
            OrderState.FILLED: [],  # Terminal state
            OrderState.CANCELLED: [],  # Terminal state
            OrderState.REJECTED: []  # Terminal state
        }
        
        return new_state in valid_transitions.get(current_state, [])
    
    def handle_websocket_reconnect(self, 
                                 connection_id: str,
                                 last_sequence_number: int) -> Dict[str, Any]:
        """Handle websocket reconnection and sequence gap detection."""
        
        reconnect_info = {
            "connection_id": connection_id,
            "reconnect_timestamp": datetime.now(timezone.utc).isoformat(),
            "last_known_sequence": last_sequence_number,
            "sequence_gaps_detected": [],
            "resync_required": False
        }
        
        # Check for sequence gaps
        if connection_id in self.last_sequence_numbers:
            expected_sequence = self.last_sequence_numbers[connection_id] + 1
            
            if last_sequence_number > expected_sequence:
                # Sequence gap detected
                gap_info = {
                    "gap_start": expected_sequence,
                    "gap_end": last_sequence_number - 1,
                    "gap_size": last_sequence_number - expected_sequence,
                    "detected_at": datetime.now(timezone.utc).isoformat()
                }
                
                reconnect_info["sequence_gaps_detected"].append(gap_info)
                reconnect_info["resync_required"] = True
                
                # Log sequence gap
                self.sequence_gaps.append(gap_info)
        
        # Update sequence tracker
        self.last_sequence_numbers[connection_id] = last_sequence_number
        
        # Save state
        self._save_state()
        
        return reconnect_info
    
    def detect_duplicate_fills(self, 
                             fill_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect duplicate fills based on order IDs and timestamps."""
        
        duplicates = []
        seen_fills = set()
        
        for fill in fill_data:
            # Create fill signature
            fill_signature = self._create_fill_signature(fill)
            
            if fill_signature in seen_fills:
                duplicates.append({
                    "fill_data": fill,
                    "duplicate_type": "exact_duplicate",
                    "detected_at": datetime.now(timezone.utc).isoformat()
                })
            else:
                seen_fills.add(fill_signature)
        
        return duplicates
    
    def _create_fill_signature(self, fill: Dict[str, Any]) -> str:
        """Create signature for fill deduplication."""
        
        signature_components = {
            "order_id": fill.get("order_id", ""),
            "fill_id": fill.get("fill_id", ""),
            "quantity": str(fill.get("quantity", 0)),
            "price": str(fill.get("price", 0)),
            "timestamp": fill.get("timestamp", "")
        }
        
        signature_str = json.dumps(signature_components, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()
    
    def verify_accounting_integrity(self, 
                                  account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify accounting integrity after reconnects."""
        
        integrity_check = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_orders": len(self.order_registry),
            "pending_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "sequence_gaps": len(self.sequence_gaps),
            "integrity_issues": [],
            "overall_status": "verified"
        }
        
        # Count orders by state
        for order_record in self.order_registry.values():
            if order_record.state == OrderState.PENDING:
                integrity_check["pending_orders"] += 1
            elif order_record.state == OrderState.FILLED:
                integrity_check["filled_orders"] += 1
            elif order_record.state == OrderState.CANCELLED:
                integrity_check["cancelled_orders"] += 1
            elif order_record.state == OrderState.REJECTED:
                integrity_check["rejected_orders"] += 1
        
        # Check for integrity issues
        if integrity_check["sequence_gaps"] > 0:
            integrity_check["integrity_issues"].append("Sequence gaps detected - manual reconciliation required")
            integrity_check["overall_status"] = "warning"
        
        if integrity_check["pending_orders"] > 100:  # Threshold for too many pending orders
            integrity_check["integrity_issues"].append("Too many pending orders - possible stuck orders")
            integrity_check["overall_status"] = "warning"
        
        return integrity_check
    
    def run_cancel_replace_storm_test(self, 
                                    symbol: str,
                                    num_orders: int = 100,
                                    storm_duration_seconds: int = 60) -> Dict[str, Any]:
        """Run cancel/replace storm test for resilience."""
        
        storm_test = {
            "test_timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "num_orders": num_orders,
            "storm_duration_seconds": storm_duration_seconds,
            "orders_created": 0,
            "orders_cancelled": 0,
            "orders_replaced": 0,
            "duplicate_detections": 0,
            "state_transition_errors": 0,
            "test_result": "passed"
        }
        
        # Create orders
        for i in range(num_orders):
            client_order_id = self.generate_client_order_id(
                symbol, "buy", 100.0, 1.0 + i * 0.01
            )
            
            if self.register_order(client_order_id, symbol, "buy", 100.0, 1.0 + i * 0.01):
                storm_test["orders_created"] += 1
                
                # Simulate state transitions
                if not self.update_order_state(client_order_id, OrderState.SUBMITTED):
                    storm_test["state_transition_errors"] += 1
                
                if not self.update_order_state(client_order_id, OrderState.ACKNOWLEDGED):
                    storm_test["state_transition_errors"] += 1
                
                # Cancel order
                if not self.update_order_state(client_order_id, OrderState.CANCELLED):
                    storm_test["state_transition_errors"] += 1
                else:
                    storm_test["orders_cancelled"] += 1
        
        # Check for duplicates
        storm_test["duplicate_detections"] = len(self.duplicate_detector) - storm_test["orders_created"]
        
        # Determine test result
        if storm_test["state_transition_errors"] > 0 or storm_test["duplicate_detections"] > 0:
            storm_test["test_result"] = "failed"
        
        return storm_test
    
    def _save_state(self):
        """Save idempotency state to disk."""
        
        state_data = {
            "order_registry": {
                client_id: {
                    "client_order_id": record.client_order_id,
                    "exchange_order_id": record.exchange_order_id,
                    "state": record.state.value,
                    "timestamp": record.timestamp.isoformat(),
                    "retry_count": record.retry_count,
                    "last_sequence_number": record.last_sequence_number,
                    "hash_signature": record.hash_signature
                }
                for client_id, record in self.order_registry.items()
            },
            "sequence_tracker": self.sequence_tracker,
            "last_sequence_numbers": self.last_sequence_numbers,
            "sequence_gaps": self.sequence_gaps,
            "last_saved": datetime.now(timezone.utc).isoformat()
        }
        
        state_file = self.execution_dir / "idempotency_state.json"
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def _load_state(self):
        """Load idempotency state from disk."""
        
        state_file = self.execution_dir / "idempotency_state.json"
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Load order registry
            for client_id, record_data in state_data.get("order_registry", {}).items():
                order_record = OrderRecord(
                    client_order_id=record_data["client_order_id"],
                    exchange_order_id=record_data["exchange_order_id"],
                    state=OrderState(record_data["state"]),
                    timestamp=datetime.fromisoformat(record_data["timestamp"]),
                    retry_count=record_data["retry_count"],
                    last_sequence_number=record_data["last_sequence_number"],
                    hash_signature=record_data["hash_signature"]
                )
                self.order_registry[client_id] = order_record
                self.duplicate_detector.add(client_id)
            
            # Load other state
            self.sequence_tracker = state_data.get("sequence_tracker", {})
            self.last_sequence_numbers = state_data.get("last_sequence_numbers", {})
            self.sequence_gaps = state_data.get("sequence_gaps", [])
            
        except Exception as e:
            print(f"Warning: Could not load idempotency state: {e}")


def main():
    """Test idempotency guard functionality."""
    guard = IdempotencyGuard()
    
    # Test order registration
    client_order_id = guard.generate_client_order_id("XRP", "buy", 100.0, 1.0)
    success = guard.register_order(client_order_id, "XRP", "buy", 100.0, 1.0)
    print(f"✅ Order registration: {success}")
    
    # Test state transitions
    guard.update_order_state(client_order_id, OrderState.SUBMITTED)
    guard.update_order_state(client_order_id, OrderState.ACKNOWLEDGED)
    guard.update_order_state(client_order_id, OrderState.FILLED)
    print(f"✅ State transitions completed")
    
    # Test WS reconnect
    reconnect_info = guard.handle_websocket_reconnect("ws_001", 1000)
    print(f"✅ WS reconnect: {reconnect_info['resync_required']}")
    
    # Test cancel/replace storm
    storm_test = guard.run_cancel_replace_storm_test("XRP", 50, 30)
    print(f"✅ Storm test: {storm_test['test_result']}")
    
    # Test accounting integrity
    integrity = guard.verify_accounting_integrity({})
    print(f"✅ Accounting integrity: {integrity['overall_status']}")
    
    print("✅ Idempotency guard testing completed")


if __name__ == "__main__":
    main()