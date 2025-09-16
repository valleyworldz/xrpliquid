"""
Idempotency Guard - Ensures Exactly-Once Execution
Prevents double-fills, duplicate orders, and stale state issues.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class OrderState:
    """Represents the state of an order."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    fills: List[Dict[str, Any]]
    reject_reason: Optional[str] = None


class IdempotencyGuard:
    """Ensures exactly-once execution and prevents duplicate operations."""
    
    def __init__(self, state_dir: str = "data/execution_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory state tracking
        self.active_orders: Dict[str, OrderState] = {}
        self.processed_operations: Set[str] = set()
        self.sequence_numbers: Dict[str, int] = {}
        
        # Load existing state
        self._load_state()
    
    def generate_client_order_id(self, 
                               symbol: str, 
                               side: str, 
                               quantity: float, 
                               price: float,
                               order_type: str) -> str:
        """Generate unique client order ID."""
        
        # Create deterministic ID based on order parameters
        timestamp = int(time.time() * 1000)  # milliseconds
        order_data = f"{symbol}_{side}_{quantity}_{price}_{order_type}_{timestamp}"
        
        # Generate hash for uniqueness
        order_hash = hashlib.md5(order_data.encode()).hexdigest()[:8]
        
        # Format: SYMBOL_SIDE_TIMESTAMP_HASH
        client_order_id = f"{symbol}_{side}_{timestamp}_{order_hash}"
        
        return client_order_id
    
    def register_order(self, 
                      client_order_id: str,
                      symbol: str,
                      side: str,
                      quantity: float,
                      price: float,
                      order_type: str) -> bool:
        """Register a new order and check for duplicates."""
        
        # Check if order already exists
        if client_order_id in self.active_orders:
            existing_order = self.active_orders[client_order_id]
            if existing_order.status in ['pending', 'partially_filled', 'filled']:
                return False  # Duplicate order
        
        # Create new order state
        order_state = OrderState(
            order_id="",  # Will be set by exchange
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            status="pending",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            fills=[]
        )
        
        # Register order
        self.active_orders[client_order_id] = order_state
        self._save_state()
        
        return True
    
    def update_order_status(self, 
                          client_order_id: str,
                          order_id: str = None,
                          status: str = None,
                          fill_data: Dict[str, Any] = None,
                          reject_reason: str = None) -> bool:
        """Update order status and handle fills."""
        
        if client_order_id not in self.active_orders:
            return False  # Order not found
        
        order = self.active_orders[client_order_id]
        
        # Update order ID if provided
        if order_id:
            order.order_id = order_id
        
        # Update status
        if status:
            order.status = status
        
        # Handle fill
        if fill_data:
            # Check for duplicate fill
            fill_id = fill_data.get('fill_id', '')
            if fill_id and any(f.get('fill_id') == fill_id for f in order.fills):
                return False  # Duplicate fill
            
            order.fills.append(fill_data)
        
        # Handle rejection
        if reject_reason:
            order.reject_reason = reject_reason
            order.status = "rejected"
        
        # Update timestamp
        order.updated_at = datetime.now(timezone.utc)
        
        # Save state
        self._save_state()
        
        return True
    
    def check_duplicate_fill(self, 
                           client_order_id: str, 
                           fill_id: str) -> bool:
        """Check if fill is duplicate."""
        
        if client_order_id not in self.active_orders:
            return False
        
        order = self.active_orders[client_order_id]
        
        # Check if fill ID already exists
        for fill in order.fills:
            if fill.get('fill_id') == fill_id:
                return True  # Duplicate fill
        
        return False
    
    def get_order_state(self, client_order_id: str) -> Optional[OrderState]:
        """Get current order state."""
        return self.active_orders.get(client_order_id)
    
    def process_operation(self, operation_id: str) -> bool:
        """Process an operation and check for duplicates."""
        
        # Check if operation already processed
        if operation_id in self.processed_operations:
            return False  # Duplicate operation
        
        # Mark as processed
        self.processed_operations.add(operation_id)
        self._save_state()
        
        return True
    
    def handle_sequence_gap(self, 
                          stream_name: str, 
                          expected_sequence: int, 
                          actual_sequence: int) -> Dict[str, Any]:
        """Handle sequence number gaps in data streams."""
        
        gap_info = {
            "stream_name": stream_name,
            "expected_sequence": expected_sequence,
            "actual_sequence": actual_sequence,
            "gap_size": actual_sequence - expected_sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_required": True
        }
        
        # Determine action based on gap size
        if gap_info["gap_size"] == 1:
            gap_info["action"] = "ignore_single_gap"
            gap_info["action_required"] = False
        elif gap_info["gap_size"] <= 10:
            gap_info["action"] = "request_replay"
        else:
            gap_info["action"] = "request_snapshot"
        
        # Update sequence number
        self.sequence_numbers[stream_name] = actual_sequence
        
        # Log gap
        self._log_sequence_gap(gap_info)
        
        return gap_info
    
    def _log_sequence_gap(self, gap_info: Dict[str, Any]):
        """Log sequence gap for monitoring."""
        
        gap_log = {
            "timestamp": gap_info["timestamp"],
            "stream_name": gap_info["stream_name"],
            "gap_size": gap_info["gap_size"],
            "action": gap_info["action"],
            "action_required": gap_info["action_required"]
        }
        
        # Save to gap log file
        gap_file = self.state_dir / "sequence_gaps.jsonl"
        with open(gap_file, 'a') as f:
            f.write(json.dumps(gap_log) + '\n')
    
    def create_reject_taxonomy(self) -> Dict[str, Dict[str, Any]]:
        """Create taxonomy of reject reasons and remediation actions."""
        
        taxonomy = {
            "INSUFFICIENT_MARGIN": {
                "description": "Not enough margin to place order",
                "remediation": "reduce_position_size",
                "auto_retry": False,
                "severity": "high"
            },
            "INVALID_TICK_SIZE": {
                "description": "Price not aligned with tick size",
                "remediation": "retick_price",
                "auto_retry": True,
                "severity": "medium"
            },
            "INVALID_LOT_SIZE": {
                "description": "Quantity not aligned with lot size",
                "remediation": "resize_quantity",
                "auto_retry": True,
                "severity": "medium"
            },
            "MIN_NOTIONAL": {
                "description": "Order value below minimum",
                "remediation": "increase_quantity_or_price",
                "auto_retry": True,
                "severity": "low"
            },
            "REDUCE_ONLY_VIOLATION": {
                "description": "Order would increase position when reduce-only",
                "remediation": "flip_side_or_remove_reduce_only",
                "auto_retry": True,
                "severity": "medium"
            },
            "POSITION_LIMIT": {
                "description": "Order would exceed position limit",
                "remediation": "reduce_quantity",
                "auto_retry": True,
                "severity": "high"
            },
            "RATE_LIMIT": {
                "description": "Too many requests",
                "remediation": "wait_and_retry",
                "auto_retry": True,
                "severity": "low"
            },
            "MARKET_CLOSED": {
                "description": "Market is closed",
                "remediation": "queue_for_open",
                "auto_retry": False,
                "severity": "medium"
            }
        }
        
        return taxonomy
    
    def handle_reject(self, 
                     client_order_id: str, 
                     reject_reason: str) -> Dict[str, Any]:
        """Handle order rejection with automatic remediation."""
        
        taxonomy = self.create_reject_taxonomy()
        
        if reject_reason not in taxonomy:
            reject_reason = "UNKNOWN"
        
        reject_info = taxonomy[reject_reason]
        
        # Update order status
        self.update_order_status(
            client_order_id=client_order_id,
            status="rejected",
            reject_reason=reject_reason
        )
        
        # Determine remediation action
        remediation = {
            "reject_reason": reject_reason,
            "remediation_action": reject_info["remediation"],
            "auto_retry": reject_info["auto_retry"],
            "severity": reject_info["severity"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Log reject for monitoring
        self._log_reject(client_order_id, remediation)
        
        return remediation
    
    def _log_reject(self, client_order_id: str, reject_info: Dict[str, Any]):
        """Log reject for monitoring and analysis."""
        
        reject_log = {
            "client_order_id": client_order_id,
            "timestamp": reject_info["timestamp"],
            "reject_reason": reject_info["reject_reason"],
            "remediation_action": reject_info["remediation_action"],
            "auto_retry": reject_info["auto_retry"],
            "severity": reject_info["severity"]
        }
        
        # Save to reject log file
        reject_file = self.state_dir / "rejects.jsonl"
        with open(reject_file, 'a') as f:
            f.write(json.dumps(reject_log) + '\n')
    
    def _save_state(self):
        """Save current state to disk."""
        
        state_data = {
            "active_orders": {
                client_id: {
                    "order_id": order.order_id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "order_type": order.order_type,
                    "status": order.status,
                    "created_at": order.created_at.isoformat(),
                    "updated_at": order.updated_at.isoformat(),
                    "fills": order.fills,
                    "reject_reason": order.reject_reason
                }
                for client_id, order in self.active_orders.items()
            },
            "processed_operations": list(self.processed_operations),
            "sequence_numbers": self.sequence_numbers,
            "last_saved": datetime.now(timezone.utc).isoformat()
        }
        
        state_file = self.state_dir / "execution_state.json"
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def _load_state(self):
        """Load state from disk."""
        
        state_file = self.state_dir / "execution_state.json"
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Load active orders
            for client_id, order_data in state_data.get("active_orders", {}).items():
                order = OrderState(
                    order_id=order_data["order_id"],
                    client_order_id=order_data["client_order_id"],
                    symbol=order_data["symbol"],
                    side=order_data["side"],
                    quantity=order_data["quantity"],
                    price=order_data["price"],
                    order_type=order_data["order_type"],
                    status=order_data["status"],
                    created_at=datetime.fromisoformat(order_data["created_at"]),
                    updated_at=datetime.fromisoformat(order_data["updated_at"]),
                    fills=order_data["fills"],
                    reject_reason=order_data.get("reject_reason")
                )
                self.active_orders[client_id] = order
            
            # Load processed operations
            self.processed_operations = set(state_data.get("processed_operations", []))
            
            # Load sequence numbers
            self.sequence_numbers = state_data.get("sequence_numbers", {})
            
        except Exception as e:
            print(f"Warning: Could not load execution state: {e}")
    
    def generate_idempotency_report(self) -> Dict[str, Any]:
        """Generate idempotency and execution correctness report."""
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "active_orders_count": len(self.active_orders),
            "processed_operations_count": len(self.processed_operations),
            "sequence_streams_count": len(self.sequence_numbers),
            "order_status_distribution": {},
            "reject_reason_distribution": {},
            "fill_duplication_checks": {
                "total_fills_checked": 0,
                "duplicate_fills_detected": 0
            }
        }
        
        # Analyze order status distribution
        for order in self.active_orders.values():
            status = order.status
            report["order_status_distribution"][status] = report["order_status_distribution"].get(status, 0) + 1
            
            # Count fills
            report["fill_duplication_checks"]["total_fills_checked"] += len(order.fills)
        
        # Analyze reject reasons
        reject_file = self.state_dir / "rejects.jsonl"
        if reject_file.exists():
            reject_reasons = {}
            with open(reject_file, 'r') as f:
                for line in f:
                    try:
                        reject_data = json.loads(line.strip())
                        reason = reject_data.get("reject_reason", "UNKNOWN")
                        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                    except:
                        continue
            report["reject_reason_distribution"] = reject_reasons
        
        return report


def main():
    """Test idempotency guard functionality."""
    guard = IdempotencyGuard()
    
    # Test order registration
    client_order_id = guard.generate_client_order_id("XRP", "buy", 100.0, 0.5, "limit")
    success = guard.register_order(client_order_id, "XRP", "buy", 100.0, 0.5, "limit")
    print(f"✅ Order registration: {success}")
    
    # Test duplicate order
    success2 = guard.register_order(client_order_id, "XRP", "buy", 100.0, 0.5, "limit")
    print(f"✅ Duplicate order prevention: {not success2}")
    
    # Test order update
    guard.update_order_status(client_order_id, order_id="ex_123", status="filled")
    order_state = guard.get_order_state(client_order_id)
    print(f"✅ Order update: {order_state.status if order_state else 'Not found'}")
    
    # Test sequence gap handling
    gap_info = guard.handle_sequence_gap("trades", 100, 103)
    print(f"✅ Sequence gap handling: {gap_info['action']}")
    
    # Test reject handling
    reject_info = guard.handle_reject(client_order_id, "INSUFFICIENT_MARGIN")
    print(f"✅ Reject handling: {reject_info['remediation_action']}")
    
    # Generate report
    report = guard.generate_idempotency_report()
    print(f"✅ Idempotency report: {report['active_orders_count']} active orders")
    
    print("✅ Idempotency guard testing completed")


if __name__ == "__main__":
    main()
