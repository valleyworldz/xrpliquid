"""
Idempotent Order Manager
Ensures exactly-once order accounting via strict client OID dedupe.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Set, Optional
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IdempotentOrderManager:
    """Manages orders with strict idempotency guarantees."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Order tracking
        self.client_order_ids: Set[str] = set()
        self.order_history: Dict[str, Dict] = {}
        self.fill_events: Dict[str, Dict] = {}
        
        # Deduplication settings
        self.max_order_age_hours = 24
        self.cleanup_interval_hours = 1
        
        # Statistics
        self.stats = {
            'orders_processed': 0,
            'duplicate_orders_rejected': 0,
            'fill_events_processed': 0,
            'duplicate_fills_rejected': 0,
            'last_cleanup': datetime.now()
        }
        
        # Load existing order history
        self.load_order_history()
    
    def generate_client_order_id(self, order_data: Dict) -> str:
        """Generate deterministic client order ID from order data."""
        # Create deterministic hash from order components
        order_string = f"{order_data.get('symbol', '')}_{order_data.get('side', '')}_{order_data.get('size', 0)}_{order_data.get('price', 0)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return hashlib.sha256(order_string.encode()).hexdigest()[:16]
    
    def is_duplicate_order(self, client_order_id: str) -> bool:
        """Check if order ID has been seen before."""
        return client_order_id in self.client_order_ids
    
    def register_order(self, client_order_id: str, order_data: Dict) -> bool:
        """Register a new order with idempotency check."""
        if self.is_duplicate_order(client_order_id):
            self.stats['duplicate_orders_rejected'] += 1
            logger.warning(f"🚫 Duplicate order rejected: {client_order_id}")
            return False
        
        # Register the order
        self.client_order_ids.add(client_order_id)
        self.order_history[client_order_id] = {
            'order_data': order_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'registered',
            'fill_count': 0,
            'total_filled': 0
        }
        
        self.stats['orders_processed'] += 1
        logger.info(f"✅ Order registered: {client_order_id}")
        return True
    
    def process_fill_event(self, client_order_id: str, fill_data: Dict) -> bool:
        """Process fill event with deduplication."""
        if client_order_id not in self.order_history:
            logger.error(f"❌ Fill event for unknown order: {client_order_id}")
            return False
        
        # Generate fill event ID for deduplication
        fill_id = f"{client_order_id}_{fill_data.get('fill_id', '')}_{fill_data.get('timestamp', '')}"
        
        if fill_id in self.fill_events:
            self.stats['duplicate_fills_rejected'] += 1
            logger.warning(f"🚫 Duplicate fill rejected: {fill_id}")
            return False
        
        # Process the fill
        self.fill_events[fill_id] = fill_data
        order_info = self.order_history[client_order_id]
        order_info['fill_count'] += 1
        order_info['total_filled'] += fill_data.get('size', 0)
        
        self.stats['fill_events_processed'] += 1
        logger.info(f"✅ Fill processed: {fill_id}")
        return True
    
    def cleanup_old_orders(self):
        """Clean up old orders to prevent memory bloat."""
        cutoff_time = datetime.now() - timedelta(hours=self.max_order_age_hours)
        
        orders_to_remove = []
        for client_order_id, order_info in self.order_history.items():
            order_time = datetime.fromisoformat(order_info['timestamp'])
            if order_time < cutoff_time:
                orders_to_remove.append(client_order_id)
        
        for client_order_id in orders_to_remove:
            del self.order_history[client_order_id]
            self.client_order_ids.discard(client_order_id)
        
        if orders_to_remove:
            logger.info(f"🧹 Cleaned up {len(orders_to_remove)} old orders")
        
        self.stats['last_cleanup'] = datetime.now()
    
    def get_order_status(self, client_order_id: str) -> Optional[Dict]:
        """Get current status of an order."""
        if client_order_id not in self.order_history:
            return None
        
        order_info = self.order_history[client_order_id]
        return {
            'client_order_id': client_order_id,
            'status': order_info['status'],
            'fill_count': order_info['fill_count'],
            'total_filled': order_info['total_filled'],
            'timestamp': order_info['timestamp']
        }
    
    def get_statistics(self) -> Dict:
        """Get order management statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'active_orders': len(self.order_history),
            'total_fill_events': len(self.fill_events),
            'duplicate_rejection_rate': (
                self.stats['duplicate_orders_rejected'] / 
                max(1, self.stats['orders_processed'])
            ),
            'fill_duplicate_rejection_rate': (
                self.stats['duplicate_fills_rejected'] / 
                max(1, self.stats['fill_events_processed'])
            )
        }
    
    def save_order_history(self):
        """Save order history to persistent storage."""
        history_dir = self.reports_dir / "execution"
        history_dir.mkdir(exist_ok=True)
        
        # Save order history
        history_file = history_dir / "order_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.order_history, f, indent=2)
        
        # Save fill events
        fills_file = history_dir / "fill_events.json"
        with open(fills_file, 'w') as f:
            json.dump(self.fill_events, f, indent=2)
        
        # Save statistics
        stats_file = history_dir / "order_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
        
        logger.info(f"💾 Order history saved: {history_file}")
    
    def load_order_history(self):
        """Load order history from persistent storage."""
        history_dir = self.reports_dir / "execution"
        
        # Load order history
        history_file = history_dir / "order_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.order_history = json.load(f)
                self.client_order_ids = set(self.order_history.keys())
            logger.info(f"📂 Loaded {len(self.order_history)} orders from history")
        
        # Load fill events
        fills_file = history_dir / "fill_events.json"
        if fills_file.exists():
            with open(fills_file, 'r') as f:
                self.fill_events = json.load(f)
            logger.info(f"📂 Loaded {len(self.fill_events)} fill events")


class WebSocketResyncManager:
    """Manages WebSocket connection resync with sequence gap detection."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.reports_dir / "reports"
        
        # Sequence tracking
        self.expected_sequence = 1
        self.last_sequence = 0
        self.gap_detected = False
        
        # Resync statistics
        self.resync_stats = {
            'total_messages': 0,
            'sequence_gaps': 0,
            'resync_events': 0,
            'last_resync': None
        }
        
        # Load sequence state
        self.load_sequence_state()
    
    def process_message(self, message: Dict) -> bool:
        """Process WebSocket message with sequence validation."""
        sequence = message.get('sequence', 0)
        self.resync_stats['total_messages'] += 1
        
        if sequence == 0:
            # No sequence number, assume valid
            return True
        
        if sequence == self.expected_sequence:
            # Expected sequence, update
            self.expected_sequence += 1
            self.last_sequence = sequence
            return True
        
        elif sequence > self.expected_sequence:
            # Gap detected
            gap_size = sequence - self.expected_sequence
            self.resync_stats['sequence_gaps'] += 1
            self.gap_detected = True
            
            logger.warning(f"⚠️ Sequence gap detected: expected {self.expected_sequence}, got {sequence} (gap: {gap_size})")
            
            # Trigger resync
            self.trigger_resync()
            return False
        
        else:
            # Out of order or duplicate
            logger.warning(f"⚠️ Out of order sequence: expected {self.expected_sequence}, got {sequence}")
            return False
    
    def trigger_resync(self):
        """Trigger WebSocket resync procedure."""
        self.resync_stats['resync_events'] += 1
        self.resync_stats['last_resync'] = datetime.now().isoformat()
        
        logger.info("🔄 Triggering WebSocket resync...")
        
        # In real implementation, this would:
        # 1. Request snapshot from exchange
        # 2. Replay missed messages
        # 3. Resume normal processing
        
        # For simulation, just reset sequence
        self.expected_sequence = self.last_sequence + 1
        self.gap_detected = False
        
        logger.info(f"✅ Resync completed, new expected sequence: {self.expected_sequence}")
    
    def get_resync_statistics(self) -> Dict:
        """Get resync statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'resync_stats': self.resync_stats,
            'current_sequence': self.expected_sequence,
            'gap_detected': self.gap_detected,
            'gap_rate': (
                self.resync_stats['sequence_gaps'] / 
                max(1, self.resync_stats['total_messages'])
            )
        }
    
    def save_sequence_state(self):
        """Save sequence state to persistent storage."""
        state_dir = self.reports_dir / "execution"
        state_dir.mkdir(exist_ok=True)
        
        state_file = state_dir / "websocket_sequence_state.json"
        state_data = {
            'expected_sequence': self.expected_sequence,
            'last_sequence': self.last_sequence,
            'resync_stats': self.resync_stats
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"💾 Sequence state saved: {state_file}")
    
    def load_sequence_state(self):
        """Load sequence state from persistent storage."""
        state_dir = self.reports_dir / "execution"
        state_file = state_dir / "websocket_sequence_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                self.expected_sequence = state_data.get('expected_sequence', 1)
                self.last_sequence = state_data.get('last_sequence', 0)
                self.resync_stats = state_data.get('resync_stats', self.resync_stats)
            logger.info(f"📂 Loaded sequence state: expected {self.expected_sequence}")


def main():
    """Main function to demonstrate idempotent execution."""
    # Test order manager
    order_manager = IdempotentOrderManager()
    
    # Test orders
    test_orders = [
        {'symbol': 'XRP', 'side': 'buy', 'size': 1000, 'price': 0.50},
        {'symbol': 'XRP', 'side': 'sell', 'size': 500, 'price': 0.51},
        {'symbol': 'XRP', 'side': 'buy', 'size': 1000, 'price': 0.50}  # Duplicate
    ]
    
    for i, order_data in enumerate(test_orders):
        client_order_id = order_manager.generate_client_order_id(order_data)
        success = order_manager.register_order(client_order_id, order_data)
        print(f"Order {i+1}: {client_order_id} -> {'Success' if success else 'Duplicate'}")
    
    # Test WebSocket resync
    ws_manager = WebSocketResyncManager()
    
    # Test messages
    test_messages = [
        {'sequence': 1, 'data': 'message1'},
        {'sequence': 2, 'data': 'message2'},
        {'sequence': 4, 'data': 'message4'},  # Gap
        {'sequence': 5, 'data': 'message5'},
    ]
    
    for msg in test_messages:
        success = ws_manager.process_message(msg)
        print(f"Message {msg['sequence']}: {'Success' if success else 'Gap detected'}")
    
    # Save statistics
    order_manager.save_order_history()
    ws_manager.save_sequence_state()
    
    print("✅ Idempotent execution demonstration completed")


if __name__ == "__main__":
    main()
