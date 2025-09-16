"""
Idempotency Property Tests
Tests for exactly-once order accounting and WS resync properties.
"""

import json
import os
import pytest
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestIdempotencyProperties:
    """Property tests for idempotent execution."""
    
    def setup_method(self):
        """Set up test environment."""
        self.repo_root = Path(".")
        self.reports_dir = self.repo_root / "reports"
        
        # Import the modules
        import sys
        sys.path.append(str(self.repo_root))
        
        from src.core.execution.idempotent_order_manager import IdempotentOrderManager, WebSocketResyncManager
        
        self.order_manager = IdempotentOrderManager()
        self.ws_manager = WebSocketResyncManager()
    
    def test_order_deduplication_property(self):
        """Test that duplicate orders are rejected."""
        logger.info("🧪 Testing order deduplication property...")
        
        # Create test order
        order_data = {'symbol': 'XRP', 'side': 'buy', 'size': 1000, 'price': 0.50}
        client_order_id = self.order_manager.generate_client_order_id(order_data)
        
        # First registration should succeed
        success1 = self.order_manager.register_order(client_order_id, order_data)
        assert success1, "First order registration should succeed"
        
        # Second registration with same ID should fail
        success2 = self.order_manager.register_order(client_order_id, order_data)
        assert not success2, "Duplicate order registration should fail"
        
        # Statistics should reflect rejection
        stats = self.order_manager.get_statistics()
        assert stats['stats']['duplicate_orders_rejected'] == 1, "Should have 1 duplicate rejection"
        assert stats['stats']['orders_processed'] == 1, "Should have 1 order processed"
        
        logger.info("✅ Order deduplication property test passed")
    
    def test_fill_deduplication_property(self):
        """Test that duplicate fills are rejected."""
        logger.info("🧪 Testing fill deduplication property...")
        
        # Register an order first
        order_data = {'symbol': 'XRP', 'side': 'buy', 'size': 1000, 'price': 0.50}
        client_order_id = self.order_manager.generate_client_order_id(order_data)
        self.order_manager.register_order(client_order_id, order_data)
        
        # Create test fill
        fill_data = {'fill_id': 'fill_001', 'size': 500, 'price': 0.50, 'timestamp': datetime.now().isoformat()}
        
        # First fill should succeed
        success1 = self.order_manager.process_fill_event(client_order_id, fill_data)
        assert success1, "First fill should succeed"
        
        # Second fill with same data should fail
        success2 = self.order_manager.process_fill_event(client_order_id, fill_data)
        assert not success2, "Duplicate fill should fail"
        
        # Statistics should reflect rejection
        stats = self.order_manager.get_statistics()
        assert stats['stats']['duplicate_fills_rejected'] == 1, "Should have 1 duplicate fill rejection"
        assert stats['stats']['fill_events_processed'] == 1, "Should have 1 fill processed"
        
        logger.info("✅ Fill deduplication property test passed")
    
    def test_websocket_sequence_gap_detection(self):
        """Test WebSocket sequence gap detection."""
        logger.info("🧪 Testing WebSocket sequence gap detection...")
        
        # Process messages in sequence
        msg1 = {'sequence': 1, 'data': 'message1'}
        msg2 = {'sequence': 2, 'data': 'message2'}
        msg3 = {'sequence': 4, 'data': 'message4'}  # Gap
        
        # First two messages should succeed
        success1 = self.ws_manager.process_message(msg1)
        success2 = self.ws_manager.process_message(msg2)
        assert success1 and success2, "Sequential messages should succeed"
        
        # Gap message should trigger resync
        success3 = self.ws_manager.process_message(msg3)
        assert not success3, "Gap message should trigger resync"
        
        # Statistics should reflect gap detection
        stats = self.ws_manager.get_resync_statistics()
        assert stats['resync_stats']['sequence_gaps'] == 1, "Should have 1 sequence gap"
        assert stats['resync_stats']['resync_events'] == 1, "Should have 1 resync event"
        
        logger.info("✅ WebSocket sequence gap detection test passed")
    
    def test_order_state_consistency(self):
        """Test that order state remains consistent."""
        logger.info("🧪 Testing order state consistency...")
        
        # Register order
        order_data = {'symbol': 'XRP', 'side': 'buy', 'size': 1000, 'price': 0.50}
        client_order_id = self.order_manager.generate_client_order_id(order_data)
        self.order_manager.register_order(client_order_id, order_data)
        
        # Check initial state
        status = self.order_manager.get_order_status(client_order_id)
        assert status is not None, "Order status should exist"
        assert status['fill_count'] == 0, "Initial fill count should be 0"
        assert status['total_filled'] == 0, "Initial total filled should be 0"
        
        # Process fill
        fill_data = {'fill_id': 'fill_001', 'size': 500, 'price': 0.50, 'timestamp': datetime.now().isoformat()}
        self.order_manager.process_fill_event(client_order_id, fill_data)
        
        # Check updated state
        status = self.order_manager.get_order_status(client_order_id)
        assert status['fill_count'] == 1, "Fill count should be 1"
        assert status['total_filled'] == 500, "Total filled should be 500"
        
        logger.info("✅ Order state consistency test passed")
    
    def test_chaos_resilience(self):
        """Test system resilience under chaotic conditions."""
        logger.info("🧪 Testing chaos resilience...")
        
        # Simulate rapid order submissions
        order_ids = []
        for i in range(100):
            order_data = {'symbol': 'XRP', 'side': 'buy', 'size': 1000, 'price': 0.50 + i * 0.001}
            client_order_id = self.order_manager.generate_client_order_id(order_data)
            success = self.order_manager.register_order(client_order_id, order_data)
            order_ids.append((client_order_id, success))
        
        # Check that all orders were processed
        successful_orders = [oid for oid, success in order_ids if success]
        assert len(successful_orders) == 100, "All orders should be processed successfully"
        
        # Simulate rapid WebSocket messages with gaps
        for i in range(50):
            if i % 10 == 0:  # Create gaps every 10 messages
                sequence = i + 2  # Skip sequence
            else:
                sequence = i + 1
            
            msg = {'sequence': sequence, 'data': f'message_{i}'}
            self.ws_manager.process_message(msg)
        
        # Check that gaps were detected
        stats = self.ws_manager.get_resync_statistics()
        assert stats['resync_stats']['sequence_gaps'] > 0, "Should have detected sequence gaps"
        
        logger.info("✅ Chaos resilience test passed")
    
    def test_persistence_property(self):
        """Test that order state persists across restarts."""
        logger.info("🧪 Testing persistence property...")
        
        # Register order and process fill
        order_data = {'symbol': 'XRP', 'side': 'buy', 'size': 1000, 'price': 0.50}
        client_order_id = self.order_manager.generate_client_order_id(order_data)
        self.order_manager.register_order(client_order_id, order_data)
        
        fill_data = {'fill_id': 'fill_001', 'size': 500, 'price': 0.50, 'timestamp': datetime.now().isoformat()}
        self.order_manager.process_fill_event(client_order_id, fill_data)
        
        # Save state
        self.order_manager.save_order_history()
        
        # Create new manager (simulating restart)
        from src.core.execution.idempotent_order_manager import IdempotentOrderManager
        new_manager = IdempotentOrderManager()
        
        # Check that order state is preserved
        status = new_manager.get_order_status(client_order_id)
        assert status is not None, "Order should exist after restart"
        assert status['fill_count'] == 1, "Fill count should be preserved"
        assert status['total_filled'] == 500, "Total filled should be preserved"
        
        # Check that duplicate is still rejected
        success = new_manager.register_order(client_order_id, order_data)
        assert not success, "Duplicate should still be rejected after restart"
        
        logger.info("✅ Persistence property test passed")


def run_property_tests():
    """Run all property tests."""
    logger.info("🚀 Running idempotency property tests...")
    
    test_suite = TestIdempotencyProperties()
    
    tests = [
        test_suite.test_order_deduplication_property,
        test_suite.test_fill_deduplication_property,
        test_suite.test_websocket_sequence_gap_detection,
        test_suite.test_order_state_consistency,
        test_suite.test_chaos_resilience,
        test_suite.test_persistence_property
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test_suite.setup_method()
            test()
            passed += 1
            logger.info(f"✅ {test.__name__} passed")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test.__name__} failed: {e}")
    
    logger.info(f"📊 Property tests completed: {passed} passed, {failed} failed")
    
    # Save test results
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'idempotency_property_tests',
        'total_tests': len(tests),
        'passed': passed,
        'failed': failed,
        'success_rate': passed / len(tests)
    }
    
    repo_root = Path(".")
    reports_dir = repo_root / "reports"
    os.makedirs(reports_dir / "tests", exist_ok=True)
    results_file = reports_dir / "tests" / "idempotency_property_tests.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Test results saved: {results_file}")
    return results


if __name__ == "__main__":
    run_property_tests()
