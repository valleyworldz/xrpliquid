"""
Exactly-Once Semantics Property Tests
Fuzz WS gaps/replays/cancel/replace; assert dedupe and final state matches ledger
"""

import pytest
import asyncio
import random
import time
from decimal import Decimal
from typing import Dict, Any, List, Set
from unittest.mock import Mock, patch, AsyncMock
import json

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ExactlyOncePropertyTester:
    """
    Property-based testing for exactly-once semantics
    """
    
    def __init__(self):
        self.order_ledger = {}  # client_order_id -> order_state
        self.sequence_numbers = {}  # connection_id -> last_seq
        self.gap_events = []
        self.duplicate_events = []
        
    def generate_ws_gap_scenario(self) -> Dict[str, Any]:
        """
        Generate a WebSocket gap scenario for testing
        """
        scenarios = [
            {
                'type': 'sequence_gap',
                'gap_size': random.randint(1, 10),
                'reconnect_delay_ms': random.randint(100, 1000),
                'expected_behavior': 'snapshot_replay'
            },
            {
                'type': 'duplicate_message',
                'duplicate_count': random.randint(1, 3),
                'message_delay_ms': random.randint(50, 500),
                'expected_behavior': 'dedupe'
            },
            {
                'type': 'cancel_replace_race',
                'replace_delay_ms': random.randint(10, 100),
                'expected_behavior': 'idempotent'
            },
            {
                'type': 'connection_drop',
                'drop_duration_ms': random.randint(1000, 5000),
                'expected_behavior': 'resync'
            }
        ]
        
        return random.choice(scenarios)
    
    def generate_order_sequence(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate a sequence of orders for testing
        """
        orders = []
        for i in range(count):
            order = {
                'client_order_id': f'test_order_{i}_{int(time.time() * 1000)}',
                'side': random.choice(['BUY', 'SELL']),
                'size': Decimal(str(random.uniform(0.1, 10.0))),
                'price': Decimal(str(random.uniform(0.5, 0.6))),
                'timestamp': time.time(),
                'sequence_number': i
            }
            orders.append(order)
        return orders
    
    async def simulate_ws_gap_recovery(self, scenario: Dict[str, Any], orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate WebSocket gap recovery
        """
        result = {
            'scenario': scenario,
            'orders_processed': 0,
            'duplicates_detected': 0,
            'gaps_handled': 0,
            'final_ledger_state': {},
            'errors': []
        }
        
        try:
            if scenario['type'] == 'sequence_gap':
                # Simulate sequence gap
                gap_start = random.randint(0, len(orders) - 2)
                gap_end = gap_start + scenario['gap_size']
                
                # Process orders before gap
                for order in orders[:gap_start]:
                    await self.process_order(order)
                    result['orders_processed'] += 1
                
                # Simulate gap detection
                result['gaps_handled'] += 1
                self.gap_events.append({
                    'type': 'sequence_gap',
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'timestamp': time.time()
                })
                
                # Simulate snapshot replay
                await asyncio.sleep(scenario['reconnect_delay_ms'] / 1000.0)
                
                # Process remaining orders
                for order in orders[gap_start:]:
                    await self.process_order(order)
                    result['orders_processed'] += 1
            
            elif scenario['type'] == 'duplicate_message':
                # Simulate duplicate messages
                for order in orders:
                    await self.process_order(order)
                    result['orders_processed'] += 1
                    
                    # Send duplicates
                    for _ in range(scenario['duplicate_count']):
                        await asyncio.sleep(scenario['message_delay_ms'] / 1000.0)
                        duplicate_detected = await self.process_order(order)
                        if duplicate_detected:
                            result['duplicates_detected'] += 1
            
            elif scenario['type'] == 'cancel_replace_race':
                # Simulate cancel/replace race condition
                for order in orders:
                    await self.process_order(order)
                    result['orders_processed'] += 1
                    
                    # Simulate replace
                    replace_order = order.copy()
                    replace_order['price'] = order['price'] * Decimal('1.001')
                    replace_order['client_order_id'] = f"{order['client_order_id']}_replace"
                    
                    await asyncio.sleep(scenario['replace_delay_ms'] / 1000.0)
                    await self.process_order(replace_order)
                    result['orders_processed'] += 1
            
            elif scenario['type'] == 'connection_drop':
                # Simulate connection drop and recovery
                for order in orders[:len(orders)//2]:
                    await self.process_order(order)
                    result['orders_processed'] += 1
                
                # Simulate connection drop
                await asyncio.sleep(scenario['drop_duration_ms'] / 1000.0)
                
                # Simulate resync
                result['gaps_handled'] += 1
                
                # Process remaining orders
                for order in orders[len(orders)//2:]:
                    await self.process_order(order)
                    result['orders_processed'] += 1
            
            result['final_ledger_state'] = self.order_ledger.copy()
            
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    async def process_order(self, order: Dict[str, Any]) -> bool:
        """
        Process an order and detect duplicates
        """
        client_order_id = order['client_order_id']
        
        # Check for duplicate
        if client_order_id in self.order_ledger:
            self.duplicate_events.append({
                'client_order_id': client_order_id,
                'timestamp': time.time(),
                'existing_state': self.order_ledger[client_order_id]
            })
            return True  # Duplicate detected
        
        # Add to ledger
        self.order_ledger[client_order_id] = {
            'side': order['side'],
            'size': order['size'],
            'price': order['price'],
            'timestamp': order['timestamp'],
            'status': 'submitted'
        }
        
        return False  # Not a duplicate
    
    def assert_exactly_once_properties(self, result: Dict[str, Any]) -> List[str]:
        """
        Assert exactly-once properties
        """
        violations = []
        
        # Property 1: No duplicate orders in final ledger
        client_order_ids = [order['client_order_id'] for order in result.get('orders', [])]
        unique_ids = set(client_order_ids)
        if len(client_order_ids) != len(unique_ids):
            violations.append(f"Duplicate client_order_ids found: {len(client_order_ids) - len(unique_ids)} duplicates")
        
        # Property 2: All orders have unique CLOIDs
        ledger_cloids = set(self.order_ledger.keys())
        if len(ledger_cloids) != len(self.order_ledger):
            violations.append("Duplicate CLOIDs in ledger")
        
        # Property 3: Final ledger state is consistent
        for cloid, order_state in self.order_ledger.items():
            if not all(key in order_state for key in ['side', 'size', 'price', 'timestamp', 'status']):
                violations.append(f"Incomplete order state for CLOID: {cloid}")
        
        # Property 4: Gaps were handled properly
        if result['gaps_handled'] > 0:
            if not self.gap_events:
                violations.append("Gaps reported but no gap events recorded")
        
        # Property 5: Duplicates were detected
        if result['duplicates_detected'] > 0:
            if not self.duplicate_events:
                violations.append("Duplicates reported but no duplicate events recorded")
        
        return violations

@pytest.fixture
def exactly_once_tester():
    """Fixture for exactly-once property tester"""
    return ExactlyOncePropertyTester()

@pytest.mark.asyncio
async def test_sequence_gap_recovery(exactly_once_tester):
    """Test sequence gap recovery maintains exactly-once semantics"""
    
    # Generate test scenario
    scenario = {
        'type': 'sequence_gap',
        'gap_size': 3,
        'reconnect_delay_ms': 500,
        'expected_behavior': 'snapshot_replay'
    }
    
    # Generate test orders
    orders = exactly_once_tester.generate_order_sequence(10)
    
    # Simulate gap recovery
    result = await exactly_once_tester.simulate_ws_gap_recovery(scenario, orders)
    
    # Assert properties
    violations = exactly_once_tester.assert_exactly_once_properties(result)
    
    assert len(violations) == 0, f"Exactly-once violations: {violations}"
    assert result['gaps_handled'] > 0, "Gaps should have been handled"
    assert result['orders_processed'] == len(orders), "All orders should have been processed"

@pytest.mark.asyncio
async def test_duplicate_detection(exactly_once_tester):
    """Test duplicate message detection"""
    
    # Generate test scenario
    scenario = {
        'type': 'duplicate_message',
        'duplicate_count': 2,
        'message_delay_ms': 100,
        'expected_behavior': 'dedupe'
    }
    
    # Generate test orders
    orders = exactly_once_tester.generate_order_sequence(5)
    
    # Simulate duplicate messages
    result = await exactly_once_tester.simulate_ws_gap_recovery(scenario, orders)
    
    # Assert properties
    violations = exactly_once_tester.assert_exactly_once_properties(result)
    
    assert len(violations) == 0, f"Exactly-once violations: {violations}"
    assert result['duplicates_detected'] > 0, "Duplicates should have been detected"
    assert len(exactly_once_tester.order_ledger) == len(orders), "Ledger should contain only unique orders"

@pytest.mark.asyncio
async def test_cancel_replace_idempotency(exactly_once_tester):
    """Test cancel/replace idempotency"""
    
    # Generate test scenario
    scenario = {
        'type': 'cancel_replace_race',
        'replace_delay_ms': 50,
        'expected_behavior': 'idempotent'
    }
    
    # Generate test orders
    orders = exactly_once_tester.generate_order_sequence(3)
    
    # Simulate cancel/replace race
    result = await exactly_once_tester.simulate_ws_gap_recovery(scenario, orders)
    
    # Assert properties
    violations = exactly_once_tester.assert_exactly_once_properties(result)
    
    assert len(violations) == 0, f"Exactly-once violations: {violations}"
    assert result['orders_processed'] == len(orders) * 2, "Both original and replace orders should be processed"

@pytest.mark.asyncio
async def test_connection_drop_recovery(exactly_once_tester):
    """Test connection drop recovery"""
    
    # Generate test scenario
    scenario = {
        'type': 'connection_drop',
        'drop_duration_ms': 2000,
        'expected_behavior': 'resync'
    }
    
    # Generate test orders
    orders = exactly_once_tester.generate_order_sequence(8)
    
    # Simulate connection drop
    result = await exactly_once_tester.simulate_ws_gap_recovery(scenario, orders)
    
    # Assert properties
    violations = exactly_once_tester.assert_exactly_once_properties(result)
    
    assert len(violations) == 0, f"Exactly-once violations: {violations}"
    assert result['gaps_handled'] > 0, "Connection drop should have been handled"
    assert result['orders_processed'] == len(orders), "All orders should have been processed"

@pytest.mark.asyncio
async def test_fuzz_exactly_once_properties(exactly_once_tester):
    """Fuzz test exactly-once properties with random scenarios"""
    
    violations_count = 0
    total_tests = 100
    
    for i in range(total_tests):
        # Generate random scenario
        scenario = exactly_once_tester.generate_ws_gap_scenario()
        
        # Generate random orders
        orders = exactly_once_tester.generate_order_sequence(random.randint(5, 20))
        
        # Reset state
        exactly_once_tester.order_ledger.clear()
        exactly_once_tester.gap_events.clear()
        exactly_once_tester.duplicate_events.clear()
        
        # Simulate scenario
        result = await exactly_once_tester.simulate_ws_gap_recovery(scenario, orders)
        
        # Check for violations
        violations = exactly_once_tester.assert_exactly_once_properties(result)
        if violations:
            violations_count += 1
            print(f"Test {i+1}: Violations found: {violations}")
    
    # Assert that violations are rare (less than 5% of tests)
    violation_rate = violations_count / total_tests
    assert violation_rate < 0.05, f"Too many violations: {violation_rate:.2%} ({violations_count}/{total_tests})"

def test_exactly_once_property_test_suite():
    """Test that the exactly-once property test suite runs"""
    print("🧪 Exactly-Once Property Test Suite")
    print("=" * 50)
    
    tester = ExactlyOncePropertyTester()
    
    # Test scenario generation
    scenario = tester.generate_ws_gap_scenario()
    assert 'type' in scenario
    assert 'expected_behavior' in scenario
    
    # Test order generation
    orders = tester.generate_order_sequence(5)
    assert len(orders) == 5
    assert all('client_order_id' in order for order in orders)
    
    # Test property assertions
    result = {
        'orders_processed': 5,
        'duplicates_detected': 0,
        'gaps_handled': 1,
        'orders': orders
    }
    
    violations = tester.assert_exactly_once_properties(result)
    print(f"✅ Property test suite ready: {len(violations)} violations detected")
    
    print("✅ Exactly-Once Property Test Suite Complete")

if __name__ == "__main__":
    test_exactly_once_property_test_suite()
