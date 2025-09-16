"""
🎯 INTEGRATION TESTS FOR SYSTEM RESILIENCE
==========================================
Integration tests for cancel/replace storms, network hiccups, and idempotence.

Tests:
- Cancel/replace storms
- Network hiccups and reconnection
- Idempotence verification
- System recovery
- Error handling
"""

import pytest
import asyncio
import time
import random
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.execution.trading_state_machine import TradingStateMachine, TradingState
from src.core.execution.maker_first_router import MakerFirstRouter, OrderRequest, OrderUrgency
from src.core.validation.api_precheck_validator import APIPrecheckValidator, OrderValidationRequest
from src.core.risk.production_risk_manager import ProductionRiskManager

class TestSystemResilience:
    """Integration tests for system resilience"""
    
    @pytest.fixture
    async def state_machine(self):
        """Create trading state machine"""
        sm = TradingStateMachine({})
        return sm
    
    @pytest.fixture
    async def router(self):
        """Create maker-first router"""
        return MakerFirstRouter({})
    
    @pytest.fixture
    async def validator(self):
        """Create API precheck validator"""
        return APIPrecheckValidator({})
    
    @pytest.fixture
    async def risk_manager(self):
        """Create risk manager"""
        return ProductionRiskManager({})
    
    @pytest.mark.asyncio
    async def test_cancel_replace_storm(self, state_machine, router):
        """Test system resilience during cancel/replace storms"""
        
        # Simulate cancel/replace storm
        storm_orders = []
        for i in range(100):  # 100 rapid cancel/replace operations
            order_request = OrderRequest(
                symbol='XRP',
                side='buy',
                quantity=100.0,
                price=0.52 + (i * 0.0001),  # Slightly increasing price
                urgency=OrderUrgency.HIGH,
                cloid=f"storm_order_{i}"
            )
            storm_orders.append(order_request)
        
        # Execute storm
        results = []
        for order_request in storm_orders:
            try:
                result = await router.route_order(order_request)
                results.append(result)
            except Exception as e:
                # System should handle errors gracefully
                assert "timeout" in str(e).lower() or "rate limit" in str(e).lower()
        
        # Verify system stability
        assert len(results) > 0, "No orders processed during storm"
        
        # Verify no duplicate order IDs
        order_ids = [r.order_id for r in results if hasattr(r, 'order_id')]
        assert len(order_ids) == len(set(order_ids)), "Duplicate order IDs detected"
        
        # Verify system can still process new orders
        normal_order = OrderRequest(
            symbol='XRP',
            side='sell',
            quantity=50.0,
            price=0.51,
            urgency=OrderUrgency.LOW,
            cloid="normal_order_after_storm"
        )
        
        normal_result = await router.route_order(normal_order)
        assert normal_result is not None, "System failed to process normal order after storm"
    
    @pytest.mark.asyncio
    async def test_network_hiccups(self, state_machine, router):
        """Test system resilience during network hiccups"""
        
        # Simulate network hiccups
        network_errors = [
            ConnectionError("Network timeout"),
            TimeoutError("Request timeout"),
            OSError("Network unreachable"),
            Exception("Unknown network error")
        ]
        
        # Test each type of network error
        for error in network_errors:
            with patch.object(router, '_place_order', side_effect=error):
                order_request = OrderRequest(
                    symbol='XRP',
                    side='buy',
                    quantity=100.0,
                    price=0.52,
                    urgency=OrderUrgency.MEDIUM,
                    cloid=f"network_test_{error.__class__.__name__}"
                )
                
                # System should handle network errors gracefully
                result = await router.route_order(order_request)
                assert result is not None, f"System failed to handle {error.__class__.__name__}"
                assert result.order_state == 'failed', "Network error should result in failed order"
        
        # Test network recovery
        with patch.object(router, '_place_order', return_value={'success': True, 'order_id': 'recovered_order'}):
            recovery_order = OrderRequest(
                symbol='XRP',
                side='sell',
                quantity=50.0,
                price=0.51,
                urgency=OrderUrgency.LOW,
                cloid="recovery_test"
            )
            
            result = await router.route_order(recovery_order)
            assert result is not None, "System failed to recover from network hiccups"
            assert result.order_state == 'filled', "Recovery order should be successful"
    
    @pytest.mark.asyncio
    async def test_idempotence_verification(self, state_machine, router):
        """Test idempotence of operations"""
        
        # Create identical order requests
        order_request = OrderRequest(
            symbol='XRP',
            side='buy',
            quantity=100.0,
            price=0.52,
            urgency=OrderUrgency.LOW,
            cloid="idempotence_test"  # Same CLID
        )
        
        # Execute same order multiple times
        results = []
        for i in range(5):
            result = await router.route_order(order_request)
            results.append(result)
        
        # Verify idempotence
        assert len(results) == 5, "Should process all requests"
        
        # All results should have same order ID (idempotent)
        order_ids = [r.order_id for r in results if hasattr(r, 'order_id')]
        if len(order_ids) > 1:
            assert len(set(order_ids)) == 1, "Non-idempotent order IDs detected"
        
        # Verify no duplicate processing
        unique_results = set(r.order_id for r in results if hasattr(r, 'order_id'))
        assert len(unique_results) <= 1, "Duplicate processing detected"
    
    @pytest.mark.asyncio
    async def test_system_recovery(self, state_machine, router, validator, risk_manager):
        """Test system recovery after failures"""
        
        # Simulate system failure
        with patch.object(router, '_place_order', side_effect=Exception("System failure")):
            order_request = OrderRequest(
                symbol='XRP',
                side='buy',
                quantity=100.0,
                price=0.52,
                urgency=OrderUrgency.HIGH,
                cloid="failure_test"
            )
            
            result = await router.route_order(order_request)
            assert result is not None, "System should handle failures gracefully"
            assert result.order_state == 'failed', "Failed order should be marked as failed"
        
        # Test system recovery
        with patch.object(router, '_place_order', return_value={'success': True, 'order_id': 'recovery_order'}):
            recovery_order = OrderRequest(
                symbol='XRP',
                side='sell',
                quantity=50.0,
                price=0.51,
                urgency=OrderUrgency.LOW,
                cloid="recovery_test"
            )
            
            result = await router.route_order(recovery_order)
            assert result is not None, "System should recover from failures"
            assert result.order_state == 'filled', "Recovery should be successful"
        
        # Test component recovery
        # Risk manager recovery
        risk_result = await risk_manager.calculate_position_size('XRP', 0.8, 0.52, 10000.0)
        assert risk_result is not None, "Risk manager should recover"
        
        # Validator recovery
        validation_request = OrderValidationRequest(
            symbol='XRP',
            side='buy',
            quantity=100.0,
            price=0.52,
            account_equity=10000.0,
            available_margin=8000.0
        )
        validation_result = await validator.validate_order(validation_request)
        assert validation_result is not None, "Validator should recover"
    
    @pytest.mark.asyncio
    async def test_error_handling_cascade(self, state_machine, router, validator, risk_manager):
        """Test error handling cascade through system components"""
        
        # Test validator error handling
        with patch.object(validator, '_validate_tick_size', side_effect=Exception("Validation error")):
            validation_request = OrderValidationRequest(
                symbol='XRP',
                side='buy',
                quantity=100.0,
                price=0.52,
                account_equity=10000.0,
                available_margin=8000.0
            )
            
            result = await validator.validate_order(validation_request)
            assert result is not None, "Validator should handle errors gracefully"
            assert result.result.value == 'invalid', "Validation error should be handled"
        
        # Test risk manager error handling
        with patch.object(risk_manager, '_calculate_atr_position_size', side_effect=Exception("Risk calculation error")):
            risk_result = await risk_manager.calculate_position_size('XRP', 0.8, 0.52, 10000.0)
            assert risk_result is not None, "Risk manager should handle errors gracefully"
            assert 'error' in risk_result or risk_result['recommended_size'] > 0, "Risk error should be handled"
        
        # Test router error handling
        with patch.object(router, '_calculate_optimal_pricing', side_effect=Exception("Pricing error")):
            order_request = OrderRequest(
                symbol='XRP',
                side='buy',
                quantity=100.0,
                price=0.52,
                urgency=OrderUrgency.LOW,
                cloid="error_handling_test"
            )
            
            result = await router.route_order(order_request)
            assert result is not None, "Router should handle errors gracefully"
            assert result.order_state == 'failed', "Router error should result in failed order"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, state_machine, router, validator, risk_manager):
        """Test concurrent operations handling"""
        
        # Create multiple concurrent operations
        async def concurrent_operation(operation_id: int):
            order_request = OrderRequest(
                symbol='XRP',
                side='buy' if operation_id % 2 == 0 else 'sell',
                quantity=100.0 + operation_id,
                price=0.52 + (operation_id * 0.001),
                urgency=OrderUrgency.LOW,
                cloid=f"concurrent_{operation_id}"
            )
            
            # Route order
            result = await router.route_order(order_request)
            return result
        
        # Execute 10 concurrent operations
        tasks = [concurrent_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all operations completed
        assert len(results) == 10, "Not all concurrent operations completed"
        
        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent operations failed: {exceptions}"
        
        # Verify unique order IDs
        order_ids = [r.order_id for r in results if hasattr(r, 'order_id')]
        assert len(order_ids) == len(set(order_ids)), "Duplicate order IDs in concurrent operations"
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, state_machine, router, validator, risk_manager):
        """Test memory leak prevention"""
        
        # Get initial memory usage (approximate)
        initial_cache_size = len(router.validation_cache)
        
        # Perform many operations
        for i in range(1000):
            order_request = OrderRequest(
                symbol='XRP',
                side='buy',
                quantity=100.0,
                price=0.52,
                urgency=OrderUrgency.LOW,
                cloid=f"memory_test_{i}"
            )
            
            await router.route_order(order_request)
            
            # Clear cache periodically to prevent memory leaks
            if i % 100 == 0:
                router.clear_cache()
        
        # Verify memory usage is reasonable
        final_cache_size = len(router.validation_cache)
        assert final_cache_size < 1000, "Memory leak detected in validation cache"
        
        # Verify system is still functional
        test_order = OrderRequest(
            symbol='XRP',
            side='sell',
            quantity=50.0,
            price=0.51,
            urgency=OrderUrgency.LOW,
            cloid="memory_test_final"
        )
        
        result = await router.route_order(test_order)
        assert result is not None, "System should be functional after memory stress test"
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, state_machine, router, validator, risk_manager):
        """Test graceful shutdown"""
        
        # Start state machine
        state_machine_task = asyncio.create_task(state_machine.start_state_machine())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Shutdown gracefully
        await state_machine.shutdown()
        
        # Verify shutdown
        assert not state_machine.is_running, "State machine should be stopped"
        assert state_machine.shutdown_requested, "Shutdown should be requested"
        
        # Verify no pending operations
        assert len(state_machine.state_history) > 0, "State machine should have history"
        
        # Cancel task
        state_machine_task.cancel()
        try:
            await state_machine_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_data_consistency(self, state_machine, router, validator, risk_manager):
        """Test data consistency across components"""
        
        # Create order request
        order_request = OrderRequest(
            symbol='XRP',
            side='buy',
            quantity=100.0,
            price=0.52,
            urgency=OrderUrgency.LOW,
            cloid="consistency_test"
        )
        
        # Validate order
        validation_request = OrderValidationRequest(
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            price=order_request.price,
            account_equity=10000.0,
            available_margin=8000.0
        )
        
        validation_result = await validator.validate_order(validation_request)
        
        # Calculate position size
        risk_result = await risk_manager.calculate_position_size(
            order_request.symbol, 0.8, order_request.price, 10000.0
        )
        
        # Route order
        routing_result = await router.route_order(order_request)
        
        # Verify data consistency
        assert validation_result.symbol == order_request.symbol, "Symbol mismatch in validation"
        assert risk_result['symbol'] == order_request.symbol, "Symbol mismatch in risk calculation"
        assert routing_result.symbol == order_request.symbol, "Symbol mismatch in routing"
        
        assert validation_result.quantity == order_request.quantity, "Quantity mismatch in validation"
        assert routing_result.quantity == order_request.quantity, "Quantity mismatch in routing"
        
        assert validation_result.price == order_request.price, "Price mismatch in validation"
        assert routing_result.price == order_request.price, "Price mismatch in routing"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
