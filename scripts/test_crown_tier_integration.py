
"""
Crown Tier Integration Test
"""

import sys
import os
import logging
from decimal import Decimal

# Add src to path
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

def test_crown_tier_integration():
    """
    Test crown tier integration
    """
    print("🧪 Crown Tier Integration Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Decimal boundary guard
    print("\n🔍 Test 1: Decimal boundary guard...")
    try:
        from src.core.utils.decimal_boundary_guard import safe_float, safe_decimal, enforce_global_decimal_context
        
        # Test safe_float
        result = safe_float(1.23)
        assert str(result) == "1.23", f"Expected 1.23, got {result}"
        
        # Test safe_decimal
        result = safe_decimal("2.34")
        assert str(result) == "2.34", f"Expected 2.34, got {result}"
        
        # Test enforce_global_decimal_context
        enforce_global_decimal_context()
        
        print("  ✅ Decimal boundary guard test passed")
        
    except Exception as e:
        print(f"  ❌ Decimal boundary guard test failed: {e}")
        all_tests_passed = False
    
    # Test 2: Engine availability guard
    print("\n🔍 Test 2: Engine availability guard...")
    try:
        from src.core.engines.engine_availability_guard import enforce_engine_availability, get_engine_status
        
        # Test engine status
        status = get_engine_status()
        assert 'environment' in status, "Engine status missing environment"
        assert 'engine_enabled' in status, "Engine status missing engine_enabled"
        
        print("  ✅ Engine availability guard test passed")
        
    except Exception as e:
        print(f"  ❌ Engine availability guard test failed: {e}")
        all_tests_passed = False
    
    # Test 3: Feasibility enforcer
    print("\n🔍 Test 3: Feasibility enforcer...")
    try:
        from src.core.validation.hard_feasibility_enforcer import check_order_feasibility
        
        # Test feasibility check
        result = check_order_feasibility(
            symbol="XRP/USD",
            side="BUY",
            size=Decimal('1000'),
            price=Decimal('0.52'),
            order_type="LIMIT",
            market_data={
                'depth': {
                    'bids': [{'price': 0.52, 'size': 5000}],
                    'asks': [{'price': 0.521, 'size': 4000}]
                },
                'timestamp': '2025-09-17T10:00:00Z'
            }
        )
        
        assert hasattr(result, 'should_submit_order'), "Feasibility result missing should_submit_order"
        assert hasattr(result, 'overall_passed'), "Feasibility result missing overall_passed"
        
        print("  ✅ Feasibility enforcer test passed")
        
    except Exception as e:
        print(f"  ❌ Feasibility enforcer test failed: {e}")
        all_tests_passed = False
    
    # Test 4: Crown tier monitor
    print("\n🔍 Test 4: Crown tier monitor...")
    try:
        from src.core.monitoring.crown_tier_monitor import (
            log_order_submitted, log_order_blocked, update_performance_score,
            get_crown_tier_report
        )
        
        # Test logging functions
        log_order_submitted("XRP/USD", "BUY", Decimal('1000'), Decimal('0.52'))
        log_order_blocked("XRP/USD", "Test block")
        update_performance_score(8.5)
        
        # Test report generation
        report = get_crown_tier_report()
        assert 'crown_tier_status' in report, "Report missing crown_tier_status"
        assert 'health_score' in report, "Report missing health_score"
        
        print("  ✅ Crown tier monitor test passed")
        
    except Exception as e:
        print(f"  ❌ Crown tier monitor test failed: {e}")
        all_tests_passed = False
    
    # Test 5: Integration modules
    print("\n🔍 Test 5: Integration modules...")
    try:
        from src.core.integration.crown_tier_monitoring_integration import integrate_crown_tier_monitoring
        from src.core.integration.feasibility_gate_integration import integrate_feasibility_gates
        from src.core.integration.engine_availability_integration import integrate_engine_availability
        
        # Test integrations
        monitoring_result = integrate_crown_tier_monitoring()
        feasibility_result = integrate_feasibility_gates()
        engine_result = integrate_engine_availability()
        
        print("  ✅ Integration modules test passed")
        
    except Exception as e:
        print(f"  ❌ Integration modules test failed: {e}")
        all_tests_passed = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🏆 CROWN TIER INTEGRATION TEST: PASSED")
        print("✅ All crown tier fixes integrated successfully")
        return 0
    else:
        print("❌ CROWN TIER INTEGRATION TEST: FAILED")
        print("❌ Some integration tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(test_crown_tier_integration())
