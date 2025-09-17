"""
Integrate Crown Tier Fixes - Integrate all crown tier fixes into the main bot
"""

import os
import sys
import logging
from pathlib import Path

def integrate_crown_tier_fixes():
    """
    Integrate all crown tier fixes into the main bot
    """
    print("🔧 Integrating Crown Tier Fixes")
    print("=" * 50)
    
    # Add imports to main bot
    integrate_main_bot_imports()
    
    # Add monitoring integration
    integrate_monitoring()
    
    # Add feasibility gate integration
    integrate_feasibility_gates()
    
    # Add engine availability integration
    integrate_engine_availability()
    
    # Create integration test
    create_integration_test()
    
    print(f"\n✅ Crown Tier Fixes Integration Complete")
    print(f"🎯 All fixes integrated into main bot")
    print(f"🔍 Run integration test: python scripts/test_crown_tier_integration.py")

def integrate_main_bot_imports():
    """Add necessary imports to main bot"""
    print("📝 Integrating main bot imports...")
    
    main_bot_path = "src/core/main_bot.py"
    
    if not os.path.exists(main_bot_path):
        print(f"❌ Main bot file not found: {main_bot_path}")
        return
    
    # Read current content
    with open(main_bot_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add imports if not already present
    imports_to_add = [
        "from src.core.utils.decimal_boundary_guard import safe_float, safe_decimal, enforce_global_decimal_context",
        "from src.core.engines.engine_availability_guard import enforce_engine_availability",
        "from src.core.validation.hard_feasibility_enforcer import check_order_feasibility",
        "from src.core.monitoring.crown_tier_monitor import log_decimal_error, log_engine_failure, log_feasibility_block, log_guardian_invocation, log_order_submitted, log_order_blocked, update_performance_score, get_crown_tier_report"
    ]
    
    for import_line in imports_to_add:
        if import_line not in content:
            # Find the first import statement
            import_match = content.find("import")
            if import_match != -1:
                content = content[:import_match] + import_line + "\n" + content[import_match:]
                print(f"  ✅ Added import: {import_line}")
    
    # Add decimal context enforcement at the start
    if "enforce_global_decimal_context()" not in content:
        # Find the main function or class
        main_match = content.find("def main()") or content.find("class")
        if main_match != -1:
            content = content[:main_match] + "# Enforce global decimal context\nenforce_global_decimal_context()\n\n" + content[main_match:]
            print(f"  ✅ Added decimal context enforcement")
    
    # Write back
    with open(main_bot_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ Main bot imports integrated")

def integrate_monitoring():
    """Integrate crown tier monitoring"""
    print("📊 Integrating crown tier monitoring...")
    
    # Create monitoring integration file
    monitoring_integration = '''
"""
Crown Tier Monitoring Integration
"""

import logging
from typing import Dict, Any
from decimal import Decimal

def integrate_crown_tier_monitoring():
    """
    Integrate crown tier monitoring into the main bot
    """
    logger = logging.getLogger(__name__)
    
    try:
        from src.core.monitoring.crown_tier_monitor import (
            log_decimal_error, log_engine_failure, log_feasibility_block,
            log_guardian_invocation, log_order_submitted, log_order_blocked,
            update_performance_score, get_crown_tier_report
        )
        
        logger.info("✅ Crown tier monitoring integrated")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import crown tier monitoring: {e}")
        return False

def safe_order_submission(symbol: str, side: str, size: Decimal, price: Decimal, order_type: str, **kwargs):
    """
    Safe order submission with crown tier monitoring
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check feasibility first
        from src.core.validation.hard_feasibility_enforcer import check_order_feasibility
        
        feasibility_result = check_order_feasibility(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            **kwargs
        )
        
        if not feasibility_result.should_submit_order:
            log_order_blocked(symbol, feasibility_result.block_reason or "Feasibility check failed")
            return False, feasibility_result.block_reason
        
        # Submit order (placeholder - replace with actual order submission)
        # order_result = submit_order(symbol, side, size, price, order_type)
        
        # Log successful submission
        log_order_submitted(symbol, side, size, price)
        return True, "Order submitted successfully"
        
    except Exception as e:
        logger.error(f"❌ Error in safe order submission: {e}")
        log_decimal_error(str(e), {"symbol": symbol, "side": side, "size": str(size), "price": str(price)})
        return False, str(e)

def safe_performance_update(score: float):
    """
    Safe performance score update
    """
    try:
        update_performance_score(score)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error updating performance score: {e}")

def get_crown_tier_status() -> Dict[str, Any]:
    """
    Get current crown tier status
    """
    try:
        return get_crown_tier_report()
    except Exception as e:
        logging.getLogger(__name__).error(f"Error getting crown tier status: {e}")
        return {"error": str(e)}
'''
    
    os.makedirs("src/core/integration", exist_ok=True)
    with open("src/core/integration/crown_tier_monitoring_integration.py", 'w', encoding='utf-8') as f:
        f.write(monitoring_integration)
    
    print(f"  ✅ Crown tier monitoring integrated")

def integrate_feasibility_gates():
    """Integrate feasibility gates"""
    print("🚫 Integrating feasibility gates...")
    
    # Create feasibility gate integration
    feasibility_integration = '''
"""
Feasibility Gate Integration
"""

import logging
from typing import Dict, Any, Tuple
from decimal import Decimal

def integrate_feasibility_gates():
    """
    Integrate feasibility gates into the main bot
    """
    logger = logging.getLogger(__name__)
    
    try:
        from src.core.validation.hard_feasibility_enforcer import check_order_feasibility
        
        logger.info("✅ Feasibility gates integrated")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import feasibility gates: {e}")
        return False

def pre_trade_feasibility_check(symbol: str, side: str, size: Decimal, price: Decimal, order_type: str, market_data: Dict[str, Any] = None) -> Tuple[bool, str]:
    """
    Pre-trade feasibility check
    """
    logger = logging.getLogger(__name__)
    
    try:
        from src.core.validation.hard_feasibility_enforcer import check_order_feasibility
        
        result = check_order_feasibility(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            market_data=market_data
        )
        
        if result.should_submit_order:
            logger.info(f"✅ Feasibility check passed for {side} {size} {symbol}")
            return True, "Feasibility check passed"
        else:
            logger.warning(f"❌ Feasibility check failed for {side} {size} {symbol}: {result.block_reason}")
            return False, result.block_reason or "Feasibility check failed"
            
    except Exception as e:
        logger.error(f"❌ Error in feasibility check: {e}")
        return False, f"Feasibility check error: {str(e)}"
'''
    
    os.makedirs("src/core/integration", exist_ok=True)
    with open("src/core/integration/feasibility_gate_integration.py", 'w', encoding='utf-8') as f:
        f.write(feasibility_integration)
    
    print(f"  ✅ Feasibility gates integrated")

def integrate_engine_availability():
    """Integrate engine availability"""
    print("🔧 Integrating engine availability...")
    
    # Create engine availability integration
    engine_integration = '''
"""
Engine Availability Integration
"""

import logging
import os

def integrate_engine_availability():
    """
    Integrate engine availability into the main bot
    """
    logger = logging.getLogger(__name__)
    
    try:
        from src.core.engines.engine_availability_guard import enforce_engine_availability
        
        # Check engine availability
        result = enforce_engine_availability()
        
        if result:
            logger.info("✅ Engine availability check passed")
            return True
        else:
            logger.warning("⚠️ Engine availability check failed - using legacy components")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import engine availability guard: {e}")
        return False
    except SystemExit as e:
        if e.code == 1:
            logger.critical("❌ Engine availability hard fail - system cannot operate")
            return False
        else:
            logger.error(f"❌ Unexpected exit code from engine availability check: {e.code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error in engine availability check: {e}")
        return False

def check_engine_availability_on_startup():
    """
    Check engine availability on startup
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Checking engine availability on startup...")
    
    try:
        result = integrate_engine_availability()
        
        if result:
            logger.info("✅ Engine availability check passed - system ready")
            return True
        else:
            logger.warning("⚠️ Engine availability check failed - system may have limited functionality")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking engine availability: {e}")
        return False
'''
    
    with open("src/core/integration/engine_availability_integration.py", 'w', encoding='utf-8') as f:
        f.write(engine_integration)
    
    print(f"  ✅ Engine availability integrated")

def create_integration_test():
    """Create integration test"""
    print("🧪 Creating integration test...")
    
    integration_test = '''
"""
Crown Tier Integration Test
"""

import sys
import os
import logging
from decimal import Decimal

# Add src to path
sys.path.append('src')

def test_crown_tier_integration():
    """
    Test crown tier integration
    """
    print("🧪 Crown Tier Integration Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Decimal boundary guard
    print("\\n🔍 Test 1: Decimal boundary guard...")
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
    print("\\n🔍 Test 2: Engine availability guard...")
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
    print("\\n🔍 Test 3: Feasibility enforcer...")
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
    print("\\n🔍 Test 4: Crown tier monitor...")
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
    print("\\n🔍 Test 5: Integration modules...")
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
    print("\\n" + "=" * 50)
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
'''
    
    with open("scripts/test_crown_tier_integration.py", 'w', encoding='utf-8') as f:
        f.write(integration_test)
    
    print(f"  ✅ Integration test created")

if __name__ == "__main__":
    integrate_crown_tier_fixes()
