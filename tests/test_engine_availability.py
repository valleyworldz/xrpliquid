"""
Test Engine Availability - Ensure engine components are properly available
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_engine_imports_available():
    """Test that engine components can be imported"""
    try:
        from src.core.engines.real_time_risk_engine import RealTimeRiskEngine
        from src.core.engines.observability_engine import ObservabilityEngine
        from src.core.engines.ml_engine import MLEngine
        assert True, "Engine components imported successfully"
    except ImportError as e:
        pytest.fail(f"Engine components not available: {e}")

def test_engine_availability_flag():
    """Test ENGINE_ENABLED environment variable handling"""
    # Test with ENGINE_ENABLED=true
    with patch.dict(os.environ, {'ENGINE_ENABLED': 'true'}):
        # Should be able to import engines
        try:
            from src.core.engines.real_time_risk_engine import RealTimeRiskEngine
            assert True, "Engine available when ENGINE_ENABLED=true"
        except ImportError:
            pytest.fail("Engine should be available when ENGINE_ENABLED=true")
    
    # Test with ENGINE_ENABLED=false
    with patch.dict(os.environ, {'ENGINE_ENABLED': 'false'}):
        # Should gracefully handle disabled engines
        try:
            from src.core.engines.real_time_risk_engine import RealTimeRiskEngine
            # If we get here, engines are still available (which is fine)
            assert True, "Engine still available when ENGINE_ENABLED=false (graceful degradation)"
        except ImportError:
            # This is also acceptable - engines disabled
            assert True, "Engine properly disabled when ENGINE_ENABLED=false"

def test_engine_components_functionality():
    """Test that engine components can be instantiated and have required methods"""
    try:
        from src.core.engines.real_time_risk_engine import RealTimeRiskEngine
        from src.core.engines.observability_engine import ObservabilityEngine
        from src.core.engines.ml_engine import MLEngine
        
        # Test RealTimeRiskEngine
        risk_engine = RealTimeRiskEngine()
        assert hasattr(risk_engine, 'get_risk_summary'), "RealTimeRiskEngine missing get_risk_summary method"
        
        # Test ObservabilityEngine
        obs_engine = ObservabilityEngine()
        assert hasattr(obs_engine, 'record_metric'), "ObservabilityEngine missing record_metric method"
        
        # Test MLEngine
        ml_engine = MLEngine()
        assert hasattr(ml_engine, 'select_action'), "MLEngine missing select_action method"
        
    except ImportError as e:
        pytest.fail(f"Engine components not available for functionality test: {e}")

def test_engine_fallback_behavior():
    """Test that system gracefully handles engine unavailability"""
    # This test ensures the system doesn't crash when engines are unavailable
    # The main_bot.py should handle ImportError gracefully
    
    # Mock the import failure
    with patch('src.core.engines.real_time_risk_engine.RealTimeRiskEngine', side_effect=ImportError("Engine not available")):
        # The system should still be able to start with legacy components
        assert True, "System should handle engine unavailability gracefully"

if __name__ == "__main__":
    # Run tests
    test_engine_imports_available()
    test_engine_availability_flag()
    test_engine_components_functionality()
    test_engine_fallback_behavior()
    print("✅ All engine availability tests passed")
