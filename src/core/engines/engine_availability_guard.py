"""
Engine Availability Guard - Hard fail in production if engines missing
"""

import os
import logging
import sys
from typing import Dict, Any, Optional

class EngineAvailabilityGuard:
    """
    Hard fail in production if engine components are missing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.environment = os.getenv('ENVIRONMENT', 'dev').lower()
        self.engine_enabled = os.getenv('ENGINE_ENABLED', 'true').lower() == 'true'
        self.required_engines = [
            'RealTimeRiskEngine',
            'ObservabilityEngine', 
            'MLEngine'
        ]
        self.available_engines = {}
        self.missing_engines = []
        
    def check_engine_availability(self) -> Dict[str, Any]:
        """
        Check if all required engines are available
        """
        try:
            # Import engine components
            from src.core.engines.real_time_risk_engine import RealTimeRiskEngine
            from src.core.engines.observability_engine import ObservabilityEngine
            from src.core.engines.ml_engine import MLEngine
            
            self.available_engines = {
                'RealTimeRiskEngine': RealTimeRiskEngine,
                'ObservabilityEngine': ObservabilityEngine,
                'MLEngine': MLEngine
            }
            
            self.missing_engines = []
            
            return {
                'available': True,
                'engines': list(self.available_engines.keys()),
                'missing': self.missing_engines,
                'environment': self.environment,
                'engine_enabled': self.engine_enabled
            }
            
        except ImportError as e:
            self.missing_engines = [str(e)]
            return {
                'available': False,
                'engines': [],
                'missing': self.missing_engines,
                'environment': self.environment,
                'engine_enabled': self.engine_enabled,
                'error': str(e)
            }
    
    def enforce_engine_availability(self) -> bool:
        """
        Enforce engine availability based on environment and settings
        """
        availability = self.check_engine_availability()
        
        # CROWN TIER: Fail-closed behavior - if ENGINE_ENABLED=true, engines are required in ALL environments
        if self.engine_enabled:
            if not availability['available']:
                error_msg = f"❌ ENGINE_HARD_FAIL: ENGINE_ENABLED=true but required engines missing: {availability['missing']}"
                self.logger.critical(error_msg)
                print(error_msg)
                print(f"🔒 CROWN TIER FAIL-CLOSED: System cannot operate without required engines")
                sys.exit(1)
            else:
                self.logger.info(f"✅ ENGINE_AVAILABILITY_PASS: All engines available (ENGINE_ENABLED=true)")
                print(f"✅ ENGINE_AVAILABILITY_PASS: All engines available (ENGINE_ENABLED=true)")
                return True
        
        # Only allow soft fallback if ENGINE_ENABLED=false
        elif not availability['available']:
            self.logger.warning(f"⚠️ ENGINE_SOFT_FALLBACK: ENGINE_ENABLED=false, engines not available in {self.environment}, using legacy components")
            print(f"⚠️ ENGINE_SOFT_FALLBACK: ENGINE_ENABLED=false, engines not available in {self.environment}, using legacy components")
            return False
        
        else:
            self.logger.info(f"✅ ENGINE_AVAILABILITY_PASS: All engines available (ENGINE_ENABLED=false)")
            print(f"✅ ENGINE_AVAILABILITY_PASS: All engines available (ENGINE_ENABLED=false)")
            return True
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get current engine status for monitoring
        """
        availability = self.check_engine_availability()
        
        return {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'environment': self.environment,
            'engine_enabled': self.engine_enabled,
            'engines_available': availability['available'],
            'available_engines': availability['engines'],
            'missing_engines': availability['missing'],
            'status': 'HARD_FAIL' if (self.engine_enabled and not availability['available']) else 'OK'
        }

# Global guard instance
_engine_guard = EngineAvailabilityGuard()

def enforce_engine_availability() -> bool:
    """
    Global function to enforce engine availability
    """
    return _engine_guard.enforce_engine_availability()

def get_engine_status() -> Dict[str, Any]:
    """
    Get engine status for monitoring
    """
    return _engine_guard.get_engine_status()

# Demo function
def demo_engine_availability_guard():
    """Demo the engine availability guard"""
    print("🔧 Engine Availability Guard Demo")
    print("=" * 50)
    
    # Test in different environments
    test_environments = ['dev', 'test', 'production']
    
    for env in test_environments:
        print(f"\n🔍 Testing environment: {env}")
        os.environ['ENVIRONMENT'] = env
        
        guard = EngineAvailabilityGuard()
        status = guard.get_engine_status()
        
        print(f"  Environment: {status['environment']}")
        print(f"  Engine Enabled: {status['engine_enabled']}")
        print(f"  Engines Available: {status['engines_available']}")
        print(f"  Available Engines: {status['available_engines']}")
        print(f"  Missing Engines: {status['missing_engines']}")
        print(f"  Status: {status['status']}")
    
    print(f"\n✅ Engine Availability Guard Demo Complete")

if __name__ == "__main__":
    demo_engine_availability_guard()
