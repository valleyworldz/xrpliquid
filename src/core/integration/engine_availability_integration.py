
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
