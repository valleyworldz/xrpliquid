"""
Decimal Guard - Import-time decimal context setup
"""

import os
import logging
from decimal import getcontext, ROUND_HALF_EVEN, setcontext, Context

# Set up decimal context at import time
def setup_decimal_context():
    """Setup decimal context with proper precision and rounding"""
    try:
        # Get current context
        C = getcontext()
        
        # Set precision to 10 decimal places
        C.prec = 10
        
        # Set rounding mode to ROUND_HALF_EVEN (banker's rounding)
        C.rounding = ROUND_HALF_EVEN
        
        # Clear traps to prevent exceptions
        C.clear_traps()
        
        # Log the setup
        logger = logging.getLogger(__name__)
        logger.info("🔢 DECIMAL_GUARD_ACTIVE: context=ROUND_HALF_EVEN, precision=10")
        
        return True
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ DECIMAL_GUARD_FAILED: {e}")
        return False

# Setup decimal context immediately on import
DECIMAL_GUARD_ACTIVE = setup_decimal_context()

# Export the context for use in other modules
DECIMAL_CONTEXT = getcontext()

# Banner for runbook
if DECIMAL_GUARD_ACTIVE:
    print("🔢 DECIMAL_NORMALIZER_ACTIVE context=ROUND_HALF_EVEN precision=10")
