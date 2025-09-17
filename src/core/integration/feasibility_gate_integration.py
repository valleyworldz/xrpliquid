
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
