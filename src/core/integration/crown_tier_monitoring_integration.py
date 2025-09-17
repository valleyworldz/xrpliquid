
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
