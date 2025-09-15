"""
Transfer Execution for Sweep Engine

Handles the actual USDC perp→spot transfers via Hyperliquid API.
"""

import time
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def transfer_perp_to_spot(exchange, amount_usdc: float, chain: str = "Mainnet") -> Dict[str, Any]:
    """
    Execute a perp→spot USDC transfer via Hyperliquid usdClassTransfer.
    
    Args:
        exchange: Hyperliquid Exchange client with usd_class_transfer method
        amount_usdc: Amount to transfer (USDC)
        chain: Chain identifier ("Mainnet" or "Testnet")
    
    Returns:
        Dict with transfer result: {"success": bool, "response": dict, "error": str}
    """
    try:
        # Use the Exchange client's usd_class_transfer method
        # toPerp=False means we're transferring FROM perps TO spot
        result = exchange.usd_class_transfer(amount=amount_usdc, to_perp=False)
        
        # Check if the response indicates success
        success = False
        if isinstance(result, dict):
            # Check various success indicators
            if result.get("status") == "ok":
                success = True
            elif result.get("response", {}).get("type") == "ok":
                success = True
            elif "error" not in result and "err" not in result:
                # Assume success if no explicit error
                success = True
        
        return {
            "success": success,
            "response": result,
            "error": result.get("error") or result.get("err") or "" if not success else ""
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Transfer failed: {error_msg}")
        
        return {
            "success": False,
            "response": {},
            "error": error_msg
        }


def validate_transfer_params(amount_usdc: float, min_amount: float = 1.0) -> Optional[str]:
    """
    Validate transfer parameters before execution.
    
    Args:
        amount_usdc: Amount to transfer
        min_amount: Minimum allowed amount
    
    Returns:
        Error message if invalid, None if valid
    """
    if amount_usdc <= 0:
        return "Amount must be positive"
    
    if amount_usdc < min_amount:
        return f"Amount {amount_usdc:.2f} below minimum {min_amount:.2f}"
    
    # Check for reasonable maximum (safety check)
    max_amount = 100000.0  # $100k safety limit
    if amount_usdc > max_amount:
        return f"Amount {amount_usdc:.2f} exceeds safety limit {max_amount:.2f}"
    
    return None


def estimate_transfer_time() -> float:
    """
    Estimate transfer completion time in seconds.
    Hyperliquid transfers are typically very fast.
    
    Returns:
        Estimated time in seconds
    """
    return 5.0  # Conservative estimate for Hyperliquid


def format_transfer_amount(amount: float) -> float:
    """
    Format transfer amount to proper precision.
    
    Args:
        amount: Raw amount
    
    Returns:
        Formatted amount (rounded to 2 decimal places)
    """
    return round(float(amount), 2)
