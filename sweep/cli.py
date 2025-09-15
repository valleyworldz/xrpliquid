"""
Command Line Interface for Sweep Engine

Standalone CLI for testing and manual sweep operations.
"""

import argparse
import time
import sys
import logging
from typing import Callable, Any, Dict, Optional

from .config import SweepCfg
from .state import SweepState  
from .engine import maybe_sweep_to_spot
from .volatility import simple_vol_ratio_from_prices


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging for CLI"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_mock_functions():
    """Create mock functions for testing when no real bot is available"""
    
    def mock_get_user_state() -> Dict[str, Any]:
        return {
            "withdrawable": "1000.0",
            "marginSummary": {"accountValue": "2000.0"},
            "assetPositions": [
                {
                    "position": {
                        "coin": "XRP",
                        "szi": "500.0",
                        "liquidationPx": "1.50"
                    }
                }
            ]
        }
    
    def mock_get_position() -> Optional[Dict[str, Any]]:
        return {
            "size": 500.0,
            "is_long": True,
            "coin": "XRP"
        }
    
    def mock_get_mark_px() -> float:
        return 2.00  # $2.00 XRP
    
    def mock_get_returns() -> list:
        # Mock daily returns for volatility calculation
        return [0.02, -0.01, 0.015, -0.005, 0.01] * 10
    
    def mock_get_next_funding() -> float:
        return 0.0001  # 1 basis point
    
    def mock_get_position_notional() -> float:
        return 1000.0  # $1000 notional
    
    return (
        mock_get_user_state,
        mock_get_position, 
        mock_get_mark_px,
        mock_get_returns,
        mock_get_next_funding,
        mock_get_position_notional
    )


class MockExchange:
    """Mock exchange for testing"""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.transfer_count = 0
    
    def usd_class_transfer(self, amount: float, to_perp: bool) -> Dict[str, Any]:
        """Mock transfer method"""
        self.transfer_count += 1
        
        if self.should_fail:
            raise Exception("Mock transfer failure")
        
        return {
            "status": "ok",
            "type": "usdClassTransfer", 
            "amount": f"{amount:.2f}",
            "toPerp": to_perp
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Sweep Engine CLI - Test and execute perp->spot sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sweep.cli --dry-run              # Test with mock data
  python -m sweep.cli --force-sweep          # Force a sweep ignoring gates
  python -m sweep.cli --mock-fail --dry-run  # Test failure handling
        """
    )
    
    parser.add_argument("--force-sweep", action="store_true", 
                       help="Force a sweep ignoring gates except staleness")
    parser.add_argument("--dry-run", action="store_true",
                       help="Do not call transfer; print decision only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--mock-fail", action="store_true",
                       help="Mock transfer failures for testing")
    parser.add_argument("--coin", default="XRP",
                       help="Coin symbol (default: XRP)")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Load configuration
    cfg = SweepCfg()
    logger.info(f"Loaded config: enabled={cfg.enabled}, chain={cfg.chain}")
    
    # Load state
    state = SweepState(cfg.accumulator_file)
    state.load()
    logger.info(f"Loaded state: last_sweep={state.last_sweep_ts}, pending={state.pending_accum}")
    
    # Create mock functions and exchange
    (get_user_state, get_position, get_mark_px, 
     get_returns, get_next_funding, get_position_notional) = create_mock_functions()
    
    exchange = MockExchange(should_fail=args.mock_fail)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual transfers will be executed")
        # Override the exchange with a dry-run version
        class DryRunExchange:
            def usd_class_transfer(self, amount: float, to_perp: bool):
                logger.info(f"DRY RUN: Would transfer ${amount:.2f} USDC (toPerp={to_perp})")
                return {"status": "ok", "dry_run": True}
        exchange = DryRunExchange()
    
    # Gather inputs
    try:
        user_state = get_user_state()
        pos = get_position()
        mark_px = get_mark_px()
        returns = get_returns()
        next_funding = get_next_funding()
        notional = get_position_notional()
        
        logger.info(f"Market data: mark_px=${mark_px:.4f}, position_notional=${notional:.2f}")
        
        # Calculate volatility ratio
        vol_ratio = simple_vol_ratio_from_prices([mark_px] * 50, 24, 48)  # Mock prices
        logger.info(f"Volatility ratio: {vol_ratio:.2f}")
        
        # Execute sweep decision
        result = maybe_sweep_to_spot(
            exchange=exchange,
            state=state,
            cfg=cfg,
            user_state=user_state,
            pos=pos,
            mark_px=mark_px,
            vol_ratio=vol_ratio,
            next_hour_funding_rate=next_funding,
            position_notional=notional,
            coin=args.coin,
            force_sweep=args.force_sweep,
        )
        
        # Print result
        action = result.get("action", "unknown")
        if action == "sweep":
            amount = result.get("amount", 0)
            mode = result.get("mode", "unknown")
            logger.info(f"✅ SWEEP EXECUTED: ${amount:.2f} USDC ({mode} mode)")
        elif action == "skip":
            reason = result.get("reason", "unknown")
            logger.info(f"⏸️ SWEEP SKIPPED: {reason}")
        elif action == "error":
            error = result.get("error", "unknown")
            logger.error(f"❌ SWEEP FAILED: {error}")
        
        # Print full result for debugging
        if args.verbose:
            logger.debug(f"Full result: {result}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
