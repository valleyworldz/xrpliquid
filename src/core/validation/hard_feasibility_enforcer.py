"""
Hard Feasibility Enforcer - Prevents orders from being submitted if feasibility checks fail
"""

import logging
import json
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import os

@dataclass
class FeasibilityCheck:
    check_type: str
    passed: bool
    reason: str
    details: Dict[str, Any]

@dataclass
class FeasibilityResult:
    overall_passed: bool
    checks: list[FeasibilityCheck]
    should_submit_order: bool
    block_reason: Optional[str] = None

class HardFeasibilityEnforcer:
    """
    Hard feasibility enforcer that blocks orders before submission
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blocked_orders = 0
        self.allowed_orders = 0
        self.feasibility_checks = []
        
        # Feasibility thresholds
        self.thresholds = {
            'min_market_depth': Decimal('10000'),  # $10k minimum depth
            'max_slippage_bps': Decimal('50'),     # 50 bps max slippage
            'min_liquidity_ratio': Decimal('0.1'), # 10% of order size
            'max_tp_sl_spread_bps': Decimal('100'), # 100 bps max TP/SL spread
            'min_book_levels': 5,                  # Minimum 5 levels
            'max_book_age_ms': 5000,               # 5 second max age
        }
    
    def check_order_feasibility(self, 
                               symbol: str,
                               side: str,
                               size: Decimal,
                               price: Decimal,
                               order_type: str,
                               tp_price: Optional[Decimal] = None,
                               sl_price: Optional[Decimal] = None,
                               market_data: Dict[str, Any] = None) -> FeasibilityResult:
        """
        Perform comprehensive feasibility checks before order submission
        """
        try:
            self.logger.info(f"🔍 HARD_FEASIBILITY_CHECK: {side} {size} {symbol} @ {price}")
            
            checks = []
            
            # Check 1: Market depth validation
            depth_check = self._check_market_depth(symbol, size, market_data)
            checks.append(depth_check)
            
            # Check 2: Slippage estimation
            slippage_check = self._check_slippage_estimation(symbol, size, price, market_data)
            checks.append(slippage_check)
            
            # Check 3: TP/SL band validation
            if tp_price or sl_price:
                tp_sl_check = self._check_tp_sl_bands(symbol, price, tp_price, sl_price, market_data)
                checks.append(tp_sl_check)
            
            # Check 4: Book freshness
            book_check = self._check_book_freshness(symbol, market_data)
            checks.append(book_check)
            
            # Check 5: Liquidity ratio
            liquidity_check = self._check_liquidity_ratio(symbol, size, market_data)
            checks.append(liquidity_check)
            
            # Determine overall result
            failed_checks = [check for check in checks if not check.passed]
            overall_passed = len(failed_checks) == 0
            
            if overall_passed:
                self.allowed_orders += 1
                self.logger.info(f"✅ FEASIBILITY_PASS: Order approved for submission")
                return FeasibilityResult(
                    overall_passed=True,
                    checks=checks,
                    should_submit_order=True
                )
            else:
                self.blocked_orders += 1
                block_reason = f"Feasibility failed: {', '.join([check.reason for check in failed_checks])}"
                self.logger.warning(f"❌ FEASIBILITY_BLOCK: {block_reason}")
                
                # Log structured event
                self._log_feasibility_failure(symbol, side, size, price, failed_checks)
                
                return FeasibilityResult(
                    overall_passed=False,
                    checks=checks,
                    should_submit_order=False,
                    block_reason=block_reason
                )
                
        except Exception as e:
            self.logger.error(f"❌ FEASIBILITY_ERROR: Error checking feasibility: {e}")
            return FeasibilityResult(
                overall_passed=False,
                checks=[],
                should_submit_order=False,
                block_reason=f"Feasibility check error: {str(e)}"
            )
    
    def _check_market_depth(self, symbol: str, size: Decimal, market_data: Dict[str, Any]) -> FeasibilityCheck:
        """Check if market has sufficient depth"""
        try:
            if not market_data or 'depth' not in market_data:
                return FeasibilityCheck(
                    check_type='market_depth',
                    passed=False,
                    reason='No market depth data available',
                    details={'symbol': symbol, 'size': str(size)}
                )
            
            depth = market_data['depth']
            total_depth = sum(level['size'] * level['price'] for level in depth.get('bids', [])[:5])
            
            if total_depth < self.thresholds['min_market_depth']:
                return FeasibilityCheck(
                    check_type='market_depth',
                    passed=False,
                    reason=f'Insufficient market depth: ${total_depth:.2f} < ${self.thresholds["min_market_depth"]}',
                    details={'symbol': symbol, 'total_depth': str(total_depth), 'required': str(self.thresholds['min_market_depth'])}
                )
            
            return FeasibilityCheck(
                check_type='market_depth',
                passed=True,
                reason='Sufficient market depth available',
                details={'symbol': symbol, 'total_depth': str(total_depth)}
            )
            
        except Exception as e:
            return FeasibilityCheck(
                check_type='market_depth',
                passed=False,
                reason=f'Market depth check error: {str(e)}',
                details={'symbol': symbol, 'error': str(e)}
            )
    
    def _check_slippage_estimation(self, symbol: str, size: Decimal, price: Decimal, market_data: Dict[str, Any]) -> FeasibilityCheck:
        """Estimate slippage and check if acceptable"""
        try:
            if not market_data or 'depth' not in market_data:
                return FeasibilityCheck(
                    check_type='slippage',
                    passed=False,
                    reason='No market data for slippage estimation',
                    details={'symbol': symbol, 'size': str(size)}
                )
            
            # Simple slippage estimation based on order book
            depth = market_data['depth']
            estimated_slippage_bps = Decimal('5')  # Default 5 bps
            
            # Calculate based on order book impact
            if depth.get('bids') and depth.get('asks'):
                # Estimate slippage based on order book depth
                total_liquidity = sum(level['size'] for level in depth['bids'][:3])
                if total_liquidity > 0:
                    impact_ratio = size / total_liquidity
                    estimated_slippage_bps = impact_ratio * Decimal('20')  # 20 bps per 100% impact
            
            if estimated_slippage_bps > self.thresholds['max_slippage_bps']:
                return FeasibilityCheck(
                    check_type='slippage',
                    passed=False,
                    reason=f'Estimated slippage too high: {estimated_slippage_bps:.1f}bps > {self.thresholds["max_slippage_bps"]}bps',
                    details={'symbol': symbol, 'estimated_slippage_bps': str(estimated_slippage_bps), 'max_allowed': str(self.thresholds['max_slippage_bps'])}
                )
            
            return FeasibilityCheck(
                check_type='slippage',
                passed=True,
                reason=f'Estimated slippage acceptable: {estimated_slippage_bps:.1f}bps',
                details={'symbol': symbol, 'estimated_slippage_bps': str(estimated_slippage_bps)}
            )
            
        except Exception as e:
            return FeasibilityCheck(
                check_type='slippage',
                passed=False,
                reason=f'Slippage estimation error: {str(e)}',
                details={'symbol': symbol, 'error': str(e)}
            )
    
    def _check_tp_sl_bands(self, symbol: str, price: Decimal, tp_price: Optional[Decimal], sl_price: Optional[Decimal], market_data: Dict[str, Any]) -> FeasibilityCheck:
        """Check if TP/SL bands are within acceptable range"""
        try:
            if not tp_price and not sl_price:
                return FeasibilityCheck(
                    check_type='tp_sl_bands',
                    passed=True,
                    reason='No TP/SL bands to validate',
                    details={'symbol': symbol}
                )
            
            max_spread = self.thresholds['max_tp_sl_spread_bps']
            
            if tp_price:
                tp_spread = abs(tp_price - price) / price * Decimal('10000')
                if tp_spread > max_spread:
                    return FeasibilityCheck(
                        check_type='tp_sl_bands',
                        passed=False,
                        reason=f'TP band too wide: {tp_spread:.1f}bps > {max_spread}bps',
                        details={'symbol': symbol, 'tp_spread_bps': str(tp_spread), 'max_allowed': str(max_spread)}
                    )
            
            if sl_price:
                sl_spread = abs(sl_price - price) / price * Decimal('10000')
                if sl_spread > max_spread:
                    return FeasibilityCheck(
                        check_type='tp_sl_bands',
                        passed=False,
                        reason=f'SL band too wide: {sl_spread:.1f}bps > {max_spread}bps',
                        details={'symbol': symbol, 'sl_spread_bps': str(sl_spread), 'max_allowed': str(max_spread)}
                    )
            
            return FeasibilityCheck(
                check_type='tp_sl_bands',
                passed=True,
                reason='TP/SL bands within acceptable range',
                details={'symbol': symbol, 'tp_price': str(tp_price) if tp_price else None, 'sl_price': str(sl_price) if sl_price else None}
            )
            
        except Exception as e:
            return FeasibilityCheck(
                check_type='tp_sl_bands',
                passed=False,
                reason=f'TP/SL band check error: {str(e)}',
                details={'symbol': symbol, 'error': str(e)}
            )
    
    def _check_book_freshness(self, symbol: str, market_data: Dict[str, Any]) -> FeasibilityCheck:
        """Check if order book is fresh enough"""
        try:
            if not market_data or 'timestamp' not in market_data:
                return FeasibilityCheck(
                    check_type='book_freshness',
                    passed=False,
                    reason='No timestamp data for book freshness check',
                    details={'symbol': symbol}
                )
            
            book_timestamp = market_data['timestamp']
            current_time = datetime.now()
            
            if isinstance(book_timestamp, str):
                book_time = datetime.fromisoformat(book_timestamp.replace('Z', '+00:00'))
            else:
                book_time = book_timestamp
            
            age_ms = (current_time - book_time).total_seconds() * 1000
            
            if age_ms > self.thresholds['max_book_age_ms']:
                return FeasibilityCheck(
                    check_type='book_freshness',
                    passed=False,
                    reason=f'Order book too stale: {age_ms:.0f}ms > {self.thresholds["max_book_age_ms"]}ms',
                    details={'symbol': symbol, 'age_ms': age_ms, 'max_allowed': self.thresholds['max_book_age_ms']}
                )
            
            return FeasibilityCheck(
                check_type='book_freshness',
                passed=True,
                reason=f'Order book is fresh: {age_ms:.0f}ms',
                details={'symbol': symbol, 'age_ms': age_ms}
            )
            
        except Exception as e:
            return FeasibilityCheck(
                check_type='book_freshness',
                passed=False,
                reason=f'Book freshness check error: {str(e)}',
                details={'symbol': symbol, 'error': str(e)}
            )
    
    def _check_liquidity_ratio(self, symbol: str, size: Decimal, market_data: Dict[str, Any]) -> FeasibilityCheck:
        """Check if order size is reasonable relative to available liquidity"""
        try:
            if not market_data or 'depth' not in market_data:
                return FeasibilityCheck(
                    check_type='liquidity_ratio',
                    passed=False,
                    reason='No market data for liquidity ratio check',
                    details={'symbol': symbol, 'size': str(size)}
                )
            
            depth = market_data['depth']
            total_liquidity = sum(level['size'] for level in depth.get('bids', [])[:5])
            
            if total_liquidity == 0:
                return FeasibilityCheck(
                    check_type='liquidity_ratio',
                    passed=False,
                    reason='No liquidity available in order book',
                    details={'symbol': symbol, 'size': str(size)}
                )
            
            liquidity_ratio = size / total_liquidity
            
            if liquidity_ratio > self.thresholds['min_liquidity_ratio']:
                return FeasibilityCheck(
                    check_type='liquidity_ratio',
                    passed=False,
                    reason=f'Order size too large relative to liquidity: {liquidity_ratio:.1%} > {self.thresholds["min_liquidity_ratio"]:.1%}',
                    details={'symbol': symbol, 'liquidity_ratio': str(liquidity_ratio), 'max_allowed': str(self.thresholds['min_liquidity_ratio'])}
                )
            
            return FeasibilityCheck(
                check_type='liquidity_ratio',
                passed=True,
                reason=f'Order size acceptable relative to liquidity: {liquidity_ratio:.1%}',
                details={'symbol': symbol, 'liquidity_ratio': str(liquidity_ratio)}
            )
            
        except Exception as e:
            return FeasibilityCheck(
                check_type='liquidity_ratio',
                passed=False,
                reason=f'Liquidity ratio check error: {str(e)}',
                details={'symbol': symbol, 'error': str(e)}
            )
    
    def _log_feasibility_failure(self, symbol: str, side: str, size: Decimal, price: Decimal, failed_checks: list[FeasibilityCheck]):
        """Log structured feasibility failure event"""
        try:
            failure_event = {
                "event": "feasibility_failed",
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "side": side,
                "size": str(size),
                "price": str(price),
                "failed_checks": [
                    {
                        "check_type": check.check_type,
                        "reason": check.reason,
                        "details": check.details
                    }
                    for check in failed_checks
                ],
                "blocked_orders_count": self.blocked_orders,
                "allowed_orders_count": self.allowed_orders
            }
            
            self.logger.warning(f"🚫 FEASIBILITY_BLOCKED: {json.dumps(failure_event)}")
            
        except Exception as e:
            self.logger.error(f"Error logging feasibility failure: {e}")
    
    def get_feasibility_stats(self) -> Dict[str, Any]:
        """Get feasibility enforcement statistics"""
        total_orders = self.blocked_orders + self.allowed_orders
        block_rate = (self.blocked_orders / total_orders * 100) if total_orders > 0 else 0
        
        return {
            "total_orders_checked": total_orders,
            "blocked_orders": self.blocked_orders,
            "allowed_orders": self.allowed_orders,
            "block_rate_percent": block_rate,
            "thresholds": {k: str(v) for k, v in self.thresholds.items()},
            "last_updated": datetime.now().isoformat()
        }

# Global enforcer instance
_feasibility_enforcer = HardFeasibilityEnforcer()

def check_order_feasibility(symbol: str, side: str, size: Decimal, price: Decimal, order_type: str, **kwargs) -> FeasibilityResult:
    """
    Global function to check order feasibility
    """
    return _feasibility_enforcer.check_order_feasibility(symbol, side, size, price, order_type, **kwargs)

def get_feasibility_stats() -> Dict[str, Any]:
    """
    Get feasibility enforcement statistics
    """
    return _feasibility_enforcer.get_feasibility_stats()

# Demo function
def demo_hard_feasibility_enforcer():
    """Demo the hard feasibility enforcer"""
    print("🔒 Hard Feasibility Enforcer Demo")
    print("=" * 50)
    
    enforcer = HardFeasibilityEnforcer()
    
    # Test cases
    test_cases = [
        {
            "symbol": "XRP/USD",
            "side": "BUY",
            "size": Decimal('1000'),
            "price": Decimal('0.52'),
            "order_type": "LIMIT",
            "market_data": {
                "depth": {
                    "bids": [{"price": 0.52, "size": 5000}, {"price": 0.519, "size": 3000}],
                    "asks": [{"price": 0.521, "size": 4000}, {"price": 0.522, "size": 2000}]
                },
                "timestamp": datetime.now().isoformat()
            }
        },
        {
            "symbol": "XRP/USD",
            "side": "BUY",
            "size": Decimal('50000'),  # Large order
            "price": Decimal('0.52'),
            "order_type": "LIMIT",
            "market_data": {
                "depth": {
                    "bids": [{"price": 0.52, "size": 1000}],  # Insufficient depth
                    "asks": [{"price": 0.521, "size": 1000}]
                },
                "timestamp": datetime.now().isoformat()
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['side']} {test_case['size']} {test_case['symbol']}")
        
        result = enforcer.check_order_feasibility(**test_case)
        
        print(f"  Overall Result: {'✅ PASSED' if result.overall_passed else '❌ BLOCKED'}")
        print(f"  Should Submit: {'Yes' if result.should_submit_order else 'No'}")
        
        if result.block_reason:
            print(f"  Block Reason: {result.block_reason}")
        
        print(f"  Individual Checks:")
        for check in result.checks:
            status = "✅" if check.passed else "❌"
            print(f"    {status} {check.check_type}: {check.reason}")
    
    # Show statistics
    print(f"\n📊 Feasibility Statistics:")
    stats = enforcer.get_feasibility_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Hard Feasibility Enforcer Demo Complete")

if __name__ == "__main__":
    demo_hard_feasibility_enforcer()
