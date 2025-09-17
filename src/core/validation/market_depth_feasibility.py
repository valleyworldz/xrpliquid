"""
Market Depth Feasibility Checker - Hard Pre-Trade Gate
Makes market-depth feasibility a hard pre-trade gate to prevent invalid TP/SL placements
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class FeasibilityResult(Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    MARGINAL = "marginal"

@dataclass
class FeasibilityCheck:
    result: FeasibilityResult
    confidence: float
    reason: str
    required_depth: Decimal
    available_depth: Decimal
    price_impact: Decimal
    snapshot_hash: str
    timestamp: str

@dataclass
class MarketDepthSnapshot:
    symbol: str
    timestamp: str
    bids: List[Tuple[Decimal, Decimal]]  # (price, size)
    asks: List[Tuple[Decimal, Decimal]]  # (price, size)
    snapshot_hash: str

class MarketDepthFeasibilityChecker:
    """
    Hard pre-trade gate for market depth feasibility
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_depth_ratio = Decimal('0.1')  # 10% of order size must be available
        self.max_price_impact = Decimal('0.05')  # 5% max price impact
        self.min_spread_ratio = Decimal('0.001')  # 0.1% minimum spread
        self.depth_levels = 5  # Check top 5 levels
        
    def check_tp_sl_feasibility(self, 
                               market_depth: MarketDepthSnapshot,
                               entry_price: Decimal,
                               tp_price: Decimal,
                               sl_price: Decimal,
                               order_size: Decimal,
                               side: str) -> FeasibilityCheck:
        """
        Check if TP/SL levels are feasible given market depth
        """
        try:
            # Calculate price distances
            tp_distance = abs(tp_price - entry_price) / entry_price
            sl_distance = abs(sl_price - entry_price) / entry_price
            
            # Check if distances are reasonable (within 10% for TP, 5% for SL)
            if tp_distance > Decimal('0.10') or sl_distance > Decimal('0.05'):
                return FeasibilityCheck(
                    result=FeasibilityResult.INFEASIBLE,
                    confidence=1.0,
                    reason=f"TP/SL distances too large: TP={tp_distance:.2%}, SL={sl_distance:.2%}",
                    required_depth=order_size,
                    available_depth=Decimal('0'),
                    price_impact=Decimal('0'),
                    snapshot_hash=market_depth.snapshot_hash,
                    timestamp=datetime.now().isoformat()
                )
            
            # Check market depth for TP/SL execution
            if side == "buy":
                # For buy orders, check ask side for TP and bid side for SL
                tp_feasible = self._check_execution_feasibility(
                    market_depth.asks, tp_price, order_size, "sell"
                )
                sl_feasible = self._check_execution_feasibility(
                    market_depth.bids, sl_price, order_size, "sell"
                )
            else:
                # For sell orders, check bid side for TP and ask side for SL
                tp_feasible = self._check_execution_feasibility(
                    market_depth.bids, tp_price, order_size, "buy"
                )
                sl_feasible = self._check_execution_feasibility(
                    market_depth.asks, sl_price, order_size, "buy"
                )
            
            # Determine overall feasibility
            if tp_feasible.result == FeasibilityResult.FEASIBLE and sl_feasible.result == FeasibilityResult.FEASIBLE:
                overall_result = FeasibilityResult.FEASIBLE
                confidence = min(tp_feasible.confidence, sl_feasible.confidence)
                reason = "Both TP and SL levels are feasible"
            elif tp_feasible.result == FeasibilityResult.INFEASIBLE or sl_feasible.result == FeasibilityResult.INFEASIBLE:
                overall_result = FeasibilityResult.INFEASIBLE
                confidence = 1.0
                reason = f"TP/SL infeasible: TP={tp_feasible.result.value}, SL={sl_feasible.result.value}"
            else:
                overall_result = FeasibilityResult.MARGINAL
                confidence = (tp_feasible.confidence + sl_feasible.confidence) / 2
                reason = f"TP/SL marginal: TP={tp_feasible.result.value}, SL={sl_feasible.result.value}"
            
            return FeasibilityCheck(
                result=overall_result,
                confidence=confidence,
                reason=reason,
                required_depth=order_size,
                available_depth=tp_feasible.available_depth + sl_feasible.available_depth,
                price_impact=max(tp_feasible.price_impact, sl_feasible.price_impact),
                snapshot_hash=market_depth.snapshot_hash,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ TP/SL feasibility check error: {e}")
            return FeasibilityCheck(
                result=FeasibilityResult.INFEASIBLE,
                confidence=0.0,
                reason=f"Error in feasibility check: {e}",
                required_depth=order_size,
                available_depth=Decimal('0'),
                price_impact=Decimal('0'),
                snapshot_hash=market_depth.snapshot_hash,
                timestamp=datetime.now().isoformat()
            )
    
    def _check_execution_feasibility(self, 
                                   depth_levels: List[Tuple[Decimal, Decimal]],
                                   target_price: Decimal,
                                   order_size: Decimal,
                                   execution_side: str) -> FeasibilityCheck:
        """
        Check if execution at target price is feasible given market depth
        """
        try:
            if not depth_levels:
                return FeasibilityCheck(
                    result=FeasibilityResult.INFEASIBLE,
                    confidence=1.0,
                    reason="No market depth available",
                    required_depth=order_size,
                    available_depth=Decimal('0'),
                    price_impact=Decimal('0'),
                    snapshot_hash="",
                    timestamp=datetime.now().isoformat()
                )
            
            # Find relevant depth levels
            relevant_levels = []
            for price, size in depth_levels:
                if execution_side == "buy":
                    # For buying, we need prices <= target_price
                    if price <= target_price:
                        relevant_levels.append((price, size))
                else:
                    # For selling, we need prices >= target_price
                    if price >= target_price:
                        relevant_levels.append((price, size))
            
            if not relevant_levels:
                return FeasibilityCheck(
                    result=FeasibilityResult.INFEASIBLE,
                    confidence=1.0,
                    reason=f"No depth levels available at target price {target_price}",
                    required_depth=order_size,
                    available_depth=Decimal('0'),
                    price_impact=Decimal('0'),
                    snapshot_hash="",
                    timestamp=datetime.now().isoformat()
                )
            
            # Calculate available depth and price impact
            total_available = Decimal('0')
            weighted_price = Decimal('0')
            remaining_size = order_size
            
            for price, size in relevant_levels:
                if remaining_size <= 0:
                    break
                
                # Use available size or remaining order size, whichever is smaller
                used_size = min(size, remaining_size)
                total_available += used_size
                weighted_price += price * used_size
                remaining_size -= used_size
            
            if total_available == 0:
                return FeasibilityCheck(
                    result=FeasibilityResult.INFEASIBLE,
                    confidence=1.0,
                    reason="No available depth for execution",
                    required_depth=order_size,
                    available_depth=Decimal('0'),
                    price_impact=Decimal('0'),
                    snapshot_hash="",
                    timestamp=datetime.now().isoformat()
                )
            
            # Calculate average execution price and price impact
            avg_execution_price = weighted_price / total_available
            price_impact = abs(avg_execution_price - target_price) / target_price
            
            # Check if we can fill the entire order
            if total_available >= order_size:
                # Full fill possible
                if price_impact <= self.max_price_impact:
                    result = FeasibilityResult.FEASIBLE
                    confidence = 0.9
                    reason = "Full fill possible with acceptable price impact"
                else:
                    result = FeasibilityResult.MARGINAL
                    confidence = 0.6
                    reason = f"Full fill possible but high price impact: {price_impact:.2%}"
            else:
                # Partial fill only
                fill_ratio = total_available / order_size
                if fill_ratio >= self.min_depth_ratio:
                    result = FeasibilityResult.MARGINAL
                    confidence = 0.7
                    reason = f"Partial fill possible: {fill_ratio:.1%} of order size"
                else:
                    result = FeasibilityResult.INFEASIBLE
                    confidence = 1.0
                    reason = f"Insufficient depth: only {fill_ratio:.1%} of order size available"
            
            return FeasibilityCheck(
                result=result,
                confidence=confidence,
                reason=reason,
                required_depth=order_size,
                available_depth=total_available,
                price_impact=price_impact,
                snapshot_hash="",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Execution feasibility check error: {e}")
            return FeasibilityCheck(
                result=FeasibilityResult.INFEASIBLE,
                confidence=0.0,
                reason=f"Error in execution check: {e}",
                required_depth=order_size,
                available_depth=Decimal('0'),
                price_impact=Decimal('0'),
                snapshot_hash="",
                timestamp=datetime.now().isoformat()
            )
    
    def check_market_conditions(self, market_depth: MarketDepthSnapshot) -> Dict[str, Any]:
        """
        Check overall market conditions for trading feasibility
        """
        try:
            if not market_depth.bids or not market_depth.asks:
                return {
                    "feasible": False,
                    "reason": "No market depth available",
                    "spread": 0.0,
                    "depth_imbalance": 0.0,
                    "market_quality": "poor"
                }
            
            # Calculate spread
            best_bid = market_depth.bids[0][0]
            best_ask = market_depth.asks[0][0]
            spread = best_ask - best_bid
            spread_ratio = spread / best_bid
            
            # Calculate depth imbalance
            total_bid_depth = sum(size for _, size in market_depth.bids[:self.depth_levels])
            total_ask_depth = sum(size for _, size in market_depth.asks[:self.depth_levels])
            depth_imbalance = abs(total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth)
            
            # Determine market quality
            if spread_ratio <= self.min_spread_ratio and depth_imbalance <= 0.3:
                market_quality = "excellent"
                feasible = True
                reason = "Excellent market conditions"
            elif spread_ratio <= self.min_spread_ratio * 2 and depth_imbalance <= 0.5:
                market_quality = "good"
                feasible = True
                reason = "Good market conditions"
            elif spread_ratio <= self.min_spread_ratio * 5 and depth_imbalance <= 0.7:
                market_quality = "fair"
                feasible = True
                reason = "Fair market conditions"
            else:
                market_quality = "poor"
                feasible = False
                reason = f"Poor market conditions: spread={spread_ratio:.2%}, imbalance={depth_imbalance:.2%}"
            
            return {
                "feasible": feasible,
                "reason": reason,
                "spread": float(spread),
                "spread_ratio": float(spread_ratio),
                "depth_imbalance": float(depth_imbalance),
                "market_quality": market_quality,
                "total_bid_depth": float(total_bid_depth),
                "total_ask_depth": float(total_ask_depth)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Market conditions check error: {e}")
            return {
                "feasible": False,
                "reason": f"Error checking market conditions: {e}",
                "spread": 0.0,
                "depth_imbalance": 0.0,
                "market_quality": "error"
            }
    
    def log_feasibility_failure(self, check: FeasibilityCheck, order_data: Dict[str, Any]):
        """
        Log feasibility failure for audit trail
        """
        try:
            failure_log = {
                "timestamp": check.timestamp,
                "event_type": "feasibility_failed",
                "result": check.result.value,
                "reason": check.reason,
                "confidence": check.confidence,
                "required_depth": float(check.required_depth),
                "available_depth": float(check.available_depth),
                "price_impact": float(check.price_impact),
                "snapshot_hash": check.snapshot_hash,
                "order_data": order_data
            }
            
            # Log as structured JSON
            self.logger.warning(f"🚫 FEASIBILITY_FAILED: {json.dumps(failure_log)}")
            
        except Exception as e:
            self.logger.error(f"❌ Feasibility failure logging error: {e}")

# Demo function
def demo_market_depth_feasibility():
    """Demo the market depth feasibility checker"""
    print("🔍 Market Depth Feasibility Checker Demo")
    print("=" * 50)
    
    # Create feasibility checker
    checker = MarketDepthFeasibilityChecker()
    
    # Create sample market depth
    market_depth = MarketDepthSnapshot(
        symbol="XRP-USD",
        timestamp=datetime.now().isoformat(),
        bids=[
            (Decimal('0.5234'), Decimal('1000')),
            (Decimal('0.5233'), Decimal('1500')),
            (Decimal('0.5232'), Decimal('2000')),
            (Decimal('0.5231'), Decimal('1200')),
            (Decimal('0.5230'), Decimal('800'))
        ],
        asks=[
            (Decimal('0.5236'), Decimal('1000')),
            (Decimal('0.5237'), Decimal('1500')),
            (Decimal('0.5238'), Decimal('2000')),
            (Decimal('0.5239'), Decimal('1200')),
            (Decimal('0.5240'), Decimal('800'))
        ],
        snapshot_hash="demo_snapshot_123"
    )
    
    # Test market conditions
    print("📊 Market Conditions Check:")
    conditions = checker.check_market_conditions(market_depth)
    print(f"  Feasible: {conditions['feasible']}")
    print(f"  Reason: {conditions['reason']}")
    print(f"  Spread: {conditions['spread']:.4f}")
    print(f"  Spread Ratio: {conditions['spread_ratio']:.2%}")
    print(f"  Depth Imbalance: {conditions['depth_imbalance']:.2%}")
    print(f"  Market Quality: {conditions['market_quality']}")
    
    # Test TP/SL feasibility
    print(f"\n🎯 TP/SL Feasibility Check:")
    entry_price = Decimal('0.5235')
    tp_price = Decimal('0.5250')  # 0.29% above entry
    sl_price = Decimal('0.5210')  # 0.48% below entry
    order_size = Decimal('500')
    
    feasibility = checker.check_tp_sl_feasibility(
        market_depth, entry_price, tp_price, sl_price, order_size, "buy"
    )
    
    print(f"  Result: {feasibility.result.value}")
    print(f"  Confidence: {feasibility.confidence:.2f}")
    print(f"  Reason: {feasibility.reason}")
    print(f"  Required Depth: {feasibility.required_depth}")
    print(f"  Available Depth: {feasibility.available_depth}")
    print(f"  Price Impact: {feasibility.price_impact:.2%}")
    
    # Test infeasible scenario
    print(f"\n❌ Infeasible Scenario Test:")
    infeasible_tp = Decimal('0.5300')  # 1.24% above entry (too far)
    infeasible_sl = Decimal('0.5200')  # 0.67% below entry (too far)
    
    infeasible_check = checker.check_tp_sl_feasibility(
        market_depth, entry_price, infeasible_tp, infeasible_sl, order_size, "buy"
    )
    
    print(f"  Result: {infeasible_check.result.value}")
    print(f"  Reason: {infeasible_check.reason}")
    
    # Log feasibility failure
    if infeasible_check.result == FeasibilityResult.INFEASIBLE:
        order_data = {
            "symbol": "XRP-USD",
            "side": "buy",
            "size": float(order_size),
            "entry_price": float(entry_price),
            "tp_price": float(infeasible_tp),
            "sl_price": float(infeasible_sl)
        }
        checker.log_feasibility_failure(infeasible_check, order_data)
    
    print("\n✅ Market Depth Feasibility Demo Complete")

if __name__ == "__main__":
    demo_market_depth_feasibility()
