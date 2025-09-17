"""
Hard Feasibility Gate - Prevents orders from being submitted if feasibility fails
"""

import logging
import json
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class FeasibilityResult(Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    MARGINAL = "marginal"

@dataclass
class FeasibilityDecision:
    result: FeasibilityResult
    reason: str
    confidence: float
    should_proceed: bool
    timestamp: str

class HardFeasibilityGate:
    """
    Hard pre-trade gate that blocks order submission if feasibility fails
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blocked_orders = 0
        self.allowed_orders = 0
        
    def check_order_feasibility(self, 
                               market_depth: Dict[str, Any],
                               order_data: Dict[str, Any]) -> FeasibilityDecision:
        """
        Check if an order is feasible and should proceed
        """
        try:
            # Extract order details
            side = order_data.get('side', '').upper()
            size = Decimal(str(order_data.get('size', 0)))
            price = Decimal(str(order_data.get('price', 0)))
            take_profit = Decimal(str(order_data.get('take_profit', 0)))
            stop_loss = Decimal(str(order_data.get('stop_loss', 0)))
            
            # Check 1: Basic order validation
            if size <= 0 or price <= 0:
                return FeasibilityDecision(
                    result=FeasibilityResult.INFEASIBLE,
                    reason="Invalid order size or price",
                    confidence=1.0,
                    should_proceed=False,
                    timestamp=datetime.now().isoformat()
                )
            
            # Check 2: TP/SL distance validation
            if take_profit > 0 and stop_loss > 0:
                tp_distance = abs(take_profit - price) / price
                sl_distance = abs(price - stop_loss) / price
                
                if tp_distance > Decimal('0.10'):  # 10% TP limit
                    return FeasibilityDecision(
                        result=FeasibilityResult.INFEASIBLE,
                        reason=f"TP distance too large: {tp_distance:.2%}",
                        confidence=1.0,
                        should_proceed=False,
                        timestamp=datetime.now().isoformat()
                    )
                
                if sl_distance > Decimal('0.05'):  # 5% SL limit
                    return FeasibilityDecision(
                        result=FeasibilityResult.INFEASIBLE,
                        reason=f"SL distance too large: {sl_distance:.2%}",
                        confidence=1.0,
                        should_proceed=False,
                        timestamp=datetime.now().isoformat()
                    )
            
            # Check 3: Market depth validation
            if market_depth:
                bids = market_depth.get('bids', [])
                asks = market_depth.get('asks', [])
                
                if not bids or not asks:
                    return FeasibilityDecision(
                        result=FeasibilityResult.INFEASIBLE,
                        reason="Insufficient market depth",
                        confidence=1.0,
                        should_proceed=False,
                        timestamp=datetime.now().isoformat()
                    )
                
                # Check if order size is reasonable relative to market depth
                if side == 'BUY':
                    available_depth = sum(Decimal(str(ask[1])) for ask in asks[:5])
                else:
                    available_depth = sum(Decimal(str(bid[1])) for bid in bids[:5])
                
                if size > available_depth * Decimal('0.1'):  # Max 10% of top 5 levels
                    return FeasibilityDecision(
                        result=FeasibilityResult.INFEASIBLE,
                        reason=f"Order size too large for market depth: {size} > {available_depth * Decimal('0.1')}",
                        confidence=1.0,
                        should_proceed=False,
                        timestamp=datetime.now().isoformat()
                    )
            
            # Check 4: Spread validation
            if market_depth and market_depth.get('bids') and market_depth.get('asks'):
                best_bid = Decimal(str(market_depth['bids'][0][0]))
                best_ask = Decimal(str(market_depth['asks'][0][0]))
                spread = (best_ask - best_bid) / best_bid
                
                if spread > Decimal('0.01'):  # 1% spread limit
                    return FeasibilityDecision(
                        result=FeasibilityResult.MARGINAL,
                        reason=f"Wide spread: {spread:.2%}",
                        confidence=0.7,
                        should_proceed=True,  # Allow but with lower confidence
                        timestamp=datetime.now().isoformat()
                    )
            
            # All checks passed
            return FeasibilityDecision(
                result=FeasibilityResult.FEASIBLE,
                reason="Order is feasible",
                confidence=0.95,
                should_proceed=True,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in feasibility check: {e}")
            return FeasibilityDecision(
                result=FeasibilityResult.INFEASIBLE,
                reason=f"Feasibility check error: {e}",
                confidence=1.0,
                should_proceed=False,
                timestamp=datetime.now().isoformat()
            )
    
    def should_proceed_with_order(self, 
                                 market_depth: Dict[str, Any],
                                 order_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Main gate function - returns (should_proceed, reason)
        """
        try:
            decision = self.check_order_feasibility(market_depth, order_data)
            
            if decision.should_proceed:
                self.allowed_orders += 1
                self.logger.info(f"✅ FEASIBILITY_PASSED: {decision.reason}")
                return True, decision.reason
            else:
                self.blocked_orders += 1
                self.logger.warning(f"❌ FEASIBILITY_BLOCKED: {decision.reason}")
                
                # Log structured JSON event
                self.log_feasibility_failure(decision, order_data)
                
                return False, decision.reason
                
        except Exception as e:
            self.logger.error(f"❌ Error in feasibility gate: {e}")
            self.blocked_orders += 1
            return False, f"Feasibility gate error: {e}"
    
    def log_feasibility_failure(self, decision: FeasibilityDecision, order_data: Dict[str, Any]):
        """
        Log structured JSON event for feasibility failure
        """
        try:
            failure_event = {
                "event": "feasibility_failed",
                "timestamp": decision.timestamp,
                "result": decision.result.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "order_data": {
                    "side": order_data.get('side'),
                    "size": str(order_data.get('size', 0)),
                    "price": str(order_data.get('price', 0)),
                    "take_profit": str(order_data.get('take_profit', 0)),
                    "stop_loss": str(order_data.get('stop_loss', 0))
                },
                "blocked_orders": self.blocked_orders,
                "allowed_orders": self.allowed_orders
            }
            
            # Log as structured JSON
            self.logger.warning(f"🚫 FEASIBILITY_FAILED: {json.dumps(failure_event)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging feasibility failure: {e}")
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """
        Get feasibility gate statistics
        """
        total_orders = self.blocked_orders + self.allowed_orders
        block_rate = (self.blocked_orders / total_orders * 100) if total_orders > 0 else 0
        
        return {
            "total_orders": total_orders,
            "blocked_orders": self.blocked_orders,
            "allowed_orders": self.allowed_orders,
            "block_rate_percent": round(block_rate, 2),
            "last_updated": datetime.now().isoformat()
        }
    
    def log_gate_statistics(self):
        """
        Log current gate statistics
        """
        stats = self.get_gate_statistics()
        self.logger.info(f"📊 FEASIBILITY_GATE_STATS: {json.dumps(stats)}")

# Global gate instance
_feasibility_gate = HardFeasibilityGate()

def should_proceed_with_order(market_depth: Dict[str, Any], order_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Global function to check if order should proceed
    """
    return _feasibility_gate.should_proceed_with_order(market_depth, order_data)

def get_feasibility_gate() -> HardFeasibilityGate:
    """
    Get the global feasibility gate instance
    """
    return _feasibility_gate

# Demo function
def demo_hard_feasibility_gate():
    """Demo the hard feasibility gate"""
    print("🛡️ Hard Feasibility Gate Demo")
    print("=" * 50)
    
    gate = HardFeasibilityGate()
    
    # Test 1: Valid order
    print("🔍 Test 1: Valid order")
    market_depth = {
        'bids': [['0.5234', '1000'], ['0.5233', '1500']],
        'asks': [['0.5236', '1000'], ['0.5237', '1500']]
    }
    
    order_data = {
        'side': 'BUY',
        'size': 100,
        'price': 0.5235,
        'take_profit': 0.5250,  # 0.29% above
        'stop_loss': 0.5210     # 0.48% below
    }
    
    should_proceed, reason = gate.should_proceed_with_order(market_depth, order_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    
    # Test 2: Invalid TP/SL distances
    print(f"\n🔍 Test 2: Invalid TP/SL distances")
    order_data['take_profit'] = 0.5300  # 1.24% above (too far)
    order_data['stop_loss'] = 0.5200    # 0.67% below (too far)
    
    should_proceed, reason = gate.should_proceed_with_order(market_depth, order_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    
    # Test 3: Insufficient market depth
    print(f"\n🔍 Test 3: Insufficient market depth")
    market_depth = {
        'bids': [['0.5234', '10']],  # Very thin
        'asks': [['0.5236', '10']]
    }
    
    order_data = {
        'side': 'BUY',
        'size': 1000,  # Large order
        'price': 0.5235,
        'take_profit': 0.5250,
        'stop_loss': 0.5210
    }
    
    should_proceed, reason = gate.should_proceed_with_order(market_depth, order_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    
    # Show statistics
    print(f"\n📊 Gate Statistics:")
    stats = gate.get_gate_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Hard Feasibility Gate Demo Complete")

if __name__ == "__main__":
    demo_hard_feasibility_gate()
