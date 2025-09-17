"""
Funding/Oracle Divergence Guard - Halt new risk during funding spikes or oracle divergence
"""

from src.core.utils.decimal_boundary_guard import safe_decimal
from src.core.utils.decimal_boundary_guard import safe_float
import logging
import json
import statistics
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class DivergenceType(Enum):
    FUNDING_SPIKE = "funding_spike"
    ORACLE_DIVERGENCE = "oracle_divergence"
    LOW_LIQUIDITY = "low_liquidity"

@dataclass
class DivergenceAlert:
    type: DivergenceType
    severity: str  # 'low', 'medium', 'high', 'critical'
    current_value: Decimal
    threshold: Decimal
    deviation_sigma: Decimal
    should_halt: bool
    reason: str

@dataclass
class DivergenceResult:
    overall_halt: bool
    alerts: List[DivergenceAlert]
    funding_rate: Decimal
    oracle_divergence_bps: Decimal
    liquidity_score: Decimal
    recommendations: List[str]

class FundingOracleDivergenceGuard:
    """
    Guard against funding spikes and oracle divergence during low liquidity
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Thresholds (configurable via environment)
        self.funding_spike_threshold = safe_decimal('0.001')  # 0.1% funding rate
        self.funding_spike_sigma = safe_decimal('3.0')  # 3 sigma threshold
        self.oracle_divergence_threshold = safe_decimal('50')  # 50 bps divergence
        self.low_liquidity_threshold = safe_decimal('0.3')  # 30% of normal liquidity
        self.funding_history_window = 24  # 24 hours of funding history
        self.oracle_history_window = 60  # 60 minutes of oracle history
        
        self.halted_orders = 0
        self.allowed_orders = 0
        self.funding_history = []
        self.oracle_history = []
        
    def update_funding_history(self, funding_rate: Decimal, timestamp: datetime = None):
        """
        Update funding rate history
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.funding_history.append({
            'rate': funding_rate,
            'timestamp': timestamp
        })
        
        # Keep only recent history
        cutoff_time = timestamp - timedelta(hours=self.funding_history_window)
        self.funding_history = [
            entry for entry in self.funding_history 
            if entry['timestamp'] > cutoff_time
        ]
    
    def update_oracle_history(self, oracle_price: Decimal, mark_price: Decimal, timestamp: datetime = None):
        """
        Update oracle price history
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if mark_price > 0:
            divergence_bps = abs(oracle_price - mark_price) / mark_price * safe_decimal('10000')
            self.oracle_history.append({
                'divergence_bps': divergence_bps,
                'oracle_price': oracle_price,
                'mark_price': mark_price,
                'timestamp': timestamp
            })
            
            # Keep only recent history
            cutoff_time = timestamp - timedelta(minutes=self.oracle_history_window)
            self.oracle_history = [
                entry for entry in self.oracle_history 
                if entry['timestamp'] > cutoff_time
            ]
    
    def detect_funding_spike(self, current_funding: Decimal) -> DivergenceAlert:
        """
        Detect funding rate spikes
        """
        if len(self.funding_history) < 10:  # Need sufficient history
            return DivergenceAlert(
                type=DivergenceType.FUNDING_SPIKE,
                severity='low',
                current_value=current_funding,
                threshold=self.funding_spike_threshold,
                deviation_sigma=safe_decimal('0'),
                should_halt=False,
                reason="Insufficient funding history"
            )
        
        # Calculate historical statistics
        historical_rates = [entry['rate'] for entry in self.funding_history]
        mean_rate = safe_decimal(str(statistics.mean([safe_float(rate) for rate in historical_rates])))
        std_rate = safe_decimal(str(statistics.stdev([safe_float(rate) for rate in historical_rates]))) if len(historical_rates) > 1 else safe_decimal('0.0001')
        
        # Calculate deviation in sigma
        if std_rate > 0:
            deviation_sigma = abs(current_funding - mean_rate) / std_rate
        else:
            deviation_sigma = safe_decimal('0')
        
        # Determine severity and halt decision
        if deviation_sigma > self.funding_spike_sigma:
            severity = 'critical' if deviation_sigma > safe_decimal('5.0') else 'high'
            should_halt = True
            reason = f"Funding spike detected: {current_funding:.4f} ({deviation_sigma:.1f}σ above mean {mean_rate:.4f})"
        elif abs(current_funding) > self.funding_spike_threshold:
            severity = 'medium'
            should_halt = True
            reason = f"Funding rate exceeds threshold: {current_funding:.4f} > {self.funding_spike_threshold:.4f}"
        else:
            severity = 'low'
            should_halt = False
            reason = f"Funding rate normal: {current_funding:.4f}"
        
        return DivergenceAlert(
            type=DivergenceType.FUNDING_SPIKE,
            severity=severity,
            current_value=current_funding,
            threshold=self.funding_spike_threshold,
            deviation_sigma=deviation_sigma,
            should_halt=should_halt,
            reason=reason
        )
    
    def detect_oracle_divergence(self, oracle_price: Decimal, mark_price: Decimal) -> DivergenceAlert:
        """
        Detect oracle price divergence
        """
        if mark_price <= 0:
            return DivergenceAlert(
                type=DivergenceType.ORACLE_DIVERGENCE,
                severity='medium',
                current_value=safe_decimal('0'),
                threshold=self.oracle_divergence_threshold,
                deviation_sigma=safe_decimal('0'),
                should_halt=True,
                reason="Invalid mark price"
            )
        
        divergence_bps = abs(oracle_price - mark_price) / mark_price * safe_decimal('10000')
        
        if len(self.oracle_history) < 5:  # Need some history
            should_halt = divergence_bps > self.oracle_divergence_threshold
            severity = 'high' if should_halt else 'low'
            reason = f"Oracle divergence: {divergence_bps:.1f} bps (threshold: {self.oracle_divergence_threshold} bps)"
        else:
            # Calculate historical statistics
            historical_divergences = [entry['divergence_bps'] for entry in self.oracle_history]
            mean_divergence = safe_decimal(str(statistics.mean([safe_float(d) for d in historical_divergences])))
            std_divergence = safe_decimal(str(statistics.stdev([safe_float(d) for d in historical_divergences]))) if len(historical_divergences) > 1 else safe_decimal('10')
            
            # Calculate deviation in sigma
            if std_divergence > 0:
                deviation_sigma = abs(divergence_bps - mean_divergence) / std_divergence
            else:
                deviation_sigma = safe_decimal('0')
            
            # Determine severity and halt decision
            if divergence_bps > self.oracle_divergence_threshold:
                severity = 'critical' if deviation_sigma > safe_decimal('3.0') else 'high'
                should_halt = True
                reason = f"Oracle divergence exceeds threshold: {divergence_bps:.1f} bps > {self.oracle_divergence_threshold} bps"
            elif deviation_sigma > safe_decimal('2.0'):
                severity = 'medium'
                should_halt = True
                reason = f"Oracle divergence spike: {divergence_bps:.1f} bps ({deviation_sigma:.1f}σ above mean {mean_divergence:.1f} bps)"
            else:
                severity = 'low'
                should_halt = False
                reason = f"Oracle divergence normal: {divergence_bps:.1f} bps"
        
        return DivergenceAlert(
            type=DivergenceType.ORACLE_DIVERGENCE,
            severity=severity,
            current_value=divergence_bps,
            threshold=self.oracle_divergence_threshold,
            deviation_sigma=deviation_sigma,
            should_halt=should_halt,
            reason=reason
        )
    
    def assess_liquidity_conditions(self, current_liquidity: Decimal, normal_liquidity: Decimal) -> DivergenceAlert:
        """
        Assess current liquidity conditions
        """
        if normal_liquidity <= 0:
            return DivergenceAlert(
                type=DivergenceType.LOW_LIQUIDITY,
                severity='medium',
                current_value=current_liquidity,
                threshold=self.low_liquidity_threshold,
                deviation_sigma=safe_decimal('0'),
                should_halt=True,
                reason="Invalid normal liquidity reference"
            )
        
        liquidity_ratio = current_liquidity / normal_liquidity
        
        if liquidity_ratio < self.low_liquidity_threshold:
            severity = 'high' if liquidity_ratio < safe_decimal('0.1') else 'medium'
            should_halt = True
            reason = f"Low liquidity detected: {liquidity_ratio:.1%} of normal ({current_liquidity} / {normal_liquidity})"
        else:
            severity = 'low'
            should_halt = False
            reason = f"Liquidity normal: {liquidity_ratio:.1%} of normal"
        
        return DivergenceAlert(
            type=DivergenceType.LOW_LIQUIDITY,
            severity=severity,
            current_value=liquidity_ratio,
            threshold=self.low_liquidity_threshold,
            deviation_sigma=safe_decimal('0'),
            should_halt=should_halt,
            reason=reason
        )
    
    def check_divergence_conditions(self, 
                                  funding_rate: Decimal,
                                  oracle_price: Decimal,
                                  mark_price: Decimal,
                                  current_liquidity: Decimal,
                                  normal_liquidity: Decimal) -> DivergenceResult:
        """
        Check all divergence conditions
        """
        alerts = []
        recommendations = []
        
        # Update histories
        self.update_funding_history(funding_rate)
        self.update_oracle_history(oracle_price, mark_price)
        
        # Check funding spike
        funding_alert = self.detect_funding_spike(funding_rate)
        alerts.append(funding_alert)
        
        # Check oracle divergence
        oracle_alert = self.detect_oracle_divergence(oracle_price, mark_price)
        alerts.append(oracle_alert)
        
        # Check liquidity conditions
        liquidity_alert = self.assess_liquidity_conditions(current_liquidity, normal_liquidity)
        alerts.append(liquidity_alert)
        
        # Determine overall halt decision
        overall_halt = any(alert.should_halt for alert in alerts)
        
        # Generate recommendations
        if funding_alert.should_halt:
            recommendations.append("Wait for funding rate to normalize")
        if oracle_alert.should_halt:
            recommendations.append("Wait for oracle price to converge")
        if liquidity_alert.should_halt:
            recommendations.append("Wait for liquidity to improve")
        
        if not overall_halt:
            recommendations.append("Market conditions normal, proceed with caution")
        
        return DivergenceResult(
            overall_halt=overall_halt,
            alerts=alerts,
            funding_rate=funding_rate,
            oracle_divergence_bps=oracle_alert.current_value,
            liquidity_score=liquidity_alert.current_value,
            recommendations=recommendations
        )
    
    def should_proceed_with_order(self, 
                                order_data: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Tuple[bool, str, DivergenceResult]:
        """
        Main gate function - returns (should_proceed, reason, divergence_result)
        """
        try:
            # Extract market data
            funding_rate = safe_decimal(str(market_data.get('funding_rate', 0)))
            oracle_price = safe_decimal(str(market_data.get('oracle_price', 0)))
            mark_price = safe_decimal(str(market_data.get('mark_price', 0)))
            current_liquidity = safe_decimal(str(market_data.get('current_liquidity', 1000000)))
            normal_liquidity = safe_decimal(str(market_data.get('normal_liquidity', 1000000)))
            
            # Check divergence conditions
            divergence_result = self.check_divergence_conditions(
                funding_rate, oracle_price, mark_price, current_liquidity, normal_liquidity
            )
            
            if not divergence_result.overall_halt:
                self.allowed_orders += 1
                self.logger.info(f"✅ DIVERGENCE_GUARD_PASSED: Market conditions normal")
                return True, "Market conditions normal", divergence_result
            else:
                self.halted_orders += 1
                reason = f"Market divergence detected: {', '.join(divergence_result.recommendations[:2])}"
                self.logger.warning(f"❌ DIVERGENCE_GUARD_HALTED: {reason}")
                
                # Log structured JSON event
                self.log_divergence_halt(divergence_result, order_data)
                
                return False, reason, divergence_result
                
        except Exception as e:
            self.logger.error(f"❌ Error in divergence guard: {e}")
            self.halted_orders += 1
            return False, f"Divergence guard error: {e}", None
    
    def log_divergence_halt(self, divergence_result: DivergenceResult, order_data: Dict[str, Any]):
        """
        Log structured JSON event for divergence halt
        """
        try:
            halt_event = {
                "event": "divergence_guard_halt",
                "timestamp": datetime.now().isoformat(),
                "overall_halt": divergence_result.overall_halt,
                "funding_rate": str(divergence_result.funding_rate),
                "oracle_divergence_bps": str(divergence_result.oracle_divergence_bps),
                "liquidity_score": str(divergence_result.liquidity_score),
                "alerts": [
                    {
                        "type": alert.type.value,
                        "severity": alert.severity,
                        "current_value": str(alert.current_value),
                        "threshold": str(alert.threshold),
                        "deviation_sigma": str(alert.deviation_sigma),
                        "should_halt": alert.should_halt,
                        "reason": alert.reason
                    }
                    for alert in divergence_result.alerts
                ],
                "recommendations": divergence_result.recommendations,
                "order_data": {
                    "asset": order_data.get('asset'),
                    "side": order_data.get('side'),
                    "size": str(order_data.get('size', 0)),
                    "price": str(order_data.get('price', 0))
                },
                "halted_orders": self.halted_orders,
                "allowed_orders": self.allowed_orders
            }
            
            # Log as structured JSON
            self.logger.warning(f"🚫 DIVERGENCE_GUARD_HALT: {json.dumps(halt_event)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging divergence halt: {e}")
    
    def get_guard_statistics(self) -> Dict[str, Any]:
        """
        Get divergence guard statistics
        """
        total_orders = self.halted_orders + self.allowed_orders
        halt_rate = (self.halted_orders / total_orders * 100) if total_orders > 0 else 0
        
        return {
            "total_orders": total_orders,
            "halted_orders": self.halted_orders,
            "allowed_orders": self.allowed_orders,
            "halt_rate_percent": round(halt_rate, 2),
            "funding_spike_threshold": str(self.funding_spike_threshold),
            "funding_spike_sigma": str(self.funding_spike_sigma),
            "oracle_divergence_threshold": str(self.oracle_divergence_threshold),
            "low_liquidity_threshold": str(self.low_liquidity_threshold),
            "funding_history_size": len(self.funding_history),
            "oracle_history_size": len(self.oracle_history),
            "last_updated": datetime.now().isoformat()
        }

# Global guard instance
_divergence_guard = FundingOracleDivergenceGuard()

def should_proceed_with_order(order_data: Dict[str, Any], market_data: Dict[str, Any]) -> Tuple[bool, str, DivergenceResult]:
    """
    Global function to check if order should proceed based on divergence conditions
    """
    return _divergence_guard.should_proceed_with_order(order_data, market_data)

def get_divergence_guard() -> FundingOracleDivergenceGuard:
    """
    Get the global divergence guard instance
    """
    return _divergence_guard

# Demo function
def demo_funding_oracle_divergence_guard():
    """Demo the funding/oracle divergence guard"""
    print("📈 Funding/Oracle Divergence Guard Demo")
    print("=" * 50)
    
    guard = FundingOracleDivergenceGuard()
    
    # Test 1: Normal market conditions
    print("🔍 Test 1: Normal market conditions")
    order_data = {
        'asset': 'XRP',
        'side': 'BUY',
        'size': 100,
        'price': 0.52
    }
    
    market_data = {
        'funding_rate': 0.0001,  # 0.01% funding
        'oracle_price': 0.52,
        'mark_price': 0.5201,
        'current_liquidity': 1000000,
        'normal_liquidity': 1000000
    }
    
    should_proceed, reason, divergence_result = guard.should_proceed_with_order(order_data, market_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ HALTED'}")
    print(f"  Reason: {reason}")
    print(f"  Funding Rate: {divergence_result.funding_rate:.4f}")
    print(f"  Oracle Divergence: {divergence_result.oracle_divergence_bps:.1f} bps")
    
    # Test 2: Funding spike
    print(f"\n🔍 Test 2: Funding spike")
    # Add some normal funding history
    for i in range(20):
        guard.update_funding_history(safe_decimal('0.0001'))
    
    spike_market_data = market_data.copy()
    spike_market_data['funding_rate'] = 0.005  # 0.5% funding spike
    
    should_proceed, reason, divergence_result = guard.should_proceed_with_order(order_data, spike_market_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ HALTED'}")
    print(f"  Reason: {reason}")
    print(f"  Alerts: {[alert.reason for alert in divergence_result.alerts if alert.should_halt]}")
    
    # Test 3: Oracle divergence
    print(f"\n🔍 Test 3: Oracle divergence")
    oracle_market_data = market_data.copy()
    oracle_market_data['oracle_price'] = 0.55  # 5% divergence
    oracle_market_data['mark_price'] = 0.52
    
    should_proceed, reason, divergence_result = guard.should_proceed_with_order(order_data, oracle_market_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ HALTED'}")
    print(f"  Reason: {reason}")
    print(f"  Oracle Divergence: {divergence_result.oracle_divergence_bps:.1f} bps")
    
    # Test 4: Low liquidity
    print(f"\n🔍 Test 4: Low liquidity")
    low_liquidity_data = market_data.copy()
    low_liquidity_data['current_liquidity'] = 200000  # 20% of normal
    low_liquidity_data['normal_liquidity'] = 1000000
    
    should_proceed, reason, divergence_result = guard.should_proceed_with_order(order_data, low_liquidity_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ HALTED'}")
    print(f"  Reason: {reason}")
    print(f"  Liquidity Score: {divergence_result.liquidity_score:.1%}")
    
    # Show statistics
    print(f"\n📊 Guard Statistics:")
    stats = guard.get_guard_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Funding/Oracle Divergence Guard Demo Complete")

if __name__ == "__main__":
    demo_funding_oracle_divergence_guard()
