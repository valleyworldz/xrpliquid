"""
Funding/Oracle Divergence Guard - Fixed version
"""

import logging
import json
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
    severity: str
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
    Simplified guard against funding spikes and oracle divergence
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.funding_spike_threshold = Decimal('0.001')  # 0.1% funding rate
        self.oracle_divergence_threshold = Decimal('50')  # 50 bps divergence
        self.low_liquidity_threshold = Decimal('0.3')  # 30% of normal liquidity
        
        self.halted_orders = 0
        self.allowed_orders = 0
        
    def detect_funding_spike(self, current_funding: Decimal) -> DivergenceAlert:
        """
        Detect funding rate spikes
        """
        if abs(current_funding) > self.funding_spike_threshold:
            severity = 'high' if abs(current_funding) > Decimal('0.005') else 'medium'
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
            deviation_sigma=Decimal('0'),
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
                current_value=Decimal('0'),
                threshold=self.oracle_divergence_threshold,
                deviation_sigma=Decimal('0'),
                should_halt=True,
                reason="Invalid mark price"
            )
        
        divergence_bps = abs(oracle_price - mark_price) / mark_price * Decimal('10000')
        
        if divergence_bps > self.oracle_divergence_threshold:
            severity = 'critical' if divergence_bps > Decimal('100') else 'high'
            should_halt = True
            reason = f"Oracle divergence exceeds threshold: {divergence_bps:.1f} bps > {self.oracle_divergence_threshold} bps"
        else:
            severity = 'low'
            should_halt = False
            reason = f"Oracle divergence normal: {divergence_bps:.1f} bps"
        
        return DivergenceAlert(
            type=DivergenceType.ORACLE_DIVERGENCE,
            severity=severity,
            current_value=divergence_bps,
            threshold=self.oracle_divergence_threshold,
            deviation_sigma=Decimal('0'),
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
                deviation_sigma=Decimal('0'),
                should_halt=True,
                reason="Invalid normal liquidity reference"
            )
        
        liquidity_ratio = current_liquidity / normal_liquidity
        
        if liquidity_ratio < self.low_liquidity_threshold:
            severity = 'high' if liquidity_ratio < Decimal('0.1') else 'medium'
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
            deviation_sigma=Decimal('0'),
            should_halt=should_halt,
            reason=reason
        )
    
    def should_proceed_with_order(self, 
                                order_data: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Tuple[bool, str, DivergenceResult]:
        """
        Main gate function - returns (should_proceed, reason, divergence_result)
        """
        try:
            # Extract market data
            funding_rate = Decimal(str(market_data.get('funding_rate', 0)))
            oracle_price = Decimal(str(market_data.get('oracle_price', 0)))
            mark_price = Decimal(str(market_data.get('mark_price', 0)))
            current_liquidity = Decimal(str(market_data.get('current_liquidity', 1000000)))
            normal_liquidity = Decimal(str(market_data.get('normal_liquidity', 1000000)))
            
            # Check all conditions
            alerts = []
            recommendations = []
            
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
            
            divergence_result = DivergenceResult(
                overall_halt=overall_halt,
                alerts=alerts,
                funding_rate=funding_rate,
                oracle_divergence_bps=oracle_alert.current_value,
                liquidity_score=liquidity_alert.current_value,
                recommendations=recommendations
            )
            
            if not overall_halt:
                self.allowed_orders += 1
                self.logger.info(f"✅ DIVERGENCE_GUARD_PASSED: Market conditions normal")
                return True, "Market conditions normal", divergence_result
            else:
                self.halted_orders += 1
                reason = f"Market divergence detected: {', '.join(divergence_result.recommendations[:2])}"
                self.logger.warning(f"❌ DIVERGENCE_GUARD_HALTED: {reason}")
                return False, reason, divergence_result
                
        except Exception as e:
            self.logger.error(f"❌ Error in divergence guard: {e}")
            self.halted_orders += 1
            return False, f"Divergence guard error: {e}", None

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
    if divergence_result:
        print(f"  Funding Rate: {divergence_result.funding_rate:.4f}")
        print(f"  Oracle Divergence: {divergence_result.oracle_divergence_bps:.1f} bps")
    
    # Test 2: Funding spike
    print(f"\n🔍 Test 2: Funding spike")
    spike_market_data = market_data.copy()
    spike_market_data['funding_rate'] = 0.005  # 0.5% funding spike
    
    should_proceed, reason, divergence_result = guard.should_proceed_with_order(order_data, spike_market_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ HALTED'}")
    print(f"  Reason: {reason}")
    if divergence_result:
        print(f"  Alerts: {[alert.reason for alert in divergence_result.alerts if alert.should_halt]}")
    
    # Test 3: Oracle divergence
    print(f"\n🔍 Test 3: Oracle divergence")
    oracle_market_data = market_data.copy()
    oracle_market_data['oracle_price'] = 0.55  # 5% divergence
    oracle_market_data['mark_price'] = 0.52
    
    should_proceed, reason, divergence_result = guard.should_proceed_with_order(order_data, oracle_market_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ HALTED'}")
    print(f"  Reason: {reason}")
    if divergence_result:
        print(f"  Oracle Divergence: {divergence_result.oracle_divergence_bps:.1f} bps")
    
    print(f"\n✅ Funding/Oracle Divergence Guard Demo Complete")

if __name__ == "__main__":
    demo_funding_oracle_divergence_guard()
