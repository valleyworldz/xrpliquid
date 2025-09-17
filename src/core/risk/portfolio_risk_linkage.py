"""
Portfolio Risk Linkage - Pre-trade portfolio risk assessment
"""

import logging
import json
from typing import Dict, Any, List, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PortfolioRiskMetrics:
    total_exposure: Decimal
    portfolio_var_95: Decimal
    concentration_risk: Decimal
    leverage_ratio: Decimal
    risk_level: RiskLevel

@dataclass
class RiskLinkageResult:
    should_proceed: bool
    risk_level: RiskLevel
    portfolio_metrics: PortfolioRiskMetrics
    risk_breaches: List[str]
    recommendations: List[str]

class PortfolioRiskLinkage:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.max_portfolio_var_95 = Decimal('0.05')
        self.max_concentration = Decimal('0.30')
        self.max_leverage = Decimal('3.0')
        
        self.asset_limits = {
            'XRP': Decimal('0.25'),
            'BTC': Decimal('0.20'),
            'ETH': Decimal('0.20')
        }
        
        self.blocked_trades = 0
        self.allowed_trades = 0
    
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]], 
                               market_data: Dict[str, Any], 
                               account_value: Decimal) -> PortfolioRiskMetrics:
        try:
            total_exposure = Decimal('0')
            for position in positions:
                size = Decimal(str(position.get('size', 0)))
                mark_price = Decimal(str(market_data.get('mark_price', 0)))
                total_exposure += abs(size * mark_price)
            
            portfolio_var_95 = total_exposure / account_value * Decimal('0.15') if account_value > 0 else Decimal('0')
            
            max_position_value = Decimal('0')
            for position in positions:
                size = Decimal(str(position.get('size', 0)))
                mark_price = Decimal(str(market_data.get('mark_price', 0)))
                position_value = abs(size * mark_price)
                max_position_value = max(max_position_value, position_value)
            
            concentration_risk = max_position_value / total_exposure if total_exposure > 0 else Decimal('0')
            leverage_ratio = total_exposure / account_value if account_value > 0 else Decimal('0')
            
            risk_level = self._determine_risk_level(portfolio_var_95, concentration_risk, leverage_ratio)
            
            return PortfolioRiskMetrics(
                total_exposure=total_exposure,
                portfolio_var_95=portfolio_var_95,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating portfolio risk: {e}")
            return PortfolioRiskMetrics(
                total_exposure=Decimal('0'),
                portfolio_var_95=Decimal('0'),
                concentration_risk=Decimal('0'),
                leverage_ratio=Decimal('0'),
                risk_level=RiskLevel.CRITICAL
            )
    
    def _determine_risk_level(self, portfolio_var: Decimal, concentration: Decimal, leverage: Decimal) -> RiskLevel:
        if (portfolio_var > self.max_portfolio_var_95 or 
            concentration > self.max_concentration or 
            leverage > self.max_leverage):
            return RiskLevel.CRITICAL
        elif (portfolio_var > self.max_portfolio_var_95 * Decimal('0.8') or 
              concentration > self.max_concentration * Decimal('0.8') or 
              leverage > self.max_leverage * Decimal('0.8')):
            return RiskLevel.HIGH
        elif (portfolio_var > self.max_portfolio_var_95 * Decimal('0.6') or 
              concentration > self.max_concentration * Decimal('0.6') or 
              leverage > self.max_leverage * Decimal('0.6')):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def should_proceed_with_trade(self, new_order: Dict[str, Any], 
                                current_positions: List[Dict[str, Any]], 
                                market_data: Dict[str, Any], 
                                account_value: Decimal) -> Tuple[bool, str, RiskLinkageResult]:
        try:
            # Simulate new position
            new_position = {
                'asset': new_order.get('asset', 'XRP'),
                'size': Decimal(str(new_order.get('size', 0))),
                'entry_price': Decimal(str(new_order.get('price', 0))),
                'side': new_order.get('side', 'BUY')
            }
            
            updated_positions = current_positions.copy()
            updated_positions.append(new_position)
            
            updated_risk = self.calculate_portfolio_risk(updated_positions, market_data, account_value)
            
            risk_breaches = []
            recommendations = []
            
            if updated_risk.portfolio_var_95 > self.max_portfolio_var_95:
                risk_breaches.append(f"Portfolio VaR 95% breach: {updated_risk.portfolio_var_95:.2%} > {self.max_portfolio_var_95:.2%}")
                recommendations.append("Reduce position size or close existing positions")
            
            if updated_risk.concentration_risk > self.max_concentration:
                risk_breaches.append(f"Concentration risk breach: {updated_risk.concentration_risk:.2%} > {self.max_concentration:.2%}")
                recommendations.append("Diversify positions across multiple assets")
            
            if updated_risk.leverage_ratio > self.max_leverage:
                risk_breaches.append(f"Leverage ratio breach: {updated_risk.leverage_ratio:.2f} > {self.max_leverage:.2f}")
                recommendations.append("Reduce leverage or increase account value")
            
            should_proceed = len(risk_breaches) == 0
            
            risk_result = RiskLinkageResult(
                should_proceed=should_proceed,
                risk_level=updated_risk.risk_level,
                portfolio_metrics=updated_risk,
                risk_breaches=risk_breaches,
                recommendations=recommendations
            )
            
            if should_proceed:
                self.allowed_trades += 1
                self.logger.info(f"✅ PORTFOLIO_RISK_PASSED: Trade within risk limits")
                return True, "Trade within portfolio risk limits", risk_result
            else:
                self.blocked_trades += 1
                reason = f"Portfolio risk limits breached: {', '.join(risk_breaches[:2])}"
                self.logger.warning(f"❌ PORTFOLIO_RISK_BLOCKED: {reason}")
                return False, reason, risk_result
                
        except Exception as e:
            self.logger.error(f"❌ Error in portfolio risk linkage: {e}")
            self.blocked_trades += 1
            return False, f"Portfolio risk linkage error: {e}", None
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        total_trades = self.blocked_trades + self.allowed_trades
        block_rate = (self.blocked_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "blocked_trades": self.blocked_trades,
            "allowed_trades": self.allowed_trades,
            "block_rate_percent": round(block_rate, 2),
            "risk_limits": {
                "max_portfolio_var_95": str(self.max_portfolio_var_95),
                "max_concentration": str(self.max_concentration),
                "max_leverage": str(self.max_leverage)
            },
            "last_updated": datetime.now().isoformat()
        }

def demo_portfolio_risk_linkage():
    print("🔗 Portfolio Risk Linkage Demo")
    print("=" * 50)
    
    risk_linkage = PortfolioRiskLinkage()
    
    # Test 1: Normal trade
    print("🔍 Test 1: Normal trade")
    current_positions = [
        {'asset': 'XRP', 'size': 100, 'entry_price': 0.52}
    ]
    
    market_data = {'mark_price': 0.52}
    account_value = Decimal('10000')
    
    new_order = {
        'asset': 'XRP',
        'side': 'BUY',
        'size': 50,
        'price': 0.52
    }
    
    should_proceed, reason, risk_result = risk_linkage.should_proceed_with_trade(
        new_order, current_positions, market_data, account_value
    )
    
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    print(f"  Risk Level: {risk_result.risk_level.value}")
    
    # Show statistics
    stats = risk_linkage.get_risk_statistics()
    print(f"\n📊 Risk Linkage Statistics: {stats}")
    
    print(f"\n✅ Portfolio Risk Linkage Demo Complete")

if __name__ == "__main__":
    demo_portfolio_risk_linkage()
