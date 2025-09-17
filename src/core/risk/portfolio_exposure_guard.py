"""
Portfolio Exposure Guard - Pre-trade portfolio caps enforcement
"""

import logging
import json
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class ExposureLimit(Enum):
    PER_ASSET = "per_asset"
    PER_FACTOR = "per_factor"
    PORTFOLIO_VAR = "portfolio_var"
    CONCENTRATION = "concentration"

@dataclass
class ExposureCheck:
    limit_type: ExposureLimit
    current_exposure: Decimal
    limit: Decimal
    utilization_pct: Decimal
    breached: bool
    asset: Optional[str] = None
    factor: Optional[str] = None

@dataclass
class PortfolioExposureResult:
    overall_breach: bool
    checks: List[ExposureCheck]
    total_exposure: Decimal
    portfolio_var_95: Decimal
    recommendations: List[str]

class PortfolioExposureGuard:
    """
    Pre-trade portfolio exposure guard with caps enforcement
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Exposure limits (configurable via environment)
        self.per_asset_limits = {
            'XRP': Decimal('0.30'),  # 30% max per asset
            'BTC': Decimal('0.25'),
            'ETH': Decimal('0.25'),
            'SOL': Decimal('0.20')
        }
        
        self.per_factor_limits = {
            'momentum': Decimal('0.40'),  # 40% max momentum exposure
            'mean_reversion': Decimal('0.30'),
            'volatility': Decimal('0.25'),
            'funding': Decimal('0.20')
        }
        
        self.portfolio_var_limit = Decimal('0.05')  # 5% portfolio VaR limit
        self.concentration_limit = Decimal('0.60')  # 60% max concentration
        
        self.blocked_orders = 0
        self.allowed_orders = 0
        
    def calculate_current_exposures(self, positions: Dict[str, Any], account_value: Decimal) -> Dict[str, Decimal]:
        """
        Calculate current portfolio exposures
        """
        exposures = {
            'per_asset': {},
            'per_factor': {},
            'total_exposure': Decimal('0'),
            'portfolio_var_95': Decimal('0')
        }
        
        try:
            # Calculate per-asset exposures
            for asset, position in positions.items():
                if isinstance(position, dict) and 'size' in position:
                    position_value = abs(Decimal(str(position['size'])) * Decimal(str(position.get('mark_price', 0))))
                    exposures['per_asset'][asset] = position_value / account_value if account_value > 0 else Decimal('0')
                    exposures['total_exposure'] += position_value
            
            # Calculate per-factor exposures (simplified)
            # In practice, this would use factor models
            for factor in self.per_factor_limits.keys():
                # Simplified factor exposure calculation
                factor_exposure = Decimal('0')
                for asset, position in positions.items():
                    if isinstance(position, dict) and 'size' in position:
                        # Assign factors based on asset characteristics
                        if asset == 'XRP':
                            factor_exposure += exposures['per_asset'].get(asset, Decimal('0')) * Decimal('0.3')
                        elif asset == 'BTC':
                            factor_exposure += exposures['per_asset'].get(asset, Decimal('0')) * Decimal('0.4')
                        elif asset == 'ETH':
                            factor_exposure += exposures['per_asset'].get(asset, Decimal('0')) * Decimal('0.3')
                
                exposures['per_factor'][factor] = factor_exposure
            
            # Calculate portfolio VaR (simplified)
            # In practice, this would use historical simulation or parametric methods
            total_exposure_pct = exposures['total_exposure'] / account_value if account_value > 0 else Decimal('0')
            exposures['portfolio_var_95'] = total_exposure_pct * Decimal('0.15')  # Simplified 15% volatility assumption
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating exposures: {e}")
        
        return exposures
    
    def check_exposure_limits(self, new_order: Dict[str, Any], current_exposures: Dict[str, Decimal], account_value: Decimal) -> PortfolioExposureResult:
        """
        Check if new order would breach exposure limits
        """
        checks = []
        recommendations = []
        
        try:
            # Extract order details
            asset = new_order.get('asset', 'XRP')
            side = new_order.get('side', 'BUY')
            size = Decimal(str(new_order.get('size', 0)))
            price = Decimal(str(new_order.get('price', 0)))
            
            # Calculate new position value
            order_value = size * price
            order_exposure_pct = order_value / account_value if account_value > 0 else Decimal('0')
            
            # Check per-asset limits
            current_asset_exposure = current_exposures['per_asset'].get(asset, Decimal('0'))
            new_asset_exposure = current_asset_exposure + order_exposure_pct
            asset_limit = self.per_asset_limits.get(asset, Decimal('0.30'))
            
            asset_check = ExposureCheck(
                limit_type=ExposureLimit.PER_ASSET,
                current_exposure=new_asset_exposure,
                limit=asset_limit,
                utilization_pct=(new_asset_exposure / asset_limit * 100) if asset_limit > 0 else Decimal('0'),
                breached=new_asset_exposure > asset_limit,
                asset=asset
            )
            checks.append(asset_check)
            
            if asset_check.breached:
                recommendations.append(f"Reduce {asset} exposure: {new_asset_exposure:.2%} > {asset_limit:.2%}")
            
            # Check per-factor limits
            for factor, factor_limit in self.per_factor_limits.items():
                current_factor_exposure = current_exposures['per_factor'].get(factor, Decimal('0'))
                # Simplified factor impact calculation
                factor_impact = order_exposure_pct * Decimal('0.3')  # Assume 30% factor loading
                new_factor_exposure = current_factor_exposure + factor_impact
                
                factor_check = ExposureCheck(
                    limit_type=ExposureLimit.PER_FACTOR,
                    current_exposure=new_factor_exposure,
                    limit=factor_limit,
                    utilization_pct=(new_factor_exposure / factor_limit * 100) if factor_limit > 0 else Decimal('0'),
                    breached=new_factor_exposure > factor_limit,
                    factor=factor
                )
                checks.append(factor_check)
                
                if factor_check.breached:
                    recommendations.append(f"Reduce {factor} factor exposure: {new_factor_exposure:.2%} > {factor_limit:.2%}")
            
            # Check portfolio VaR limit
            current_portfolio_var = current_exposures['portfolio_var_95']
            new_portfolio_var = current_portfolio_var + (order_exposure_pct * Decimal('0.15'))  # Simplified VaR impact
            
            var_check = ExposureCheck(
                limit_type=ExposureLimit.PORTFOLIO_VAR,
                current_exposure=new_portfolio_var,
                limit=self.portfolio_var_limit,
                utilization_pct=(new_portfolio_var / self.portfolio_var_limit * 100) if self.portfolio_var_limit > 0 else Decimal('0'),
                breached=new_portfolio_var > self.portfolio_var_limit
            )
            checks.append(var_check)
            
            if var_check.breached:
                recommendations.append(f"Portfolio VaR limit breached: {new_portfolio_var:.2%} > {self.portfolio_var_limit:.2%}")
            
            # Check concentration limit
            total_exposure = current_exposures['total_exposure'] + order_value
            concentration_pct = total_exposure / account_value if account_value > 0 else Decimal('0')
            
            concentration_check = ExposureCheck(
                limit_type=ExposureLimit.CONCENTRATION,
                current_exposure=concentration_pct,
                limit=self.concentration_limit,
                utilization_pct=(concentration_pct / self.concentration_limit * 100) if self.concentration_limit > 0 else Decimal('0'),
                breached=concentration_pct > self.concentration_limit
            )
            checks.append(concentration_check)
            
            if concentration_check.breached:
                recommendations.append(f"Concentration limit breached: {concentration_pct:.2%} > {self.concentration_limit:.2%}")
            
            # Determine overall breach
            overall_breach = any(check.breached for check in checks)
            
            return PortfolioExposureResult(
                overall_breach=overall_breach,
                checks=checks,
                total_exposure=total_exposure,
                portfolio_var_95=new_portfolio_var,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error checking exposure limits: {e}")
            return PortfolioExposureResult(
                overall_breach=True,
                checks=[],
                total_exposure=Decimal('0'),
                portfolio_var_95=Decimal('0'),
                recommendations=[f"Error checking limits: {e}"]
            )
    
    def should_proceed_with_order(self, new_order: Dict[str, Any], positions: Dict[str, Any], account_value: Decimal) -> Tuple[bool, str, PortfolioExposureResult]:
        """
        Main gate function - returns (should_proceed, reason, exposure_result)
        """
        try:
            # Calculate current exposures
            current_exposures = self.calculate_current_exposures(positions, account_value)
            
            # Check exposure limits
            exposure_result = self.check_exposure_limits(new_order, current_exposures, account_value)
            
            if not exposure_result.overall_breach:
                self.allowed_orders += 1
                self.logger.info(f"✅ PORTFOLIO_EXPOSURE_PASSED: Order within limits")
                return True, "Order within portfolio exposure limits", exposure_result
            else:
                self.blocked_orders += 1
                reason = f"Portfolio exposure limits breached: {', '.join(exposure_result.recommendations[:2])}"
                self.logger.warning(f"❌ PORTFOLIO_EXPOSURE_BLOCKED: {reason}")
                
                # Log structured JSON event
                self.log_exposure_breach(exposure_result, new_order)
                
                return False, reason, exposure_result
                
        except Exception as e:
            self.logger.error(f"❌ Error in portfolio exposure gate: {e}")
            self.blocked_orders += 1
            return False, f"Portfolio exposure gate error: {e}", None
    
    def log_exposure_breach(self, exposure_result: PortfolioExposureResult, order_data: Dict[str, Any]):
        """
        Log structured JSON event for exposure breach
        """
        try:
            breach_event = {
                "event": "portfolio_exposure_breach",
                "timestamp": datetime.now().isoformat(),
                "overall_breach": exposure_result.overall_breach,
                "total_exposure": str(exposure_result.total_exposure),
                "portfolio_var_95": str(exposure_result.portfolio_var_95),
                "breached_limits": [
                    {
                        "limit_type": check.limit_type.value,
                        "current_exposure": str(check.current_exposure),
                        "limit": str(check.limit),
                        "utilization_pct": str(check.utilization_pct),
                        "asset": check.asset,
                        "factor": check.factor
                    }
                    for check in exposure_result.checks if check.breached
                ],
                "recommendations": exposure_result.recommendations,
                "order_data": {
                    "asset": order_data.get('asset'),
                    "side": order_data.get('side'),
                    "size": str(order_data.get('size', 0)),
                    "price": str(order_data.get('price', 0))
                },
                "blocked_orders": self.blocked_orders,
                "allowed_orders": self.allowed_orders
            }
            
            # Log as structured JSON
            self.logger.warning(f"🚫 PORTFOLIO_EXPOSURE_BREACH: {json.dumps(breach_event)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging exposure breach: {e}")
    
    def get_guard_statistics(self) -> Dict[str, Any]:
        """
        Get portfolio exposure guard statistics
        """
        total_orders = self.blocked_orders + self.allowed_orders
        block_rate = (self.blocked_orders / total_orders * 100) if total_orders > 0 else 0
        
        return {
            "total_orders": total_orders,
            "blocked_orders": self.blocked_orders,
            "allowed_orders": self.allowed_orders,
            "block_rate_percent": round(block_rate, 2),
            "per_asset_limits": {k: str(v) for k, v in self.per_asset_limits.items()},
            "per_factor_limits": {k: str(v) for k, v in self.per_factor_limits.items()},
            "portfolio_var_limit": str(self.portfolio_var_limit),
            "concentration_limit": str(self.concentration_limit),
            "last_updated": datetime.now().isoformat()
        }

# Global guard instance
_exposure_guard = PortfolioExposureGuard()

def should_proceed_with_order(new_order: Dict[str, Any], positions: Dict[str, Any], account_value: Decimal) -> Tuple[bool, str, PortfolioExposureResult]:
    """
    Global function to check if order should proceed based on portfolio exposure
    """
    return _exposure_guard.should_proceed_with_order(new_order, positions, account_value)

def get_exposure_guard() -> PortfolioExposureGuard:
    """
    Get the global exposure guard instance
    """
    return _exposure_guard

# Demo function
def demo_portfolio_exposure_guard():
    """Demo the portfolio exposure guard"""
    print("📊 Portfolio Exposure Guard Demo")
    print("=" * 50)
    
    guard = PortfolioExposureGuard()
    
    # Test 1: Order within limits
    print("🔍 Test 1: Order within limits")
    positions = {
        'XRP': {'size': 100, 'mark_price': 0.52},
        'BTC': {'size': 0.1, 'mark_price': 50000}
    }
    account_value = Decimal('10000')
    
    new_order = {
        'asset': 'XRP',
        'side': 'BUY',
        'size': 50,
        'price': 0.52
    }
    
    should_proceed, reason, exposure_result = guard.should_proceed_with_order(new_order, positions, account_value)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    print(f"  Portfolio VaR: {exposure_result.portfolio_var_95:.2%}")
    
    # Test 2: Order exceeds asset limit
    print(f"\n🔍 Test 2: Order exceeds asset limit")
    large_order = {
        'asset': 'XRP',
        'side': 'BUY',
        'size': 1000,  # Large order
        'price': 0.52
    }
    
    should_proceed, reason, exposure_result = guard.should_proceed_with_order(large_order, positions, account_value)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    print(f"  Recommendations: {exposure_result.recommendations[:2]}")
    
    # Test 3: Order exceeds portfolio VaR limit
    print(f"\n🔍 Test 3: Order exceeds portfolio VaR limit")
    var_order = {
        'asset': 'BTC',
        'side': 'BUY',
        'size': 1.0,  # Large BTC order
        'price': 50000
    }
    
    should_proceed, reason, exposure_result = guard.should_proceed_with_order(var_order, positions, account_value)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    print(f"  Portfolio VaR: {exposure_result.portfolio_var_95:.2%}")
    
    # Show statistics
    print(f"\n📊 Guard Statistics:")
    stats = guard.get_guard_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Portfolio Exposure Guard Demo Complete")

if __name__ == "__main__":
    demo_portfolio_exposure_guard()
