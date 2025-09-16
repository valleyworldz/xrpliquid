"""
VaR/ES & Funding-Directional Guardrails
Implements regulatory-grade risk metrics and funding-directional controls.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.stats import norm


@dataclass
class VaRESMetrics:
    """VaR and ES risk metrics."""
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    expected_return: float
    volatility: float
    skewness: float
    kurtosis: float
    confidence_level: float


@dataclass
class FundingDirectionalMetrics:
    """Funding vs directional risk metrics."""
    funding_pnl: float
    directional_pnl: float
    funding_directional_correlation: float
    net_funding_exposure: float
    directional_exposure: float
    risk_ratio: float


class VaRESGuardrails:
    """Implements VaR/ES and funding-directional guardrails."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.risk_dir = self.reports_dir / "risk"
        self.risk_dir.mkdir(parents=True, exist_ok=True)
        
        # Risk limits
        self.risk_limits = {
            "daily_var_95_limit": 0.05,      # 5% daily VaR limit
            "daily_var_99_limit": 0.10,      # 10% daily VaR limit
            "daily_es_95_limit": 0.08,       # 8% daily ES limit
            "daily_es_99_limit": 0.15,       # 15% daily ES limit
            "funding_directional_ratio_limit": 0.3,  # 30% funding vs directional ratio
            "max_funding_exposure": 0.20,    # 20% max funding exposure
            "max_directional_exposure": 0.30  # 30% max directional exposure
        }
        
        # Historical VaR/ES data
        self.historical_metrics = []
    
    def calculate_var_es(self, 
                        returns: pd.Series,
                        confidence_levels: List[float] = [0.95, 0.99]) -> VaRESMetrics:
        """Calculate Value-at-Risk and Expected Shortfall."""
        
        if returns.empty:
            return VaRESMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0.95)
        
        # Calculate basic statistics
        expected_return = returns.mean()
        volatility = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Parametric VaR/ES (assuming normal distribution)
        var_95 = norm.ppf(0.05) * volatility + expected_return
        var_99 = norm.ppf(0.01) * volatility + expected_return
        
        # Expected Shortfall (Conditional VaR)
        es_95 = -expected_return + volatility * norm.pdf(norm.ppf(0.05)) / 0.05
        es_99 = -expected_return + volatility * norm.pdf(norm.ppf(0.01)) / 0.01
        
        # Historical VaR/ES (non-parametric)
        historical_var_95 = np.percentile(returns, 5)
        historical_var_99 = np.percentile(returns, 1)
        
        # Use the more conservative estimate
        final_var_95 = min(var_95, historical_var_95)
        final_var_99 = min(var_99, historical_var_99)
        
        # Historical ES
        historical_es_95 = returns[returns <= historical_var_95].mean()
        historical_es_99 = returns[returns <= historical_var_99].mean()
        
        final_es_95 = min(es_95, abs(historical_es_95))
        final_es_99 = min(es_99, abs(historical_es_99))
        
        return VaRESMetrics(
            var_95=final_var_95,
            var_99=final_var_99,
            es_95=final_es_95,
            es_99=final_es_99,
            expected_return=expected_return,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            confidence_level=0.95
        )
    
    def calculate_funding_directional_metrics(self, 
                                            funding_pnl: pd.Series,
                                            directional_pnl: pd.Series) -> FundingDirectionalMetrics:
        """Calculate funding vs directional risk metrics."""
        
        if funding_pnl.empty or directional_pnl.empty:
            return FundingDirectionalMetrics(0, 0, 0, 0, 0, 0)
        
        # Calculate PnL components
        total_funding_pnl = funding_pnl.sum()
        total_directional_pnl = directional_pnl.sum()
        
        # Calculate correlation
        correlation = funding_pnl.corr(directional_pnl) if len(funding_pnl) > 1 else 0.0
        
        # Calculate exposures (as percentage of total PnL)
        total_pnl = total_funding_pnl + total_directional_pnl
        if total_pnl != 0:
            funding_exposure = abs(total_funding_pnl) / abs(total_pnl)
            directional_exposure = abs(total_directional_pnl) / abs(total_pnl)
        else:
            funding_exposure = 0.0
            directional_exposure = 0.0
        
        # Calculate risk ratio
        if abs(total_directional_pnl) > 0:
            risk_ratio = abs(total_funding_pnl) / abs(total_directional_pnl)
        else:
            risk_ratio = float('inf') if total_funding_pnl != 0 else 0.0
        
        return FundingDirectionalMetrics(
            funding_pnl=total_funding_pnl,
            directional_pnl=total_directional_pnl,
            funding_directional_correlation=correlation,
            net_funding_exposure=funding_exposure,
            directional_exposure=directional_exposure,
            risk_ratio=risk_ratio
        )
    
    def check_risk_limits(self, 
                         var_es_metrics: VaRESMetrics,
                         funding_directional_metrics: FundingDirectionalMetrics) -> Dict[str, Any]:
        """Check risk limits and generate alerts."""
        
        alerts = []
        violations = []
        
        # VaR limit checks
        if abs(var_es_metrics.var_95) > self.risk_limits["daily_var_95_limit"]:
            violations.append({
                "type": "var_95_limit",
                "current_value": abs(var_es_metrics.var_95),
                "limit": self.risk_limits["daily_var_95_limit"],
                "severity": "high"
            })
            alerts.append(f"VaR 95% limit exceeded: {abs(var_es_metrics.var_95):.3f} > {self.risk_limits['daily_var_95_limit']:.3f}")
        
        if abs(var_es_metrics.var_99) > self.risk_limits["daily_var_99_limit"]:
            violations.append({
                "type": "var_99_limit",
                "current_value": abs(var_es_metrics.var_99),
                "limit": self.risk_limits["daily_var_99_limit"],
                "severity": "critical"
            })
            alerts.append(f"VaR 99% limit exceeded: {abs(var_es_metrics.var_99):.3f} > {self.risk_limits['daily_var_99_limit']:.3f}")
        
        # ES limit checks
        if abs(var_es_metrics.es_95) > self.risk_limits["daily_es_95_limit"]:
            violations.append({
                "type": "es_95_limit",
                "current_value": abs(var_es_metrics.es_95),
                "limit": self.risk_limits["daily_es_95_limit"],
                "severity": "high"
            })
            alerts.append(f"ES 95% limit exceeded: {abs(var_es_metrics.es_95):.3f} > {self.risk_limits['daily_es_95_limit']:.3f}")
        
        if abs(var_es_metrics.es_99) > self.risk_limits["daily_es_99_limit"]:
            violations.append({
                "type": "es_99_limit",
                "current_value": abs(var_es_metrics.es_99),
                "limit": self.risk_limits["daily_es_99_limit"],
                "severity": "critical"
            })
            alerts.append(f"ES 99% limit exceeded: {abs(var_es_metrics.es_99):.3f} > {self.risk_limits['daily_es_99_limit']:.3f}")
        
        # Funding-directional limit checks
        if funding_directional_metrics.risk_ratio > self.risk_limits["funding_directional_ratio_limit"]:
            violations.append({
                "type": "funding_directional_ratio",
                "current_value": funding_directional_metrics.risk_ratio,
                "limit": self.risk_limits["funding_directional_ratio_limit"],
                "severity": "medium"
            })
            alerts.append(f"Funding-directional ratio exceeded: {funding_directional_metrics.risk_ratio:.3f} > {self.risk_limits['funding_directional_ratio_limit']:.3f}")
        
        if funding_directional_metrics.net_funding_exposure > self.risk_limits["max_funding_exposure"]:
            violations.append({
                "type": "max_funding_exposure",
                "current_value": funding_directional_metrics.net_funding_exposure,
                "limit": self.risk_limits["max_funding_exposure"],
                "severity": "high"
            })
            alerts.append(f"Max funding exposure exceeded: {funding_directional_metrics.net_funding_exposure:.3f} > {self.risk_limits['max_funding_exposure']:.3f}")
        
        if funding_directional_metrics.directional_exposure > self.risk_limits["max_directional_exposure"]:
            violations.append({
                "type": "max_directional_exposure",
                "current_value": funding_directional_metrics.directional_exposure,
                "limit": self.risk_limits["max_directional_exposure"],
                "severity": "high"
            })
            alerts.append(f"Max directional exposure exceeded: {funding_directional_metrics.directional_exposure:.3f} > {self.risk_limits['max_directional_exposure']:.3f}")
        
        # Determine overall status
        critical_violations = [v for v in violations if v["severity"] == "critical"]
        high_violations = [v for v in violations if v["severity"] == "high"]
        
        if critical_violations:
            status = "critical"
        elif high_violations:
            status = "warning"
        elif violations:
            status = "caution"
        else:
            status = "normal"
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "violations": violations,
            "alerts": alerts,
            "var_es_metrics": {
                "var_95": var_es_metrics.var_95,
                "var_99": var_es_metrics.var_99,
                "es_95": var_es_metrics.es_95,
                "es_99": var_es_metrics.es_99,
                "volatility": var_es_metrics.volatility,
                "skewness": var_es_metrics.skewness,
                "kurtosis": var_es_metrics.kurtosis
            },
            "funding_directional_metrics": {
                "funding_pnl": funding_directional_metrics.funding_pnl,
                "directional_pnl": funding_directional_metrics.directional_pnl,
                "correlation": funding_directional_metrics.funding_directional_correlation,
                "funding_exposure": funding_directional_metrics.net_funding_exposure,
                "directional_exposure": funding_directional_metrics.directional_exposure,
                "risk_ratio": funding_directional_metrics.risk_ratio
            }
        }
    
    def generate_daily_risk_snapshot(self, 
                                   returns: pd.Series,
                                   funding_pnl: pd.Series,
                                   directional_pnl: pd.Series) -> Dict[str, Any]:
        """Generate daily VaR/ES snapshot."""
        
        # Calculate VaR/ES metrics
        var_es_metrics = self.calculate_var_es(returns)
        
        # Calculate funding-directional metrics
        funding_directional_metrics = self.calculate_funding_directional_metrics(funding_pnl, directional_pnl)
        
        # Check risk limits
        risk_check = self.check_risk_limits(var_es_metrics, funding_directional_metrics)
        
        # Store historical metrics
        self.historical_metrics.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "var_95": var_es_metrics.var_95,
            "var_99": var_es_metrics.var_99,
            "es_95": var_es_metrics.es_95,
            "es_99": var_es_metrics.es_99,
            "volatility": var_es_metrics.volatility,
            "funding_pnl": funding_directional_metrics.funding_pnl,
            "directional_pnl": funding_directional_metrics.directional_pnl,
            "risk_ratio": funding_directional_metrics.risk_ratio
        })
        
        # Keep only last 252 days (1 year)
        if len(self.historical_metrics) > 252:
            self.historical_metrics = self.historical_metrics[-252:]
        
        # Generate snapshot
        snapshot = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "risk_check": risk_check,
            "historical_summary": self._calculate_historical_summary(),
            "recommendations": self._generate_risk_recommendations(risk_check)
        }
        
        # Save snapshot
        snapshot_file = self.risk_dir / f"var_es_snapshot_{datetime.now().strftime('%Y%m%d')}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        return snapshot
    
    def _calculate_historical_summary(self) -> Dict[str, Any]:
        """Calculate historical risk summary."""
        
        if not self.historical_metrics:
            return {"error": "No historical data available"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.historical_metrics)
        
        return {
            "days_analyzed": len(df),
            "avg_var_95": float(df["var_95"].mean()),
            "max_var_95": float(df["var_95"].max()),
            "avg_var_99": float(df["var_99"].mean()),
            "max_var_99": float(df["var_99"].max()),
            "avg_es_95": float(df["es_95"].mean()),
            "max_es_95": float(df["es_95"].max()),
            "avg_volatility": float(df["volatility"].mean()),
            "max_volatility": float(df["volatility"].max()),
            "avg_risk_ratio": float(df["risk_ratio"].mean()),
            "max_risk_ratio": float(df["risk_ratio"].max())
        }
    
    def _generate_risk_recommendations(self, risk_check: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations."""
        
        recommendations = []
        
        if risk_check["status"] == "critical":
            recommendations.append("IMMEDIATE ACTION REQUIRED: Critical risk limits exceeded - consider reducing positions")
            recommendations.append("Review and potentially halt trading until risk metrics normalize")
        
        elif risk_check["status"] == "warning":
            recommendations.append("High risk levels detected - consider reducing position sizes")
            recommendations.append("Monitor risk metrics closely and be prepared to take action")
        
        elif risk_check["status"] == "caution":
            recommendations.append("Risk levels elevated - monitor closely")
            recommendations.append("Consider reducing leverage or position sizes")
        
        # Specific recommendations based on violations
        for violation in risk_check["violations"]:
            if violation["type"] == "funding_directional_ratio":
                recommendations.append("Reduce funding exposure relative to directional exposure")
            elif violation["type"] == "max_funding_exposure":
                recommendations.append("Reduce funding position sizes")
            elif violation["type"] == "max_directional_exposure":
                recommendations.append("Reduce directional position sizes")
        
        return recommendations


def main():
    """Test VaR/ES guardrails functionality."""
    guardrails = VaRESGuardrails()
    
    # Create sample data
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100) * 0.02)
    funding_pnl = pd.Series(np.random.randn(100) * 0.01)
    directional_pnl = pd.Series(np.random.randn(100) * 0.015)
    
    # Calculate VaR/ES
    var_es_metrics = guardrails.calculate_var_es(returns)
    print(f"✅ VaR 95%: {var_es_metrics.var_95:.4f}, ES 95%: {var_es_metrics.es_95:.4f}")
    
    # Calculate funding-directional metrics
    funding_metrics = guardrails.calculate_funding_directional_metrics(funding_pnl, directional_pnl)
    print(f"✅ Funding PnL: {funding_metrics.funding_pnl:.2f}, Risk ratio: {funding_metrics.risk_ratio:.2f}")
    
    # Check risk limits
    risk_check = guardrails.check_risk_limits(var_es_metrics, funding_metrics)
    print(f"✅ Risk status: {risk_check['status']}")
    
    # Generate daily snapshot
    snapshot = guardrails.generate_daily_risk_snapshot(returns, funding_pnl, directional_pnl)
    print(f"✅ Daily risk snapshot generated")
    
    print("✅ VaR/ES guardrails testing completed")


if __name__ == "__main__":
    main()
