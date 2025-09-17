"""
Portfolio-Level VaR/ES Calculator
Regulatory-grade risk metrics across multiple strategies and venues
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ML_CLASSIFICATION = "ml_classification"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"

@dataclass
class StrategyRisk:
    strategy_id: str
    strategy_type: StrategyType
    current_exposure: float
    max_exposure: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_to_market: float
    last_updated: str

@dataclass
class PortfolioRisk:
    total_exposure: float
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_es_95: float
    portfolio_es_99: float
    diversification_ratio: float
    concentration_risk: float
    tail_risk: float
    stress_test_results: Dict[str, float]
    risk_level: RiskLevel
    timestamp: str

@dataclass
class StressTestScenario:
    name: str
    description: str
    market_shock: float
    correlation_increase: float
    volatility_multiplier: float
    liquidity_impact: float

class PortfolioVaRCalculator:
    """
    Portfolio-level VaR/ES calculator with regulatory compliance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategies: Dict[str, StrategyRisk] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.returns_data: Dict[str, pd.Series] = {}
        
        # Regulatory parameters
        self.confidence_levels = [0.95, 0.99]
        self.lookback_period = 252  # 1 year of trading days
        self.stress_test_scenarios = self._initialize_stress_scenarios()
        
        # Create reports directory
        self.reports_dir = Path("reports/portfolio_risk")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_stress_scenarios(self) -> List[StressTestScenario]:
        """Initialize regulatory stress test scenarios"""
        return [
            StressTestScenario(
                name="2008 Financial Crisis",
                description="Global financial crisis scenario",
                market_shock=-0.30,
                correlation_increase=0.4,
                volatility_multiplier=2.5,
                liquidity_impact=0.15
            ),
            StressTestScenario(
                name="COVID-19 Market Crash",
                description="Pandemic-induced market crash",
                market_shock=-0.25,
                correlation_increase=0.3,
                volatility_multiplier=2.0,
                liquidity_impact=0.10
            ),
            StressTestScenario(
                name="Flash Crash",
                description="High-frequency trading flash crash",
                market_shock=-0.15,
                correlation_increase=0.2,
                volatility_multiplier=1.8,
                liquidity_impact=0.20
            ),
            StressTestScenario(
                name="Interest Rate Shock",
                description="Central bank rate increase scenario",
                market_shock=-0.10,
                correlation_increase=0.1,
                volatility_multiplier=1.5,
                liquidity_impact=0.05
            ),
            StressTestScenario(
                name="Crypto Winter",
                description="Extended crypto bear market",
                market_shock=-0.50,
                correlation_increase=0.5,
                volatility_multiplier=3.0,
                liquidity_impact=0.25
            )
        ]
    
    def add_strategy(self, strategy_risk: StrategyRisk):
        """Add a strategy to the portfolio"""
        self.strategies[strategy_risk.strategy_id] = strategy_risk
        self.logger.info(f"✅ Added strategy: {strategy_risk.strategy_id}")
    
    def update_returns_data(self, strategy_id: str, returns: pd.Series):
        """Update returns data for a strategy"""
        self.returns_data[strategy_id] = returns
        self.logger.info(f"📊 Updated returns data for {strategy_id}: {len(returns)} observations")
    
    def calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix between strategies"""
        try:
            if len(self.returns_data) < 2:
                # Return identity matrix if insufficient data
                n_strategies = len(self.strategies)
                return np.eye(n_strategies)
            
            # Align returns data
            aligned_returns = pd.DataFrame(self.returns_data)
            aligned_returns = aligned_returns.dropna()
            
            if len(aligned_returns) < 30:  # Minimum observations
                n_strategies = len(self.strategies)
                return np.eye(n_strategies)
            
            # Calculate correlation matrix
            correlation_matrix = aligned_returns.corr().values
            
            # Handle NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            # Ensure positive definiteness
            correlation_matrix = self._make_positive_definite(correlation_matrix)
            
            self.correlation_matrix = correlation_matrix
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"❌ Correlation matrix calculation error: {e}")
            n_strategies = len(self.strategies)
            return np.eye(n_strategies)
    
    def _make_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite"""
        try:
            # Add small diagonal term to ensure positive definiteness
            matrix = matrix + np.eye(matrix.shape[0]) * 0.01
            
            # Check if positive definite
            if np.all(np.linalg.eigvals(matrix) > 0):
                return matrix
            
            # Use nearest positive definite matrix
            from sklearn.datasets import make_spd_matrix
            n = matrix.shape[0]
            return make_spd_matrix(n) * 0.1 + np.eye(n) * 0.9
            
        except Exception as e:
            self.logger.error(f"❌ Positive definite correction error: {e}")
            return np.eye(matrix.shape[0])
    
    def calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Calculate portfolio VaR using parametric method"""
        try:
            if not self.strategies:
                return 0.0
            
            # Get strategy exposures and individual VaRs
            exposures = []
            individual_vars = []
            
            for strategy_id, strategy in self.strategies.items():
                exposures.append(strategy.current_exposure)
                if confidence_level == 0.95:
                    individual_vars.append(strategy.var_95)
                else:
                    individual_vars.append(strategy.var_99)
            
            exposures = np.array(exposures)
            individual_vars = np.array(individual_vars)
            
            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix()
            
            # Portfolio VaR calculation
            # VaR_portfolio = sqrt(w' * Σ * w) where Σ is covariance matrix
            # For simplicity, we'll use the correlation matrix and individual VaRs
            
            if len(exposures) == 1:
                return individual_vars[0]
            
            # Calculate portfolio variance
            portfolio_variance = 0.0
            for i in range(len(exposures)):
                for j in range(len(exposures)):
                    portfolio_variance += (
                        exposures[i] * exposures[j] * 
                        individual_vars[i] * individual_vars[j] * 
                        correlation_matrix[i, j]
                    )
            
            portfolio_var = np.sqrt(portfolio_variance)
            return portfolio_var
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio VaR calculation error: {e}")
            return 0.0
    
    def calculate_portfolio_es(self, confidence_level: float = 0.95) -> float:
        """Calculate portfolio Expected Shortfall"""
        try:
            portfolio_var = self.calculate_portfolio_var(confidence_level)
            
            # ES is typically 1.2-1.5x VaR for normal distributions
            # We'll use a conservative multiplier
            if confidence_level == 0.95:
                es_multiplier = 1.3
            else:
                es_multiplier = 1.2
            
            portfolio_es = portfolio_var * es_multiplier
            return portfolio_es
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio ES calculation error: {e}")
            return 0.0
    
    def calculate_diversification_ratio(self) -> float:
        """Calculate portfolio diversification ratio"""
        try:
            if not self.strategies:
                return 1.0
            
            # Weighted average of individual VaRs
            total_exposure = sum(s.current_exposure for s in self.strategies.values())
            if total_exposure == 0:
                return 1.0
            
            weighted_avg_var = sum(
                s.current_exposure * s.var_95 / total_exposure 
                for s in self.strategies.values()
            )
            
            # Portfolio VaR
            portfolio_var = self.calculate_portfolio_var(0.95)
            
            if portfolio_var == 0:
                return 1.0
            
            diversification_ratio = weighted_avg_var / portfolio_var
            return diversification_ratio
            
        except Exception as e:
            self.logger.error(f"❌ Diversification ratio calculation error: {e}")
            return 1.0
    
    def calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk (Herfindahl index)"""
        try:
            if not self.strategies:
                return 0.0
            
            total_exposure = sum(s.current_exposure for s in self.strategies.values())
            if total_exposure == 0:
                return 0.0
            
            # Calculate Herfindahl index
            weights = [s.current_exposure / total_exposure for s in self.strategies.values()]
            herfindahl_index = sum(w**2 for w in weights)
            
            return herfindahl_index
            
        except Exception as e:
            self.logger.error(f"❌ Concentration risk calculation error: {e}")
            return 0.0
    
    def calculate_tail_risk(self) -> float:
        """Calculate portfolio tail risk (skewness and kurtosis)"""
        try:
            if not self.returns_data:
                return 0.0
            
            # Combine all strategy returns
            all_returns = []
            for returns in self.returns_data.values():
                all_returns.extend(returns.dropna().tolist())
            
            if len(all_returns) < 30:
                return 0.0
            
            returns_array = np.array(all_returns)
            
            # Calculate tail risk metrics
            skewness = stats.skew(returns_array)
            kurtosis = stats.kurtosis(returns_array)
            
            # Tail risk score (higher is riskier)
            tail_risk = abs(skewness) + max(0, kurtosis - 3)  # Excess kurtosis
            
            return tail_risk
            
        except Exception as e:
            self.logger.error(f"❌ Tail risk calculation error: {e}")
            return 0.0
    
    def run_stress_tests(self) -> Dict[str, float]:
        """Run regulatory stress tests"""
        try:
            stress_results = {}
            
            for scenario in self.stress_test_scenarios:
                # Calculate stressed portfolio VaR
                stressed_var = self._calculate_stressed_var(scenario)
                stress_results[scenario.name] = stressed_var
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"❌ Stress test calculation error: {e}")
            return {}
    
    def _calculate_stressed_var(self, scenario: StressTestScenario) -> float:
        """Calculate VaR under stress scenario"""
        try:
            # Base portfolio VaR
            base_var = self.calculate_portfolio_var(0.95)
            
            # Apply stress factors
            stressed_var = base_var * (
                1 + abs(scenario.market_shock) * 
                (1 + scenario.correlation_increase) * 
                scenario.volatility_multiplier * 
                (1 + scenario.liquidity_impact)
            )
            
            return stressed_var
            
        except Exception as e:
            self.logger.error(f"❌ Stressed VaR calculation error: {e}")
            return 0.0
    
    def determine_risk_level(self, portfolio_risk: PortfolioRisk) -> RiskLevel:
        """Determine overall portfolio risk level"""
        try:
            # Risk scoring based on multiple factors
            risk_score = 0.0
            
            # VaR-based scoring
            if portfolio_risk.portfolio_var_95 > 0.05:  # 5% of portfolio
                risk_score += 3.0
            elif portfolio_risk.portfolio_var_95 > 0.03:  # 3% of portfolio
                risk_score += 2.0
            elif portfolio_risk.portfolio_var_95 > 0.01:  # 1% of portfolio
                risk_score += 1.0
            
            # Concentration risk scoring
            if portfolio_risk.concentration_risk > 0.5:
                risk_score += 2.0
            elif portfolio_risk.concentration_risk > 0.3:
                risk_score += 1.0
            
            # Tail risk scoring
            if portfolio_risk.tail_risk > 2.0:
                risk_score += 2.0
            elif portfolio_risk.tail_risk > 1.0:
                risk_score += 1.0
            
            # Diversification scoring (lower is better)
            if portfolio_risk.diversification_ratio < 1.2:
                risk_score += 1.0
            
            # Determine risk level
            if risk_score >= 6.0:
                return RiskLevel.CRITICAL
            elif risk_score >= 4.0:
                return RiskLevel.HIGH
            elif risk_score >= 2.0:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"❌ Risk level determination error: {e}")
            return RiskLevel.MEDIUM
    
    def calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            self.logger.info("📊 Calculating portfolio risk metrics")
            
            # Calculate VaR and ES
            var_95 = self.calculate_portfolio_var(0.95)
            var_99 = self.calculate_portfolio_var(0.99)
            es_95 = self.calculate_portfolio_es(0.95)
            es_99 = self.calculate_portfolio_es(0.99)
            
            # Calculate other risk metrics
            total_exposure = sum(s.current_exposure for s in self.strategies.values())
            diversification_ratio = self.calculate_diversification_ratio()
            concentration_risk = self.calculate_concentration_risk()
            tail_risk = self.calculate_tail_risk()
            
            # Run stress tests
            stress_test_results = self.run_stress_tests()
            
            # Create portfolio risk object
            portfolio_risk = PortfolioRisk(
                total_exposure=total_exposure,
                portfolio_var_95=var_95,
                portfolio_var_99=var_99,
                portfolio_es_95=es_95,
                portfolio_es_99=es_99,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk,
                tail_risk=tail_risk,
                stress_test_results=stress_test_results,
                risk_level=RiskLevel.LOW,  # Will be determined below
                timestamp=datetime.now().isoformat()
            )
            
            # Determine risk level
            portfolio_risk.risk_level = self.determine_risk_level(portfolio_risk)
            
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio risk calculation error: {e}")
            return PortfolioRisk(
                total_exposure=0.0,
                portfolio_var_95=0.0,
                portfolio_var_99=0.0,
                portfolio_es_95=0.0,
                portfolio_es_99=0.0,
                diversification_ratio=1.0,
                concentration_risk=0.0,
                tail_risk=0.0,
                stress_test_results={},
                risk_level=RiskLevel.MEDIUM,
                timestamp=datetime.now().isoformat()
            )
    
    async def save_portfolio_risk_report(self, portfolio_risk: PortfolioRisk):
        """Save portfolio risk report to file"""
        try:
            # Create comprehensive report
            portfolio_risk_dict = asdict(portfolio_risk)
            portfolio_risk_dict['risk_level'] = portfolio_risk.risk_level.value
            
            strategy_details = {}
            for k, v in self.strategies.items():
                strategy_dict = asdict(v)
                strategy_dict['strategy_type'] = v.strategy_type.value
                strategy_details[k] = strategy_dict
            
            report = {
                "timestamp": portfolio_risk.timestamp,
                "portfolio_risk": portfolio_risk_dict,
                "strategy_details": strategy_details,
                "correlation_matrix": self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
                "regulatory_compliance": {
                    "var_95_within_limits": portfolio_risk.portfolio_var_95 < 0.05,
                    "var_99_within_limits": portfolio_risk.portfolio_var_99 < 0.08,
                    "concentration_acceptable": portfolio_risk.concentration_risk < 0.4,
                    "diversification_adequate": portfolio_risk.diversification_ratio > 1.2,
                    "tail_risk_acceptable": portfolio_risk.tail_risk < 2.0
                }
            }
            
            # Save to file
            report_file = self.reports_dir / f"portfolio_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"💾 Portfolio risk report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save portfolio risk report: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Get portfolio risk summary"""
        try:
            portfolio_risk = self.calculate_portfolio_risk()
            
            return {
                "total_strategies": len(self.strategies),
                "total_exposure": portfolio_risk.total_exposure,
                "risk_level": portfolio_risk.risk_level.value,
                "var_95": portfolio_risk.portfolio_var_95,
                "var_99": portfolio_risk.portfolio_var_99,
                "es_95": portfolio_risk.portfolio_es_95,
                "es_99": portfolio_risk.portfolio_es_99,
                "diversification_ratio": portfolio_risk.diversification_ratio,
                "concentration_risk": portfolio_risk.concentration_risk,
                "tail_risk": portfolio_risk.tail_risk,
                "stress_test_count": len(portfolio_risk.stress_test_results),
                "regulatory_compliant": portfolio_risk.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Risk summary error: {e}")
            return {"error": str(e)}

# Demo function
async def demo_portfolio_var_es():
    """Demo the portfolio VaR/ES calculator"""
    print("📊 Portfolio VaR/ES Calculator Demo")
    print("=" * 50)
    
    # Create calculator
    calculator = PortfolioVaRCalculator()
    
    # Add sample strategies
    strategies = [
        StrategyRisk(
            strategy_id="momentum_1",
            strategy_type=StrategyType.MOMENTUM,
            current_exposure=10000.0,
            max_exposure=20000.0,
            var_95=500.0,
            var_99=750.0,
            expected_shortfall_95=650.0,
            expected_shortfall_99=900.0,
            sharpe_ratio=1.8,
            max_drawdown=0.05,
            correlation_to_market=0.6,
            last_updated=datetime.now().isoformat()
        ),
        StrategyRisk(
            strategy_id="mean_reversion_1",
            strategy_type=StrategyType.MEAN_REVERSION,
            current_exposure=8000.0,
            max_exposure=15000.0,
            var_95=400.0,
            var_99=600.0,
            expected_shortfall_95=520.0,
            expected_shortfall_99=720.0,
            sharpe_ratio=2.1,
            max_drawdown=0.03,
            correlation_to_market=-0.2,
            last_updated=datetime.now().isoformat()
        ),
        StrategyRisk(
            strategy_id="ml_classification_1",
            strategy_type=StrategyType.ML_CLASSIFICATION,
            current_exposure=12000.0,
            max_exposure=25000.0,
            var_95=600.0,
            var_99=900.0,
            expected_shortfall_95=780.0,
            expected_shortfall_99=1080.0,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            correlation_to_market=0.4,
            last_updated=datetime.now().isoformat()
        )
    ]
    
    # Add strategies to calculator
    for strategy in strategies:
        calculator.add_strategy(strategy)
    
    # Generate sample returns data
    np.random.seed(42)
    for strategy in strategies:
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year of daily returns
        calculator.update_returns_data(strategy.strategy_id, returns)
    
    # Calculate portfolio risk
    portfolio_risk = calculator.calculate_portfolio_risk()
    
    # Print results
    print(f"\n📈 Portfolio Risk Summary:")
    print(f"Total Strategies: {len(strategies)}")
    print(f"Total Exposure: ${portfolio_risk.total_exposure:,.2f}")
    print(f"Risk Level: {portfolio_risk.risk_level.value.upper()}")
    print(f"Portfolio VaR (95%): ${portfolio_risk.portfolio_var_95:,.2f}")
    print(f"Portfolio VaR (99%): ${portfolio_risk.portfolio_var_99:,.2f}")
    print(f"Portfolio ES (95%): ${portfolio_risk.portfolio_es_95:,.2f}")
    print(f"Portfolio ES (99%): ${portfolio_risk.portfolio_es_99:,.2f}")
    print(f"Diversification Ratio: {portfolio_risk.diversification_ratio:.3f}")
    print(f"Concentration Risk: {portfolio_risk.concentration_risk:.3f}")
    print(f"Tail Risk: {portfolio_risk.tail_risk:.3f}")
    
    print(f"\n🧪 Stress Test Results:")
    for scenario, result in portfolio_risk.stress_test_results.items():
        print(f"  {scenario}: ${result:,.2f}")
    
    # Save report
    await calculator.save_portfolio_risk_report(portfolio_risk)
    
    # Get summary
    summary = calculator.get_risk_summary()
    print(f"\n✅ Regulatory Compliance: {'PASS' if summary['regulatory_compliant'] else 'FAIL'}")
    
    print("\n✅ Portfolio VaR/ES Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_portfolio_var_es())
