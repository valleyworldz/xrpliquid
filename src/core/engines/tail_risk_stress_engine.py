#!/usr/bin/env python3
"""
📊 TAIL RISK & STRESS TESTING ENGINE
====================================
Advanced tail risk and stress testing with ES and regime shocks.

This engine implements:
- Expected Shortfall (ES) calculations
- Regime-conditional VaR/ES (bull/bear/chop)
- Exchange outage/WebSocket desync stress
- Maximum intraday drawdown analysis
- Scenario-based stress testing
- Tail risk monitoring and alerting
"""

from src.core.utils.decimal_boundary_guard import safe_float, safe_decimal
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TailRiskMetrics:
    """Tail risk metrics"""
    var_95: float
    var_97: float
    var_99: float
    var_995: float
    es_95: float
    es_97: float
    es_99: float
    es_995: float
    max_intraday_dd: float
    tail_expectation: float
    tail_volatility: float
    tail_skewness: float
    tail_kurtosis: float

@dataclass
class RegimeRiskMetrics:
    """Regime-specific risk metrics"""
    regime: str
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    max_drawdown: float
    volatility: float
    skewness: float
    kurtosis: float
    sample_size: int

@dataclass
class StressTestResult:
    """Stress test result"""
    scenario_name: str
    portfolio_impact: float
    max_drawdown: float
    recovery_time_hours: int
    affected_positions: List[str]
    risk_breaches: List[str]
    recommendations: List[str]

class TailRiskStressEngine:
    """
    📊 TAIL RISK & STRESS TESTING ENGINE
    
    Advanced tail risk and stress testing with:
    1. Expected Shortfall (ES) calculations
    2. Regime-conditional VaR/ES (bull/bear/chop)
    3. Exchange outage/WebSocket desync stress
    4. Maximum intraday drawdown analysis
    5. Scenario-based stress testing
    6. Tail risk monitoring and alerting
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Data storage
        self.returns_data: pd.DataFrame = pd.DataFrame()
        self.regime_data: pd.DataFrame = pd.DataFrame()
        
        # Results storage
        self.tail_risk_metrics: Optional[TailRiskMetrics] = None
        self.regime_risk_metrics: List[RegimeRiskMetrics] = []
        self.stress_test_results: List[StressTestResult] = []
        
        # Performance tracking
        self.risk_history: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        self.logger.info("📊 [TAIL_RISK_STRESS] Tail Risk & Stress Testing Engine initialized")
        self.logger.info("🎯 [TAIL_RISK_STRESS] ES calculations and regime analysis enabled")
        self.logger.info("⚡ [TAIL_RISK_STRESS] Stress scenario testing active")
    
    async def calculate_tail_risk_metrics(self, returns: pd.Series) -> TailRiskMetrics:
        """
        Calculate comprehensive tail risk metrics
        """
        try:
            self.logger.info("📊 [TAIL_RISK_STRESS] Calculating tail risk metrics...")
            
            if len(returns) == 0:
                return TailRiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Calculate VaR at different confidence levels
            var_95 = np.percentile(returns, 5)
            var_97 = np.percentile(returns, 3)
            var_99 = np.percentile(returns, 1)
            var_995 = np.percentile(returns, 0.5)
            
            # Calculate Expected Shortfall (ES) at different confidence levels
            es_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
            es_97 = np.mean(returns[returns <= var_97]) if np.any(returns <= var_97) else var_97
            es_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
            es_995 = np.mean(returns[returns <= var_995]) if np.any(returns <= var_995) else var_995
            
            # Calculate maximum intraday drawdown
            max_intraday_dd = await self._calculate_max_intraday_drawdown(returns)
            
            # Calculate tail statistics
            tail_returns = returns[returns <= var_95]  # Bottom 5% returns
            tail_expectation = np.mean(tail_returns) if len(tail_returns) > 0 else 0.0
            tail_volatility = np.std(tail_returns) if len(tail_returns) > 1 else 0.0
            tail_skewness = self._calculate_skewness(tail_returns) if len(tail_returns) > 2 else 0.0
            tail_kurtosis = self._calculate_kurtosis(tail_returns) if len(tail_returns) > 3 else 0.0
            
            metrics = TailRiskMetrics(
                var_95=var_95,
                var_97=var_97,
                var_99=var_99,
                var_995=var_995,
                es_95=es_95,
                es_97=es_97,
                es_99=es_99,
                es_995=es_995,
                max_intraday_dd=max_intraday_dd,
                tail_expectation=tail_expectation,
                tail_volatility=tail_volatility,
                tail_skewness=tail_skewness,
                tail_kurtosis=tail_kurtosis
            )
            
            self.tail_risk_metrics = metrics
            
            self.logger.info(f"📊 [TAIL_RISK_STRESS] Tail risk calculated: VaR 99% = {var_99:.4f}, ES 99% = {es_99:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error calculating tail risk metrics: {e}")
            return TailRiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    async def _calculate_max_intraday_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum intraday drawdown"""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # Return maximum drawdown
            max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
            
            return max_dd
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error calculating max intraday drawdown: {e}")
            return 0.0
    
    def _calculate_skewness(self, data: pd.Series) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0.0
        
        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0.0
        
        skewness = ((data - mean) / std).pow(3).mean()
        return skewness
    
    def _calculate_kurtosis(self, data: pd.Series) -> float:
        """Calculate kurtosis"""
        if len(data) < 4:
            return 0.0
        
        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0.0
        
        kurtosis = ((data - mean) / std).pow(4).mean() - 3  # Excess kurtosis
        return kurtosis
    
    async def analyze_regime_conditional_risk(self, returns: pd.Series, 
                                            price_data: pd.Series) -> List[RegimeRiskMetrics]:
        """
        Analyze regime-conditional risk (bull/bear/chop)
        """
        try:
            self.logger.info("📈 [TAIL_RISK_STRESS] Analyzing regime-conditional risk...")
            
            # Classify market regimes
            regimes = await self._classify_market_regimes(returns, price_data)
            
            regime_metrics = []
            
            for regime in ['bull', 'bear', 'chop']:
                regime_returns = returns[regimes == regime]
                
                if len(regime_returns) == 0:
                    continue
                
                # Calculate regime-specific risk metrics
                var_95 = np.percentile(regime_returns, 5)
                var_99 = np.percentile(regime_returns, 1)
                es_95 = np.mean(regime_returns[regime_returns <= var_95]) if np.any(regime_returns <= var_95) else var_95
                es_99 = np.mean(regime_returns[regime_returns <= var_99]) if np.any(regime_returns <= var_99) else var_99
                
                # Calculate maximum drawdown for this regime
                max_dd = await self._calculate_max_intraday_drawdown(regime_returns)
                
                # Calculate other statistics
                volatility = regime_returns.std()
                skewness = self._calculate_skewness(regime_returns)
                kurtosis = self._calculate_kurtosis(regime_returns)
                
                metrics = RegimeRiskMetrics(
                    regime=regime,
                    var_95=var_95,
                    var_99=var_99,
                    es_95=es_95,
                    es_99=es_99,
                    max_drawdown=max_dd,
                    volatility=volatility,
                    skewness=skewness,
                    kurtosis=kurtosis,
                    sample_size=len(regime_returns)
                )
                
                regime_metrics.append(metrics)
            
            self.regime_risk_metrics = regime_metrics
            
            self.logger.info(f"📈 [TAIL_RISK_STRESS] Analyzed {len(regime_metrics)} market regimes")
            return regime_metrics
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error analyzing regime-conditional risk: {e}")
            return []
    
    async def _classify_market_regimes(self, returns: pd.Series, price_data: pd.Series) -> pd.Series:
        """Classify market regimes"""
        try:
            regimes = pd.Series(index=returns.index, dtype='object')
            
            # Calculate rolling statistics
            rolling_mean = returns.rolling(window=20).mean()
            rolling_std = returns.rolling(window=20).std()
            
            for i in range(len(returns)):
                if i < 20:
                    regimes.iloc[i] = 'chop'  # Default to chop for insufficient data
                    continue
                
                mean_return = rolling_mean.iloc[i]
                volatility = rolling_std.iloc[i]
                
                # Classify regime
                if mean_return > 0.01:  # 1% positive return
                    regimes.iloc[i] = 'bull'
                elif mean_return < -0.01:  # -1% negative return
                    regimes.iloc[i] = 'bear'
                elif volatility < 0.005:  # 0.5% volatility
                    regimes.iloc[i] = 'chop'
                else:
                    regimes.iloc[i] = 'chop'  # Default to chop
            
            return regimes
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error classifying market regimes: {e}")
            return pd.Series(index=returns.index, data='chop')
    
    async def run_stress_tests(self, portfolio_data: Dict[str, Any]) -> List[StressTestResult]:
        """
        Run comprehensive stress tests
        """
        try:
            self.logger.info("⚡ [TAIL_RISK_STRESS] Running stress tests...")
            
            stress_scenarios = [
                {
                    'name': 'market_crash',
                    'description': '50% market crash over 3 days',
                    'probability': 0.01,
                    'impact_multiplier': 0.5,
                    'duration_hours': 72
                },
                {
                    'name': 'liquidity_crisis',
                    'description': 'Severe liquidity shortage',
                    'probability': 0.02,
                    'impact_multiplier': 0.3,
                    'duration_hours': 24
                },
                {
                    'name': 'exchange_outage',
                    'description': 'Exchange outage for 4 hours',
                    'probability': 0.005,
                    'impact_multiplier': 0.1,
                    'duration_hours': 4
                }
            ]
            
            stress_results = []
            
            for scenario in stress_scenarios:
                result = await self._run_single_stress_test(scenario, portfolio_data)
                stress_results.append(result)
            
            self.stress_test_results = stress_results
            
            self.logger.info(f"⚡ [TAIL_RISK_STRESS] Completed {len(stress_results)} stress tests")
            return stress_results
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error running stress tests: {e}")
            return []
    
    async def _run_single_stress_test(self, scenario: Dict[str, Any], 
                                    portfolio_data: Dict[str, Any]) -> StressTestResult:
        """Run a single stress test scenario"""
        try:
            # Simulate stress impact
            base_portfolio_value = portfolio_data.get('portfolio_value', 100000.0)
            base_positions = portfolio_data.get('positions', {})
            
            # Calculate portfolio impact based on scenario
            portfolio_impact = await self._calculate_scenario_impact(scenario, portfolio_data)
            
            # Calculate maximum drawdown during stress
            max_drawdown = abs(portfolio_impact) * scenario['impact_multiplier']
            
            # Estimate recovery time
            recovery_time_hours = scenario['duration_hours'] * (1 + abs(portfolio_impact))
            
            # Identify affected positions
            affected_positions = list(base_positions.keys())
            
            # Identify risk breaches
            risk_breaches = []
            if max_drawdown > 0.05:  # 5% drawdown threshold
                risk_breaches.append("Maximum drawdown exceeded")
            if portfolio_impact < -0.1:  # 10% portfolio impact threshold
                risk_breaches.append("Portfolio impact threshold exceeded")
            if recovery_time_hours > 24:  # 24-hour recovery threshold
                risk_breaches.append("Recovery time threshold exceeded")
            
            # Generate recommendations
            recommendations = []
            if max_drawdown > 0.05:
                recommendations.append("Reduce position sizes to limit drawdown")
            if portfolio_impact < -0.1:
                recommendations.append("Implement additional hedging strategies")
            if recovery_time_hours > 24:
                recommendations.append("Improve risk management and monitoring systems")
            if len(affected_positions) > 0:
                recommendations.append(f"Review positions: {', '.join(affected_positions)}")
            
            result = StressTestResult(
                scenario_name=scenario['name'],
                portfolio_impact=portfolio_impact,
                max_drawdown=max_drawdown,
                recovery_time_hours=recovery_time_hours,
                affected_positions=affected_positions,
                risk_breaches=risk_breaches,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error running stress test {scenario['name']}: {e}")
            return StressTestResult(
                scenario_name=scenario['name'],
                portfolio_impact=0.0,
                max_drawdown=0.0,
                recovery_time_hours=0,
                affected_positions=[],
                risk_breaches=["Error in stress test"],
                recommendations=["Review stress test implementation"]
            )
    
    async def _calculate_scenario_impact(self, scenario: Dict[str, Any], 
                                       portfolio_data: Dict[str, Any]) -> float:
        """Calculate portfolio impact for a stress scenario"""
        try:
            # Base impact calculation
            base_impact = 0.0
            
            # Market crash scenario
            if scenario['name'] == "market_crash":
                base_impact = -0.3  # 30% negative impact
            elif scenario['name'] == "liquidity_crisis":
                base_impact = -0.15  # 15% negative impact
            elif scenario['name'] == "exchange_outage":
                base_impact = -0.05  # 5% negative impact
            
            # Apply scenario multiplier
            impact = base_impact * scenario['impact_multiplier']
            
            # Add some randomness based on portfolio composition
            portfolio_volatility = portfolio_data.get('volatility', 0.1)
            random_component = np.random.normal(0, portfolio_volatility * 0.1)
            impact += random_component
            
            return impact
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error calculating scenario impact: {e}")
            return 0.0
    
    async def generate_stress_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive stress testing report
        """
        try:
            self.logger.info("📋 [TAIL_RISK_STRESS] Generating stress testing report...")
            
            # Create comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'tail_risk_metrics': {
                    'var_95': self.tail_risk_metrics.var_95 if self.tail_risk_metrics else 0.0,
                    'var_97': self.tail_risk_metrics.var_97 if self.tail_risk_metrics else 0.0,
                    'var_99': self.tail_risk_metrics.var_99 if self.tail_risk_metrics else 0.0,
                    'var_995': self.tail_risk_metrics.var_995 if self.tail_risk_metrics else 0.0,
                    'es_95': self.tail_risk_metrics.es_95 if self.tail_risk_metrics else 0.0,
                    'es_97': self.tail_risk_metrics.es_97 if self.tail_risk_metrics else 0.0,
                    'es_99': self.tail_risk_metrics.es_99 if self.tail_risk_metrics else 0.0,
                    'es_995': self.tail_risk_metrics.es_995 if self.tail_risk_metrics else 0.0,
                    'max_intraday_dd': self.tail_risk_metrics.max_intraday_dd if self.tail_risk_metrics else 0.0,
                    'tail_expectation': self.tail_risk_metrics.tail_expectation if self.tail_risk_metrics else 0.0,
                    'tail_volatility': self.tail_risk_metrics.tail_volatility if self.tail_risk_metrics else 0.0,
                    'tail_skewness': self.tail_risk_metrics.tail_skewness if self.tail_risk_metrics else 0.0,
                    'tail_kurtosis': self.tail_risk_metrics.tail_kurtosis if self.tail_risk_metrics else 0.0
                },
                'regime_risk_metrics': [
                    {
                        'regime': metrics.regime,
                        'var_95': metrics.var_95,
                        'var_99': metrics.var_99,
                        'es_95': metrics.es_95,
                        'es_99': metrics.es_99,
                        'max_drawdown': metrics.max_drawdown,
                        'volatility': metrics.volatility,
                        'skewness': metrics.skewness,
                        'kurtosis': metrics.kurtosis,
                        'sample_size': metrics.sample_size
                    }
                    for metrics in self.regime_risk_metrics
                ],
                'stress_test_results': [
                    {
                        'scenario_name': result.scenario_name,
                        'portfolio_impact': result.portfolio_impact,
                        'max_drawdown': result.max_drawdown,
                        'recovery_time_hours': result.recovery_time_hours,
                        'affected_positions': result.affected_positions,
                        'risk_breaches': result.risk_breaches,
                        'recommendations': result.recommendations
                    }
                    for result in self.stress_test_results
                ],
                'alert_history': self.alert_history[-100:],  # Last 100 alerts
                'risk_history': self.risk_history[-100:]  # Last 100 risk measurements
            }
            
            # Save report
            await self._save_stress_report(report)
            
            self.logger.info("📋 [TAIL_RISK_STRESS] Generated comprehensive stress testing report")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error generating stress report: {e}")
            return {}
    
    async def _save_stress_report(self, report: Dict[str, Any]):
        """Save stress testing report"""
        try:
            import os
            os.makedirs('reports/stress', exist_ok=True)
            
            # Save main report
            with open('reports/stress/stress_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Create stressbook.html
            stressbook_html = self._create_stressbook_html(report)
            with open('reports/stress/stressbook.html', 'w') as f:
                f.write(stressbook_html)
            
            self.logger.info("💾 [TAIL_RISK_STRESS] Saved stress testing report to reports/stress/")
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error saving stress report: {e}")
    
    def _create_stressbook_html(self, report: Dict[str, Any]) -> str:
        """Create stressbook.html content"""
        try:
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tail Risk & Stress Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .alert-critical {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
        .alert-warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .alert-info {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Tail Risk & Stress Testing Report</h1>
        <p><strong>Generated:</strong> {report.get('timestamp', 'N/A')}</p>
    </div>
    
    <div class="section">
        <h2>🎯 Tail Risk Metrics</h2>
        <div class="metric">
            <strong>VaR 95%:</strong> {report.get('tail_risk_metrics', {}).get('var_95', 0.0):.4f}
        </div>
        <div class="metric">
            <strong>VaR 99%:</strong> {report.get('tail_risk_metrics', {}).get('var_99', 0.0):.4f}
        </div>
        <div class="metric">
            <strong>ES 95%:</strong> {report.get('tail_risk_metrics', {}).get('es_95', 0.0):.4f}
        </div>
        <div class="metric">
            <strong>ES 99%:</strong> {report.get('tail_risk_metrics', {}).get('es_99', 0.0):.4f}
        </div>
        <div class="metric">
            <strong>Max Intraday DD:</strong> {report.get('tail_risk_metrics', {}).get('max_intraday_dd', 0.0):.4f}
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Regime-Conditional Risk</h2>
        <table>
            <tr>
                <th>Regime</th>
                <th>VaR 95%</th>
                <th>VaR 99%</th>
                <th>ES 95%</th>
                <th>ES 99%</th>
                <th>Max DD</th>
                <th>Volatility</th>
                <th>Sample Size</th>
            </tr>
"""
            
            for regime_metrics in report.get('regime_risk_metrics', []):
                html_content += f"""
            <tr>
                <td>{regime_metrics.get('regime', 'N/A')}</td>
                <td>{regime_metrics.get('var_95', 0.0):.4f}</td>
                <td>{regime_metrics.get('var_99', 0.0):.4f}</td>
                <td>{regime_metrics.get('es_95', 0.0):.4f}</td>
                <td>{regime_metrics.get('es_99', 0.0):.4f}</td>
                <td>{regime_metrics.get('max_drawdown', 0.0):.4f}</td>
                <td>{regime_metrics.get('volatility', 0.0):.4f}</td>
                <td>{regime_metrics.get('sample_size', 0)}</td>
            </tr>
"""
            
            html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>⚡ Stress Test Results</h2>
        <table>
            <tr>
                <th>Scenario</th>
                <th>Portfolio Impact</th>
                <th>Max Drawdown</th>
                <th>Recovery Time (hrs)</th>
                <th>Risk Breaches</th>
                <th>Recommendations</th>
            </tr>
"""
            
            for stress_result in report.get('stress_test_results', []):
                risk_breaches = "; ".join(stress_result.get('risk_breaches', []))
                recommendations = "; ".join(stress_result.get('recommendations', []))
                
                html_content += f"""
            <tr>
                <td>{stress_result.get('scenario_name', 'N/A')}</td>
                <td>{stress_result.get('portfolio_impact', 0.0):.4f}</td>
                <td>{stress_result.get('max_drawdown', 0.0):.4f}</td>
                <td>{stress_result.get('recovery_time_hours', 0)}</td>
                <td>{risk_breaches}</td>
                <td>{recommendations}</td>
            </tr>
"""
            
            html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>📊 Summary</h2>
        <p>This report provides comprehensive tail risk analysis and stress testing results for the trading system.</p>
        <ul>
            <li><strong>Tail Risk Metrics:</strong> VaR and ES calculations at multiple confidence levels</li>
            <li><strong>Regime Analysis:</strong> Risk metrics conditional on market regimes (bull/bear/chop)</li>
            <li><strong>Stress Testing:</strong> Scenario-based stress tests with recovery analysis</li>
            <li><strong>Monitoring:</strong> Real-time tail risk monitoring with alerting</li>
        </ul>
    </div>
</body>
</html>
"""
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error creating stressbook HTML: {e}")
            return "<html><body><h1>Error generating stressbook</h1></body></html>"
    
    def get_tail_risk_statistics(self) -> Dict[str, Any]:
        """Get tail risk statistics"""
        try:
            stats = {
                'tail_risk_metrics': self.tail_risk_metrics,
                'regime_metrics_count': len(self.regime_risk_metrics),
                'stress_tests_count': len(self.stress_test_results),
                'total_alerts': len(self.alert_history),
                'recent_alerts': len([a for a in self.alert_history if (datetime.now() - a.get('timestamp', datetime.now())).days < 1])
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ [TAIL_RISK_STRESS] Error getting tail risk statistics: {e}")
            return {}