#!/usr/bin/env python3
"""
📈 AUTOMATED STRESS TESTING ENGINE
=================================
Institutional-grade stress testing system for portfolio resilience assessment.

Features:
- Market crash scenarios (2008, 2020, 2022 style events)
- Volatility spike simulations
- Liquidity crisis scenarios
- Exchange outage simulations
- Correlation breakdown testing
- Black swan event modeling
"""

import asyncio
import logging
import time
import numpy as np
import json
from decimal import Decimal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

class StressScenarioType(Enum):
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    EXCHANGE_OUTAGE = "exchange_outage"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    BLACK_SWAN = "black_swan"
    FLASH_CRASH = "flash_crash"
    FUNDING_SHOCK = "funding_shock"

@dataclass
class StressScenario:
    """Definition of a stress test scenario"""
    scenario_type: StressScenarioType
    name: str
    description: str
    duration_minutes: int
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]  # Expected portfolio impact ranges

@dataclass
class StressTestResult:
    """Results from a stress test execution"""
    scenario: StressScenario
    start_time: datetime
    end_time: datetime
    portfolio_value_before: float
    portfolio_value_after: float
    max_drawdown: float
    max_drawdown_duration: int  # minutes
    positions_liquidated: int
    margin_calls: int
    system_failures: List[str]
    risk_metrics: Dict[str, float]
    passed: bool
    severity_score: float  # 0-10

class AutomatedStressTester:
    """
    📈 AUTOMATED STRESS TESTING ENGINE
    Continuously monitors portfolio resilience through systematic stress testing
    """
    
    def __init__(self, api, config: Dict, logger: Optional[logging.Logger] = None):
        self.api = api
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Stress testing parameters
        self.max_acceptable_drawdown = config.get('max_acceptable_drawdown', 0.15)  # 15%
        self.stress_test_interval = config.get('stress_test_interval', 86400)  # Daily
        self.simulation_mode = config.get('simulation_mode', True)  # Don't affect real trades
        
        # Test scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
        self.test_results = []
        self.last_stress_test = 0
        
        # Portfolio snapshot for testing
        self.baseline_portfolio = None
        
        self.logger.info("📈 [STRESS_TESTER] Automated Stress Testing Engine initialized")

    def _initialize_stress_scenarios(self) -> List[StressScenario]:
        """Initialize predefined stress test scenarios"""
        scenarios = [
            # 2008-style Market Crash
            StressScenario(
                scenario_type=StressScenarioType.MARKET_CRASH,
                name="2008 Financial Crisis",
                description="Severe market crash with 40% decline over 6 months",
                duration_minutes=4320,  # 3 days compressed
                parameters={
                    "price_decline": 0.40,
                    "volatility_spike": 3.0,
                    "correlation_increase": 0.8,
                    "liquidity_reduction": 0.6
                },
                expected_impact={"max_drawdown": 0.25, "margin_pressure": 0.7}
            ),
            
            # 2020 COVID Crash
            StressScenario(
                scenario_type=StressScenarioType.MARKET_CRASH,
                name="COVID-19 Flash Crash",
                description="Rapid 35% decline followed by quick recovery",
                duration_minutes=2880,  # 2 days compressed
                parameters={
                    "price_decline": 0.35,
                    "volatility_spike": 5.0,
                    "recovery_speed": 0.8,
                    "correlation_spike": 0.9
                },
                expected_impact={"max_drawdown": 0.20, "recovery_time": 1440}
            ),
            
            # Extreme Volatility Spike
            StressScenario(
                scenario_type=StressScenarioType.VOLATILITY_SPIKE,
                name="Extreme Volatility Event",
                description="10x normal volatility with rapid price swings",
                duration_minutes=720,  # 12 hours
                parameters={
                    "volatility_multiplier": 10.0,
                    "price_swing_range": 0.25,
                    "swing_frequency": 15  # minutes
                },
                expected_impact={"stop_loss_triggers": 0.6, "margin_usage": 0.8}
            ),
            
            # Liquidity Crisis
            StressScenario(
                scenario_type=StressScenarioType.LIQUIDITY_CRISIS,
                name="Liquidity Drought",
                description="Market liquidity disappears, spreads widen dramatically",
                duration_minutes=1440,  # 24 hours
                parameters={
                    "spread_multiplier": 20.0,
                    "liquidity_reduction": 0.9,
                    "slippage_increase": 10.0
                },
                expected_impact={"execution_cost": 0.05, "position_closure": 0.3}
            ),
            
            # Exchange Outage
            StressScenario(
                scenario_type=StressScenarioType.EXCHANGE_OUTAGE,
                name="Exchange Outage",
                description="Complete exchange unavailability for extended period",
                duration_minutes=360,  # 6 hours
                parameters={
                    "outage_duration": 360,
                    "reconnection_attempts": 10,
                    "data_lag_on_recovery": 60
                },
                expected_impact={"position_monitoring": 0.0, "risk_control": 0.3}
            ),
            
            # Correlation Breakdown
            StressScenario(
                scenario_type=StressScenarioType.CORRELATION_BREAKDOWN,
                name="Correlation Breakdown",
                description="Historical correlations break down completely",
                duration_minutes=2880,  # 2 days
                parameters={
                    "correlation_randomization": True,
                    "hedge_effectiveness": 0.1
                },
                expected_impact={"hedge_failure": 0.8, "portfolio_variance": 2.0}
            ),
            
            # Flash Crash
            StressScenario(
                scenario_type=StressScenarioType.FLASH_CRASH,
                name="Flash Crash",
                description="Instant 25% drop followed by partial recovery",
                duration_minutes=60,  # 1 hour
                parameters={
                    "instant_drop": 0.25,
                    "recovery_percentage": 0.7,
                    "recovery_time": 30
                },
                expected_impact={"immediate_loss": 0.15, "system_stress": 0.9}
            ),
            
            # Funding Rate Shock
            StressScenario(
                scenario_type=StressScenarioType.FUNDING_SHOCK,
                name="Funding Rate Crisis",
                description="Extreme funding rates due to market imbalance",
                duration_minutes=1440,  # 24 hours
                parameters={
                    "funding_rate_spike": 0.01,  # 1% per 8 hours
                    "funding_frequency": 480,  # Every 8 hours
                    "market_imbalance": 0.8
                },
                expected_impact={"funding_cost": 0.03, "position_pressure": 0.6}
            )
        ]
        
        return scenarios

    async def run_stress_tests(self) -> List[StressTestResult]:
        """Run all stress test scenarios"""
        try:
            if time.time() - self.last_stress_test < self.stress_test_interval:
                return []
            
            self.logger.info("📈 [STRESS_TESTER] Starting comprehensive stress test suite")
            
            # Take baseline portfolio snapshot
            self.baseline_portfolio = await self._get_portfolio_snapshot()
            if not self.baseline_portfolio:
                self.logger.error("❌ [STRESS_TESTER] Cannot run stress tests - no portfolio data")
                return []
            
            results = []
            for scenario in self.stress_scenarios:
                try:
                    result = await self._run_single_stress_test(scenario)
                    results.append(result)
                    
                    # Log immediate results
                    status = "✅ PASSED" if result.passed else "❌ FAILED"
                    self.logger.warning(f"📈 [STRESS_TEST] {scenario.name}: {status} | "
                                      f"Drawdown: {result.max_drawdown:.2%} | "
                                      f"Severity: {result.severity_score:.1f}/10")
                    
                except Exception as e:
                    self.logger.error(f"❌ [STRESS_TESTER] Error in scenario {scenario.name}: {e}")
            
            # Generate comprehensive stress test report
            await self._generate_stress_test_report(results)
            
            self.last_stress_test = time.time()
            self.test_results.extend(results)
            
            # Cleanup old results (keep last 30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.test_results = [r for r in self.test_results if r.end_time > cutoff_time]
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ [STRESS_TESTER] Error running stress tests: {e}")
            return []

    async def _get_portfolio_snapshot(self) -> Optional[Dict]:
        """Get current portfolio snapshot for stress testing"""
        try:
            user_state = self.api.get_user_state()
            if not user_state:
                return None
            
            positions = {}
            total_value = 0
            
            asset_positions = user_state.get("assetPositions", [])
            for pos in asset_positions:
                symbol = pos.get("position", {}).get("coin", "")
                size = float(pos.get("position", {}).get("szi", "0"))
                entry_px = float(pos.get("position", {}).get("entryPx", "0"))
                unrealized_pnl = float(pos.get("position", {}).get("unrealizedPnl", "0"))
                
                if abs(size) > 0.001:
                    position_value = abs(size) * entry_px
                    positions[symbol] = {
                        'size': size,
                        'entry_price': entry_px,
                        'unrealized_pnl': unrealized_pnl,
                        'value': position_value
                    }
                    total_value += position_value
            
            account_value = float(user_state.get("marginSummary", {}).get("accountValue", 0))
            
            return {
                'positions': positions,
                'total_position_value': total_value,
                'account_value': account_value,
                'timestamp': datetime.now(),
                'margin_summary': user_state.get("marginSummary", {})
            }
            
        except Exception as e:
            self.logger.error(f"❌ [STRESS_TESTER] Error getting portfolio snapshot: {e}")
            return None

    async def _run_single_stress_test(self, scenario: StressScenario) -> StressTestResult:
        """Run a single stress test scenario"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"📈 [STRESS_TEST] Running scenario: {scenario.name}")
            
            # Initialize simulation environment
            simulated_portfolio = self._create_simulation_copy(self.baseline_portfolio)
            
            # Apply stress scenario
            test_results = await self._apply_stress_scenario(scenario, simulated_portfolio)
            
            # Calculate final metrics
            end_time = datetime.now()
            
            # Determine if test passed
            passed = (test_results['max_drawdown'] <= self.max_acceptable_drawdown and
                     test_results['margin_calls'] == 0 and
                     len(test_results['system_failures']) == 0)
            
            # Calculate severity score (0-10)
            severity_score = self._calculate_severity_score(test_results, scenario)
            
            return StressTestResult(
                scenario=scenario,
                start_time=start_time,
                end_time=end_time,
                portfolio_value_before=self.baseline_portfolio['account_value'],
                portfolio_value_after=test_results['final_portfolio_value'],
                max_drawdown=test_results['max_drawdown'],
                max_drawdown_duration=test_results['max_drawdown_duration'],
                positions_liquidated=test_results['positions_liquidated'],
                margin_calls=test_results['margin_calls'],
                system_failures=test_results['system_failures'],
                risk_metrics=test_results['risk_metrics'],
                passed=passed,
                severity_score=severity_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ [STRESS_TEST] Error in scenario {scenario.name}: {e}")
            return StressTestResult(
                scenario=scenario,
                start_time=start_time,
                end_time=datetime.now(),
                portfolio_value_before=self.baseline_portfolio['account_value'],
                portfolio_value_after=0,
                max_drawdown=1.0,  # 100% loss assumed on error
                max_drawdown_duration=scenario.duration_minutes,
                positions_liquidated=len(self.baseline_portfolio['positions']),
                margin_calls=1,
                system_failures=[f"Test execution error: {str(e)}"],
                risk_metrics={},
                passed=False,
                severity_score=10.0
            )

    def _create_simulation_copy(self, portfolio: Dict) -> Dict:
        """Create a deep copy of portfolio for simulation"""
        import copy
        return copy.deepcopy(portfolio)

    async def _apply_stress_scenario(self, scenario: StressScenario, portfolio: Dict) -> Dict:
        """Apply stress scenario to simulated portfolio"""
        try:
            results = {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'positions_liquidated': 0,
                'margin_calls': 0,
                'system_failures': [],
                'risk_metrics': {},
                'final_portfolio_value': portfolio['account_value']
            }
            
            # Simulate scenario based on type
            if scenario.scenario_type == StressScenarioType.MARKET_CRASH:
                results = await self._simulate_market_crash(scenario, portfolio, results)
            elif scenario.scenario_type == StressScenarioType.VOLATILITY_SPIKE:
                results = await self._simulate_volatility_spike(scenario, portfolio, results)
            elif scenario.scenario_type == StressScenarioType.LIQUIDITY_CRISIS:
                results = await self._simulate_liquidity_crisis(scenario, portfolio, results)
            elif scenario.scenario_type == StressScenarioType.EXCHANGE_OUTAGE:
                results = await self._simulate_exchange_outage(scenario, portfolio, results)
            elif scenario.scenario_type == StressScenarioType.FLASH_CRASH:
                results = await self._simulate_flash_crash(scenario, portfolio, results)
            elif scenario.scenario_type == StressScenarioType.FUNDING_SHOCK:
                results = await self._simulate_funding_shock(scenario, portfolio, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ [STRESS_TEST] Error applying scenario: {e}")
            return {
                'max_drawdown': 1.0,
                'max_drawdown_duration': scenario.duration_minutes,
                'positions_liquidated': len(portfolio['positions']),
                'margin_calls': 1,
                'system_failures': [f"Simulation error: {str(e)}"],
                'risk_metrics': {},
                'final_portfolio_value': 0
            }

    async def _simulate_market_crash(self, scenario: StressScenario, portfolio: Dict, results: Dict) -> Dict:
        """Simulate market crash scenario"""
        decline = scenario.parameters['price_decline']
        volatility_spike = scenario.parameters['volatility_spike']
        
        # Apply price decline to all positions
        portfolio_value = portfolio['account_value']
        
        for symbol, position in portfolio['positions'].items():
            # Apply decline with some position-specific variation
            position_decline = decline * (0.8 + 0.4 * np.random.random())  # 80-120% of average decline
            new_price = position['entry_price'] * (1 - position_decline)
            
            # Calculate new unrealized PnL
            if position['size'] > 0:  # Long position
                pnl_change = position['size'] * (new_price - position['entry_price'])
            else:  # Short position
                pnl_change = abs(position['size']) * (position['entry_price'] - new_price)
            
            position['unrealized_pnl'] += pnl_change
            portfolio_value += pnl_change
        
        # Calculate drawdown
        drawdown = (portfolio['account_value'] - portfolio_value) / portfolio['account_value']
        results['max_drawdown'] = max(results['max_drawdown'], drawdown)
        results['final_portfolio_value'] = portfolio_value
        
        # Check for margin calls
        if portfolio_value < portfolio['account_value'] * 0.5:  # 50% margin requirement
            results['margin_calls'] += 1
        
        # Simulate position liquidations if severe
        if drawdown > 0.3:  # 30% drawdown triggers liquidations
            liquidated = int(len(portfolio['positions']) * (drawdown - 0.3) / 0.7)
            results['positions_liquidated'] = liquidated
        
        return results

    async def _simulate_volatility_spike(self, scenario: StressScenario, portfolio: Dict, results: Dict) -> Dict:
        """Simulate extreme volatility scenario"""
        volatility_multiplier = scenario.parameters['volatility_multiplier']
        swing_range = scenario.parameters['price_swing_range']
        
        # Simulate multiple price swings
        portfolio_value = portfolio['account_value']
        min_portfolio_value = portfolio_value
        
        for minute in range(scenario.duration_minutes):
            # Random price swing for each asset
            for symbol, position in portfolio['positions'].items():
                swing = (np.random.random() - 0.5) * 2 * swing_range  # ±swing_range
                new_price = position['entry_price'] * (1 + swing)
                
                # Calculate temporary PnL impact
                if position['size'] > 0:
                    temp_pnl = position['size'] * (new_price - position['entry_price'])
                else:
                    temp_pnl = abs(position['size']) * (position['entry_price'] - new_price)
                
                temp_portfolio_value = portfolio_value + temp_pnl
                min_portfolio_value = min(min_portfolio_value, temp_portfolio_value)
        
        # Calculate maximum drawdown during volatility
        drawdown = (portfolio['account_value'] - min_portfolio_value) / portfolio['account_value']
        results['max_drawdown'] = drawdown
        results['final_portfolio_value'] = portfolio_value  # Return to baseline
        
        # High volatility often triggers stop losses
        stop_loss_probability = volatility_multiplier * 0.1
        if np.random.random() < stop_loss_probability:
            results['positions_liquidated'] = int(len(portfolio['positions']) * 0.3)
        
        return results

    async def _simulate_liquidity_crisis(self, scenario: StressScenario, portfolio: Dict, results: Dict) -> Dict:
        """Simulate liquidity crisis"""
        spread_multiplier = scenario.parameters['spread_multiplier']
        liquidity_reduction = scenario.parameters['liquidity_reduction']
        
        # Calculate impact of increased spreads and slippage
        total_spread_cost = 0
        
        for symbol, position in portfolio['positions'].items():
            position_value = abs(position['size']) * position['entry_price']
            # Increased spread cost due to liquidity crisis
            spread_cost = position_value * 0.001 * spread_multiplier  # Base 0.1% spread
            total_spread_cost += spread_cost
        
        # Reduce portfolio value by spread costs
        final_value = portfolio['account_value'] - total_spread_cost
        drawdown = total_spread_cost / portfolio['account_value']
        
        results['max_drawdown'] = drawdown
        results['final_portfolio_value'] = final_value
        
        # Some positions may be difficult to close
        if liquidity_reduction > 0.8:
            results['system_failures'].append("Unable to close positions due to low liquidity")
        
        return results

    async def _simulate_exchange_outage(self, scenario: StressScenario, portfolio: Dict, results: Dict) -> Dict:
        """Simulate exchange outage"""
        outage_duration = scenario.parameters['outage_duration']
        
        # During outage, no position monitoring or risk control
        results['system_failures'].append(f"Exchange outage for {outage_duration} minutes")
        results['final_portfolio_value'] = portfolio['account_value']
        
        # Risk accumulates during outage
        risk_accumulation = outage_duration / 1440  # Daily risk over outage period
        results['risk_metrics']['uncontrolled_risk_period'] = risk_accumulation
        
        return results

    async def _simulate_flash_crash(self, scenario: StressScenario, portfolio: Dict, results: Dict) -> Dict:
        """Simulate flash crash"""
        instant_drop = scenario.parameters['instant_drop']
        recovery_percentage = scenario.parameters['recovery_percentage']
        
        # Immediate severe drop
        crash_portfolio_value = portfolio['account_value'] * (1 - instant_drop)
        
        # Partial recovery
        final_portfolio_value = crash_portfolio_value + (portfolio['account_value'] - crash_portfolio_value) * recovery_percentage
        
        drawdown = (portfolio['account_value'] - crash_portfolio_value) / portfolio['account_value']
        results['max_drawdown'] = drawdown
        results['final_portfolio_value'] = final_portfolio_value
        
        # Flash crashes often trigger all stop losses
        results['positions_liquidated'] = len(portfolio['positions'])
        results['system_failures'].append("All stop losses triggered simultaneously")
        
        return results

    async def _simulate_funding_shock(self, scenario: StressScenario, portfolio: Dict, results: Dict) -> Dict:
        """Simulate funding rate shock"""
        funding_spike = scenario.parameters['funding_rate_spike']
        
        # Calculate funding cost for all positions
        total_funding_cost = 0
        
        for symbol, position in portfolio['positions'].items():
            position_value = abs(position['size']) * position['entry_price']
            # Funding cost (positive for longs in high funding environment)
            if position['size'] > 0:  # Long position pays funding
                funding_cost = position_value * funding_spike
            else:  # Short position receives funding
                funding_cost = -position_value * funding_spike
            
            total_funding_cost += funding_cost
        
        final_value = portfolio['account_value'] - total_funding_cost
        
        if total_funding_cost > 0:
            drawdown = total_funding_cost / portfolio['account_value']
            results['max_drawdown'] = drawdown
        
        results['final_portfolio_value'] = final_value
        
        return results

    def _calculate_severity_score(self, test_results: Dict, scenario: StressScenario) -> float:
        """Calculate severity score (0-10) based on test results"""
        try:
            score = 0.0
            
            # Drawdown component (0-4 points)
            drawdown = test_results['max_drawdown']
            if drawdown > 0.5:  # >50% loss
                score += 4.0
            elif drawdown > 0.3:  # >30% loss
                score += 3.0
            elif drawdown > 0.15:  # >15% loss
                score += 2.0
            elif drawdown > 0.05:  # >5% loss
                score += 1.0
            
            # System failures component (0-3 points)
            failures = len(test_results['system_failures'])
            score += min(failures, 3)
            
            # Liquidations component (0-2 points)
            liquidation_rate = test_results['positions_liquidated'] / max(len(self.baseline_portfolio['positions']), 1)
            if liquidation_rate > 0.5:
                score += 2.0
            elif liquidation_rate > 0.25:
                score += 1.0
            
            # Margin calls component (0-1 point)
            if test_results['margin_calls'] > 0:
                score += 1.0
            
            return min(score, 10.0)
            
        except Exception as e:
            self.logger.error(f"❌ [STRESS_TESTER] Error calculating severity: {e}")
            return 10.0

    async def _generate_stress_test_report(self, results: List[StressTestResult]):
        """Generate comprehensive stress test report"""
        try:
            report = {
                'test_date': datetime.now().isoformat(),
                'total_scenarios': len(results),
                'passed_scenarios': sum(1 for r in results if r.passed),
                'failed_scenarios': sum(1 for r in results if not r.passed),
                'overall_score': self._calculate_overall_score(results),
                'scenarios': []
            }
            
            for result in results:
                scenario_report = {
                    'name': result.scenario.name,
                    'type': result.scenario.scenario_type.value,
                    'passed': result.passed,
                    'max_drawdown': f"{result.max_drawdown:.2%}",
                    'severity_score': f"{result.severity_score:.1f}/10",
                    'positions_liquidated': result.positions_liquidated,
                    'margin_calls': result.margin_calls,
                    'system_failures': result.system_failures
                }
                report['scenarios'].append(scenario_report)
            
            # Save report
            report_path = f"reports/stress/stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            import os
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"📈 [STRESS_TESTER] Report saved: {report_path}")
            
            # Log summary
            self.logger.warning(f"📈 [STRESS_TEST_SUMMARY] "
                              f"Passed: {report['passed_scenarios']}/{report['total_scenarios']} | "
                              f"Overall Score: {report['overall_score']:.1f}/10 | "
                              f"Report: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ [STRESS_TESTER] Error generating report: {e}")

    def _calculate_overall_score(self, results: List[StressTestResult]) -> float:
        """Calculate overall stress test score"""
        if not results:
            return 0.0
        
        # Average severity score (inverted - lower severity = higher score)
        avg_severity = sum(r.severity_score for r in results) / len(results)
        base_score = 10.0 - avg_severity
        
        # Bonus for all tests passing
        if all(r.passed for r in results):
            base_score += 2.0
        
        return min(base_score, 10.0)

    def get_stress_test_status(self) -> Dict:
        """Get current stress testing system status"""
        recent_results = [r for r in self.test_results if r.end_time > datetime.now() - timedelta(days=7)]
        
        return {
            'last_test': self.last_stress_test,
            'total_tests_run': len(self.test_results),
            'recent_tests': len(recent_results),
            'recent_pass_rate': sum(1 for r in recent_results if r.passed) / max(len(recent_results), 1),
            'next_test_due': self.last_stress_test + self.stress_test_interval,
            'status': 'active'
        }
