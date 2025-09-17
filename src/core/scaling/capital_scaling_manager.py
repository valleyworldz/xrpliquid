"""
Capital Scaling Manager - Tiered Scaling with Real Returns Linkage
Implements capital allocation ladder (100→1K→10K→100K) with auto-promotion rules
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
import warnings
warnings.filterwarnings('ignore')

class CapitalTier(Enum):
    SEED = "seed"           # $100 - $1,000
    GROWTH = "growth"       # $1,000 - $10,000
    SCALE = "scale"         # $10,000 - $100,000
    INSTITUTIONAL = "institutional"  # $100,000+

class PromotionStatus(Enum):
    ELIGIBLE = "eligible"
    PROMOTED = "promoted"
    DEMOTED = "demoted"
    MAINTAINED = "maintained"

@dataclass
class TierConfig:
    tier: CapitalTier
    min_capital: float
    max_capital: float
    min_sharpe_ratio: float
    min_win_rate: float
    max_drawdown: float
    min_trades: int
    min_days: int
    promotion_threshold: float  # Additional capital to promote
    demotion_threshold: float   # Loss threshold to demote

@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    average_trade_pnl: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    trading_days: int
    last_updated: str

@dataclass
class CapitalAllocation:
    current_tier: CapitalTier
    allocated_capital: float
    available_capital: float
    utilized_capital: float
    performance_metrics: PerformanceMetrics
    promotion_status: PromotionStatus
    next_review_date: str
    auto_promotion_enabled: bool

@dataclass
class ScalingDecision:
    action: str  # "promote", "demote", "maintain", "pause"
    new_tier: Optional[CapitalTier]
    new_capital: Optional[float]
    reasoning: str
    confidence: float
    effective_date: str
    review_period_days: int

class CapitalScalingManager:
    """
    Manages capital scaling with real returns linkage and auto-promotion
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tier_configs = self._initialize_tier_configs()
        self.allocations: Dict[str, CapitalAllocation] = {}
        self.scaling_history: List[ScalingDecision] = []
        
        # Create reports directory
        self.reports_dir = Path("reports/capital_scaling")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_tier_configs(self) -> Dict[CapitalTier, TierConfig]:
        """Initialize tier configurations"""
        return {
            CapitalTier.SEED: TierConfig(
                tier=CapitalTier.SEED,
                min_capital=100.0,
                max_capital=1000.0,
                min_sharpe_ratio=1.0,
                min_win_rate=0.55,
                max_drawdown=0.10,
                min_trades=50,
                min_days=30,
                promotion_threshold=0.20,  # 20% profit to promote
                demotion_threshold=0.15    # 15% loss to demote
            ),
            CapitalTier.GROWTH: TierConfig(
                tier=CapitalTier.GROWTH,
                min_capital=1000.0,
                max_capital=10000.0,
                min_sharpe_ratio=1.5,
                min_win_rate=0.60,
                max_drawdown=0.08,
                min_trades=100,
                min_days=60,
                promotion_threshold=0.25,  # 25% profit to promote
                demotion_threshold=0.12    # 12% loss to demote
            ),
            CapitalTier.SCALE: TierConfig(
                tier=CapitalTier.SCALE,
                min_capital=10000.0,
                max_capital=100000.0,
                min_sharpe_ratio=2.0,
                min_win_rate=0.65,
                max_drawdown=0.06,
                min_trades=200,
                min_days=90,
                promotion_threshold=0.30,  # 30% profit to promote
                demotion_threshold=0.10    # 10% loss to demote
            ),
            CapitalTier.INSTITUTIONAL: TierConfig(
                tier=CapitalTier.INSTITUTIONAL,
                min_capital=100000.0,
                max_capital=1000000.0,
                min_sharpe_ratio=2.5,
                min_win_rate=0.70,
                max_drawdown=0.05,
                min_trades=500,
                min_days=180,
                promotion_threshold=0.35,  # 35% profit to promote
                demotion_threshold=0.08    # 8% loss to demote
            )
        }
    
    def add_strategy(self, strategy_id: str, initial_capital: float = 100.0):
        """Add a new strategy with initial capital allocation"""
        try:
            # Determine initial tier
            initial_tier = self._determine_tier(initial_capital)
            
            # Create initial allocation
            allocation = CapitalAllocation(
                current_tier=initial_tier,
                allocated_capital=initial_capital,
                available_capital=initial_capital,
                utilized_capital=0.0,
                performance_metrics=PerformanceMetrics(
                    total_trades=0,
                    winning_trades=0,
                    total_pnl=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    average_trade_pnl=0.0,
                    volatility=0.0,
                    calmar_ratio=0.0,
                    sortino_ratio=0.0,
                    trading_days=0,
                    last_updated=datetime.now().isoformat()
                ),
                promotion_status=PromotionStatus.MAINTAINED,
                next_review_date=(datetime.now() + timedelta(days=30)).isoformat(),
                auto_promotion_enabled=True
            )
            
            self.allocations[strategy_id] = allocation
            self.logger.info(f"✅ Added strategy {strategy_id} with {initial_capital} in {initial_tier.value} tier")
            
        except Exception as e:
            self.logger.error(f"❌ Error adding strategy {strategy_id}: {e}")
    
    def _determine_tier(self, capital: float) -> CapitalTier:
        """Determine tier based on capital amount"""
        if capital < 1000:
            return CapitalTier.SEED
        elif capital < 10000:
            return CapitalTier.GROWTH
        elif capital < 100000:
            return CapitalTier.SCALE
        else:
            return CapitalTier.INSTITUTIONAL
    
    def update_performance(self, strategy_id: str, trade_data: List[Dict]):
        """Update strategy performance with real trade data"""
        try:
            if strategy_id not in self.allocations:
                self.logger.error(f"❌ Strategy {strategy_id} not found")
                return
            
            allocation = self.allocations[strategy_id]
            
            # Calculate performance metrics from trade data
            metrics = self._calculate_performance_metrics(trade_data)
            allocation.performance_metrics = metrics
            
            # Update utilized capital
            allocation.utilized_capital = sum(trade.get('size', 0) for trade in trade_data)
            
            # Check for scaling decisions
            scaling_decision = self._evaluate_scaling_decision(strategy_id)
            if scaling_decision:
                self._execute_scaling_decision(strategy_id, scaling_decision)
            
            self.logger.info(f"📊 Updated performance for {strategy_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating performance for {strategy_id}: {e}")
    
    def _calculate_performance_metrics(self, trade_data: List[Dict]) -> PerformanceMetrics:
        """Calculate performance metrics from trade data"""
        try:
            if not trade_data:
                return PerformanceMetrics(
                    total_trades=0,
                    winning_trades=0,
                    total_pnl=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    average_trade_pnl=0.0,
                    volatility=0.0,
                    calmar_ratio=0.0,
                    sortino_ratio=0.0,
                    trading_days=0,
                    last_updated=datetime.now().isoformat()
                )
            
            # Extract trade PnLs
            pnls = [trade.get('pnl', 0.0) for trade in trade_data]
            winning_trades = sum(1 for pnl in pnls if pnl > 0)
            total_trades = len(pnls)
            
            # Basic metrics
            total_pnl = sum(pnls)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            average_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
            
            # Risk metrics
            if len(pnls) > 1:
                volatility = np.std(pnls)
                
                # Sharpe ratio (assuming risk-free rate = 0)
                sharpe_ratio = np.mean(pnls) / volatility if volatility > 0 else 0.0
                
                # Max drawdown
                cumulative_pnl = np.cumsum(pnls)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdowns = (cumulative_pnl - running_max) / (running_max + 1e-8)
                max_drawdown = abs(np.min(drawdowns))
                
                # Calmar ratio
                calmar_ratio = np.mean(pnls) / max_drawdown if max_drawdown > 0 else 0.0
                
                # Sortino ratio (downside deviation)
                downside_returns = [pnl for pnl in pnls if pnl < 0]
                downside_volatility = np.std(downside_returns) if downside_returns else 0.0
                sortino_ratio = np.mean(pnls) / downside_volatility if downside_volatility > 0 else 0.0
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
                calmar_ratio = 0.0
                sortino_ratio = 0.0
            
            # Trading days (simplified)
            trading_days = len(set(trade.get('date', '')[:10] for trade in trade_data))
            
            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                total_pnl=total_pnl,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                average_trade_pnl=average_trade_pnl,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                trading_days=trading_days,
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Performance calculation error: {e}")
            return PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                total_pnl=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                average_trade_pnl=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                trading_days=0,
                last_updated=datetime.now().isoformat()
            )
    
    def _evaluate_scaling_decision(self, strategy_id: str) -> Optional[ScalingDecision]:
        """Evaluate if strategy should be promoted, demoted, or maintained"""
        try:
            allocation = self.allocations[strategy_id]
            config = self.tier_configs[allocation.current_tier]
            metrics = allocation.performance_metrics
            
            # Check if enough data for evaluation
            if (metrics.total_trades < config.min_trades or 
                metrics.trading_days < config.min_days):
                return None
            
            # Check promotion criteria
            if self._meets_promotion_criteria(allocation, config, metrics):
                new_tier = self._get_next_tier(allocation.current_tier)
                new_capital = self._calculate_promotion_capital(allocation, new_tier)
                
                return ScalingDecision(
                    action="promote",
                    new_tier=new_tier,
                    new_capital=new_capital,
                    reasoning=f"Met promotion criteria: Sharpe={metrics.sharpe_ratio:.2f}, Win Rate={metrics.win_rate:.2%}, DD={metrics.max_drawdown:.2%}",
                    confidence=0.9,
                    effective_date=(datetime.now() + timedelta(days=1)).isoformat(),
                    review_period_days=30
                )
            
            # Check demotion criteria
            elif self._meets_demotion_criteria(allocation, config, metrics):
                new_tier = self._get_previous_tier(allocation.current_tier)
                new_capital = self._calculate_demotion_capital(allocation, new_tier)
                
                return ScalingDecision(
                    action="demote",
                    new_tier=new_tier,
                    new_capital=new_capital,
                    reasoning=f"Met demotion criteria: Max DD={metrics.max_drawdown:.2%}, Win Rate={metrics.win_rate:.2%}",
                    confidence=0.8,
                    effective_date=(datetime.now() + timedelta(days=1)).isoformat(),
                    review_period_days=30
                )
            
            # Check for pause (severe underperformance)
            elif (metrics.max_drawdown > config.max_drawdown * 1.5 or 
                  metrics.win_rate < config.min_win_rate * 0.7):
                return ScalingDecision(
                    action="pause",
                    new_tier=None,
                    new_capital=None,
                    reasoning=f"Severe underperformance: Max DD={metrics.max_drawdown:.2%}, Win Rate={metrics.win_rate:.2%}",
                    confidence=0.95,
                    effective_date=datetime.now().isoformat(),
                    review_period_days=7
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Scaling decision evaluation error: {e}")
            return None
    
    def _meets_promotion_criteria(self, allocation: CapitalAllocation, config: TierConfig, metrics: PerformanceMetrics) -> bool:
        """Check if strategy meets promotion criteria"""
        try:
            # Check all promotion criteria
            criteria_met = (
                metrics.sharpe_ratio >= config.min_sharpe_ratio and
                metrics.win_rate >= config.min_win_rate and
                metrics.max_drawdown <= config.max_drawdown and
                metrics.total_pnl >= allocation.allocated_capital * config.promotion_threshold
            )
            
            return criteria_met
            
        except Exception as e:
            self.logger.error(f"❌ Promotion criteria check error: {e}")
            return False
    
    def _meets_demotion_criteria(self, allocation: CapitalAllocation, config: TierConfig, metrics: PerformanceMetrics) -> bool:
        """Check if strategy meets demotion criteria"""
        try:
            # Check demotion criteria
            criteria_met = (
                metrics.max_drawdown > config.max_drawdown * 1.2 or
                metrics.win_rate < config.min_win_rate * 0.8 or
                metrics.total_pnl <= -allocation.allocated_capital * config.demotion_threshold
            )
            
            return criteria_met
            
        except Exception as e:
            self.logger.error(f"❌ Demotion criteria check error: {e}")
            return False
    
    def _get_next_tier(self, current_tier: CapitalTier) -> Optional[CapitalTier]:
        """Get the next tier for promotion"""
        tier_order = [CapitalTier.SEED, CapitalTier.GROWTH, CapitalTier.SCALE, CapitalTier.INSTITUTIONAL]
        current_index = tier_order.index(current_tier)
        
        if current_index < len(tier_order) - 1:
            return tier_order[current_index + 1]
        return None
    
    def _get_previous_tier(self, current_tier: CapitalTier) -> Optional[CapitalTier]:
        """Get the previous tier for demotion"""
        tier_order = [CapitalTier.SEED, CapitalTier.GROWTH, CapitalTier.SCALE, CapitalTier.INSTITUTIONAL]
        current_index = tier_order.index(current_tier)
        
        if current_index > 0:
            return tier_order[current_index - 1]
        return CapitalTier.SEED  # Can't go below seed tier
    
    def _calculate_promotion_capital(self, allocation: CapitalAllocation, new_tier: CapitalTier) -> float:
        """Calculate new capital allocation for promotion"""
        try:
            config = self.tier_configs[new_tier]
            
            # Start with minimum capital for new tier
            new_capital = config.min_capital
            
            # Add performance bonus
            performance_bonus = allocation.performance_metrics.total_pnl * 0.5  # 50% of profits
            new_capital += performance_bonus
            
            # Cap at maximum for tier
            new_capital = min(new_capital, config.max_capital)
            
            return new_capital
            
        except Exception as e:
            self.logger.error(f"❌ Promotion capital calculation error: {e}")
            return allocation.allocated_capital
    
    def _calculate_demotion_capital(self, allocation: CapitalAllocation, new_tier: CapitalTier) -> float:
        """Calculate new capital allocation for demotion"""
        try:
            config = self.tier_configs[new_tier]
            
            # Reduce capital based on losses
            loss_factor = max(0.5, 1.0 + allocation.performance_metrics.total_pnl / allocation.allocated_capital)
            new_capital = allocation.allocated_capital * loss_factor
            
            # Ensure within tier limits
            new_capital = max(config.min_capital, min(new_capital, config.max_capital))
            
            return new_capital
            
        except Exception as e:
            self.logger.error(f"❌ Demotion capital calculation error: {e}")
            return allocation.allocated_capital * 0.5
    
    def _execute_scaling_decision(self, strategy_id: str, decision: ScalingDecision):
        """Execute scaling decision"""
        try:
            allocation = self.allocations[strategy_id]
            
            if decision.action == "promote":
                allocation.current_tier = decision.new_tier
                allocation.allocated_capital = decision.new_capital
                allocation.available_capital = decision.new_capital
                allocation.promotion_status = PromotionStatus.PROMOTED
                self.logger.info(f"🚀 Promoted {strategy_id} to {decision.new_tier.value} tier with ${decision.new_capital:,.2f}")
                
            elif decision.action == "demote":
                allocation.current_tier = decision.new_tier
                allocation.allocated_capital = decision.new_capital
                allocation.available_capital = decision.new_capital
                allocation.promotion_status = PromotionStatus.DEMOTED
                self.logger.info(f"📉 Demoted {strategy_id} to {decision.new_tier.value} tier with ${decision.new_capital:,.2f}")
                
            elif decision.action == "pause":
                allocation.auto_promotion_enabled = False
                allocation.promotion_status = PromotionStatus.MAINTAINED
                self.logger.warning(f"⏸️ Paused auto-promotion for {strategy_id}")
            
            # Update review date
            allocation.next_review_date = (datetime.now() + timedelta(days=decision.review_period_days)).isoformat()
            
            # Record decision
            self.scaling_history.append(decision)
            
        except Exception as e:
            self.logger.error(f"❌ Scaling decision execution error: {e}")
    
    def get_scaling_summary(self) -> Dict:
        """Get capital scaling summary"""
        try:
            total_strategies = len(self.allocations)
            tier_distribution = {}
            total_capital = 0.0
            
            for allocation in self.allocations.values():
                tier = allocation.current_tier.value
                tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
                total_capital += allocation.allocated_capital
            
            recent_decisions = [d for d in self.scaling_history if 
                              datetime.fromisoformat(d.effective_date) > datetime.now() - timedelta(days=30)]
            
            return {
                "total_strategies": total_strategies,
                "total_capital": total_capital,
                "tier_distribution": tier_distribution,
                "recent_decisions": len(recent_decisions),
                "promotions": len([d for d in recent_decisions if d.action == "promote"]),
                "demotions": len([d for d in recent_decisions if d.action == "demote"]),
                "pauses": len([d for d in recent_decisions if d.action == "pause"])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Scaling summary error: {e}")
            return {"error": str(e)}
    
    async def save_scaling_report(self):
        """Save capital scaling report"""
        try:
            # Convert allocations with enum handling
            allocations_dict = {}
            for k, v in self.allocations.items():
                allocation_dict = asdict(v)
                allocation_dict['current_tier'] = v.current_tier.value
                allocation_dict['promotion_status'] = v.promotion_status.value
                allocations_dict[k] = allocation_dict
            
            # Convert scaling history with enum handling
            scaling_history_dict = []
            for d in self.scaling_history:
                decision_dict = asdict(d)
                decision_dict['new_tier'] = d.new_tier.value if d.new_tier else None
                scaling_history_dict.append(decision_dict)
            
            # Convert tier configs with enum handling
            tier_configs_dict = {}
            for k, v in self.tier_configs.items():
                config_dict = asdict(v)
                config_dict['tier'] = v.tier.value
                tier_configs_dict[k.value] = config_dict
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "allocations": allocations_dict,
                "scaling_history": scaling_history_dict,
                "tier_configs": tier_configs_dict,
                "summary": self.get_scaling_summary()
            }
            
            report_file = self.reports_dir / f"capital_scaling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"💾 Capital scaling report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save scaling report: {e}")

# Demo function
async def demo_capital_scaling():
    """Demo the capital scaling manager"""
    print("💰 Capital Scaling Manager Demo")
    print("=" * 50)
    
    # Create scaling manager
    manager = CapitalScalingManager()
    
    # Add sample strategies
    strategies = ["momentum_1", "mean_reversion_1", "ml_classification_1"]
    for strategy in strategies:
        manager.add_strategy(strategy, initial_capital=100.0)
    
    # Simulate trading performance over time
    print("📊 Simulating trading performance...")
    
    for day in range(90):  # 90 days of trading
        for strategy_id in strategies:
            # Generate sample trade data
            trades = []
            for trade_num in range(np.random.randint(1, 5)):  # 1-4 trades per day
                pnl = np.random.normal(2.0, 5.0)  # Positive expected return
                trades.append({
                    'pnl': pnl,
                    'size': 10.0,
                    'date': (datetime.now() - timedelta(days=90-day)).isoformat()
                })
            
            # Update performance
            manager.update_performance(strategy_id, trades)
        
        # Show progress every 30 days
        if day % 30 == 0:
            print(f"Day {day}: Processing trades...")
    
    # Show final results
    print(f"\n📈 Capital Scaling Summary:")
    summary = manager.get_scaling_summary()
    print(f"Total Strategies: {summary['total_strategies']}")
    print(f"Total Capital: ${summary['total_capital']:,.2f}")
    print(f"Tier Distribution: {summary['tier_distribution']}")
    print(f"Recent Decisions: {summary['recent_decisions']}")
    print(f"Promotions: {summary['promotions']}")
    print(f"Demotions: {summary['demotions']}")
    
    # Show individual strategy details
    print(f"\n🎯 Strategy Details:")
    for strategy_id, allocation in manager.allocations.items():
        metrics = allocation.performance_metrics
        print(f"  {strategy_id}:")
        print(f"    Tier: {allocation.current_tier.value}")
        print(f"    Capital: ${allocation.allocated_capital:,.2f}")
        print(f"    Trades: {metrics.total_trades}")
        print(f"    Win Rate: {metrics.win_rate:.1%}")
        print(f"    Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"    Max DD: {metrics.max_drawdown:.1%}")
        print(f"    Total PnL: ${metrics.total_pnl:,.2f}")
    
    # Save report
    await manager.save_scaling_report()
    
    print("\n✅ Capital Scaling Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_capital_scaling())
