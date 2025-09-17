"""
Capital Scaling Plan
Defines safe bankroll tiers for compounding growth strategy.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalingTier(Enum):
    """Capital scaling tiers."""
    SEED = "seed"           # $20 - $100
    GROWTH = "growth"       # $100 - $500
    SCALE = "scale"         # $500 - $1,000
    PROFESSIONAL = "professional"  # $1,000 - $10,000
    INSTITUTIONAL = "institutional"  # $10,000+


@dataclass
class TierConfiguration:
    """Configuration for a capital scaling tier."""
    tier: ScalingTier
    min_capital: float
    max_capital: float
    
    # Risk parameters
    max_daily_risk: float  # % of capital at risk per day
    max_position_size: float  # % of capital per position
    max_total_exposure: float  # % of capital total exposure
    
    # Position sizing
    buy_cap_multiplier: float
    scalp_cap_multiplier: float
    funding_arb_cap_multiplier: float
    
    # Performance requirements
    min_win_rate: float  # Minimum win rate to advance
    min_sharpe_ratio: float  # Minimum Sharpe ratio to advance
    min_trades: int  # Minimum trades before advancement
    min_days: int  # Minimum days in tier before advancement
    
    # Drawdown limits
    max_daily_drawdown: float
    max_rolling_drawdown: float
    kill_switch_threshold: float
    
    # Advancement criteria
    advancement_threshold: float  # Performance score needed to advance
    demotion_threshold: float  # Performance score that triggers demotion


class CapitalScalingPlan:
    """Capital scaling plan with safe bankroll tiers for compounding growth."""
    
    def __init__(self, config_path: str = "config/capital_scaling.json",
                 reports_dir: str = "reports"):
        self.config_path = Path(config_path)
        self.reports_dir = Path(reports_dir)
        
        # Load or initialize configuration
        self.tier_configs = self._load_tier_configurations()
        
        # Current state
        self.current_tier = ScalingTier.SEED
        self.current_capital = 20.0  # Starting capital
        self.tier_start_date = datetime.now()
        self.tier_performance_history: List[Dict[str, Any]] = []
        
        logger.info(f"💰 Capital Scaling Plan initialized - Tier: {self.current_tier.value}, Capital: ${self.current_capital}")
    
    def _load_tier_configurations(self) -> Dict[ScalingTier, TierConfiguration]:
        """Load tier configurations."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                tier_configs = {}
                for tier_name, config in config_data.get('tiers', {}).items():
                    tier = ScalingTier(tier_name)
                    tier_configs[tier] = TierConfiguration(
                        tier=tier,
                        min_capital=config['min_capital'],
                        max_capital=config['max_capital'],
                        max_daily_risk=config['max_daily_risk'],
                        max_position_size=config['max_position_size'],
                        max_total_exposure=config['max_total_exposure'],
                        buy_cap_multiplier=config['buy_cap_multiplier'],
                        scalp_cap_multiplier=config['scalp_cap_multiplier'],
                        funding_arb_cap_multiplier=config['funding_arb_cap_multiplier'],
                        min_win_rate=config['min_win_rate'],
                        min_sharpe_ratio=config['min_sharpe_ratio'],
                        min_trades=config['min_trades'],
                        min_days=config['min_days'],
                        max_daily_drawdown=config['max_daily_drawdown'],
                        max_rolling_drawdown=config['max_rolling_drawdown'],
                        kill_switch_threshold=config['kill_switch_threshold'],
                        advancement_threshold=config['advancement_threshold'],
                        demotion_threshold=config['demotion_threshold']
                    )
                
                return tier_configs
                
            except Exception as e:
                logger.warning(f"Could not load tier configurations: {e}")
        
        # Default tier configurations
        return self._create_default_tier_configurations()
    
    def _create_default_tier_configurations(self) -> Dict[ScalingTier, TierConfiguration]:
        """Create default tier configurations."""
        tier_configs = {}
        
        # Seed Tier ($20 - $100)
        tier_configs[ScalingTier.SEED] = TierConfiguration(
            tier=ScalingTier.SEED,
            min_capital=20.0,
            max_capital=100.0,
            max_daily_risk=0.02,  # 2% daily risk
            max_position_size=0.10,  # 10% per position
            max_total_exposure=0.20,  # 20% total exposure
            buy_cap_multiplier=0.5,  # Conservative sizing
            scalp_cap_multiplier=0.3,
            funding_arb_cap_multiplier=0.4,
            min_win_rate=35.0,  # 35% win rate
            min_sharpe_ratio=1.0,  # 1.0 Sharpe ratio
            min_trades=50,  # 50 trades minimum
            min_days=30,  # 30 days minimum
            max_daily_drawdown=0.03,  # 3% max daily DD
            max_rolling_drawdown=0.08,  # 8% max rolling DD
            kill_switch_threshold=0.10,  # 10% kill switch
            advancement_threshold=75.0,  # 75% performance score
            demotion_threshold=25.0  # 25% performance score
        )
        
        # Growth Tier ($100 - $500)
        tier_configs[ScalingTier.GROWTH] = TierConfiguration(
            tier=ScalingTier.GROWTH,
            min_capital=100.0,
            max_capital=500.0,
            max_daily_risk=0.025,  # 2.5% daily risk
            max_position_size=0.15,  # 15% per position
            max_total_exposure=0.30,  # 30% total exposure
            buy_cap_multiplier=0.7,
            scalp_cap_multiplier=0.5,
            funding_arb_cap_multiplier=0.6,
            min_win_rate=40.0,  # 40% win rate
            min_sharpe_ratio=1.2,  # 1.2 Sharpe ratio
            min_trades=100,  # 100 trades minimum
            min_days=45,  # 45 days minimum
            max_daily_drawdown=0.04,  # 4% max daily DD
            max_rolling_drawdown=0.10,  # 10% max rolling DD
            kill_switch_threshold=0.12,  # 12% kill switch
            advancement_threshold=80.0,  # 80% performance score
            demotion_threshold=30.0  # 30% performance score
        )
        
        # Scale Tier ($500 - $1,000)
        tier_configs[ScalingTier.SCALE] = TierConfiguration(
            tier=ScalingTier.SCALE,
            min_capital=500.0,
            max_capital=1000.0,
            max_daily_risk=0.03,  # 3% daily risk
            max_position_size=0.20,  # 20% per position
            max_total_exposure=0.40,  # 40% total exposure
            buy_cap_multiplier=1.0,  # Full sizing
            scalp_cap_multiplier=0.8,
            funding_arb_cap_multiplier=0.9,
            min_win_rate=45.0,  # 45% win rate
            min_sharpe_ratio=1.5,  # 1.5 Sharpe ratio
            min_trades=200,  # 200 trades minimum
            min_days=60,  # 60 days minimum
            max_daily_drawdown=0.05,  # 5% max daily DD
            max_rolling_drawdown=0.12,  # 12% max rolling DD
            kill_switch_threshold=0.15,  # 15% kill switch
            advancement_threshold=85.0,  # 85% performance score
            demotion_threshold=35.0  # 35% performance score
        )
        
        # Professional Tier ($1,000 - $10,000)
        tier_configs[ScalingTier.PROFESSIONAL] = TierConfiguration(
            tier=ScalingTier.PROFESSIONAL,
            min_capital=1000.0,
            max_capital=10000.0,
            max_daily_risk=0.035,  # 3.5% daily risk
            max_position_size=0.25,  # 25% per position
            max_total_exposure=0.50,  # 50% total exposure
            buy_cap_multiplier=1.2,  # Aggressive sizing
            scalp_cap_multiplier=1.0,
            funding_arb_cap_multiplier=1.1,
            min_win_rate=50.0,  # 50% win rate
            min_sharpe_ratio=1.8,  # 1.8 Sharpe ratio
            min_trades=500,  # 500 trades minimum
            min_days=90,  # 90 days minimum
            max_daily_drawdown=0.06,  # 6% max daily DD
            max_rolling_drawdown=0.15,  # 15% max rolling DD
            kill_switch_threshold=0.18,  # 18% kill switch
            advancement_threshold=90.0,  # 90% performance score
            demotion_threshold=40.0  # 40% performance score
        )
        
        # Institutional Tier ($10,000+)
        tier_configs[ScalingTier.INSTITUTIONAL] = TierConfiguration(
            tier=ScalingTier.INSTITUTIONAL,
            min_capital=10000.0,
            max_capital=safe_float('inf'),
            max_daily_risk=0.04,  # 4% daily risk
            max_position_size=0.30,  # 30% per position
            max_total_exposure=0.60,  # 60% total exposure
            buy_cap_multiplier=1.5,  # Maximum sizing
            scalp_cap_multiplier=1.2,
            funding_arb_cap_multiplier=1.3,
            min_win_rate=55.0,  # 55% win rate
            min_sharpe_ratio=2.0,  # 2.0 Sharpe ratio
            min_trades=1000,  # 1000 trades minimum
            min_days=120,  # 120 days minimum
            max_daily_drawdown=0.07,  # 7% max daily DD
            max_rolling_drawdown=0.18,  # 18% max rolling DD
            kill_switch_threshold=0.20,  # 20% kill switch
            advancement_threshold=95.0,  # 95% performance score
            demotion_threshold=45.0  # 45% performance score
        )
        
        # Save default configurations
        self._save_tier_configurations(tier_configs)
        
        return tier_configs
    
    def _save_tier_configurations(self, tier_configs: Dict[ScalingTier, TierConfiguration]):
        """Save tier configurations to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'tiers': {}
            }
            
            for tier, config in tier_configs.items():
                config_data['tiers'][tier.value] = asdict(config)
                # Remove the tier field to avoid duplication
                del config_data['tiers'][tier.value]['tier']
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"💾 Tier configurations saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Could not save tier configurations: {e}")
    
    def _load_performance_data(self, days: int = 30) -> pd.DataFrame:
        """Load recent performance data."""
        try:
            # Load trade ledger
            ledger_path = self.reports_dir / "ledgers" / "trades.parquet"
            if ledger_path.exists():
                trades_df = pd.read_parquet(ledger_path)
            else:
                ledger_path = self.reports_dir / "ledgers" / "trades.csv"
                if ledger_path.exists():
                    trades_df = pd.read_csv(ledger_path)
                else:
                    return pd.DataFrame()
            
            # Filter to recent data
            cutoff_time = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            recent_trades = trades_df[trades_df['ts'] >= cutoff_timestamp]
            return recent_trades
            
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")
            return pd.DataFrame()
    
    def _calculate_tier_performance(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for tier evaluation."""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_pnl': 0.0,
                'days_active': 0,
                'performance_score': 0.0
            }
        
        returns = trades_df['pnl_realized']
        
        # Basic metrics
        total_trades = len(trades_df)
        win_rate = (returns > 0).mean() * 100
        total_pnl = returns.sum()
        
        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * (252**0.5)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = returns.cumsum()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0
        
        # Days active
        if not trades_df.empty:
            first_trade = datetime.fromtimestamp(trades_df['ts'].min())
            last_trade = datetime.fromtimestamp(trades_df['ts'].max())
            days_active = (last_trade - first_trade).days + 1
        else:
            days_active = 0
        
        # Performance score (0-100)
        performance_score = self._calculate_performance_score(
            total_trades, win_rate, sharpe_ratio, max_drawdown, days_active
        )
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_pnl': total_pnl,
            'days_active': days_active,
            'performance_score': performance_score
        }
    
    def _calculate_performance_score(self, total_trades: int, win_rate: float, 
                                   sharpe_ratio: float, max_drawdown: float, 
                                   days_active: int) -> float:
        """Calculate performance score for tier advancement."""
        if total_trades < 10:
            return 0.0
        
        score = 0.0
        
        # Win rate component (30% weight)
        win_rate_score = min(win_rate / 60.0, 1.0) * 30
        score += win_rate_score
        
        # Sharpe ratio component (25% weight)
        sharpe_score = min(sharpe_ratio / 2.5, 1.0) * 25
        score += sharpe_score
        
        # Drawdown penalty (20% weight)
        dd_penalty = min(max_drawdown / 0.15, 1.0) * 20
        score -= dd_penalty
        
        # Trade frequency component (15% weight)
        trade_frequency = min(total_trades / 200.0, 1.0) * 15
        score += trade_frequency
        
        # Consistency component (10% weight)
        consistency_score = min(days_active / 90.0, 1.0) * 10
        score += consistency_score
        
        return max(0.0, min(100.0, score))
    
    def evaluate_tier_advancement(self) -> Tuple[bool, bool, str]:
        """Evaluate if tier advancement or demotion is warranted.
        
        Returns:
            (should_advance, should_demote, reason)
        """
        current_config = self.tier_configs[self.current_tier]
        
        # Load recent performance data
        trades_df = self._load_performance_data(current_config.min_days)
        
        if trades_df.empty:
            return False, False, "Insufficient trade data"
        
        # Calculate performance metrics
        performance = self._calculate_tier_performance(trades_df)
        
        # Check minimum requirements
        if performance['total_trades'] < current_config.min_trades:
            return False, False, f"Insufficient trades: {performance['total_trades']}/{current_config.min_trades}"
        
        if performance['days_active'] < current_config.min_days:
            return False, False, f"Insufficient days: {performance['days_active']}/{current_config.min_days}"
        
        if performance['win_rate'] < current_config.min_win_rate:
            return False, True, f"Win rate too low: {performance['win_rate']:.1f}%/{current_config.min_win_rate}%"
        
        if performance['sharpe_ratio'] < current_config.min_sharpe_ratio:
            return False, True, f"Sharpe ratio too low: {performance['sharpe_ratio']:.2f}/{current_config.min_sharpe_ratio}"
        
        # Check advancement criteria
        if performance['performance_score'] >= current_config.advancement_threshold:
            return True, False, f"Performance score: {performance['performance_score']:.1f}/{current_config.advancement_threshold}"
        
        # Check demotion criteria
        if performance['performance_score'] <= current_config.demotion_threshold:
            return False, True, f"Performance score too low: {performance['performance_score']:.1f}/{current_config.demotion_threshold}"
        
        return False, False, f"Performance adequate: {performance['performance_score']:.1f}"
    
    def advance_tier(self) -> bool:
        """Advance to the next tier."""
        current_tier_order = [ScalingTier.SEED, ScalingTier.GROWTH, ScalingTier.SCALE, 
                             ScalingTier.PROFESSIONAL, ScalingTier.INSTITUTIONAL]
        
        current_index = current_tier_order.index(self.current_tier)
        
        if current_index >= len(current_tier_order) - 1:
            logger.info("🎉 Already at highest tier!")
            return False
        
        next_tier = current_tier_order[current_index + 1]
        next_config = self.tier_configs[next_tier]
        
        # Update current state
        old_tier = self.current_tier
        self.current_tier = next_tier
        self.current_capital = next_config.min_capital
        self.tier_start_date = datetime.now()
        
        logger.info(f"🚀 Advanced from {old_tier.value} to {next_tier.value} tier!")
        logger.info(f"💰 New capital target: ${self.current_capital}")
        
        # Save tier advancement
        self._save_tier_state()
        
        return True
    
    def demote_tier(self) -> bool:
        """Demote to the previous tier."""
        current_tier_order = [ScalingTier.SEED, ScalingTier.GROWTH, ScalingTier.SCALE, 
                             ScalingTier.PROFESSIONAL, ScalingTier.INSTITUTIONAL]
        
        current_index = current_tier_order.index(self.current_tier)
        
        if current_index <= 0:
            logger.warning("⚠️ Already at lowest tier!")
            return False
        
        prev_tier = current_tier_order[current_index - 1]
        prev_config = self.tier_configs[prev_tier]
        
        # Update current state
        old_tier = self.current_tier
        self.current_tier = prev_tier
        self.current_capital = prev_config.min_capital
        self.tier_start_date = datetime.now()
        
        logger.warning(f"⬇️ Demoted from {old_tier.value} to {prev_tier.value} tier!")
        logger.info(f"💰 New capital target: ${self.current_capital}")
        
        # Save tier demotion
        self._save_tier_state()
        
        return True
    
    def _save_tier_state(self):
        """Save current tier state."""
        try:
            state_path = self.reports_dir / "capital_scaling" / "tier_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                'current_tier': self.current_tier.value,
                'current_capital': self.current_capital,
                'tier_start_date': self.tier_start_date.isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(state_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"💾 Tier state saved to {state_path}")
            
        except Exception as e:
            logger.error(f"Could not save tier state: {e}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current tier-based parameters."""
        current_config = self.tier_configs[self.current_tier]
        
        return {
            'tier': self.current_tier.value,
            'capital': self.current_capital,
            'max_daily_risk': current_config.max_daily_risk,
            'max_position_size': current_config.max_position_size,
            'max_total_exposure': current_config.max_total_exposure,
            'buy_cap_xrp': self.current_capital * current_config.buy_cap_multiplier * 0.01,  # Convert to XRP
            'scalp_cap_xrp': self.current_capital * current_config.scalp_cap_multiplier * 0.01,
            'funding_arb_cap_xrp': self.current_capital * current_config.funding_arb_cap_multiplier * 0.01,
            'max_daily_drawdown': current_config.max_daily_drawdown,
            'max_rolling_drawdown': current_config.max_rolling_drawdown,
            'kill_switch_threshold': current_config.kill_switch_threshold
        }
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get capital scaling summary."""
        current_config = self.tier_configs[self.current_tier]
        
        # Load recent performance
        trades_df = self._load_performance_data(30)
        performance = self._calculate_tier_performance(trades_df)
        
        # Evaluate advancement
        should_advance, should_demote, reason = self.evaluate_tier_advancement()
        
        return {
            'current_tier': self.current_tier.value,
            'current_capital': self.current_capital,
            'tier_start_date': self.tier_start_date.isoformat(),
            'days_in_tier': (datetime.now() - self.tier_start_date).days,
            'performance': performance,
            'advancement_evaluation': {
                'should_advance': should_advance,
                'should_demote': should_demote,
                'reason': reason
            },
            'tier_requirements': {
                'min_trades': current_config.min_trades,
                'min_days': current_config.min_days,
                'min_win_rate': current_config.min_win_rate,
                'min_sharpe_ratio': current_config.min_sharpe_ratio,
                'advancement_threshold': current_config.advancement_threshold
            }
        }


def main():
    """Main function for testing capital scaling."""
    scaling_plan = CapitalScalingPlan()
    
    # Evaluate tier advancement
    should_advance, should_demote, reason = scaling_plan.evaluate_tier_advancement()
    
    print(f"🎯 Tier Evaluation: {reason}")
    print(f"📊 Should advance: {should_advance}")
    print(f"📉 Should demote: {should_demote}")
    
    # Get current parameters
    params = scaling_plan.get_current_parameters()
    print(f"💰 Current parameters: {params}")
    
    # Get scaling summary
    summary = scaling_plan.get_scaling_summary()
    print(f"📋 Scaling summary: {summary}")


if __name__ == "__main__":
    main()
