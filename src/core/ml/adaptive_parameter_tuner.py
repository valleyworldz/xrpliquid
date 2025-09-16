"""
Adaptive Parameter Tuning System
Uses regime detection to auto-adjust caps and thresholds based on performance.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """Parameter set for a specific regime."""
    regime: str
    timestamp: str
    
    # Position sizing parameters
    buy_cap_xrp: float = 10.0
    scalp_cap_xrp: float = 0.5
    funding_arb_cap_xrp: float = 0.8
    
    # Risk parameters
    daily_dd_limit: float = 0.02  # 2%
    rolling_dd_limit: float = 0.05  # 5%
    kill_switch_threshold: float = 0.08  # 8%
    
    # Execution parameters
    maker_ratio_target: float = 0.8  # 80%
    max_slippage_bps: float = 5.0  # 5 bps
    latency_threshold_ms: float = 100.0  # 100ms
    
    # Strategy-specific parameters
    funding_rate_threshold: float = 0.0001  # 0.01%
    momentum_lookback: int = 20
    mean_reversion_threshold: float = 0.02  # 2%
    
    # Performance tracking
    performance_score: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0


class AdaptiveParameterTuner:
    """Adaptive parameter tuning system using regime detection and performance feedback."""
    
    def __init__(self, config_path: str = "config/adaptive_tuning.json", 
                 reports_dir: str = "reports"):
        self.config_path = Path(config_path)
        self.reports_dir = Path(reports_dir)
        
        # Load or initialize configuration
        self.config = self._load_config()
        
        # Parameter history
        self.parameter_history: List[ParameterSet] = []
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Tuning state
        self.current_regime = "normal"
        self.last_tuning_time = datetime.now()
        self.tuning_interval_hours = 24  # Tune every 24 hours
        
        logger.info("🎯 Adaptive Parameter Tuner initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load adaptive tuning configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        # Default configuration
        default_config = {
            "tuning_enabled": True,
            "tuning_interval_hours": 24,
            "min_trades_for_tuning": 100,
            "performance_lookback_days": 7,
            "regime_confidence_threshold": 0.7,
            "parameter_adjustment_rate": 0.1,  # 10% adjustment per tuning cycle
            "max_parameter_change": 0.5,  # 50% max change
            "regimes": {
                "bull": {
                    "buy_cap_multiplier": 1.2,
                    "scalp_cap_multiplier": 0.8,
                    "funding_arb_cap_multiplier": 1.1,
                    "risk_tolerance_multiplier": 1.1
                },
                "bear": {
                    "buy_cap_multiplier": 0.7,
                    "scalp_cap_multiplier": 1.3,
                    "funding_arb_cap_multiplier": 0.9,
                    "risk_tolerance_multiplier": 0.8
                },
                "sideways": {
                    "buy_cap_multiplier": 0.9,
                    "scalp_cap_multiplier": 1.1,
                    "funding_arb_cap_multiplier": 1.0,
                    "risk_tolerance_multiplier": 0.9
                },
                "high_volatility": {
                    "buy_cap_multiplier": 0.6,
                    "scalp_cap_multiplier": 1.4,
                    "funding_arb_cap_multiplier": 0.8,
                    "risk_tolerance_multiplier": 0.7
                },
                "low_volatility": {
                    "buy_cap_multiplier": 1.3,
                    "scalp_cap_multiplier": 0.7,
                    "funding_arb_cap_multiplier": 1.2,
                    "risk_tolerance_multiplier": 1.2
                }
            }
        }
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _detect_current_regime(self) -> Tuple[str, float]:
        """Detect current market regime."""
        try:
            # Load regime analysis
            regime_path = self.reports_dir / "regime" / "regime_analysis.json"
            if regime_path.exists():
                with open(regime_path, 'r') as f:
                    regime_data = json.load(f)
                
                current_regime = regime_data.get('current_regime', 'normal')
                confidence = regime_data.get('confidence', 0.5)
                
                return current_regime, confidence
        except Exception as e:
            logger.warning(f"Could not detect regime: {e}")
        
        return "normal", 0.5
    
    def _load_performance_data(self, lookback_days: int = 7) -> pd.DataFrame:
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
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            recent_trades = trades_df[trades_df['ts'] >= cutoff_timestamp]
            return recent_trades
            
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")
            return pd.DataFrame()
    
    def _calculate_performance_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from trades."""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_pnl': 0.0,
                'avg_trade': 0.0,
                'volatility': 0.0
            }
        
        returns = trades_df['pnl_realized']
        
        metrics = {
            'total_trades': len(trades_df),
            'win_rate': (returns > 0).mean() * 100,
            'total_pnl': returns.sum(),
            'avg_trade': returns.mean(),
            'volatility': returns.std() if len(returns) > 1 else 0.0
        }
        
        # Sharpe ratio
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['avg_trade'] / metrics['volatility']) * (252**0.5)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Max drawdown
        cumulative = returns.cumsum()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        metrics['max_drawdown'] = abs(drawdown.min()) if not drawdown.empty else 0.0
        
        return metrics
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score (0-100)."""
        if metrics['total_trades'] < 10:
            return 50.0  # Neutral score for insufficient data
        
        # Weighted performance score
        score = 0.0
        
        # Win rate (30% weight)
        win_rate_score = min(metrics['win_rate'] / 50.0, 1.0) * 30
        score += win_rate_score
        
        # Sharpe ratio (25% weight)
        sharpe_score = min(metrics['sharpe_ratio'] / 2.0, 1.0) * 25
        score += sharpe_score
        
        # Total P&L (20% weight)
        pnl_score = min(max(metrics['total_pnl'] / 1000.0, -1.0), 1.0) * 20
        score += pnl_score
        
        # Drawdown penalty (15% weight)
        dd_penalty = min(metrics['max_drawdown'] / 0.1, 1.0) * 15
        score -= dd_penalty
        
        # Trade frequency (10% weight)
        trade_score = min(metrics['total_trades'] / 100.0, 1.0) * 10
        score += trade_score
        
        return max(0.0, min(100.0, score))
    
    def _get_regime_parameters(self, regime: str) -> Dict[str, float]:
        """Get parameter multipliers for a specific regime."""
        return self.config.get('regimes', {}).get(regime, {
            'buy_cap_multiplier': 1.0,
            'scalp_cap_multiplier': 1.0,
            'funding_arb_cap_multiplier': 1.0,
            'risk_tolerance_multiplier': 1.0
        })
    
    def _adjust_parameters(self, current_params: ParameterSet, 
                          regime: str, performance_score: float) -> ParameterSet:
        """Adjust parameters based on regime and performance."""
        regime_multipliers = self._get_regime_parameters(regime)
        adjustment_rate = self.config.get('parameter_adjustment_rate', 0.1)
        max_change = self.config.get('max_parameter_change', 0.5)
        
        # Create new parameter set
        new_params = ParameterSet(
            regime=regime,
            timestamp=datetime.now().isoformat(),
            performance_score=performance_score
        )
        
        # Adjust position caps based on regime
        new_params.buy_cap_xrp = current_params.buy_cap_xrp * regime_multipliers.get('buy_cap_multiplier', 1.0)
        new_params.scalp_cap_xrp = current_params.scalp_cap_xrp * regime_multipliers.get('scalp_cap_multiplier', 1.0)
        new_params.funding_arb_cap_xrp = current_params.funding_arb_cap_xrp * regime_multipliers.get('funding_arb_cap_multiplier', 1.0)
        
        # Adjust risk parameters based on performance
        risk_multiplier = regime_multipliers.get('risk_tolerance_multiplier', 1.0)
        
        if performance_score > 70:  # Good performance - can take more risk
            risk_adjustment = 1 + adjustment_rate
        elif performance_score < 30:  # Poor performance - reduce risk
            risk_adjustment = 1 - adjustment_rate
        else:  # Neutral performance
            risk_adjustment = 1.0
        
        # Apply risk adjustments with limits
        new_params.daily_dd_limit = max(0.01, min(0.05, 
            current_params.daily_dd_limit * risk_multiplier * risk_adjustment))
        new_params.rolling_dd_limit = max(0.02, min(0.10, 
            current_params.rolling_dd_limit * risk_multiplier * risk_adjustment))
        new_params.kill_switch_threshold = max(0.05, min(0.15, 
            current_params.kill_switch_threshold * risk_multiplier * risk_adjustment))
        
        # Adjust execution parameters based on performance
        if performance_score > 80:
            new_params.maker_ratio_target = min(0.95, current_params.maker_ratio_target + 0.05)
            new_params.max_slippage_bps = max(2.0, current_params.max_slippage_bps - 0.5)
        elif performance_score < 40:
            new_params.maker_ratio_target = max(0.6, current_params.maker_ratio_target - 0.05)
            new_params.max_slippage_bps = min(10.0, current_params.max_slippage_bps + 1.0)
        
        # Regime-specific strategy adjustments
        if regime == "high_volatility":
            new_params.momentum_lookback = max(10, current_params.momentum_lookback - 5)
            new_params.mean_reversion_threshold = min(0.05, current_params.mean_reversion_threshold + 0.01)
        elif regime == "low_volatility":
            new_params.momentum_lookback = min(50, current_params.momentum_lookback + 10)
            new_params.mean_reversion_threshold = max(0.01, current_params.mean_reversion_threshold - 0.005)
        
        return new_params
    
    def should_tune(self) -> bool:
        """Check if it's time to tune parameters."""
        if not self.config.get('tuning_enabled', True):
            return False
        
        time_since_last = datetime.now() - self.last_tuning_time
        return time_since_last.total_seconds() >= (self.tuning_interval_hours * 3600)
    
    def tune_parameters(self) -> ParameterSet:
        """Perform parameter tuning based on current regime and performance."""
        logger.info("🎯 Starting parameter tuning...")
        
        # Detect current regime
        regime, confidence = self._detect_current_regime()
        self.current_regime = regime
        
        logger.info(f"📊 Detected regime: {regime} (confidence: {confidence:.2f})")
        
        # Load recent performance data
        lookback_days = self.config.get('performance_lookback_days', 7)
        trades_df = self._load_performance_data(lookback_days)
        
        if trades_df.empty:
            logger.warning("No recent trade data available for tuning")
            return self._get_current_parameters()
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(trades_df)
        performance_score = self._calculate_performance_score(metrics)
        
        logger.info(f"📈 Performance score: {performance_score:.1f}/100")
        logger.info(f"📊 Metrics: {metrics['total_trades']} trades, {metrics['win_rate']:.1f}% win rate, {metrics['sharpe_ratio']:.2f} Sharpe")
        
        # Get current parameters
        current_params = self._get_current_parameters()
        
        # Adjust parameters
        new_params = self._adjust_parameters(current_params, regime, performance_score)
        
        # Update performance tracking
        new_params.trades_count = metrics['total_trades']
        new_params.win_rate = metrics['win_rate']
        new_params.sharpe_ratio = metrics['sharpe_ratio']
        new_params.max_drawdown = metrics['max_drawdown']
        
        # Save parameter history
        self.parameter_history.append(new_params)
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'regime': regime,
            'performance_score': performance_score,
            'metrics': metrics
        })
        
        # Update last tuning time
        self.last_tuning_time = datetime.now()
        
        # Save updated parameters
        self._save_parameters(new_params)
        
        logger.info("✅ Parameter tuning completed")
        logger.info(f"🎯 New parameters: BUY={new_params.buy_cap_xrp:.1f}XRP, SCALP={new_params.scalp_cap_xrp:.1f}XRP, FUNDING={new_params.funding_arb_cap_xrp:.1f}XRP")
        
        return new_params
    
    def _get_current_parameters(self) -> ParameterSet:
        """Get current parameter set."""
        if self.parameter_history:
            return self.parameter_history[-1]
        
        # Return default parameters
        return ParameterSet(
            regime=self.current_regime,
            timestamp=datetime.now().isoformat()
        )
    
    def _save_parameters(self, params: ParameterSet):
        """Save parameters to file."""
        try:
            params_path = self.reports_dir / "adaptive_tuning" / "current_parameters.json"
            params_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(params_path, 'w') as f:
                json.dump(asdict(params), f, indent=2)
            
            # Save history
            history_path = self.reports_dir / "adaptive_tuning" / "parameter_history.json"
            history_data = [asdict(p) for p in self.parameter_history]
            
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Save performance history
            perf_path = self.reports_dir / "adaptive_tuning" / "performance_history.json"
            with open(perf_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            logger.info(f"💾 Parameters saved to {params_path}")
            
        except Exception as e:
            logger.error(f"Could not save parameters: {e}")
    
    def get_current_parameters(self) -> ParameterSet:
        """Get current parameter set (public interface)."""
        return self._get_current_parameters()
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get summary of tuning activity."""
        return {
            'current_regime': self.current_regime,
            'last_tuning_time': self.last_tuning_time.isoformat(),
            'tuning_enabled': self.config.get('tuning_enabled', True),
            'parameter_history_count': len(self.parameter_history),
            'performance_history_count': len(self.performance_history),
            'current_parameters': asdict(self._get_current_parameters()) if self.parameter_history else None
        }


def main():
    """Main function for testing adaptive tuning."""
    tuner = AdaptiveParameterTuner()
    
    if tuner.should_tune():
        new_params = tuner.tune_parameters()
        print(f"🎯 Tuned parameters: {asdict(new_params)}")
    else:
        current_params = tuner.get_current_parameters()
        print(f"📊 Current parameters: {asdict(current_params)}")
    
    summary = tuner.get_tuning_summary()
    print(f"📋 Tuning summary: {summary}")


if __name__ == "__main__":
    main()
