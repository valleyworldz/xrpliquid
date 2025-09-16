"""
Regime-Aware Sizing - Volatility-Adaptive Position Sizing
Implements regime detection and adaptive position sizing to prevent whipsaw.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class VolatilityRegime(Enum):
    """Volatility regime enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RegimeMetrics:
    """Regime detection metrics."""
    regime: VolatilityRegime
    volatility: float
    confidence: float
    regime_duration_days: int
    transition_probability: float


class RegimeAwareSizing:
    """Implements regime-aware position sizing with hysteresis."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.risk_dir = self.reports_dir / "risk"
        self.risk_dir.mkdir(parents=True, exist_ok=True)
        
        # Regime detection parameters
        self.regime_thresholds = {
            "low": 0.15,      # < 15% daily volatility
            "medium": 0.25,   # 15-25% daily volatility
            "high": 0.40,     # 25-40% daily volatility
            "extreme": 0.60   # > 40% daily volatility
        }
        
        # Position sizing by regime
        self.regime_sizing = {
            VolatilityRegime.LOW: {
                "base_size_multiplier": 1.0,
                "max_position_size": 0.20,  # 20% of capital
                "volatility_target": 0.15,
                "hysteresis_factor": 0.8
            },
            VolatilityRegime.MEDIUM: {
                "base_size_multiplier": 0.7,
                "max_position_size": 0.15,  # 15% of capital
                "volatility_target": 0.20,
                "hysteresis_factor": 0.9
            },
            VolatilityRegime.HIGH: {
                "base_size_multiplier": 0.5,
                "max_position_size": 0.10,  # 10% of capital
                "volatility_target": 0.25,
                "hysteresis_factor": 1.1
            },
            VolatilityRegime.EXTREME: {
                "base_size_multiplier": 0.3,
                "max_position_size": 0.05,  # 5% of capital
                "volatility_target": 0.30,
                "hysteresis_factor": 1.2
            }
        }
        
        # Hysteresis state
        self.current_regime = VolatilityRegime.MEDIUM
        self.regime_transition_count = 0
        self.last_regime_change = datetime.now(timezone.utc)
        self.hysteresis_buffer = 0.05  # 5% buffer for regime changes
    
    def detect_volatility_regime(self, 
                               returns: pd.Series,
                               lookback_days: int = 30) -> RegimeMetrics:
        """Detect current volatility regime."""
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=lookback_days).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0.0
        
        # Determine regime based on thresholds
        if current_vol < self.regime_thresholds["low"]:
            detected_regime = VolatilityRegime.LOW
        elif current_vol < self.regime_thresholds["medium"]:
            detected_regime = VolatilityRegime.MEDIUM
        elif current_vol < self.regime_thresholds["high"]:
            detected_regime = VolatilityRegime.HIGH
        else:
            detected_regime = VolatilityRegime.EXTREME
        
        # Apply hysteresis
        final_regime = self._apply_hysteresis(detected_regime, current_vol)
        
        # Calculate confidence
        confidence = self._calculate_regime_confidence(rolling_vol, final_regime)
        
        # Calculate regime duration
        regime_duration = (datetime.now(timezone.utc) - self.last_regime_change).days
        
        # Calculate transition probability
        transition_prob = self._calculate_transition_probability(rolling_vol, final_regime)
        
        return RegimeMetrics(
            regime=final_regime,
            volatility=current_vol,
            confidence=confidence,
            regime_duration_days=regime_duration,
            transition_probability=transition_prob
        )
    
    def _apply_hysteresis(self, 
                         detected_regime: VolatilityRegime,
                         current_vol: float) -> VolatilityRegime:
        """Apply hysteresis to prevent regime whipsaw."""
        
        # Get hysteresis factor for current regime
        current_sizing = self.regime_sizing[self.current_regime]
        hysteresis_factor = current_sizing["hysteresis_factor"]
        
        # Calculate adjusted thresholds
        if detected_regime != self.current_regime:
            # Check if change is significant enough
            vol_threshold = self.regime_thresholds[detected_regime.value]
            adjusted_threshold = vol_threshold * hysteresis_factor
            
            if current_vol > adjusted_threshold:
                # Regime change confirmed
                self.current_regime = detected_regime
                self.regime_transition_count += 1
                self.last_regime_change = datetime.now(timezone.utc)
        
        return self.current_regime
    
    def _calculate_regime_confidence(self, 
                                   rolling_vol: pd.Series,
                                   regime: VolatilityRegime) -> float:
        """Calculate confidence in regime detection."""
        
        if rolling_vol.empty:
            return 0.0
        
        # Calculate how consistently the volatility stays in the regime
        regime_vol_range = self._get_regime_volatility_range(regime)
        
        # Count observations within regime range
        in_regime_count = 0
        for vol in rolling_vol.dropna():
            if regime_vol_range[0] <= vol <= regime_vol_range[1]:
                in_regime_count += 1
        
        confidence = in_regime_count / len(rolling_vol.dropna())
        return min(confidence, 1.0)
    
    def _get_regime_volatility_range(self, regime: VolatilityRegime) -> Tuple[float, float]:
        """Get volatility range for a regime."""
        
        if regime == VolatilityRegime.LOW:
            return (0.0, self.regime_thresholds["low"])
        elif regime == VolatilityRegime.MEDIUM:
            return (self.regime_thresholds["low"], self.regime_thresholds["medium"])
        elif regime == VolatilityRegime.HIGH:
            return (self.regime_thresholds["medium"], self.regime_thresholds["high"])
        else:  # EXTREME
            return (self.regime_thresholds["high"], 1.0)
    
    def _calculate_transition_probability(self, 
                                        rolling_vol: pd.Series,
                                        regime: VolatilityRegime) -> float:
        """Calculate probability of regime transition."""
        
        if len(rolling_vol) < 10:
            return 0.0
        
        # Calculate volatility trend
        recent_vol = rolling_vol.tail(5).mean()
        older_vol = rolling_vol.head(-5).mean()
        
        vol_change = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0.0
        
        # Higher volatility change increases transition probability
        transition_prob = min(abs(vol_change) * 2, 1.0)
        
        return transition_prob
    
    def calculate_regime_aware_position_size(self, 
                                           base_position_size: float,
                                           current_volatility: float,
                                           account_equity: float,
                                           regime_metrics: RegimeMetrics) -> Dict[str, Any]:
        """Calculate position size based on regime and volatility."""
        
        # Get regime-specific sizing parameters
        regime_params = self.regime_sizing[regime_metrics.regime]
        
        # Calculate volatility-adjusted size
        vol_adjustment = regime_params["volatility_target"] / max(current_volatility, 0.01)
        vol_adjusted_size = base_position_size * vol_adjustment
        
        # Apply regime multiplier
        regime_adjusted_size = vol_adjusted_size * regime_params["base_size_multiplier"]
        
        # Apply confidence adjustment
        confidence_adjustment = 0.5 + (regime_metrics.confidence * 0.5)  # 0.5 to 1.0
        confidence_adjusted_size = regime_adjusted_size * confidence_adjustment
        
        # Apply maximum position size limit
        max_size_usd = account_equity * regime_params["max_position_size"]
        final_position_size = min(confidence_adjusted_size, max_size_usd)
        
        # Calculate position size as percentage of equity
        position_size_pct = final_position_size / account_equity
        
        return {
            "regime": regime_metrics.regime.value,
            "base_position_size": base_position_size,
            "volatility_adjustment": vol_adjustment,
            "regime_multiplier": regime_params["base_size_multiplier"],
            "confidence_adjustment": confidence_adjustment,
            "final_position_size_usd": final_position_size,
            "final_position_size_pct": position_size_pct,
            "max_position_size_usd": max_size_usd,
            "regime_confidence": regime_metrics.confidence,
            "transition_probability": regime_metrics.transition_probability
        }
    
    def generate_regime_sizing_report(self, 
                                    returns: pd.Series,
                                    account_equity: float = 100000.0) -> Dict[str, Any]:
        """Generate comprehensive regime sizing report."""
        
        # Detect current regime
        regime_metrics = self.detect_volatility_regime(returns)
        
        # Calculate position sizes for different base sizes
        base_sizes = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20% of equity
        position_sizes = {}
        
        for base_size_pct in base_sizes:
            base_size_usd = account_equity * base_size_pct
            current_vol = returns.std() * np.sqrt(252) if not returns.empty else 0.0
            
            sizing_result = self.calculate_regime_aware_position_size(
                base_size_usd, current_vol, account_equity, regime_metrics
            )
            
            position_sizes[f"{base_size_pct*100:.0f}%"] = sizing_result
        
        # Generate regime transition analysis
        regime_transitions = self._analyze_regime_transitions(returns)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_regime": {
                "regime": regime_metrics.regime.value,
                "volatility": regime_metrics.volatility,
                "confidence": regime_metrics.confidence,
                "duration_days": regime_metrics.regime_duration_days,
                "transition_probability": regime_metrics.transition_probability
            },
            "regime_sizing_parameters": {
                regime.value: params for regime, params in self.regime_sizing.items()
            },
            "position_sizing_results": position_sizes,
            "regime_transitions": regime_transitions,
            "hysteresis_state": {
                "current_regime": self.current_regime.value,
                "transition_count": self.regime_transition_count,
                "last_change": self.last_regime_change.isoformat(),
                "hysteresis_buffer": self.hysteresis_buffer
            }
        }
        
        # Save report
        report_file = self.risk_dir / f"regime_sizing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _analyze_regime_transitions(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze historical regime transitions."""
        
        if len(returns) < 60:  # Need at least 60 days
            return {"error": "Insufficient data for transition analysis"}
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        
        # Detect regime changes
        regime_changes = []
        current_regime = None
        
        for i, vol in enumerate(rolling_vol.dropna()):
            if vol < self.regime_thresholds["low"]:
                detected_regime = VolatilityRegime.LOW
            elif vol < self.regime_thresholds["medium"]:
                detected_regime = VolatilityRegime.MEDIUM
            elif vol < self.regime_thresholds["high"]:
                detected_regime = VolatilityRegime.HIGH
            else:
                detected_regime = VolatilityRegime.EXTREME
            
            if current_regime is None:
                current_regime = detected_regime
            elif detected_regime != current_regime:
                regime_changes.append({
                    "from_regime": current_regime.value,
                    "to_regime": detected_regime.value,
                    "volatility": vol,
                    "day_index": i
                })
                current_regime = detected_regime
        
        return {
            "total_transitions": len(regime_changes),
            "transition_frequency_days": len(rolling_vol) / max(len(regime_changes), 1),
            "transitions": regime_changes[-10:] if regime_changes else []  # Last 10 transitions
        }


def main():
    """Test regime-aware sizing functionality."""
    sizing = RegimeAwareSizing()
    
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
    
    # Detect regime
    regime_metrics = sizing.detect_volatility_regime(returns)
    print(f"✅ Detected regime: {regime_metrics.regime.value} (confidence: {regime_metrics.confidence:.2f})")
    
    # Calculate position size
    sizing_result = sizing.calculate_regime_aware_position_size(
        base_position_size=10000.0,
        current_volatility=0.20,
        account_equity=100000.0,
        regime_metrics=regime_metrics
    )
    print(f"✅ Position size: ${sizing_result['final_position_size_usd']:.2f} ({sizing_result['final_position_size_pct']:.1%})")
    
    # Generate report
    report = sizing.generate_regime_sizing_report(returns)
    print(f"✅ Regime sizing report generated")
    
    print("✅ Regime-aware sizing testing completed")


if __name__ == "__main__":
    main()
