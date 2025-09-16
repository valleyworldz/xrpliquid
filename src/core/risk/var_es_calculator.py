"""
VaR/ES Calculator
Calculates Value-at-Risk and Expected Shortfall for risk management.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from scipy import stats
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VaRESCalculator:
    """Calculates VaR and ES for risk management."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Risk parameters
        self.var_confidence_level = 0.95
        self.es_confidence_level = 0.99
        self.lookback_days = 252  # 1 year of trading days
        
        # Load regime sizing config
        self.regime_config = self.load_regime_config()
    
    def load_regime_config(self) -> Dict:
        """Load regime sizing configuration."""
        config_file = self.repo_root / "config" / "sizing_by_regime.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default config
            return {
                'risk_parameters': {
                    'var_confidence_level': 0.95,
                    'es_confidence_level': 0.99
                }
            }
    
    def calculate_var_historical(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate VaR using historical simulation."""
        if len(returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Calculate VaR as the percentile
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[var_index]  # Negative because VaR is typically reported as positive
        
        return var
    
    def calculate_es_historical(self, returns: np.ndarray, confidence_level: float = 0.99) -> float:
        """Calculate Expected Shortfall using historical simulation."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate VaR first
        var = self.calculate_var_historical(returns, confidence_level)
        
        # Calculate ES as the average of returns below VaR
        var_threshold = -var  # Convert back to return space
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var
        
        es = -np.mean(tail_returns)  # Negative because ES is typically reported as positive
        
        return es
    
    def calculate_var_parametric(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate VaR using parametric method (assuming normal distribution)."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mean_return + z_score * std_return)
        
        return var
    
    def calculate_es_parametric(self, returns: np.ndarray, confidence_level: float = 0.99) -> float:
        """Calculate Expected Shortfall using parametric method."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Calculate ES using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        es = -(mean_return + std_return * stats.norm.pdf(z_score) / (1 - confidence_level))
        
        return es
    
    def calculate_regime_var_es(self, returns: np.ndarray, volatility_regime: str) -> Dict:
        """Calculate VaR/ES adjusted for volatility regime."""
        # Get regime-specific parameters
        regime_params = self.regime_config.get('regime_sizing_config', {}).get(volatility_regime, {})
        position_multiplier = regime_params.get('position_size_multiplier', 1.0)
        
        # Calculate base VaR/ES
        var_95 = self.calculate_var_historical(returns, 0.95)
        var_99 = self.calculate_var_historical(returns, 0.99)
        es_95 = self.calculate_es_historical(returns, 0.95)
        es_99 = self.calculate_es_historical(returns, 0.99)
        
        # Adjust for regime
        adjusted_var_95 = var_95 * position_multiplier
        adjusted_var_99 = var_99 * position_multiplier
        adjusted_es_95 = es_95 * position_multiplier
        adjusted_es_99 = es_99 * position_multiplier
        
        return {
            'regime': volatility_regime,
            'position_multiplier': position_multiplier,
            'var_95': adjusted_var_95,
            'var_99': adjusted_var_99,
            'es_95': adjusted_es_95,
            'es_99': adjusted_es_99,
            'base_var_95': var_95,
            'base_var_99': var_99,
            'base_es_95': es_95,
            'base_es_99': es_99
        }
    
    def calculate_funding_directional_guardrail(self, funding_pnl: float, directional_pnl: float, 
                                              net_threshold: float = -0.02) -> Dict:
        """Calculate funding-directional guardrail."""
        net_pnl = funding_pnl + directional_pnl
        
        # Check if net PnL breaches threshold
        breach = net_pnl < net_threshold
        
        # Calculate exposure reduction if breached
        exposure_reduction = 0.0
        if breach:
            # Reduce exposure by 50% if threshold breached
            exposure_reduction = 0.5
        
        return {
            'funding_pnl': funding_pnl,
            'directional_pnl': directional_pnl,
            'net_pnl': net_pnl,
            'threshold': net_threshold,
            'breach': breach,
            'exposure_reduction': exposure_reduction,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_daily_var_es_report(self, returns: np.ndarray, volatility_regime: str = 'normal_volatility') -> Dict:
        """Generate daily VaR/ES report."""
        logger.info("📊 Generating daily VaR/ES report...")
        
        # Calculate VaR/ES for different confidence levels
        var_95 = self.calculate_var_historical(returns, 0.95)
        var_99 = self.calculate_var_historical(returns, 0.99)
        es_95 = self.calculate_es_historical(returns, 0.95)
        es_99 = self.calculate_es_historical(returns, 0.99)
        
        # Calculate regime-adjusted VaR/ES
        regime_var_es = self.calculate_regime_var_es(returns, volatility_regime)
        
        # Calculate additional risk metrics
        max_drawdown = self.calculate_max_drawdown(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'volatility_regime': volatility_regime,
            'sample_size': len(returns),
            'var_es_metrics': {
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99
            },
            'regime_adjusted': regime_var_es,
            'additional_metrics': {
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'volatility': np.std(returns) if len(returns) > 0 else 0.0
            },
            'risk_assessment': {
                'risk_level': self.assess_risk_level(var_95, max_drawdown),
                'recommendations': self.generate_risk_recommendations(var_95, max_drawdown, volatility_regime)
            }
        }
        
        return report
    
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Return maximum drawdown
        return abs(np.min(drawdown))
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns)
    
    def assess_risk_level(self, var_95: float, max_drawdown: float) -> str:
        """Assess overall risk level."""
        if var_95 > 0.05 or max_drawdown > 0.1:
            return 'HIGH'
        elif var_95 > 0.03 or max_drawdown > 0.05:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_risk_recommendations(self, var_95: float, max_drawdown: float, volatility_regime: str) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if var_95 > 0.05:
            recommendations.append("Reduce position sizes - VaR exceeds 5% threshold")
        
        if max_drawdown > 0.1:
            recommendations.append("Implement stricter stop-losses - max drawdown exceeds 10%")
        
        if volatility_regime == 'high_volatility':
            recommendations.append("Increase cooldown periods between trades")
            recommendations.append("Consider reducing position sizes by 50%")
        
        if not recommendations:
            recommendations.append("Risk levels are within acceptable limits")
        
        return recommendations
    
    def save_var_es_report(self, report: Dict) -> Path:
        """Save VaR/ES report to file."""
        risk_dir = self.reports_dir / "risk"
        risk_dir.mkdir(exist_ok=True)
        
        report_file = risk_dir / "var_es.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 VaR/ES report saved: {report_file}")
        return report_file


def main():
    """Main function to demonstrate VaR/ES calculation."""
    calculator = VaRESCalculator()
    
    # Generate sample returns (in real implementation, use actual trade returns)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
    
    # Calculate VaR/ES
    var_95 = calculator.calculate_var_historical(returns, 0.95)
    es_99 = calculator.calculate_es_historical(returns, 0.99)
    
    print(f"VaR (95%): {var_95:.4f}")
    print(f"ES (99%): {es_99:.4f}")
    
    # Generate daily report
    report = calculator.generate_daily_var_es_report(returns, 'normal_volatility')
    calculator.save_var_es_report(report)
    
    print("✅ VaR/ES calculation completed")


if __name__ == "__main__":
    main()
