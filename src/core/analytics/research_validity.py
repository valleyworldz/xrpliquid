"""
Research Validity & Anti-P-Hacking
Implements deflated Sharpe, PSR, parameter stability, and trial counting.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy import stats
from scipy.stats import jarque_bera, kstest
import warnings
warnings.filterwarnings('ignore')


class ResearchValidity:
    """Ensures research quality and prevents p-hacking."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.research_dir = self.reports_dir / "research"
        self.research_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_deflated_sharpe_with_trials(self, 
                                            returns: pd.Series, 
                                            n_trials: int = 1,
                                            n_observations: int = None) -> Dict[str, float]:
        """Calculate deflated Sharpe ratio accounting for multiple trials."""
        
        if n_observations is None:
            n_observations = len(returns)
        
        if n_observations < 2:
            return {
                "original_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "deflation_factor": 1.0,
                "n_trials": n_trials,
                "n_observations": n_observations,
                "trial_adjusted": True
            }
        
        # Original Sharpe ratio
        mean_return = returns.mean()
        std_return = returns.std()
        original_sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
        
        # Deflation factor (Bailey & López de Prado)
        # Accounts for multiple testing and finite sample bias
        deflation_factor = np.sqrt((1 - 1/n_observations) * (1 - 1/n_trials))
        
        # Deflated Sharpe ratio
        deflated_sharpe = original_sharpe * deflation_factor
        
        # Confidence interval for deflated Sharpe
        confidence_interval = self._calculate_sharpe_confidence_interval(
            deflated_sharpe, n_observations, n_trials
        )
        
        return {
            "original_sharpe": original_sharpe,
            "deflated_sharpe": deflated_sharpe,
            "deflation_factor": deflation_factor,
            "n_trials": n_trials,
            "n_observations": n_observations,
            "trial_adjusted": True,
            "confidence_interval": confidence_interval,
            "significance_level": 0.05
        }
    
    def _calculate_sharpe_confidence_interval(self, 
                                            sharpe_ratio: float, 
                                            n_obs: int, 
                                            n_trials: int) -> Dict[str, float]:
        """Calculate confidence interval for Sharpe ratio."""
        
        # Standard error of Sharpe ratio
        se_sharpe = np.sqrt((1 + 0.5 * sharpe_ratio**2) / n_obs)
        
        # Adjust for multiple trials
        se_sharpe_adjusted = se_sharpe * np.sqrt(n_trials)
        
        # 95% confidence interval
        z_score = 1.96
        ci_lower = sharpe_ratio - z_score * se_sharpe_adjusted
        ci_upper = sharpe_ratio + z_score * se_sharpe_adjusted
        
        return {
            "lower_95": ci_lower,
            "upper_95": ci_upper,
            "standard_error": se_sharpe_adjusted
        }
    
    def calculate_probabilistic_sharpe_ratio(self, 
                                           returns: pd.Series, 
                                           benchmark_sharpe: float = 0.0,
                                           n_trials: int = 1) -> Dict[str, float]:
        """Calculate Probabilistic Sharpe Ratio with trial adjustment."""
        
        if len(returns) < 2:
            return {"psr": 0.0, "confidence": 0.0, "n_trials": n_trials}
        
        # Calculate Sharpe ratio
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
        
        # PSR calculation (Bailey & López de Prado)
        n_observations = len(returns)
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # PSR formula with trial adjustment
        psr_numerator = (sharpe_ratio - benchmark_sharpe) * np.sqrt(n_observations - 1)
        psr_denominator = np.sqrt(1 - skewness * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio**2)
        
        if psr_denominator > 0:
            psr = psr_numerator / psr_denominator
            # Adjust for multiple trials
            psr_adjusted = psr / np.sqrt(n_trials)
            confidence = stats.norm.cdf(psr_adjusted)
        else:
            psr = 0.0
            psr_adjusted = 0.0
            confidence = 0.0
        
        return {
            "psr": psr,
            "psr_adjusted": psr_adjusted,
            "confidence": confidence,
            "sharpe_ratio": sharpe_ratio,
            "benchmark_sharpe": benchmark_sharpe,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "n_observations": n_observations,
            "n_trials": n_trials,
            "trial_adjusted": True
        }
    
    def generate_parameter_stability_plots(self, 
                                         parameter_values: List[float],
                                         performance_metrics: List[float],
                                         parameter_name: str,
                                         regime_labels: List[str] = None) -> Dict[str, Any]:
        """Generate parameter stability plots with confidence bands."""
        
        if len(parameter_values) != len(performance_metrics):
            raise ValueError("Parameter values and performance metrics must have same length")
        
        # Create DataFrame
        df = pd.DataFrame({
            'parameter': parameter_values,
            'performance': performance_metrics
        })
        
        if regime_labels and len(regime_labels) == len(parameter_values):
            df['regime'] = regime_labels
        
        # Calculate statistics
        correlation = df['parameter'].corr(df['performance'])
        
        # Fit polynomial regression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Try different polynomial degrees
        max_degree = min(3, len(df) - 1)
        r2_scores = []
        models = []
        
        for degree in range(1, max_degree + 1):
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(df[['parameter']])
            
            model = LinearRegression()
            model.fit(X_poly, df['performance'])
            y_pred = model.predict(X_poly)
            r2 = r2_score(df['performance'], y_pred)
            
            r2_scores.append(r2)
            models.append((model, poly_features))
        
        # Overfitting risk assessment
        overfitting_risk = "low"
        if len(r2_scores) > 1 and r2_scores[-1] - r2_scores[0] > 0.3:
            overfitting_risk = "high"
        elif len(r2_scores) > 1 and r2_scores[-1] - r2_scores[0] > 0.1:
            overfitting_risk = "medium"
        
        # Generate plot
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        if regime_labels:
            for regime in df['regime'].unique():
                regime_data = df[df['regime'] == regime]
                plt.scatter(regime_data['parameter'], regime_data['performance'], 
                           label=f'Regime: {regime}', alpha=0.7)
        else:
            plt.scatter(df['parameter'], df['performance'], alpha=0.7)
        
        # Confidence bands
        param_range = np.linspace(df['parameter'].min(), df['parameter'].max(), 100)
        
        # Use best model for confidence bands
        best_degree = np.argmax(r2_scores) + 1
        best_model, best_poly = models[best_degree - 1]
        
        X_range = best_poly.fit_transform(param_range.reshape(-1, 1))
        y_pred_range = best_model.predict(X_range)
        
        plt.plot(param_range, y_pred_range, 'r-', linewidth=2, 
                label=f'Best Fit (Degree {best_degree}, R² = {r2_scores[best_degree-1]:.3f})')
        
        # Add confidence bands (simplified)
        y_std = df['performance'].std()
        plt.fill_between(param_range, y_pred_range - 1.96*y_std, y_pred_range + 1.96*y_std, 
                        alpha=0.2, color='red', label='95% Confidence Band')
        
        plt.xlabel(f'{parameter_name}')
        plt.ylabel('Performance Metric')
        plt.title(f'Parameter Stability: {parameter_name}\nCorrelation: {correlation:.3f}, Overfitting Risk: {overfitting_risk}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = self.research_dir / f"param_stability_{parameter_name.lower().replace(' ', '_')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "parameter_name": parameter_name,
            "correlation": correlation,
            "r2_scores_by_degree": r2_scores,
            "overfitting_risk": overfitting_risk,
            "best_degree": best_degree,
            "plot_file": str(plot_file),
            "parameter_range": {
                "min": float(df['parameter'].min()),
                "max": float(df['parameter'].max()),
                "mean": float(df['parameter'].mean()),
                "std": float(df['parameter'].std())
            },
            "performance_range": {
                "min": float(df['performance'].min()),
                "max": float(df['performance'].max()),
                "mean": float(df['performance'].mean()),
                "std": float(df['performance'].std())
            }
        }
    
    def track_trial_count(self, 
                         strategy_name: str,
                         parameter_combinations: List[Dict[str, Any]],
                         backtest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track number of trials to prevent p-hacking."""
        
        trial_record = {
            "strategy_name": strategy_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trials": len(parameter_combinations),
            "parameter_space_size": self._calculate_parameter_space_size(parameter_combinations),
            "backtest_results": backtest_results,
            "trial_adjustment_required": len(parameter_combinations) > 1
        }
        
        # Calculate multiple testing correction
        if len(parameter_combinations) > 1:
            trial_record["multiple_testing_correction"] = {
                "bonferroni_alpha": 0.05 / len(parameter_combinations),
                "holm_alpha": 0.05 / len(parameter_combinations),
                "fdr_alpha": 0.05 * len(parameter_combinations) / len(parameter_combinations)
            }
        
        # Save trial record
        trial_file = self.research_dir / f"trial_count_{strategy_name}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_record, f, indent=2)
        
        return trial_record
    
    def _calculate_parameter_space_size(self, parameter_combinations: List[Dict[str, Any]]) -> int:
        """Calculate the size of the parameter space."""
        if not parameter_combinations:
            return 0
        
        # Count unique parameter combinations
        unique_combinations = set()
        for combo in parameter_combinations:
            combo_str = str(sorted(combo.items()))
            unique_combinations.add(combo_str)
        
        return len(unique_combinations)
    
    def generate_research_quality_report(self, 
                                       strategy_results: Dict[str, Any],
                                       trial_count: int = 1) -> Dict[str, Any]:
        """Generate comprehensive research quality report."""
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy_name": strategy_results.get("strategy_name", "unknown"),
            "trial_count": trial_count,
            "research_quality_metrics": {}
        }
        
        # Add deflated Sharpe if returns available
        if "returns" in strategy_results:
            returns = pd.Series(strategy_results["returns"])
            report["research_quality_metrics"]["deflated_sharpe"] = self.calculate_deflated_sharpe_with_trials(
                returns, trial_count
            )
            report["research_quality_metrics"]["probabilistic_sharpe_ratio"] = self.calculate_probabilistic_sharpe_ratio(
                returns, n_trials=trial_count
            )
        
        # Add parameter stability if available
        if "parameter_stability" in strategy_results:
            report["research_quality_metrics"]["parameter_stability"] = strategy_results["parameter_stability"]
        
        # Add trial tracking
        report["trial_tracking"] = {
            "total_trials": trial_count,
            "multiple_testing_correction_applied": trial_count > 1,
            "recommended_alpha": 0.05 / trial_count if trial_count > 1 else 0.05
        }
        
        # Research quality assessment
        report["quality_assessment"] = self._assess_research_quality(report["research_quality_metrics"])
        
        # Save report
        report_file = self.research_dir / "research_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _assess_research_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall research quality."""
        
        quality_score = 0
        max_score = 0
        issues = []
        
        # Check deflated Sharpe
        if "deflated_sharpe" in metrics:
            max_score += 1
            deflated_sharpe = metrics["deflated_sharpe"]["deflated_sharpe"]
            if deflated_sharpe > 1.0:
                quality_score += 1
            elif deflated_sharpe < 0.5:
                issues.append("Low deflated Sharpe ratio")
        
        # Check PSR
        if "probabilistic_sharpe_ratio" in metrics:
            max_score += 1
            confidence = metrics["probabilistic_sharpe_ratio"]["confidence"]
            if confidence > 0.8:
                quality_score += 1
            elif confidence < 0.6:
                issues.append("Low PSR confidence")
        
        # Check parameter stability
        if "parameter_stability" in metrics:
            max_score += 1
            overfitting_risk = metrics["parameter_stability"]["overfitting_risk"]
            if overfitting_risk == "low":
                quality_score += 1
            elif overfitting_risk == "high":
                issues.append("High overfitting risk detected")
        
        return {
            "quality_score": quality_score,
            "max_score": max_score,
            "quality_percentage": (quality_score / max_score * 100) if max_score > 0 else 0,
            "issues": issues,
            "overall_assessment": "high" if quality_score == max_score else "medium" if quality_score >= max_score * 0.7 else "low"
        }


def main():
    """Test research validity functionality."""
    validator = ResearchValidity()
    
    # Generate sample data
    np.random.seed(42)
    n_obs = 1000
    returns = pd.Series(np.random.normal(0.001, 0.02, n_obs))
    
    # Test deflated Sharpe with trials
    n_trials = 10
    deflated_sharpe = validator.calculate_deflated_sharpe_with_trials(returns, n_trials)
    print(f"✅ Deflated Sharpe (n_trials={n_trials}): {deflated_sharpe['deflated_sharpe']:.3f}")
    
    # Test PSR with trials
    psr = validator.calculate_probabilistic_sharpe_ratio(returns, n_trials=n_trials)
    print(f"✅ PSR (n_trials={n_trials}): {psr['psr_adjusted']:.3f}, Confidence: {psr['confidence']:.3f}")
    
    # Test parameter stability
    param_values = np.linspace(0.1, 0.9, 20)
    performance = 0.5 + 0.3 * param_values + np.random.normal(0, 0.1, 20)
    
    stability = validator.generate_parameter_stability_plots(
        param_values.tolist(), performance.tolist(), "Risk Parameter"
    )
    print(f"✅ Parameter stability: {stability['overfitting_risk']} risk")
    
    # Test trial tracking
    trial_record = validator.track_trial_count(
        "test_strategy",
        [{"param1": 0.1}, {"param1": 0.2}, {"param1": 0.3}],
        [{"sharpe": 1.0}, {"sharpe": 1.1}, {"sharpe": 1.2}]
    )
    print(f"✅ Trial tracking: {trial_record['total_trials']} trials")
    
    print("✅ Research validity testing completed")


if __name__ == "__main__":
    main()
