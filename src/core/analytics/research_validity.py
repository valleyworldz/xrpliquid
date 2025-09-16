"""
Research Validity - Deflated Sharpe, PSR, and Parameter Stability
Implements research-grade validation metrics to prevent p-hacking.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ResearchValidityMetrics:
    """Research validity metrics for backtesting."""
    deflated_sharpe: float
    probabilistic_sharpe_ratio: float
    parameter_stability_score: float
    multiple_testing_penalty: float
    sample_size_adequacy: float
    stationarity_p_value: float
    autocorrelation_p_value: float


class ResearchValidityAnalyzer:
    """Analyzes research validity and prevents p-hacking."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.research_dir = self.reports_dir / "research"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameter stability tracking
        self.parameter_trials = []
        self.performance_history = []
    
    def calculate_deflated_sharpe(self, 
                                returns: pd.Series,
                                num_trials: int = 1,
                                confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate deflated Sharpe ratio to account for multiple testing."""
        
        # Basic Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Deflated Sharpe ratio (Bailey & López de Prado)
        n = len(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Variance of Sharpe ratio
        var_sharpe = (1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + 
                     (kurtosis - 1) / 4 * sharpe_ratio**2) / n
        
        # Multiple testing penalty
        if num_trials > 1:
            # Bonferroni correction
            alpha = 1 - confidence_level
            corrected_alpha = alpha / num_trials
            
            # Adjusted critical value
            critical_value = stats.norm.ppf(1 - corrected_alpha / 2)
        else:
            critical_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Deflated Sharpe ratio
        deflated_sharpe = sharpe_ratio - np.sqrt(var_sharpe) * critical_value
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "deflated_sharpe": deflated_sharpe,
            "variance_sharpe": var_sharpe,
            "multiple_testing_penalty": sharpe_ratio - deflated_sharpe,
            "num_trials": num_trials,
            "confidence_level": confidence_level,
            "critical_value": critical_value
        }
    
    def calculate_probabilistic_sharpe_ratio(self, 
                                           returns: pd.Series,
                                           benchmark_sharpe: float = 0.0) -> Dict[str, float]:
        """Calculate Probabilistic Sharpe Ratio (PSR)."""
        
        # Calculate Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Calculate PSR
        n = len(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Variance of Sharpe ratio
        var_sharpe = (1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + 
                     (kurtosis - 1) / 4 * sharpe_ratio**2) / n
        
        # PSR calculation
        if var_sharpe > 0:
            psr = stats.norm.cdf((sharpe_ratio - benchmark_sharpe) / np.sqrt(var_sharpe))
        else:
            psr = 0.0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "benchmark_sharpe": benchmark_sharpe,
            "probabilistic_sharpe_ratio": psr,
            "variance_sharpe": var_sharpe,
            "sample_size": n,
            "skewness": skewness,
            "kurtosis": kurtosis
        }
    
    def analyze_parameter_stability(self, 
                                  parameter_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter stability across different values."""
        
        if not parameter_results:
            return {
                "stability_score": 0.0,
                "parameter_ranges": {},
                "performance_correlation": 0.0,
                "stability_analysis": "insufficient_data"
            }
        
        # Extract parameter values and performance metrics
        param_data = []
        for result in parameter_results:
            param_data.append({
                "parameters": result.get("parameters", {}),
                "sharpe_ratio": result.get("sharpe_ratio", 0.0),
                "max_drawdown": result.get("max_drawdown", 0.0),
                "total_return": result.get("total_return", 0.0)
            })
        
        # Calculate parameter ranges
        param_ranges = {}
        for param_name in param_data[0]["parameters"].keys():
            values = [data["parameters"][param_name] for data in param_data]
            param_ranges[param_name] = {
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
                "std": np.std(values)
            }
        
        # Calculate performance correlation with parameters
        performance_correlations = {}
        for param_name in param_data[0]["parameters"].keys():
            param_values = [data["parameters"][param_name] for data in param_data]
            sharpe_values = [data["sharpe_ratio"] for data in param_data]
            
            if len(param_values) > 1:
                correlation, p_value = stats.pearsonr(param_values, sharpe_values)
                performance_correlations[param_name] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(param_ranges, performance_correlations)
        
        return {
            "stability_score": stability_score,
            "parameter_ranges": param_ranges,
            "performance_correlations": performance_correlations,
            "stability_analysis": "stable" if stability_score > 0.7 else "unstable"
        }
    
    def _calculate_stability_score(self, 
                                 param_ranges: Dict[str, Any],
                                 performance_correlations: Dict[str, Any]) -> float:
        """Calculate overall parameter stability score."""
        
        if not param_ranges:
            return 0.0
        
        # Score based on parameter range consistency
        range_scores = []
        for param_name, param_info in param_ranges.items():
            # Lower coefficient of variation indicates more stability
            if param_info["std"] > 0:
                cv = param_info["std"] / np.mean([param_info["min"], param_info["max"]])
                range_score = max(0, 1 - cv)
            else:
                range_score = 1.0
            range_scores.append(range_score)
        
        # Score based on performance correlation
        correlation_scores = []
        for param_name, corr_info in performance_correlations.items():
            # Lower absolute correlation indicates more stability
            abs_corr = abs(corr_info["correlation"])
            correlation_score = max(0, 1 - abs_corr)
            correlation_scores.append(correlation_score)
        
        # Combine scores
        if range_scores and correlation_scores:
            stability_score = (np.mean(range_scores) + np.mean(correlation_scores)) / 2
        elif range_scores:
            stability_score = np.mean(range_scores)
        else:
            stability_score = 0.0
        
        return stability_score
    
    def test_stationarity(self, returns: pd.Series) -> Dict[str, Any]:
        """Test for stationarity in returns."""
        
        # Augmented Dickey-Fuller test
        adf_stat, adf_pvalue, _, _, adf_critical, _ = stats.adfuller(returns.dropna())
        
        # Kwiatkowski-Phillips-Schmidt-Shin test
        from statsmodels.tsa.stattools import kpss
        kpss_stat, kpss_pvalue, _, kpss_critical = kpss(returns.dropna())
        
        # Determine stationarity
        is_stationary_adf = adf_pvalue < 0.05
        is_stationary_kpss = kpss_pvalue > 0.05
        
        # Overall stationarity
        is_stationary = is_stationary_adf and is_stationary_kpss
        
        return {
            "is_stationary": is_stationary,
            "adf_test": {
                "statistic": adf_stat,
                "p_value": adf_pvalue,
                "critical_values": adf_critical,
                "is_stationary": is_stationary_adf
            },
            "kpss_test": {
                "statistic": kpss_stat,
                "p_value": kpss_pvalue,
                "critical_values": kpss_critical,
                "is_stationary": is_stationary_kpss
            }
        }
    
    def test_autocorrelation(self, returns: pd.Series, max_lags: int = 10) -> Dict[str, Any]:
        """Test for autocorrelation in returns."""
        
        # Ljung-Box test
        from statsmodels.stats.diagnostic import acorr_ljungbox
        ljung_box_result = acorr_ljungbox(returns.dropna(), lags=max_lags, return_df=True)
        
        # Durbin-Watson test
        from statsmodels.stats.diagnostic import durbin_watson
        dw_stat = durbin_watson(returns.dropna())
        
        # Determine autocorrelation
        has_autocorrelation = any(ljung_box_result['lb_pvalue'] < 0.05)
        
        return {
            "has_autocorrelation": has_autocorrelation,
            "ljung_box_test": {
                "statistics": ljung_box_result['lb_stat'].tolist(),
                "p_values": ljung_box_result['lb_pvalue'].tolist(),
                "significant_lags": ljung_box_result[ljung_box_result['lb_pvalue'] < 0.05].index.tolist()
            },
            "durbin_watson": {
                "statistic": dw_stat,
                "interpretation": "no_autocorrelation" if 1.5 < dw_stat < 2.5 else "autocorrelation_present"
            }
        }
    
    def generate_parameter_stability_plots(self, 
                                         parameter_results: List[Dict[str, Any]],
                                         output_dir: str = None) -> List[str]:
        """Generate parameter stability plots."""
        
        if output_dir is None:
            output_dir = self.research_dir / "param_stability"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        if not parameter_results:
            return plot_files
        
        # Extract parameter data
        param_data = []
        for result in parameter_results:
            param_data.append({
                "parameters": result.get("parameters", {}),
                "sharpe_ratio": result.get("sharpe_ratio", 0.0),
                "max_drawdown": result.get("max_drawdown", 0.0),
                "total_return": result.get("total_return", 0.0)
            })
        
        # Get parameter names
        param_names = list(param_data[0]["parameters"].keys())
        
        # Create plots for each parameter
        for param_name in param_names:
            param_values = [data["parameters"][param_name] for data in param_data]
            sharpe_values = [data["sharpe_ratio"] for data in param_data]
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(param_values, sharpe_values, alpha=0.7)
            plt.xlabel(param_name)
            plt.ylabel('Sharpe Ratio')
            plt.title(f'Parameter Stability: {param_name}')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            if len(param_values) > 1:
                z = np.polyfit(param_values, sharpe_values, 1)
                p = np.poly1d(z)
                plt.plot(param_values, p(param_values), "r--", alpha=0.8)
            
            # Save plot
            plot_file = output_dir / f"param_stability_{param_name}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_files.append(str(plot_file))
        
        return plot_files
    
    def generate_research_validity_report(self, 
                                        returns: pd.Series,
                                        parameter_results: List[Dict[str, Any]] = None,
                                        num_trials: int = 1) -> Dict[str, Any]:
        """Generate comprehensive research validity report."""
        
        # Calculate deflated Sharpe
        deflated_sharpe = self.calculate_deflated_sharpe(returns, num_trials)
        
        # Calculate PSR
        psr = self.calculate_probabilistic_sharpe_ratio(returns)
        
        # Analyze parameter stability
        param_stability = self.analyze_parameter_stability(parameter_results or [])
        
        # Test stationarity
        stationarity = self.test_stationarity(returns)
        
        # Test autocorrelation
        autocorrelation = self.test_autocorrelation(returns)
        
        # Generate report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "deflated_sharpe_analysis": deflated_sharpe,
            "probabilistic_sharpe_ratio": psr,
            "parameter_stability": param_stability,
            "stationarity_test": stationarity,
            "autocorrelation_test": autocorrelation,
            "research_validity_score": self._calculate_research_validity_score(
                deflated_sharpe, psr, param_stability, stationarity, autocorrelation
            ),
            "recommendations": self._generate_recommendations(
                deflated_sharpe, psr, param_stability, stationarity, autocorrelation
            )
        }
        
        # Save report
        report_file = self.research_dir / f"research_validity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _calculate_research_validity_score(self, 
                                         deflated_sharpe: Dict[str, Any],
                                         psr: Dict[str, Any],
                                         param_stability: Dict[str, Any],
                                         stationarity: Dict[str, Any],
                                         autocorrelation: Dict[str, Any]) -> float:
        """Calculate overall research validity score."""
        
        scores = []
        
        # Deflated Sharpe score
        if deflated_sharpe["deflated_sharpe"] > 0:
            scores.append(1.0)
        elif deflated_sharpe["deflated_sharpe"] > -0.5:
            scores.append(0.5)
        else:
            scores.append(0.0)
        
        # PSR score
        if psr["probabilistic_sharpe_ratio"] > 0.8:
            scores.append(1.0)
        elif psr["probabilistic_sharpe_ratio"] > 0.6:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        # Parameter stability score
        scores.append(param_stability["stability_score"])
        
        # Stationarity score
        if stationarity["is_stationary"]:
            scores.append(1.0)
        else:
            scores.append(0.3)
        
        # Autocorrelation score
        if not autocorrelation["has_autocorrelation"]:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        return np.mean(scores)
    
    def _generate_recommendations(self, 
                                deflated_sharpe: Dict[str, Any],
                                psr: Dict[str, Any],
                                param_stability: Dict[str, Any],
                                stationarity: Dict[str, Any],
                                autocorrelation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        
        recommendations = []
        
        # Deflated Sharpe recommendations
        if deflated_sharpe["deflated_sharpe"] < 0:
            recommendations.append("Consider reducing multiple testing or improving strategy performance")
        
        # PSR recommendations
        if psr["probabilistic_sharpe_ratio"] < 0.6:
            recommendations.append("Increase sample size or improve strategy robustness")
        
        # Parameter stability recommendations
        if param_stability["stability_score"] < 0.7:
            recommendations.append("Parameter sensitivity detected - consider regularization or ensemble methods")
        
        # Stationarity recommendations
        if not stationarity["is_stationary"]:
            recommendations.append("Returns are not stationary - consider detrending or regime-based models")
        
        # Autocorrelation recommendations
        if autocorrelation["has_autocorrelation"]:
            recommendations.append("Autocorrelation detected - consider lag features or time series models")
        
        return recommendations


def main():
    """Test research validity analyzer functionality."""
    analyzer = ResearchValidityAnalyzer()
    
    # Create sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.randn(1000) * 0.02 + 0.001)
    
    # Test deflated Sharpe
    deflated_sharpe = analyzer.calculate_deflated_sharpe(returns, num_trials=10)
    print(f"✅ Deflated Sharpe: {deflated_sharpe['deflated_sharpe']:.4f}")
    
    # Test PSR
    psr = analyzer.calculate_probabilistic_sharpe_ratio(returns)
    print(f"✅ PSR: {psr['probabilistic_sharpe_ratio']:.4f}")
    
    # Test parameter stability
    param_results = [
        {"parameters": {"param1": 0.1, "param2": 0.5}, "sharpe_ratio": 1.2, "max_drawdown": 0.05},
        {"parameters": {"param1": 0.2, "param2": 0.6}, "sharpe_ratio": 1.1, "max_drawdown": 0.06},
        {"parameters": {"param1": 0.15, "param2": 0.55}, "sharpe_ratio": 1.15, "max_drawdown": 0.055}
    ]
    
    param_stability = analyzer.analyze_parameter_stability(param_results)
    print(f"✅ Parameter stability: {param_stability['stability_score']:.4f}")
    
    # Test stationarity
    stationarity = analyzer.test_stationarity(returns)
    print(f"✅ Stationarity: {stationarity['is_stationary']}")
    
    # Test autocorrelation
    autocorrelation = analyzer.test_autocorrelation(returns)
    print(f"✅ Autocorrelation: {autocorrelation['has_autocorrelation']}")
    
    # Generate comprehensive report
    report = analyzer.generate_research_validity_report(returns, param_results, num_trials=10)
    print(f"✅ Research validity score: {report['research_validity_score']:.4f}")
    
    print("✅ Research validity analyzer testing completed")


if __name__ == "__main__":
    main()