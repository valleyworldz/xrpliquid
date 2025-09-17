"""
Statistical Validation and Research Quality
Implements deflated Sharpe, PSR, parameter stability, and stationarity monitoring.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy import stats
from scipy.stats import jarque_bera, kstest, anderson
import warnings
warnings.filterwarnings('ignore')


class StatisticalValidator:
    """Comprehensive statistical validation for trading system research quality."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.validation_dir = self.reports_dir / "statistical_validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_deflated_sharpe(self, returns: pd.Series, n_trials: int = 1) -> Dict[str, float]:
        """Calculate deflated Sharpe ratio to adjust for multiple testing."""
        
        if len(returns) < 2:
            return {"deflated_sharpe": 0.0, "original_sharpe": 0.0, "deflation_factor": 1.0}
        
        # Original Sharpe ratio
        mean_return = returns.mean()
        std_return = returns.std()
        original_sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
        
        # Deflation factor (Bailey & López de Prado)
        n_observations = len(returns)
        deflation_factor = np.sqrt((1 - 1/n_observations) * (1 - 1/n_trials))
        
        # Deflated Sharpe ratio
        deflated_sharpe = original_sharpe * deflation_factor
        
        return {
            "original_sharpe": original_sharpe,
            "deflated_sharpe": deflated_sharpe,
            "deflation_factor": deflation_factor,
            "n_observations": n_observations,
            "n_trials": n_trials
        }
    
    def calculate_probabilistic_sharpe_ratio(self, returns: pd.Series, benchmark_sharpe: float = 0.0) -> Dict[str, float]:
        """Calculate Probabilistic Sharpe Ratio (PSR)."""
        
        if len(returns) < 2:
            return {"psr": 0.0, "confidence": 0.0}
        
        # Calculate Sharpe ratio
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
        
        # PSR calculation (Bailey & López de Prado)
        n_observations = len(returns)
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # PSR formula
        psr_numerator = (sharpe_ratio - benchmark_sharpe) * np.sqrt(n_observations - 1)
        psr_denominator = np.sqrt(1 - skewness * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio**2)
        
        if psr_denominator > 0:
            psr = psr_numerator / psr_denominator
            confidence = stats.norm.cdf(psr)
        else:
            psr = 0.0
            confidence = 0.0
        
        return {
            "psr": psr,
            "confidence": confidence,
            "sharpe_ratio": sharpe_ratio,
            "benchmark_sharpe": benchmark_sharpe,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "n_observations": n_observations
        }
    
    def parameter_stability_analysis(self, 
                                   parameter_values: List[float],
                                   performance_metrics: List[float],
                                   parameter_name: str) -> Dict[str, Any]:
        """Analyze parameter stability and overfitting risk."""
        
        if len(parameter_values) != len(performance_metrics):
            raise ValueError("Parameter values and performance metrics must have same length")
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'parameter': parameter_values,
            'performance': performance_metrics
        })
        
        # Calculate correlation
        correlation = df['parameter'].corr(df['performance'])
        
        # Fit polynomial regression to detect overfitting
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Try different polynomial degrees
        max_degree = min(3, len(df) - 1)
        r2_scores = []
        
        for degree in range(1, max_degree + 1):
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(df[['parameter']])
            
            model = LinearRegression()
            model.fit(X_poly, df['performance'])
            y_pred = model.predict(X_poly)
            r2 = r2_score(df['performance'], y_pred)
            r2_scores.append(r2)
        
        # Overfitting risk assessment
        overfitting_risk = "low"
        if len(r2_scores) > 1 and r2_scores[-1] - r2_scores[0] > 0.3:
            overfitting_risk = "high"
        elif len(r2_scores) > 1 and r2_scores[-1] - r2_scores[0] > 0.1:
            overfitting_risk = "medium"
        
        # Confidence bands (simplified)
        mean_performance = df['performance'].mean()
        std_performance = df['performance'].std()
        confidence_bands = {
            "lower_95": mean_performance - 1.96 * std_performance,
            "upper_95": mean_performance + 1.96 * std_performance,
            "lower_99": mean_performance - 2.58 * std_performance,
            "upper_99": mean_performance + 2.58 * std_performance
        }
        
        return {
            "parameter_name": parameter_name,
            "correlation": correlation,
            "r2_scores_by_degree": r2_scores,
            "overfitting_risk": overfitting_risk,
            "confidence_bands": confidence_bands,
            "parameter_range": {
                "min": safe_float(df['parameter'].min()),
                "max": safe_float(df['parameter'].max()),
                "mean": safe_float(df['parameter'].mean()),
                "std": safe_float(df['parameter'].std())
            },
            "performance_range": {
                "min": safe_float(df['performance'].min()),
                "max": safe_float(df['performance'].max()),
                "mean": safe_float(df['performance'].mean()),
                "std": safe_float(df['performance'].std())
            }
        }
    
    def stationarity_tests(self, time_series: pd.Series, series_name: str) -> Dict[str, Any]:
        """Perform stationarity tests on time series data."""
        
        # Remove NaN values
        clean_series = time_series.dropna()
        
        if len(clean_series) < 10:
            return {
                "series_name": series_name,
                "error": "Insufficient data for stationarity tests",
                "tests": {}
            }
        
        test_results = {}
        
        # Augmented Dickey-Fuller test
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, adf_pvalue, adf_critical, adf_used_lag, adf_nobs, adf_icbest = adfuller(clean_series)
            test_results["adf"] = {
                "statistic": adf_stat,
                "p_value": adf_pvalue,
                "critical_values": adf_critical,
                "is_stationary": adf_pvalue < 0.05
            }
        except Exception as e:
            test_results["adf"] = {"error": str(e)}
        
        # KPSS test
        try:
            from statsmodels.tsa.stattools import kpss
            kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(clean_series)
            test_results["kpss"] = {
                "statistic": kpss_stat,
                "p_value": kpss_pvalue,
                "critical_values": kpss_critical,
                "is_stationary": kpss_pvalue > 0.05
            }
        except Exception as e:
            test_results["kpss"] = {"error": str(e)}
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = jarque_bera(clean_series)
            test_results["jarque_bera"] = {
                "statistic": jb_stat,
                "p_value": jb_pvalue,
                "is_normal": jb_pvalue > 0.05
            }
        except Exception as e:
            test_results["jarque_bera"] = {"error": str(e)}
        
        # Overall stationarity assessment
        is_stationary = True
        if "adf" in test_results and "is_stationary" in test_results["adf"]:
            is_stationary = is_stationary and test_results["adf"]["is_stationary"]
        if "kpss" in test_results and "is_stationary" in test_results["kpss"]:
            is_stationary = is_stationary and test_results["kpss"]["is_stationary"]
        
        return {
            "series_name": series_name,
            "n_observations": len(clean_series),
            "is_stationary": is_stationary,
            "tests": test_results,
            "recommendation": "Stationary" if is_stationary else "Non-stationary - consider differencing"
        }
    
    def edge_decay_monitoring(self, 
                            returns: pd.Series,
                            window_size: int = 252,
                            min_observations: int = 50) -> Dict[str, Any]:
        """Monitor edge decay over time with rolling statistics."""
        
        if len(returns) < min_observations:
            return {
                "error": f"Insufficient data: {len(returns)} < {min_observations}",
                "rolling_stats": {}
            }
        
        # Calculate rolling statistics
        rolling_stats = {}
        
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window=window_size).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
        )
        rolling_stats["sharpe_ratio"] = {
            "values": rolling_sharpe.dropna().tolist(),
            "timestamps": rolling_sharpe.dropna().index.tolist(),
            "current": safe_float(rolling_sharpe.iloc[-1]) if not rolling_sharpe.empty else 0.0,
            "mean": safe_float(rolling_sharpe.mean()) if not rolling_sharpe.empty else 0.0,
            "std": safe_float(rolling_sharpe.std()) if not rolling_sharpe.empty else 0.0
        }
        
        # Rolling win rate
        rolling_win_rate = returns.rolling(window=window_size).apply(
            lambda x: (x > 0).mean() * 100
        )
        rolling_stats["win_rate"] = {
            "values": rolling_win_rate.dropna().tolist(),
            "timestamps": rolling_win_rate.dropna().index.tolist(),
            "current": safe_float(rolling_win_rate.iloc[-1]) if not rolling_win_rate.empty else 0.0,
            "mean": safe_float(rolling_win_rate.mean()) if not rolling_win_rate.empty else 0.0,
            "std": safe_float(rolling_win_rate.std()) if not rolling_win_rate.empty else 0.0
        }
        
        # Rolling expectancy
        rolling_expectancy = returns.rolling(window=window_size).apply(
            lambda x: x.mean()
        )
        rolling_stats["expectancy"] = {
            "values": rolling_expectancy.dropna().tolist(),
            "timestamps": rolling_expectancy.dropna().index.tolist(),
            "current": safe_float(rolling_expectancy.iloc[-1]) if not rolling_expectancy.empty else 0.0,
            "mean": safe_float(rolling_expectancy.mean()) if not rolling_expectancy.empty else 0.0,
            "std": safe_float(rolling_expectancy.std()) if not rolling_expectancy.empty else 0.0
        }
        
        # Edge decay detection
        current_sharpe = rolling_stats["sharpe_ratio"]["current"]
        mean_sharpe = rolling_stats["sharpe_ratio"]["mean"]
        std_sharpe = rolling_stats["sharpe_ratio"]["std"]
        
        # Alert if current performance is significantly below historical mean
        decay_threshold = 2.0  # 2 standard deviations
        edge_decay_detected = current_sharpe < (mean_sharpe - decay_threshold * std_sharpe)
        
        return {
            "window_size": window_size,
            "n_observations": len(returns),
            "edge_decay_detected": edge_decay_detected,
            "decay_threshold": decay_threshold,
            "current_vs_mean_sharpe": current_sharpe - mean_sharpe,
            "rolling_stats": rolling_stats,
            "alert_required": edge_decay_detected
        }
    
    def multiple_testing_correction(self, 
                                  p_values: List[float],
                                  method: str = "bonferroni") -> Dict[str, Any]:
        """Apply multiple testing correction to p-values."""
        
        p_values = np.array(p_values)
        
        if method == "bonferroni":
            corrected_p = p_values * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected_p = np.zeros_like(p_values)
            
            for i, p in enumerate(sorted_p):
                corrected_p[sorted_indices[i]] = p * (len(p_values) - i)
            
            corrected_p = np.minimum(corrected_p, 1.0)
        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected_p = np.zeros_like(p_values)
            
            for i, p in enumerate(sorted_p):
                corrected_p[sorted_indices[i]] = p * len(p_values) / (i + 1)
            
            corrected_p = np.minimum(corrected_p, 1.0)
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return {
            "method": method,
            "original_p_values": p_values.tolist(),
            "corrected_p_values": corrected_p.tolist(),
            "significant_original": (p_values < 0.05).sum(),
            "significant_corrected": (corrected_p < 0.05).sum(),
            "correction_factor": len(p_values)
        }
    
    def generate_research_quality_report(self, 
                                       returns: pd.Series,
                                       parameter_stability_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive research quality report."""
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_summary": {
                "n_observations": len(returns),
                "time_span_days": (returns.index[-1] - returns.index[0]).days if len(returns) > 1 else 0,
                "mean_return": safe_float(returns.mean()),
                "std_return": safe_float(returns.std()),
                "skewness": safe_float(returns.skew()),
                "kurtosis": safe_float(returns.kurtosis())
            }
        }
        
        # Deflated Sharpe and PSR
        report["deflated_sharpe"] = self.calculate_deflated_sharpe(returns)
        report["probabilistic_sharpe_ratio"] = self.calculate_probabilistic_sharpe_ratio(returns)
        
        # Stationarity tests
        report["stationarity_tests"] = self.stationarity_tests(returns, "returns")
        
        # Edge decay monitoring
        report["edge_decay_monitoring"] = self.edge_decay_monitoring(returns)
        
        # Parameter stability (if provided)
        if parameter_stability_data:
            report["parameter_stability"] = parameter_stability_data
        
        # Multiple testing correction note
        report["multiple_testing_note"] = {
            "message": "All statistical tests should be corrected for multiple testing",
            "recommended_methods": ["bonferroni", "holm", "fdr_bh"],
            "current_corrections_applied": ["deflated_sharpe", "probabilistic_sharpe_ratio"]
        }
        
        # Save report
        report_file = self.validation_dir / "research_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def main():
    """Test statistical validation functionality."""
    validator = StatisticalValidator()
    
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)
    
    # Test deflated Sharpe
    deflated_sharpe = validator.calculate_deflated_sharpe(returns, n_trials=10)
    print(f"✅ Deflated Sharpe: {deflated_sharpe['deflated_sharpe']:.3f}")
    
    # Test PSR
    psr = validator.calculate_probabilistic_sharpe_ratio(returns)
    print(f"✅ PSR: {psr['psr']:.3f}, Confidence: {psr['confidence']:.3f}")
    
    # Test stationarity
    stationarity = validator.stationarity_tests(returns, "sample_returns")
    print(f"✅ Stationarity: {stationarity['is_stationary']}")
    
    # Test edge decay monitoring
    edge_decay = validator.edge_decay_monitoring(returns)
    print(f"✅ Edge Decay Detected: {edge_decay['edge_decay_detected']}")
    
    # Generate full report
    report = validator.generate_research_quality_report(returns)
    print(f"✅ Research quality report generated: {len(report)} sections")


if __name__ == "__main__":
    main()
