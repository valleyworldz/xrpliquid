"""
Walk-Forward Analysis with Deflated Sharpe and Probabilistic Sharpe Ratio (PSR)
Advanced statistical analysis for strategy robustness.
"""

import json
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class WalkForwardPSRAnalyzer:
    """Analyzes walk-forward results with advanced statistical metrics."""
    
    def __init__(self, folds_file: str = "reports/walkforward/folds.json"):
        self.folds_file = Path(folds_file)
        self.folds_data = None
        self.psr_results = None
        self.deflated_sharpe = None
    
    def load_folds_data(self):
        """Load walk-forward folds data."""
        if not self.folds_file.exists():
            logger.error(f"Folds file not found: {self.folds_file}")
            return False
        
        try:
            with open(self.folds_file, 'r') as f:
                self.folds_data = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Error loading folds data: {e}")
            return False
    
    def calculate_deflated_sharpe(self, sharpe_ratios: List[float], n_trials: int = None) -> Dict:
        """Calculate Deflated Sharpe Ratio accounting for multiple testing."""
        
        if n_trials is None:
            n_trials = len(sharpe_ratios)
        
        # Calculate average Sharpe
        avg_sharpe = np.mean(sharpe_ratios)
        
        # Calculate standard error
        se_sharpe = np.std(sharpe_ratios) / math.sqrt(len(sharpe_ratios))
        
        # Deflated Sharpe Ratio formula
        # DS = (SR - sqrt(2*log(n_trials))) / sqrt(1 - sqrt(2*log(n_trials)/n_trials))
        log_n = math.log(n_trials)
        bias_term = math.sqrt(2 * log_n)
        
        if bias_term >= n_trials:
            # Edge case: bias term too large
            deflated_sharpe = avg_sharpe
        else:
            denominator = math.sqrt(1 - bias_term / n_trials)
            if denominator > 0:
                deflated_sharpe = (avg_sharpe - bias_term) / denominator
            else:
                deflated_sharpe = avg_sharpe
        
        return {
            "average_sharpe": avg_sharpe,
            "deflated_sharpe": deflated_sharpe,
            "bias_term": bias_term,
            "n_trials": n_trials,
            "standard_error": se_sharpe,
            "deflation_factor": bias_term / avg_sharpe if avg_sharpe > 0 else 0
        }
    
    def calculate_probabilistic_sharpe_ratio(self, sharpe_ratio: float, n_observations: int, 
                                           skewness: float = 0, kurtosis: float = 3) -> Dict:
        """Calculate Probabilistic Sharpe Ratio (PSR)."""
        
        # PSR formula: PSR(SR*) = Φ((SR - SR*) * sqrt(n-1) / sqrt(1 - γ3*SR + (γ4-1)/4*SR²))
        # Where SR* is the benchmark Sharpe (typically 0)
        benchmark_sharpe = 0.0
        
        # Calculate the denominator
        denominator_term1 = 1 - skewness * sharpe_ratio
        denominator_term2 = (kurtosis - 1) / 4 * sharpe_ratio**2
        denominator = math.sqrt(denominator_term1 + denominator_term2)
        
        if denominator <= 0:
            # Edge case: invalid denominator
            psr = 0.5  # Neutral probability
        else:
            # Calculate PSR using normal CDF approximation
            z_score = (sharpe_ratio - benchmark_sharpe) * math.sqrt(n_observations - 1) / denominator
            psr = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
        
        return {
            "probabilistic_sharpe_ratio": psr,
            "benchmark_sharpe": benchmark_sharpe,
            "z_score": z_score if 'z_score' in locals() else 0,
            "n_observations": n_observations,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "confidence_level": psr * 100
        }
    
    def analyze_walkforward_robustness(self) -> Dict:
        """Analyze walk-forward robustness with advanced metrics."""
        
        if not self.folds_data:
            logger.error("No folds data loaded")
            return {}
        
        folds = self.folds_data["fold_results"]
        
        # Extract Sharpe ratios
        train_sharpes = [fold["train_sharpe"] for fold in folds]
        test_sharpes = [fold["test_sharpe"] for fold in folds]
        
        # Calculate Deflated Sharpe for test results
        deflated_sharpe_test = self.calculate_deflated_sharpe(test_sharpes, len(test_sharpes))
        
        # Calculate PSR for average test Sharpe
        avg_test_sharpe = np.mean(test_sharpes)
        total_observations = sum(fold["test_trades"] for fold in folds)
        
        # Calculate skewness and kurtosis of test Sharpe ratios
        test_sharpe_skewness = self._calculate_skewness(test_sharpes)
        test_sharpe_kurtosis = self._calculate_kurtosis(test_sharpes)
        
        psr_analysis = self.calculate_probabilistic_sharpe_ratio(
            avg_test_sharpe, total_observations, test_sharpe_skewness, test_sharpe_kurtosis
        )
        
        # Calculate stability metrics
        sharpe_stability = 1 - (np.std(test_sharpes) / np.mean(test_sharpes))
        return_stability = 1 - (np.std([fold["test_return"] for fold in folds]) / 
                               np.mean([fold["test_return"] for fold in folds]))
        
        # Calculate degradation metrics
        avg_train_sharpe = np.mean(train_sharpes)
        sharpe_degradation = (avg_train_sharpe - avg_test_sharpe) / avg_train_sharpe
        
        return {
            "deflated_sharpe_analysis": deflated_sharpe_test,
            "probabilistic_sharpe_analysis": psr_analysis,
            "stability_metrics": {
                "sharpe_stability": sharpe_stability,
                "return_stability": return_stability,
                "sharpe_degradation": sharpe_degradation
            },
            "fold_statistics": {
                "total_folds": len(folds),
                "avg_train_sharpe": avg_train_sharpe,
                "avg_test_sharpe": avg_test_sharpe,
                "sharpe_std": np.std(test_sharpes),
                "min_test_sharpe": min(test_sharpes),
                "max_test_sharpe": max(test_sharpes)
            }
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skewness = np.mean([(x - mean)**3 for x in data]) / (std**3)
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 3.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 3.0
        
        kurtosis = np.mean([(x - mean)**4 for x in data]) / (std**4)
        return kurtosis
    
    def generate_walkforward_summary_html(self, analysis_results: Dict) -> str:
        """Generate HTML summary of walk-forward analysis."""
        
        if not analysis_results:
            return "<p>No analysis results available</p>"
        
        deflated = analysis_results["deflated_sharpe_analysis"]
        psr = analysis_results["probabilistic_sharpe_analysis"]
        stability = analysis_results["stability_metrics"]
        fold_stats = analysis_results["fold_statistics"]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Walk-Forward Analysis Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                               gap: 15px; margin-bottom: 20px; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 8px; 
                              box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2E8B57; }}
                .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
                .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .positive {{ color: #2E8B57; }}
                .negative {{ color: #DC143C; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Walk-Forward Analysis Summary</h1>
                <p>Advanced Statistical Analysis with Deflated Sharpe & PSR</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value positive">{deflated['deflated_sharpe']:.3f}</div>
                    <div class="metric-label">Deflated Sharpe Ratio</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value positive">{psr['confidence_level']:.1f}%</div>
                    <div class="metric-label">PSR Confidence</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value positive">{stability['sharpe_stability']:.3f}</div>
                    <div class="metric-label">Sharpe Stability</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {'positive' if stability['sharpe_degradation'] < 0.1 else 'negative'}">
                        {stability['sharpe_degradation']:.1%}
                    </div>
                    <div class="metric-label">Sharpe Degradation</div>
                </div>
            </div>
            
            <div class="section">
                <h3>📈 Statistical Analysis</h3>
                <p><strong>Average Test Sharpe:</strong> {fold_stats['avg_test_sharpe']:.3f}</p>
                <p><strong>Sharpe Standard Deviation:</strong> {fold_stats['sharpe_std']:.3f}</p>
                <p><strong>Sharpe Range:</strong> {fold_stats['min_test_sharpe']:.3f} to {fold_stats['max_test_sharpe']:.3f}</p>
                <p><strong>Total Folds:</strong> {fold_stats['total_folds']}</p>
            </div>
            
            <div class="section">
                <h3>🔬 Advanced Metrics</h3>
                <p><strong>Deflated Sharpe:</strong> {deflated['deflated_sharpe']:.3f} (accounts for {deflated['n_trials']} trials)</p>
                <p><strong>PSR Confidence:</strong> {psr['confidence_level']:.1f}% (probability of positive Sharpe)</p>
                <p><strong>Bias Term:</strong> {deflated['bias_term']:.3f}</p>
                <p><strong>Deflation Factor:</strong> {deflated['deflation_factor']:.1%}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def run_analysis(self) -> Dict:
        """Run complete walk-forward analysis."""
        
        if not self.load_folds_data():
            return {}
        
        analysis_results = self.analyze_walkforward_robustness()
        
        # Generate HTML summary
        html_content = self.generate_walkforward_summary_html(analysis_results)
        
        # Save HTML summary
        output_file = Path("reports/walkforward/walkforward_summary.html")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save analysis results
        results_file = Path("reports/walkforward/psr_analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Walk-forward analysis completed. Results saved to {results_file}")
        logger.info(f"HTML summary saved to {output_file}")
        
        return analysis_results

def main():
    """Run walk-forward PSR analysis."""
    analyzer = WalkForwardPSRAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        print("✅ Walk-forward PSR analysis completed")
        print(f"   Deflated Sharpe: {results['deflated_sharpe_analysis']['deflated_sharpe']:.3f}")
        print(f"   PSR Confidence: {results['probabilistic_sharpe_analysis']['confidence_level']:.1f}%")
        print(f"   Sharpe Stability: {results['stability_metrics']['sharpe_stability']:.3f}")
        return 0
    else:
        print("❌ Walk-forward PSR analysis failed")
        return 1

if __name__ == "__main__":
    exit(main())
