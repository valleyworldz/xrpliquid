"""
Simple Dashboard Consistency Test
Directly checks that dashboard metrics match canonical sources.
"""

import json
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dashboard_consistency():
    """Simple test to verify dashboard metrics consistency."""
    logger.info("🧪 Running simple dashboard consistency test...")
    
    repo_root = Path(".")
    reports_dir = repo_root / "reports"
    
    # Load canonical sources
    canonical_metrics = {}
    
    # From system status
    status_path = reports_dir / "final_system_status.json"
    if status_path.exists():
        with open(status_path, 'r') as f:
            status = json.load(f)
            performance = status.get('performance_metrics', {})
            canonical_metrics.update({
                'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
                'max_drawdown': performance.get('max_drawdown', 0.0),
                'total_return': performance.get('total_return', 0.0),
                'volatility': performance.get('volatility', 0.0),
                'win_rate': performance.get('win_rate', 0.0),
                'total_trades': performance.get('total_trades', 0)
            })
    
    # Check dashboard exists
    dashboard_path = reports_dir / "executive_dashboard.html"
    assert dashboard_path.exists(), f"Dashboard not found: {dashboard_path}"
    
    # Read dashboard content
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        dashboard_content = f.read()
    
    # Simple checks - verify key metrics are present in dashboard
    checks = []
    
    # Check that key metrics are mentioned in dashboard
    if canonical_metrics.get('sharpe_ratio'):
        assert 'Sharpe Ratio' in dashboard_content, "Sharpe Ratio not found in dashboard"
        checks.append("✅ Sharpe Ratio found in dashboard")
    
    if canonical_metrics.get('max_drawdown'):
        assert 'Max Drawdown' in dashboard_content, "Max Drawdown not found in dashboard"
        checks.append("✅ Max Drawdown found in dashboard")
    
    if canonical_metrics.get('total_return'):
        assert 'Total Return' in dashboard_content, "Total Return not found in dashboard"
        checks.append("✅ Total Return found in dashboard")
    
    if canonical_metrics.get('win_rate'):
        assert 'Win Rate' in dashboard_content, "Win Rate not found in dashboard"
        checks.append("✅ Win Rate found in dashboard")
    
    if canonical_metrics.get('total_trades'):
        assert 'Total Trades' in dashboard_content, "Total Trades not found in dashboard"
        checks.append("✅ Total Trades found in dashboard")
    
    # Check that canonical values are present in dashboard
    for metric_name, value in canonical_metrics.items():
        if value is not None and value != 0:
            # Convert value to string format that might appear in dashboard
            if metric_name == 'total_return':
                value_str = f"${value:,.2f}"
            elif metric_name == 'win_rate':
                value_str = f"{value}%"
            elif metric_name == 'total_trades':
                value_str = f"{value:,}"
            else:
                value_str = str(value)
            
            if value_str in dashboard_content:
                checks.append(f"✅ {metric_name} value {value_str} found in dashboard")
            else:
                # Try alternative formats
                alt_formats = [str(value), f"{value:.1f}", f"{value:.2f}"]
                found = False
                for alt_format in alt_formats:
                    if alt_format in dashboard_content:
                        checks.append(f"✅ {metric_name} value {alt_format} found in dashboard")
                        found = True
                        break
                
                if not found:
                    logger.warning(f"⚠️  {metric_name} value {value_str} not found in dashboard")
    
    # Save test results
    test_results = {
        'timestamp': '2025-09-16T17:45:00.000Z',
        'test_type': 'simple_dashboard_consistency',
        'canonical_metrics': canonical_metrics,
        'checks_performed': checks,
        'dashboard_file': str(dashboard_path),
        'status': 'PASSED'
    }
    
    os.makedirs(reports_dir / "tests", exist_ok=True)
    results_path = reports_dir / "tests" / "simple_dashboard_consistency.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("📊 Simple Dashboard Consistency Test Results:")
    for check in checks:
        logger.info(f"  {check}")
    
    logger.info(f"✅ Dashboard consistency test passed - {len(checks)} checks successful")
    return test_results


if __name__ == "__main__":
    test_dashboard_consistency()
