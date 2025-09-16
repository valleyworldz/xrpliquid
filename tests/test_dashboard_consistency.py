"""
Dashboard Consistency Test
Verifies that dashboard metrics match canonical sources.
"""

import json
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dashboard_consistency():
    """Test that dashboard metrics match canonical sources."""
    logger.info("🧪 Running dashboard consistency test...")
    
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
                'win_rate': performance.get('win_rate', 0.0),
                'total_trades': performance.get('total_trades', 0),
                'maker_ratio': performance.get('maker_ratio', 0.0)
            })
    
    # From latency analysis
    latency_path = reports_dir / "latency" / "latency_analysis.json"
    if latency_path.exists():
        with open(latency_path, 'r') as f:
            latency = json.load(f)
            metrics = latency.get('metrics', {})
            canonical_metrics.update({
                'p50_latency': metrics.get('p50_loop_ms', 0.0),
                'p95_latency': metrics.get('p95_loop_ms', 0.0),
                'p99_latency': metrics.get('p99_loop_ms', 0.0)
            })
    
    # Check dashboard exists
    dashboard_path = reports_dir / "executive_dashboard.html"
    assert dashboard_path.exists(), f"Dashboard not found: {dashboard_path}"
    
    # Read dashboard content
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        dashboard_content = f.read()
    
    # Test consistency
    checks = []
    tolerance = 0.1  # 10% tolerance for formatting differences
    
    # Test Sharpe ratio
    expected_sharpe = canonical_metrics.get('sharpe_ratio', 0.0)
    if expected_sharpe:
        sharpe_str = f"{expected_sharpe:.2f}"
        assert sharpe_str in dashboard_content, f"Sharpe ratio {sharpe_str} not found in dashboard"
        checks.append(f"✅ Sharpe ratio: {sharpe_str}")
    
    # Test P95 latency
    expected_p95 = canonical_metrics.get('p95_latency', 0.0)
    if expected_p95:
        p95_str = f"{expected_p95:.1f}ms"
        assert p95_str in dashboard_content, f"P95 latency {p95_str} not found in dashboard"
        checks.append(f"✅ P95 latency: {p95_str}")
    
    # Test maker ratio
    expected_maker = canonical_metrics.get('maker_ratio', 0.0)
    if expected_maker:
        maker_str = f"{expected_maker:.1f}%"
        assert maker_str in dashboard_content, f"Maker ratio {maker_str} not found in dashboard"
        checks.append(f"✅ Maker ratio: {maker_str}")
    
    # Test total return
    expected_return = canonical_metrics.get('total_return', 0.0)
    if expected_return:
        return_str = f"${expected_return:,.2f}"
        assert return_str in dashboard_content, f"Total return {return_str} not found in dashboard"
        checks.append(f"✅ Total return: {return_str}")
    
    # Test max drawdown
    expected_dd = canonical_metrics.get('max_drawdown', 0.0)
    if expected_dd:
        dd_str = f"{expected_dd:.1f}%"
        assert dd_str in dashboard_content, f"Max drawdown {dd_str} not found in dashboard"
        checks.append(f"✅ Max drawdown: {dd_str}")
    
    # Test win rate
    expected_win = canonical_metrics.get('win_rate', 0.0)
    if expected_win:
        win_str = f"{expected_win:.1f}%"
        assert win_str in dashboard_content, f"Win rate {win_str} not found in dashboard"
        checks.append(f"✅ Win rate: {win_str}")
    
    # Test total trades
    expected_trades = canonical_metrics.get('total_trades', 0)
    if expected_trades:
        trades_str = f"{expected_trades:,}"
        assert trades_str in dashboard_content, f"Total trades {trades_str} not found in dashboard"
        checks.append(f"✅ Total trades: {trades_str}")
    
    # Test P50 latency
    expected_p50 = canonical_metrics.get('p50_latency', 0.0)
    if expected_p50:
        p50_str = f"{expected_p50:.1f}ms"
        assert p50_str in dashboard_content, f"P50 latency {p50_str} not found in dashboard"
        checks.append(f"✅ P50 latency: {p50_str}")
    
    # Test P99 latency
    expected_p99 = canonical_metrics.get('p99_latency', 0.0)
    if expected_p99:
        p99_str = f"{expected_p99:.1f}ms"
        assert p99_str in dashboard_content, f"P99 latency {p99_str} not found in dashboard"
        checks.append(f"✅ P99 latency: {p99_str}")
    
    # Save test results
    test_results = {
        'timestamp': '2025-09-16T18:00:00.000Z',
        'test_type': 'dashboard_consistency',
        'canonical_metrics': canonical_metrics,
        'checks_performed': checks,
        'dashboard_file': str(dashboard_path),
        'status': 'PASSED',
        'total_checks': len(checks)
    }
    
    os.makedirs(reports_dir / "tests", exist_ok=True)
    results_path = reports_dir / "tests" / "dashboard_consistency.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("📊 Dashboard Consistency Test Results:")
    for check in checks:
        logger.info(f"  {check}")
    
    logger.info(f"✅ Dashboard consistency test passed - {len(checks)} checks successful")
    return test_results


if __name__ == "__main__":
    test_dashboard_consistency()