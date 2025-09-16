"""
Dashboard Consistency Test
Cross-checks dashboard headline metrics against canonical sources.
"""

import json
import pandas as pd
from pathlib import Path
import sys


def test_dashboard_consistency():
    """Test that dashboard metrics match canonical sources."""
    
    print("🔍 Testing dashboard consistency...")
    
    reports_dir = Path("reports")
    errors = []
    
    # Load canonical sources
    canonical_metrics = {}
    
    # 1. Load tearsheet metrics
    try:
        tearsheet_path = reports_dir / "tearsheets" / "comprehensive_tearsheet.html"
        if tearsheet_path.exists():
            with open(tearsheet_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract metrics from tearsheet
            import re
            sharpe_match = re.search(r'Sharpe Ratio[:\s]*([\d.-]+)', content)
            if sharpe_match:
                canonical_metrics['sharpe_ratio'] = float(sharpe_match.group(1))
            
            dd_match = re.search(r'Max Drawdown[:\s]*([\d.-]+)%', content)
            if dd_match:
                canonical_metrics['max_drawdown'] = float(dd_match.group(1))
            
            win_match = re.search(r'Win Rate[:\s]*([\d.-]+)%', content)
            if win_match:
                canonical_metrics['win_rate'] = float(win_match.group(1))
            
            trades_match = re.search(r'Total Trades[:\s]*([\d,]+)', content)
            if trades_match:
                canonical_metrics['total_trades'] = int(trades_match.group(1).replace(',', ''))
    except Exception as e:
        errors.append(f"Failed to load tearsheet: {e}")
    
    # 2. Load latency metrics
    try:
        latency_path = reports_dir / "latency" / "latency_analysis.json"
        if latency_path.exists():
            with open(latency_path, 'r') as f:
                latency_data = json.load(f)
                canonical_metrics.update({
                    'p95_latency_ms': latency_data.get('p95_latency_ms', 0),
                    'p99_latency_ms': latency_data.get('p99_latency_ms', 0)
                })
    except Exception as e:
        errors.append(f"Failed to load latency data: {e}")
    
    # 3. Load backtest results
    try:
        backtest_path = reports_dir / "backtest_results.json"
        if backtest_path.exists():
            with open(backtest_path, 'r') as f:
                backtest_data = json.load(f)
                canonical_metrics.update({
                    'total_return': backtest_data.get('total_return', 0),
                    'maker_ratio': backtest_data.get('maker_ratio', 0) * 100
                })
    except Exception as e:
        errors.append(f"Failed to load backtest results: {e}")
    
    # 4. Load current dashboard
    try:
        dashboard_path = reports_dir / "executive_dashboard.html"
        if dashboard_path.exists():
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                dashboard_content = f.read()
                
            # Extract metrics from dashboard
            dashboard_metrics = {}
            sharpe_match = re.search(r'Sharpe Ratio.*?([\d.-]+)', dashboard_content)
            if sharpe_match:
                dashboard_metrics['sharpe_ratio'] = float(sharpe_match.group(1))
            
            p95_match = re.search(r'P95 Latency.*?([\d.-]+)\s*ms', dashboard_content)
            if p95_match:
                dashboard_metrics['p95_latency_ms'] = float(p95_match.group(1))
            
            maker_match = re.search(r'Maker Ratio.*?([\d.-]+)%', dashboard_content)
            if maker_match:
                dashboard_metrics['maker_ratio'] = float(maker_match.group(1))
    except Exception as e:
        errors.append(f"Failed to load dashboard: {e}")
    
    # Compare metrics
    print(f"📊 Canonical metrics: {canonical_metrics}")
    print(f"📊 Dashboard metrics: {dashboard_metrics}")
    
    # Check for inconsistencies
    for metric, canonical_value in canonical_metrics.items():
        if metric in dashboard_metrics:
            dashboard_value = dashboard_metrics[metric]
            if abs(canonical_value - dashboard_value) > 0.01:  # Allow small rounding differences
                errors.append(f"Inconsistent {metric}: canonical={canonical_value}, dashboard={dashboard_value}")
        else:
            errors.append(f"Missing metric in dashboard: {metric}")
    
    # Check for zero/placeholder values
    for metric, value in dashboard_metrics.items():
        if value == 0.0 and metric in ['p95_latency_ms', 'maker_ratio']:
            errors.append(f"Dashboard shows placeholder value for {metric}: {value}")
    
    # Summary
    if errors:
        print(f"\n❌ DASHBOARD CONSISTENCY TEST FAILED")
        print(f"📊 {len(errors)} inconsistencies found:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print(f"\n✅ DASHBOARD CONSISTENCY TEST PASSED")
        print(f"📊 All metrics consistent between canonical sources and dashboard")
        return True


def main():
    """Main function."""
    success = test_dashboard_consistency()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
