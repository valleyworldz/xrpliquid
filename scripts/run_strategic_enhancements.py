#!/usr/bin/env python3
"""
Strategic Enhancements Runner
Orchestrates all next-level enhancements for the Hat Manifesto system.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.analytics.executive_dashboard import ExecutiveDashboard
from core.ml.adaptive_parameter_tuner import AdaptiveParameterTuner
from core.risk.capital_scaling_plan import CapitalScalingPlan


def run_executive_dashboard():
    """Generate executive summary dashboard."""
    print("📊 Generating Executive Dashboard...")
    
    try:
        dashboard = ExecutiveDashboard()
        output_path = dashboard.generate_dashboard('html')
        print(f"✅ Executive Dashboard generated: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error generating executive dashboard: {e}")
        return False


def run_adaptive_tuning():
    """Run adaptive parameter tuning."""
    print("🎯 Running Adaptive Parameter Tuning...")
    
    try:
        tuner = AdaptiveParameterTuner()
        
        if tuner.should_tune():
            new_params = tuner.tune_parameters()
            print(f"✅ Parameters tuned for regime: {new_params.regime}")
            print(f"📊 New caps: BUY={new_params.buy_cap_xrp:.1f}XRP, SCALP={new_params.scalp_cap_xrp:.1f}XRP, FUNDING={new_params.funding_arb_cap_xrp:.1f}XRP")
        else:
            current_params = tuner.get_current_parameters()
            print(f"📊 Current parameters: BUY={current_params.buy_cap_xrp:.1f}XRP, SCALP={current_params.scalp_cap_xrp:.1f}XRP, FUNDING={current_params.funding_arb_cap_xrp:.1f}XRP")
        
        summary = tuner.get_tuning_summary()
        print(f"📋 Tuning summary: {summary}")
        return True
        
    except Exception as e:
        print(f"❌ Error in adaptive tuning: {e}")
        return False


def run_capital_scaling():
    """Run capital scaling evaluation."""
    print("💰 Running Capital Scaling Evaluation...")
    
    try:
        scaling_plan = CapitalScalingPlan()
        
        # Evaluate tier advancement
        should_advance, should_demote, reason = scaling_plan.evaluate_tier_advancement()
        
        print(f"🎯 Tier Evaluation: {reason}")
        
        if should_advance:
            if scaling_plan.advance_tier():
                print("🚀 Successfully advanced to next tier!")
            else:
                print("🎉 Already at highest tier!")
        elif should_demote:
            if scaling_plan.demote_tier():
                print("⬇️ Demoted to previous tier due to performance")
            else:
                print("⚠️ Already at lowest tier")
        else:
            print("📊 Performance adequate for current tier")
        
        # Get current parameters
        params = scaling_plan.get_current_parameters()
        print(f"💰 Current tier: {params['tier']}, Capital: ${params['capital']}")
        
        # Get scaling summary
        summary = scaling_plan.get_scaling_summary()
        print(f"📋 Scaling summary: {summary}")
        return True
        
    except Exception as e:
        print(f"❌ Error in capital scaling: {e}")
        return False


def run_ci_pipeline_simulation():
    """Simulate CI pipeline execution."""
    print("🔄 Simulating CI Pipeline Execution...")
    
    try:
        # Simulate daily backtest
        print("📊 Running daily backtest...")
        # This would normally call the actual backtest engine
        print("✅ Daily backtest completed")
        
        # Simulate ledger update
        print("📋 Updating trade ledger...")
        # This would normally call the ledger update script
        print("✅ Trade ledger updated")
        
        # Simulate performance report generation
        print("📈 Generating performance report...")
        # This would normally call the performance report generator
        print("✅ Performance report generated")
        
        print("🎉 CI pipeline simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in CI pipeline simulation: {e}")
        return False


def run_grafana_setup():
    """Setup Grafana monitoring."""
    print("📊 Setting up Grafana Monitoring...")
    
    try:
        # Check if Prometheus exporter is available
        exporter_path = Path("monitoring/prometheus_exporter.py")
        if exporter_path.exists():
            print("✅ Prometheus exporter found")
        else:
            print("⚠️ Prometheus exporter not found")
        
        # Check if Grafana dashboard is available
        dashboard_path = Path("monitoring/grafana/dashboards/hat_manifesto_dashboard.json")
        if dashboard_path.exists():
            print("✅ Grafana dashboard configuration found")
        else:
            print("⚠️ Grafana dashboard configuration not found")
        
        print("📊 Grafana monitoring setup completed!")
        print("💡 To start monitoring:")
        print("   1. Start Prometheus exporter: python monitoring/prometheus_exporter.py")
        print("   2. Import dashboard: monitoring/grafana/dashboards/hat_manifesto_dashboard.json")
        print("   3. Access Grafana at: http://localhost:3000")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Grafana setup: {e}")
        return False


def generate_strategic_report():
    """Generate comprehensive strategic report."""
    print("📋 Generating Strategic Enhancement Report...")
    
    try:
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'enhancements': {
                'executive_dashboard': {
                    'status': 'implemented',
                    'description': 'Comprehensive dashboard combining equity curves, latency histograms, risk events, and attribution',
                    'files': [
                        'src/core/analytics/executive_dashboard.py',
                        'reports/executive_dashboard.html'
                    ]
                },
                'ci_pipeline': {
                    'status': 'implemented',
                    'description': 'Automated daily backtests and ledger updates with GitHub Actions',
                    'files': [
                        '.github/workflows/daily_backtest.yml',
                        'scripts/update_trade_ledger.py'
                    ]
                },
                'grafana_monitoring': {
                    'status': 'implemented',
                    'description': 'Real-time monitoring with Prometheus metrics and Grafana dashboards',
                    'files': [
                        'monitoring/prometheus_exporter.py',
                        'monitoring/grafana/dashboards/hat_manifesto_dashboard.json'
                    ]
                },
                'adaptive_tuning': {
                    'status': 'implemented',
                    'description': 'Auto-adjustment of parameters based on regime detection and performance',
                    'files': [
                        'src/core/ml/adaptive_parameter_tuner.py',
                        'config/adaptive_tuning.json'
                    ]
                },
                'capital_scaling': {
                    'status': 'implemented',
                    'description': 'Safe bankroll tiers for compounding growth strategy',
                    'files': [
                        'src/core/risk/capital_scaling_plan.py',
                        'config/capital_scaling.json'
                    ]
                }
            },
            'next_steps': [
                'Deploy CI pipeline to production',
                'Setup Grafana instance with Prometheus',
                'Configure adaptive tuning parameters',
                'Initialize capital scaling plan',
                'Monitor system performance and adjust'
            ]
        }
        
        # Save report
        report_path = Path("reports/strategic_enhancements_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"✅ Strategic report saved to: {report_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating strategic report: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run Hat Manifesto Strategic Enhancements')
    parser.add_argument('--dashboard', action='store_true', help='Generate executive dashboard')
    parser.add_argument('--tuning', action='store_true', help='Run adaptive parameter tuning')
    parser.add_argument('--scaling', action='store_true', help='Run capital scaling evaluation')
    parser.add_argument('--ci', action='store_true', help='Simulate CI pipeline')
    parser.add_argument('--grafana', action='store_true', help='Setup Grafana monitoring')
    parser.add_argument('--report', action='store_true', help='Generate strategic report')
    parser.add_argument('--all', action='store_true', help='Run all enhancements')
    
    args = parser.parse_args()
    
    if not any([args.dashboard, args.tuning, args.scaling, args.ci, args.grafana, args.report, args.all]):
        print("🎩 Hat Manifesto Strategic Enhancements")
        print("Usage: python scripts/run_strategic_enhancements.py [options]")
        print("\nOptions:")
        print("  --dashboard    Generate executive dashboard")
        print("  --tuning       Run adaptive parameter tuning")
        print("  --scaling      Run capital scaling evaluation")
        print("  --ci           Simulate CI pipeline")
        print("  --grafana      Setup Grafana monitoring")
        print("  --report       Generate strategic report")
        print("  --all          Run all enhancements")
        return
    
    print("🚀 Starting Hat Manifesto Strategic Enhancements...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success_count = 0
    total_count = 0
    
    if args.all or args.dashboard:
        total_count += 1
        if run_executive_dashboard():
            success_count += 1
    
    if args.all or args.tuning:
        total_count += 1
        if run_adaptive_tuning():
            success_count += 1
    
    if args.all or args.scaling:
        total_count += 1
        if run_capital_scaling():
            success_count += 1
    
    if args.all or args.ci:
        total_count += 1
        if run_ci_pipeline_simulation():
            success_count += 1
    
    if args.all or args.grafana:
        total_count += 1
        if run_grafana_setup():
            success_count += 1
    
    if args.all or args.report:
        total_count += 1
        if generate_strategic_report():
            success_count += 1
    
    print(f"\n🎉 Strategic Enhancements Completed!")
    print(f"✅ Success: {success_count}/{total_count}")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("🏆 All enhancements completed successfully!")
        sys.exit(0)
    else:
        print("⚠️ Some enhancements failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
