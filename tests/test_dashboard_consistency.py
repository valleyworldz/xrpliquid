"""
Unit test to verify dashboard consistency with canonical sources.
"""

import json
import re
from pathlib import Path
import unittest


class TestDashboardConsistency(unittest.TestCase):
    """Test that dashboard metrics match canonical sources."""
    
    def setUp(self):
        self.reports_dir = Path("reports")
        self.dashboard_path = self.reports_dir / "executive_dashboard.html"
        self.tearsheet_path = self.reports_dir / "tearsheets" / "comprehensive_tearsheet.html"
        self.latency_path = self.reports_dir / "latency" / "latency_analysis.json"
        self.status_path = self.reports_dir / "final_system_status.json"
    
    def test_dashboard_shows_correct_sharpe(self):
        """Test that dashboard shows the correct Sharpe ratio (1.80)."""
        if not self.dashboard_path.exists():
            self.skipTest("Dashboard file not found")
        
        with open(self.dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_content = f.read()
        
        # Check that the dashboard contains the correct Sharpe ratio
        self.assertIn("1.80", dashboard_content, "Dashboard should show Sharpe ratio 1.80")
        self.assertIn("Sharpe Ratio", dashboard_content, "Dashboard should contain Sharpe Ratio label")
    
    def test_dashboard_shows_correct_drawdown(self):
        """Test that dashboard shows the correct max drawdown (5.00%)."""
        if not self.dashboard_path.exists():
            self.skipTest("Dashboard file not found")
        
        with open(self.dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_content = f.read()
        
        # Check that the dashboard contains the correct max drawdown
        self.assertIn("5.00%", dashboard_content, "Dashboard should show max drawdown 5.00%")
        self.assertIn("Max Drawdown", dashboard_content, "Dashboard should contain Max Drawdown label")
    
    def test_dashboard_shows_correct_latency(self):
        """Test that dashboard shows the correct P95 latency (89.7ms)."""
        if not self.dashboard_path.exists():
            self.skipTest("Dashboard file not found")
        
        with open(self.dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_content = f.read()
        
        # Check that the dashboard contains the correct P95 latency
        self.assertIn("89.7ms", dashboard_content, "Dashboard should show P95 latency 89.7ms")
        self.assertIn("P95 Latency", dashboard_content, "Dashboard should contain P95 Latency label")
    
    def test_dashboard_shows_correct_trades(self):
        """Test that dashboard shows the correct total trades (1,000)."""
        if not self.dashboard_path.exists():
            self.skipTest("Dashboard file not found")
        
        with open(self.dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_content = f.read()
        
        # Check that the dashboard contains the correct total trades
        self.assertIn("1,000", dashboard_content, "Dashboard should show total trades 1,000")
        self.assertIn("Total Trades", dashboard_content, "Dashboard should contain Total Trades label")
    
    def test_dashboard_data_source_attribution(self):
        """Test that dashboard shows data source attribution."""
        if not self.dashboard_path.exists():
            self.skipTest("Dashboard file not found")
        
        with open(self.dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_content = f.read()
        
        # Check that the dashboard shows data source attribution
        self.assertIn("Source: final_system_status.json", dashboard_content, 
                     "Dashboard should show data source attribution")
        self.assertIn("Source: latency_analysis.json", dashboard_content, 
                     "Dashboard should show latency data source attribution")
    
    def test_canonical_sources_exist(self):
        """Test that all canonical data sources exist."""
        self.assertTrue(self.status_path.exists(), "final_system_status.json should exist")
        self.assertTrue(self.latency_path.exists(), "latency_analysis.json should exist")
        self.assertTrue(self.tearsheet_path.exists(), "comprehensive_tearsheet.html should exist")
    
    def test_expected_performance_values(self):
        """Test that performance metrics meet expected targets."""
        if not self.status_path.exists():
            self.skipTest("Status file not found")
        
        with open(self.status_path, 'r') as f:
            status_data = json.load(f)
        
        metrics = status_data['performance_metrics']
        
        # Performance targets
        self.assertGreaterEqual(metrics['sharpe_ratio'], 1.5, "Sharpe ratio below target (1.5)")
        self.assertLessEqual(metrics['max_drawdown'], 10.0, "Max drawdown above target (10%)")
        self.assertLessEqual(metrics['p95_latency_ms'], 100.0, "P95 latency above target (100ms)")
        self.assertGreaterEqual(metrics['maker_ratio'], 60.0, "Maker ratio below target (60%)")


if __name__ == "__main__":
    unittest.main()