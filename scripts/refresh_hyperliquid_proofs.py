#!/usr/bin/env python3
"""
Hyperliquid Proof Artifacts Refresh Script
==========================================
Refreshes all Hyperliquid proof artifacts in one go.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.data.hyperliquid_provenance_verifier import HyperliquidProvenanceVerifier
from src.core.portfolio.hyperliquid_portfolio_manager import HyperliquidPortfolioManager
from src.core.ml.hyperliquid_drift_monitor import HyperliquidDriftMonitor

def refresh_provenance():
    """Refresh data provenance artifacts"""
    print("📊 Refreshing data provenance...")
    verifier = HyperliquidProvenanceVerifier()
    snapshot = verifier.capture_daily_snapshot()
    print(f"✅ Captured snapshot with hash: {snapshot.get('snapshot_hash', 'N/A')[:16]}...")

def refresh_portfolio():
    """Refresh portfolio risk artifacts"""
    print("⚖️ Refreshing portfolio risk analysis...")
    pm = HyperliquidPortfolioManager()
    
    # Sample multi-market data
    sample_returns = {
        "XRP": [0.001, -0.002, 0.0005, 0.0015, -0.0008, 0.0021, -0.0012, 0.0009],
        "BTC": [0.002, -0.001, 0.0018, 0.0025, -0.0015, 0.0032, -0.0008, 0.0014],
        "ETH": [0.0015, -0.0015, 0.0012, 0.0028, -0.0012, 0.0028, -0.0015, 0.0011],
        "SOL": [0.0025, -0.002, 0.0022, 0.0035, -0.002, 0.0041, -0.0018, 0.0022]
    }
    
    mark_prices = {"XRP": 0.65, "BTC": 45000, "ETH": 3200, "SOL": 95}
    positions = {"XRP": 1000, "BTC": 0.1, "ETH": 0.5, "SOL": 10}
    
    risk_metrics = pm.calculate_portfolio_risk(positions, mark_prices)
    print(f"✅ Portfolio VaR 95%: {risk_metrics.get('portfolio_var_95', 'N/A')}")

def refresh_ml_drift():
    """Refresh ML drift monitoring"""
    print("🧠 Refreshing ML drift analysis...")
    drift_monitor = HyperliquidDriftMonitor()
    
    # Sample feature data
    baseline_features = {
        "spread": [0.001, 0.0012, 0.0008, 0.0015, 0.0009],
        "depth_imbalance": [0.1, 0.2, -0.1, 0.3, -0.2],
        "funding_rate": [0.0001, 0.0002, -0.0001, 0.0003, -0.0002]
    }
    
    current_features = {
        "spread": [0.0012, 0.0014, 0.001, 0.0017, 0.0011],
        "depth_imbalance": [0.15, 0.25, -0.05, 0.35, -0.15],
        "funding_rate": [0.0002, 0.0003, 0.0, 0.0004, -0.0001]
    }
    
    drift_results = drift_monitor.monitor_feature_drift(current_features)
    print(f"✅ Drift detected in {sum(1 for r in drift_results.values() if r.get('drift_detected', False))} features")

def main():
    """Main refresh function"""
    print("🎩 ULTIMATEBACKTESTBUILDER-2025: Refreshing Hyperliquid Proof Artifacts")
    print("=" * 70)
    
    try:
        refresh_provenance()
        refresh_portfolio()
        refresh_ml_drift()
        
        print("=" * 70)
        print("🎉 Successfully refreshed all core Hyperliquid proof artifacts!")
        print("📁 Check reports/ directory for updated files")
        
    except Exception as e:
        print(f"❌ Error during refresh: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
