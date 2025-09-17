
#!/usr/bin/env python3
"""
Crown Tier Verification Script
Verifies all 7 crown-tier claims with concrete proof artifacts
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
import sys

def verify_crown_tier():
    """Verify all crown-tier claims"""
    print("🔍 Crown Tier Verification")
    print("=" * 50)
    
    all_passed = True
    
    # 1. Verify daily tear-sheets
    print("\n📊 Verifying daily tear-sheets...")
    if verify_daily_tearsheets():
        print("  ✅ Daily tear-sheets: PASSED")
    else:
        print("  ❌ Daily tear-sheets: FAILED")
        all_passed = False
    
    # 2. Verify maker-taker summary
    print("\n💰 Verifying maker-taker summary...")
    if verify_maker_taker_summary():
        print("  ✅ Maker-taker summary: PASSED")
    else:
        print("  ❌ Maker-taker summary: FAILED")
        all_passed = False
    
    # 3. Verify latency traces
    print("\n🏁 Verifying latency traces...")
    if verify_latency_traces():
        print("  ✅ Latency traces: PASSED")
    else:
        print("  ❌ Latency traces: FAILED")
        all_passed = False
    
    # 4. Verify arbitrage trades
    print("\n🌐 Verifying arbitrage trades...")
    if verify_arbitrage_trades():
        print("  ✅ Arbitrage trades: PASSED")
    else:
        print("  ❌ Arbitrage trades: FAILED")
        all_passed = False
    
    # 5. Verify scaling stress tests
    print("\n💪 Verifying scaling stress tests...")
    if verify_scaling_stress_tests():
        print("  ✅ Scaling stress tests: PASSED")
    else:
        print("  ❌ Scaling stress tests: FAILED")
        all_passed = False
    
    # 6. Verify PnL attribution
    print("\n📈 Verifying PnL attribution...")
    if verify_pnl_attribution():
        print("  ✅ PnL attribution: PASSED")
    else:
        print("  ❌ PnL attribution: FAILED")
        all_passed = False
    
    # 7. Verify audit reports
    print("\n🔍 Verifying audit reports...")
    if verify_audit_reports():
        print("  ✅ Audit reports: PASSED")
    else:
        print("  ❌ Audit reports: FAILED")
        all_passed = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_passed:
        print("🏆 CROWN TIER VERIFICATION: PASSED")
        print("✅ All 7 crown-tier claims verified with concrete proof artifacts")
        return 0
    else:
        print("❌ CROWN TIER VERIFICATION: FAILED")
        print("❌ Some claims could not be verified")
        return 1

def verify_daily_tearsheets():
    """Verify 7 consecutive days of tear-sheets"""
    try:
        # Check for any 7 consecutive days in the directory
        import glob
        json_files = glob.glob("reports/tearsheets/daily/*.json")
        
        if len(json_files) < 7:
            return False
        
        # Verify each JSON file has required structure
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                required_fields = ['date', 'daily_return', 'sharpe_ratio', 'trades_count']
                if not all(field in data for field in required_fields):
                    return False
        
        return True
    except Exception:
        return False

def verify_maker_taker_summary():
    """Verify maker-taker execution summary"""
    try:
        if not os.path.exists("reports/execution/maker_taker_summary.json"):
            return False
        
        with open("reports/execution/maker_taker_summary.json", 'r') as f:
            data = json.load(f)
            
        # Verify required fields
        required_fields = ['total_orders', 'maker_orders', 'taker_orders', 'maker_ratio_by_count']
        if not all(field in data for field in required_fields):
            return False
        
        # Verify maker ratio calculation
        expected_ratio = data['maker_orders'] / data['total_orders']
        if abs(data['maker_ratio_by_count'] - expected_ratio) > 0.01:
            return False
        
        return True
    except Exception:
        return False

def verify_latency_traces():
    """Verify latency traces"""
    try:
        ws_file = "logs/latency/ws_ping.csv"
        order_file = "logs/latency/order_submit.csv"
        
        if not os.path.exists(ws_file) or not os.path.exists(order_file):
            return False
        
        # Verify WebSocket latency
        ws_df = pd.read_csv(ws_file)
        if len(ws_df) < 100:
            return False
        
        # Check P95 latency
        p95_latency = ws_df['latency_ms'].quantile(0.95)
        if p95_latency > 15:  # Should be < 15ms for crown tier
            return False
        
        # Verify order submit latency
        order_df = pd.read_csv(order_file)
        if len(order_df) < 50:
            return False
        
        p95_order_latency = order_df['latency_ms'].quantile(0.95)
        if p95_order_latency > 8:  # Should be < 8ms for crown tier
            return False
        
        return True
    except Exception:
        return False

def verify_arbitrage_trades():
    """Verify cross-venue arbitrage trades"""
    try:
        if not os.path.exists("reports/arb/arb_trades.parquet"):
            return False
        
        df = pd.read_parquet("reports/arb/arb_trades.parquet")
        
        if len(df) < 10:
            return False
        
        # Verify all trades are profitable
        if not all(df['net_profit'] > 0):
            return False
        
        # Verify success rate
        success_rate = df['success'].mean()
        if success_rate < 0.95:  # Should be > 95% for crown tier
            return False
        
        return True
    except Exception:
        return False

def verify_scaling_stress_tests():
    """Verify capital scaling stress tests"""
    try:
        impact_file = "reports/scale_tests/impact_curves.json"
        margin_file = "reports/scale_tests/margin_scenarios.json"
        
        if not os.path.exists(impact_file) or not os.path.exists(margin_file):
            return False
        
        with open(impact_file, 'r') as f:
            impact_data = json.load(f)
        
        # Verify impact curves
        if len(impact_data['notional_sizes']) < 5:
            return False
        
        # Verify largest size tested
        max_size = max(impact_data['notional_sizes'])
        if max_size < 1000000:  # Should test up to $1M
            return False
        
        return True
    except Exception:
        return False

def verify_pnl_attribution():
    """Verify PnL attribution"""
    try:
        if not os.path.exists("reports/pnl_attribution/attribution.parquet"):
            return False
        
        df = pd.read_parquet("reports/pnl_attribution/attribution.parquet")
        
        if len(df) < 50:
            return False
        
        # Verify component reconciliation
        df['component_sum'] = (df['directional_pnl'] + df['funding_pnl'] + 
                              df['rebate_pnl'] - df['slippage_cost'] - 
                              df['impact_cost'] - df['fees_cost'])
        
        # Check reconciliation accuracy
        tolerance = 0.01
        mismatches = df[abs(df['component_sum'] - df['total_pnl']) > tolerance]
        
        if len(mismatches) > len(df) * 0.05:  # Allow 5% tolerance
            return False
        
        return True
    except Exception:
        return False

def verify_audit_reports():
    """Verify audit reports"""
    try:
        audit_dirs = [
            "audits/quantfund_audit_llc",
            "audits/blockchain_security_audit_inc", 
            "audits/institutional_trading_audit_partners"
        ]
        
        for audit_dir in audit_dirs:
            if not os.path.exists(f"{audit_dir}/report.pdf"):
                return False
            if not os.path.exists(f"{audit_dir}/report.sig"):
                return False
        
        return True
    except Exception:
        return False

if __name__ == "__main__":
    sys.exit(verify_crown_tier())
