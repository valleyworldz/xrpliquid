#!/usr/bin/env python3
"""
PnL Attribution Reconciliation Proof
Verifies that six-component PnL breakdown sums to ledger PnL
"""

import pandas as pd
import numpy as np
from decimal import Decimal
import json

def verify_attribution_reconciliation():
    """
    Verify that PnL components sum to total PnL
    """
    print("🔍 PnL Attribution Reconciliation Proof")
    print("=" * 50)
    
    # Load per-trade ledger
    df = pd.read_csv('reports/crown/attribution/per_trade_ledger.csv')
    
    # Calculate component sum for each trade
    df['component_sum'] = (df['directional_pnl'] + df['funding_pnl'] + 
                          df['rebate_pnl'] - df['slippage_cost'] - 
                          df['impact_cost'] - df['fees_cost'])
    
    # Check reconciliation
    tolerance = 0.01  # 1 cent tolerance
    mismatches = df[abs(df['component_sum'] - df['total_pnl']) > tolerance]
    
    print(f"📊 Reconciliation Results:")
    print(f"  Total Trades: {len(df)}")
    print(f"  Mismatches: {len(mismatches)}")
    print(f"  Tolerance: ${tolerance}")
    
    if len(mismatches) == 0:
        print("✅ PnL attribution reconciliation: PASSED")
        print("✅ All trades reconcile within tolerance")
        
        # Calculate summary statistics
        total_pnl = df['total_pnl'].sum()
        total_directional = df['directional_pnl'].sum()
        total_funding = df['funding_pnl'].sum()
        total_rebate = df['rebate_pnl'].sum()
        total_slippage = df['slippage_cost'].sum()
        total_impact = df['impact_cost'].sum()
        total_fees = df['fees_cost'].sum()
        
        print(f"\n📈 Component Summary:")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Directional: ${total_directional:.2f} ({total_directional/total_pnl*100:.1f}%)")
        print(f"  Funding: ${total_funding:.2f} ({total_funding/total_pnl*100:.1f}%)")
        print(f"  Rebate: ${total_rebate:.2f} ({total_rebate/total_pnl*100:.1f}%)")
        print(f"  Slippage: ${total_slippage:.2f} ({total_slippage/total_pnl*100:.1f}%)")
        print(f"  Impact: ${total_impact:.2f} ({total_impact/total_pnl*100:.1f}%)")
        print(f"  Fees: ${total_fees:.2f} ({total_fees/total_pnl*100:.1f}%)")
        
        return True
    else:
        print("❌ PnL attribution reconciliation: FAILED")
        print(f"❌ {len(mismatches)} trades failed reconciliation")
        
        for idx, row in mismatches.iterrows():
            print(f"  Trade {row['trade_id']}: Component sum ${row['component_sum']:.2f} != Ledger PnL ${row['total_pnl']:.2f}")
        
        return False

if __name__ == "__main__":
    success = verify_attribution_reconciliation()
    exit(0 if success else 1)
