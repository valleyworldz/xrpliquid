
import pandas as pd
import numpy as np

def verify_attribution_reconciliation():
    """Verify that PnL components sum to total PnL"""
    df = pd.read_parquet('reports/pnl_attribution/attribution.parquet')
    
    # Calculate component sum for each trade
    df['component_sum'] = (df['directional_pnl'] + df['funding_pnl'] + 
                          df['rebate_pnl'] - df['slippage_cost'] - 
                          df['impact_cost'] - df['fees_cost'])
    
    # Check reconciliation
    tolerance = 0.01  # 1 cent tolerance
    mismatches = df[abs(df['component_sum'] - df['total_pnl']) > tolerance]
    
    if len(mismatches) == 0:
        print("✅ PnL attribution reconciliation: PASSED")
        return True
    else:
        print(f"❌ PnL attribution reconciliation: FAILED ({len(mismatches)} mismatches)")
        return False

if __name__ == "__main__":
    verify_attribution_reconciliation()
