#!/usr/bin/env python3
"""
Trade Ledger Update Script
Updates the canonical trade ledger with new backtest results.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.analytics.trade_ledger import CanonicalTradeLedger


def update_trade_ledger(input_dir: str, output_path: str):
    """Update trade ledger with new backtest results."""
    print(f"📊 Updating trade ledger from {input_dir} to {output_path}")
    
    input_path = Path(input_dir)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize trade ledger
    ledger = CanonicalTradeLedger()
    
    # Load existing ledger if it exists
    if output_path.exists():
        try:
            if output_path.suffix == '.parquet':
                existing_trades = pd.read_parquet(output_path)
            else:
                existing_trades = pd.read_csv(output_path)
            print(f"📋 Loaded {len(existing_trades)} existing trades")
        except Exception as e:
            print(f"⚠️ Could not load existing ledger: {e}")
            existing_trades = pd.DataFrame()
    else:
        existing_trades = pd.DataFrame()
    
    # Find new trade files
    trade_files = list(input_path.glob("**/*trades*.csv")) + list(input_path.glob("**/*trades*.parquet"))
    
    if not trade_files:
        print("❌ No trade files found in input directory")
        return False
    
    new_trades = []
    
    for trade_file in trade_files:
        try:
            print(f"📁 Processing {trade_file}")
            
            if trade_file.suffix == '.parquet':
                trades = pd.read_parquet(trade_file)
            else:
                trades = pd.read_csv(trade_file)
            
            # Ensure required columns exist
            required_columns = [
                'ts', 'strategy_name', 'side', 'qty', 'price', 'fee', 'fee_bps',
                'funding', 'slippage_bps', 'pnl_realized', 'pnl_unrealized',
                'reason_code', 'maker_flag', 'order_state', 'regime_label', 'cloid'
            ]
            
            for col in required_columns:
                if col not in trades.columns:
                    if col == 'cloid':
                        trades[col] = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    elif col == 'regime_label':
                        trades[col] = 'normal'
                    else:
                        trades[col] = 0.0
            
            new_trades.append(trades)
            print(f"✅ Loaded {len(trades)} trades from {trade_file.name}")
            
        except Exception as e:
            print(f"❌ Error processing {trade_file}: {e}")
            continue
    
    if not new_trades:
        print("❌ No valid trade data found")
        return False
    
    # Combine all new trades
    combined_new_trades = pd.concat(new_trades, ignore_index=True)
    print(f"📊 Combined {len(combined_new_trades)} new trades")
    
    # Remove duplicates based on timestamp and strategy
    if not existing_trades.empty:
        # Combine existing and new trades
        all_trades = pd.concat([existing_trades, combined_new_trades], ignore_index=True)
        
        # Remove duplicates (keep last occurrence)
        all_trades = all_trades.drop_duplicates(
            subset=['ts', 'strategy_name', 'side', 'qty', 'price'],
            keep='last'
        )
        
        print(f"🔄 Deduplicated: {len(all_trades)} total trades")
    else:
        all_trades = combined_new_trades
    
    # Sort by timestamp
    all_trades = all_trades.sort_values('ts').reset_index(drop=True)
    
    # Save updated ledger
    try:
        if output_path.suffix == '.parquet':
            all_trades.to_parquet(output_path, index=False)
        else:
            all_trades.to_csv(output_path, index=False)
        
        print(f"✅ Updated trade ledger saved to {output_path}")
        print(f"📊 Total trades: {len(all_trades)}")
        
        # Generate summary statistics
        summary = {
            'total_trades': len(all_trades),
            'total_pnl': all_trades['pnl_realized'].sum(),
            'win_rate': (all_trades['pnl_realized'] > 0).mean() * 100,
            'avg_trade': all_trades['pnl_realized'].mean(),
            'total_fees': all_trades['fee'].sum(),
            'total_funding': all_trades['funding'].sum(),
            'maker_ratio': (all_trades['maker_flag'] == True).mean() * 100,
            'last_updated': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Summary saved to {summary_path}")
        print(f"💰 Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"🎯 Win Rate: {summary['win_rate']:.1f}%")
        print(f"📈 Maker Ratio: {summary['maker_ratio']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving updated ledger: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Update trade ledger with new backtest results')
    parser.add_argument('--input-dir', required=True, help='Input directory containing trade files')
    parser.add_argument('--output', required=True, help='Output path for updated ledger')
    parser.add_argument('--format', choices=['csv', 'parquet'], default='parquet', help='Output format')
    
    args = parser.parse_args()
    
    # Ensure output has correct extension
    output_path = args.output
    if not output_path.endswith(f'.{args.format}'):
        output_path = f"{output_path}.{args.format}"
    
    success = update_trade_ledger(args.input_dir, output_path)
    
    if success:
        print("🎉 Trade ledger update completed successfully!")
        sys.exit(0)
    else:
        print("❌ Trade ledger update failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
