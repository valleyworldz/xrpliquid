"""
Daily Reconciliation
Auto exchange vs ledger reconciliation with PnL taxonomy.
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyReconciliation:
    """Performs daily reconciliation between exchange and ledger."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Reconciliation tolerances
        self.tolerance_bps = 1.0  # 1 basis point tolerance
        self.tolerance_usd = 0.01  # $0.01 tolerance
        
        # PnL taxonomy categories
        self.pnl_categories = [
            'directional_pnl',
            'fees_paid',
            'fees_received',
            'funding_pnl',
            'borrow_costs',
            'rebates',
            'slippage',
            'inventory_pnl'
        ]
    
    def load_exchange_statement(self, date: str) -> Dict:
        """Load exchange statement for a given date."""
        # In real implementation, this would fetch from exchange API
        # For simulation, create sample data
        exchange_data = {
            'date': date,
            'trades': [
                {
                    'trade_id': 'trade_001',
                    'symbol': 'XRP',
                    'side': 'buy',
                    'size': 1000,
                    'price': 0.50,
                    'fee': 0.25,
                    'timestamp': f"{date}T10:30:00Z"
                },
                {
                    'trade_id': 'trade_002',
                    'symbol': 'XRP',
                    'side': 'sell',
                    'size': 500,
                    'price': 0.51,
                    'fee': 0.125,
                    'timestamp': f"{date}T14:45:00Z"
                }
            ],
            'funding_payments': [
                {
                    'payment_id': 'funding_001',
                    'symbol': 'XRP',
                    'amount': 2.50,
                    'timestamp': f"{date}T08:00:00Z"
                }
            ],
            'fees_summary': {
                'trading_fees': 0.375,
                'funding_fees': 0.0,
                'total_fees': 0.375
            },
            'pnl_summary': {
                'realized_pnl': 5.0,
                'unrealized_pnl': 0.0,
                'total_pnl': 5.0
            }
        }
        
        return exchange_data
    
    def load_ledger_data(self, date: str) -> Dict:
        """Load ledger data for a given date."""
        ledger_file = self.reports_dir / "ledgers" / f"trades_{date}.parquet"
        
        if ledger_file.exists():
            df = pd.read_parquet(ledger_file)
            
            # Convert to dictionary format
            ledger_data = {
                'date': date,
                'trades': df.to_dict('records'),
                'pnl_summary': {
                    'realized_pnl': df['pnl_realized'].sum() if 'pnl_realized' in df.columns else 0.0,
                    'unrealized_pnl': df['pnl_unrealized'].sum() if 'pnl_unrealized' in df.columns else 0.0,
                    'total_pnl': 0.0
                }
            }
            
            # Calculate total PnL
            ledger_data['pnl_summary']['total_pnl'] = (
                ledger_data['pnl_summary']['realized_pnl'] + 
                ledger_data['pnl_summary']['unrealized_pnl']
            )
            
            return ledger_data
        else:
            # Return empty data if file doesn't exist
            return {
                'date': date,
                'trades': [],
                'pnl_summary': {
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0,
                    'total_pnl': 0.0
                }
            }
    
    def reconcile_trades(self, exchange_trades: List[Dict], ledger_trades: List[Dict]) -> Dict:
        """Reconcile individual trades between exchange and ledger."""
        reconciliation = {
            'total_exchange_trades': len(exchange_trades),
            'total_ledger_trades': len(ledger_trades),
            'matched_trades': 0,
            'unmatched_trades': [],
            'discrepancies': []
        }
        
        # Create lookup dictionaries
        exchange_lookup = {trade['trade_id']: trade for trade in exchange_trades}
        ledger_lookup = {trade.get('trade_id', ''): trade for trade in ledger_trades}
        
        # Check for matches
        for trade_id, exchange_trade in exchange_lookup.items():
            if trade_id in ledger_lookup:
                ledger_trade = ledger_lookup[trade_id]
                
                # Check for discrepancies
                discrepancy = self.check_trade_discrepancy(exchange_trade, ledger_trade)
                if discrepancy:
                    reconciliation['discrepancies'].append(discrepancy)
                else:
                    reconciliation['matched_trades'] += 1
            else:
                reconciliation['unmatched_trades'].append({
                    'trade_id': trade_id,
                    'source': 'exchange_only',
                    'trade': exchange_trade
                })
        
        # Check for ledger-only trades
        for trade_id, ledger_trade in ledger_lookup.items():
            if trade_id not in exchange_lookup and trade_id:
                reconciliation['unmatched_trades'].append({
                    'trade_id': trade_id,
                    'source': 'ledger_only',
                    'trade': ledger_trade
                })
        
        return reconciliation
    
    def check_trade_discrepancy(self, exchange_trade: Dict, ledger_trade: Dict) -> Optional[Dict]:
        """Check for discrepancies between exchange and ledger trades."""
        discrepancies = []
        
        # Check price discrepancy
        exchange_price = exchange_trade.get('price', 0)
        ledger_price = ledger_trade.get('price', 0)
        price_diff = abs(exchange_price - ledger_price)
        
        if price_diff > self.tolerance_usd:
            discrepancies.append({
                'field': 'price',
                'exchange_value': exchange_price,
                'ledger_value': ledger_price,
                'difference': price_diff
            })
        
        # Check size discrepancy
        exchange_size = exchange_trade.get('size', 0)
        ledger_size = ledger_trade.get('size', 0)
        size_diff = abs(exchange_size - ledger_size)
        
        if size_diff > 0:
            discrepancies.append({
                'field': 'size',
                'exchange_value': exchange_size,
                'ledger_value': ledger_size,
                'difference': size_diff
            })
        
        # Check fee discrepancy
        exchange_fee = exchange_trade.get('fee', 0)
        ledger_fee = ledger_trade.get('fee', 0)
        fee_diff = abs(exchange_fee - ledger_fee)
        
        if fee_diff > self.tolerance_usd:
            discrepancies.append({
                'field': 'fee',
                'exchange_value': exchange_fee,
                'ledger_value': ledger_fee,
                'difference': fee_diff
            })
        
        if discrepancies:
            return {
                'trade_id': exchange_trade.get('trade_id', 'unknown'),
                'discrepancies': discrepancies
            }
        
        return None
    
    def reconcile_pnl(self, exchange_pnl: Dict, ledger_pnl: Dict) -> Dict:
        """Reconcile PnL between exchange and ledger."""
        reconciliation = {
            'exchange_pnl': exchange_pnl,
            'ledger_pnl': ledger_pnl,
            'discrepancies': [],
            'reconciliation_status': 'CLEAN'
        }
        
        # Check realized PnL
        exchange_realized = exchange_pnl.get('realized_pnl', 0)
        ledger_realized = ledger_pnl.get('realized_pnl', 0)
        realized_diff = abs(exchange_realized - ledger_realized)
        
        if realized_diff > self.tolerance_usd:
            reconciliation['discrepancies'].append({
                'field': 'realized_pnl',
                'exchange_value': exchange_realized,
                'ledger_value': ledger_realized,
                'difference': realized_diff
            })
            reconciliation['reconciliation_status'] = 'DIRTY'
        
        # Check unrealized PnL
        exchange_unrealized = exchange_pnl.get('unrealized_pnl', 0)
        ledger_unrealized = ledger_pnl.get('unrealized_pnl', 0)
        unrealized_diff = abs(exchange_unrealized - ledger_unrealized)
        
        if unrealized_diff > self.tolerance_usd:
            reconciliation['discrepancies'].append({
                'field': 'unrealized_pnl',
                'exchange_value': exchange_unrealized,
                'ledger_value': ledger_unrealized,
                'difference': unrealized_diff
            })
            reconciliation['reconciliation_status'] = 'DIRTY'
        
        # Check total PnL
        exchange_total = exchange_pnl.get('total_pnl', 0)
        ledger_total = ledger_pnl.get('total_pnl', 0)
        total_diff = abs(exchange_total - ledger_total)
        
        if total_diff > self.tolerance_usd:
            reconciliation['discrepancies'].append({
                'field': 'total_pnl',
                'exchange_value': exchange_total,
                'ledger_value': ledger_total,
                'difference': total_diff
            })
            reconciliation['reconciliation_status'] = 'DIRTY'
        
        return reconciliation
    
    def decompose_pnl(self, trades: List[Dict]) -> Dict:
        """Decompose PnL into taxonomy categories."""
        pnl_decomposition = {category: 0.0 for category in self.pnl_categories}
        
        for trade in trades:
            # Directional PnL
            if 'pnl_realized' in trade:
                pnl_decomposition['directional_pnl'] += trade['pnl_realized']
            
            # Fees
            if 'fee' in trade:
                if trade.get('side') == 'buy':
                    pnl_decomposition['fees_paid'] += trade['fee']
                else:
                    pnl_decomposition['fees_received'] += trade['fee']
            
            # Funding PnL
            if 'funding' in trade:
                pnl_decomposition['funding_pnl'] += trade['funding']
            
            # Slippage
            if 'slippage_bps' in trade:
                slippage_usd = (trade['slippage_bps'] / 10000) * trade.get('size', 0) * trade.get('price', 0)
                pnl_decomposition['slippage'] += slippage_usd
        
        return pnl_decomposition
    
    def perform_daily_reconciliation(self, date: str) -> Dict:
        """Perform complete daily reconciliation."""
        logger.info(f"🔄 Performing daily reconciliation for {date}...")
        
        # Load data
        exchange_data = self.load_exchange_statement(date)
        ledger_data = self.load_ledger_data(date)
        
        # Reconcile trades
        trade_reconciliation = self.reconcile_trades(
            exchange_data['trades'],
            ledger_data['trades']
        )
        
        # Reconcile PnL
        pnl_reconciliation = self.reconcile_pnl(
            exchange_data['pnl_summary'],
            ledger_data['pnl_summary']
        )
        
        # Decompose PnL
        pnl_decomposition = self.decompose_pnl(ledger_data['trades'])
        
        # Generate reconciliation report
        reconciliation_report = {
            'timestamp': datetime.now().isoformat(),
            'date': date,
            'trade_reconciliation': trade_reconciliation,
            'pnl_reconciliation': pnl_reconciliation,
            'pnl_decomposition': pnl_decomposition,
            'overall_status': 'CLEAN' if pnl_reconciliation['reconciliation_status'] == 'CLEAN' else 'DIRTY',
            'tolerances': {
                'tolerance_bps': self.tolerance_bps,
                'tolerance_usd': self.tolerance_usd
            }
        }
        
        return reconciliation_report
    
    def save_reconciliation_report(self, report: Dict) -> Path:
        """Save reconciliation report to file."""
        reconciliation_dir = self.reports_dir / "reconciliation"
        reconciliation_dir.mkdir(exist_ok=True)
        
        report_file = reconciliation_dir / f"exchange_vs_ledger_{report['date']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Reconciliation report saved: {report_file}")
        return report_file
    
    def generate_reconciliation_summary(self, date: str) -> Dict:
        """Generate reconciliation summary for dashboard."""
        reconciliation_file = self.reports_dir / "reconciliation" / f"exchange_vs_ledger_{date}.json"
        
        if reconciliation_file.exists():
            with open(reconciliation_file, 'r') as f:
                report = json.load(f)
            
            return {
                'date': date,
                'reconciliation_status': report['overall_status'],
                'matched_trades': report['trade_reconciliation']['matched_trades'],
                'total_trades': report['trade_reconciliation']['total_exchange_trades'],
                'pnl_discrepancies': len(report['pnl_reconciliation']['discrepancies']),
                'last_updated': report['timestamp']
            }
        else:
            return {
                'date': date,
                'reconciliation_status': 'PENDING',
                'matched_trades': 0,
                'total_trades': 0,
                'pnl_discrepancies': 0,
                'last_updated': datetime.now().isoformat()
            }


def main():
    """Main function to demonstrate daily reconciliation."""
    reconciler = DailyReconciliation()
    
    # Perform reconciliation for today
    today = datetime.now().strftime('%Y-%m-%d')
    report = reconciler.perform_daily_reconciliation(today)
    
    # Save report
    reconciler.save_reconciliation_report(report)
    
    # Generate summary
    summary = reconciler.generate_reconciliation_summary(today)
    
    print(f"Reconciliation Status: {report['overall_status']}")
    print(f"Matched Trades: {report['trade_reconciliation']['matched_trades']}")
    print(f"PnL Discrepancies: {len(report['pnl_reconciliation']['discrepancies'])}")
    
    print("✅ Daily reconciliation completed")


if __name__ == "__main__":
    main()
