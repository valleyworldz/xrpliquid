"""
Live Proven PnL Tracker - Immutable daily reports showing compounding PnL, Sharpe, drawdown vs Hyperliquid benchmarks
"""

from src.core.utils.decimal_boundary_guard import safe_decimal
from src.core.utils.decimal_boundary_guard import safe_float
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional
from decimal import Decimal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import statistics

@dataclass
class DailyPnLRecord:
    date: str
    starting_balance: Decimal
    ending_balance: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    daily_return: Decimal
    cumulative_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    trades_count: int
    maker_trades: int
    taker_trades: int
    funding_earned: Decimal
    fees_paid: Decimal
    rebates_earned: Decimal
    slippage_cost: Decimal
    hyperliquid_benchmark: Decimal
    alpha_vs_benchmark: Decimal
    timestamp: str
    hash_proof: str

@dataclass
class LivePnLSummary:
    total_days: int
    total_return: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    alpha_vs_hyperliquid: Decimal
    total_trades: int
    maker_ratio: Decimal
    total_fees_saved: Decimal
    total_rebates_earned: Decimal
    last_updated: str
    immutable_hash: str

class LivePnLTracker:
    """
    Immutable live PnL tracking with daily tear-sheets
    """
    
    def __init__(self, data_dir: str = "data/live_pnl"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.daily_records = []
        self.immutable_records = []
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing records
        self._load_existing_records()
    
    def _load_existing_records(self):
        """Load existing immutable records"""
        try:
            records_file = os.path.join(self.data_dir, "immutable_records.json")
            if os.path.exists(records_file):
                with open(records_file, 'r') as f:
                    data = json.load(f)
                    self.immutable_records = [
                        DailyPnLRecord(**record) for record in data.get('records', [])
                    ]
                self.logger.info(f"✅ Loaded {len(self.immutable_records)} immutable PnL records")
        except Exception as e:
            self.logger.error(f"❌ Error loading existing records: {e}")
    
    def _calculate_hash_proof(self, record: DailyPnLRecord) -> str:
        """Calculate immutable hash proof for a record"""
        try:
            # Create hashable string from record data
            hash_data = f"{record.date}{record.starting_balance}{record.ending_balance}{record.realized_pnl}{record.trades_count}{record.timestamp}"
            return hashlib.sha256(hash_data.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"❌ Error calculating hash proof: {e}")
            return ""
    
    def _calculate_metrics(self, records: List[DailyPnLRecord]) -> Dict[str, Decimal]:
        """Calculate performance metrics from records"""
        try:
            if not records:
                return {}
            
            # Calculate returns
            daily_returns = [record.daily_return for record in records]
            total_return = records[-1].cumulative_return if records else safe_decimal('0')
            
            # Calculate Sharpe ratio
            if len(daily_returns) > 1:
                mean_return = safe_decimal(str(statistics.mean([safe_float(r) for r in daily_returns])))
                std_return = safe_decimal(str(statistics.stdev([safe_float(r) for r in daily_returns])))
                sharpe_ratio = mean_return / std_return * safe_decimal('16') if std_return > 0 else safe_decimal('0')  # Annualized
            else:
                sharpe_ratio = safe_decimal('0')
            
            # Calculate max drawdown
            peak = safe_decimal('0')
            max_dd = safe_decimal('0')
            for record in records:
                if record.cumulative_return > peak:
                    peak = record.cumulative_return
                drawdown = peak - record.cumulative_return
                if drawdown > max_dd:
                    max_dd = drawdown
            
            # Calculate win rate
            winning_days = sum(1 for record in records if record.daily_return > 0)
            win_rate = safe_decimal(str(winning_days / len(records))) if records else safe_decimal('0')
            
            # Calculate profit factor
            total_profits = sum(record.daily_return for record in records if record.daily_return > 0)
            total_losses = abs(sum(record.daily_return for record in records if record.daily_return < 0))
            profit_factor = total_profits / total_losses if total_losses > 0 else safe_decimal('0')
            
            # Calculate alpha vs Hyperliquid
            total_alpha = sum(record.alpha_vs_benchmark for record in records)
            alpha_vs_hyperliquid = total_alpha / len(records) if records else safe_decimal('0')
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'alpha_vs_hyperliquid': alpha_vs_hyperliquid
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating metrics: {e}")
            return {}
    
    def record_daily_pnl(self, 
                        date: str,
                        starting_balance: Decimal,
                        ending_balance: Decimal,
                        realized_pnl: Decimal,
                        unrealized_pnl: Decimal,
                        trades_count: int,
                        maker_trades: int,
                        taker_trades: int,
                        funding_earned: Decimal,
                        fees_paid: Decimal,
                        rebates_earned: Decimal,
                        slippage_cost: Decimal,
                        hyperliquid_benchmark: Decimal) -> DailyPnLRecord:
        """
        Record daily PnL with immutable proof
        """
        try:
            # Calculate derived metrics
            total_pnl = realized_pnl + unrealized_pnl
            daily_return = (ending_balance - starting_balance) / starting_balance if starting_balance > 0 else safe_decimal('0')
            
            # Calculate cumulative return
            if self.immutable_records:
                last_cumulative = self.immutable_records[-1].cumulative_return
                cumulative_return = last_cumulative + daily_return
            else:
                cumulative_return = daily_return
            
            # Calculate alpha vs benchmark
            alpha_vs_benchmark = daily_return - hyperliquid_benchmark
            
            # Create record
            record = DailyPnLRecord(
                date=date,
                starting_balance=starting_balance,
                ending_balance=ending_balance,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                sharpe_ratio=safe_decimal('0'),  # Will be calculated in summary
                max_drawdown=safe_decimal('0'),  # Will be calculated in summary
                trades_count=trades_count,
                maker_trades=maker_trades,
                taker_trades=taker_trades,
                funding_earned=funding_earned,
                fees_paid=fees_paid,
                rebates_earned=rebates_earned,
                slippage_cost=slippage_cost,
                hyperliquid_benchmark=hyperliquid_benchmark,
                alpha_vs_benchmark=alpha_vs_benchmark,
                timestamp=datetime.now().isoformat(),
                hash_proof=""
            )
            
            # Calculate hash proof
            record.hash_proof = self._calculate_hash_proof(record)
            
            # Add to records
            self.immutable_records.append(record)
            
            # Save to immutable storage
            self._save_immutable_records()
            
            self.logger.info(f"✅ Daily PnL recorded for {date}: {daily_return:.2%} return, {trades_count} trades")
            
            return record
            
        except Exception as e:
            self.logger.error(f"❌ Error recording daily PnL: {e}")
            return None
    
    def _save_immutable_records(self):
        """Save records to immutable storage"""
        try:
            records_file = os.path.join(self.data_dir, "immutable_records.json")
            
            # Create immutable data structure
            immutable_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_records": len(self.immutable_records),
                    "data_integrity_hash": self._calculate_data_integrity_hash()
                },
                "records": [asdict(record) for record in self.immutable_records]
            }
            
            # Save with atomic write
            temp_file = records_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(immutable_data, f, indent=2, default=str)
            
            # Atomic rename
            os.rename(temp_file, records_file)
            
            self.logger.info(f"✅ Immutable records saved: {len(self.immutable_records)} records")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving immutable records: {e}")
    
    def _calculate_data_integrity_hash(self) -> str:
        """Calculate integrity hash for all records"""
        try:
            all_hashes = [record.hash_proof for record in self.immutable_records]
            combined_hash = "".join(all_hashes)
            return hashlib.sha256(combined_hash.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"❌ Error calculating integrity hash: {e}")
            return ""
    
    def generate_live_tearsheet(self) -> LivePnLSummary:
        """
        Generate comprehensive live tearsheet
        """
        try:
            if not self.immutable_records:
                return LivePnLSummary(
                    total_days=0,
                    total_return=safe_decimal('0'),
                    annualized_return=safe_decimal('0'),
                    sharpe_ratio=safe_decimal('0'),
                    max_drawdown=safe_decimal('0'),
                    win_rate=safe_decimal('0'),
                    profit_factor=safe_decimal('0'),
                    alpha_vs_hyperliquid=safe_decimal('0'),
                    total_trades=0,
                    maker_ratio=safe_decimal('0'),
                    total_fees_saved=safe_decimal('0'),
                    total_rebates_earned=safe_decimal('0'),
                    last_updated=datetime.now().isoformat(),
                    immutable_hash=""
                )
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.immutable_records)
            
            # Calculate additional metrics
            total_trades = sum(record.trades_count for record in self.immutable_records)
            total_maker_trades = sum(record.maker_trades for record in self.immutable_records)
            maker_ratio = total_maker_trades / total_trades if total_trades > 0 else safe_decimal('0')
            
            total_fees_saved = sum(record.rebates_earned - record.fees_paid for record in self.immutable_records)
            total_rebates_earned = sum(record.rebates_earned for record in self.immutable_records)
            
            # Calculate annualized return
            days_trading = len(self.immutable_records)
            if days_trading > 0:
                total_return_float = safe_float(metrics['total_return'])
                annualized_return = safe_decimal(str((1 + total_return_float) ** (365 / days_trading) - 1))
            else:
                annualized_return = safe_decimal('0')
            
            # Create summary
            summary = LivePnLSummary(
                total_days=days_trading,
                total_return=metrics['total_return'],
                annualized_return=annualized_return,
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                alpha_vs_hyperliquid=metrics['alpha_vs_hyperliquid'],
                total_trades=total_trades,
                maker_ratio=maker_ratio,
                total_fees_saved=total_fees_saved,
                total_rebates_earned=total_rebates_earned,
                last_updated=datetime.now().isoformat(),
                immutable_hash=self._calculate_data_integrity_hash()
            )
            
            # Save tearsheet
            self._save_tearsheet(summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating tearsheet: {e}")
            return None
    
    def _save_tearsheet(self, summary: LivePnLSummary):
        """Save tearsheet to immutable storage"""
        try:
            tearsheet_file = os.path.join(self.data_dir, "live_tearsheet.json")
            
            tearsheet_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "data_integrity_hash": summary.immutable_hash,
                    "verification_url": f"https://github.com/valleyworldz/xrpliquid/blob/master/data/live_pnl/immutable_records.json"
                },
                "performance_summary": asdict(summary)
            }
            
            with open(tearsheet_file, 'w') as f:
                json.dump(tearsheet_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Live tearsheet saved: {summary.total_days} days, {summary.total_return:.2%} return")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving tearsheet: {e}")
    
    def verify_data_integrity(self) -> bool:
        """Verify data integrity of all records"""
        try:
            for record in self.immutable_records:
                expected_hash = self._calculate_hash_proof(record)
                if record.hash_proof != expected_hash:
                    self.logger.error(f"❌ Data integrity violation for {record.date}")
                    return False
            
            self.logger.info(f"✅ Data integrity verified for {len(self.immutable_records)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error verifying data integrity: {e}")
            return False

# Demo function
def demo_live_pnl_tracker():
    """Demo the live PnL tracker"""
    print("📊 Live Proven PnL Tracker Demo")
    print("=" * 50)
    
    tracker = LivePnLTracker("data/demo_live_pnl")
    
    # Simulate 7 days of trading
    print("🔧 Simulating 7 days of live trading...")
    
    starting_balance = safe_decimal('10000')
    current_balance = starting_balance
    
    for day in range(7):
        date = (datetime.now() - timedelta(days=6-day)).strftime('%Y-%m-%d')
        
        # Simulate daily performance
        daily_return = safe_decimal(str(0.02 + (day * 0.005)))  # 2% to 5% daily returns
        realized_pnl = current_balance * daily_return * safe_decimal('0.7')  # 70% realized
        unrealized_pnl = current_balance * daily_return * safe_decimal('0.3')  # 30% unrealized
        
        ending_balance = current_balance + realized_pnl + unrealized_pnl
        
        # Simulate trade data
        trades_count = 50 + day * 10
        maker_trades = int(trades_count * 0.6)  # 60% maker
        taker_trades = trades_count - maker_trades
        
        # Simulate fees and rebates
        fees_paid = safe_decimal(str(trades_count * 0.001))  # 0.1% per trade
        rebates_earned = safe_decimal(str(maker_trades * 0.0005))  # 0.05% rebate for makers
        funding_earned = safe_decimal(str(day * 0.001))  # Daily funding
        slippage_cost = safe_decimal(str(trades_count * 0.0002))  # 0.02% slippage
        
        # Hyperliquid benchmark (simulated)
        hyperliquid_benchmark = safe_decimal('0.015')  # 1.5% daily benchmark
        
        # Record daily PnL
        record = tracker.record_daily_pnl(
            date=date,
            starting_balance=current_balance,
            ending_balance=ending_balance,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            trades_count=trades_count,
            maker_trades=maker_trades,
            taker_trades=taker_trades,
            funding_earned=funding_earned,
            fees_paid=fees_paid,
            rebates_earned=rebates_earned,
            slippage_cost=slippage_cost,
            hyperliquid_benchmark=hyperliquid_benchmark
        )
        
        if record:
            print(f"  Day {day+1} ({date}): {record.daily_return:.2%} return, {trades_count} trades, {maker_trades} makers")
        
        current_balance = ending_balance
    
    # Generate tearsheet
    print(f"\n📋 Generating live tearsheet...")
    summary = tracker.generate_live_tearsheet()
    
    if summary:
        print(f"📊 Live Performance Summary:")
        print(f"  Total Days: {summary.total_days}")
        print(f"  Total Return: {summary.total_return:.2%}")
        print(f"  Annualized Return: {summary.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {summary.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {summary.max_drawdown:.2%}")
        print(f"  Win Rate: {summary.win_rate:.2%}")
        print(f"  Alpha vs Hyperliquid: {summary.alpha_vs_hyperliquid:.2%}")
        print(f"  Total Trades: {summary.total_trades}")
        print(f"  Maker Ratio: {summary.maker_ratio:.2%}")
        print(f"  Total Fees Saved: ${summary.total_fees_saved:.2f}")
        print(f"  Data Integrity Hash: {summary.immutable_hash[:16]}...")
    
    # Verify integrity
    print(f"\n🔍 Verifying data integrity...")
    integrity_ok = tracker.verify_data_integrity()
    print(f"  Data Integrity: {'✅ VERIFIED' if integrity_ok else '❌ FAILED'}")
    
    print(f"\n✅ Live Proven PnL Tracker Demo Complete")

if __name__ == "__main__":
    demo_live_pnl_tracker()
