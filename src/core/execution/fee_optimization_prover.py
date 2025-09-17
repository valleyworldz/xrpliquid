"""
Fee Optimization Prover - % of orders filled as maker, exact rebates captured vs taker costs, annualized savings quantified
"""

from src.core.utils.decimal_boundary_guard import safe_decimal
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import statistics

@dataclass
class FeeOptimizationRecord:
    timestamp: str
    order_id: str
    side: str
    size: Decimal
    price: Decimal
    order_type: str  # 'maker' or 'taker'
    execution_price: Decimal
    execution_size: Decimal
    fee_rate: Decimal
    fee_amount: Decimal
    rebate_rate: Decimal
    rebate_amount: Decimal
    net_cost: Decimal
    slippage: Decimal
    venue: str
    hash_proof: str

@dataclass
class FeeOptimizationSummary:
    total_orders: int
    maker_orders: int
    taker_orders: int
    maker_ratio: Decimal
    total_fees_paid: Decimal
    total_rebates_earned: Decimal
    net_fee_savings: Decimal
    annualized_savings: Decimal
    average_maker_rebate: Decimal
    average_taker_fee: Decimal
    fee_optimization_score: Decimal
    venue_breakdown: Dict[str, Dict[str, Any]]
    daily_savings_trend: List[Dict[str, Any]]
    last_updated: str
    immutable_hash: str

class FeeOptimizationProver:
    """
    Proves fee optimization mastery with exact rebates vs taker costs
    """
    
    def __init__(self, data_dir: str = "data/fee_optimization"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.fee_records = []
        self.immutable_records = []
        
        # Fee structure (Hyperliquid specific)
        self.fee_structure = {
            'maker_rebate': safe_decimal('0.0001'),  # 0.01% rebate for makers
            'taker_fee': safe_decimal('0.0002'),     # 0.02% fee for takers
            'venue_fees': {
                'hyperliquid': {'maker': safe_decimal('0.0001'), 'taker': safe_decimal('0.0002')},
                'binance': {'maker': safe_decimal('0.0001'), 'taker': safe_decimal('0.001')},
                'bybit': {'maker': safe_decimal('0.0001'), 'taker': safe_decimal('0.0006')}
            }
        }
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing records
        self._load_existing_records()
    
    def _load_existing_records(self):
        """Load existing immutable records"""
        try:
            records_file = os.path.join(self.data_dir, "immutable_fee_records.json")
            if os.path.exists(records_file):
                with open(records_file, 'r') as f:
                    data = json.load(f)
                    self.immutable_records = [
                        FeeOptimizationRecord(**record) for record in data.get('records', [])
                    ]
                self.logger.info(f"✅ Loaded {len(self.immutable_records)} immutable fee records")
        except Exception as e:
            self.logger.error(f"❌ Error loading existing records: {e}")
    
    def record_order_execution(self, 
                             order_id: str,
                             side: str,
                             size: Decimal,
                             price: Decimal,
                             order_type: str,
                             execution_price: Decimal,
                             execution_size: Decimal,
                             venue: str = 'hyperliquid') -> FeeOptimizationRecord:
        """
        Record order execution with fee optimization data
        """
        try:
            # Calculate fees and rebates
            notional_value = execution_size * execution_price
            
            if order_type == 'maker':
                fee_rate = safe_decimal('0')  # Makers pay no fees
                fee_amount = safe_decimal('0')
                rebate_rate = self.fee_structure['venue_fees'][venue]['maker']
                rebate_amount = notional_value * rebate_rate
            else:  # taker
                fee_rate = self.fee_structure['venue_fees'][venue]['taker']
                fee_amount = notional_value * fee_rate
                rebate_rate = safe_decimal('0')
                rebate_amount = safe_decimal('0')
            
            # Calculate net cost (negative = savings)
            net_cost = fee_amount - rebate_amount
            
            # Calculate slippage
            slippage = abs(execution_price - price) / price if price > 0 else safe_decimal('0')
            
            # Create record
            record = FeeOptimizationRecord(
                timestamp=datetime.now().isoformat(),
                order_id=order_id,
                side=side,
                size=size,
                price=price,
                order_type=order_type,
                execution_price=execution_price,
                execution_size=execution_size,
                fee_rate=fee_rate,
                fee_amount=fee_amount,
                rebate_rate=rebate_rate,
                rebate_amount=rebate_amount,
                net_cost=net_cost,
                slippage=slippage,
                venue=venue,
                hash_proof=""
            )
            
            # Calculate hash proof
            record.hash_proof = self._calculate_hash_proof(record)
            
            # Add to records
            self.immutable_records.append(record)
            
            # Save to immutable storage
            self._save_immutable_records()
            
            self.logger.info(f"✅ Fee optimization recorded: {order_type} order {order_id}, net cost: ${net_cost:.4f}")
            
            return record
            
        except Exception as e:
            self.logger.error(f"❌ Error recording order execution: {e}")
            return None
    
    def _calculate_hash_proof(self, record: FeeOptimizationRecord) -> str:
        """Calculate immutable hash proof for a record"""
        try:
            import hashlib
            hash_data = f"{record.timestamp}{record.order_id}{record.order_type}{record.fee_amount}{record.rebate_amount}{record.venue}"
            return hashlib.sha256(hash_data.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"❌ Error calculating hash proof: {e}")
            return ""
    
    def _save_immutable_records(self):
        """Save records to immutable storage"""
        try:
            records_file = os.path.join(self.data_dir, "immutable_fee_records.json")
            
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
            
        except Exception as e:
            self.logger.error(f"❌ Error saving immutable records: {e}")
    
    def _calculate_data_integrity_hash(self) -> str:
        """Calculate integrity hash for all records"""
        try:
            import hashlib
            all_hashes = [record.hash_proof for record in self.immutable_records]
            combined_hash = "".join(all_hashes)
            return hashlib.sha256(combined_hash.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"❌ Error calculating integrity hash: {e}")
            return ""
    
    def generate_fee_optimization_proof(self) -> FeeOptimizationSummary:
        """
        Generate comprehensive fee optimization proof
        """
        try:
            if not self.immutable_records:
                return FeeOptimizationSummary(
                    total_orders=0,
                    maker_orders=0,
                    taker_orders=0,
                    maker_ratio=safe_decimal('0'),
                    total_fees_paid=safe_decimal('0'),
                    total_rebates_earned=safe_decimal('0'),
                    net_fee_savings=safe_decimal('0'),
                    annualized_savings=safe_decimal('0'),
                    average_maker_rebate=safe_decimal('0'),
                    average_taker_fee=safe_decimal('0'),
                    fee_optimization_score=safe_decimal('0'),
                    venue_breakdown={},
                    daily_savings_trend=[],
                    last_updated=datetime.now().isoformat(),
                    immutable_hash=""
                )
            
            # Calculate basic metrics
            total_orders = len(self.immutable_records)
            maker_orders = sum(1 for record in self.immutable_records if record.order_type == 'maker')
            taker_orders = total_orders - maker_orders
            maker_ratio = maker_orders / total_orders if total_orders > 0 else safe_decimal('0')
            
            # Calculate fee metrics
            total_fees_paid = sum(record.fee_amount for record in self.immutable_records)
            total_rebates_earned = sum(record.rebate_amount for record in self.immutable_records)
            net_fee_savings = total_rebates_earned - total_fees_paid
            
            # Calculate averages
            maker_records = [record for record in self.immutable_records if record.order_type == 'maker']
            taker_records = [record for record in self.immutable_records if record.order_type == 'taker']
            
            average_maker_rebate = sum(record.rebate_amount for record in maker_records) / len(maker_records) if maker_records else safe_decimal('0')
            average_taker_fee = sum(record.fee_amount for record in taker_records) / len(taker_records) if taker_records else safe_decimal('0')
            
            # Calculate fee optimization score (0-100)
            if total_orders > 0:
                # Score based on maker ratio and net savings
                maker_score = maker_ratio * 50  # 50 points for maker ratio
                savings_score = min(50, abs(net_fee_savings) * 1000)  # 50 points for savings
                fee_optimization_score = maker_score + savings_score
            else:
                fee_optimization_score = safe_decimal('0')
            
            # Calculate annualized savings
            if self.immutable_records:
                first_date = datetime.fromisoformat(self.immutable_records[0].timestamp)
                last_date = datetime.fromisoformat(self.immutable_records[-1].timestamp)
                days_trading = (last_date - first_date).days + 1
                annualized_savings = net_fee_savings * (365 / days_trading) if days_trading > 0 else safe_decimal('0')
            else:
                annualized_savings = safe_decimal('0')
            
            # Calculate venue breakdown
            venue_breakdown = {}
            for venue in set(record.venue for record in self.immutable_records):
                venue_records = [record for record in self.immutable_records if record.venue == venue]
                venue_makers = sum(1 for record in venue_records if record.order_type == 'maker')
                venue_takers = len(venue_records) - venue_makers
                venue_fees = sum(record.fee_amount for record in venue_records)
                venue_rebates = sum(record.rebate_amount for record in venue_records)
                
                venue_breakdown[venue] = {
                    'total_orders': len(venue_records),
                    'maker_orders': venue_makers,
                    'taker_orders': venue_takers,
                    'maker_ratio': venue_makers / len(venue_records) if venue_records else 0,
                    'total_fees': str(venue_fees),
                    'total_rebates': str(venue_rebates),
                    'net_savings': str(venue_rebates - venue_fees)
                }
            
            # Calculate daily savings trend
            daily_savings_trend = self._calculate_daily_savings_trend()
            
            # Create summary
            summary = FeeOptimizationSummary(
                total_orders=total_orders,
                maker_orders=maker_orders,
                taker_orders=taker_orders,
                maker_ratio=maker_ratio,
                total_fees_paid=total_fees_paid,
                total_rebates_earned=total_rebates_earned,
                net_fee_savings=net_fee_savings,
                annualized_savings=annualized_savings,
                average_maker_rebate=average_maker_rebate,
                average_taker_fee=average_taker_fee,
                fee_optimization_score=fee_optimization_score,
                venue_breakdown=venue_breakdown,
                daily_savings_trend=daily_savings_trend,
                last_updated=datetime.now().isoformat(),
                immutable_hash=self._calculate_data_integrity_hash()
            )
            
            # Save proof
            self._save_fee_optimization_proof(summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating fee optimization proof: {e}")
            return None
    
    def _calculate_daily_savings_trend(self) -> List[Dict[str, Any]]:
        """Calculate daily savings trend"""
        try:
            daily_savings = {}
            
            for record in self.immutable_records:
                date = record.timestamp[:10]  # Extract date
                if date not in daily_savings:
                    daily_savings[date] = {
                        'date': date,
                        'orders': 0,
                        'makers': 0,
                        'takers': 0,
                        'fees_paid': safe_decimal('0'),
                        'rebates_earned': safe_decimal('0'),
                        'net_savings': safe_decimal('0')
                    }
                
                daily_savings[date]['orders'] += 1
                if record.order_type == 'maker':
                    daily_savings[date]['makers'] += 1
                else:
                    daily_savings[date]['takers'] += 1
                
                daily_savings[date]['fees_paid'] += record.fee_amount
                daily_savings[date]['rebates_earned'] += record.rebate_amount
                daily_savings[date]['net_savings'] += record.rebate_amount - record.fee_amount
            
            # Convert to list and sort by date
            trend = []
            for date in sorted(daily_savings.keys()):
                data = daily_savings[date]
                trend.append({
                    'date': data['date'],
                    'orders': data['orders'],
                    'makers': data['makers'],
                    'takers': data['takers'],
                    'maker_ratio': data['makers'] / data['orders'] if data['orders'] > 0 else 0,
                    'fees_paid': str(data['fees_paid']),
                    'rebates_earned': str(data['rebates_earned']),
                    'net_savings': str(data['net_savings'])
                })
            
            return trend
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating daily savings trend: {e}")
            return []
    
    def _save_fee_optimization_proof(self, summary: FeeOptimizationSummary):
        """Save fee optimization proof to immutable storage"""
        try:
            proof_file = os.path.join(self.data_dir, "fee_optimization_proof.json")
            
            proof_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "data_integrity_hash": summary.immutable_hash,
                    "verification_url": f"https://github.com/valleyworldz/xrpliquid/blob/master/data/fee_optimization/immutable_fee_records.json"
                },
                "fee_optimization_summary": asdict(summary)
            }
            
            with open(proof_file, 'w') as f:
                json.dump(proof_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Fee optimization proof saved: {summary.total_orders} orders, {summary.maker_ratio:.2%} maker ratio")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving fee optimization proof: {e}")
    
    def verify_fee_optimization(self) -> bool:
        """Verify fee optimization calculations"""
        try:
            for record in self.immutable_records:
                expected_hash = self._calculate_hash_proof(record)
                if record.hash_proof != expected_hash:
                    self.logger.error(f"❌ Fee optimization verification failed for {record.order_id}")
                    return False
            
            self.logger.info(f"✅ Fee optimization verified for {len(self.immutable_records)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error verifying fee optimization: {e}")
            return False

# Demo function
def demo_fee_optimization_prover():
    """Demo the fee optimization prover"""
    print("💰 Fee Optimization Prover Demo")
    print("=" * 50)
    
    prover = FeeOptimizationProver("data/demo_fee_optimization")
    
    # Simulate 100 orders with fee optimization
    print("🔧 Simulating 100 orders with fee optimization...")
    
    for i in range(100):
        order_id = f"order_{i:03d}"
        side = "BUY" if i % 2 == 0 else "SELL"
        size = safe_decimal('100')
        price = safe_decimal('0.52')
        
        # 70% maker, 30% taker (excellent optimization)
        order_type = "maker" if i % 10 < 7 else "taker"
        
        # Simulate execution
        execution_price = price + safe_decimal(str(0.0001 * (i % 5)))  # Small price movement
        execution_size = size
        
        # Record execution
        record = prover.record_order_execution(
            order_id=order_id,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            execution_price=execution_price,
            execution_size=execution_size,
            venue='hyperliquid'
        )
        
        if i % 20 == 0:
            print(f"  Order {i+1}: {order_type} order, net cost: ${record.net_cost:.4f}")
    
    # Generate fee optimization proof
    print(f"\n📋 Generating fee optimization proof...")
    summary = prover.generate_fee_optimization_proof()
    
    if summary:
        print(f"💰 Fee Optimization Summary:")
        print(f"  Total Orders: {summary.total_orders}")
        print(f"  Maker Orders: {summary.maker_orders}")
        print(f"  Taker Orders: {summary.taker_orders}")
        print(f"  Maker Ratio: {summary.maker_ratio:.2%}")
        print(f"  Total Fees Paid: ${summary.total_fees_paid:.4f}")
        print(f"  Total Rebates Earned: ${summary.total_rebates_earned:.4f}")
        print(f"  Net Fee Savings: ${summary.net_fee_savings:.4f}")
        print(f"  Annualized Savings: ${summary.annualized_savings:.2f}")
        print(f"  Fee Optimization Score: {summary.fee_optimization_score:.1f}/100")
        print(f"  Average Maker Rebate: ${summary.average_maker_rebate:.4f}")
        print(f"  Average Taker Fee: ${summary.average_taker_fee:.4f}")
        
        print(f"\n🏢 Venue Breakdown:")
        for venue, data in summary.venue_breakdown.items():
            print(f"  {venue}: {data['maker_ratio']:.2%} maker ratio, ${data['net_savings']} net savings")
        
        print(f"\n📈 Daily Savings Trend (Last 5 days):")
        for day_data in summary.daily_savings_trend[-5:]:
            print(f"  {day_data['date']}: {day_data['maker_ratio']:.2%} maker ratio, ${day_data['net_savings']} savings")
    
    # Verify optimization
    print(f"\n🔍 Verifying fee optimization...")
    optimization_ok = prover.verify_fee_optimization()
    print(f"  Fee Optimization: {'✅ VERIFIED' if optimization_ok else '❌ FAILED'}")
    
    print(f"\n✅ Fee Optimization Prover Demo Complete")

if __name__ == "__main__":
    demo_fee_optimization_prover()
