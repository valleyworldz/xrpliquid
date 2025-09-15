"""
🎯 COMPREHENSIVE TRADE LEDGER SYSTEM
===================================
Canonical schema with all required fields for paper + live trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import os
import json
from pathlib import Path

@dataclass
class ComprehensiveTradeRecord:
    """Canonical trade record schema with all required fields"""
    # Core trade data
    timestamp: datetime
    strategy: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    
    # Execution quality
    expected_price: float
    fill_price: float
    slippage_bps: float
    maker_flag: bool
    queue_jump: bool = False
    
    # Costs and fees
    fee: float
    funding: float
    
    # PnL tracking
    pnl_realized: float
    pnl_unrealized: float
    
    # Trade context
    reason_code: str
    market_regime: str = "normal"
    volatility_percent: float = 0.0
    
    # Risk metrics
    position_size_usd: float = 0.0
    risk_unit_size: float = 0.0
    equity_at_risk: float = 0.0
    
    # Additional metadata
    order_id: Optional[str] = None
    exchange: str = "hyperliquid"
    symbol: str = "XRP"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class ComprehensiveTradeLedger:
    """Comprehensive trade ledger with canonical schema"""
    
    def __init__(self, data_dir: str = "data/trades"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades: List[ComprehensiveTradeRecord] = []
        self.daily_summaries: Dict[str, Dict[str, Any]] = {}
        
        # Load existing trades
        self._load_existing_trades()
    
    def _load_existing_trades(self) -> None:
        """Load existing trades from storage"""
        csv_file = self.data_dir / "comprehensive_trades.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                for _, row in df.iterrows():
                    trade = ComprehensiveTradeRecord(
                        timestamp=row['timestamp'],
                        strategy=row['strategy'],
                        side=row['side'],
                        quantity=row['quantity'],
                        price=row['price'],
                        expected_price=row['expected_price'],
                        fill_price=row['fill_price'],
                        slippage_bps=row['slippage_bps'],
                        maker_flag=row['maker_flag'],
                        queue_jump=row.get('queue_jump', False),
                        fee=row['fee'],
                        funding=row['funding'],
                        pnl_realized=row['pnl_realized'],
                        pnl_unrealized=row['pnl_unrealized'],
                        reason_code=row['reason_code'],
                        market_regime=row.get('market_regime', 'normal'),
                        volatility_percent=row.get('volatility_percent', 0.0),
                        position_size_usd=row.get('position_size_usd', 0.0),
                        risk_unit_size=row.get('risk_unit_size', 0.0),
                        equity_at_risk=row.get('equity_at_risk', 0.0),
                        order_id=row.get('order_id'),
                        exchange=row.get('exchange', 'hyperliquid'),
                        symbol=row.get('symbol', 'XRP')
                    )
                    self.trades.append(trade)
                
                print(f"✅ Loaded {len(self.trades)} existing trades")
            except Exception as e:
                print(f"⚠️ Error loading existing trades: {e}")
    
    def record_trade(self, trade: ComprehensiveTradeRecord) -> None:
        """Record a new trade"""
        self.trades.append(trade)
        
        # Auto-save every 10 trades
        if len(self.trades) % 10 == 0:
            self._save_trades()
    
    def _save_trades(self) -> None:
        """Save trades to CSV and Parquet"""
        if not self.trades:
            return
        
        # Convert to DataFrame
        trade_data = [trade.to_dict() for trade in self.trades]
        df = pd.DataFrame(trade_data)
        
        # Save as CSV
        csv_file = self.data_dir / "comprehensive_trades.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as Parquet
        parquet_file = self.data_dir / "comprehensive_trades.parquet"
        df.to_parquet(parquet_file, index=False)
        
        print(f"✅ Saved {len(self.trades)} trades to {csv_file}")
    
    def generate_daily_tearsheet(self, date: datetime) -> Dict[str, Any]:
        """Generate daily tearsheet for a specific date"""
        date_str = date.strftime('%Y-%m-%d')
        
        # Filter trades for the date
        day_trades = [
            t for t in self.trades 
            if t.timestamp.date() == date.date()
        ]
        
        if not day_trades:
            return {"date": date_str, "trades": 0}
        
        # Calculate daily metrics
        total_pnl = sum(t.pnl_realized for t in day_trades)
        total_fees = sum(t.fee for t in day_trades)
        total_funding = sum(t.funding for t in day_trades)
        
        # Execution quality
        avg_slippage = np.mean([t.slippage_bps for t in day_trades])
        maker_ratio = len([t for t in day_trades if t.maker_flag]) / len(day_trades)
        
        # Strategy breakdown
        strategy_pnl = {}
        for trade in day_trades:
            if trade.strategy not in strategy_pnl:
                strategy_pnl[trade.strategy] = 0.0
            strategy_pnl[trade.strategy] += trade.pnl_realized
        
        # Regime analysis
        regime_pnl = {}
        for trade in day_trades:
            if trade.market_regime not in regime_pnl:
                regime_pnl[trade.market_regime] = 0.0
            regime_pnl[trade.market_regime] += trade.pnl_realized
        
        tearsheet = {
            "date": date_str,
            "trades": len(day_trades),
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "total_funding": total_funding,
            "net_pnl": total_pnl - total_fees + total_funding,
            "avg_slippage_bps": avg_slippage,
            "maker_ratio": maker_ratio,
            "strategy_breakdown": strategy_pnl,
            "regime_breakdown": regime_pnl,
            "win_rate": len([t for t in day_trades if t.pnl_realized > 0]) / len(day_trades)
        }
        
        # Store daily summary
        self.daily_summaries[date_str] = tearsheet
        
        return tearsheet
    
    def generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""
        if not self.trades:
            return {"error": "No trades recorded"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([trade.to_dict() for trade in self.trades])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Overall performance
        total_pnl = df['pnl_realized'].sum()
        total_fees = df['fee'].sum()
        total_funding = df['funding'].sum()
        net_pnl = total_pnl - total_fees + total_funding
        
        # Execution quality
        avg_slippage = df['slippage_bps'].mean()
        max_slippage = df['slippage_bps'].max()
        maker_ratio = df['maker_flag'].mean()
        
        # Strategy performance
        strategy_performance = df.groupby('strategy').agg({
            'pnl_realized': ['sum', 'count', 'mean'],
            'fee': 'sum',
            'slippage_bps': 'mean'
        }).round(4)
        
        # Regime performance
        regime_performance = df.groupby('market_regime').agg({
            'pnl_realized': ['sum', 'count', 'mean'],
            'slippage_bps': 'mean'
        }).round(4)
        
        # Risk metrics
        position_sizes = df['position_size_usd'].dropna()
        risk_units = df['risk_unit_size'].dropna()
        
        # Time series analysis
        df['date'] = df['timestamp'].dt.date
        daily_pnl = df.groupby('date')['pnl_realized'].sum()
        
        # Calculate Sharpe ratio (simplified)
        if len(daily_pnl) > 1:
            sharpe_ratio = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
        else:
            sharpe_ratio = 0.0
        
        # Drawdown analysis
        cumulative_pnl = daily_pnl.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "summary": {
                "total_trades": len(df),
                "total_pnl": total_pnl,
                "total_fees": total_fees,
                "total_funding": total_funding,
                "net_pnl": net_pnl,
                "win_rate": (df['pnl_realized'] > 0).mean(),
                "avg_trade_pnl": df['pnl_realized'].mean(),
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown
            },
            "execution_quality": {
                "avg_slippage_bps": avg_slippage,
                "max_slippage_bps": max_slippage,
                "maker_ratio": maker_ratio,
                "total_fees": total_fees,
                "avg_fee_per_trade": df['fee'].mean()
            },
            "strategy_performance": strategy_performance.to_dict(),
            "regime_performance": regime_performance.to_dict(),
            "risk_metrics": {
                "avg_position_size_usd": position_sizes.mean() if len(position_sizes) > 0 else 0,
                "max_position_size_usd": position_sizes.max() if len(position_sizes) > 0 else 0,
                "avg_risk_unit_size": risk_units.mean() if len(risk_units) > 0 else 0
            }
        }
    
    def save_daily_tearsheets(self) -> None:
        """Save all daily tearsheets to reports directory"""
        if not self.daily_summaries:
            return
        
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        json_file = reports_dir / "daily_tearsheets.json"
        with open(json_file, 'w') as f:
            json.dump(self.daily_summaries, f, indent=2, default=str)
        
        # Save as CSV
        csv_file = reports_dir / "daily_tearsheets.csv"
        df = pd.DataFrame(list(self.daily_summaries.values()))
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Saved {len(self.daily_summaries)} daily tearsheets to {json_file}")
    
    def export_trade_ledger(self, format: str = "both") -> None:
        """Export trade ledger in specified format"""
        if not self.trades:
            print("⚠️ No trades to export")
            return
        
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ["csv", "both"]:
            csv_file = reports_dir / f"trade_ledger_{timestamp}.csv"
            df = pd.DataFrame([trade.to_dict() for trade in self.trades])
            df.to_csv(csv_file, index=False)
            print(f"✅ Exported trade ledger to {csv_file}")
        
        if format in ["parquet", "both"]:
            parquet_file = reports_dir / f"trade_ledger_{timestamp}.parquet"
            df = pd.DataFrame([trade.to_dict() for trade in self.trades])
            df.to_parquet(parquet_file, index=False)
            print(f"✅ Exported trade ledger to {parquet_file}")
        
        if format in ["json", "both"]:
            json_file = reports_dir / f"trade_ledger_{timestamp}.json"
            trade_data = [trade.to_dict() for trade in self.trades]
            with open(json_file, 'w') as f:
                json.dump(trade_data, f, indent=2, default=str)
            print(f"✅ Exported trade ledger to {json_file}")
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trade statistics"""
        if not self.trades:
            return {"trades": 0}
        
        df = pd.DataFrame([trade.to_dict() for trade in self.trades])
        
        return {
            "total_trades": len(df),
            "total_pnl": df['pnl_realized'].sum(),
            "total_fees": df['fee'].sum(),
            "avg_slippage_bps": df['slippage_bps'].mean(),
            "maker_ratio": df['maker_flag'].mean(),
            "win_rate": (df['pnl_realized'] > 0).mean(),
            "strategies": df['strategy'].value_counts().to_dict(),
            "regimes": df['market_regime'].value_counts().to_dict(),
            "date_range": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            }
        }
