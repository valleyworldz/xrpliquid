"""
🎯 TRADE LEDGER SYSTEM
Comprehensive trade tracking and analytics for all simulated and live trades
"""

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

@dataclass
class TradeRecord:
    """Comprehensive trade record schema"""
    # Trade Identification
    trade_id: str
    timestamp: float
    datetime_utc: str
    
    # Trade Classification
    trade_type: str  # 'BUY', 'SELL', 'SCALP', 'FUNDING_ARBITRAGE', 'GRID', 'MEAN_REVERSION'
    strategy: str    # Strategy that generated the trade
    hat_role: str    # Which specialized role executed the trade
    
    # Market Data
    symbol: str
    side: str        # 'BUY' or 'SELL'
    quantity: float
    price: float
    mark_price: float
    
    # Execution Details
    order_type: str  # 'MARKET', 'LIMIT', 'POST_ONLY', 'STOP_LIMIT'
    order_id: str
    execution_time: float
    slippage: float
    fees_paid: float
    
    # Position Management
    position_size_before: float
    position_size_after: float
    avg_entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Risk Management
    margin_used: float
    margin_ratio: float
    risk_score: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    
    # Performance Metrics
    profit_loss: float
    profit_loss_percent: float
    win_loss: str    # 'WIN', 'LOSS', 'BREAKEVEN'
    trade_duration: float
    
    # Market Conditions
    funding_rate: float
    volatility: float
    volume_24h: float
    market_regime: str
    
    # System State
    system_score: float
    confidence_score: float
    emergency_mode: bool
    cycle_count: int
    
    # Data Source
    data_source: str  # 'live_hyperliquid', 'simulated', 'backtest'
    is_live_trade: bool
    
    # Additional Metadata
    notes: str
    tags: List[str]
    metadata: Dict[str, Any]

class TradeLedgerManager:
    """
    🎯 TRADE LEDGER MANAGER
    Comprehensive trade tracking, storage, and analytics system
    """
    
    def __init__(self, data_dir: str = "data/trades", logger=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize storage paths
        self.csv_path = self.data_dir / "trade_ledger.csv"
        self.parquet_path = self.data_dir / "trade_ledger.parquet"
        self.json_path = self.data_dir / "trade_metadata.json"
        
        # Initialize data structures
        self.trades: List[TradeRecord] = []
        self.trade_counter = 0
        
        # Load existing data
        self._load_existing_data()
        
        self.logger.info("🎯 Trade Ledger Manager initialized")
    
    def _load_existing_data(self):
        """Load existing trade data from storage"""
        try:
            if self.parquet_path.exists():
                df = pd.read_parquet(self.parquet_path)
                self.trades = [self._df_row_to_trade_record(row) for _, row in df.iterrows()]
                self.trade_counter = len(self.trades)
                self.logger.info(f"📊 Loaded {len(self.trades)} existing trades from ledger")
            elif self.csv_path.exists():
                df = pd.read_csv(self.csv_path)
                self.trades = [self._df_row_to_trade_record(row) for _, row in df.iterrows()]
                self.trade_counter = len(self.trades)
                self.logger.info(f"📊 Loaded {len(self.trades)} existing trades from CSV")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load existing trade data: {e}")
            self.trades = []
            self.trade_counter = 0
    
    def _df_row_to_trade_record(self, row: pd.Series) -> TradeRecord:
        """Convert DataFrame row to TradeRecord"""
        return TradeRecord(
            trade_id=row.get('trade_id', ''),
            timestamp=row.get('timestamp', 0),
            datetime_utc=row.get('datetime_utc', ''),
            trade_type=row.get('trade_type', ''),
            strategy=row.get('strategy', ''),
            hat_role=row.get('hat_role', ''),
            symbol=row.get('symbol', ''),
            side=row.get('side', ''),
            quantity=row.get('quantity', 0),
            price=row.get('price', 0),
            mark_price=row.get('mark_price', 0),
            order_type=row.get('order_type', ''),
            order_id=row.get('order_id', ''),
            execution_time=row.get('execution_time', 0),
            slippage=row.get('slippage', 0),
            fees_paid=row.get('fees_paid', 0),
            position_size_before=row.get('position_size_before', 0),
            position_size_after=row.get('position_size_after', 0),
            avg_entry_price=row.get('avg_entry_price', 0),
            unrealized_pnl=row.get('unrealized_pnl', 0),
            realized_pnl=row.get('realized_pnl', 0),
            margin_used=row.get('margin_used', 0),
            margin_ratio=row.get('margin_ratio', 0),
            risk_score=row.get('risk_score', 0),
            stop_loss_price=row.get('stop_loss_price'),
            take_profit_price=row.get('take_profit_price'),
            profit_loss=row.get('profit_loss', 0),
            profit_loss_percent=row.get('profit_loss_percent', 0),
            win_loss=row.get('win_loss', ''),
            trade_duration=row.get('trade_duration', 0),
            funding_rate=row.get('funding_rate', 0),
            volatility=row.get('volatility', 0),
            volume_24h=row.get('volume_24h', 0),
            market_regime=row.get('market_regime', ''),
            system_score=row.get('system_score', 0),
            confidence_score=row.get('confidence_score', 0),
            emergency_mode=row.get('emergency_mode', False),
            cycle_count=row.get('cycle_count', 0),
            data_source=row.get('data_source', ''),
            is_live_trade=row.get('is_live_trade', False),
            notes=row.get('notes', ''),
            tags=json.loads(row.get('tags', '[]')),
            metadata=json.loads(row.get('metadata', '{}'))
        )
    
    def record_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Record a new trade in the ledger
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            trade_id: Unique identifier for the recorded trade
        """
        try:
            # Generate unique trade ID
            self.trade_counter += 1
            trade_id = f"TRADE_{int(time.time())}_{self.trade_counter:06d}"
            
            # Create trade record
            trade_record = TradeRecord(
                trade_id=trade_id,
                timestamp=time.time(),
                datetime_utc=datetime.now(timezone.utc).isoformat(),
                trade_type=trade_data.get('trade_type', 'UNKNOWN'),
                strategy=trade_data.get('strategy', 'UNKNOWN'),
                hat_role=trade_data.get('hat_role', 'UNKNOWN'),
                symbol=trade_data.get('symbol', 'XRP'),
                side=trade_data.get('side', 'BUY'),
                quantity=trade_data.get('quantity', 0),
                price=trade_data.get('price', 0),
                mark_price=trade_data.get('mark_price', 0),
                order_type=trade_data.get('order_type', 'MARKET'),
                order_id=trade_data.get('order_id', ''),
                execution_time=trade_data.get('execution_time', 0),
                slippage=trade_data.get('slippage', 0),
                fees_paid=trade_data.get('fees_paid', 0),
                position_size_before=trade_data.get('position_size_before', 0),
                position_size_after=trade_data.get('position_size_after', 0),
                avg_entry_price=trade_data.get('avg_entry_price', 0),
                unrealized_pnl=trade_data.get('unrealized_pnl', 0),
                realized_pnl=trade_data.get('realized_pnl', 0),
                margin_used=trade_data.get('margin_used', 0),
                margin_ratio=trade_data.get('margin_ratio', 0),
                risk_score=trade_data.get('risk_score', 0),
                stop_loss_price=trade_data.get('stop_loss_price'),
                take_profit_price=trade_data.get('take_profit_price'),
                profit_loss=trade_data.get('profit_loss', 0),
                profit_loss_percent=trade_data.get('profit_loss_percent', 0),
                win_loss=trade_data.get('win_loss', 'BREAKEVEN'),
                trade_duration=trade_data.get('trade_duration', 0),
                funding_rate=trade_data.get('funding_rate', 0),
                volatility=trade_data.get('volatility', 0),
                volume_24h=trade_data.get('volume_24h', 0),
                market_regime=trade_data.get('market_regime', 'UNKNOWN'),
                system_score=trade_data.get('system_score', 0),
                confidence_score=trade_data.get('confidence_score', 0),
                emergency_mode=trade_data.get('emergency_mode', False),
                cycle_count=trade_data.get('cycle_count', 0),
                data_source=trade_data.get('data_source', 'simulated'),
                is_live_trade=trade_data.get('is_live_trade', False),
                notes=trade_data.get('notes', ''),
                tags=trade_data.get('tags', []),
                metadata=trade_data.get('metadata', {})
            )
            
            # Add to trades list
            self.trades.append(trade_record)
            
            # Log the trade
            self.logger.info(f"📊 Trade recorded: {trade_id} - {trade_record.side} {trade_record.quantity} {trade_record.symbol} @ {trade_record.price}")
            
            return trade_id
            
        except Exception as e:
            self.logger.error(f"❌ Error recording trade: {e}")
            return ""
    
    def save_to_csv(self) -> bool:
        """Save all trades to CSV file"""
        try:
            if not self.trades:
                return True
            
            df = pd.DataFrame([asdict(trade) for trade in self.trades])
            
            # Convert complex fields to JSON strings
            df['tags'] = df['tags'].apply(json.dumps)
            df['metadata'] = df['metadata'].apply(json.dumps)
            
            df.to_csv(self.csv_path, index=False)
            self.logger.info(f"💾 Saved {len(self.trades)} trades to CSV: {self.csv_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving to CSV: {e}")
            return False
    
    def save_to_parquet(self) -> bool:
        """Save all trades to Parquet file"""
        try:
            if not self.trades:
                return True
            
            df = pd.DataFrame([asdict(trade) for trade in self.trades])
            
            # Convert complex fields to JSON strings
            df['tags'] = df['tags'].apply(json.dumps)
            df['metadata'] = df['metadata'].apply(json.dumps)
            
            df.to_parquet(self.parquet_path, index=False)
            self.logger.info(f"💾 Saved {len(self.trades)} trades to Parquet: {self.parquet_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving to Parquet: {e}")
            return False
    
    def get_trade_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive trade analytics"""
        try:
            if not self.trades:
                return {"error": "No trades recorded"}
            
            df = pd.DataFrame([asdict(trade) for trade in self.trades])
            
            # Basic statistics
            total_trades = len(df)
            live_trades = len(df[df['is_live_trade'] == True])
            simulated_trades = len(df[df['is_live_trade'] == False])
            
            # Performance metrics
            total_pnl = df['profit_loss'].sum()
            total_pnl_percent = df['profit_loss_percent'].sum()
            avg_pnl_per_trade = df['profit_loss'].mean()
            
            # Win/Loss analysis
            wins = len(df[df['win_loss'] == 'WIN'])
            losses = len(df[df['win_loss'] == 'LOSS'])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            # Strategy performance
            strategy_performance = df.groupby('strategy').agg({
                'profit_loss': ['sum', 'mean', 'count'],
                'win_loss': lambda x: (x == 'WIN').sum()
            }).round(4)
            
            # Hat role performance
            hat_performance = df.groupby('hat_role').agg({
                'profit_loss': ['sum', 'mean', 'count'],
                'win_loss': lambda x: (x == 'WIN').sum()
            }).round(4)
            
            # Time-based analysis
            df['datetime'] = pd.to_datetime(df['datetime_utc'])
            daily_pnl = df.groupby(df['datetime'].dt.date)['profit_loss'].sum()
            
            # Risk metrics
            max_drawdown = df['profit_loss'].cumsum().expanding().max() - df['profit_loss'].cumsum()
            max_drawdown = max_drawdown.max()
            
            # Market condition analysis
            market_regime_performance = df.groupby('market_regime').agg({
                'profit_loss': ['sum', 'mean', 'count'],
                'win_loss': lambda x: (x == 'WIN').sum()
            }).round(4)
            
            analytics = {
                "summary": {
                    "total_trades": total_trades,
                    "live_trades": live_trades,
                    "simulated_trades": simulated_trades,
                    "total_pnl": total_pnl,
                    "total_pnl_percent": total_pnl_percent,
                    "avg_pnl_per_trade": avg_pnl_per_trade,
                    "win_rate": win_rate,
                    "wins": wins,
                    "losses": losses,
                    "max_drawdown": max_drawdown
                },
                "strategy_performance": strategy_performance.to_dict(),
                "hat_role_performance": hat_performance.to_dict(),
                "market_regime_performance": market_regime_performance.to_dict(),
                "daily_pnl": daily_pnl.to_dict(),
                "recent_trades": df.tail(10).to_dict('records')
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"❌ Error generating analytics: {e}")
            return {"error": str(e)}
    
    def export_trades(self, format: str = "both", start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, str]:
        """
        Export trades in specified format
        
        Args:
            format: 'csv', 'parquet', or 'both'
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            Dictionary with file paths
        """
        try:
            df = pd.DataFrame([asdict(trade) for trade in self.trades])
            
            # Apply date filters if provided
            if start_date or end_date:
                df['datetime'] = pd.to_datetime(df['datetime_utc'])
                if start_date:
                    df = df[df['datetime'] >= start_date]
                if end_date:
                    df = df[df['datetime'] <= end_date]
            
            # Convert complex fields to JSON strings
            df['tags'] = df['tags'].apply(json.dumps)
            df['metadata'] = df['metadata'].apply(json.dumps)
            
            exported_files = {}
            
            if format in ['csv', 'both']:
                csv_path = self.data_dir / f"trade_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_path, index=False)
                exported_files['csv'] = str(csv_path)
            
            if format in ['parquet', 'both']:
                parquet_path = self.data_dir / f"trade_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.to_parquet(parquet_path, index=False)
                exported_files['parquet'] = str(parquet_path)
            
            self.logger.info(f"📤 Exported {len(df)} trades: {exported_files}")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting trades: {e}")
            return {"error": str(e)}
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades"""
        try:
            recent_trades = self.trades[-limit:] if self.trades else []
            return [asdict(trade) for trade in recent_trades]
        except Exception as e:
            self.logger.error(f"❌ Error getting recent trades: {e}")
            return []
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get specific trade by ID"""
        try:
            for trade in self.trades:
                if trade.trade_id == trade_id:
                    return asdict(trade)
            return None
        except Exception as e:
            self.logger.error(f"❌ Error getting trade by ID: {e}")
            return None
    
    def cleanup_old_trades(self, days_to_keep: int = 30):
        """Remove trades older than specified days"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            original_count = len(self.trades)
            self.trades = [trade for trade in self.trades if trade.timestamp > cutoff_time]
            removed_count = original_count - len(self.trades)
            
            if removed_count > 0:
                self.logger.info(f"🧹 Cleaned up {removed_count} old trades (kept {len(self.trades)})")
                self.save_to_parquet()
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up old trades: {e}")
    
    def get_ledger_summary(self) -> Dict[str, Any]:
        """Get comprehensive ledger summary"""
        try:
            analytics = self.get_trade_analytics()
            
            summary = {
                "ledger_info": {
                    "total_trades": len(self.trades),
                    "data_dir": str(self.data_dir),
                    "csv_path": str(self.csv_path),
                    "parquet_path": str(self.parquet_path),
                    "last_updated": datetime.now().isoformat()
                },
                "analytics": analytics,
                "recent_trades": self.get_recent_trades(5)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating ledger summary: {e}")
            return {"error": str(e)}
