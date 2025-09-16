"""
📊 CANONICAL TRADE LEDGER
=========================
Production-grade trade ledger with comprehensive schema for paper + live trading.

Schema: ts, symbol, side, qty, px, fee, fee_bps, funding, slippage_bps, 
        pnl_realized, pnl_unrealized, reason_code, maker_flag, cloid, order_state
"""

import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from enum import Enum
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger

class OrderState(Enum):
    """Order state enumeration"""
    PENDING = "pending"
    PLACED = "placed"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ReasonCode(Enum):
    """Trade reason code enumeration"""
    SIGNAL_ENTRY = "signal_entry"
    SIGNAL_EXIT = "signal_exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    FUNDING_ARBITRAGE = "funding_arbitrage"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    EMERGENCY_EXIT = "emergency_exit"
    MANUAL_OVERRIDE = "manual_override"

@dataclass
class TradeRecord:
    """Canonical trade record with comprehensive schema"""
    
    # Core trade data
    ts: float                           # Timestamp (Unix)
    symbol: str                         # Trading symbol (e.g., "XRP")
    side: str                           # "buy" or "sell"
    qty: float                          # Quantity
    px: float                           # Price
    
    # Fee and cost data
    fee: float                          # Total fee paid
    fee_bps: float                      # Fee in basis points
    funding: float                      # Funding payment received/paid
    slippage_bps: float                 # Slippage in basis points
    
    # P&L data
    pnl_realized: float                 # Realized P&L
    pnl_unrealized: float               # Unrealized P&L
    
    # Order metadata
    reason_code: str                    # Reason for trade
    maker_flag: bool                    # True if maker order
    cloid: str                          # Client order ID
    order_state: str                    # Order state
    
    # Additional metadata
    order_id: Optional[str] = None      # Exchange order ID
    fill_id: Optional[str] = None       # Fill ID
    commission_asset: str = "USDC"      # Commission asset
    funding_rate: float = 0.0           # Funding rate at time of trade
    leverage: float = 1.0               # Leverage used
    margin_used: float = 0.0            # Margin used
    position_size: float = 0.0          # Position size after trade
    account_balance: float = 0.0        # Account balance after trade
    
    # Performance tracking
    latency_ms: float = 0.0             # Order latency in milliseconds
    retry_count: int = 0                # Number of retries
    error_code: Optional[str] = None    # Error code if any
    
    # Risk metrics
    var_95: float = 0.0                 # Value at Risk (95%)
    max_drawdown: float = 0.0           # Maximum drawdown
    sharpe_ratio: float = 0.0           # Sharpe ratio
    
    # Market conditions
    volatility: float = 0.0             # Market volatility
    volume_24h: float = 0.0             # 24h volume
    spread_bps: float = 0.0             # Bid-ask spread
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'ts': self.ts,
            'symbol': self.symbol,
            'side': self.side,
            'qty': self.qty,
            'px': self.px,
            'fee': self.fee,
            'fee_bps': self.fee_bps,
            'funding': self.funding,
            'slippage_bps': self.slippage_bps,
            'pnl_realized': self.pnl_realized,
            'pnl_unrealized': self.pnl_unrealized,
            'reason_code': self.reason_code,
            'maker_flag': self.maker_flag,
            'cloid': self.cloid,
            'order_state': self.order_state,
            'order_id': self.order_id,
            'fill_id': self.fill_id,
            'commission_asset': self.commission_asset,
            'funding_rate': self.funding_rate,
            'leverage': self.leverage,
            'margin_used': self.margin_used,
            'position_size': self.position_size,
            'account_balance': self.account_balance,
            'latency_ms': self.latency_ms,
            'retry_count': self.retry_count,
            'error_code': self.error_code,
            'var_95': self.var_95,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'volatility': self.volatility,
            'volume_24h': self.volume_24h,
            'spread_bps': self.spread_bps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create from dictionary"""
        return cls(**data)

class CanonicalTradeLedger:
    """
    📊 CANONICAL TRADE LEDGER
    
    Production-grade trade ledger with comprehensive schema for paper + live trading.
    Persists to CSV/Parquet under reports/ledgers/ with full observability.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or Logger()
        
        # Ledger configuration
        self.ledger_config = {
            'persist_format': 'both',  # 'csv', 'parquet', 'both'
            'persist_interval': 60,    # Persist every 60 seconds
            'max_memory_records': 10000,  # Max records in memory
            'ledger_directory': 'reports/ledgers',
            'backup_interval': 3600,   # Backup every hour
            'retention_days': 365,     # Keep records for 1 year
        }
        
        # Ledger state
        self.trade_records: List[TradeRecord] = []
        self.last_persist_time = time.time()
        self.last_backup_time = time.time()
        self.total_trades = 0
        self.total_volume = 0.0
        self.total_fees = 0.0
        self.total_pnl = 0.0
        
        # Performance tracking
        self.performance_metrics = {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'maker_ratio': 0.0,
            'avg_slippage_bps': 0.0,
            'avg_fee_bps': 0.0,
        }
        
        # Initialize ledger directory
        self._initialize_ledger_directory()
        
        self.logger.info("📊 [TRADE_LEDGER] Canonical Trade Ledger initialized")
        self.logger.info(f"📊 [TRADE_LEDGER] Ledger directory: {self.ledger_config['ledger_directory']}")
    
    def _initialize_ledger_directory(self):
        """Initialize ledger directory structure"""
        try:
            ledger_dir = Path(self.ledger_config['ledger_directory'])
            ledger_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (ledger_dir / 'csv').mkdir(exist_ok=True)
            (ledger_dir / 'parquet').mkdir(exist_ok=True)
            (ledger_dir / 'backups').mkdir(exist_ok=True)
            (ledger_dir / 'reports').mkdir(exist_ok=True)
            
            self.logger.info(f"📊 [LEDGER_DIR] Initialized ledger directory: {ledger_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ [LEDGER_DIR] Error initializing ledger directory: {e}")
    
    async def record_trade(self, trade_record: TradeRecord) -> bool:
        """
        📊 Record a trade in the canonical ledger
        
        Args:
            trade_record: TradeRecord with comprehensive trade data
            
        Returns:
            bool: True if successfully recorded
        """
        try:
            # Add to memory ledger
            self.trade_records.append(trade_record)
            
            # Update statistics
            self.total_trades += 1
            self.total_volume += trade_record.qty * trade_record.px
            self.total_fees += trade_record.fee
            self.total_pnl += trade_record.pnl_realized
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Check if we need to persist
            current_time = time.time()
            if (current_time - self.last_persist_time) >= self.ledger_config['persist_interval']:
                await self._persist_ledger()
                self.last_persist_time = current_time
            
            # Check if we need to backup
            if (current_time - self.last_backup_time) >= self.ledger_config['backup_interval']:
                await self._backup_ledger()
                self.last_backup_time = current_time
            
            # Check memory limit
            if len(self.trade_records) >= self.ledger_config['max_memory_records']:
                await self._persist_ledger()
                self.trade_records.clear()
            
            self.logger.info(f"📊 [TRADE_RECORD] Recorded trade: {trade_record.symbol} {trade_record.side} {trade_record.qty} @ {trade_record.px}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [TRADE_RECORD] Error recording trade: {e}")
            return False
    
    async def _update_performance_metrics(self):
        """Update performance metrics from trade records"""
        try:
            if not self.trade_records:
                return
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([record.to_dict() for record in self.trade_records])
            
            # Calculate performance metrics
            winning_trades = df[df['pnl_realized'] > 0]
            losing_trades = df[df['pnl_realized'] < 0]
            
            self.performance_metrics.update({
                'win_rate': len(winning_trades) / len(df) if len(df) > 0 else 0,
                'avg_win': winning_trades['pnl_realized'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl_realized'].mean() if len(losing_trades) > 0 else 0,
                'profit_factor': abs(winning_trades['pnl_realized'].sum() / losing_trades['pnl_realized'].sum()) if len(losing_trades) > 0 and losing_trades['pnl_realized'].sum() != 0 else 0,
                'maker_ratio': len(df[df['maker_flag'] == True]) / len(df) if len(df) > 0 else 0,
                'avg_slippage_bps': df['slippage_bps'].mean(),
                'avg_fee_bps': df['fee_bps'].mean(),
            })
            
            # Calculate Sharpe ratio (simplified)
            if len(df) > 1:
                returns = df['pnl_realized'].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    self.performance_metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
            
            # Calculate max drawdown
            cumulative_pnl = df['pnl_realized'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            self.performance_metrics['max_drawdown'] = drawdown.min()
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_METRICS] Error updating performance metrics: {e}")
    
    async def _persist_ledger(self):
        """Persist ledger to disk"""
        try:
            if not self.trade_records:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Convert to DataFrame
            df = pd.DataFrame([record.to_dict() for record in self.trade_records])
            
            # Persist CSV
            if self.ledger_config['persist_format'] in ['csv', 'both']:
                csv_path = Path(self.ledger_config['ledger_directory']) / 'csv' / f'trades_{timestamp}.csv'
                df.to_csv(csv_path, index=False)
                self.logger.info(f"📊 [PERSIST] Saved CSV ledger: {csv_path}")
            
            # Persist Parquet
            if self.ledger_config['persist_format'] in ['parquet', 'both']:
                parquet_path = Path(self.ledger_config['ledger_directory']) / 'parquet' / f'trades_{timestamp}.parquet'
                df.to_parquet(parquet_path, index=False)
                self.logger.info(f"📊 [PERSIST] Saved Parquet ledger: {parquet_path}")
            
            # Update master ledger
            await self._update_master_ledger(df)
            
        except Exception as e:
            self.logger.error(f"❌ [PERSIST] Error persisting ledger: {e}")
    
    async def _update_master_ledger(self, new_df: pd.DataFrame):
        """Update master ledger with new trades"""
        try:
            master_csv_path = Path(self.ledger_config['ledger_directory']) / 'master_trades.csv'
            master_parquet_path = Path(self.ledger_config['ledger_directory']) / 'master_trades.parquet'
            
            # Load existing master ledger
            if master_csv_path.exists():
                existing_df = pd.read_csv(master_csv_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Save updated master ledger
            combined_df.to_csv(master_csv_path, index=False)
            combined_df.to_parquet(master_parquet_path, index=False)
            
            self.logger.info(f"📊 [MASTER_LEDGER] Updated master ledger with {len(new_df)} new trades")
            
        except Exception as e:
            self.logger.error(f"❌ [MASTER_LEDGER] Error updating master ledger: {e}")
    
    async def _backup_ledger(self):
        """Create backup of ledger"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = Path(self.ledger_config['ledger_directory']) / 'backups' / f'backup_{timestamp}'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy master ledger to backup
            master_csv = Path(self.ledger_config['ledger_directory']) / 'master_trades.csv'
            master_parquet = Path(self.ledger_config['ledger_directory']) / 'master_trades.parquet'
            
            if master_csv.exists():
                import shutil
                shutil.copy2(master_csv, backup_dir / 'master_trades.csv')
                shutil.copy2(master_parquet, backup_dir / 'master_trades.parquet')
                
                # Save performance metrics
                with open(backup_dir / 'performance_metrics.json', 'w') as f:
                    json.dump(self.performance_metrics, f, indent=2)
                
                self.logger.info(f"📊 [BACKUP] Created backup: {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ [BACKUP] Error creating backup: {e}")
    
    async def get_trade_history(self, symbol: Optional[str] = None, 
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None) -> List[TradeRecord]:
        """Get trade history with optional filters"""
        try:
            # Load from master ledger if memory is empty
            if not self.trade_records:
                await self._load_from_master_ledger()
            
            # Filter trades
            filtered_trades = self.trade_records
            
            if symbol:
                filtered_trades = [t for t in filtered_trades if t.symbol == symbol]
            
            if start_time:
                filtered_trades = [t for t in filtered_trades if t.ts >= start_time]
            
            if end_time:
                filtered_trades = [t for t in filtered_trades if t.ts <= end_time]
            
            return filtered_trades
            
        except Exception as e:
            self.logger.error(f"❌ [TRADE_HISTORY] Error getting trade history: {e}")
            return []
    
    async def _load_from_master_ledger(self):
        """Load trades from master ledger"""
        try:
            master_csv_path = Path(self.ledger_config['ledger_directory']) / 'master_trades.csv'
            
            if master_csv_path.exists():
                df = pd.read_csv(master_csv_path)
                self.trade_records = [TradeRecord.from_dict(row.to_dict()) for _, row in df.iterrows()]
                self.logger.info(f"📊 [LOAD_MASTER] Loaded {len(self.trade_records)} trades from master ledger")
            
        except Exception as e:
            self.logger.error(f"❌ [LOAD_MASTER] Error loading from master ledger: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'total_fees': self.total_fees,
            'total_pnl': self.total_pnl,
            'performance_metrics': self.performance_metrics,
            'ledger_config': self.ledger_config,
            'last_persist_time': self.last_persist_time,
            'last_backup_time': self.last_backup_time,
        }
    
    async def generate_trade_report(self) -> Dict[str, Any]:
        """Generate comprehensive trade report"""
        try:
            if not self.trade_records:
                await self._load_from_master_ledger()
            
            if not self.trade_records:
                return {'error': 'No trades found'}
            
            # Convert to DataFrame
            df = pd.DataFrame([record.to_dict() for record in self.trade_records])
            
            # Generate report
            report = {
                'summary': {
                    'total_trades': len(df),
                    'total_volume': df['qty'].multiply(df['px']).sum(),
                    'total_fees': df['fee'].sum(),
                    'total_pnl': df['pnl_realized'].sum(),
                    'date_range': {
                        'start': datetime.fromtimestamp(df['ts'].min()).isoformat(),
                        'end': datetime.fromtimestamp(df['ts'].max()).isoformat(),
                    }
                },
                'performance': self.performance_metrics,
                'by_symbol': df.groupby('symbol').agg({
                    'qty': 'sum',
                    'px': 'mean',
                    'fee': 'sum',
                    'pnl_realized': 'sum',
                    'slippage_bps': 'mean',
                    'fee_bps': 'mean',
                }).to_dict(),
                'by_reason': df.groupby('reason_code').agg({
                    'qty': 'sum',
                    'pnl_realized': 'sum',
                    'fee': 'sum',
                }).to_dict(),
                'maker_analysis': {
                    'maker_trades': len(df[df['maker_flag'] == True]),
                    'taker_trades': len(df[df['maker_flag'] == False]),
                    'maker_ratio': len(df[df['maker_flag'] == True]) / len(df) if len(df) > 0 else 0,
                    'avg_maker_fee_bps': df[df['maker_flag'] == True]['fee_bps'].mean(),
                    'avg_taker_fee_bps': df[df['maker_flag'] == False]['fee_bps'].mean(),
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ [TRADE_REPORT] Error generating trade report: {e}")
            return {'error': str(e)}
