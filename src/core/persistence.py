#!/usr/bin/env python3
"""
Persistence Module
=================

This module handles CSV/DB logging with async-safe I/O operations.
Offloads CSV/DB writes with asyncio.to_thread to prevent blocking the event loop.
"""

import asyncio
import csv
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict, is_dataclass

from src.core.state import RuntimeState


class AsyncPersistenceManager:
    """Async-safe persistence manager for trade logging and state management"""
    
    def __init__(self, data_dir: str = "data", logger: Optional[logging.Logger] = None):
        self.data_dir = data_dir
        self.logger = logger or logging.getLogger(__name__)
        self.csv_lock = threading.Lock()
        self.db_lock = threading.Lock()
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "trade_logs"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "state_backups"), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for trade logging"""
        db_path = os.path.join(self.data_dir, "trading_history.db")
        
        with self.db_lock:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    trade_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    pnl REAL,
                    fees REAL,
                    duration_seconds REAL,
                    signal_confidence REAL,
                    risk_reward_ratio REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    daily_trades INTEGER NOT NULL,
                    consecutive_losses INTEGER NOT NULL,
                    max_drawdown REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create state snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    state_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
    async def log_trade_async(self, trade_data: Dict[str, Any]) -> bool:
        """Log trade data asynchronously"""
        try:
            return await asyncio.to_thread(self._log_trade_sync, trade_data)
        except Exception as e:
            self.logger.error(f"❌ Error logging trade asynchronously: {e}")
            return False
            
    def _log_trade_sync(self, trade_data: Dict[str, Any]) -> bool:
        """Synchronous trade logging"""
        try:
            # Log to CSV
            csv_success = self._write_trade_to_csv(trade_data)
            
            # Log to database
            db_success = self._write_trade_to_db(trade_data)
            
            return csv_success and db_success
            
        except Exception as e:
            self.logger.error(f"❌ Error in synchronous trade logging: {e}")
            return False
            
    def _write_trade_to_csv(self, trade_data: Dict[str, Any]) -> bool:
        """Write trade data to CSV file"""
        try:
            csv_file = os.path.join(self.data_dir, "trade_logs", "trades.csv")
            
            # Define fieldnames to ensure consistent schema
            fieldnames = [
                'timestamp', 'trade_type', 'symbol', 'side', 'size', 'entry_price',
                'exit_price', 'pnl', 'fees', 'duration_seconds', 'signal_confidence',
                'risk_reward_ratio', 'atr_pct', 'funding_rate', 'margin_ratio'
            ]
            
            with self.csv_lock:
                file_exists = os.path.exists(csv_file)
                
                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                        
                    # Normalize trade data to match schema
                    normalized_data = {}
                    for field in fieldnames:
                        normalized_data[field] = trade_data.get(field, '')
                        
                    writer.writerow(normalized_data)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error writing to CSV: {e}")
            return False
            
    def _write_trade_to_db(self, trade_data: Dict[str, Any]) -> bool:
        """Write trade data to SQLite database"""
        try:
            db_path = os.path.join(self.data_dir, "trading_history.db")
            
            with self.db_lock:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trades (
                        timestamp, trade_type, symbol, side, size, entry_price,
                        exit_price, pnl, fees, duration_seconds, signal_confidence,
                        risk_reward_ratio, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('timestamp', time.time()),
                    trade_data.get('trade_type', ''),
                    trade_data.get('symbol', 'XRP'),
                    trade_data.get('side', ''),
                    trade_data.get('size', 0),
                    trade_data.get('entry_price', 0.0),
                    trade_data.get('exit_price'),
                    trade_data.get('pnl'),
                    trade_data.get('fees', 0.0),
                    trade_data.get('duration_seconds'),
                    trade_data.get('signal_confidence'),
                    trade_data.get('risk_reward_ratio'),
                    json.dumps(trade_data.get('metadata', {}))
                ))
                
                conn.commit()
                conn.close()
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error writing to database: {e}")
            return False
            
    async def log_performance_async(self, state: RuntimeState) -> bool:
        """Log performance metrics asynchronously"""
        try:
            return await asyncio.to_thread(self._log_performance_sync, state)
        except Exception as e:
            self.logger.error(f"❌ Error logging performance asynchronously: {e}")
            return False
            
    def _log_performance_sync(self, state: RuntimeState) -> bool:
        """Synchronous performance logging"""
        try:
            db_path = os.path.join(self.data_dir, "trading_history.db")
            
            with self.db_lock:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance (
                        timestamp, total_pnl, daily_pnl, win_rate, total_trades,
                        daily_trades, consecutive_losses, max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    time.time(),
                    state.total_pnl,
                    state.daily_pnl,
                    state.win_rate,
                    state.total_trades,
                    state.daily_trades,
                    state.consecutive_losses,
                    state.max_drawdown
                ))
                
                conn.commit()
                conn.close()
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error logging performance: {e}")
            return False
            
    async def save_state_snapshot_async(self, state: RuntimeState) -> bool:
        """Save state snapshot asynchronously"""
        try:
            return await asyncio.to_thread(self._save_state_snapshot_sync, state)
        except Exception as e:
            self.logger.error(f"❌ Error saving state snapshot asynchronously: {e}")
            return False
            
    def _save_state_snapshot_sync(self, state: RuntimeState) -> bool:
        """Synchronous state snapshot saving"""
        try:
            # Convert state to dict
            state_dict = state.to_dict()
            state_dict['timestamp'] = time.time()
            
            # Save to database
            db_path = os.path.join(self.data_dir, "trading_history.db")
            
            with self.db_lock:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO state_snapshots (timestamp, state_data)
                    VALUES (?, ?)
                ''', (time.time(), json.dumps(state_dict)))
                
                conn.commit()
                conn.close()
                
            # Save to JSON file as backup
            json_file = os.path.join(
                self.data_dir, 
                "state_backups", 
                f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(json_file, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving state snapshot: {e}")
            return False
            
    async def load_latest_state_async(self) -> Optional[Dict[str, Any]]:
        """Load latest state snapshot asynchronously"""
        try:
            return await asyncio.to_thread(self._load_latest_state_sync)
        except Exception as e:
            self.logger.error(f"❌ Error loading state asynchronously: {e}")
            return None
            
    def _load_latest_state_sync(self) -> Optional[Dict[str, Any]]:
        """Synchronous state loading"""
        try:
            db_path = os.path.join(self.data_dir, "trading_history.db")
            
            with self.db_lock:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT state_data FROM state_snapshots 
                    ORDER BY timestamp DESC LIMIT 1
                ''')
                
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    return json.loads(result[0])
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error loading state: {e}")
            return None
            
    async def get_trade_history_async(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history asynchronously"""
        try:
            return await asyncio.to_thread(self._get_trade_history_sync, limit)
        except Exception as e:
            self.logger.error(f"❌ Error getting trade history asynchronously: {e}")
            return []
            
    def _get_trade_history_sync(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Synchronous trade history retrieval"""
        try:
            db_path = os.path.join(self.data_dir, "trading_history.db")
            
            with self.db_lock:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                conn.close()
                
                trades = []
                for row in rows:
                    trade_dict = dict(zip(columns, row))
                    # Parse metadata JSON
                    if trade_dict.get('metadata'):
                        try:
                            trade_dict['metadata'] = json.loads(trade_dict['metadata'])
                        except:
                            trade_dict['metadata'] = {}
                    trades.append(trade_dict)
                    
                return trades
                
        except Exception as e:
            self.logger.error(f"❌ Error getting trade history: {e}")
            return []
            
    async def cleanup_old_data_async(self, days_to_keep: int = 30) -> bool:
        """Clean up old data asynchronously"""
        try:
            return await asyncio.to_thread(self._cleanup_old_data_sync, days_to_keep)
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up old data asynchronously: {e}")
            return False
            
    def _cleanup_old_data_sync(self, days_to_keep: int = 30) -> bool:
        """Synchronous data cleanup"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            db_path = os.path.join(self.data_dir, "trading_history.db")
            
            with self.db_lock:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Clean up old state snapshots
                cursor.execute('''
                    DELETE FROM state_snapshots 
                    WHERE timestamp < ?
                ''', (cutoff_time,))
                
                # Clean up old performance data
                cursor.execute('''
                    DELETE FROM performance 
                    WHERE timestamp < ?
                ''', (cutoff_time,))
                
                conn.commit()
                conn.close()
                
            # Clean up old JSON files
            backup_dir = os.path.join(self.data_dir, "state_backups")
            for filename in os.listdir(backup_dir):
                file_path = os.path.join(backup_dir, filename)
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up old data: {e}")
            return False 