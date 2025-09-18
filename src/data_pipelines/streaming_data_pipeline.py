#!/usr/bin/env python3
"""
🔄 STREAMING DATA PIPELINE
==========================
High-performance data pipeline for processing, transforming, and storing real-time market data.

Features:
- Real-time data ingestion
- Data transformation and enrichment
- Time-series database storage
- Data quality validation
- Compression and archival
- Real-time analytics
"""

import asyncio
import time
import json
import logging
import threading
import queue
import sqlite3
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pickle
import gzip
import csv

from ..streaming.high_frequency_data_engine import MarketTick, OrderBookSnapshot, TradeData, DataType, ExchangeType
from ..streaming.market_data_feed_manager import MarketDepthAnalysis, LiquidityMetrics, CrossExchangeArbitrage

@dataclass
class ProcessedTick:
    """Processed and enriched market tick"""
    original_tick: MarketTick
    enriched_data: Dict[str, Any]
    quality_score: float  # 0-1 data quality score
    processing_timestamp: float
    storage_key: str

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    total_ticks: int
    valid_ticks: int
    invalid_ticks: int
    duplicate_ticks: int
    out_of_order_ticks: int
    latency_violations: int
    quality_score: float
    last_updated: float

class StreamingDataPipeline:
    """
    🔄 STREAMING DATA PIPELINE
    Processes and stores high-frequency market data with institutional quality
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Database and storage
        self.db_path = config.get('database_path', 'data/streaming/market_data.db')
        self.archive_path = config.get('archive_path', 'data/streaming/archive')
        self.db_connection = None
        
        # Processing queues
        self.ingestion_queue = queue.Queue(maxsize=100000)
        self.processing_queue = queue.Queue(maxsize=50000)
        self.storage_queue = queue.Queue(maxsize=25000)
        
        # Processing threads
        self.running = False
        self.ingestion_thread = None
        self.processing_threads = []
        self.storage_thread = None
        self.archival_thread = None
        
        # Data validation and quality
        self.quality_metrics: Dict[str, DataQualityMetrics] = {}
        self.duplicate_detection: Dict[str, set] = defaultdict(set)  # Sliding window for duplicates
        self.sequence_tracking: Dict[str, int] = {}  # Track sequence numbers
        
        # Performance tracking
        self.processed_count = 0
        self.stored_count = 0
        self.processing_latency: deque = deque(maxlen=1000)
        self.storage_latency: deque = deque(maxlen=1000)
        
        # Configuration parameters
        self.max_processing_threads = config.get('max_processing_threads', 4)
        self.batch_size = config.get('batch_size', 100)
        self.archival_days = config.get('archival_days', 7)
        self.quality_threshold = config.get('quality_threshold', 0.8)
        
        # Real-time analytics
        self.analytics_callbacks: List[Callable] = []
        self.real_time_aggregates: Dict[str, Any] = {}
        
        self.logger.info("🔄 [PIPELINE] Streaming Data Pipeline initialized")

    async def start_pipeline(self):
        """Start the data processing pipeline"""
        try:
            self.running = True
            
            # Initialize database
            await self._initialize_database()
            
            # Start processing threads
            self.ingestion_thread = threading.Thread(target=self._ingestion_worker, daemon=True)
            self.ingestion_thread.start()
            
            for i in range(self.max_processing_threads):
                thread = threading.Thread(target=self._processing_worker, daemon=True)
                thread.start()
                self.processing_threads.append(thread)
            
            self.storage_thread = threading.Thread(target=self._storage_worker, daemon=True)
            self.storage_thread.start()
            
            self.archival_thread = threading.Thread(target=self._archival_worker, daemon=True)
            self.archival_thread.start()
            
            # Start monitoring
            asyncio.create_task(self._monitor_pipeline_performance())
            
            self.logger.info(f"🔄 [PIPELINE] Started with {self.max_processing_threads} processing threads")
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Error starting pipeline: {e}")

    async def _initialize_database(self):
        """Initialize time-series database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Create tables for different data types
            await self._create_tables()
            
            self.logger.info("🔄 [PIPELINE] Database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Database initialization error: {e}")

    async def _create_tables(self):
        """Create database tables for market data"""
        try:
            cursor = self.db_connection.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL,
                    trade_id TEXT,
                    is_taker BOOLEAN,
                    latency_us REAL,
                    quality_score REAL,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Order book snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    sequence_id INTEGER,
                    bids_json TEXT,
                    asks_json TEXT,
                    spread REAL,
                    mid_price REAL,
                    latency_us REAL,
                    quality_score REAL,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Market depth analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_depth_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    bid_depth_5 REAL,
                    ask_depth_5 REAL,
                    bid_depth_10 REAL,
                    ask_depth_10 REAL,
                    imbalance_5 REAL,
                    imbalance_10 REAL,
                    spread_bps REAL,
                    mid_price REAL,
                    volatility_1min REAL,
                    volume_1min REAL,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Arbitrage opportunities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    buy_exchange TEXT NOT NULL,
                    sell_exchange TEXT NOT NULL,
                    buy_price REAL NOT NULL,
                    sell_price REAL NOT NULL,
                    profit_bps REAL NOT NULL,
                    available_size REAL,
                    confidence REAL,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Data quality metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    total_ticks INTEGER,
                    valid_ticks INTEGER,
                    invalid_ticks INTEGER,
                    duplicate_ticks INTEGER,
                    out_of_order_ticks INTEGER,
                    latency_violations INTEGER,
                    quality_score REAL,
                    created_at REAL DEFAULT (julianday('now'))
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_timestamp ON orderbook_snapshots(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_depth_symbol_timestamp ON market_depth_analysis(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_arbitrage_timestamp ON arbitrage_opportunities(timestamp)')
            
            self.db_connection.commit()
            
            self.logger.info("🔄 [PIPELINE] Database tables created")
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Error creating tables: {e}")

    def ingest_tick(self, tick: MarketTick):
        """Ingest a market tick into the pipeline"""
        try:
            if not self.ingestion_queue.full():
                self.ingestion_queue.put(tick)
            else:
                self.logger.warning("⚠️ [PIPELINE] Ingestion queue full - dropping tick")
                
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Error ingesting tick: {e}")

    def _ingestion_worker(self):
        """Worker thread for initial tick ingestion and validation"""
        while self.running:
            try:
                tick = self.ingestion_queue.get(timeout=1.0)
                
                # Basic validation
                if self._validate_tick(tick):
                    # Check for duplicates
                    if not self._is_duplicate(tick):
                        # Add to processing queue
                        if not self.processing_queue.full():
                            self.processing_queue.put(tick)
                        else:
                            self.logger.warning("⚠️ [PIPELINE] Processing queue full")
                
                self.ingestion_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ [PIPELINE] Ingestion worker error: {e}")

    def _processing_worker(self):
        """Worker thread for tick processing and enrichment"""
        while self.running:
            try:
                tick = self.processing_queue.get(timeout=1.0)
                
                processing_start = time.time()
                
                # Process and enrich the tick
                processed_tick = self._process_tick(tick)
                
                # Calculate processing latency
                processing_latency = (time.time() - processing_start) * 1_000_000  # microseconds
                self.processing_latency.append(processing_latency)
                
                # Add to storage queue
                if processed_tick and not self.storage_queue.full():
                    self.storage_queue.put(processed_tick)
                
                self.processed_count += 1
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ [PIPELINE] Processing worker error: {e}")

    def _storage_worker(self):
        """Worker thread for batch storage of processed ticks"""
        batch = []
        last_storage = time.time()
        
        while self.running:
            try:
                # Collect batch or timeout
                try:
                    tick = self.storage_queue.get(timeout=0.1)
                    batch.append(tick)
                    self.storage_queue.task_done()
                except queue.Empty:
                    pass
                
                # Store batch if full or timeout
                current_time = time.time()
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_storage > 1.0)):  # 1 second timeout
                    
                    storage_start = time.time()
                    self._store_batch(batch)
                    storage_latency = (time.time() - storage_start) * 1000  # milliseconds
                    self.storage_latency.append(storage_latency)
                    
                    self.stored_count += len(batch)
                    batch.clear()
                    last_storage = current_time
                
            except Exception as e:
                self.logger.error(f"❌ [PIPELINE] Storage worker error: {e}")

    def _archival_worker(self):
        """Worker thread for data archival and cleanup"""
        while self.running:
            try:
                # Run archival every hour
                time.sleep(3600)
                
                # Archive old data
                cutoff_time = time.time() - (self.archival_days * 86400)
                await self._archive_old_data(cutoff_time)
                
                # Update data quality metrics
                await self._update_quality_metrics()
                
            except Exception as e:
                self.logger.error(f"❌ [PIPELINE] Archival worker error: {e}")

    def _validate_tick(self, tick: MarketTick) -> bool:
        """Validate incoming tick data"""
        try:
            # Basic validation
            if not tick.symbol or not tick.exchange:
                return False
            
            if tick.timestamp <= 0:
                return False
            
            # Data type specific validation
            if tick.data_type == DataType.TRADE:
                trade_data = TradeData(**tick.data)
                if trade_data.price <= 0 or trade_data.size <= 0:
                    return False
            
            elif tick.data_type == DataType.ORDER_BOOK:
                # Basic order book validation
                orderbook_data = tick.data
                if not orderbook_data.get('bids') and not orderbook_data.get('asks'):
                    return False
            
            # Check for reasonable timestamp (not too far in future/past)
            current_time = time.time()
            if abs(tick.timestamp - current_time) > 300:  # 5 minutes
                self.logger.warning(f"⚠️ [PIPELINE] Tick timestamp out of range: {tick.timestamp}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Validation error: {e}")
            return False

    def _is_duplicate(self, tick: MarketTick) -> bool:
        """Check if tick is a duplicate"""
        try:
            # Create unique key for tick
            if tick.data_type == DataType.TRADE:
                trade_data = TradeData(**tick.data)
                key = f"{tick.exchange.value}:{tick.symbol}:{trade_data.trade_id}:{tick.timestamp}"
            else:
                key = f"{tick.exchange.value}:{tick.symbol}:{tick.data_type.value}:{tick.timestamp}:{tick.sequence_id}"
            
            source_key = f"{tick.exchange.value}:{tick.symbol}"
            
            # Check sliding window (last 1000 ticks)
            if key in self.duplicate_detection[source_key]:
                return True
            
            # Add to detection set
            self.duplicate_detection[source_key].add(key)
            
            # Maintain sliding window size
            if len(self.duplicate_detection[source_key]) > 1000:
                # Remove oldest (simplified - just clear and restart)
                self.duplicate_detection[source_key].clear()
            
            return False
            
        except Exception:
            return False  # Assume not duplicate on error

    def _process_tick(self, tick: MarketTick) -> Optional[ProcessedTick]:
        """Process and enrich a market tick"""
        try:
            enriched_data = {}
            quality_score = 1.0
            
            # Add timestamp information
            enriched_data['processing_timestamp'] = time.time()
            enriched_data['age_ms'] = (enriched_data['processing_timestamp'] - tick.timestamp) * 1000
            
            # Data type specific processing
            if tick.data_type == DataType.TRADE:
                trade_data = TradeData(**tick.data)
                
                # Enrich trade data
                enriched_data['usd_volume'] = float(trade_data.price * trade_data.size)
                enriched_data['price_tier'] = self._categorize_price(trade_data.price)
                enriched_data['size_tier'] = self._categorize_size(trade_data.size)
                
                # Quality assessment
                if tick.latency_us > 100000:  # 100ms
                    quality_score *= 0.8
                
            elif tick.data_type == DataType.ORDER_BOOK:
                orderbook = OrderBookSnapshot(**tick.data)
                
                # Calculate derived metrics
                if orderbook.bids and orderbook.asks:
                    best_bid = orderbook.bids[0].price
                    best_ask = orderbook.asks[0].price
                    spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / 2
                    
                    enriched_data['spread'] = float(spread)
                    enriched_data['mid_price'] = float(mid_price)
                    enriched_data['spread_bps'] = float(spread / mid_price * 10000)
                    
                    # Calculate depth metrics
                    bid_depth_5 = sum(level.size for level in orderbook.bids[:5])
                    ask_depth_5 = sum(level.size for level in orderbook.asks[:5])
                    enriched_data['total_depth_5'] = float(bid_depth_5 + ask_depth_5)
                    
                    # Quality assessment
                    if enriched_data['spread_bps'] > 100:  # Wide spread
                        quality_score *= 0.9
            
            # Sequence validation
            source_key = f"{tick.exchange.value}:{tick.symbol}"
            if tick.sequence_id:
                last_sequence = self.sequence_tracking.get(source_key, 0)
                if tick.sequence_id <= last_sequence:
                    quality_score *= 0.7  # Out of order
                self.sequence_tracking[source_key] = tick.sequence_id
            
            # Create storage key
            storage_key = f"{tick.exchange.value}_{tick.symbol}_{tick.data_type.value}_{int(tick.timestamp)}"
            
            return ProcessedTick(
                original_tick=tick,
                enriched_data=enriched_data,
                quality_score=quality_score,
                processing_timestamp=time.time(),
                storage_key=storage_key
            )
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Error processing tick: {e}")
            return None

    def _categorize_price(self, price: Decimal) -> str:
        """Categorize price into tiers"""
        price_float = float(price)
        if price_float < 0.1:
            return "micro"
        elif price_float < 1.0:
            return "low"
        elif price_float < 10.0:
            return "medium"
        elif price_float < 100.0:
            return "high"
        else:
            return "premium"

    def _categorize_size(self, size: Decimal) -> str:
        """Categorize trade size into tiers"""
        size_float = float(size)
        if size_float < 10:
            return "small"
        elif size_float < 100:
            return "medium"
        elif size_float < 1000:
            return "large"
        else:
            return "whale"

    def _store_batch(self, batch: List[ProcessedTick]):
        """Store a batch of processed ticks to database"""
        try:
            if not self.db_connection:
                return
            
            cursor = self.db_connection.cursor()
            
            # Group by data type for efficient storage
            trades = []
            orderbooks = []
            
            for processed_tick in batch:
                tick = processed_tick.original_tick
                
                if tick.data_type == DataType.TRADE:
                    trade_data = TradeData(**tick.data)
                    trades.append((
                        tick.symbol,
                        tick.exchange.value,
                        tick.timestamp,
                        float(trade_data.price),
                        float(trade_data.size),
                        trade_data.side,
                        trade_data.trade_id,
                        trade_data.is_taker,
                        tick.latency_us,
                        processed_tick.quality_score
                    ))
                
                elif tick.data_type == DataType.ORDER_BOOK:
                    orderbook = OrderBookSnapshot(**tick.data)
                    orderbooks.append((
                        tick.symbol,
                        tick.exchange.value,
                        tick.timestamp,
                        tick.sequence_id,
                        json.dumps([asdict(bid) for bid in orderbook.bids]),
                        json.dumps([asdict(ask) for ask in orderbook.asks]),
                        processed_tick.enriched_data.get('spread'),
                        processed_tick.enriched_data.get('mid_price'),
                        tick.latency_us,
                        processed_tick.quality_score
                    ))
            
            # Batch insert trades
            if trades:
                cursor.executemany('''
                    INSERT INTO trades (symbol, exchange, timestamp, price, size, side, 
                                      trade_id, is_taker, latency_us, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', trades)
            
            # Batch insert order books
            if orderbooks:
                cursor.executemany('''
                    INSERT INTO orderbook_snapshots (symbol, exchange, timestamp, sequence_id,
                                                    bids_json, asks_json, spread, mid_price,
                                                    latency_us, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', orderbooks)
            
            self.db_connection.commit()
            
            # Call analytics callbacks
            for callback in self.analytics_callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    self.logger.error(f"❌ [PIPELINE] Analytics callback error: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Batch storage error: {e}")

    async def _archive_old_data(self, cutoff_time: float):
        """Archive old data to compressed files"""
        try:
            if not self.db_connection:
                return
            
            os.makedirs(self.archive_path, exist_ok=True)
            
            cursor = self.db_connection.cursor()
            
            # Archive trades
            cursor.execute('''
                SELECT * FROM trades WHERE timestamp < ? ORDER BY timestamp
            ''', (cutoff_time,))
            
            old_trades = cursor.fetchall()
            if old_trades:
                archive_file = os.path.join(self.archive_path, f"trades_{int(cutoff_time)}.csv.gz")
                with gzip.open(archive_file, 'wt', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([desc[0] for desc in cursor.description])  # Header
                    writer.writerows(old_trades)
                
                # Delete archived data
                cursor.execute('DELETE FROM trades WHERE timestamp < ?', (cutoff_time,))
                
                self.logger.info(f"🔄 [PIPELINE] Archived {len(old_trades)} trades to {archive_file}")
            
            # Archive order books
            cursor.execute('''
                SELECT * FROM orderbook_snapshots WHERE timestamp < ? ORDER BY timestamp
            ''', (cutoff_time,))
            
            old_orderbooks = cursor.fetchall()
            if old_orderbooks:
                archive_file = os.path.join(self.archive_path, f"orderbooks_{int(cutoff_time)}.csv.gz")
                with gzip.open(archive_file, 'wt', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([desc[0] for desc in cursor.description])  # Header
                    writer.writerows(old_orderbooks)
                
                # Delete archived data
                cursor.execute('DELETE FROM orderbook_snapshots WHERE timestamp < ?', (cutoff_time,))
                
                self.logger.info(f"🔄 [PIPELINE] Archived {len(old_orderbooks)} orderbooks to {archive_file}")
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Archival error: {e}")

    async def _update_quality_metrics(self):
        """Update data quality metrics"""
        try:
            current_time = time.time()
            
            for source_key, metrics in self.quality_metrics.items():
                # Store quality metrics to database
                if self.db_connection:
                    cursor = self.db_connection.cursor()
                    cursor.execute('''
                        INSERT INTO data_quality_metrics 
                        (source, timestamp, total_ticks, valid_ticks, invalid_ticks,
                         duplicate_ticks, out_of_order_ticks, latency_violations, quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        source_key, current_time,
                        metrics.total_ticks, metrics.valid_ticks, metrics.invalid_ticks,
                        metrics.duplicate_ticks, metrics.out_of_order_ticks,
                        metrics.latency_violations, metrics.quality_score
                    ))
                    self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Quality metrics update error: {e}")

    async def _monitor_pipeline_performance(self):
        """Monitor pipeline performance"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                # Calculate queue sizes and processing rates
                ingestion_size = self.ingestion_queue.qsize()
                processing_size = self.processing_queue.qsize()
                storage_size = self.storage_queue.qsize()
                
                # Calculate average latencies
                avg_processing_latency = sum(self.processing_latency) / len(self.processing_latency) if self.processing_latency else 0
                avg_storage_latency = sum(self.storage_latency) / len(self.storage_latency) if self.storage_latency else 0
                
                self.logger.info(
                    f"🔄 [PIPELINE] Performance: "
                    f"Processed: {self.processed_count}, "
                    f"Stored: {self.stored_count}, "
                    f"Queues: I:{ingestion_size} P:{processing_size} S:{storage_size}, "
                    f"Latency: Proc:{avg_processing_latency:.1f}μs Stor:{avg_storage_latency:.1f}ms"
                )
                
            except Exception as e:
                self.logger.error(f"❌ [PIPELINE] Performance monitoring error: {e}")

    def register_analytics_callback(self, callback: Callable[[List[ProcessedTick]], None]):
        """Register callback for real-time analytics"""
        self.analytics_callbacks.append(callback)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "running": self.running,
            "processed_count": self.processed_count,
            "stored_count": self.stored_count,
            "queue_sizes": {
                "ingestion": self.ingestion_queue.qsize(),
                "processing": self.processing_queue.qsize(),
                "storage": self.storage_queue.qsize()
            },
            "avg_processing_latency_us": sum(self.processing_latency) / len(self.processing_latency) if self.processing_latency else 0,
            "avg_storage_latency_ms": sum(self.storage_latency) / len(self.storage_latency) if self.storage_latency else 0,
            "quality_metrics": {k: asdict(v) for k, v in self.quality_metrics.items()}
        }

    async def stop_pipeline(self):
        """Stop the data pipeline"""
        try:
            self.running = False
            
            # Wait for threads to finish
            if self.ingestion_thread:
                self.ingestion_thread.join(timeout=5)
            
            for thread in self.processing_threads:
                thread.join(timeout=5)
            
            if self.storage_thread:
                self.storage_thread.join(timeout=5)
            
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
            
            self.logger.info("🔄 [PIPELINE] Pipeline stopped")
            
        except Exception as e:
            self.logger.error(f"❌ [PIPELINE] Error stopping pipeline: {e}")
