"""
Multi-Venue Manager
Parallel connector to multiple exchanges for resilience and benchmarking
"""

from src.core.utils.decimal_boundary_guard import safe_decimal
from src.core.utils.decimal_boundary_guard import safe_float
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import websockets
from decimal import Decimal

class VenueType(Enum):
    HYPERLIQUID = "hyperliquid"
    BINANCE = "binance"
    BYBIT = "bybit"
    COINBASE = "coinbase"

class ConnectionStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

@dataclass
class VenueConfig:
    venue_type: VenueType
    name: str
    api_key: str
    api_secret: str
    base_url: str
    ws_url: str
    testnet: bool = True
    enabled: bool = True
    priority: int = 1  # 1 = primary, 2 = secondary, etc.

@dataclass
class MarketData:
    venue: VenueType
    symbol: str
    timestamp: str
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    last_price: Decimal
    volume_24h: Decimal
    spread_bps: float

@dataclass
class OrderBook:
    venue: VenueType
    symbol: str
    timestamp: str
    bids: List[Tuple[Decimal, Decimal]]  # (price, size)
    asks: List[Tuple[Decimal, Decimal]]  # (price, size)
    sequence: int

@dataclass
class Trade:
    venue: VenueType
    symbol: str
    timestamp: str
    side: str  # 'buy' or 'sell'
    price: Decimal
    size: Decimal
    trade_id: str

@dataclass
class VenueHealth:
    venue: VenueType
    status: ConnectionStatus
    last_heartbeat: str
    latency_ms: float
    error_count: int
    success_rate: float
    last_error: Optional[str] = None

class MultiVenueManager:
    """
    Manages connections to multiple venues for resilience and benchmarking
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.venues: Dict[VenueType, VenueConfig] = {}
        self.connections: Dict[VenueType, Any] = {}
        self.venue_health: Dict[VenueType, VenueHealth] = {}
        self.market_data: Dict[VenueType, Dict[str, MarketData]] = {}
        self.order_books: Dict[VenueType, Dict[str, OrderBook]] = {}
        self.trades: Dict[VenueType, List[Trade]] = {}
        
        # Create reports directory
        self.reports_dir = Path("reports/multi_venue")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize venue health tracking
        for venue_type in VenueType:
            self.venue_health[venue_type] = VenueHealth(
                venue=venue_type,
                status=ConnectionStatus.DISCONNECTED,
                last_heartbeat=datetime.now().isoformat(),
                latency_ms=0.0,
                error_count=0,
                success_rate=0.0
            )
            self.market_data[venue_type] = {}
            self.order_books[venue_type] = {}
            self.trades[venue_type] = []
    
    def add_venue(self, config: VenueConfig):
        """Add a venue configuration"""
        self.venues[config.venue_type] = config
        self.logger.info(f"✅ Added venue: {config.name} ({config.venue_type.value})")
    
    async def start_all_venues(self):
        """Start connections to all enabled venues"""
        self.logger.info("🚀 Starting multi-venue connections")
        
        tasks = []
        for venue_type, config in self.venues.items():
            if config.enabled:
                task = asyncio.create_task(self._start_venue(venue_type, config))
                tasks.append(task)
        
        # Start all venues concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _start_venue(self, venue_type: VenueType, config: VenueConfig):
        """Start connection to a specific venue"""
        try:
            self.logger.info(f"🔄 Starting {config.name} connection")
            
            if venue_type == VenueType.HYPERLIQUID:
                await self._start_hyperliquid(config)
            elif venue_type == VenueType.BINANCE:
                await self._start_binance(config)
            elif venue_type == VenueType.BYBIT:
                await self._start_bybit(config)
            elif venue_type == VenueType.COINBASE:
                await self._start_coinbase(config)
            else:
                self.logger.error(f"❌ Unsupported venue type: {venue_type}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start {config.name}: {e}")
            self.venue_health[venue_type].status = ConnectionStatus.FAILED
            self.venue_health[venue_type].last_error = str(e)
    
    async def _start_hyperliquid(self, config: VenueConfig):
        """Start Hyperliquid connection"""
        self.logger.info("🔄 Starting Hyperliquid connection")
        
        # Simulate Hyperliquid connection
        # In a real implementation, this would connect to Hyperliquid API/WS
        self.venue_health[VenueType.HYPERLIQUID].status = ConnectionStatus.CONNECTED
        self.venue_health[VenueType.HYPERLIQUID].last_heartbeat = datetime.now().isoformat()
        
        # Start market data feed
        asyncio.create_task(self._hyperliquid_market_data_loop())
        asyncio.create_task(self._hyperliquid_orderbook_loop())
        asyncio.create_task(self._hyperliquid_trades_loop())
    
    async def _start_binance(self, config: VenueConfig):
        """Start Binance connection"""
        self.logger.info("🔄 Starting Binance connection")
        
        # Simulate Binance connection
        # In a real implementation, this would connect to Binance API/WS
        self.venue_health[VenueType.BINANCE].status = ConnectionStatus.CONNECTED
        self.venue_health[VenueType.BINANCE].last_heartbeat = datetime.now().isoformat()
        
        # Start market data feed
        asyncio.create_task(self._binance_market_data_loop())
        asyncio.create_task(self._binance_orderbook_loop())
        asyncio.create_task(self._binance_trades_loop())
    
    async def _start_bybit(self, config: VenueConfig):
        """Start Bybit connection"""
        self.logger.info("🔄 Starting Bybit connection")
        
        # Simulate Bybit connection
        # In a real implementation, this would connect to Bybit API/WS
        self.venue_health[VenueType.BYBIT].status = ConnectionStatus.CONNECTED
        self.venue_health[VenueType.BYBIT].last_heartbeat = datetime.now().isoformat()
        
        # Start market data feed
        asyncio.create_task(self._bybit_market_data_loop())
        asyncio.create_task(self._bybit_orderbook_loop())
        asyncio.create_task(self._bybit_trades_loop())
    
    async def _start_coinbase(self, config: VenueConfig):
        """Start Coinbase connection"""
        self.logger.info("🔄 Starting Coinbase connection")
        
        # Simulate Coinbase connection
        # In a real implementation, this would connect to Coinbase API/WS
        self.venue_health[VenueType.COINBASE].status = ConnectionStatus.CONNECTED
        self.venue_health[VenueType.COINBASE].last_heartbeat = datetime.now().isoformat()
        
        # Start market data feed
        asyncio.create_task(self._coinbase_market_data_loop())
        asyncio.create_task(self._coinbase_orderbook_loop())
        asyncio.create_task(self._coinbase_trades_loop())
    
    # Market data loops for each venue
    async def _hyperliquid_market_data_loop(self):
        """Hyperliquid market data feed"""
        while self.venue_health[VenueType.HYPERLIQUID].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate market data
                market_data = MarketData(
                    venue=VenueType.HYPERLIQUID,
                    symbol="XRP-USD",
                    timestamp=datetime.now().isoformat(),
                    bid_price=safe_decimal("0.5234"),
                    ask_price=safe_decimal("0.5236"),
                    bid_size=safe_decimal("1000"),
                    ask_size=safe_decimal("1000"),
                    last_price=safe_decimal("0.5235"),
                    volume_24h=safe_decimal("1000000"),
                    spread_bps=3.8
                )
                
                self.market_data[VenueType.HYPERLIQUID]["XRP-USD"] = market_data
                await self._update_venue_health(VenueType.HYPERLIQUID, success=True)
                
                await asyncio.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                await self._update_venue_health(VenueType.HYPERLIQUID, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _binance_market_data_loop(self):
        """Binance market data feed"""
        while self.venue_health[VenueType.BINANCE].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate market data
                market_data = MarketData(
                    venue=VenueType.BINANCE,
                    symbol="XRPUSDT",
                    timestamp=datetime.now().isoformat(),
                    bid_price=safe_decimal("0.5233"),
                    ask_price=safe_decimal("0.5237"),
                    bid_size=safe_decimal("2000"),
                    ask_size=safe_decimal("2000"),
                    last_price=safe_decimal("0.5235"),
                    volume_24h=safe_decimal("2000000"),
                    spread_bps=7.6
                )
                
                self.market_data[VenueType.BINANCE]["XRPUSDT"] = market_data
                await self._update_venue_health(VenueType.BINANCE, success=True)
                
                await asyncio.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                await self._update_venue_health(VenueType.BINANCE, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _bybit_market_data_loop(self):
        """Bybit market data feed"""
        while self.venue_health[VenueType.BYBIT].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate market data
                market_data = MarketData(
                    venue=VenueType.BYBIT,
                    symbol="XRPUSDT",
                    timestamp=datetime.now().isoformat(),
                    bid_price=safe_decimal("0.5232"),
                    ask_price=safe_decimal("0.5238"),
                    bid_size=safe_decimal("1500"),
                    ask_size=safe_decimal("1500"),
                    last_price=safe_decimal("0.5235"),
                    volume_24h=safe_decimal("1500000"),
                    spread_bps=11.4
                )
                
                self.market_data[VenueType.BYBIT]["XRPUSDT"] = market_data
                await self._update_venue_health(VenueType.BYBIT, success=True)
                
                await asyncio.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                await self._update_venue_health(VenueType.BYBIT, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _coinbase_market_data_loop(self):
        """Coinbase market data feed"""
        while self.venue_health[VenueType.COINBASE].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate market data
                market_data = MarketData(
                    venue=VenueType.COINBASE,
                    symbol="XRP-USD",
                    timestamp=datetime.now().isoformat(),
                    bid_price=safe_decimal("0.5231"),
                    ask_price=safe_decimal("0.5239"),
                    bid_size=safe_decimal("800"),
                    ask_size=safe_decimal("800"),
                    last_price=safe_decimal("0.5235"),
                    volume_24h=safe_decimal("800000"),
                    spread_bps=15.2
                )
                
                self.market_data[VenueType.COINBASE]["XRP-USD"] = market_data
                await self._update_venue_health(VenueType.COINBASE, success=True)
                
                await asyncio.sleep(0.1)  # 10 updates per second
                
            except Exception as e:
                await self._update_venue_health(VenueType.COINBASE, success=False, error=str(e))
                await asyncio.sleep(1)
    
    # Order book loops (simplified)
    async def _hyperliquid_orderbook_loop(self):
        """Hyperliquid order book feed"""
        while self.venue_health[VenueType.HYPERLIQUID].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate order book updates
                await asyncio.sleep(0.5)  # 2 updates per second
            except Exception as e:
                await self._update_venue_health(VenueType.HYPERLIQUID, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _binance_orderbook_loop(self):
        """Binance order book feed"""
        while self.venue_health[VenueType.BINANCE].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate order book updates
                await asyncio.sleep(0.5)  # 2 updates per second
            except Exception as e:
                await self._update_venue_health(VenueType.BINANCE, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _bybit_orderbook_loop(self):
        """Bybit order book feed"""
        while self.venue_health[VenueType.BYBIT].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate order book updates
                await asyncio.sleep(0.5)  # 2 updates per second
            except Exception as e:
                await self._update_venue_health(VenueType.BYBIT, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _coinbase_orderbook_loop(self):
        """Coinbase order book feed"""
        while self.venue_health[VenueType.COINBASE].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate order book updates
                await asyncio.sleep(0.5)  # 2 updates per second
            except Exception as e:
                await self._update_venue_health(VenueType.COINBASE, success=False, error=str(e))
                await asyncio.sleep(1)
    
    # Trade loops (simplified)
    async def _hyperliquid_trades_loop(self):
        """Hyperliquid trades feed"""
        while self.venue_health[VenueType.HYPERLIQUID].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate trade updates
                await asyncio.sleep(1)  # 1 update per second
            except Exception as e:
                await self._update_venue_health(VenueType.HYPERLIQUID, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _binance_trades_loop(self):
        """Binance trades feed"""
        while self.venue_health[VenueType.BINANCE].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate trade updates
                await asyncio.sleep(1)  # 1 update per second
            except Exception as e:
                await self._update_venue_health(VenueType.BINANCE, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _bybit_trades_loop(self):
        """Bybit trades feed"""
        while self.venue_health[VenueType.BYBIT].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate trade updates
                await asyncio.sleep(1)  # 1 update per second
            except Exception as e:
                await self._update_venue_health(VenueType.BYBIT, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _coinbase_trades_loop(self):
        """Coinbase trades feed"""
        while self.venue_health[VenueType.COINBASE].status == ConnectionStatus.CONNECTED:
            try:
                # Simulate trade updates
                await asyncio.sleep(1)  # 1 update per second
            except Exception as e:
                await self._update_venue_health(VenueType.COINBASE, success=False, error=str(e))
                await asyncio.sleep(1)
    
    async def _update_venue_health(self, venue_type: VenueType, success: bool, error: str = None):
        """Update venue health metrics"""
        health = self.venue_health[venue_type]
        
        if success:
            health.error_count = max(0, health.error_count - 1)
            health.success_rate = min(1.0, health.success_rate + 0.01)
            health.last_heartbeat = datetime.now().isoformat()
        else:
            health.error_count += 1
            health.success_rate = max(0.0, health.success_rate - 0.01)
            health.last_error = error
            
            if health.error_count > 5:
                health.status = ConnectionStatus.FAILED
    
    def get_best_venue(self, symbol: str) -> Optional[VenueType]:
        """Get the best venue for a symbol based on health and spread"""
        best_venue = None
        best_score = safe_float('inf')
        
        for venue_type, health in self.venue_health.items():
            if health.status != ConnectionStatus.CONNECTED:
                continue
            
            # Get market data for this venue
            venue_market_data = self.market_data.get(venue_type, {})
            if symbol not in venue_market_data:
                continue
            
            market_data = venue_market_data[symbol]
            
            # Score based on spread and health
            score = market_data.spread_bps + (1 - health.success_rate) * 100
            
            if score < best_score:
                best_score = score
                best_venue = venue_type
        
        return best_venue
    
    def get_venue_comparison(self, symbol: str) -> Dict:
        """Get comparison of all venues for a symbol"""
        comparison = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "venues": []
        }
        
        for venue_type, health in self.venue_health.items():
            if health.status != ConnectionStatus.CONNECTED:
                continue
            
            venue_market_data = self.market_data.get(venue_type, {})
            if symbol not in venue_market_data:
                continue
            
            market_data = venue_market_data[symbol]
            
            venue_info = {
                "venue": venue_type.value,
                "status": health.status.value,
                "bid_price": safe_float(market_data.bid_price),
                "ask_price": safe_float(market_data.ask_price),
                "spread_bps": market_data.spread_bps,
                "volume_24h": safe_float(market_data.volume_24h),
                "success_rate": health.success_rate,
                "latency_ms": health.latency_ms
            }
            
            comparison["venues"].append(venue_info)
        
        # Sort by spread
        comparison["venues"].sort(key=lambda x: x["spread_bps"])
        
        return comparison
    
    async def save_venue_metrics(self):
        """Save venue metrics to file"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "venue_health": {},
                "market_data": {},
                "comparison": self.get_venue_comparison("XRP-USD")
            }
            
            # Convert venue health to serializable format
            for venue_type, health in self.venue_health.items():
                metrics["venue_health"][venue_type.value] = {
                    "status": health.status.value,
                    "last_heartbeat": health.last_heartbeat,
                    "latency_ms": health.latency_ms,
                    "error_count": health.error_count,
                    "success_rate": health.success_rate,
                    "last_error": health.last_error
                }
            
            # Convert market data to serializable format
            for venue_type, venue_data in self.market_data.items():
                metrics["market_data"][venue_type.value] = {}
                for symbol, data in venue_data.items():
                    metrics["market_data"][venue_type.value][symbol] = {
                        "timestamp": data.timestamp,
                        "bid_price": safe_float(data.bid_price),
                        "ask_price": safe_float(data.ask_price),
                        "bid_size": safe_float(data.bid_size),
                        "ask_size": safe_float(data.ask_size),
                        "last_price": safe_float(data.last_price),
                        "volume_24h": safe_float(data.volume_24h),
                        "spread_bps": data.spread_bps
                    }
            
            # Save to file
            metrics_file = self.reports_dir / "venue_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            self.logger.info(f"💾 Venue metrics saved: {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save venue metrics: {e}")

# Demo function
async def demo_multi_venue_manager():
    """Demo the multi-venue manager"""
    print("🌐 Multi-Venue Manager Demo")
    print("=" * 50)
    
    manager = MultiVenueManager()
    
    # Add venue configurations
    venues = [
        VenueConfig(
            venue_type=VenueType.HYPERLIQUID,
            name="Hyperliquid",
            api_key="demo_key",
            api_secret="demo_secret",
            base_url="https://api.hyperliquid.xyz",
            ws_url="wss://api.hyperliquid.xyz/ws",
            testnet=True,
            priority=1
        ),
        VenueConfig(
            venue_type=VenueType.BINANCE,
            name="Binance Testnet",
            api_key="demo_key",
            api_secret="demo_secret",
            base_url="https://testnet.binance.vision",
            ws_url="wss://testnet.binance.vision/ws",
            testnet=True,
            priority=2
        ),
        VenueConfig(
            venue_type=VenueType.BYBIT,
            name="Bybit Testnet",
            api_key="demo_key",
            api_secret="demo_secret",
            base_url="https://api-testnet.bybit.com",
            ws_url="wss://stream-testnet.bybit.com",
            testnet=True,
            priority=3
        )
    ]
    
    # Add venues
    for venue in venues:
        manager.add_venue(venue)
    
    # Start all venues
    print("🔄 Starting venue connections...")
    await manager.start_all_venues()
    
    # Wait for data
    await asyncio.sleep(5)
    
    # Get venue comparison
    comparison = manager.get_venue_comparison("XRP-USD")
    print(f"\n📊 Venue Comparison for XRP-USD:")
    for venue in comparison["venues"]:
        print(f"  {venue['venue']}: {venue['spread_bps']:.1f}bps spread, {venue['success_rate']:.1%} success rate")
    
    # Get best venue
    best_venue = manager.get_best_venue("XRP-USD")
    print(f"\n🏆 Best Venue: {best_venue.value if best_venue else 'None'}")
    
    # Save metrics
    await manager.save_venue_metrics()
    
    print("\n✅ Multi-Venue Manager Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_multi_venue_manager())
