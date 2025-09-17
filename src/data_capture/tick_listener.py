"""
Tick Tape Capture - Hyperliquid WebSocket Market Data
Captures tick-for-tick market data for replay power and audit trails.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import json
import asyncio
import websockets
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import gzip
import time


@dataclass
class TickData:
    """Represents a single tick data point."""
    timestamp_ms: int
    price: float
    quantity: float
    side: str
    best_bid: float
    best_ask: float
    spread: float
    depth_bid: float
    depth_ask: float
    symbol: str


class TickListener:
    """Listens to Hyperliquid WebSocket streams and captures tick data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.ticks_dir = self.data_dir / "ticks"
        self.warehouse_dir = self.data_dir / "warehouse"
        
        # Create directories
        self.ticks_dir.mkdir(parents=True, exist_ok=True)
        self.warehouse_dir.mkdir(parents=True, exist_ok=True)
        
        # WebSocket configuration
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.symbols = ["XRP"]  # Focus on XRP for now
        
        # Data capture state
        self.current_file = None
        self.current_date = None
        self.tick_count = 0
        self.drop_count = 0
        self.start_time = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def connect_websocket(self) -> websockets.WebSocketServerProtocol:
        """Connect to Hyperliquid WebSocket."""
        
        try:
            websocket = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.logger.info(f"Connected to Hyperliquid WebSocket: {self.ws_url}")
            return websocket
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def subscribe_to_market_data(self, websocket: websockets.WebSocketServerProtocol):
        """Subscribe to market data streams."""
        
        # Subscribe to trades
        trade_subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": "XRP"
            }
        }
        
        # Subscribe to order book
        book_subscription = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": "XRP"
            }
        }
        
        try:
            await websocket.send(json.dumps(trade_subscription))
            await websocket.send(json.dumps(book_subscription))
            self.logger.info("Subscribed to XRP trades and order book")
        except Exception as e:
            self.logger.error(f"Failed to subscribe to market data: {e}")
            raise
    
    def _get_current_file_path(self) -> Path:
        """Get current tick file path based on date."""
        
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if self.current_date != current_date:
            # Date changed, close current file and open new one
            if self.current_file:
                self.current_file.close()
            
            self.current_date = current_date
            filename = f"{current_date}_xrp.jsonl"
            file_path = self.ticks_dir / filename
            self.current_file = open(file_path, 'a', encoding='utf-8')
            self.logger.info(f"Opened new tick file: {file_path}")
        
        return self.ticks_dir / f"{current_date}_xrp.jsonl"
    
    def _parse_trade_data(self, message: Dict[str, Any]) -> Optional[TickData]:
        """Parse trade data from WebSocket message."""
        
        try:
            if message.get("channel") != "trades":
                return None
            
            data = message.get("data", [])
            if not data:
                return None
            
            # Get latest trade
            trade = data[-1] if isinstance(data, list) else data
            
            # Extract trade information
            timestamp_ms = int(trade.get("time", time.time() * 1000))
            price = safe_float(trade.get("px", 0))
            quantity = safe_float(trade.get("sz", 0))
            side = trade.get("side", "unknown")
            
            # For now, we'll use placeholder values for bid/ask
            # In a real implementation, you'd maintain the latest order book state
            best_bid = price * 0.9999  # Placeholder
            best_ask = price * 1.0001  # Placeholder
            spread = best_ask - best_bid
            depth_bid = quantity * 10  # Placeholder
            depth_ask = quantity * 10  # Placeholder
            
            return TickData(
                timestamp_ms=timestamp_ms,
                price=price,
                quantity=quantity,
                side=side,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                depth_bid=depth_bid,
                depth_ask=depth_ask,
                symbol="XRP"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse trade data: {e}")
            return None
    
    def _parse_book_data(self, message: Dict[str, Any]) -> Optional[TickData]:
        """Parse order book data from WebSocket message."""
        
        try:
            if message.get("channel") != "l2Book":
                return None
            
            data = message.get("data", {})
            if not data:
                return None
            
            # Extract order book information
            timestamp_ms = int(time.time() * 1000)
            
            # Get best bid/ask
            levels = data.get("levels", [])
            if len(levels) >= 2:
                best_bid = safe_float(levels[0][0])  # [price, size]
                best_ask = safe_float(levels[1][0])
                depth_bid = safe_float(levels[0][1])
                depth_ask = safe_float(levels[1][1])
            else:
                return None
            
            spread = best_ask - best_bid
            
            # Use mid-price as reference
            price = (best_bid + best_ask) / 2
            quantity = 0  # No trade quantity for book updates
            
            return TickData(
                timestamp_ms=timestamp_ms,
                price=price,
                quantity=quantity,
                side="book_update",
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                depth_bid=depth_bid,
                depth_ask=depth_ask,
                symbol="XRP"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse book data: {e}")
            return None
    
    def _write_tick_data(self, tick_data: TickData):
        """Write tick data to JSONL file."""
        
        try:
            # Ensure file is open
            self._get_current_file_path()
            
            # Convert to dictionary
            tick_dict = {
                "timestamp_ms": tick_data.timestamp_ms,
                "price": tick_data.price,
                "quantity": tick_data.quantity,
                "side": tick_data.side,
                "best_bid": tick_data.best_bid,
                "best_ask": tick_data.best_ask,
                "spread": tick_data.spread,
                "depth_bid": tick_data.depth_bid,
                "depth_ask": tick_data.depth_ask,
                "symbol": tick_data.symbol
            }
            
            # Write to JSONL file
            if self.current_file:
                self.current_file.write(json.dumps(tick_dict) + '\n')
                self.current_file.flush()
                self.tick_count += 1
            
        except Exception as e:
            self.logger.error(f"Failed to write tick data: {e}")
            self.drop_count += 1
    
    async def process_message(self, message: str):
        """Process incoming WebSocket message."""
        
        try:
            data = json.loads(message)
            
            # Parse trade data
            trade_tick = self._parse_trade_data(data)
            if trade_tick:
                self._write_tick_data(trade_tick)
            
            # Parse book data
            book_tick = self._parse_book_data(data)
            if book_tick:
                self._write_tick_data(book_tick)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
            self.drop_count += 1
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            self.drop_count += 1
    
    async def listen_to_ticks(self):
        """Main loop to listen to tick data."""
        
        self.start_time = datetime.now(timezone.utc)
        self.logger.info("Starting tick data capture...")
        
        while True:
            try:
                websocket = await self.connect_websocket()
                await self.subscribe_to_market_data(websocket)
                
                async for message in websocket:
                    await self.process_message(message)
                    
                    # Log progress every 1000 ticks
                    if self.tick_count % 1000 == 0 and self.tick_count > 0:
                        self.logger.info(f"Captured {self.tick_count} ticks, {self.drop_count} drops")
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Error in tick listener: {e}")
                await asyncio.sleep(10)
    
    def convert_to_parquet(self, date: str = None):
        """Convert JSONL tick data to Parquet format."""
        
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        jsonl_file = self.ticks_dir / f"{date}_xrp.jsonl"
        parquet_file = self.ticks_dir / f"xrp_{date}.parquet"
        
        if not jsonl_file.exists():
            self.logger.warning(f"JSONL file not found: {jsonl_file}")
            return
        
        try:
            # Read JSONL data
            data = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            if not data:
                self.logger.warning(f"No data found in {jsonl_file}")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
            
            # Save as Parquet
            df.to_parquet(parquet_file, index=False)
            self.logger.info(f"Converted {len(data)} ticks to Parquet: {parquet_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to convert to Parquet: {e}")
    
    def create_provenance_record(self):
        """Create provenance record for captured data."""
        
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date_dir = self.warehouse_dir / current_date
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        runtime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600 if self.start_time else 0
        drop_rate = self.drop_count / max(self.tick_count + self.drop_count, 1)
        
        provenance = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "hyperliquid_websocket",
            "endpoints": [self.ws_url],
            "timezone": "UTC",
            "symbols": self.symbols,
            "capture_period": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "runtime_hours": runtime_hours
            },
            "data_quality": {
                "total_ticks": self.tick_count,
                "dropped_ticks": self.drop_count,
                "drop_rate": drop_rate,
                "tick_rate_per_second": self.tick_count / max(runtime_hours * 3600, 1)
            },
            "files": {
                "jsonl_file": f"{current_date}_xrp.jsonl",
                "parquet_file": f"xrp_{current_date}.parquet"
            },
            "schema_version": "1.0",
            "capture_method": "websocket_streaming"
        }
        
        # Save provenance
        provenance_file = date_dir / "_provenance.json"
        with open(provenance_file, 'w') as f:
            json.dump(provenance, f, indent=2)
        
        self.logger.info(f"Created provenance record: {provenance_file}")
        return provenance


async def main():
    """Main function to run tick listener."""
    listener = TickListener()
    
    try:
        # Start listening
        await listener.listen_to_ticks()
    except KeyboardInterrupt:
        listener.logger.info("Stopping tick listener...")
        
        # Convert to Parquet
        listener.convert_to_parquet()
        
        # Create provenance record
        listener.create_provenance_record()
        
        # Close file
        if listener.current_file:
            listener.current_file.close()
        
        listener.logger.info("Tick listener stopped")


if __name__ == "__main__":
    asyncio.run(main())
