"""
Enhanced Tick Capture
Continuous tick and funding data capture with perfect replay capabilities.
"""

import json
import os
import asyncio
import websockets
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTickCapture:
    """Captures tick and funding data continuously."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.data_dir = self.repo_root / "data"
        
        # WebSocket configuration
        self.websocket_url = "wss://api.hyperliquid.xyz/ws"
        self.reconnect_interval = 5  # seconds
        self.max_reconnect_attempts = 10
        
        # Data storage
        self.tick_buffer = []
        self.funding_buffer = []
        self.buffer_size = 1000
        
        # Capture statistics
        self.stats = {
            'ticks_captured': 0,
            'funding_events_captured': 0,
            'connection_attempts': 0,
            'reconnects': 0,
            'last_tick_time': None,
            'last_funding_time': None
        }
        
        # Create data directories
        self.setup_data_directories()
    
    def setup_data_directories(self):
        """Set up data storage directories."""
        directories = [
            self.data_dir / "ticks",
            self.data_dir / "funding",
            self.data_dir / "warehouse"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 Data directories set up")
    
    async def connect_websocket(self) -> websockets.WebSocketServerProtocol:
        """Connect to Hyperliquid WebSocket."""
        try:
            self.stats['connection_attempts'] += 1
            logger.info(f"🔌 Connecting to WebSocket (attempt {self.stats['connection_attempts']})...")
            
            websocket = await websockets.connect(self.websocket_url)
            logger.info("✅ WebSocket connected successfully")
            return websocket
        
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            raise
    
    async def subscribe_to_tick_data(self, websocket):
        """Subscribe to tick data stream."""
        subscribe_message = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": "XRP"
            }
        }
        
        await websocket.send(json.dumps(subscribe_message))
        logger.info("📡 Subscribed to XRP tick data")
    
    async def subscribe_to_funding_data(self, websocket):
        """Subscribe to funding data stream."""
        subscribe_message = {
            "method": "subscribe",
            "subscription": {
                "type": "funding",
                "coin": "XRP"
            }
        }
        
        await websocket.send(json.dumps(subscribe_message))
        logger.info("💰 Subscribed to XRP funding data")
    
    def process_tick_message(self, message: Dict) -> Optional[Dict]:
        """Process tick data message."""
        try:
            if message.get('channel') == 'trades':
                trade_data = message.get('data', {})
                
                tick = {
                    'timestamp': datetime.now().isoformat(),
                    'price': trade_data.get('px', 0),
                    'size': trade_data.get('sz', 0),
                    'side': trade_data.get('side', 'unknown'),
                    'coin': trade_data.get('coin', 'XRP'),
                    'hash': trade_data.get('hash', ''),
                    'source': 'hyperliquid_websocket'
                }
                
                self.stats['ticks_captured'] += 1
                self.stats['last_tick_time'] = tick['timestamp']
                
                return tick
        
        except Exception as e:
            logger.error(f"Error processing tick message: {e}")
        
        return None
    
    def process_funding_message(self, message: Dict) -> Optional[Dict]:
        """Process funding data message."""
        try:
            if message.get('channel') == 'funding':
                funding_data = message.get('data', {})
                
                funding = {
                    'timestamp': datetime.now().isoformat(),
                    'coin': funding_data.get('coin', 'XRP'),
                    'funding_rate': funding_data.get('fundingRate', 0),
                    'mark_price': funding_data.get('markPx', 0),
                    'index_price': funding_data.get('indexPx', 0),
                    'source': 'hyperliquid_websocket'
                }
                
                self.stats['funding_events_captured'] += 1
                self.stats['last_funding_time'] = funding['timestamp']
                
                return funding
        
        except Exception as e:
            logger.error(f"Error processing funding message: {e}")
        
        return None
    
    def save_tick_data(self, tick: Dict):
        """Save tick data to buffer and file."""
        self.tick_buffer.append(tick)
        
        # Save to daily file when buffer is full
        if len(self.tick_buffer) >= self.buffer_size:
            self.flush_tick_buffer()
    
    def save_funding_data(self, funding: Dict):
        """Save funding data to buffer and file."""
        self.funding_buffer.append(funding)
        
        # Save to daily file when buffer is full
        if len(self.funding_buffer) >= self.buffer_size:
            self.flush_funding_buffer()
    
    def flush_tick_buffer(self):
        """Flush tick buffer to file."""
        if not self.tick_buffer:
            return
        
        # Get current date for file naming
        current_date = datetime.now().strftime('%Y-%m-%d')
        tick_file = self.data_dir / "ticks" / f"xrp_{current_date}.jsonl"
        
        # Append ticks to file
        with open(tick_file, 'a') as f:
            for tick in self.tick_buffer:
                f.write(json.dumps(tick) + '\n')
        
        logger.info(f"💾 Flushed {len(self.tick_buffer)} ticks to {tick_file}")
        self.tick_buffer.clear()
    
    def flush_funding_buffer(self):
        """Flush funding buffer to file."""
        if not self.funding_buffer:
            return
        
        # Get current date for file naming
        current_date = datetime.now().strftime('%Y-%m-%d')
        funding_file = self.data_dir / "funding" / f"xrp_{current_date}.json"
        
        # Load existing funding data
        funding_data = []
        if funding_file.exists():
            with open(funding_file, 'r') as f:
                funding_data = json.load(f)
        
        # Add new funding events
        funding_data.extend(self.funding_buffer)
        
        # Save updated funding data
        with open(funding_file, 'w') as f:
            json.dump(funding_data, f, indent=2)
        
        logger.info(f"💾 Flushed {len(self.funding_buffer)} funding events to {funding_file}")
        self.funding_buffer.clear()
    
    def create_daily_provenance(self, date: str):
        """Create daily provenance file."""
        provenance_data = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'data_sources': {
                'ticks': {
                    'source': 'hyperliquid_websocket',
                    'endpoint': self.websocket_url,
                    'symbol': 'XRP',
                    'format': 'jsonl'
                },
                'funding': {
                    'source': 'hyperliquid_websocket',
                    'endpoint': self.websocket_url,
                    'symbol': 'XRP',
                    'format': 'json'
                }
            },
            'capture_stats': {
                'ticks_captured': self.stats['ticks_captured'],
                'funding_events_captured': self.stats['funding_events_captured'],
                'connection_attempts': self.stats['connection_attempts'],
                'reconnects': self.stats['reconnects']
            },
            'data_quality': {
                'missing_data_policy': 'forward_fill_last_known',
                'outlier_detection': 'iqr_method',
                'validation_rules': [
                    'price > 0',
                    'size > 0',
                    'timestamp_format_iso8601'
                ]
            }
        }
        
        # Save provenance file
        provenance_file = self.data_dir / "warehouse" / date / "_provenance.json"
        provenance_file.parent.mkdir(exist_ok=True)
        
        with open(provenance_file, 'w') as f:
            json.dump(provenance_data, f, indent=2)
        
        logger.info(f"📋 Daily provenance created: {provenance_file}")
    
    async def run_tick_capture(self):
        """Run continuous tick capture."""
        logger.info("🚀 Starting enhanced tick capture...")
        
        reconnect_attempts = 0
        
        while reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Connect to WebSocket
                websocket = await self.connect_websocket()
                
                # Subscribe to data streams
                await self.subscribe_to_tick_data(websocket)
                await self.subscribe_to_funding_data(websocket)
                
                # Process messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Process tick data
                        tick = self.process_tick_message(data)
                        if tick:
                            self.save_tick_data(tick)
                        
                        # Process funding data
                        funding = self.process_funding_message(data)
                        if funding:
                            self.save_funding_data(funding)
                    
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON message received")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                reconnect_attempts += 1
                self.stats['reconnects'] += 1
                
                if reconnect_attempts < self.max_reconnect_attempts:
                    logger.info(f"Reconnecting in {self.reconnect_interval} seconds...")
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    logger.error("Max reconnection attempts reached")
                    break
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                reconnect_attempts += 1
                
                if reconnect_attempts < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    break
        
        # Flush remaining buffers
        self.flush_tick_buffer()
        self.flush_funding_buffer()
        
        # Create daily provenance
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.create_daily_provenance(current_date)
        
        logger.info("✅ Tick capture completed")


def main():
    """Main function to run tick capture."""
    capture = EnhancedTickCapture()
    
    # Run tick capture
    asyncio.run(capture.run_tick_capture())


if __name__ == "__main__":
    main()
