"""
Funding Schedule Capture - Hyperliquid Funding Rate Monitoring
Captures funding rate snapshots aligned to settlement windows.
"""

import json
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import time


@dataclass
class FundingData:
    """Represents funding rate data."""
    timestamp: datetime
    symbol: str
    funding_rate: float
    mark_price: float
    spot_price: float
    funding_time: datetime
    next_funding_time: datetime
    funding_interval_hours: int


class FundingLogger:
    """Logs funding rate snapshots from Hyperliquid API."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.funding_dir = self.data_dir / "funding"
        self.warehouse_dir = self.data_dir / "warehouse"
        
        # Create directories
        self.funding_dir.mkdir(parents=True, exist_ok=True)
        self.warehouse_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.api_url = "https://api.hyperliquid.xyz/info"
        self.symbols = ["XRP"]
        self.funding_interval_hours = 1  # Hyperliquid uses 1-hour funding
        
        # Capture state
        self.funding_data = []
        self.start_time = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def fetch_funding_data(self) -> Optional[FundingData]:
        """Fetch current funding data from Hyperliquid API."""
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get funding rates
                funding_url = f"{self.api_url}?type=meta"
                async with session.get(funding_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract funding rate for XRP
                        for coin_info in data.get("universe", []):
                            if coin_info.get("name") == "XRP":
                                funding_rate = coin_info.get("funding", 0.0)
                                break
                        else:
                            funding_rate = 0.0
                
                # Get mark prices
                mark_url = f"{self.api_url}?type=allMids"
                async with session.get(mark_url) as response:
                    if response.status == 200:
                        mark_data = await response.json()
                        mark_price = mark_data.get("XRP", 0.0)
                    else:
                        mark_price = 0.0
                
                # Get spot prices (approximate with mark price for now)
                spot_price = mark_price
                
                # Calculate funding times
                now = datetime.now(timezone.utc)
                current_hour = now.replace(minute=0, second=0, microsecond=0)
                next_funding_time = current_hour + timedelta(hours=1)
                
                return FundingData(
                    timestamp=now,
                    symbol="XRP",
                    funding_rate=funding_rate,
                    mark_price=mark_price,
                    spot_price=spot_price,
                    funding_time=current_hour,
                    next_funding_time=next_funding_time,
                    funding_interval_hours=self.funding_interval_hours
                )
                
        except Exception as e:
            self.logger.error(f"Failed to fetch funding data: {e}")
            return None
    
    def _get_current_file_path(self) -> Path:
        """Get current funding file path based on date."""
        
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filename = f"{current_date}_xrp.json"
        return self.funding_dir / filename
    
    def _save_funding_data(self, funding_data: FundingData):
        """Save funding data to JSON file."""
        
        try:
            file_path = self._get_current_file_path()
            
            # Convert to dictionary
            funding_dict = {
                "timestamp": funding_data.timestamp.isoformat(),
                "symbol": funding_data.symbol,
                "funding_rate": funding_data.funding_rate,
                "mark_price": funding_data.mark_price,
                "spot_price": funding_data.spot_price,
                "funding_time": funding_data.funding_time.isoformat(),
                "next_funding_time": funding_data.next_funding_time.isoformat(),
                "funding_interval_hours": funding_data.funding_interval_hours
            }
            
            # Load existing data or create new list
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # Add new data
            existing_data.append(funding_dict)
            
            # Save updated data
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            self.funding_data.append(funding_data)
            self.logger.info(f"Saved funding data: {funding_data.funding_rate:.6f} at {funding_data.timestamp}")
            
        except Exception as e:
            self.logger.error(f"Failed to save funding data: {e}")
    
    async def log_funding_rates(self, interval_minutes: int = 5):
        """Continuously log funding rates at specified interval."""
        
        self.start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting funding rate logging (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                # Fetch funding data
                funding_data = await self.fetch_funding_data()
                
                if funding_data:
                    self._save_funding_data(funding_data)
                    
                    # Log funding rate info
                    self.logger.info(
                        f"Funding rate: {funding_data.funding_rate:.6f} "
                        f"(Mark: ${funding_data.mark_price:.4f}, "
                        f"Next funding: {funding_data.next_funding_time.strftime('%H:%M UTC')})"
                    )
                else:
                    self.logger.warning("Failed to fetch funding data")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in funding logger: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def create_funding_summary(self, date: str = None) -> Dict[str, Any]:
        """Create funding summary for a specific date."""
        
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        file_path = self.funding_dir / f"{date}_xrp.json"
        
        if not file_path.exists():
            return {"error": f"No funding data found for {date}"}
        
        try:
            with open(file_path, 'r') as f:
                funding_records = json.load(f)
            
            if not funding_records:
                return {"error": "No funding records found"}
            
            # Calculate statistics
            funding_rates = [record["funding_rate"] for record in funding_records]
            mark_prices = [record["mark_price"] for record in funding_records]
            
            summary = {
                "date": date,
                "total_records": len(funding_records),
                "funding_rate_stats": {
                    "min": min(funding_rates),
                    "max": max(funding_rates),
                    "mean": sum(funding_rates) / len(funding_rates),
                    "std": pd.Series(funding_rates).std() if len(funding_rates) > 1 else 0.0
                },
                "mark_price_stats": {
                    "min": min(mark_prices),
                    "max": max(mark_prices),
                    "mean": sum(mark_prices) / len(mark_prices),
                    "std": pd.Series(mark_prices).std() if len(mark_prices) > 1 else 0.0
                },
                "funding_events": len([r for r in funding_records if r["funding_rate"] != 0.0]),
                "capture_period": {
                    "start": funding_records[0]["timestamp"],
                    "end": funding_records[-1]["timestamp"]
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to create funding summary: {e}")
            return {"error": str(e)}
    
    def create_provenance_record(self, date: str = None):
        """Create provenance record for funding data."""
        
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        date_dir = self.warehouse_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Get funding summary
        funding_summary = self.create_funding_summary(date)
        
        # Calculate runtime
        runtime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600 if self.start_time else 0
        
        provenance = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "hyperliquid_funding_api",
            "endpoints": [self.api_url],
            "timezone": "UTC",
            "symbols": self.symbols,
            "funding_interval_hours": self.funding_interval_hours,
            "capture_period": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "runtime_hours": runtime_hours
            },
            "data_quality": {
                "total_records": funding_summary.get("total_records", 0),
                "funding_events": funding_summary.get("funding_events", 0),
                "capture_rate": funding_summary.get("total_records", 0) / max(runtime_hours * 12, 1)  # Expected 12 captures per hour (5-min intervals)
            },
            "funding_summary": funding_summary,
            "files": {
                "funding_file": f"{date}_xrp.json"
            },
            "schema_version": "1.0",
            "capture_method": "api_polling"
        }
        
        # Save provenance
        provenance_file = date_dir / "_provenance.json"
        with open(provenance_file, 'w') as f:
            json.dump(provenance, f, indent=2)
        
        self.logger.info(f"Created funding provenance record: {provenance_file}")
        return provenance


async def main():
    """Main function to run funding logger."""
    logger = FundingLogger()
    
    try:
        # Start logging funding rates every 5 minutes
        await logger.log_funding_rates(interval_minutes=5)
    except KeyboardInterrupt:
        logger.logger.info("Stopping funding logger...")
        
        # Create provenance record
        logger.create_provenance_record()
        
        logger.logger.info("Funding logger stopped")


if __name__ == "__main__":
    asyncio.run(main())
