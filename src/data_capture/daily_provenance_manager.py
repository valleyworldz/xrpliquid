"""
Daily Data Provenance Manager
Manages daily data partitions with provenance tracking and freshness monitoring.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DailyProvenanceManager:
    """Manages daily data provenance and freshness monitoring."""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.ticks_dir = self.data_root / "ticks"
        self.funding_dir = self.data_root / "funding"
        
        # Create directories
        self.ticks_dir.mkdir(parents=True, exist_ok=True)
        self.funding_dir.mkdir(parents=True, exist_ok=True)
    
    def create_daily_partition(self, date: str = None) -> Dict:
        """Create daily partition with provenance metadata."""
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        partition_info = {
            "date": date,
            "created_at": datetime.now().isoformat(),
            "partitions": {}
        }
        
        # Create ticks partition
        ticks_partition = self.ticks_dir / date
        ticks_partition.mkdir(exist_ok=True)
        
        ticks_provenance = {
            "data_type": "tick_data",
            "source": "hyperliquid_websocket",
            "endpoints": [
                "wss://api.hyperliquid.xyz/ws",
                "wss://api.hyperliquid.xyz/ws/trades"
            ],
            "symbols": ["XRP"],
            "timezone": "UTC",
            "missing_data_policy": "interpolate",
            "resampling_rules": "1s_ohlc",
            "expected_frequency_hz": 10,
            "compression": "gzip",
            "format": "jsonl"
        }
        
        # Create funding partition
        funding_partition = self.funding_dir / date
        funding_partition.mkdir(exist_ok=True)
        
        funding_provenance = {
            "data_type": "funding_rates",
            "source": "hyperliquid_api",
            "endpoints": [
                "https://api.hyperliquid.xyz/info",
                "https://api.hyperliquid.xyz/fundingHistory"
            ],
            "symbols": ["XRP"],
            "timezone": "UTC",
            "missing_data_policy": "forward_fill",
            "funding_interval_hours": 1,
            "expected_updates_per_day": 24,
            "format": "json"
        }
        
        # Save provenance files
        ticks_provenance_file = ticks_partition / "_provenance.json"
        with open(ticks_provenance_file, 'w') as f:
            json.dump(ticks_provenance, f, indent=2)
        
        funding_provenance_file = funding_partition / "_provenance.json"
        with open(funding_provenance_file, 'w') as f:
            json.dump(funding_provenance, f, indent=2)
        
        partition_info["partitions"]["ticks"] = {
            "path": str(ticks_partition),
            "provenance_file": str(ticks_provenance_file)
        }
        
        partition_info["partitions"]["funding"] = {
            "path": str(funding_partition),
            "provenance_file": str(funding_provenance_file)
        }
        
        logger.info(f"Created daily partition for {date}")
        return partition_info
    
    def check_data_freshness(self, date: str = None) -> Dict:
        """Check data freshness for a given date."""
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        freshness_report = {
            "date": date,
            "checked_at": datetime.now().isoformat(),
            "partitions": {},
            "overall_status": "unknown"
        }
        
        # Check ticks freshness
        ticks_partition = self.ticks_dir / date
        ticks_freshness = self._check_partition_freshness(ticks_partition, "ticks")
        freshness_report["partitions"]["ticks"] = ticks_freshness
        
        # Check funding freshness
        funding_partition = self.funding_dir / date
        funding_freshness = self._check_partition_freshness(funding_partition, "funding")
        freshness_report["partitions"]["funding"] = funding_freshness
        
        # Determine overall status
        all_fresh = all(
            partition["status"] == "fresh" 
            for partition in freshness_report["partitions"].values()
        )
        
        freshness_report["overall_status"] = "fresh" if all_fresh else "stale"
        
        return freshness_report
    
    def _check_partition_freshness(self, partition_path: Path, data_type: str) -> Dict:
        """Check freshness of a specific partition."""
        
        if not partition_path.exists():
            return {
                "status": "missing",
                "message": f"Partition {data_type} does not exist",
                "last_update": None,
                "age_hours": None
            }
        
        # Check for data files
        data_files = list(partition_path.glob("*.json*")) + list(partition_path.glob("*.csv*"))
        
        if not data_files:
            return {
                "status": "empty",
                "message": f"Partition {data_type} exists but has no data files",
                "last_update": None,
                "age_hours": None
            }
        
        # Find most recent file
        most_recent_file = max(data_files, key=lambda f: f.stat().st_mtime)
        last_update = datetime.fromtimestamp(most_recent_file.stat().st_mtime)
        age_hours = (datetime.now() - last_update).total_seconds() / 3600
        
        # Determine freshness based on data type
        if data_type == "ticks":
            # Ticks should be updated within 1 hour
            max_age_hours = 1.0
        elif data_type == "funding":
            # Funding should be updated within 2 hours
            max_age_hours = 2.0
        else:
            max_age_hours = 24.0
        
        status = "fresh" if age_hours <= max_age_hours else "stale"
        
        return {
            "status": status,
            "message": f"Partition {data_type} is {status}",
            "last_update": last_update.isoformat(),
            "age_hours": round(age_hours, 2),
            "max_age_hours": max_age_hours,
            "most_recent_file": most_recent_file.name
        }
    
    def generate_freshness_badge(self, freshness_report: Dict) -> str:
        """Generate HTML badge for data freshness."""
        
        status = freshness_report["overall_status"]
        date = freshness_report["date"]
        
        if status == "fresh":
            badge_color = "#2E8B57"  # Green
            badge_text = "Data Freshness OK"
        elif status == "stale":
            badge_color = "#FF8C00"  # Orange
            badge_text = "Data Stale"
        else:
            badge_color = "#DC143C"  # Red
            badge_text = "Data Missing"
        
        badge_html = f"""
        <div style="display: inline-block; background-color: {badge_color}; color: white; 
                    padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; 
                    margin: 2px;">
            {badge_text} ({date})
        </div>
        """
        
        return badge_html
    
    def create_sample_data(self, date: str = None) -> Dict:
        """Create sample data files for testing."""
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Create sample tick data
        ticks_partition = self.ticks_dir / date
        ticks_partition.mkdir(exist_ok=True)
        
        sample_ticks = []
        base_time = datetime.strptime(f"{date} 00:00:00", "%Y-%m-%d %H:%M:%S")
        
        for i in range(100):
            tick_time = base_time + timedelta(seconds=i*60)  # Every minute
            sample_ticks.append({
                "timestamp": tick_time.isoformat(),
                "price": 0.52 + (i % 10) * 0.001,
                "qty": 100 + (i % 5) * 50,
                "side": "buy" if i % 2 == 0 else "sell"
            })
        
        ticks_file = ticks_partition / f"xrp_ticks_{date}.jsonl"
        with open(ticks_file, 'w') as f:
            for tick in sample_ticks:
                f.write(json.dumps(tick) + "\n")
        
        # Create sample funding data
        funding_partition = self.funding_dir / date
        funding_partition.mkdir(exist_ok=True)
        
        sample_funding = []
        for hour in range(24):
            funding_time = base_time + timedelta(hours=hour)
            sample_funding.append({
                "timestamp": funding_time.isoformat(),
                "funding_rate": 0.0001 + (hour % 3) * 0.00005,
                "mark_price": 0.52 + (hour % 10) * 0.001,
                "index_price": 0.52 + (hour % 8) * 0.0008
            })
        
        funding_file = funding_partition / f"xrp_funding_{date}.json"
        with open(funding_file, 'w') as f:
            json.dump(sample_funding, f, indent=2)
        
        logger.info(f"Created sample data for {date}")
        
        return {
            "ticks_file": str(ticks_file),
            "funding_file": str(funding_file),
            "ticks_count": len(sample_ticks),
            "funding_count": len(sample_funding)
        }
    
    def run_daily_provenance_check(self) -> Dict:
        """Run complete daily provenance check."""
        
        # Create today's partition
        partition_info = self.create_daily_partition()
        
        # Create sample data for demonstration
        sample_data = self.create_sample_data()
        
        # Check freshness
        freshness_report = self.check_data_freshness()
        
        # Generate badge
        badge_html = self.generate_freshness_badge(freshness_report)
        
        # Save reports
        reports_dir = Path("reports/data_provenance")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save partition info
        partition_file = reports_dir / f"partition_info_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(partition_file, 'w') as f:
            json.dump(partition_info, f, indent=2)
        
        # Save freshness report
        freshness_file = reports_dir / f"freshness_report_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(freshness_file, 'w') as f:
            json.dump(freshness_report, f, indent=2)
        
        # Save badge HTML
        badge_file = reports_dir / "data_freshness_badge.html"
        with open(badge_file, 'w') as f:
            f.write(badge_html)
        
        logger.info("Daily provenance check completed")
        
        return {
            "partition_info": partition_info,
            "freshness_report": freshness_report,
            "badge_html": badge_html,
            "sample_data": sample_data
        }

def main():
    """Run daily provenance management."""
    manager = DailyProvenanceManager()
    results = manager.run_daily_provenance_check()
    
    print("✅ Daily provenance management completed")
    print(f"   Overall status: {results['freshness_report']['overall_status']}")
    print(f"   Ticks status: {results['freshness_report']['partitions']['ticks']['status']}")
    print(f"   Funding status: {results['freshness_report']['partitions']['funding']['status']}")
    
    return 0

if __name__ == "__main__":
    exit(main())
