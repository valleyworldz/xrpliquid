"""
Data Lineage Manager - Immutable Snapshots & Provenance
Manages immutable data snapshots with complete provenance tracking.
"""

import json
import hashlib
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DataProvenance:
    """Represents data provenance information."""
    timestamp: datetime
    source: str
    endpoints: List[str]
    timezone: str
    missing_candle_policy: str
    resampling_rules: Dict[str, Any]
    data_hash: str
    record_count: int
    schema_version: str


@dataclass
class WalkForwardSplit:
    """Represents a walk-forward split."""
    split_id: str
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    gap_days: int
    train_samples: int
    test_samples: int
    data_hash: str


class DataLineageManager:
    """Manages data lineage and immutable snapshots."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.data_dir = Path("data")
        self.warehouse_dir = self.data_dir / "warehouse"
        self.splits_dir = self.reports_dir / "splits"
        
        # Create directories
        self.warehouse_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
    
    def create_immutable_snapshot(self, 
                                data: pd.DataFrame,
                                source: str,
                                endpoints: List[str],
                                timezone_str: str = "UTC",
                                missing_candle_policy: str = "forward_fill",
                                resampling_rules: Dict[str, Any] = None,
                                schema_version: str = "1.0") -> str:
        """Create immutable data snapshot with provenance."""
        
        timestamp = datetime.now(timezone.utc)
        date_str = timestamp.strftime("%Y-%m-%d")
        
        # Create date directory
        date_dir = self.warehouse_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate snapshot ID
        snapshot_id = f"{source}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate data hash
        data_hash = self._calculate_dataframe_hash(data)
        
        # Create provenance
        provenance = DataProvenance(
            timestamp=timestamp,
            source=source,
            endpoints=endpoints,
            timezone=timezone_str,
            missing_candle_policy=missing_candle_policy,
            resampling_rules=resampling_rules or {},
            data_hash=data_hash,
            record_count=len(data),
            schema_version=schema_version
        )
        
        # Save data snapshot
        data_file = date_dir / f"{snapshot_id}.parquet"
        data.to_parquet(data_file, index=False)
        
        # Save provenance
        provenance_file = date_dir / f"{snapshot_id}_provenance.json"
        provenance_dict = {
            "timestamp": provenance.timestamp.isoformat(),
            "source": provenance.source,
            "endpoints": provenance.endpoints,
            "timezone": provenance.timezone,
            "missing_candle_policy": provenance.missing_candle_policy,
            "resampling_rules": provenance.resampling_rules,
            "data_hash": provenance.data_hash,
            "record_count": provenance.record_count,
            "schema_version": provenance.schema_version,
            "data_file": str(data_file),
            "snapshot_id": snapshot_id
        }
        
        with open(provenance_file, 'w') as f:
            json.dump(provenance_dict, f, indent=2)
        
        return str(data_file)
    
    def create_walk_forward_splits(self, 
                                 data: pd.DataFrame,
                                 time_column: str,
                                 target_column: str,
                                 train_days: int = 30,
                                 test_days: int = 7,
                                 gap_days: int = 1,
                                 min_train_samples: int = 1000) -> List[WalkForwardSplit]:
        """Create walk-forward splits for backtesting."""
        
        # Sort data by time
        data_sorted = data.sort_values(time_column)
        
        # Get date range
        start_date = pd.to_datetime(data_sorted[time_column].min())
        end_date = pd.to_datetime(data_sorted[time_column].max())
        
        splits = []
        current_date = start_date
        
        split_id = 0
        
        while current_date + timedelta(days=train_days + test_days + gap_days) <= end_date:
            # Calculate split dates
            train_start = current_date
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=gap_days)
            test_end = test_start + timedelta(days=test_days)
            
            # Create masks
            train_mask = (data_sorted[time_column] >= train_start) & (data_sorted[time_column] < train_end)
            test_mask = (data_sorted[time_column] >= test_start) & (data_sorted[time_column] < test_end)
            
            # Extract splits
            train_data = data_sorted[train_mask]
            test_data = data_sorted[test_mask]
            
            # Check minimum samples
            if len(train_data) >= min_train_samples and len(test_data) > 0:
                # Calculate data hashes
                train_hash = self._calculate_dataframe_hash(train_data)
                test_hash = self._calculate_dataframe_hash(test_data)
                
                # Create split
                split = WalkForwardSplit(
                    split_id=f"split_{split_id:03d}",
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    gap_days=gap_days,
                    train_samples=len(train_data),
                    test_samples=len(test_data),
                    data_hash=f"{train_hash}_{test_hash}"
                )
                
                splits.append(split)
                split_id += 1
            
            # Move to next split
            current_date += timedelta(days=test_days)
        
        # Save splits
        self._save_walk_forward_splits(splits)
        
        return splits
    
    def _save_walk_forward_splits(self, splits: List[WalkForwardSplit]):
        """Save walk-forward splits to file."""
        
        splits_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_splits": len(splits),
            "splits": [
                {
                    "split_id": split.split_id,
                    "train_start": split.train_start.isoformat(),
                    "train_end": split.train_end.isoformat(),
                    "test_start": split.test_start.isoformat(),
                    "test_end": split.test_end.isoformat(),
                    "gap_days": split.gap_days,
                    "train_samples": split.train_samples,
                    "test_samples": split.test_samples,
                    "data_hash": split.data_hash
                }
                for split in splits
            ]
        }
        
        splits_file = self.splits_dir / "train_test_splits.json"
        with open(splits_file, 'w') as f:
            json.dump(splits_data, f, indent=2)
    
    def verify_data_integrity(self, snapshot_file: str) -> Dict[str, Any]:
        """Verify data integrity of a snapshot."""
        
        snapshot_path = Path(snapshot_file)
        if not snapshot_path.exists():
            return {
                "status": "error",
                "message": f"Snapshot file not found: {snapshot_file}"
            }
        
        # Load data
        try:
            data = pd.read_parquet(snapshot_path)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load snapshot: {str(e)}"
            }
        
        # Calculate current hash
        current_hash = self._calculate_dataframe_hash(data)
        
        # Load provenance
        provenance_file = snapshot_path.parent / f"{snapshot_path.stem}_provenance.json"
        if not provenance_file.exists():
            return {
                "status": "error",
                "message": f"Provenance file not found: {provenance_file}"
            }
        
        with open(provenance_file, 'r') as f:
            provenance = json.load(f)
        
        # Verify hash
        hash_match = current_hash == provenance["data_hash"]
        
        return {
            "status": "verified" if hash_match else "corrupted",
            "snapshot_file": str(snapshot_path),
            "provenance_file": str(provenance_file),
            "current_hash": current_hash,
            "stored_hash": provenance["data_hash"],
            "hash_match": hash_match,
            "record_count": len(data),
            "expected_record_count": provenance["record_count"],
            "record_count_match": len(data) == provenance["record_count"]
        }
    
    def get_data_lineage(self, snapshot_file: str) -> Dict[str, Any]:
        """Get complete data lineage for a snapshot."""
        
        snapshot_path = Path(snapshot_file)
        provenance_file = snapshot_path.parent / f"{snapshot_path.stem}_provenance.json"
        
        if not provenance_file.exists():
            return {
                "status": "error",
                "message": f"Provenance file not found: {provenance_file}"
            }
        
        with open(provenance_file, 'r') as f:
            provenance = json.load(f)
        
        # Get parent snapshots (if any)
        parent_snapshots = self._find_parent_snapshots(provenance)
        
        # Get child snapshots (if any)
        child_snapshots = self._find_child_snapshots(provenance)
        
        return {
            "snapshot_id": provenance["snapshot_id"],
            "timestamp": provenance["timestamp"],
            "source": provenance["source"],
            "endpoints": provenance["endpoints"],
            "timezone": provenance["timezone"],
            "missing_candle_policy": provenance["missing_candle_policy"],
            "resampling_rules": provenance["resampling_rules"],
            "data_hash": provenance["data_hash"],
            "record_count": provenance["record_count"],
            "schema_version": provenance["schema_version"],
            "parent_snapshots": parent_snapshots,
            "child_snapshots": child_snapshots,
            "lineage_depth": len(parent_snapshots)
        }
    
    def _find_parent_snapshots(self, provenance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find parent snapshots in the lineage."""
        
        # This is a simplified implementation
        # In practice, you'd track parent-child relationships explicitly
        
        parent_snapshots = []
        
        # Look for snapshots from the same source with earlier timestamps
        source = provenance["source"]
        timestamp = datetime.fromisoformat(provenance["timestamp"])
        
        # Search in warehouse directory
        for date_dir in self.warehouse_dir.iterdir():
            if date_dir.is_dir():
                for file_path in date_dir.glob(f"{source}_*_provenance.json"):
                    try:
                        with open(file_path, 'r') as f:
                            parent_provenance = json.load(f)
                        
                        parent_timestamp = datetime.fromisoformat(parent_provenance["timestamp"])
                        
                        if parent_timestamp < timestamp:
                            parent_snapshots.append({
                                "snapshot_id": parent_provenance["snapshot_id"],
                                "timestamp": parent_provenance["timestamp"],
                                "data_hash": parent_provenance["data_hash"],
                                "record_count": parent_provenance["record_count"]
                            })
                    except Exception:
                        continue
        
        # Sort by timestamp
        parent_snapshots.sort(key=lambda x: x["timestamp"])
        
        return parent_snapshots
    
    def _find_child_snapshots(self, provenance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find child snapshots in the lineage."""
        
        child_snapshots = []
        
        # Look for snapshots from the same source with later timestamps
        source = provenance["source"]
        timestamp = datetime.fromisoformat(provenance["timestamp"])
        
        # Search in warehouse directory
        for date_dir in self.warehouse_dir.iterdir():
            if date_dir.is_dir():
                for file_path in date_dir.glob(f"{source}_*_provenance.json"):
                    try:
                        with open(file_path, 'r') as f:
                            child_provenance = json.load(f)
                        
                        child_timestamp = datetime.fromisoformat(child_provenance["timestamp"])
                        
                        if child_timestamp > timestamp:
                            child_snapshots.append({
                                "snapshot_id": child_provenance["snapshot_id"],
                                "timestamp": child_provenance["timestamp"],
                                "data_hash": child_provenance["data_hash"],
                                "record_count": child_provenance["record_count"]
                            })
                    except Exception:
                        continue
        
        # Sort by timestamp
        child_snapshots.sort(key=lambda x: x["timestamp"])
        
        return child_snapshots
    
    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Calculate SHA-256 hash of a DataFrame."""
        
        # Convert DataFrame to string representation
        df_str = df.to_string()
        
        # Calculate hash
        return hashlib.sha256(df_str.encode()).hexdigest()


def main():
    """Test data lineage manager functionality."""
    lineage_manager = DataLineageManager()
    
    # Create sample data
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Create immutable snapshot
    snapshot_file = lineage_manager.create_immutable_snapshot(
        data=sample_data,
        source="hyperliquid_xrp_trades",
        endpoints=["https://api.hyperliquid.xyz/info", "https://api.hyperliquid.xyz/exchange"],
        timezone="UTC",
        missing_candle_policy="forward_fill",
        resampling_rules={"frequency": "1H", "method": "ohlc"},
        schema_version="1.0"
    )
    
    print(f"✅ Created immutable snapshot: {snapshot_file}")
    
    # Create walk-forward splits
    splits = lineage_manager.create_walk_forward_splits(
        data=sample_data,
        time_column="timestamp",
        target_column="price",
        train_days=7,
        test_days=1,
        gap_days=0
    )
    
    print(f"✅ Created {len(splits)} walk-forward splits")
    
    # Verify data integrity
    integrity_result = lineage_manager.verify_data_integrity(snapshot_file)
    print(f"✅ Data integrity verification: {integrity_result['status']}")
    
    # Get data lineage
    lineage = lineage_manager.get_data_lineage(snapshot_file)
    print(f"✅ Data lineage: {lineage['lineage_depth']} parent snapshots")
    
    print("✅ Data lineage manager testing completed")


if __name__ == "__main__":
    main()
