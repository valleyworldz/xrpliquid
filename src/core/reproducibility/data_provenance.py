"""
Data Provenance Tracker
Maintains lineage and provenance of all data sources and transformations.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib


class DataProvenance:
    """Tracks data lineage and provenance for audit trails."""
    
    def __init__(self, data_dir: str = "data/warehouse"):
        self.data_dir = Path(data_dir)
        self.provenance_dir = self.data_dir / "provenance"
        self.provenance_dir.mkdir(parents=True, exist_ok=True)
    
    def create_data_snapshot(self, 
                           exchange: str,
                           symbol: str,
                           data_type: str,
                           start_time: datetime,
                           end_time: datetime,
                           source_endpoints: List[str],
                           missing_candle_policy: str = "forward_fill",
                           resampling_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a data snapshot with full provenance."""
        
        snapshot_id = f"{exchange}_{symbol}_{data_type}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        snapshot_dir = self.data_dir / datetime.now().strftime('%Y-%m-%d') / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        provenance_record = {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
            "symbol": symbol,
            "data_type": data_type,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "timezone": str(start_time.tzinfo) if start_time.tzinfo else "UTC"
            },
            "source_endpoints": source_endpoints,
            "data_policies": {
                "missing_candle_policy": missing_candle_policy,
                "resampling_rules": resampling_rules or {},
                "timezone_handling": "UTC_normalized"
            },
            "quality_checks": {
                "duplicate_removal": True,
                "gap_detection": True,
                "outlier_detection": True,
                "volume_validation": True
            },
            "file_locations": {
                "raw_data": str(snapshot_dir / "raw_data.csv"),
                "processed_data": str(snapshot_dir / "processed_data.parquet"),
                "quality_report": str(snapshot_dir / "quality_report.json")
            }
        }
        
        # Save provenance record
        provenance_file = self.provenance_dir / f"{snapshot_id}_provenance.json"
        with open(provenance_file, 'w') as f:
            json.dump(provenance_record, f, indent=2)
        
        return provenance_record
    
    def add_transformation_step(self, 
                              snapshot_id: str,
                              transformation_name: str,
                              input_files: List[str],
                              output_files: List[str],
                              parameters: Dict[str, Any],
                              validation_checks: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a transformation step to the provenance chain."""
        
        transformation_record = {
            "transformation_id": f"{snapshot_id}_{transformation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "snapshot_id": snapshot_id,
            "transformation_name": transformation_name,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "input_files": input_files,
            "output_files": output_files,
            "parameters": parameters,
            "validation_checks": validation_checks or {},
            "checksums": {
                "input_checksums": {f: self._calculate_file_hash(f) for f in input_files},
                "output_checksums": {f: self._calculate_file_hash(f) for f in output_files}
            }
        }
        
        # Load existing provenance
        provenance_file = self.provenance_dir / f"{snapshot_id}_provenance.json"
        if provenance_file.exists():
            with open(provenance_file, 'r') as f:
                provenance = json.load(f)
        else:
            provenance = {"snapshot_id": snapshot_id, "transformations": []}
        
        # Add transformation
        if "transformations" not in provenance:
            provenance["transformations"] = []
        
        provenance["transformations"].append(transformation_record)
        
        # Save updated provenance
        with open(provenance_file, 'w') as f:
            json.dump(provenance, f, indent=2)
        
        return transformation_record
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception:
            return "FILE_NOT_FOUND"
    
    def create_walk_forward_splits(self, 
                                 start_date: datetime,
                                 end_date: datetime,
                                 train_window_days: int = 252,
                                 test_window_days: int = 30,
                                 step_size_days: int = 30) -> Dict[str, Any]:
        """Create walk-forward analysis splits with full provenance."""
        
        splits = []
        current_start = start_date
        
        while current_start + pd.Timedelta(days=train_window_days + test_window_days) <= end_date:
            train_start = current_start
            train_end = train_start + pd.Timedelta(days=train_window_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=test_window_days)
            
            split_record = {
                "split_id": f"wf_{train_start.strftime('%Y%m%d')}_{test_end.strftime('%Y%m%d')}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "train_period": {
                    "start": train_start.isoformat(),
                    "end": train_end.isoformat(),
                    "duration_days": train_window_days
                },
                "test_period": {
                    "start": test_start.isoformat(),
                    "end": test_end.isoformat(),
                    "duration_days": test_window_days
                },
                "rationale": f"Walk-forward split {len(splits) + 1}: {train_window_days} day train, {test_window_days} day test",
                "wall_clock_timestamps": {
                    "split_created": datetime.now(timezone.utc).isoformat(),
                    "train_start_wall_clock": train_start.isoformat(),
                    "test_start_wall_clock": test_start.isoformat()
                }
            }
            
            splits.append(split_record)
            current_start += pd.Timedelta(days=step_size_days)
        
        splits_manifest = {
            "splits_created_at": datetime.now(timezone.utc).isoformat(),
            "total_splits": len(splits),
            "parameters": {
                "train_window_days": train_window_days,
                "test_window_days": test_window_days,
                "step_size_days": step_size_days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "splits": splits
        }
        
        # Save splits manifest
        splits_file = Path("reports/splits/train_test_splits.json")
        splits_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(splits_file, 'w') as f:
            json.dump(splits_manifest, f, indent=2)
        
        return splits_manifest
    
    def track_clock_skew(self, 
                        exchange_timestamp: datetime,
                        system_timestamp: datetime,
                        threshold_ms: float = 100.0) -> Dict[str, Any]:
        """Track clock skew between exchange and system."""
        
        skew_ms = (system_timestamp - exchange_timestamp).total_seconds() * 1000
        
        clock_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange_timestamp": exchange_timestamp.isoformat(),
            "system_timestamp": system_timestamp.isoformat(),
            "skew_ms": skew_ms,
            "threshold_ms": threshold_ms,
            "within_threshold": abs(skew_ms) <= threshold_ms,
            "action_required": abs(skew_ms) > threshold_ms
        }
        
        # Save to clock skew log
        clock_file = Path("reports/latency/clock_skew_ms.json")
        clock_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing records
        if clock_file.exists():
            with open(clock_file, 'r') as f:
                clock_records = json.load(f)
        else:
            clock_records = {"records": []}
        
        clock_records["records"].append(clock_record)
        
        # Keep only last 1000 records
        if len(clock_records["records"]) > 1000:
            clock_records["records"] = clock_records["records"][-1000:]
        
        with open(clock_file, 'w') as f:
            json.dump(clock_records, f, indent=2)
        
        return clock_record
    
    def validate_data_integrity(self, file_path: str) -> Dict[str, Any]:
        """Validate data integrity and quality."""
        
        validation_result = {
            "file_path": file_path,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }
        
        try:
            # Load data
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Basic checks
            validation_result["checks"]["row_count"] = len(df)
            validation_result["checks"]["column_count"] = len(df.columns)
            validation_result["checks"]["memory_usage_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Data quality checks
            validation_result["checks"]["duplicates"] = df.duplicated().sum()
            validation_result["checks"]["null_values"] = df.isnull().sum().to_dict()
            validation_result["checks"]["data_types"] = df.dtypes.astype(str).to_dict()
            
            # Time series specific checks
            if 'ts' in df.columns:
                validation_result["checks"]["time_range"] = {
                    "start": pd.to_datetime(df['ts']).min().isoformat(),
                    "end": pd.to_datetime(df['ts']).max().isoformat()
                }
                validation_result["checks"]["time_gaps"] = self._detect_time_gaps(df['ts'])
            
            # Financial data checks
            if 'price' in df.columns:
                validation_result["checks"]["price_stats"] = {
                    "min": safe_float(df['price'].min()),
                    "max": safe_float(df['price'].max()),
                    "mean": safe_float(df['price'].mean()),
                    "std": safe_float(df['price'].std())
                }
                validation_result["checks"]["negative_prices"] = (df['price'] < 0).sum()
            
            validation_result["valid"] = True
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["error"] = str(e)
        
        return validation_result
    
    def _detect_time_gaps(self, timestamps: pd.Series) -> Dict[str, Any]:
        """Detect gaps in time series data."""
        ts_series = pd.to_datetime(timestamps).sort_values()
        time_diffs = ts_series.diff().dropna()
        
        # Define expected frequency (assuming 1-minute data)
        expected_freq = pd.Timedelta(minutes=1)
        
        gaps = time_diffs[time_diffs > expected_freq * 2]  # More than 2x expected frequency
        
        return {
            "total_gaps": len(gaps),
            "largest_gap_minutes": safe_float(gaps.max().total_seconds() / 60) if len(gaps) > 0 else 0,
            "gap_locations": gaps.index.tolist()[:10]  # First 10 gaps
        }


def main():
    """Test data provenance functionality."""
    provenance = DataProvenance()
    
    # Create a sample data snapshot
    snapshot = provenance.create_data_snapshot(
        exchange="hyperliquid",
        symbol="XRP",
        data_type="trades",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        source_endpoints=["https://api.hyperliquid.xyz/info"],
        missing_candle_policy="forward_fill"
    )
    
    print(f"✅ Data snapshot created: {snapshot['snapshot_id']}")
    
    # Create walk-forward splits
    splits = provenance.create_walk_forward_splits(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        train_window_days=252,
        test_window_days=30
    )
    
    print(f"✅ Walk-forward splits created: {splits['total_splits']} splits")
    
    # Track clock skew
    clock_record = provenance.track_clock_skew(
        exchange_timestamp=datetime.now(),
        system_timestamp=datetime.now()
    )
    
    print(f"✅ Clock skew tracked: {clock_record['skew_ms']:.2f}ms")


if __name__ == "__main__":
    main()
