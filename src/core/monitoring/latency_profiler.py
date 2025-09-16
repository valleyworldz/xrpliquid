"""
Latency Profiler - Stage Timings & Performance Monitoring
Implements comprehensive latency tracking with p50/p95/p99 metrics.
"""

import time
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
import threading


@dataclass
class LatencyMeasurement:
    """Represents a single latency measurement."""
    stage: str
    duration_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics for a stage."""
    stage: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    last_updated: datetime


class LatencyProfiler:
    """Comprehensive latency profiling and monitoring system."""
    
    def __init__(self, reports_dir: str = "reports", max_samples: int = 10000):
        self.reports_dir = Path(reports_dir)
        self.latency_dir = self.reports_dir / "latency"
        self.latency_dir.mkdir(parents=True, exist_ok=True)
        
        # Latency tracking
        self.measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.stage_stats: Dict[str, LatencyStats] = {}
        
        # SLO definitions
        self.slos = {
            "websocket_receive": {"p95_ms": 10.0, "p99_ms": 50.0},
            "order_processing": {"p95_ms": 20.0, "p99_ms": 100.0},
            "api_request": {"p95_ms": 100.0, "p99_ms": 500.0},
            "decision_engine": {"p95_ms": 50.0, "p99_ms": 200.0},
            "risk_calculation": {"p95_ms": 10.0, "p99_ms": 50.0},
            "total_loop": {"p95_ms": 250.0, "p99_ms": 500.0}
        }
        
        # Thread safety
        self._lock = threading.Lock()
    
    @contextmanager
    def measure_stage(self, stage: str, metadata: Dict[str, Any] = None):
        """Context manager for measuring stage latency."""
        start_time = time.perf_counter()
        start_timestamp = datetime.now(timezone.utc)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            measurement = LatencyMeasurement(
                stage=stage,
                duration_ms=duration_ms,
                timestamp=start_timestamp,
                metadata=metadata or {}
            )
            
            self.record_measurement(measurement)
    
    def record_measurement(self, measurement: LatencyMeasurement):
        """Record a latency measurement."""
        with self._lock:
            self.measurements[measurement.stage].append(measurement)
            self._update_stage_stats(measurement.stage)
    
    def _update_stage_stats(self, stage: str):
        """Update statistics for a stage."""
        measurements = list(self.measurements[stage])
        if not measurements:
            return
        
        durations = [m.duration_ms for m in measurements]
        
        stats = LatencyStats(
            stage=stage,
            count=len(durations),
            p50_ms=statistics.quantiles(durations, n=2)[0],
            p95_ms=statistics.quantiles(durations, n=20)[18],
            p99_ms=statistics.quantiles(durations, n=100)[98],
            min_ms=min(durations),
            max_ms=max(durations),
            mean_ms=statistics.mean(durations),
            last_updated=datetime.now(timezone.utc)
        )
        
        self.stage_stats[stage] = stats
    
    def check_slo_compliance(self, stage: str) -> Dict[str, Any]:
        """Check SLO compliance for a stage."""
        if stage not in self.slos or stage not in self.stage_stats:
            return {"stage": stage, "status": "unknown"}
        
        slo = self.slos[stage]
        stats = self.stage_stats[stage]
        
        p95_compliant = stats.p95_ms <= slo["p95_ms"]
        p99_compliant = stats.p99_ms <= slo["p99_ms"]
        overall_compliant = p95_compliant and p99_compliant
        
        return {
            "stage": stage,
            "status": "compliant" if overall_compliant else "violation",
            "p95_actual": stats.p95_ms,
            "p95_target": slo["p95_ms"],
            "p99_actual": stats.p99_ms,
            "p99_target": slo["p99_ms"],
            "measurement_count": stats.count
        }
    
    def generate_latency_report(self) -> Dict[str, Any]:
        """Generate comprehensive latency report."""
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stage_statistics": {
                stage: {
                    "count": stats.count,
                    "p50_ms": stats.p50_ms,
                    "p95_ms": stats.p95_ms,
                    "p99_ms": stats.p99_ms,
                    "min_ms": stats.min_ms,
                    "max_ms": stats.max_ms,
                    "mean_ms": stats.mean_ms,
                    "last_updated": stats.last_updated.isoformat()
                }
                for stage, stats in self.stage_stats.items()
            },
            "slo_compliance": {
                stage: self.check_slo_compliance(stage)
                for stage in self.slos.keys()
            }
        }
        
        # Save report
        report_file = self.latency_dir / f"latency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def create_prometheus_metrics(self) -> str:
        """Create Prometheus metrics format."""
        metrics = []
        
        for stage, stats in self.stage_stats.items():
            metrics.append(f"trading_latency_p95_ms{{stage=\"{stage}\"}} {stats.p95_ms}")
            metrics.append(f"trading_latency_p99_ms{{stage=\"{stage}\"}} {stats.p99_ms}")
            metrics.append(f"trading_latency_count{{stage=\"{stage}\"}} {stats.count}")
            
            compliance = self.check_slo_compliance(stage)
            slo_compliant = 1 if compliance["status"] == "compliant" else 0
            metrics.append(f"trading_slo_compliant{{stage=\"{stage}\"}} {slo_compliant}")
        
        return "\n".join(metrics)


def main():
    """Test latency profiler functionality."""
    profiler = LatencyProfiler()
    
    # Simulate some measurements
    import random
    
    for _ in range(100):
        with profiler.measure_stage("websocket_receive"):
            time.sleep(random.uniform(0.001, 0.01))  # 1-10ms
        
        with profiler.measure_stage("order_processing"):
            time.sleep(random.uniform(0.005, 0.02))  # 5-20ms
    
    # Generate report
    report = profiler.generate_latency_report()
    print(f"✅ Latency report generated: {len(report['stage_statistics'])} stages")
    
    # Check SLO compliance
    compliance = profiler.check_slo_compliance("websocket_receive")
    print(f"✅ SLO compliance: {compliance['status']}")
    
    print("✅ Latency profiler testing completed")


if __name__ == "__main__":
    main()