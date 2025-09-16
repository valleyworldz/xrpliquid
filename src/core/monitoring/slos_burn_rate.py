"""
SLOs & Burn-Rate Alerts - Service Level Objectives and Performance Monitoring
Implements measurable service reliability with latency and error rate monitoring.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time


class SLOStatus(Enum):
    """SLO status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACHED = "breached"


@dataclass
class SLOMetric:
    """SLO metric definition."""
    name: str
    target: float
    window_minutes: int
    current_value: float
    status: SLOStatus
    breach_count: int
    last_breach: Optional[datetime]


@dataclass
class BurnRateAlert:
    """Burn rate alert definition."""
    metric_name: str
    current_rate: float
    threshold_rate: float
    time_window_minutes: int
    alert_level: str
    triggered_at: datetime


class SLOsBurnRateMonitor:
    """Monitors SLOs and burn rates for service reliability."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.monitoring_dir = self.reports_dir / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # SLO definitions
        self.slo_definitions = {
            "latency_p95": {
                "target": 250.0,  # 250ms p95 latency
                "window_minutes": 5,
                "breach_threshold": 300.0
            },
            "latency_p99": {
                "target": 500.0,  # 500ms p99 latency
                "window_minutes": 5,
                "breach_threshold": 600.0
            },
            "error_rate": {
                "target": 0.01,  # 1% error rate
                "window_minutes": 5,
                "breach_threshold": 0.05
            },
            "success_rate": {
                "target": 0.99,  # 99% success rate
                "window_minutes": 5,
                "breach_threshold": 0.95
            },
            "throughput": {
                "target": 1000.0,  # 1000 orders/minute
                "window_minutes": 5,
                "breach_threshold": 500.0
            }
        }
        
        # Burn rate thresholds
        self.burn_rate_thresholds = {
            "latency_burn_rate": 0.1,  # 10% increase per minute
            "error_rate_burn_rate": 0.05,  # 5% increase per minute
            "throughput_burn_rate": 0.2,  # 20% decrease per minute
            "memory_burn_rate": 0.15,  # 15% increase per minute
            "cpu_burn_rate": 0.2  # 20% increase per minute
        }
        
        # Metric history (circular buffers)
        self.metric_history = {
            "latency": deque(maxlen=1000),
            "errors": deque(maxlen=1000),
            "throughput": deque(maxlen=1000),
            "memory_usage": deque(maxlen=1000),
            "cpu_usage": deque(maxlen=1000)
        }
        
        # SLO state
        self.slo_metrics = {}
        self.burn_rate_alerts = []
        
        # Initialize SLO metrics
        self._initialize_slo_metrics()
    
    def _initialize_slo_metrics(self):
        """Initialize SLO metrics."""
        for name, definition in self.slo_definitions.items():
            self.slo_metrics[name] = SLOMetric(
                name=name,
                target=definition["target"],
                window_minutes=definition["window_minutes"],
                current_value=0.0,
                status=SLOStatus.HEALTHY,
                breach_count=0,
                last_breach=None
            )
    
    def record_metric(self, 
                     metric_name: str,
                     value: float,
                     timestamp: datetime = None):
        """Record a metric value."""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if metric_name in self.metric_history:
            self.metric_history[metric_name].append({
                "value": value,
                "timestamp": timestamp
            })
    
    def calculate_slo_metrics(self) -> Dict[str, SLOMetric]:
        """Calculate current SLO metrics."""
        
        current_time = datetime.now(timezone.utc)
        
        for slo_name, slo_metric in self.slo_metrics.items():
            # Get relevant metric data
            if slo_name == "latency_p95":
                metric_data = list(self.metric_history["latency"])
            elif slo_name == "latency_p99":
                metric_data = list(self.metric_history["latency"])
            elif slo_name == "error_rate":
                metric_data = list(self.metric_history["errors"])
            elif slo_name == "success_rate":
                metric_data = list(self.metric_history["errors"])
            elif slo_name == "throughput":
                metric_data = list(self.metric_history["throughput"])
            else:
                continue
            
            if not metric_data:
                continue
            
            # Filter data by time window
            window_start = current_time - timedelta(minutes=slo_metric.window_minutes)
            recent_data = [
                d for d in metric_data
                if d["timestamp"] >= window_start
            ]
            
            if not recent_data:
                continue
            
            # Calculate metric value
            values = [d["value"] for d in recent_data]
            
            if slo_name in ["latency_p95", "latency_p99"]:
                if slo_name == "latency_p95":
                    current_value = np.percentile(values, 95)
                else:  # p99
                    current_value = np.percentile(values, 99)
            elif slo_name == "error_rate":
                current_value = np.mean(values)
            elif slo_name == "success_rate":
                current_value = 1.0 - np.mean(values)
            elif slo_name == "throughput":
                current_value = np.sum(values) / slo_metric.window_minutes  # per minute
            
            # Update SLO metric
            slo_metric.current_value = current_value
            
            # Check SLO status
            definition = self.slo_definitions[slo_name]
            if current_value > definition["breach_threshold"]:
                slo_metric.status = SLOStatus.BREACHED
                slo_metric.breach_count += 1
                slo_metric.last_breach = current_time
            elif current_value > slo_metric.target:
                slo_metric.status = SLOStatus.WARNING
            else:
                slo_metric.status = SLOStatus.HEALTHY
        
        return self.slo_metrics
    
    def calculate_burn_rates(self) -> List[BurnRateAlert]:
        """Calculate burn rates and generate alerts."""
        
        current_time = datetime.now(timezone.utc)
        new_alerts = []
        
        for metric_name, history in self.metric_history.items():
            if len(history) < 10:  # Need minimum data points
                continue
            
            # Get recent data (last 10 minutes)
            recent_data = [
                d for d in history
                if d["timestamp"] >= current_time - timedelta(minutes=10)
            ]
            
            if len(recent_data) < 5:
                continue
            
            # Calculate burn rate (rate of change per minute)
            values = [d["value"] for d in recent_data]
            timestamps = [d["timestamp"] for d in recent_data]
            
            # Linear regression to find trend
            time_seconds = [(ts - timestamps[0]).total_seconds() / 60.0 for ts in timestamps]
            
            if len(time_seconds) > 1:
                # Calculate slope (burn rate)
                slope = np.polyfit(time_seconds, values, 1)[0]
                
                # Check against threshold
                burn_rate_key = f"{metric_name}_burn_rate"
                if burn_rate_key in self.burn_rate_thresholds:
                    threshold = self.burn_rate_thresholds[burn_rate_key]
                    
                    # Determine alert level
                    if abs(slope) > threshold * 2:
                        alert_level = "critical"
                    elif abs(slope) > threshold:
                        alert_level = "warning"
                    else:
                        continue
                    
                    # Create burn rate alert
                    alert = BurnRateAlert(
                        metric_name=metric_name,
                        current_rate=slope,
                        threshold_rate=threshold,
                        time_window_minutes=10,
                        alert_level=alert_level,
                        triggered_at=current_time
                    )
                    
                    new_alerts.append(alert)
        
        # Add new alerts to history
        self.burn_rate_alerts.extend(new_alerts)
        
        # Keep only last 100 alerts
        if len(self.burn_rate_alerts) > 100:
            self.burn_rate_alerts = self.burn_rate_alerts[-100:]
        
        return new_alerts
    
    def generate_slo_report(self) -> Dict[str, Any]:
        """Generate comprehensive SLO report."""
        
        # Calculate current SLO metrics
        slo_metrics = self.calculate_slo_metrics()
        
        # Calculate burn rates
        burn_rate_alerts = self.calculate_burn_rates()
        
        # Calculate SLO compliance
        total_slos = len(slo_metrics)
        healthy_slos = len([m for m in slo_metrics.values() if m.status == SLOStatus.HEALTHY])
        warning_slos = len([m for m in slo_metrics.values() if m.status == SLOStatus.WARNING])
        breached_slos = len([m for m in slo_metrics.values() if m.status == SLOStatus.BREACHED])
        
        compliance_rate = healthy_slos / total_slos if total_slos > 0 else 0.0
        
        # Calculate uptime
        uptime_hours = self._calculate_uptime()
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "slo_compliance": {
                "total_slos": total_slos,
                "healthy_slos": healthy_slos,
                "warning_slos": warning_slos,
                "breached_slos": breached_slos,
                "compliance_rate": compliance_rate
            },
            "uptime": {
                "uptime_hours": uptime_hours,
                "uptime_percentage": min(uptime_hours / 24.0, 1.0) * 100
            },
            "slo_metrics": {
                name: {
                    "current_value": metric.current_value,
                    "target": metric.target,
                    "status": metric.status.value,
                    "breach_count": metric.breach_count,
                    "last_breach": metric.last_breach.isoformat() if metric.last_breach else None
                }
                for name, metric in slo_metrics.items()
            },
            "burn_rate_alerts": [
                {
                    "metric_name": alert.metric_name,
                    "current_rate": alert.current_rate,
                    "threshold_rate": alert.threshold_rate,
                    "alert_level": alert.alert_level,
                    "triggered_at": alert.triggered_at.isoformat()
                }
                for alert in burn_rate_alerts
            ],
            "recommendations": self._generate_slo_recommendations(slo_metrics, burn_rate_alerts)
        }
        
        # Save report
        report_file = self.monitoring_dir / f"slo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours."""
        
        # This is a simplified calculation
        # In practice, you'd track actual system uptime
        
        current_time = datetime.now(timezone.utc)
        start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Assume system has been running since start of day
        uptime_seconds = (current_time - start_of_day).total_seconds()
        uptime_hours = uptime_seconds / 3600.0
        
        return uptime_hours
    
    def _generate_slo_recommendations(self, 
                                    slo_metrics: Dict[str, SLOMetric],
                                    burn_rate_alerts: List[BurnRateAlert]) -> List[str]:
        """Generate SLO recommendations."""
        
        recommendations = []
        
        # SLO-based recommendations
        for name, metric in slo_metrics.items():
            if metric.status == SLOStatus.BREACHED:
                recommendations.append(f"CRITICAL: {name} SLO breached - immediate action required")
            elif metric.status == SLOStatus.WARNING:
                recommendations.append(f"WARNING: {name} approaching breach threshold - monitor closely")
        
        # Burn rate recommendations
        critical_alerts = [a for a in burn_rate_alerts if a.alert_level == "critical"]
        warning_alerts = [a for a in burn_rate_alerts if a.alert_level == "warning"]
        
        if critical_alerts:
            recommendations.append("CRITICAL: Multiple burn rate alerts - system may be degrading")
        
        if warning_alerts:
            recommendations.append("WARNING: Burn rate alerts detected - monitor system performance")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All SLOs healthy - continue monitoring")
        
        return recommendations
    
    def simulate_metrics(self, duration_minutes: int = 60):
        """Simulate metric data for testing."""
        
        np.random.seed(42)
        start_time = datetime.now(timezone.utc)
        
        for minute in range(duration_minutes):
            current_time = start_time + timedelta(minutes=minute)
            
            # Simulate latency (increasing trend)
            base_latency = 200 + minute * 2  # Gradual increase
            latency = base_latency + np.random.normal(0, 20)
            self.record_metric("latency", max(latency, 50), current_time)
            
            # Simulate errors (sporadic)
            error_rate = 0.005 + np.random.exponential(0.01) if np.random.random() < 0.1 else 0.001
            self.record_metric("errors", error_rate, current_time)
            
            # Simulate throughput (decreasing trend)
            base_throughput = 1000 - minute * 5  # Gradual decrease
            throughput = base_throughput + np.random.normal(0, 50)
            self.record_metric("throughput", max(throughput, 100), current_time)
            
            # Simulate memory usage (increasing trend)
            memory_usage = 0.5 + minute * 0.01 + np.random.normal(0, 0.05)
            self.record_metric("memory_usage", min(memory_usage, 1.0), current_time)
            
            # Simulate CPU usage (variable)
            cpu_usage = 0.3 + np.random.normal(0, 0.1)
            self.record_metric("cpu_usage", max(min(cpu_usage, 1.0), 0.0), current_time)


def main():
    """Test SLOs and burn rate monitoring functionality."""
    monitor = SLOsBurnRateMonitor()
    
    # Simulate metrics
    print("Simulating metrics for 60 minutes...")
    monitor.simulate_metrics(60)
    
    # Generate SLO report
    report = monitor.generate_slo_report()
    print(f"✅ SLO compliance rate: {report['slo_compliance']['compliance_rate']:.1%}")
    print(f"✅ Uptime: {report['uptime']['uptime_hours']:.1f} hours")
    print(f"✅ Burn rate alerts: {len(report['burn_rate_alerts'])}")
    
    # Print SLO status
    for name, metric in report['slo_metrics'].items():
        print(f"✅ {name}: {metric['current_value']:.2f} (target: {metric['target']:.2f}) - {metric['status']}")
    
    print("✅ SLOs and burn rate monitoring testing completed")


if __name__ == "__main__":
    main()
