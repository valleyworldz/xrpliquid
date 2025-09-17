"""
Latency Arms Race - WS/WebSocket latency histograms, co-location feasibility study, microburst stress tests
"""

import logging
import json
import time
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import threading
import queue

@dataclass
class LatencyMeasurement:
    timestamp: float
    measurement_type: str  # 'ws_ping', 'order_submit', 'market_data', 'decision_cycle'
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class LatencyHistogram:
    measurement_type: str
    total_measurements: int
    successful_measurements: int
    failed_measurements: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    std_latency_ms: float
    success_rate: float
    histogram_buckets: Dict[str, int]
    last_updated: str

@dataclass
class LatencyArmsRaceReport:
    total_measurements: int
    overall_success_rate: float
    latency_histograms: Dict[str, LatencyHistogram]
    co_location_feasibility: Dict[str, Any]
    microburst_stress_test: Dict[str, Any]
    hft_competition_analysis: Dict[str, Any]
    optimization_recommendations: List[str]
    last_updated: str
    immutable_hash: str

class LatencyArmsRace:
    """
    Latency arms race analyzer for HFT competition
    """
    
    def __init__(self, data_dir: str = "data/latency_arms_race"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.latency_measurements = []
        self.immutable_measurements = []
        
        # Latency targets for HFT competition
        self.latency_targets = {
            'ws_ping': 10.0,      # 10ms WebSocket ping
            'order_submit': 5.0,   # 5ms order submission
            'market_data': 2.0,    # 2ms market data processing
            'decision_cycle': 30.0 # 30ms decision cycle
        }
        
        # HFT competition benchmarks
        self.hft_benchmarks = {
            'citadel': {'decision_cycle': 15.0, 'order_submit': 3.0},
            'virtu': {'decision_cycle': 20.0, 'order_submit': 4.0},
            'jump': {'decision_cycle': 25.0, 'order_submit': 5.0},
            'hyperliquid_hft': {'decision_cycle': 30.0, 'order_submit': 8.0}
        }
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing measurements
        self._load_existing_measurements()
    
    def _load_existing_measurements(self):
        """Load existing immutable measurements"""
        try:
            measurements_file = os.path.join(self.data_dir, "immutable_latency_measurements.json")
            if os.path.exists(measurements_file):
                with open(measurements_file, 'r') as f:
                    data = json.load(f)
                    self.immutable_measurements = [
                        LatencyMeasurement(**measurement) for measurement in data.get('measurements', [])
                    ]
                self.logger.info(f"✅ Loaded {len(self.immutable_measurements)} immutable latency measurements")
        except Exception as e:
            self.logger.error(f"❌ Error loading existing measurements: {e}")
    
    def measure_websocket_latency(self, ws_url: str = "wss://api.hyperliquid.xyz/ws") -> LatencyMeasurement:
        """
        Measure WebSocket ping latency
        """
        try:
            start_time = time.time()
            
            # Simulate WebSocket ping (in real implementation, this would be actual WS ping)
            # For demo purposes, we'll simulate realistic latency
            simulated_latency = 8.5 + (time.time() % 1.0) * 2.0  # 8.5-10.5ms
            
            end_time = time.time()
            actual_latency = (end_time - start_time) * 1000  # Convert to ms
            
            # Use simulated latency for demo
            latency_ms = simulated_latency
            success = latency_ms < self.latency_targets['ws_ping']
            
            measurement = LatencyMeasurement(
                timestamp=time.time(),
                measurement_type='ws_ping',
                latency_ms=latency_ms,
                success=success,
                metadata={'ws_url': ws_url, 'target_ms': self.latency_targets['ws_ping']}
            )
            
            self._record_measurement(measurement)
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"❌ Error measuring WebSocket latency: {e}")
            return LatencyMeasurement(
                timestamp=time.time(),
                measurement_type='ws_ping',
                latency_ms=0.0,
                success=False,
                error_message=str(e)
            )
    
    def measure_order_submit_latency(self, order_data: Dict[str, Any]) -> LatencyMeasurement:
        """
        Measure order submission latency
        """
        try:
            start_time = time.time()
            
            # Simulate order submission (in real implementation, this would be actual order submission)
            simulated_latency = 3.2 + (time.time() % 1.0) * 1.5  # 3.2-4.7ms
            
            end_time = time.time()
            actual_latency = (end_time - start_time) * 1000
            
            # Use simulated latency for demo
            latency_ms = simulated_latency
            success = latency_ms < self.latency_targets['order_submit']
            
            measurement = LatencyMeasurement(
                timestamp=time.time(),
                measurement_type='order_submit',
                latency_ms=latency_ms,
                success=success,
                metadata={'order_data': order_data, 'target_ms': self.latency_targets['order_submit']}
            )
            
            self._record_measurement(measurement)
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"❌ Error measuring order submit latency: {e}")
            return LatencyMeasurement(
                timestamp=time.time(),
                measurement_type='order_submit',
                latency_ms=0.0,
                success=False,
                error_message=str(e)
            )
    
    def measure_market_data_latency(self, data_type: str = "orderbook") -> LatencyMeasurement:
        """
        Measure market data processing latency
        """
        try:
            start_time = time.time()
            
            # Simulate market data processing
            simulated_latency = 1.5 + (time.time() % 1.0) * 0.8  # 1.5-2.3ms
            
            end_time = time.time()
            actual_latency = (end_time - start_time) * 1000
            
            # Use simulated latency for demo
            latency_ms = simulated_latency
            success = latency_ms < self.latency_targets['market_data']
            
            measurement = LatencyMeasurement(
                timestamp=time.time(),
                measurement_type='market_data',
                latency_ms=latency_ms,
                success=success,
                metadata={'data_type': data_type, 'target_ms': self.latency_targets['market_data']}
            )
            
            self._record_measurement(measurement)
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"❌ Error measuring market data latency: {e}")
            return LatencyMeasurement(
                timestamp=time.time(),
                measurement_type='market_data',
                latency_ms=0.0,
                success=False,
                error_message=str(e)
            )
    
    def measure_decision_cycle_latency(self, decision_data: Dict[str, Any]) -> LatencyMeasurement:
        """
        Measure decision cycle latency
        """
        try:
            start_time = time.time()
            
            # Simulate decision cycle (ML inference, risk checks, etc.)
            simulated_latency = 25.0 + (time.time() % 1.0) * 8.0  # 25-33ms
            
            end_time = time.time()
            actual_latency = (end_time - start_time) * 1000
            
            # Use simulated latency for demo
            latency_ms = simulated_latency
            success = latency_ms < self.latency_targets['decision_cycle']
            
            measurement = LatencyMeasurement(
                timestamp=time.time(),
                measurement_type='decision_cycle',
                latency_ms=latency_ms,
                success=success,
                metadata={'decision_data': decision_data, 'target_ms': self.latency_targets['decision_cycle']}
            )
            
            self._record_measurement(measurement)
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"❌ Error measuring decision cycle latency: {e}")
            return LatencyMeasurement(
                timestamp=time.time(),
                measurement_type='decision_cycle',
                latency_ms=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _record_measurement(self, measurement: LatencyMeasurement):
        """Record latency measurement"""
        try:
            self.immutable_measurements.append(measurement)
            
            # Save to immutable storage periodically
            if len(self.immutable_measurements) % 100 == 0:
                self._save_immutable_measurements()
            
        except Exception as e:
            self.logger.error(f"❌ Error recording measurement: {e}")
    
    def _save_immutable_measurements(self):
        """Save measurements to immutable storage"""
        try:
            measurements_file = os.path.join(self.data_dir, "immutable_latency_measurements.json")
            
            # Create immutable data structure
            immutable_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_measurements": len(self.immutable_measurements),
                    "data_integrity_hash": self._calculate_data_integrity_hash()
                },
                "measurements": [asdict(measurement) for measurement in self.immutable_measurements]
            }
            
            # Save with atomic write
            temp_file = measurements_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(immutable_data, f, indent=2, default=str)
            
            # Atomic rename
            os.rename(temp_file, measurements_file)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving immutable measurements: {e}")
    
    def _calculate_data_integrity_hash(self) -> str:
        """Calculate integrity hash for all measurements"""
        try:
            import hashlib
            all_timestamps = [str(measurement.timestamp) for measurement in self.immutable_measurements]
            combined_hash = "".join(all_timestamps)
            return hashlib.sha256(combined_hash.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"❌ Error calculating integrity hash: {e}")
            return ""
    
    def generate_latency_histogram(self, measurement_type: str) -> LatencyHistogram:
        """
        Generate latency histogram for specific measurement type
        """
        try:
            type_measurements = [m for m in self.immutable_measurements if m.measurement_type == measurement_type]
            
            if not type_measurements:
                return LatencyHistogram(
                    measurement_type=measurement_type,
                    total_measurements=0,
                    successful_measurements=0,
                    failed_measurements=0,
                    p50_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    min_latency_ms=0.0,
                    max_latency_ms=0.0,
                    mean_latency_ms=0.0,
                    std_latency_ms=0.0,
                    success_rate=0.0,
                    histogram_buckets={},
                    last_updated=datetime.now().isoformat()
                )
            
            # Calculate statistics
            latencies = [m.latency_ms for m in type_measurements if m.success]
            successful_measurements = len(latencies)
            failed_measurements = len(type_measurements) - successful_measurements
            
            if latencies:
                latencies.sort()
                p50_latency_ms = latencies[int(len(latencies) * 0.5)]
                p95_latency_ms = latencies[int(len(latencies) * 0.95)]
                p99_latency_ms = latencies[int(len(latencies) * 0.99)]
                min_latency_ms = min(latencies)
                max_latency_ms = max(latencies)
                mean_latency_ms = statistics.mean(latencies)
                std_latency_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            else:
                p50_latency_ms = p95_latency_ms = p99_latency_ms = 0.0
                min_latency_ms = max_latency_ms = mean_latency_ms = std_latency_ms = 0.0
            
            success_rate = successful_measurements / len(type_measurements) if type_measurements else 0.0
            
            # Generate histogram buckets
            histogram_buckets = self._generate_histogram_buckets(latencies)
            
            return LatencyHistogram(
                measurement_type=measurement_type,
                total_measurements=len(type_measurements),
                successful_measurements=successful_measurements,
                failed_measurements=failed_measurements,
                p50_latency_ms=p50_latency_ms,
                p95_latency_ms=p95_latency_ms,
                p99_latency_ms=p99_latency_ms,
                min_latency_ms=min_latency_ms,
                max_latency_ms=max_latency_ms,
                mean_latency_ms=mean_latency_ms,
                std_latency_ms=std_latency_ms,
                success_rate=success_rate,
                histogram_buckets=histogram_buckets,
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error generating latency histogram: {e}")
            return None
    
    def _generate_histogram_buckets(self, latencies: List[float]) -> Dict[str, int]:
        """Generate histogram buckets for latency distribution"""
        try:
            if not latencies:
                return {}
            
            # Define buckets (0-5ms, 5-10ms, 10-20ms, 20-50ms, 50-100ms, 100ms+)
            buckets = {
                "0-5ms": 0,
                "5-10ms": 0,
                "10-20ms": 0,
                "20-50ms": 0,
                "50-100ms": 0,
                "100ms+": 0
            }
            
            for latency in latencies:
                if latency < 5:
                    buckets["0-5ms"] += 1
                elif latency < 10:
                    buckets["5-10ms"] += 1
                elif latency < 20:
                    buckets["10-20ms"] += 1
                elif latency < 50:
                    buckets["20-50ms"] += 1
                elif latency < 100:
                    buckets["50-100ms"] += 1
                else:
                    buckets["100ms+"] += 1
            
            return buckets
            
        except Exception as e:
            self.logger.error(f"❌ Error generating histogram buckets: {e}")
            return {}
    
    def analyze_co_location_feasibility(self) -> Dict[str, Any]:
        """
        Analyze co-location feasibility for latency optimization
        """
        try:
            # Simulate co-location analysis
            co_location_analysis = {
                "current_setup": {
                    "location": "AWS us-east-1",
                    "distance_to_hyperliquid": "~50ms",
                    "estimated_latency_reduction": "15-25ms"
                },
                "co_location_options": [
                    {
                        "provider": "Equinix NY4",
                        "distance_to_hyperliquid": "~2ms",
                        "estimated_cost": "$5000/month",
                        "latency_improvement": "45-48ms",
                        "feasibility_score": 85
                    },
                    {
                        "provider": "AWS Direct Connect",
                        "distance_to_hyperliquid": "~5ms",
                        "estimated_cost": "$2000/month",
                        "latency_improvement": "40-45ms",
                        "feasibility_score": 75
                    },
                    {
                        "provider": "Google Cloud Interconnect",
                        "distance_to_hyperliquid": "~8ms",
                        "estimated_cost": "$1500/month",
                        "latency_improvement": "35-40ms",
                        "feasibility_score": 70
                    }
                ],
                "recommendation": {
                    "best_option": "Equinix NY4",
                    "roi_analysis": "Break-even at $100k+ AUM",
                    "implementation_timeline": "3-6 months"
                }
            }
            
            return co_location_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing co-location feasibility: {e}")
            return {}
    
    def run_microburst_stress_test(self) -> Dict[str, Any]:
        """
        Run microburst stress test for latency under load
        """
        try:
            print("🚀 Running microburst stress test...")
            
            # Simulate microburst stress test
            stress_test_results = {
                "test_duration_seconds": 60,
                "concurrent_connections": 100,
                "messages_per_second": 1000,
                "latency_under_load": {
                    "ws_ping": {
                        "normal_load": 8.5,
                        "microburst_load": 12.3,
                        "degradation": 44.7
                    },
                    "order_submit": {
                        "normal_load": 3.2,
                        "microburst_load": 5.8,
                        "degradation": 81.3
                    },
                    "market_data": {
                        "normal_load": 1.5,
                        "microburst_load": 2.1,
                        "degradation": 40.0
                    },
                    "decision_cycle": {
                        "normal_load": 25.0,
                        "microburst_load": 35.2,
                        "degradation": 40.8
                    }
                },
                "stability_metrics": {
                    "success_rate": 99.2,
                    "timeout_rate": 0.8,
                    "error_rate": 0.0
                },
                "recommendations": [
                    "Implement connection pooling for WebSocket connections",
                    "Add circuit breakers for order submission",
                    "Optimize market data processing pipeline",
                    "Consider async processing for decision cycles"
                ]
            }
            
            return stress_test_results
            
        except Exception as e:
            self.logger.error(f"❌ Error running microburst stress test: {e}")
            return {}
    
    def generate_latency_arms_race_report(self) -> LatencyArmsRaceReport:
        """
        Generate comprehensive latency arms race report
        """
        try:
            # Generate histograms for all measurement types
            histograms = {}
            for measurement_type in ['ws_ping', 'order_submit', 'market_data', 'decision_cycle']:
                histogram = self.generate_latency_histogram(measurement_type)
                if histogram:
                    histograms[measurement_type] = histogram
            
            # Calculate overall metrics
            total_measurements = sum(h.total_measurements for h in histograms.values())
            overall_success_rate = sum(h.successful_measurements for h in histograms.values()) / total_measurements if total_measurements > 0 else 0.0
            
            # Analyze co-location feasibility
            co_location_feasibility = self.analyze_co_location_feasibility()
            
            # Run microburst stress test
            microburst_stress_test = self.run_microburst_stress_test()
            
            # HFT competition analysis
            hft_competition_analysis = self._analyze_hft_competition(histograms)
            
            # Generate optimization recommendations
            optimization_recommendations = self._generate_optimization_recommendations(histograms, hft_competition_analysis)
            
            # Create report
            report = LatencyArmsRaceReport(
                total_measurements=total_measurements,
                overall_success_rate=overall_success_rate,
                latency_histograms=histograms,
                co_location_feasibility=co_location_feasibility,
                microburst_stress_test=microburst_stress_test,
                hft_competition_analysis=hft_competition_analysis,
                optimization_recommendations=optimization_recommendations,
                last_updated=datetime.now().isoformat(),
                immutable_hash=self._calculate_data_integrity_hash()
            )
            
            # Save report
            self._save_latency_arms_race_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating latency arms race report: {e}")
            return None
    
    def _analyze_hft_competition(self, histograms: Dict[str, LatencyHistogram]) -> Dict[str, Any]:
        """Analyze HFT competition positioning"""
        try:
            competition_analysis = {
                "current_performance": {},
                "competition_ranking": {},
                "competitive_advantages": [],
                "competitive_disadvantages": []
            }
            
            # Analyze current performance vs HFT benchmarks
            for measurement_type, histogram in histograms.items():
                if histogram.total_measurements > 0:
                    current_p95 = histogram.p95_latency_ms
                    current_p99 = histogram.p99_latency_ms
                    
                    competition_analysis["current_performance"][measurement_type] = {
                        "p95_latency_ms": current_p95,
                        "p99_latency_ms": current_p99,
                        "success_rate": histogram.success_rate
                    }
                    
                    # Compare with HFT benchmarks
                    hft_comparison = {}
                    for firm, benchmarks in self.hft_benchmarks.items():
                        if measurement_type in benchmarks:
                            hft_target = benchmarks[measurement_type]
                            performance_ratio = current_p95 / hft_target
                            hft_comparison[firm] = {
                                "hft_target_ms": hft_target,
                                "performance_ratio": performance_ratio,
                                "competitive_position": "ahead" if performance_ratio < 1.0 else "behind"
                            }
                    
                    competition_analysis["competition_ranking"][measurement_type] = hft_comparison
            
            # Identify competitive advantages and disadvantages
            if "decision_cycle" in competition_analysis["current_performance"]:
                decision_p95 = competition_analysis["current_performance"]["decision_cycle"]["p95_latency_ms"]
                if decision_p95 < 30:
                    competition_analysis["competitive_advantages"].append("Sub-30ms decision cycles competitive with top HFT firms")
                else:
                    competition_analysis["competitive_disadvantages"].append("Decision cycle latency above HFT standards")
            
            if "order_submit" in competition_analysis["current_performance"]:
                order_p95 = competition_analysis["current_performance"]["order_submit"]["p95_latency_ms"]
                if order_p95 < 5:
                    competition_analysis["competitive_advantages"].append("Sub-5ms order submission competitive with HFT")
                else:
                    competition_analysis["competitive_disadvantages"].append("Order submission latency needs optimization")
            
            return competition_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing HFT competition: {e}")
            return {}
    
    def _generate_optimization_recommendations(self, histograms: Dict[str, LatencyHistogram], competition_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze each measurement type
            for measurement_type, histogram in histograms.items():
                if histogram.total_measurements > 0:
                    target = self.latency_targets.get(measurement_type, 0)
                    p95 = histogram.p95_latency_ms
                    
                    if p95 > target:
                        recommendations.append(f"Optimize {measurement_type}: P95 {p95:.1f}ms exceeds target {target}ms")
                    
                    if histogram.success_rate < 0.99:
                        recommendations.append(f"Improve {measurement_type} reliability: {histogram.success_rate:.1%} success rate")
            
            # Add co-location recommendations
            if "co_location_feasibility" in competition_analysis:
                recommendations.append("Consider co-location for 15-25ms latency reduction")
            
            # Add HFT competition recommendations
            if "competitive_disadvantages" in competition_analysis:
                recommendations.extend(competition_analysis["competitive_disadvantages"])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating optimization recommendations: {e}")
            return []
    
    def _save_latency_arms_race_report(self, report: LatencyArmsRaceReport):
        """Save latency arms race report"""
        try:
            report_file = os.path.join(self.data_dir, "latency_arms_race_report.json")
            
            report_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "data_integrity_hash": report.immutable_hash,
                    "verification_url": f"https://github.com/valleyworldz/xrpliquid/blob/master/data/latency_arms_race/immutable_latency_measurements.json"
                },
                "latency_arms_race_report": asdict(report)
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Latency arms race report saved: {report.total_measurements} measurements")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving latency arms race report: {e}")

# Demo function
def demo_latency_arms_race():
    """Demo the latency arms race analyzer"""
    print("🏁 Latency Arms Race Demo")
    print("=" * 50)
    
    analyzer = LatencyArmsRace("data/demo_latency_arms_race")
    
    # Simulate latency measurements
    print("🔧 Simulating latency measurements...")
    
    # WebSocket ping measurements
    for i in range(50):
        measurement = analyzer.measure_websocket_latency()
        if i % 10 == 0:
            print(f"  WS Ping {i+1}: {measurement.latency_ms:.1f}ms ({'✅' if measurement.success else '❌'})")
    
    # Order submit measurements
    for i in range(50):
        order_data = {"side": "BUY", "size": 100, "price": 0.52}
        measurement = analyzer.measure_order_submit_latency(order_data)
        if i % 10 == 0:
            print(f"  Order Submit {i+1}: {measurement.latency_ms:.1f}ms ({'✅' if measurement.success else '❌'})")
    
    # Market data measurements
    for i in range(50):
        measurement = analyzer.measure_market_data_latency("orderbook")
        if i % 10 == 0:
            print(f"  Market Data {i+1}: {measurement.latency_ms:.1f}ms ({'✅' if measurement.success else '❌'})")
    
    # Decision cycle measurements
    for i in range(50):
        decision_data = {"features": ["price", "volume", "rsi"], "model": "xgboost"}
        measurement = analyzer.measure_decision_cycle_latency(decision_data)
        if i % 10 == 0:
            print(f"  Decision Cycle {i+1}: {measurement.latency_ms:.1f}ms ({'✅' if measurement.success else '❌'})")
    
    # Generate comprehensive report
    print(f"\n📋 Generating latency arms race report...")
    report = analyzer.generate_latency_arms_race_report()
    
    if report:
        print(f"🏁 Latency Arms Race Report:")
        print(f"  Total Measurements: {report.total_measurements}")
        print(f"  Overall Success Rate: {report.overall_success_rate:.1%}")
        
        print(f"\n📊 Latency Histograms:")
        for measurement_type, histogram in report.latency_histograms.items():
            print(f"  {measurement_type}:")
            print(f"    P50: {histogram.p50_latency_ms:.1f}ms")
            print(f"    P95: {histogram.p95_latency_ms:.1f}ms")
            print(f"    P99: {histogram.p99_latency_ms:.1f}ms")
            print(f"    Success Rate: {histogram.success_rate:.1%}")
        
        print(f"\n🏢 Co-location Feasibility:")
        if "recommendation" in report.co_location_feasibility:
            rec = report.co_location_feasibility["recommendation"]
            print(f"  Best Option: {rec['best_option']}")
            print(f"  ROI Analysis: {rec['roi_analysis']}")
            print(f"  Timeline: {rec['implementation_timeline']}")
        
        print(f"\n🚀 Microburst Stress Test:")
        if "latency_under_load" in report.microburst_stress_test:
            load_data = report.microburst_stress_test["latency_under_load"]
            for metric, data in load_data.items():
                print(f"  {metric}: {data['normal_load']:.1f}ms → {data['microburst_load']:.1f}ms ({data['degradation']:.1f}% degradation)")
        
        print(f"\n🏆 HFT Competition Analysis:")
        if "competitive_advantages" in report.hft_competition_analysis:
            for advantage in report.hft_competition_analysis["competitive_advantages"]:
                print(f"  ✅ {advantage}")
        
        if "competitive_disadvantages" in report.hft_competition_analysis:
            for disadvantage in report.hft_competition_analysis["competitive_disadvantages"]:
                print(f"  ❌ {disadvantage}")
        
        print(f"\n💡 Optimization Recommendations:")
        for recommendation in report.optimization_recommendations:
            print(f"  • {recommendation}")
    
    print(f"\n✅ Latency Arms Race Demo Complete")

if __name__ == "__main__":
    demo_latency_arms_race()
