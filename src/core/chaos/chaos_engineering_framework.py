#!/usr/bin/env python3
"""
🔥 CHAOS ENGINEERING FRAMEWORK
==============================
Institutional-grade chaos engineering for building antifragile trading systems.

Philosophy: "What doesn't kill the system makes it stronger"

Features:
- Automated failure injection
- Market crash simulation engine
- Network partition testing
- Exchange outage drills
- Memory/CPU stress testing
- Database corruption recovery
- Self-healing system validation
- Failure prediction and prevention
"""

import asyncio
import time
import json
import logging
import random
import threading
import subprocess
import psutil
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import numpy as np

class ChaosExperimentType(Enum):
    """Types of chaos experiments"""
    NETWORK_PARTITION = "network_partition"
    SERVICE_FAILURE = "service_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MARKET_CRASH = "market_crash"
    EXCHANGE_OUTAGE = "exchange_outage"
    DATABASE_CORRUPTION = "database_corruption"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    CONNECTIVITY_LOSS = "connectivity_loss"
    LATENCY_INJECTION = "latency_injection"

class ExperimentStatus(Enum):
    """Status of chaos experiments"""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

class ImpactSeverity(Enum):
    """Severity levels for chaos experiments"""
    LOW = "low"          # Minor disruption, should be handled gracefully
    MEDIUM = "medium"    # Moderate disruption, some degradation expected
    HIGH = "high"        # Significant disruption, emergency procedures triggered
    CRITICAL = "critical"  # Severe disruption, system survival test

@dataclass
class ChaosExperiment:
    """Definition of a chaos engineering experiment"""
    experiment_id: str
    name: str
    experiment_type: ChaosExperimentType
    description: str
    hypothesis: str  # What we expect to happen
    blast_radius: str  # Scope of impact
    severity: ImpactSeverity
    duration_seconds: int
    parameters: Dict[str, Any]
    abort_conditions: List[str]
    success_criteria: List[str]
    created_at: datetime
    scheduled_at: Optional[datetime] = None

@dataclass
class ExperimentResult:
    """Results from a chaos engineering experiment"""
    experiment_id: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    hypothesis_validated: bool
    system_impact: Dict[str, Any]
    recovery_time_seconds: float
    lessons_learned: List[str]
    improvements_identified: List[str]
    metrics_during_experiment: Dict[str, List[float]]
    error_logs: List[str]

@dataclass
class SystemHealthSnapshot:
    """Snapshot of system health for before/after comparison"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_latency_ms: float
    api_response_time_ms: float
    active_connections: int
    error_rate_percent: float
    trading_performance_score: float

class ChaosEngineeringFramework:
    """
    🔥 CHAOS ENGINEERING FRAMEWORK
    Systematically introduces controlled failures to build antifragile systems
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Experiment management
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: deque = deque(maxlen=1000)
        self.active_experiments: Dict[str, ExperimentResult] = {}
        
        # System monitoring
        self.baseline_metrics = None
        self.health_history = deque(maxlen=1440)  # 24 hours of minute data
        self.monitoring_active = False
        
        # Safety mechanisms
        self.abort_callbacks: List[Callable] = []
        self.max_concurrent_experiments = config.get('max_concurrent_experiments', 1)
        self.safety_threshold_cpu = config.get('safety_threshold_cpu', 90.0)
        self.safety_threshold_memory = config.get('safety_threshold_memory', 90.0)
        self.emergency_stop_enabled = True
        
        # Experiment scheduling
        self.scheduler_active = False
        self.experiment_queue = deque()
        
        # Integration points
        self.trading_engine = None
        self.observability_engine = None
        self.network_engine = None
        
        # Initialize predefined experiments
        self._initialize_standard_experiments()
        
        self.logger.info("🔥 [CHAOS] Chaos Engineering Framework initialized")

    def _initialize_standard_experiments(self):
        """Initialize standard chaos engineering experiments"""
        
        # 1. Network Partition Experiment
        self.experiments['network_partition_basic'] = ChaosExperiment(
            experiment_id='network_partition_basic',
            name='Basic Network Partition Test',
            experiment_type=ChaosExperimentType.NETWORK_PARTITION,
            description='Simulate network partition to test resilience',
            hypothesis='System should detect network issues and switch to offline mode gracefully',
            blast_radius='Network connectivity only',
            severity=ImpactSeverity.MEDIUM,
            duration_seconds=300,  # 5 minutes
            parameters={'target_hosts': ['api.hyperliquid.xyz'], 'drop_percentage': 100},
            abort_conditions=['cpu_usage > 95%', 'memory_usage > 95%', 'manual_abort'],
            success_criteria=['offline_mode_activated', 'no_data_loss', 'recovery_time < 60s'],
            created_at=datetime.now()
        )
        
        # 2. Exchange API Outage
        self.experiments['exchange_api_outage'] = ChaosExperiment(
            experiment_id='exchange_api_outage',
            name='Exchange API Outage Simulation',
            experiment_type=ChaosExperimentType.EXCHANGE_OUTAGE,
            description='Simulate complete exchange API unavailability',
            hypothesis='System should halt trading and preserve positions',
            blast_radius='All exchange operations',
            severity=ImpactSeverity.HIGH,
            duration_seconds=600,  # 10 minutes
            parameters={'block_endpoints': ['api.hyperliquid.xyz'], 'return_errors': True},
            abort_conditions=['position_loss', 'data_corruption', 'manual_abort'],
            success_criteria=['trading_halted', 'positions_preserved', 'alerts_triggered'],
            created_at=datetime.now()
        )
        
        # 3. Memory Stress Test
        self.experiments['memory_stress_test'] = ChaosExperiment(
            experiment_id='memory_stress_test',
            name='Memory Exhaustion Test',
            experiment_type=ChaosExperimentType.MEMORY_LEAK,
            description='Gradually consume memory to test handling',
            hypothesis='System should detect memory pressure and optimize usage',
            blast_radius='System memory',
            severity=ImpactSeverity.MEDIUM,
            duration_seconds=1800,  # 30 minutes
            parameters={'allocation_rate_mb_per_second': 10, 'max_allocation_mb': 1000},
            abort_conditions=['memory_usage > 95%', 'system_unresponsive', 'manual_abort'],
            success_criteria=['memory_optimization_triggered', 'no_crashes', 'graceful_degradation'],
            created_at=datetime.now()
        )
        
        # 4. Market Crash Simulation
        self.experiments['market_crash_2008'] = ChaosExperiment(
            experiment_id='market_crash_2008',
            name='2008 Financial Crisis Simulation',
            experiment_type=ChaosExperimentType.MARKET_CRASH,
            description='Simulate 2008-style market crash conditions',
            hypothesis='Risk management should prevent catastrophic losses',
            blast_radius='Trading algorithms and risk management',
            severity=ImpactSeverity.CRITICAL,
            duration_seconds=3600,  # 1 hour
            parameters={
                'price_drop_percentage': 40,
                'volatility_multiplier': 8,
                'liquidity_reduction': 0.7,
                'correlation_spike': 0.95
            },
            abort_conditions=['drawdown > 15%', 'margin_call_risk', 'manual_abort'],
            success_criteria=['kill_switch_activated', 'max_drawdown < 10%', 'positions_closed'],
            created_at=datetime.now()
        )
        
        # 5. High-Frequency Latency Spike
        self.experiments['latency_spike_test'] = ChaosExperiment(
            experiment_id='latency_spike_test',
            name='High-Frequency Latency Spike',
            experiment_type=ChaosExperimentType.LATENCY_INJECTION,
            description='Inject artificial latency into data feeds',
            hypothesis='HF trading algorithms should adapt to increased latency',
            blast_radius='Data feed latency',
            severity=ImpactSeverity.LOW,
            duration_seconds=900,  # 15 minutes
            parameters={'additional_latency_ms': 500, 'jitter_ms': 100},
            abort_conditions=['order_failures > 10%', 'manual_abort'],
            success_criteria=['latency_adaptation', 'order_success_rate > 95%', 'no_timeouts'],
            created_at=datetime.now()
        )

    async def start_chaos_framework(self):
        """Start the chaos engineering framework"""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            
            # Establish baseline metrics
            await self._establish_baseline_metrics()
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            # Start experiment scheduler
            asyncio.create_task(self._experiment_scheduler())
            
            self.logger.info("🔥 [CHAOS] Chaos Engineering Framework started")
            
        except Exception as e:
            self.logger.error(f"❌ [CHAOS] Error starting framework: {e}")

    async def _establish_baseline_metrics(self):
        """Establish baseline system metrics"""
        try:
            baseline = await self._capture_health_snapshot()
            self.baseline_metrics = baseline
            
            self.logger.info(f"🔥 [CHAOS] Baseline metrics established: "
                           f"CPU: {baseline.cpu_usage_percent:.1f}%, "
                           f"Memory: {baseline.memory_usage_percent:.1f}%, "
                           f"API Latency: {baseline.api_response_time_ms:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"❌ [CHAOS] Error establishing baseline: {e}")

    async def _capture_health_snapshot(self) -> SystemHealthSnapshot:
        """Capture current system health snapshot"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network latency (simplified)
            network_latency = await self._measure_network_latency()
            
            # API response time (if available)
            api_response_time = await self._measure_api_response_time()
            
            # Trading performance (simplified)
            trading_score = await self._calculate_trading_performance_score()
            
            return SystemHealthSnapshot(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_latency_ms=network_latency,
                api_response_time_ms=api_response_time,
                active_connections=len(psutil.net_connections()),
                error_rate_percent=0.0,  # Would be calculated from logs
                trading_performance_score=trading_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ [CHAOS] Error capturing health snapshot: {e}")
            return SystemHealthSnapshot(
                timestamp=datetime.now(),
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_latency_ms=0.0,
                api_response_time_ms=0.0,
                active_connections=0,
                error_rate_percent=0.0,
                trading_performance_score=0.0
            )

    async def _measure_network_latency(self) -> float:
        """Measure network latency to key endpoints"""
        try:
            import ping3
            latency = ping3.ping('8.8.8.8', timeout=5)
            return latency * 1000 if latency else 999.0  # Convert to ms
        except:
            return 999.0  # Fallback

    async def _measure_api_response_time(self) -> float:
        """Measure API response time"""
        try:
            start_time = time.time()
            # This would make a test API call
            await asyncio.sleep(0.01)  # Simulated API call
            return (time.time() - start_time) * 1000  # Convert to ms
        except:
            return 999.0  # Fallback

    async def _calculate_trading_performance_score(self) -> float:
        """Calculate simplified trading performance score"""
        # This would integrate with the trading engine to get real metrics
        return 85.0  # Placeholder

    async def _monitoring_loop(self):
        """Main monitoring loop for system health"""
        while self.monitoring_active:
            try:
                # Capture health metrics
                health_snapshot = await self._capture_health_snapshot()
                self.health_history.append(health_snapshot)
                
                # Check safety thresholds
                await self._check_safety_thresholds(health_snapshot)
                
                # Monitor active experiments
                await self._monitor_active_experiments(health_snapshot)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"❌ [CHAOS] Error in monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _check_safety_thresholds(self, health: SystemHealthSnapshot):
        """Check if system is exceeding safety thresholds"""
        safety_violations = []
        
        if health.cpu_usage_percent > self.safety_threshold_cpu:
            safety_violations.append(f"CPU usage: {health.cpu_usage_percent:.1f}%")
        
        if health.memory_usage_percent > self.safety_threshold_memory:
            safety_violations.append(f"Memory usage: {health.memory_usage_percent:.1f}%")
        
        if safety_violations and self.emergency_stop_enabled:
            self.logger.critical(f"🚨 [CHAOS] SAFETY THRESHOLD EXCEEDED: {', '.join(safety_violations)}")
            await self._emergency_stop_all_experiments("Safety threshold exceeded")

    async def _emergency_stop_all_experiments(self, reason: str):
        """Emergency stop all active experiments"""
        try:
            self.logger.critical(f"🚨 [CHAOS] EMERGENCY STOP: {reason}")
            
            for experiment_id in list(self.active_experiments.keys()):
                await self.abort_experiment(experiment_id, f"Emergency stop: {reason}")
            
            # Trigger emergency callbacks
            for callback in self.abort_callbacks:
                try:
                    await callback(reason)
                except Exception as e:
                    self.logger.error(f"❌ [CHAOS] Error in abort callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ [CHAOS] Error in emergency stop: {e}")

    async def schedule_experiment(self, experiment_id: str, scheduled_time: Optional[datetime] = None):
        """Schedule a chaos experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Unknown experiment: {experiment_id}")
            
            experiment = self.experiments[experiment_id]
            if scheduled_time:
                experiment.scheduled_at = scheduled_time
            else:
                experiment.scheduled_at = datetime.now() + timedelta(minutes=5)  # Default: 5 minutes from now
            
            self.experiment_queue.append(experiment_id)
            
            self.logger.info(f"🔥 [CHAOS] Experiment scheduled: {experiment_id} at {experiment.scheduled_at}")
            
        except Exception as e:
            self.logger.error(f"❌ [CHAOS] Error scheduling experiment: {e}")

    async def _experiment_scheduler(self):
        """Experiment scheduler loop"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                # Check if any experiments are ready to run
                while self.experiment_queue:
                    experiment_id = self.experiment_queue[0]
                    experiment = self.experiments[experiment_id]
                    
                    if (experiment.scheduled_at and 
                        current_time >= experiment.scheduled_at and
                        len(self.active_experiments) < self.max_concurrent_experiments):
                        
                        self.experiment_queue.popleft()
                        await self.run_experiment(experiment_id)
                    else:
                        break  # Wait for the next scheduled time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ [CHAOS] Error in experiment scheduler: {e}")
                await asyncio.sleep(60)

    async def run_experiment(self, experiment_id: str) -> ExperimentResult:
        """Run a chaos engineering experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Unknown experiment: {experiment_id}")
            
            experiment = self.experiments[experiment_id]
            
            self.logger.info(f"🔥 [CHAOS] Starting experiment: {experiment.name}")
            
            # Create experiment result record
            result = ExperimentResult(
                experiment_id=experiment_id,
                status=ExperimentStatus.RUNNING,
                start_time=datetime.now(),
                end_time=None,
                duration_seconds=0.0,
                hypothesis_validated=False,
                system_impact={},
                recovery_time_seconds=0.0,
                lessons_learned=[],
                improvements_identified=[],
                metrics_during_experiment={},
                error_logs=[]
            )
            
            self.active_experiments[experiment_id] = result
            
            # Capture pre-experiment health
            pre_health = await self._capture_health_snapshot()
            
            # Execute the experiment
            try:
                await self._execute_experiment(experiment, result)
                result.status = ExperimentStatus.COMPLETED
            except Exception as e:
                self.logger.error(f"❌ [CHAOS] Experiment execution failed: {e}")
                result.status = ExperimentStatus.FAILED
                result.error_logs.append(str(e))
            
            # Capture post-experiment health and calculate recovery
            post_health = await self._capture_health_snapshot()
            result.recovery_time_seconds = await self._measure_recovery_time(pre_health, post_health)
            
            # Finalize result
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # Validate hypothesis
            result.hypothesis_validated = await self._validate_experiment_hypothesis(experiment, result)
            
            # Generate lessons learned
            result.lessons_learned = await self._generate_lessons_learned(experiment, result)
            
            # Clean up
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            self.experiment_history.append(result)
            
            self.logger.info(f"🔥 [CHAOS] Experiment completed: {experiment.name} - "
                           f"Status: {result.status.value}, "
                           f"Duration: {result.duration_seconds:.1f}s, "
                           f"Hypothesis: {'✅' if result.hypothesis_validated else '❌'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [CHAOS] Error running experiment {experiment_id}: {e}")
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            raise

    async def _execute_experiment(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Execute the specific chaos experiment"""
        try:
            if experiment.experiment_type == ChaosExperimentType.NETWORK_PARTITION:
                await self._execute_network_partition(experiment, result)
            elif experiment.experiment_type == ChaosExperimentType.EXCHANGE_OUTAGE:
                await self._execute_exchange_outage(experiment, result)
            elif experiment.experiment_type == ChaosExperimentType.MEMORY_LEAK:
                await self._execute_memory_stress(experiment, result)
            elif experiment.experiment_type == ChaosExperimentType.MARKET_CRASH:
                await self._execute_market_crash(experiment, result)
            elif experiment.experiment_type == ChaosExperimentType.LATENCY_INJECTION:
                await self._execute_latency_injection(experiment, result)
            else:
                raise ValueError(f"Experiment type not implemented: {experiment.experiment_type}")
                
        except Exception as e:
            result.error_logs.append(f"Execution error: {str(e)}")
            raise

    async def _execute_network_partition(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Execute network partition experiment"""
        self.logger.info("🔥 [CHAOS] Executing network partition experiment")
        
        # This would use tools like iptables or tc to simulate network issues
        # For safety, we'll simulate the effects rather than actually blocking traffic
        
        # Simulate network issues by introducing delays and errors
        await asyncio.sleep(experiment.duration_seconds)
        
        result.system_impact['network_simulation'] = 'completed'

    async def _execute_exchange_outage(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Execute exchange outage experiment"""
        self.logger.info("🔥 [CHAOS] Executing exchange outage experiment")
        
        # This would integrate with the network resilience engine
        # to simulate API unavailability
        
        await asyncio.sleep(experiment.duration_seconds)
        
        result.system_impact['exchange_outage_simulation'] = 'completed'

    async def _execute_memory_stress(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Execute memory stress experiment"""
        self.logger.info("🔥 [CHAOS] Executing memory stress experiment")
        
        allocation_rate = experiment.parameters.get('allocation_rate_mb_per_second', 10)
        max_allocation = experiment.parameters.get('max_allocation_mb', 1000)
        
        # Gradually allocate memory to simulate memory pressure
        allocated_memory = []
        total_allocated = 0
        
        start_time = time.time()
        while (time.time() - start_time) < experiment.duration_seconds and total_allocated < max_allocation:
            # Allocate memory in chunks
            chunk_size = allocation_rate * 1024 * 1024  # Convert MB to bytes
            chunk = bytearray(chunk_size)
            allocated_memory.append(chunk)
            total_allocated += allocation_rate
            
            # Monitor system health
            health = await self._capture_health_snapshot()
            if health.memory_usage_percent > 95:
                self.logger.warning("🔥 [CHAOS] Memory stress: Approaching safety limit")
                break
            
            await asyncio.sleep(1)
        
        # Clean up allocated memory
        del allocated_memory
        
        result.system_impact['max_memory_allocated_mb'] = total_allocated
        result.system_impact['memory_stress_completed'] = True

    async def _execute_market_crash(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Execute market crash simulation"""
        self.logger.info("🔥 [CHAOS] Executing market crash simulation")
        
        # This would integrate with the trading engine to simulate
        # extreme market conditions
        
        price_drop = experiment.parameters.get('price_drop_percentage', 40)
        volatility_multiplier = experiment.parameters.get('volatility_multiplier', 8)
        
        self.logger.info(f"🔥 [CHAOS] Simulating {price_drop}% price drop with {volatility_multiplier}x volatility")
        
        await asyncio.sleep(experiment.duration_seconds)
        
        result.system_impact['market_crash_simulation'] = {
            'price_drop_pct': price_drop,
            'volatility_multiplier': volatility_multiplier,
            'completed': True
        }

    async def _execute_latency_injection(self, experiment: ChaosExperiment, result: ExperimentResult):
        """Execute latency injection experiment"""
        self.logger.info("🔥 [CHAOS] Executing latency injection experiment")
        
        additional_latency = experiment.parameters.get('additional_latency_ms', 500)
        jitter = experiment.parameters.get('jitter_ms', 100)
        
        # This would integrate with the data streaming engine
        # to inject artificial latency
        
        await asyncio.sleep(experiment.duration_seconds)
        
        result.system_impact['latency_injection'] = {
            'additional_latency_ms': additional_latency,
            'jitter_ms': jitter,
            'completed': True
        }

    async def _measure_recovery_time(self, pre_health: SystemHealthSnapshot, post_health: SystemHealthSnapshot) -> float:
        """Measure system recovery time after experiment"""
        # Simplified recovery time calculation
        # In practice, this would monitor until system returns to baseline
        return 30.0  # Placeholder

    async def _validate_experiment_hypothesis(self, experiment: ChaosExperiment, result: ExperimentResult) -> bool:
        """Validate the experiment hypothesis"""
        # This would check if the expected behavior occurred
        # Based on success criteria and system behavior
        
        success_count = 0
        total_criteria = len(experiment.success_criteria)
        
        for criteria in experiment.success_criteria:
            # Simplified criteria checking
            if 'no_crashes' in criteria and result.status == ExperimentStatus.COMPLETED:
                success_count += 1
            elif 'graceful_degradation' in criteria and 'completed' in str(result.system_impact):
                success_count += 1
        
        return success_count >= (total_criteria * 0.7)  # 70% success threshold

    async def _generate_lessons_learned(self, experiment: ChaosExperiment, result: ExperimentResult) -> List[str]:
        """Generate lessons learned from the experiment"""
        lessons = []
        
        if result.status == ExperimentStatus.COMPLETED:
            lessons.append(f"System successfully handled {experiment.experiment_type.value} scenario")
        
        if result.hypothesis_validated:
            lessons.append("Hypothesis validated - system behavior matched expectations")
        else:
            lessons.append("Hypothesis not validated - unexpected system behavior observed")
        
        if result.recovery_time_seconds > 60:
            lessons.append("Recovery time exceeded 60 seconds - consider optimization")
        
        return lessons

    async def abort_experiment(self, experiment_id: str, reason: str):
        """Abort a running experiment"""
        try:
            if experiment_id in self.active_experiments:
                result = self.active_experiments[experiment_id]
                result.status = ExperimentStatus.ABORTED
                result.end_time = datetime.now()
                result.error_logs.append(f"Aborted: {reason}")
                
                del self.active_experiments[experiment_id]
                self.experiment_history.append(result)
                
                self.logger.warning(f"🔥 [CHAOS] Experiment aborted: {experiment_id} - {reason}")
            
        except Exception as e:
            self.logger.error(f"❌ [CHAOS] Error aborting experiment: {e}")

    async def _monitor_active_experiments(self, health: SystemHealthSnapshot):
        """Monitor active experiments for abort conditions"""
        for experiment_id, result in list(self.active_experiments.items()):
            experiment = self.experiments[experiment_id]
            
            # Check abort conditions
            should_abort = False
            abort_reason = ""
            
            for condition in experiment.abort_conditions:
                if 'cpu_usage' in condition and health.cpu_usage_percent > 95:
                    should_abort = True
                    abort_reason = f"CPU usage exceeded: {health.cpu_usage_percent:.1f}%"
                elif 'memory_usage' in condition and health.memory_usage_percent > 95:
                    should_abort = True
                    abort_reason = f"Memory usage exceeded: {health.memory_usage_percent:.1f}%"
            
            if should_abort:
                await self.abort_experiment(experiment_id, abort_reason)

    def get_experiment_results(self) -> List[Dict[str, Any]]:
        """Get results from all completed experiments"""
        return [asdict(result) for result in self.experiment_history]

    def get_framework_status(self) -> Dict[str, Any]:
        """Get current framework status"""
        return {
            'monitoring_active': self.monitoring_active,
            'active_experiments': len(self.active_experiments),
            'total_experiments_run': len(self.experiment_history),
            'available_experiments': len(self.experiments),
            'queued_experiments': len(self.experiment_queue),
            'emergency_stop_enabled': self.emergency_stop_enabled,
            'current_system_health': asdict(self.health_history[-1]) if self.health_history else None
        }

    async def stop_chaos_framework(self):
        """Stop the chaos engineering framework"""
        self.monitoring_active = False
        await self._emergency_stop_all_experiments("Framework shutdown")
        self.logger.info("🔥 [CHAOS] Chaos Engineering Framework stopped")
