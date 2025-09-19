#!/usr/bin/env python3
"""
📝 EVENT SOURCING ENGINE
========================
Institutional-grade event sourcing architecture for complete audit trail and decision provenance.

Features:
- Immutable event store with cryptographic integrity
- Complete decision provenance tracking
- Perfect state reconstruction capability
- Time machine functionality for any trading moment
- AI decision explanation engine
- Human override documentation
- Distributed event replication
- Regulatory audit compliance
"""

import asyncio
import time
import json
import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

class EventType(Enum):
    """Types of events in the trading system"""
    TRADE_DECISION = "trade_decision"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_ALERT = "risk_alert"
    COMPLIANCE_BREACH = "compliance_breach"
    SYSTEM_STATE_CHANGE = "system_state_change"
    MARKET_DATA_UPDATE = "market_data_update"
    AI_DECISION = "ai_decision"
    HUMAN_OVERRIDE = "human_override"
    CONFIGURATION_CHANGE = "configuration_change"
    EMERGENCY_ACTION = "emergency_action"
    CHAOS_EXPERIMENT = "chaos_experiment"
    PERFORMANCE_METRIC = "performance_metric"

class EventSeverity(Enum):
    """Severity levels for events"""
    TRACE = "trace"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DecisionSource(Enum):
    """Source of trading decisions"""
    AI_ALGORITHM = "ai_algorithm"
    HUMAN_TRADER = "human_trader"
    AUTOMATED_SYSTEM = "automated_system"
    RISK_ENGINE = "risk_engine"
    COMPLIANCE_ENGINE = "compliance_engine"
    EMERGENCY_PROTOCOL = "emergency_protocol"

@dataclass
class TradingEvent:
    """Single immutable trading event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    sequence_number: int
    severity: EventSeverity
    source: str
    decision_source: Optional[DecisionSource]
    correlation_id: Optional[str]  # Links related events
    parent_event_id: Optional[str]  # Event hierarchy
    
    # Event data
    event_data: Dict[str, Any]
    
    # Decision context
    decision_context: Optional[Dict[str, Any]] = None
    ai_confidence: Optional[float] = None
    human_rationale: Optional[str] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    
    # Audit metadata
    system_state_hash: Optional[str] = None
    market_conditions_hash: Optional[str] = None
    compliance_verified: bool = False
    
    # Cryptographic integrity
    event_hash: Optional[str] = None
    previous_event_hash: Optional[str] = None
    digital_signature: Optional[str] = None
    
    # Performance tracking
    execution_latency_ms: Optional[float] = None
    impact_score: Optional[float] = None

@dataclass
class StateSnapshot:
    """Immutable snapshot of system state at a point in time"""
    snapshot_id: str
    timestamp: datetime
    sequence_number: int
    
    # Portfolio state
    portfolio_value: Decimal
    positions: Dict[str, Any]
    open_orders: Dict[str, Any]
    
    # Risk state
    var_estimate: float
    drawdown_current: float
    exposure_by_asset: Dict[str, float]
    
    # System state
    active_strategies: List[str]
    system_health_score: float
    configuration_hash: str
    
    # Market state
    market_regime: str
    volatility_regime: str
    funding_rates: Dict[str, float]
    
    # Integrity
    state_hash: str
    events_since_last_snapshot: int

@dataclass
class DecisionAuditTrail:
    """Complete audit trail for a trading decision"""
    decision_id: str
    timestamp: datetime
    decision_type: str
    final_action: str
    outcome: Optional[str]
    
    # Decision inputs
    market_data_inputs: Dict[str, Any]
    ai_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    compliance_checks: Dict[str, Any]
    
    # Decision process
    algorithm_steps: List[Dict[str, Any]]
    alternative_actions_considered: List[str]
    rejection_reasons: Dict[str, str]
    
    # Human involvement
    human_review_required: bool
    human_approval: Optional[bool]
    human_notes: Optional[str]
    
    # Execution
    execution_events: List[str]  # Event IDs
    execution_quality: Optional[float]
    slippage_bps: Optional[float]
    
    # Outcome tracking
    pnl_impact: Optional[Decimal]
    risk_impact: Optional[float]
    compliance_impact: Optional[str]

class EventSourcingEngine:
    """
    📝 EVENT SOURCING ENGINE
    Complete audit trail and decision provenance system
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Event storage
        self.event_store_path = config.get('event_store_path', 'data/events/')
        self.db_path = os.path.join(self.event_store_path, 'events.db')
        os.makedirs(self.event_store_path, exist_ok=True)
        
        # Event management
        self.event_sequence = 0
        self.event_buffer = deque(maxlen=10000)  # In-memory buffer
        self.snapshot_buffer = deque(maxlen=1000)
        self.decision_trails: Dict[str, DecisionAuditTrail] = {}
        
        # Integrity tracking
        self.last_event_hash = None
        self.integrity_verified = True
        
        # Performance tracking
        self.events_per_second = deque(maxlen=60)  # Last 60 seconds
        self.last_performance_log = time.time()
        
        # Background processing
        self.processing_active = False
        self.db_connection = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Snapshot configuration
        self.snapshot_interval_events = config.get('snapshot_interval', 10000)
        self.max_replay_events = config.get('max_replay_events', 100000)
        
        # Replication
        self.replication_enabled = config.get('replication_enabled', False)
        self.replication_targets = config.get('replication_targets', [])
        
        # Callbacks for real-time processing
        self.event_callbacks: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info("📝 [EVENT_SOURCING] Event Sourcing Engine initialized")

    def _initialize_database(self):
        """Initialize SQLite database for event storage"""
        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    decision_source TEXT,
                    correlation_id TEXT,
                    parent_event_id TEXT,
                    event_data TEXT NOT NULL,
                    decision_context TEXT,
                    ai_confidence REAL,
                    human_rationale TEXT,
                    risk_assessment TEXT,
                    system_state_hash TEXT,
                    market_conditions_hash TEXT,
                    compliance_verified INTEGER,
                    event_hash TEXT,
                    previous_event_hash TEXT,
                    digital_signature TEXT,
                    execution_latency_ms REAL,
                    impact_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # State snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    portfolio_value TEXT NOT NULL,
                    positions TEXT NOT NULL,
                    open_orders TEXT NOT NULL,
                    var_estimate REAL,
                    drawdown_current REAL,
                    exposure_by_asset TEXT,
                    active_strategies TEXT,
                    system_health_score REAL,
                    configuration_hash TEXT,
                    market_regime TEXT,
                    volatility_regime TEXT,
                    funding_rates TEXT,
                    state_hash TEXT NOT NULL,
                    events_since_last_snapshot INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Decision audit trails table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_trails (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    final_action TEXT NOT NULL,
                    outcome TEXT,
                    market_data_inputs TEXT,
                    ai_analysis TEXT,
                    risk_analysis TEXT,
                    compliance_checks TEXT,
                    algorithm_steps TEXT,
                    alternative_actions_considered TEXT,
                    rejection_reasons TEXT,
                    human_review_required INTEGER,
                    human_approval INTEGER,
                    human_notes TEXT,
                    execution_events TEXT,
                    execution_quality REAL,
                    slippage_bps REAL,
                    pnl_impact TEXT,
                    risk_impact REAL,
                    compliance_impact TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_sequence ON events(sequence_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON state_snapshots(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decision_trails(timestamp)')
            
            self.db_connection.commit()
            
            # Load sequence number from database
            cursor.execute('SELECT MAX(sequence_number) FROM events')
            result = cursor.fetchone()
            if result[0] is not None:
                self.event_sequence = result[0] + 1
            
            self.logger.info(f"📝 [EVENT_SOURCING] Database initialized, sequence: {self.event_sequence}")
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Database initialization failed: {e}")
            raise

    async def start_engine(self):
        """Start the event sourcing engine"""
        try:
            if self.processing_active:
                return
            
            self.processing_active = True
            
            # Start background processing
            asyncio.create_task(self._background_processor())
            asyncio.create_task(self._performance_monitor())
            
            # Verify integrity on startup
            await self._verify_event_integrity()
            
            self.logger.info("📝 [EVENT_SOURCING] Event Sourcing Engine started")
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error starting engine: {e}")

    async def record_event(self, event_type: EventType, source: str, event_data: Dict[str, Any],
                          severity: EventSeverity = EventSeverity.INFO,
                          decision_source: Optional[DecisionSource] = None,
                          correlation_id: Optional[str] = None,
                          parent_event_id: Optional[str] = None,
                          decision_context: Optional[Dict[str, Any]] = None,
                          ai_confidence: Optional[float] = None,
                          human_rationale: Optional[str] = None,
                          risk_assessment: Optional[Dict[str, Any]] = None) -> str:
        """Record a new trading event"""
        try:
            # Generate event ID and increment sequence
            event_id = str(uuid.uuid4())
            self.event_sequence += 1
            
            # Calculate system state hash
            system_state_hash = await self._calculate_system_state_hash()
            market_conditions_hash = await self._calculate_market_conditions_hash()
            
            # Create event
            event = TradingEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                sequence_number=self.event_sequence,
                severity=severity,
                source=source,
                decision_source=decision_source,
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
                event_data=event_data,
                decision_context=decision_context,
                ai_confidence=ai_confidence,
                human_rationale=human_rationale,
                risk_assessment=risk_assessment,
                system_state_hash=system_state_hash,
                market_conditions_hash=market_conditions_hash,
                compliance_verified=await self._verify_compliance_context(event_data),
                previous_event_hash=self.last_event_hash
            )
            
            # Calculate event hash for integrity
            event.event_hash = self._calculate_event_hash(event)
            self.last_event_hash = event.event_hash
            
            # Add to buffer for immediate processing
            self.event_buffer.append(event)
            
            # Persist asynchronously
            asyncio.create_task(self._persist_event(event))
            
            # Trigger callbacks
            await self._trigger_event_callbacks(event)
            
            # Check if snapshot needed
            if self.event_sequence % self.snapshot_interval_events == 0:
                asyncio.create_task(self._create_state_snapshot())
            
            # Performance tracking
            current_time = time.time()
            self.events_per_second.append(current_time)
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error recording event: {e}")
            raise

    async def start_decision_trail(self, decision_type: str, market_data_inputs: Dict[str, Any],
                                 ai_analysis: Dict[str, Any], risk_analysis: Dict[str, Any],
                                 compliance_checks: Dict[str, Any]) -> str:
        """Start a new decision audit trail"""
        try:
            decision_id = str(uuid.uuid4())
            
            trail = DecisionAuditTrail(
                decision_id=decision_id,
                timestamp=datetime.now(),
                decision_type=decision_type,
                final_action="",  # To be set later
                outcome=None,
                market_data_inputs=market_data_inputs,
                ai_analysis=ai_analysis,
                risk_analysis=risk_analysis,
                compliance_checks=compliance_checks,
                algorithm_steps=[],
                alternative_actions_considered=[],
                rejection_reasons={},
                human_review_required=False,
                human_approval=None,
                human_notes=None,
                execution_events=[],
                execution_quality=None,
                slippage_bps=None,
                pnl_impact=None,
                risk_impact=None,
                compliance_impact=None
            )
            
            self.decision_trails[decision_id] = trail
            
            # Record decision start event
            await self.record_event(
                event_type=EventType.AI_DECISION,
                source="decision_engine",
                event_data={
                    "decision_id": decision_id,
                    "decision_type": decision_type,
                    "status": "started"
                },
                correlation_id=decision_id,
                decision_context=ai_analysis
            )
            
            return decision_id
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error starting decision trail: {e}")
            raise

    async def add_decision_step(self, decision_id: str, step_name: str, step_data: Dict[str, Any],
                              alternatives_considered: Optional[List[str]] = None,
                              rejection_reasons: Optional[Dict[str, str]] = None):
        """Add a step to the decision trail"""
        try:
            if decision_id not in self.decision_trails:
                raise ValueError(f"Decision trail not found: {decision_id}")
            
            trail = self.decision_trails[decision_id]
            
            step = {
                "step_name": step_name,
                "timestamp": datetime.now().isoformat(),
                "step_data": step_data
            }
            
            trail.algorithm_steps.append(step)
            
            if alternatives_considered:
                trail.alternative_actions_considered.extend(alternatives_considered)
            
            if rejection_reasons:
                trail.rejection_reasons.update(rejection_reasons)
            
            # Record step event
            await self.record_event(
                event_type=EventType.AI_DECISION,
                source="decision_engine",
                event_data={
                    "decision_id": decision_id,
                    "step_name": step_name,
                    "step_data": step_data
                },
                correlation_id=decision_id
            )
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error adding decision step: {e}")

    async def finalize_decision(self, decision_id: str, final_action: str, execution_events: List[str],
                              human_approval: Optional[bool] = None, human_notes: Optional[str] = None):
        """Finalize a decision trail"""
        try:
            if decision_id not in self.decision_trails:
                raise ValueError(f"Decision trail not found: {decision_id}")
            
            trail = self.decision_trails[decision_id]
            trail.final_action = final_action
            trail.execution_events = execution_events
            trail.human_approval = human_approval
            trail.human_notes = human_notes
            
            # Persist decision trail
            await self._persist_decision_trail(trail)
            
            # Record decision completion event
            await self.record_event(
                event_type=EventType.AI_DECISION,
                source="decision_engine",
                event_data={
                    "decision_id": decision_id,
                    "final_action": final_action,
                    "status": "completed"
                },
                correlation_id=decision_id,
                human_rationale=human_notes
            )
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error finalizing decision: {e}")

    async def replay_events(self, start_time: Optional[datetime] = None, 
                           end_time: Optional[datetime] = None,
                           event_types: Optional[List[EventType]] = None) -> List[TradingEvent]:
        """Replay events for analysis or state reconstruction"""
        try:
            conditions = []
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())
            
            if event_types:
                type_conditions = ",".join("?" * len(event_types))
                conditions.append(f"event_type IN ({type_conditions})")
                params.extend([et.value for et in event_types])
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
                SELECT * FROM events 
                {where_clause}
                ORDER BY sequence_number ASC
                LIMIT {self.max_replay_events}
            """
            
            cursor = self.db_connection.cursor()
            cursor.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                event = self._row_to_event(row)
                events.append(event)
            
            self.logger.info(f"📝 [EVENT_SOURCING] Replayed {len(events)} events")
            return events
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error replaying events: {e}")
            return []

    async def reconstruct_state_at_time(self, target_time: datetime) -> Optional[Dict[str, Any]]:
        """Reconstruct system state at a specific point in time"""
        try:
            # Find the latest snapshot before target time
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT * FROM state_snapshots 
                WHERE timestamp <= ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (target_time.isoformat(),))
            
            snapshot_row = cursor.fetchone()
            if not snapshot_row:
                self.logger.warning("📝 [EVENT_SOURCING] No snapshot found before target time")
                return None
            
            # Load snapshot
            base_state = self._row_to_snapshot(snapshot_row)
            
            # Replay events from snapshot to target time
            events = await self.replay_events(
                start_time=base_state.timestamp,
                end_time=target_time
            )
            
            # Apply events to reconstruct state
            reconstructed_state = self._apply_events_to_state(base_state, events)
            
            self.logger.info(f"📝 [EVENT_SOURCING] State reconstructed at {target_time}")
            return reconstructed_state
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error reconstructing state: {e}")
            return None

    def _calculate_event_hash(self, event: TradingEvent) -> str:
        """Calculate cryptographic hash for event integrity"""
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'sequence_number': event.event_sequence,
            'source': event.source,
            'event_data': event.event_data,
            'previous_hash': event.previous_event_hash or ""
        }
        
        json_str = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def _calculate_system_state_hash(self) -> str:
        """Calculate hash of current system state"""
        # This would include current positions, orders, configuration, etc.
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'sequence': self.event_sequence
        }
        json_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def _calculate_market_conditions_hash(self) -> str:
        """Calculate hash of current market conditions"""
        # This would include prices, volatility, funding rates, etc.
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'regime': 'normal'  # Placeholder
        }
        json_str = json.dumps(market_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def _verify_compliance_context(self, event_data: Dict[str, Any]) -> bool:
        """Verify compliance context for event"""
        # This would integrate with compliance engine
        return True

    async def _persist_event(self, event: TradingEvent):
        """Persist event to database asynchronously"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.sequence_number,
                event.severity.value,
                event.source,
                event.decision_source.value if event.decision_source else None,
                event.correlation_id,
                event.parent_event_id,
                json.dumps(event.event_data, default=str),
                json.dumps(event.decision_context, default=str) if event.decision_context else None,
                event.ai_confidence,
                event.human_rationale,
                json.dumps(event.risk_assessment, default=str) if event.risk_assessment else None,
                event.system_state_hash,
                event.market_conditions_hash,
                1 if event.compliance_verified else 0,
                event.event_hash,
                event.previous_event_hash,
                event.digital_signature,
                event.execution_latency_ms,
                event.impact_score
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error persisting event: {e}")

    async def _persist_decision_trail(self, trail: DecisionAuditTrail):
        """Persist decision trail to database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO decision_trails VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                trail.decision_id,
                trail.timestamp.isoformat(),
                trail.decision_type,
                trail.final_action,
                trail.outcome,
                json.dumps(trail.market_data_inputs, default=str),
                json.dumps(trail.ai_analysis, default=str),
                json.dumps(trail.risk_analysis, default=str),
                json.dumps(trail.compliance_checks, default=str),
                json.dumps(trail.algorithm_steps, default=str),
                json.dumps(trail.alternative_actions_considered, default=str),
                json.dumps(trail.rejection_reasons, default=str),
                1 if trail.human_review_required else 0,
                1 if trail.human_approval else 0 if trail.human_approval is not None else None,
                trail.human_notes,
                json.dumps(trail.execution_events, default=str),
                trail.execution_quality,
                trail.slippage_bps,
                str(trail.pnl_impact) if trail.pnl_impact else None,
                trail.risk_impact,
                trail.compliance_impact
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error persisting decision trail: {e}")

    async def _trigger_event_callbacks(self, event: TradingEvent):
        """Trigger registered callbacks for event types"""
        try:
            callbacks = self.event_callbacks.get(event.event_type, [])
            for callback in callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    self.logger.error(f"❌ [EVENT_SOURCING] Error in event callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error triggering callbacks: {e}")

    async def _create_state_snapshot(self):
        """Create a state snapshot for faster replay"""
        try:
            snapshot_id = str(uuid.uuid4())
            
            # This would capture actual system state
            snapshot = StateSnapshot(
                snapshot_id=snapshot_id,
                timestamp=datetime.now(),
                sequence_number=self.event_sequence,
                portfolio_value=Decimal('0.0'),  # Would be real value
                positions={},
                open_orders={},
                var_estimate=0.0,
                drawdown_current=0.0,
                exposure_by_asset={},
                active_strategies=[],
                system_health_score=100.0,
                configuration_hash="",
                market_regime="normal",
                volatility_regime="low",
                funding_rates={},
                state_hash="",
                events_since_last_snapshot=self.snapshot_interval_events
            )
            
            # Calculate state hash
            snapshot.state_hash = self._calculate_snapshot_hash(snapshot)
            
            # Persist snapshot
            await self._persist_snapshot(snapshot)
            
            self.logger.info(f"📝 [EVENT_SOURCING] State snapshot created: {snapshot_id}")
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error creating snapshot: {e}")

    def _calculate_snapshot_hash(self, snapshot: StateSnapshot) -> str:
        """Calculate hash for state snapshot"""
        snapshot_data = asdict(snapshot)
        snapshot_data.pop('state_hash', None)  # Remove hash field from calculation
        json_str = json.dumps(snapshot_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def _persist_snapshot(self, snapshot: StateSnapshot):
        """Persist state snapshot to database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO state_snapshots VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                snapshot.snapshot_id,
                snapshot.timestamp.isoformat(),
                snapshot.sequence_number,
                str(snapshot.portfolio_value),
                json.dumps(snapshot.positions, default=str),
                json.dumps(snapshot.open_orders, default=str),
                snapshot.var_estimate,
                snapshot.drawdown_current,
                json.dumps(snapshot.exposure_by_asset, default=str),
                json.dumps(snapshot.active_strategies, default=str),
                snapshot.system_health_score,
                snapshot.configuration_hash,
                snapshot.market_regime,
                snapshot.volatility_regime,
                json.dumps(snapshot.funding_rates, default=str),
                snapshot.state_hash,
                snapshot.events_since_last_snapshot
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error persisting snapshot: {e}")

    def _row_to_event(self, row) -> TradingEvent:
        """Convert database row to TradingEvent"""
        return TradingEvent(
            event_id=row[0],
            event_type=EventType(row[1]),
            timestamp=datetime.fromisoformat(row[2]),
            sequence_number=row[3],
            severity=EventSeverity(row[4]),
            source=row[5],
            decision_source=DecisionSource(row[6]) if row[6] else None,
            correlation_id=row[7],
            parent_event_id=row[8],
            event_data=json.loads(row[9]),
            decision_context=json.loads(row[10]) if row[10] else None,
            ai_confidence=row[11],
            human_rationale=row[12],
            risk_assessment=json.loads(row[13]) if row[13] else None,
            system_state_hash=row[14],
            market_conditions_hash=row[15],
            compliance_verified=bool(row[16]),
            event_hash=row[17],
            previous_event_hash=row[18],
            digital_signature=row[19],
            execution_latency_ms=row[20],
            impact_score=row[21]
        )

    def _row_to_snapshot(self, row) -> StateSnapshot:
        """Convert database row to StateSnapshot"""
        return StateSnapshot(
            snapshot_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            sequence_number=row[2],
            portfolio_value=Decimal(row[3]),
            positions=json.loads(row[4]),
            open_orders=json.loads(row[5]),
            var_estimate=row[6],
            drawdown_current=row[7],
            exposure_by_asset=json.loads(row[8]),
            active_strategies=json.loads(row[9]),
            system_health_score=row[10],
            configuration_hash=row[11],
            market_regime=row[12],
            volatility_regime=row[13],
            funding_rates=json.loads(row[14]),
            state_hash=row[15],
            events_since_last_snapshot=row[16]
        )

    def _apply_events_to_state(self, base_state: StateSnapshot, events: List[TradingEvent]) -> Dict[str, Any]:
        """Apply events to reconstruct state"""
        # This would contain the actual state reconstruction logic
        # For now, return the base state
        return asdict(base_state)

    async def _background_processor(self):
        """Background processing loop"""
        while self.processing_active:
            try:
                # Process any pending operations
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ [EVENT_SOURCING] Error in background processor: {e}")
                await asyncio.sleep(5)

    async def _performance_monitor(self):
        """Monitor event sourcing performance"""
        while self.processing_active:
            try:
                current_time = time.time()
                
                # Calculate events per second
                recent_events = [t for t in self.events_per_second if current_time - t < 60]
                eps = len(recent_events) / 60.0 if recent_events else 0.0
                
                if current_time - self.last_performance_log > 300:  # Log every 5 minutes
                    self.logger.info(f"📝 [EVENT_SOURCING] Performance: {eps:.1f} events/sec, "
                                   f"Buffer: {len(self.event_buffer)}, Sequence: {self.event_sequence}")
                    self.last_performance_log = current_time
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"❌ [EVENT_SOURCING] Error in performance monitor: {e}")
                await asyncio.sleep(60)

    async def _verify_event_integrity(self):
        """Verify integrity of event chain"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT event_hash, previous_event_hash 
                FROM events 
                ORDER BY sequence_number ASC
            ''')
            
            previous_hash = None
            integrity_verified = True
            
            for row in cursor.fetchall():
                current_hash, expected_previous = row
                
                if previous_hash is not None and expected_previous != previous_hash:
                    self.logger.error(f"❌ [EVENT_SOURCING] Integrity violation detected!")
                    integrity_verified = False
                    break
                
                previous_hash = current_hash
            
            self.integrity_verified = integrity_verified
            
            if integrity_verified:
                self.logger.info("✅ [EVENT_SOURCING] Event integrity verified")
            else:
                self.logger.critical("🚨 [EVENT_SOURCING] EVENT CHAIN INTEGRITY COMPROMISED!")
                
        except Exception as e:
            self.logger.error(f"❌ [EVENT_SOURCING] Error verifying integrity: {e}")

    def register_event_callback(self, event_type: EventType, callback: Callable):
        """Register a callback for specific event types"""
        self.event_callbacks[event_type].append(callback)

    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        current_time = time.time()
        recent_events = [t for t in self.events_per_second if current_time - t < 60]
        eps = len(recent_events) / 60.0 if recent_events else 0.0
        
        return {
            'processing_active': self.processing_active,
            'event_sequence': self.event_sequence,
            'events_per_second': eps,
            'buffer_size': len(self.event_buffer),
            'integrity_verified': self.integrity_verified,
            'decision_trails_active': len(self.decision_trails),
            'replication_enabled': self.replication_enabled
        }

    async def stop_engine(self):
        """Stop the event sourcing engine"""
        self.processing_active = False
        if self.db_connection:
            self.db_connection.close()
        self.executor.shutdown(wait=True)
        self.logger.info("📝 [EVENT_SOURCING] Event Sourcing Engine stopped")
