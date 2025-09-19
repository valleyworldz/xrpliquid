"""
📝 AUDIT MODULE
===============
Comprehensive audit and event sourcing system for institutional trading.

This module provides:
- Immutable event store with cryptographic integrity
- Complete decision provenance tracking
- Perfect state reconstruction capability
- Time machine functionality
- AI decision explanation engine
- Human override documentation
- Regulatory audit compliance
"""

from .event_sourcing_engine import (
    EventSourcingEngine,
    TradingEvent,
    StateSnapshot,
    DecisionAuditTrail,
    EventType,
    EventSeverity,
    DecisionSource
)

__all__ = [
    'EventSourcingEngine',
    'TradingEvent',
    'StateSnapshot', 
    'DecisionAuditTrail',
    'EventType',
    'EventSeverity',
    'DecisionSource'
]
