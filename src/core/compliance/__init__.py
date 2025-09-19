"""
📋 COMPLIANCE MODULE
===================
Comprehensive regulatory compliance system for institutional trading.

This module provides:
- MiFID II compliance automation
- SEC regulatory framework
- Real-time compliance monitoring
- Automated regulatory reporting
- Transaction cost analysis (TCA)
- Best execution reporting
"""

from .regulatory_compliance_engine import (
    RegulatoryComplianceEngine,
    RegulatoryFramework,
    ComplianceStatus,
    ComplianceRule,
    ComplianceBreach,
    TransactionCostAnalysis,
    BestExecutionReport
)

__all__ = [
    'RegulatoryComplianceEngine',
    'RegulatoryFramework',
    'ComplianceStatus', 
    'ComplianceRule',
    'ComplianceBreach',
    'TransactionCostAnalysis',
    'BestExecutionReport'
]
