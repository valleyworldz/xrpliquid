"""
🔄 DATA PIPELINES MODULE
========================
High-performance data processing and storage pipelines.

This module provides:
- Real-time data ingestion and processing
- Data quality validation and monitoring
- Time-series database storage
- Data archival and compression
- Real-time analytics capabilities
"""

from .streaming_data_pipeline import (
    StreamingDataPipeline,
    ProcessedTick,
    DataQualityMetrics
)

__all__ = [
    'StreamingDataPipeline',
    'ProcessedTick',
    'DataQualityMetrics'
]
