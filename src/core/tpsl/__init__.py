"""
🎯 TP/SL MODULE
==============
Advanced Take Profit / Stop Loss system with high-frequency data integration.

This module provides:
- Real-time market data integration
- Order book depth-aware TP/SL placement
- ML confidence-based dynamic adjustments
- Cross-exchange optimization
- Liquidity-aware execution
- Market microstructure intelligence
"""

from .advanced_tpsl_engine import (
    AdvancedTPSLEngine,
    TPSLOrder,
    TPSLPerformance,
    TPSLStrategy,
    TPSLStatus
)

__all__ = [
    'AdvancedTPSLEngine',
    'TPSLOrder',
    'TPSLPerformance', 
    'TPSLStrategy',
    'TPSLStatus'
]
