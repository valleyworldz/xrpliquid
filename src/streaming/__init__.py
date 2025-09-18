"""
⚡ STREAMING MODULE
==================
High-frequency data streaming infrastructure for institutional trading.

This module provides:
- Real-time market data streaming
- Multi-exchange WebSocket management
- Order book and trade data processing
- Latency optimization and monitoring
- Cross-exchange arbitrage detection
"""

from .high_frequency_data_engine import (
    HighFrequencyDataEngine,
    MarketTick,
    OrderBookSnapshot,
    OrderBookLevel,
    TradeData,
    DataType,
    ExchangeType,
    calculate_spread,
    calculate_mid_price,
    calculate_order_book_imbalance
)

from .market_data_feed_manager import (
    MarketDataFeedManager,
    MarketDepthAnalysis,
    LiquidityMetrics,
    CrossExchangeArbitrage
)

__all__ = [
    'HighFrequencyDataEngine',
    'MarketDataFeedManager',
    'MarketTick',
    'OrderBookSnapshot',
    'OrderBookLevel',
    'TradeData',
    'DataType',
    'ExchangeType',
    'MarketDepthAnalysis',
    'LiquidityMetrics',
    'CrossExchangeArbitrage',
    'calculate_spread',
    'calculate_mid_price',
    'calculate_order_book_imbalance'
]
