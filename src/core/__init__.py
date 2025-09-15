#!/usr/bin/env python3
"""
Core Module for XRP Trading Bot
===============================

This module contains the core configuration and interfaces for the trading bot.
"""

from .config import (
    TradingConfig, FeeRates, VolatilityRegimeFilters,
    WebSocketConfig, ExchangeConfig, config, fee_rates,
    volatility_filters, ws_config, exchange_config,
    min_profitable_move, get_config_from_env, validate_config
)

from .interfaces import (
    ILogger, IAPIClient, IWebSocketManager, IRiskManager,
    IPatternAnalyzer, IPerformanceTracker, IFeeOptimizer,
    IOrderManager, IPositionManager, ISignalGenerator,
    IStrategyExecutor, ITradingBot,
    BaseComponent, BaseAPIClient, BaseRiskManager,
    BasePatternAnalyzer, BasePerformanceTracker
)

__all__ = [
    # Configuration
    'TradingConfig',
    'FeeRates', 
    'VolatilityRegimeFilters',
    'WebSocketConfig',
    'ExchangeConfig',
    'config',
    'fee_rates',
    'volatility_filters',
    'ws_config',
    'exchange_config',
    'min_profitable_move',
    'get_config_from_env',
    'validate_config',
    
    # Interfaces
    'ILogger',
    'IAPIClient',
    'IWebSocketManager',
    'IRiskManager',
    'IPatternAnalyzer',
    'IPerformanceTracker',
    'IFeeOptimizer',
    'IOrderManager',
    'IPositionManager',
    'ISignalGenerator',
    'IStrategyExecutor',
    'ITradingBot',
    
    # Base Classes
    'BaseComponent',
    'BaseAPIClient',
    'BaseRiskManager',
    'BasePatternAnalyzer',
    'BasePerformanceTracker'
] 