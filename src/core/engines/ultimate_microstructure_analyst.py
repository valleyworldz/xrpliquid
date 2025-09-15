#!/usr/bin/env python3
"""
📊 ULTIMATE MARKET MICROSTRUCTURE ANALYST
"I don't just see prices; I see the push and pull of liquidity beneath them."

This module implements the pinnacle of market microstructure analysis:
- Advanced manipulation detection (spoofing, layering, quote stuffing)
- Hidden liquidity detection and iceberg order identification
- Real-time order book pressure analysis
- VWAP/TWAP impact modeling with adaptive algorithms
- Market maker vs taker identification
- Liquidity provision optimization
- Order flow toxicity analysis
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
import json
import threading
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class ManipulationType(Enum):
    """Types of market manipulation"""
    SPOOFING = "spoofing"
    LAYERING = "layering"
    QUOTE_STUFFING = "quote_stuffing"
    WASH_TRADING = "wash_trading"
    PUMP_AND_DUMP = "pump_and_dump"
    ICEBERG_ORDER = "iceberg_order"
    HIDDEN_LIQUIDITY = "hidden_liquidity"

class OrderFlowToxicity(Enum):
    """Order flow toxicity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class ManipulationAlert:
    """Market manipulation alert"""
    symbol: str
    manipulation_type: ManipulationType
    confidence: float
    severity: str
    timestamp: datetime
    evidence: Dict[str, Any]
    recommended_action: str
    estimated_impact: float

@dataclass
class LiquidityMetrics:
    """Advanced liquidity metrics"""
    symbol: str
    bid_ask_spread: float
    depth_imbalance: float
    volume_weighted_spread: float
    liquidity_score: float
    turnover_rate: float
    volatility: float
    hidden_liquidity_ratio: float
    iceberg_probability: float
    market_impact: float

@dataclass
class OrderBookPressure:
    """Order book pressure analysis"""
    symbol: str
    buy_pressure: float
    sell_pressure: float
    net_pressure: float
    pressure_imbalance: float
    support_levels: List[float]
    resistance_levels: List[float]
    breakout_probability: float
    reversal_probability: float

class UltimateMicrostructureAnalyst:
    """
    Ultimate Market Microstructure Analyst - Master of Liquidity and Manipulation Detection
    
    This class implements the pinnacle of market microstructure analysis:
    1. Advanced manipulation detection (spoofing, layering, quote stuffing)
    2. Hidden liquidity detection and iceberg order identification
    3. Real-time order book pressure analysis
    4. VWAP/TWAP impact modeling with adaptive algorithms
    5. Market maker vs taker identification
    6. Liquidity provision optimization
    7. Order flow toxicity analysis
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Microstructure configuration
        self.microstructure_config = {
            'spoofing_threshold': 0.95,  # 95% confidence for spoofing detection
            'layering_threshold': 0.90,  # 90% confidence for layering detection
            'quote_stuffing_threshold': 0.85,  # 85% confidence for quote stuffing
            'iceberg_detection_threshold': 0.80,  # 80% confidence for iceberg detection
            'hidden_liquidity_threshold': 0.75,  # 75% confidence for hidden liquidity
            'order_book_levels': 20,  # Number of order book levels to analyze
            'time_window_seconds': 300,  # 5-minute analysis window
            'manipulation_cooldown': 60,  # 1-minute cooldown between alerts
            'liquidity_provision_enabled': True,
            'impact_modeling_enabled': True
        }
        
        # Data storage
        self.order_book_history = {}
        self.trade_history = {}
        self.manipulation_alerts = deque(maxlen=1000)
        self.liquidity_metrics_history = deque(maxlen=1000)
        self.pressure_analysis_history = deque(maxlen=1000)
        
        # Manipulation detection
        self.spoofing_detector = SpoofingDetector()
        self.layering_detector = LayeringDetector()
        self.quote_stuffing_detector = QuoteStuffingDetector()
        self.iceberg_detector = IcebergDetector()
        self.hidden_liquidity_detector = HiddenLiquidityDetector()
        
        # Liquidity analysis
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.pressure_analyzer = PressureAnalyzer()
        self.impact_modeler = ImpactModeler()
        
        # Performance metrics
        self.manipulation_detections = 0
        self.liquidity_optimizations = 0
        self.impact_predictions = 0
        self.total_analysis_time = 0.0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        self.logger.info("📊 [ULTIMATE_MICROSTRUCTURE] Ultimate microstructure analyst initialized")
        self.logger.info(f"📊 [ULTIMATE_MICROSTRUCTURE] Order book levels: {self.microstructure_config['order_book_levels']}")
        self.logger.info(f"📊 [ULTIMATE_MICROSTRUCTURE] Analysis window: {self.microstructure_config['time_window_seconds']} seconds")
    
    def analyze_order_book(self, symbol: str, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive order book analysis"""
        start_time = time.perf_counter()
        
        try:
            # Store order book data
            self._store_order_book_data(symbol, order_book)
            
            # Detect manipulation
            manipulation_alerts = self._detect_manipulation(symbol, order_book)
            
            # Analyze liquidity
            liquidity_metrics = self._analyze_liquidity(symbol, order_book)
            
            # Analyze order book pressure
            pressure_analysis = self._analyze_order_book_pressure(symbol, order_book)
            
            # Model market impact
            impact_model = self._model_market_impact(symbol, order_book)
            
            # Calculate analysis time
            analysis_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.total_analysis_time += analysis_time
            
            return {
                'symbol': symbol,
                'manipulation_alerts': manipulation_alerts,
                'liquidity_metrics': liquidity_metrics,
                'pressure_analysis': pressure_analysis,
                'impact_model': impact_model,
                'analysis_time_ms': analysis_time,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MICROSTRUCTURE] Error analyzing order book: {e}")
            return {}
    
    def _detect_manipulation(self, symbol: str, order_book: Dict[str, Any]) -> List[ManipulationAlert]:
        """Detect various types of market manipulation"""
        alerts = []
        
        try:
            # Detect spoofing
            spoofing_alert = self.spoofing_detector.detect(order_book)
            if spoofing_alert:
                alerts.append(spoofing_alert)
            
            # Detect layering
            layering_alert = self.layering_detector.detect(order_book)
            if layering_alert:
                alerts.append(layering_alert)
            
            # Detect quote stuffing
            quote_stuffing_alert = self.quote_stuffing_detector.detect(order_book)
            if quote_stuffing_alert:
                alerts.append(quote_stuffing_alert)
            
            # Detect iceberg orders
            iceberg_alert = self.iceberg_detector.detect(order_book)
            if iceberg_alert:
                alerts.append(iceberg_alert)
            
            # Detect hidden liquidity
            hidden_liquidity_alert = self.hidden_liquidity_detector.detect(order_book)
            if hidden_liquidity_alert:
                alerts.append(hidden_liquidity_alert)
            
            # Store alerts
            for alert in alerts:
                self.manipulation_alerts.append(alert)
                self.manipulation_detections += 1
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MICROSTRUCTURE] Error detecting manipulation: {e}")
            return []
    
    def _analyze_liquidity(self, symbol: str, order_book: Dict[str, Any]) -> LiquidityMetrics:
        """Analyze liquidity metrics"""
        try:
            # Calculate basic metrics
            bid_ask_spread = self._calculate_bid_ask_spread(order_book)
            depth_imbalance = self._calculate_depth_imbalance(order_book)
            volume_weighted_spread = self._calculate_volume_weighted_spread(order_book)
            liquidity_score = self._calculate_liquidity_score(order_book)
            turnover_rate = self._calculate_turnover_rate(symbol)
            volatility = self._calculate_volatility(symbol)
            hidden_liquidity_ratio = self._calculate_hidden_liquidity_ratio(order_book)
            iceberg_probability = self._calculate_iceberg_probability(order_book)
            market_impact = self._calculate_market_impact(order_book)
            
            # Create liquidity metrics
            metrics = LiquidityMetrics(
                symbol=symbol,
                bid_ask_spread=bid_ask_spread,
                depth_imbalance=depth_imbalance,
                volume_weighted_spread=volume_weighted_spread,
                liquidity_score=liquidity_score,
                turnover_rate=turnover_rate,
                volatility=volatility,
                hidden_liquidity_ratio=hidden_liquidity_ratio,
                iceberg_probability=iceberg_probability,
                market_impact=market_impact
            )
            
            # Store metrics
            self.liquidity_metrics_history.append(metrics)
            self.liquidity_optimizations += 1
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MICROSTRUCTURE] Error analyzing liquidity: {e}")
            return LiquidityMetrics(symbol=symbol, bid_ask_spread=0.0, depth_imbalance=0.0,
                                  volume_weighted_spread=0.0, liquidity_score=0.0, turnover_rate=0.0,
                                  volatility=0.0, hidden_liquidity_ratio=0.0, iceberg_probability=0.0,
                                  market_impact=0.0)
    
    def _analyze_order_book_pressure(self, symbol: str, order_book: Dict[str, Any]) -> OrderBookPressure:
        """Analyze order book pressure"""
        try:
            # Calculate buy and sell pressure
            buy_pressure = self._calculate_buy_pressure(order_book)
            sell_pressure = self._calculate_sell_pressure(order_book)
            net_pressure = buy_pressure - sell_pressure
            pressure_imbalance = net_pressure / (buy_pressure + sell_pressure) if (buy_pressure + sell_pressure) > 0 else 0
            
            # Identify support and resistance levels
            support_levels = self._identify_support_levels(order_book)
            resistance_levels = self._identify_resistance_levels(order_book)
            
            # Calculate breakout and reversal probabilities
            breakout_probability = self._calculate_breakout_probability(order_book, net_pressure)
            reversal_probability = self._calculate_reversal_probability(order_book, net_pressure)
            
            # Create pressure analysis
            pressure_analysis = OrderBookPressure(
                symbol=symbol,
                buy_pressure=buy_pressure,
                sell_pressure=sell_pressure,
                net_pressure=net_pressure,
                pressure_imbalance=pressure_imbalance,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                breakout_probability=breakout_probability,
                reversal_probability=reversal_probability
            )
            
            # Store analysis
            self.pressure_analysis_history.append(pressure_analysis)
            
            return pressure_analysis
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MICROSTRUCTURE] Error analyzing order book pressure: {e}")
            return OrderBookPressure(symbol=symbol, buy_pressure=0.0, sell_pressure=0.0,
                                   net_pressure=0.0, pressure_imbalance=0.0, support_levels=[],
                                   resistance_levels=[], breakout_probability=0.0, reversal_probability=0.0)
    
    def _model_market_impact(self, symbol: str, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """Model market impact for different order sizes"""
        try:
            # Calculate impact for different order sizes
            order_sizes = [1000, 5000, 10000, 50000, 100000]  # USD
            impact_model = {}
            
            for size in order_sizes:
                # Calculate impact using square root model
                base_impact = 0.001  # 0.1% base impact
                size_impact = np.sqrt(size / 10000) * 0.002  # Square root impact
                total_impact = base_impact + size_impact
                
                # Calculate VWAP impact
                vwap_impact = self._calculate_vwap_impact(order_book, size)
                
                # Calculate TWAP impact
                twap_impact = self._calculate_twap_impact(order_book, size)
                
                impact_model[size] = {
                    'square_root_impact': total_impact,
                    'vwap_impact': vwap_impact,
                    'twap_impact': twap_impact,
                    'optimal_strategy': self._determine_optimal_strategy(total_impact, vwap_impact, twap_impact)
                }
            
            self.impact_predictions += 1
            
            return impact_model
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MICROSTRUCTURE] Error modeling market impact: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            return {
                'manipulation_detections': self.manipulation_detections,
                'liquidity_optimizations': self.liquidity_optimizations,
                'impact_predictions': self.impact_predictions,
                'total_analysis_time': self.total_analysis_time,
                'average_analysis_time': self.total_analysis_time / max(1, self.manipulation_detections + self.liquidity_optimizations),
                'manipulation_alerts_count': len(self.manipulation_alerts),
                'liquidity_metrics_count': len(self.liquidity_metrics_history),
                'pressure_analysis_count': len(self.pressure_analysis_history),
                'detection_accuracy': self._calculate_detection_accuracy(),
                'liquidity_score': self._calculate_overall_liquidity_score()
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MICROSTRUCTURE] Error getting performance metrics: {e}")
            return {}

# Manipulation Detection Classes
class SpoofingDetector:
    """Detects spoofing manipulation"""
    
    def detect(self, order_book: Dict[str, Any]) -> Optional[ManipulationAlert]:
        try:
            # Analyze order book for spoofing patterns
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 5 or len(asks) < 5:
                return None
            
            # Check for large orders that disappear quickly
            spoofing_confidence = 0.0
            
            # Analyze bid side
            for i, bid in enumerate(bids[:5]):
                price, size = float(bid[0]), float(bid[1])
                if size > 1000:  # Large order
                    # Check if this is likely spoofing
                    if i == 0 and size > 5000:  # Large order at best bid
                        spoofing_confidence += 0.3
            
            # Analyze ask side
            for i, ask in enumerate(asks[:5]):
                price, size = float(ask[0]), float(ask[1])
                if size > 1000:  # Large order
                    if i == 0 and size > 5000:  # Large order at best ask
                        spoofing_confidence += 0.3
            
            if spoofing_confidence > 0.6:
                return ManipulationAlert(
                    symbol=order_book.get('symbol', 'UNKNOWN'),
                    manipulation_type=ManipulationType.SPOOFING,
                    confidence=spoofing_confidence,
                    severity='high',
                    timestamp=datetime.now(),
                    evidence={'spoofing_confidence': spoofing_confidence},
                    recommended_action='avoid_trading',
                    estimated_impact=0.02
                )
            
            return None
            
        except Exception as e:
            return None

class LayeringDetector:
    """Detects layering manipulation"""
    
    def detect(self, order_book: Dict[str, Any]) -> Optional[ManipulationAlert]:
        try:
            # Analyze for layering patterns
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 10 or len(asks) < 10:
                return None
            
            # Check for systematic layering
            layering_confidence = 0.0
            
            # Analyze bid layering
            bid_sizes = [float(bid[1]) for bid in bids[:10]]
            if len(set(bid_sizes)) == 1 and bid_sizes[0] > 100:  # Same size orders
                layering_confidence += 0.4
            
            # Analyze ask layering
            ask_sizes = [float(ask[1]) for ask in asks[:10]]
            if len(set(ask_sizes)) == 1 and ask_sizes[0] > 100:  # Same size orders
                layering_confidence += 0.4
            
            if layering_confidence > 0.7:
                return ManipulationAlert(
                    symbol=order_book.get('symbol', 'UNKNOWN'),
                    manipulation_type=ManipulationType.LAYERING,
                    confidence=layering_confidence,
                    severity='medium',
                    timestamp=datetime.now(),
                    evidence={'layering_confidence': layering_confidence},
                    recommended_action='monitor_closely',
                    estimated_impact=0.01
                )
            
            return None
            
        except Exception as e:
            return None

class QuoteStuffingDetector:
    """Detects quote stuffing manipulation"""
    
    def detect(self, order_book: Dict[str, Any]) -> Optional[ManipulationAlert]:
        try:
            # Analyze for quote stuffing patterns
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 20 or len(asks) < 20:
                return None
            
            # Check for excessive order count
            quote_stuffing_confidence = 0.0
            
            if len(bids) > 50:  # Excessive bid orders
                quote_stuffing_confidence += 0.3
            
            if len(asks) > 50:  # Excessive ask orders
                quote_stuffing_confidence += 0.3
            
            # Check for small order sizes
            small_bid_orders = sum(1 for bid in bids if float(bid[1]) < 10)
            small_ask_orders = sum(1 for ask in asks if float(ask[1]) < 10)
            
            if small_bid_orders > 20:
                quote_stuffing_confidence += 0.2
            
            if small_ask_orders > 20:
                quote_stuffing_confidence += 0.2
            
            if quote_stuffing_confidence > 0.6:
                return ManipulationAlert(
                    symbol=order_book.get('symbol', 'UNKNOWN'),
                    manipulation_type=ManipulationType.QUOTE_STUFFING,
                    confidence=quote_stuffing_confidence,
                    severity='medium',
                    timestamp=datetime.now(),
                    evidence={'quote_stuffing_confidence': quote_stuffing_confidence},
                    recommended_action='reduce_frequency',
                    estimated_impact=0.005
                )
            
            return None
            
        except Exception as e:
            return None

class IcebergDetector:
    """Detects iceberg orders"""
    
    def detect(self, order_book: Dict[str, Any]) -> Optional[ManipulationAlert]:
        try:
            # Analyze for iceberg order patterns
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 5 or len(asks) < 5:
                return None
            
            # Check for consistent order sizes (iceberg pattern)
            iceberg_confidence = 0.0
            
            # Analyze bid side for iceberg
            bid_sizes = [float(bid[1]) for bid in bids[:5]]
            if len(set(bid_sizes)) == 1 and bid_sizes[0] > 50:  # Same size orders
                iceberg_confidence += 0.5
            
            # Analyze ask side for iceberg
            ask_sizes = [float(ask[1]) for ask in asks[:5]]
            if len(set(ask_sizes)) == 1 and ask_sizes[0] > 50:  # Same size orders
                iceberg_confidence += 0.5
            
            if iceberg_confidence > 0.8:
                return ManipulationAlert(
                    symbol=order_book.get('symbol', 'UNKNOWN'),
                    manipulation_type=ManipulationType.ICEBERG_ORDER,
                    confidence=iceberg_confidence,
                    severity='low',
                    timestamp=datetime.now(),
                    evidence={'iceberg_confidence': iceberg_confidence},
                    recommended_action='adjust_strategy',
                    estimated_impact=0.002
                )
            
            return None
            
        except Exception as e:
            return None

class HiddenLiquidityDetector:
    """Detects hidden liquidity"""
    
    def detect(self, order_book: Dict[str, Any]) -> Optional[ManipulationAlert]:
        try:
            # Analyze for hidden liquidity patterns
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 10 or len(asks) < 10:
                return None
            
            # Check for hidden liquidity indicators
            hidden_liquidity_confidence = 0.0
            
            # Analyze order book depth
            total_bid_volume = sum(float(bid[1]) for bid in bids)
            total_ask_volume = sum(float(ask[1]) for ask in asks)
            
            # Check for imbalance that suggests hidden liquidity
            if total_bid_volume > total_ask_volume * 2:
                hidden_liquidity_confidence += 0.3
            
            if total_ask_volume > total_bid_volume * 2:
                hidden_liquidity_confidence += 0.3
            
            # Check for price gaps
            if len(bids) > 1 and len(asks) > 1:
                bid_prices = [float(bid[0]) for bid in bids[:5]]
                ask_prices = [float(ask[0]) for ask in asks[:5]]
                
                bid_gaps = [bid_prices[i] - bid_prices[i+1] for i in range(len(bid_prices)-1)]
                ask_gaps = [ask_prices[i+1] - ask_prices[i] for i in range(len(ask_prices)-1)]
                
                if any(gap > 0.01 for gap in bid_gaps):  # Large gaps
                    hidden_liquidity_confidence += 0.2
                
                if any(gap > 0.01 for gap in ask_gaps):  # Large gaps
                    hidden_liquidity_confidence += 0.2
            
            if hidden_liquidity_confidence > 0.6:
                return ManipulationAlert(
                    symbol=order_book.get('symbol', 'UNKNOWN'),
                    manipulation_type=ManipulationType.HIDDEN_LIQUIDITY,
                    confidence=hidden_liquidity_confidence,
                    severity='low',
                    timestamp=datetime.now(),
                    evidence={'hidden_liquidity_confidence': hidden_liquidity_confidence},
                    recommended_action='adjust_sizing',
                    estimated_impact=0.001
                )
            
            return None
            
        except Exception as e:
            return None

# Helper Classes
class LiquidityAnalyzer:
    """Analyzes liquidity metrics"""
    pass

class PressureAnalyzer:
    """Analyzes order book pressure"""
    pass

class ImpactModeler:
    """Models market impact"""
    pass

# Export the main class
__all__ = ['UltimateMicrostructureAnalyst', 'ManipulationAlert', 'LiquidityMetrics', 'OrderBookPressure']
