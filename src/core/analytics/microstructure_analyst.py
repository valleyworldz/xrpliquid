"""
📈 MARKET MICROSTRUCTURE ANALYST
"I don't just see prices; I see the push and pull of liquidity beneath them."

This module implements advanced market microstructure analysis:
- Order book depth analysis
- Slippage prediction and optimization
- Market impact modeling
- Liquidity provision strategies
- Order type optimization
- Spread analysis and prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import asyncio
import time
from enum import Enum

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

@dataclass
class OrderBookLevel:
    """Represents a single level in the order book"""
    price: float
    size: float
    orders: int

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    symbol: str
    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float
    total_bid_size: float
    total_ask_size: float

@dataclass
class SlippageAnalysis:
    """Slippage analysis results"""
    expected_slippage: float
    market_impact: float
    liquidity_cost: float
    optimal_order_type: OrderType
    optimal_size: float
    confidence: float

@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a symbol"""
    symbol: str
    bid_ask_spread: float
    depth_imbalance: float
    volume_weighted_spread: float
    liquidity_score: float
    volatility: float
    turnover_rate: float

class MarketMicrostructureAnalyst:
    """
    Market Microstructure Analyst - Master of Liquidity and Execution
    
    This class analyzes market microstructure to optimize execution:
    1. Order book depth analysis
    2. Slippage prediction and optimization
    3. Market impact modeling
    4. Liquidity provision strategies
    5. Order type optimization
    6. Spread analysis and prediction
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Order book data storage
        self.order_book_history: Dict[str, deque] = {}
        self.spread_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        
        # Market impact parameters
        self.impact_config = {
            'alpha': 0.5,  # Market impact coefficient
            'beta': 0.5,   # Temporary impact coefficient
            'gamma': 0.1,  # Permanent impact coefficient
            'decay_rate': 0.1  # Impact decay rate
        }
        
        # Slippage parameters
        self.slippage_config = {
            'base_slippage': 0.0001,  # 0.01% base slippage
            'size_multiplier': 0.0002,  # Slippage per size unit
            'volatility_multiplier': 0.5,  # Volatility impact
            'liquidity_threshold': 1000.0  # Minimum liquidity threshold
        }
        
        # Liquidity parameters
        self.liquidity_config = {
            'min_spread': 0.0001,  # 0.01% minimum spread
            'max_spread': 0.01,    # 1% maximum spread
            'depth_threshold': 100.0,  # Minimum depth threshold
            'imbalance_threshold': 0.3  # Maximum imbalance threshold
        }
        
        # Performance tracking
        self.analyst_metrics = {
            'total_orders_analyzed': 0,
            'slippage_saved': 0.0,
            'liquidity_provided': 0.0,
            'spread_captured': 0.0,
            'execution_improvements': 0
        }
        
        # Initialize analysis models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize microstructure analysis models"""
        try:
            self.logger.info("Initializing microstructure analysis models...")
            
            # Initialize order book analysis
            self.order_book_analyzer = OrderBookAnalyzer()
            
            # Initialize slippage predictor
            self.slippage_predictor = SlippagePredictor()
            
            # Initialize market impact model
            self.market_impact_model = MarketImpactModel()
            
            # Initialize liquidity analyzer
            self.liquidity_analyzer = LiquidityAnalyzer()
            
            self.logger.info("Microstructure analysis models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing microstructure models: {e}")
    
    def update_order_book(self, symbol: str, order_book: OrderBookSnapshot):
        """Update order book data for analysis"""
        try:
            # Store order book snapshot
            if symbol not in self.order_book_history:
                self.order_book_history[symbol] = deque(maxlen=100)
            
            self.order_book_history[symbol].append(order_book)
            
            # Update spread history
            if symbol not in self.spread_history:
                self.spread_history[symbol] = deque(maxlen=1000)
            
            self.spread_history[symbol].append(order_book.spread)
            
            # Analyze order book
            self._analyze_order_book(symbol, order_book)
            
        except Exception as e:
            self.logger.error(f"Error updating order book for {symbol}: {e}")
    
    def _analyze_order_book(self, symbol: str, order_book: OrderBookSnapshot):
        """Analyze order book for microstructure insights"""
        try:
            # Calculate depth imbalance
            depth_imbalance = self._calculate_depth_imbalance(order_book)
            
            # Calculate volume weighted spread
            vw_spread = self._calculate_volume_weighted_spread(order_book)
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(order_book)
            
            # Store metrics
            metrics = LiquidityMetrics(
                symbol=symbol,
                bid_ask_spread=order_book.spread,
                depth_imbalance=depth_imbalance,
                volume_weighted_spread=vw_spread,
                liquidity_score=liquidity_score,
                volatility=self._calculate_volatility(symbol),
                turnover_rate=self._calculate_turnover_rate(symbol)
            )
            
            # Update liquidity analyzer
            self.liquidity_analyzer.update_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error analyzing order book for {symbol}: {e}")
    
    def _calculate_depth_imbalance(self, order_book: OrderBookSnapshot) -> float:
        """Calculate depth imbalance between bids and asks"""
        try:
            if order_book.total_bid_size == 0 and order_book.total_ask_size == 0:
                return 0.0
            
            total_size = order_book.total_bid_size + order_book.total_ask_size
            imbalance = (order_book.total_bid_size - order_book.total_ask_size) / total_size
            
            return imbalance
            
        except Exception as e:
            self.logger.error(f"Error calculating depth imbalance: {e}")
            return 0.0
    
    def _calculate_volume_weighted_spread(self, order_book: OrderBookSnapshot) -> float:
        """Calculate volume weighted spread"""
        try:
            if not order_book.bids or not order_book.asks:
                return 0.0
            
            # Calculate volume weighted prices
            bid_vw_price = sum(level.price * level.size for level in order_book.bids[:5])
            bid_total_size = sum(level.size for level in order_book.bids[:5])
            
            ask_vw_price = sum(level.price * level.size for level in order_book.asks[:5])
            ask_total_size = sum(level.size for level in order_book.asks[:5])
            
            if bid_total_size == 0 or ask_total_size == 0:
                return 0.0
            
            bid_vw_price /= bid_total_size
            ask_vw_price /= ask_total_size
            
            vw_spread = (ask_vw_price - bid_vw_price) / order_book.mid_price
            
            return vw_spread
            
        except Exception as e:
            self.logger.error(f"Error calculating volume weighted spread: {e}")
            return 0.0
    
    def _calculate_liquidity_score(self, order_book: OrderBookSnapshot) -> float:
        """Calculate overall liquidity score"""
        try:
            # Factors affecting liquidity
            spread_factor = 1.0 / (1.0 + order_book.spread * 1000)  # Lower spread = higher score
            depth_factor = min(1.0, (order_book.total_bid_size + order_book.total_ask_size) / 10000)  # Higher depth = higher score
            balance_factor = 1.0 - abs(self._calculate_depth_imbalance(order_book))  # More balanced = higher score
            
            # Weighted combination
            liquidity_score = (0.4 * spread_factor + 0.4 * depth_factor + 0.2 * balance_factor)
            
            return liquidity_score
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.0
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate recent volatility"""
        try:
            if symbol not in self.spread_history or len(self.spread_history[symbol]) < 10:
                return 0.0
            
            spreads = list(self.spread_history[symbol])[-20:]
            volatility = np.std(spreads) if len(spreads) > 1 else 0.0
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.0
    
    def _calculate_turnover_rate(self, symbol: str) -> float:
        """Calculate turnover rate"""
        try:
            # This would calculate actual turnover rate
            # For now, return a mock value
            return 0.1  # 10% turnover rate
            
        except Exception as e:
            self.logger.error(f"Error calculating turnover rate for {symbol}: {e}")
            return 0.0
    
    def analyze_slippage(self, symbol: str, side: str, size: float, 
                        current_price: float) -> SlippageAnalysis:
        """Analyze expected slippage for a trade"""
        try:
            # Get current order book
            if symbol not in self.order_book_history or not self.order_book_history[symbol]:
                return self._default_slippage_analysis()
            
            order_book = self.order_book_history[symbol][-1]
            
            # Calculate expected slippage
            expected_slippage = self._calculate_expected_slippage(order_book, side, size)
            
            # Calculate market impact
            market_impact = self._calculate_market_impact(symbol, side, size)
            
            # Calculate liquidity cost
            liquidity_cost = self._calculate_liquidity_cost(order_book, side, size)
            
            # Determine optimal order type
            optimal_order_type = self._determine_optimal_order_type(
                order_book, side, size, expected_slippage
            )
            
            # Calculate optimal size
            optimal_size = self._calculate_optimal_size(order_book, side, size)
            
            # Calculate confidence
            confidence = self._calculate_slippage_confidence(order_book, side, size)
            
            analysis = SlippageAnalysis(
                expected_slippage=expected_slippage,
                market_impact=market_impact,
                liquidity_cost=liquidity_cost,
                optimal_order_type=optimal_order_type,
                optimal_size=optimal_size,
                confidence=confidence
            )
            
            self.analyst_metrics['total_orders_analyzed'] += 1
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing slippage for {symbol}: {e}")
            return self._default_slippage_analysis()
    
    def _calculate_expected_slippage(self, order_book: OrderBookSnapshot, 
                                   side: str, size: float) -> float:
        """Calculate expected slippage for a trade"""
        try:
            if side == 'buy':
                levels = order_book.asks
            else:
                levels = order_book.bids
            
            if not levels:
                return self.slippage_config['base_slippage']
            
            # Calculate slippage by walking the order book
            remaining_size = size
            total_cost = 0.0
            weighted_price = 0.0
            
            for level in levels:
                if remaining_size <= 0:
                    break
                
                level_size = min(remaining_size, level.size)
                total_cost += level_size * level.price
                remaining_size -= level_size
            
            if size == 0:
                return 0.0
            
            # Calculate average execution price
            avg_execution_price = total_cost / (size - remaining_size)
            
            # Calculate slippage
            if side == 'buy':
                slippage = (avg_execution_price - order_book.mid_price) / order_book.mid_price
            else:
                slippage = (order_book.mid_price - avg_execution_price) / order_book.mid_price
            
            # Add base slippage and volatility adjustment
            volatility = self._calculate_volatility(order_book.symbol)
            adjusted_slippage = (
                self.slippage_config['base_slippage'] + 
                slippage + 
                volatility * self.slippage_config['volatility_multiplier']
            )
            
            return max(0.0, adjusted_slippage)
            
        except Exception as e:
            self.logger.error(f"Error calculating expected slippage: {e}")
            return self.slippage_config['base_slippage']
    
    def _calculate_market_impact(self, symbol: str, side: str, size: float) -> float:
        """Calculate market impact of a trade"""
        try:
            # Get recent volatility
            volatility = self._calculate_volatility(symbol)
            
            # Calculate market impact using square root model
            # Impact = alpha * sqrt(size) * volatility
            impact = (
                self.impact_config['alpha'] * 
                np.sqrt(size) * 
                volatility
            )
            
            return impact
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact: {e}")
            return 0.0
    
    def _calculate_liquidity_cost(self, order_book: OrderBookSnapshot, 
                                side: str, size: float) -> float:
        """Calculate liquidity cost"""
        try:
            # Liquidity cost is the spread cost
            spread_cost = order_book.spread / 2.0  # Half spread for each side
            
            # Adjust for size
            size_cost = size * self.slippage_config['size_multiplier']
            
            total_cost = spread_cost + size_cost
            
            return total_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity cost: {e}")
            return 0.0
    
    def _determine_optimal_order_type(self, order_book: OrderBookSnapshot, 
                                    side: str, size: float, 
                                    expected_slippage: float) -> OrderType:
        """Determine optimal order type based on market conditions"""
        try:
            # Analyze market conditions
            spread = order_book.spread
            depth = order_book.total_bid_size + order_book.total_ask_size
            imbalance = abs(self._calculate_depth_imbalance(order_book))
            
            # Decision logic
            if spread < self.liquidity_config['min_spread'] * 2:
                # Tight spread, use limit order
                return OrderType.LIMIT
            elif expected_slippage > 0.005:  # 0.5% slippage
                # High slippage, use post-only
                return OrderType.POST_ONLY
            elif depth < self.liquidity_config['depth_threshold']:
                # Low depth, use IOC
                return OrderType.IOC
            elif imbalance > self.liquidity_config['imbalance_threshold']:
                # High imbalance, use FOK
                return OrderType.FOK
            else:
                # Default to limit order
                return OrderType.LIMIT
                
        except Exception as e:
            self.logger.error(f"Error determining optimal order type: {e}")
            return OrderType.LIMIT
    
    def _calculate_optimal_size(self, order_book: OrderBookSnapshot, 
                              side: str, size: float) -> float:
        """Calculate optimal trade size to minimize market impact"""
        try:
            if side == 'buy':
                available_liquidity = order_book.total_ask_size
            else:
                available_liquidity = order_book.total_bid_size
            
            # Optimal size is a fraction of available liquidity
            optimal_fraction = 0.1  # 10% of available liquidity
            optimal_size = min(size, available_liquidity * optimal_fraction)
            
            # Ensure minimum size
            min_size = 1.0
            optimal_size = max(optimal_size, min_size)
            
            return optimal_size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal size: {e}")
            return size
    
    def _calculate_slippage_confidence(self, order_book: OrderBookSnapshot, 
                                     side: str, size: float) -> float:
        """Calculate confidence in slippage prediction"""
        try:
            # Factors affecting confidence
            depth_factor = min(1.0, (order_book.total_bid_size + order_book.total_ask_size) / 10000)
            spread_factor = 1.0 / (1.0 + order_book.spread * 1000)
            size_factor = 1.0 / (1.0 + size / 1000)
            
            # Weighted combination
            confidence = (0.4 * depth_factor + 0.3 * spread_factor + 0.3 * size_factor)
            
            return min(confidence, 0.95)  # Cap at 95%
            
        except Exception as e:
            self.logger.error(f"Error calculating slippage confidence: {e}")
            return 0.5
    
    def _default_slippage_analysis(self) -> SlippageAnalysis:
        """Return default slippage analysis when data is unavailable"""
        return SlippageAnalysis(
            expected_slippage=self.slippage_config['base_slippage'],
            market_impact=0.0,
            liquidity_cost=0.0,
            optimal_order_type=OrderType.LIMIT,
            optimal_size=1.0,
            confidence=0.5
        )
    
    def optimize_execution_strategy(self, symbol: str, side: str, size: float,
                                  urgency: str = 'normal') -> Dict[str, Any]:
        """Optimize execution strategy based on market microstructure"""
        try:
            # Get slippage analysis
            slippage_analysis = self.analyze_slippage(symbol, side, size, 0.0)
            
            # Get liquidity metrics
            liquidity_metrics = self.liquidity_analyzer.get_metrics(symbol)
            
            # Determine execution strategy based on urgency
            if urgency == 'urgent':
                strategy = self._urgent_execution_strategy(slippage_analysis, liquidity_metrics)
            elif urgency == 'patient':
                strategy = self._patient_execution_strategy(slippage_analysis, liquidity_metrics)
            else:
                strategy = self._normal_execution_strategy(slippage_analysis, liquidity_metrics)
            
            # Add optimization recommendations
            strategy['recommendations'] = self._generate_recommendations(
                slippage_analysis, liquidity_metrics
            )
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error optimizing execution strategy: {e}")
            return {}
    
    def _urgent_execution_strategy(self, slippage_analysis: SlippageAnalysis,
                                 liquidity_metrics: LiquidityMetrics) -> Dict[str, Any]:
        """Strategy for urgent execution"""
        return {
            'order_type': OrderType.MARKET.value,
            'size': slippage_analysis.optimal_size,
            'expected_slippage': slippage_analysis.expected_slippage,
            'execution_time': 'immediate',
            'risk_level': 'high',
            'cost_priority': 'low'
        }
    
    def _patient_execution_strategy(self, slippage_analysis: SlippageAnalysis,
                                  liquidity_metrics: LiquidityMetrics) -> Dict[str, Any]:
        """Strategy for patient execution"""
        return {
            'order_type': OrderType.POST_ONLY.value,
            'size': slippage_analysis.optimal_size,
            'expected_slippage': slippage_analysis.expected_slippage * 0.5,
            'execution_time': 'variable',
            'risk_level': 'low',
            'cost_priority': 'high'
        }
    
    def _normal_execution_strategy(self, slippage_analysis: SlippageAnalysis,
                                 liquidity_metrics: LiquidityMetrics) -> Dict[str, Any]:
        """Strategy for normal execution"""
        return {
            'order_type': slippage_analysis.optimal_order_type.value,
            'size': slippage_analysis.optimal_size,
            'expected_slippage': slippage_analysis.expected_slippage,
            'execution_time': 'moderate',
            'risk_level': 'medium',
            'cost_priority': 'medium'
        }
    
    def _generate_recommendations(self, slippage_analysis: SlippageAnalysis,
                                liquidity_metrics: LiquidityMetrics) -> List[str]:
        """Generate execution recommendations"""
        recommendations = []
        
        if slippage_analysis.expected_slippage > 0.01:  # 1% slippage
            recommendations.append("Consider splitting order into smaller chunks")
        
        if liquidity_metrics.liquidity_score < 0.5:
            recommendations.append("Low liquidity detected - consider alternative timing")
        
        if liquidity_metrics.depth_imbalance > 0.3:
            recommendations.append("High order book imbalance - adjust execution strategy")
        
        if slippage_analysis.confidence < 0.7:
            recommendations.append("Low confidence in slippage prediction - use conservative sizing")
        
        return recommendations
    
    def get_analyst_metrics(self) -> Dict[str, Any]:
        """Get analyst performance metrics"""
        return self.analyst_metrics.copy()


class OrderBookAnalyzer:
    """Analyzes order book patterns and dynamics"""
    
    def __init__(self):
        self.patterns = {}
    
    def analyze_patterns(self, order_book: OrderBookSnapshot) -> Dict[str, Any]:
        """Analyze order book patterns"""
        try:
            patterns = {}
            
            # Analyze bid-ask spread patterns
            patterns['spread_pattern'] = self._analyze_spread_pattern(order_book)
            
            # Analyze depth patterns
            patterns['depth_pattern'] = self._analyze_depth_pattern(order_book)
            
            # Analyze order size patterns
            patterns['size_pattern'] = self._analyze_size_pattern(order_book)
            
            return patterns
            
        except Exception as e:
            return {}
    
    def _analyze_spread_pattern(self, order_book: OrderBookSnapshot) -> str:
        """Analyze spread pattern"""
        spread = order_book.spread
        if spread < 0.0001:
            return 'tight'
        elif spread < 0.001:
            return 'normal'
        else:
            return 'wide'
    
    def _analyze_depth_pattern(self, order_book: OrderBookSnapshot) -> str:
        """Analyze depth pattern"""
        total_depth = order_book.total_bid_size + order_book.total_ask_size
        if total_depth > 10000:
            return 'deep'
        elif total_depth > 1000:
            return 'moderate'
        else:
            return 'shallow'
    
    def _analyze_size_pattern(self, order_book: OrderBookSnapshot) -> str:
        """Analyze order size pattern"""
        avg_bid_size = np.mean([level.size for level in order_book.bids[:5]]) if order_book.bids else 0
        avg_ask_size = np.mean([level.size for level in order_book.asks[:5]]) if order_book.asks else 0
        
        if avg_bid_size > 1000 or avg_ask_size > 1000:
            return 'large'
        elif avg_bid_size > 100 or avg_ask_size > 100:
            return 'medium'
        else:
            return 'small'


class SlippagePredictor:
    """Predicts slippage based on historical data"""
    
    def __init__(self):
        self.historical_slippage = {}
    
    def predict_slippage(self, symbol: str, side: str, size: float) -> float:
        """Predict slippage based on historical data"""
        try:
            # This would use historical data to predict slippage
            # For now, return a simple prediction
            base_slippage = 0.0001
            size_impact = size * 0.00001
            return base_slippage + size_impact
            
        except Exception as e:
            return 0.0001


class MarketImpactModel:
    """Models market impact of trades"""
    
    def __init__(self):
        self.impact_history = {}
    
    def calculate_impact(self, symbol: str, side: str, size: float) -> float:
        """Calculate market impact"""
        try:
            # Simple market impact model
            # Impact = alpha * sqrt(size) * volatility
            alpha = 0.5
            volatility = 0.02  # 2% volatility
            impact = alpha * np.sqrt(size) * volatility
            return impact
            
        except Exception as e:
            return 0.0


class LiquidityAnalyzer:
    """Analyzes liquidity conditions"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def update_metrics(self, metrics: LiquidityMetrics):
        """Update liquidity metrics"""
        try:
            symbol = metrics.symbol
            if symbol not in self.metrics_history:
                self.metrics_history[symbol] = deque(maxlen=100)
            
            self.metrics_history[symbol].append(metrics)
            
        except Exception as e:
            pass
    
    def get_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """Get latest liquidity metrics"""
        try:
            if symbol in self.metrics_history and self.metrics_history[symbol]:
                return self.metrics_history[symbol][-1]
            return None
            
        except Exception as e:
            return None

