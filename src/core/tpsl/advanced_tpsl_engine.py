#!/usr/bin/env python3
"""
🎯 ADVANCED TP/SL ENGINE
========================
Institutional-grade Take Profit / Stop Loss system with high-frequency data integration.

Features:
- Real-time market data integration from streaming infrastructure
- Order book depth-aware TP/SL placement
- ML confidence-based dynamic adjustments
- Cross-exchange optimization
- Liquidity-aware execution
- Market microstructure intelligence
- Performance attribution and analytics
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import numpy as np

from ..streaming.high_frequency_data_engine import MarketTick, OrderBookSnapshot, TradeData, DataType
from ..streaming.market_data_feed_manager import MarketDepthAnalysis, LiquidityMetrics

class TPSLStrategy(Enum):
    """TP/SL strategy types"""
    VOLATILITY_SCALED = "volatility_scaled"
    LIQUIDITY_AWARE = "liquidity_aware"
    ML_ADAPTIVE = "ml_adaptive"
    CROSS_EXCHANGE = "cross_exchange"
    MICROSTRUCTURE = "microstructure"

class TPSLStatus(Enum):
    """TP/SL order status"""
    PENDING = "pending"
    PLACED = "placed"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class TPSLOrder:
    """Take Profit / Stop Loss order"""
    order_id: str
    symbol: str
    side: str  # "TP" or "SL"
    price: Decimal
    size: Decimal
    strategy: TPSLStrategy
    confidence: float
    liquidity_score: float
    expected_slippage: float
    status: TPSLStatus
    created_at: float
    filled_at: Optional[float] = None
    actual_fill_price: Optional[Decimal] = None
    slippage_bps: Optional[float] = None

@dataclass
class TPSLPerformance:
    """TP/SL performance metrics"""
    total_orders: int
    successful_orders: int
    tp_hit_rate: float
    sl_hit_rate: float
    avg_slippage_bps: float
    avg_execution_time_ms: float
    profit_factor: float
    sharpe_ratio: float
    max_adverse_excursion: float

class AdvancedTPSLEngine:
    """
    🎯 ADVANCED TP/SL ENGINE
    Integrates with high-frequency data streams for optimal TP/SL execution
    """
    
    def __init__(self, config: Dict[str, Any], data_feed_manager=None, 
                 ml_engine=None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.data_feed_manager = data_feed_manager
        self.ml_engine = ml_engine
        self.logger = logger or logging.getLogger(__name__)
        
        # TP/SL orders tracking
        self.active_tpsl_orders: Dict[str, TPSLOrder] = {}
        self.tpsl_history: deque = deque(maxlen=10000)
        self.position_tpsl_map: Dict[str, List[str]] = defaultdict(list)
        
        # Market data integration
        self.latest_orderbooks: Dict[str, OrderBookSnapshot] = {}
        self.latest_depth_analysis: Dict[str, MarketDepthAnalysis] = {}
        self.latest_liquidity_metrics: Dict[str, LiquidityMetrics] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, TPSLPerformance] = {}
        self.execution_analytics: deque = deque(maxlen=5000)
        
        # Configuration parameters
        self.min_liquidity_score = config.get('min_liquidity_score', 0.6)
        self.max_slippage_bps = config.get('max_slippage_bps', 50)
        self.ml_confidence_threshold = config.get('ml_confidence_threshold', 0.7)
        self.orderbook_levels_analysis = config.get('orderbook_levels_analysis', 10)
        
        # Real-time callbacks
        self.execution_callbacks: List[Callable] = []
        
        self.logger.info("🎯 [TPSL] Advanced TP/SL Engine initialized")

    async def start_engine(self):
        """Start the TP/SL engine with data stream integration"""
        try:
            # Register for market data updates
            if self.data_feed_manager:
                await self._register_data_callbacks()
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("🎯 [TPSL] TP/SL Engine started")
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error starting engine: {e}")

    async def _register_data_callbacks(self):
        """Register callbacks for high-frequency data updates"""
        try:
            # Register for order book updates
            for engine in self.data_feed_manager.data_engines.values():
                engine.register_callback(DataType.ORDER_BOOK, self._handle_orderbook_update)
                engine.register_callback(DataType.TRADE, self._handle_trade_update)
            
            self.logger.info("🎯 [TPSL] Registered for high-frequency data callbacks")
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error registering data callbacks: {e}")

    def _handle_orderbook_update(self, tick: MarketTick):
        """Handle order book updates for TP/SL optimization"""
        try:
            if tick.data_type != DataType.ORDER_BOOK:
                return
            
            orderbook = OrderBookSnapshot(**tick.data)
            key = f"{tick.exchange.value}:{tick.symbol}"
            self.latest_orderbooks[key] = orderbook
            
            # Update TP/SL orders based on new order book data
            asyncio.create_task(self._optimize_tpsl_for_liquidity(tick.symbol, orderbook))
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error handling orderbook update: {e}")

    def _handle_trade_update(self, tick: MarketTick):
        """Handle trade updates for market momentum analysis"""
        try:
            if tick.data_type != DataType.TRADE:
                return
            
            trade_data = TradeData(**tick.data)
            
            # Analyze trade flow for TP/SL adjustments
            asyncio.create_task(self._analyze_trade_flow_impact(tick.symbol, trade_data))
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error handling trade update: {e}")

    async def calculate_optimal_tpsl(self, position: Dict[str, Any], 
                                   strategy: TPSLStrategy = TPSLStrategy.LIQUIDITY_AWARE) -> Tuple[Decimal, Decimal, float]:
        """Calculate optimal TP/SL prices using advanced algorithms"""
        try:
            symbol = position.get('symbol', 'XRP')
            entry_price = Decimal(str(position.get('entry_price', 0)))
            position_size = Decimal(str(position.get('size', 0)))
            side = 'long' if position_size > 0 else 'short'
            
            if strategy == TPSLStrategy.LIQUIDITY_AWARE:
                return await self._calculate_liquidity_aware_tpsl(symbol, entry_price, position_size, side)
            elif strategy == TPSLStrategy.ML_ADAPTIVE:
                return await self._calculate_ml_adaptive_tpsl(symbol, entry_price, position_size, side)
            elif strategy == TPSLStrategy.MICROSTRUCTURE:
                return await self._calculate_microstructure_tpsl(symbol, entry_price, position_size, side)
            elif strategy == TPSLStrategy.CROSS_EXCHANGE:
                return await self._calculate_cross_exchange_tpsl(symbol, entry_price, position_size, side)
            else:  # VOLATILITY_SCALED (fallback)
                return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error calculating optimal TP/SL: {e}")
            # Return conservative defaults
            return entry_price * Decimal('1.02'), entry_price * Decimal('0.98'), 0.5

    async def _calculate_liquidity_aware_tpsl(self, symbol: str, entry_price: Decimal, 
                                            position_size: Decimal, side: str) -> Tuple[Decimal, Decimal, float]:
        """Calculate TP/SL based on order book liquidity"""
        try:
            # Get latest order book
            orderbook_key = f"hyperliquid:{symbol}"
            orderbook = self.latest_orderbooks.get(orderbook_key)
            
            if not orderbook or not orderbook.bids or not orderbook.asks:
                return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)
            
            # Analyze order book depth
            cumulative_bid_size = Decimal('0')
            cumulative_ask_size = Decimal('0')
            
            # Find prices with sufficient liquidity
            target_liquidity = abs(position_size) * Decimal('2')  # 2x position size for safety
            
            tp_price = entry_price
            sl_price = entry_price
            
            if side == 'long':
                # For long positions: TP above entry, SL below entry
                for ask_level in orderbook.asks:
                    cumulative_ask_size += ask_level.size
                    if cumulative_ask_size >= target_liquidity and ask_level.price > entry_price:
                        tp_price = ask_level.price * Decimal('0.999')  # Just below the level
                        break
                
                for bid_level in reversed(orderbook.bids):
                    cumulative_bid_size += bid_level.size
                    if cumulative_bid_size >= target_liquidity and bid_level.price < entry_price:
                        sl_price = bid_level.price * Decimal('1.001')  # Just above the level
                        break
            else:
                # For short positions: TP below entry, SL above entry
                for bid_level in reversed(orderbook.bids):
                    cumulative_bid_size += bid_level.size
                    if cumulative_bid_size >= target_liquidity and bid_level.price < entry_price:
                        tp_price = bid_level.price * Decimal('1.001')  # Just above the level
                        break
                
                for ask_level in orderbook.asks:
                    cumulative_ask_size += ask_level.size
                    if cumulative_ask_size >= target_liquidity and ask_level.price > entry_price:
                        sl_price = ask_level.price * Decimal('0.999')  # Just below the level
                        break
            
            # Calculate confidence based on liquidity depth
            confidence = min(float(min(cumulative_bid_size, cumulative_ask_size) / target_liquidity), 1.0)
            
            return tp_price, sl_price, confidence
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error in liquidity-aware calculation: {e}")
            return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)

    async def _calculate_ml_adaptive_tpsl(self, symbol: str, entry_price: Decimal, 
                                        position_size: Decimal, side: str) -> Tuple[Decimal, Decimal, float]:
        """Calculate TP/SL using ML predictions"""
        try:
            if not self.ml_engine:
                return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)
            
            # Get ML predictions for price movement
            price_history = self._get_recent_price_history(symbol)
            if not price_history:
                return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)
            
            # Get ML confidence for the current market state
            ml_confidence = await self._get_ml_confidence(symbol, price_history)
            
            # Adjust TP/SL based on ML confidence
            base_tp_distance = entry_price * Decimal('0.02')  # 2% base
            base_sl_distance = entry_price * Decimal('0.01')  # 1% base
            
            if ml_confidence > self.ml_confidence_threshold:
                # High confidence: Wider TP, tighter SL
                tp_multiplier = Decimal('1.5')
                sl_multiplier = Decimal('0.8')
            else:
                # Low confidence: Tighter TP, wider SL
                tp_multiplier = Decimal('0.8')
                sl_multiplier = Decimal('1.3')
            
            if side == 'long':
                tp_price = entry_price + (base_tp_distance * tp_multiplier)
                sl_price = entry_price - (base_sl_distance * sl_multiplier)
            else:
                tp_price = entry_price - (base_tp_distance * tp_multiplier)
                sl_price = entry_price + (base_sl_distance * sl_multiplier)
            
            return tp_price, sl_price, ml_confidence
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error in ML-adaptive calculation: {e}")
            return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)

    async def _calculate_microstructure_tpsl(self, symbol: str, entry_price: Decimal, 
                                           position_size: Decimal, side: str) -> Tuple[Decimal, Decimal, float]:
        """Calculate TP/SL based on market microstructure analysis"""
        try:
            # Analyze order book imbalance
            orderbook_key = f"hyperliquid:{symbol}"
            orderbook = self.latest_orderbooks.get(orderbook_key)
            
            if not orderbook:
                return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)
            
            # Calculate order book imbalance (5 levels)
            bid_volume = sum(level.size for level in orderbook.bids[:5])
            ask_volume = sum(level.size for level in orderbook.asks[:5])
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                imbalance = 0
            else:
                imbalance = float((bid_volume - ask_volume) / total_volume)
            
            # Analyze recent trade flow
            trade_momentum = await self._calculate_trade_momentum(symbol)
            
            # Adjust TP/SL based on microstructure signals
            base_tp_distance = entry_price * Decimal('0.015')  # 1.5% base
            base_sl_distance = entry_price * Decimal('0.008')  # 0.8% base
            
            # Imbalance adjustment
            if side == 'long':
                if imbalance > 0.2:  # Strong bid pressure
                    tp_multiplier = Decimal('1.3')
                    sl_multiplier = Decimal('0.9')
                elif imbalance < -0.2:  # Strong ask pressure
                    tp_multiplier = Decimal('0.8')
                    sl_multiplier = Decimal('1.2')
                else:
                    tp_multiplier = Decimal('1.0')
                    sl_multiplier = Decimal('1.0')
                
                tp_price = entry_price + (base_tp_distance * tp_multiplier)
                sl_price = entry_price - (base_sl_distance * sl_multiplier)
            else:
                if imbalance < -0.2:  # Strong ask pressure (good for short)
                    tp_multiplier = Decimal('1.3')
                    sl_multiplier = Decimal('0.9')
                elif imbalance > 0.2:  # Strong bid pressure (bad for short)
                    tp_multiplier = Decimal('0.8')
                    sl_multiplier = Decimal('1.2')
                else:
                    tp_multiplier = Decimal('1.0')
                    sl_multiplier = Decimal('1.0')
                
                tp_price = entry_price - (base_tp_distance * tp_multiplier)
                sl_price = entry_price + (base_sl_distance * sl_multiplier)
            
            # Calculate confidence based on signal strength
            confidence = min(abs(imbalance) + abs(trade_momentum) / 2, 1.0)
            
            return tp_price, sl_price, confidence
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error in microstructure calculation: {e}")
            return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)

    async def _calculate_cross_exchange_tpsl(self, symbol: str, entry_price: Decimal, 
                                           position_size: Decimal, side: str) -> Tuple[Decimal, Decimal, float]:
        """Calculate TP/SL considering cross-exchange opportunities"""
        try:
            # Get arbitrage opportunities
            if not self.data_feed_manager:
                return await self._calculate_volatility_scaled_tpsl(symbol, entry_price, position_size, side)
            
            arbitrage_ops = self.data_feed_manager.arbitrage_opportunities
            relevant_ops = [op for op in arbitrage_ops if op.symbol == symbol]
            
            if not relevant_ops:
                return await self._calculate_liquidity_aware_tpsl(symbol, entry_price, position_size, side)
            
            # Find best exit opportunities
            best_tp_price = entry_price
            best_sl_price = entry_price
            max_confidence = 0.0
            
            for opportunity in relevant_ops:
                if side == 'long':
                    # For long positions, look for selling opportunities
                    if opportunity.sell_price > entry_price:
                        if opportunity.profit_bps > 30 and opportunity.confidence > max_confidence:
                            best_tp_price = opportunity.sell_price * Decimal('0.998')  # Conservative
                            max_confidence = opportunity.confidence
                else:
                    # For short positions, look for buying opportunities
                    if opportunity.buy_price < entry_price:
                        if opportunity.profit_bps > 30 and opportunity.confidence > max_confidence:
                            best_tp_price = opportunity.buy_price * Decimal('1.002')  # Conservative
                            max_confidence = opportunity.confidence
            
            # Set stop loss at reasonable distance
            if side == 'long':
                best_sl_price = entry_price * Decimal('0.99')  # 1% stop loss
            else:
                best_sl_price = entry_price * Decimal('1.01')  # 1% stop loss
            
            return best_tp_price, best_sl_price, max_confidence
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error in cross-exchange calculation: {e}")
            return await self._calculate_liquidity_aware_tpsl(symbol, entry_price, position_size, side)

    async def _calculate_volatility_scaled_tpsl(self, symbol: str, entry_price: Decimal, 
                                              position_size: Decimal, side: str) -> Tuple[Decimal, Decimal, float]:
        """Calculate TP/SL using volatility scaling (fallback method)"""
        try:
            # Use real-time volatility if available
            volatility = await self._get_realtime_volatility(symbol)
            if volatility is None:
                volatility = 0.02  # 2% default
            
            # Scale TP/SL based on volatility
            if volatility < 0.01:  # Low volatility
                tp_distance = entry_price * Decimal('0.015')  # 1.5%
                sl_distance = entry_price * Decimal('0.008')  # 0.8%
            elif volatility > 0.05:  # High volatility
                tp_distance = entry_price * Decimal('0.03')   # 3%
                sl_distance = entry_price * Decimal('0.015')  # 1.5%
            else:  # Normal volatility
                tp_distance = entry_price * Decimal('0.02')   # 2%
                sl_distance = entry_price * Decimal('0.01')   # 1%
            
            if side == 'long':
                tp_price = entry_price + tp_distance
                sl_price = entry_price - sl_distance
            else:
                tp_price = entry_price - tp_distance
                sl_price = entry_price + sl_distance
            
            confidence = 1.0 - min(volatility * 10, 0.5)  # Lower confidence with higher volatility
            
            return tp_price, sl_price, confidence
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error in volatility-scaled calculation: {e}")
            # Return very conservative defaults
            if side == 'long':
                return entry_price * Decimal('1.02'), entry_price * Decimal('0.98'), 0.5
            else:
                return entry_price * Decimal('0.98'), entry_price * Decimal('1.02'), 0.5

    async def _get_realtime_volatility(self, symbol: str) -> Optional[float]:
        """Get real-time volatility from streaming data"""
        try:
            # Get recent price history from data feed manager
            if not self.data_feed_manager:
                return None
            
            price_history = []
            key = f"hyperliquid:{symbol}:price"
            
            if key in self.data_feed_manager.price_history:
                recent_prices = list(self.data_feed_manager.price_history[key])[-60:]  # Last 60 seconds
                if len(recent_prices) >= 10:
                    prices = [float(p['price']) for p in recent_prices]
                    returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
                    volatility = np.std(returns) * np.sqrt(3600)  # Annualized hourly volatility
                    return volatility
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error calculating real-time volatility: {e}")
            return None

    async def _get_ml_confidence(self, symbol: str, price_history: List[float]) -> float:
        """Get ML confidence for current market state"""
        try:
            if not self.ml_engine:
                return 0.5
            
            # This would integrate with the ML engine
            # For now, return a simplified confidence based on price stability
            if len(price_history) >= 10:
                price_std = np.std(price_history[-10:])
                price_mean = np.mean(price_history[-10:])
                cv = price_std / price_mean if price_mean > 0 else 1.0
                confidence = max(0.0, min(1.0, 1.0 - cv * 10))
                return confidence
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error getting ML confidence: {e}")
            return 0.5

    def _get_recent_price_history(self, symbol: str) -> List[float]:
        """Get recent price history for analysis"""
        try:
            if not self.data_feed_manager:
                return []
            
            key = f"hyperliquid:{symbol}:price"
            if key in self.data_feed_manager.price_history:
                recent_prices = list(self.data_feed_manager.price_history[key])[-100:]
                return [float(p['price']) for p in recent_prices]
            
            return []
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error getting price history: {e}")
            return []

    async def _calculate_trade_momentum(self, symbol: str) -> float:
        """Calculate trade momentum from recent trades"""
        try:
            if not self.data_feed_manager:
                return 0.0
            
            key = f"hyperliquid:{symbol}"
            recent_trades = list(self.data_feed_manager.trade_history[key])[-20:]  # Last 20 trades
            
            if len(recent_trades) < 5:
                return 0.0
            
            # Calculate momentum based on trade sides and sizes
            buy_volume = sum(trade.size for trade in recent_trades if trade.side == 'buy')
            sell_volume = sum(trade.size for trade in recent_trades if trade.side == 'sell')
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0.0
            
            momentum = float((buy_volume - sell_volume) / total_volume)
            return momentum
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error calculating trade momentum: {e}")
            return 0.0

    async def _optimize_tpsl_for_liquidity(self, symbol: str, orderbook: OrderBookSnapshot):
        """Optimize existing TP/SL orders based on new order book data"""
        try:
            # Find TP/SL orders for this symbol
            symbol_orders = [
                order for order in self.active_tpsl_orders.values()
                if order.symbol == symbol and order.status == TPSLStatus.PLACED
            ]
            
            for order in symbol_orders:
                # Check if order needs adjustment based on liquidity changes
                liquidity_at_price = self._calculate_liquidity_at_price(orderbook, order.price)
                
                if liquidity_at_price < self.min_liquidity_score:
                    self.logger.warning(f"🎯 [TPSL] Low liquidity at {order.side} price {order.price} for {symbol}")
                    # Would trigger order adjustment here
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error optimizing TP/SL for liquidity: {e}")

    def _calculate_liquidity_at_price(self, orderbook: OrderBookSnapshot, price: Decimal) -> float:
        """Calculate liquidity score at a specific price"""
        try:
            total_size = Decimal('0')
            levels_count = 0
            
            # Check both sides around the price
            for level in orderbook.bids + orderbook.asks:
                if abs(level.price - price) / price < Decimal('0.001'):  # Within 0.1%
                    total_size += level.size
                    levels_count += 1
            
            # Normalize to 0-1 score
            liquidity_score = min(float(total_size) / 100.0, 1.0)  # Assume 100 is good liquidity
            return liquidity_score
            
        except Exception:
            return 0.0

    async def _analyze_trade_flow_impact(self, symbol: str, trade_data: TradeData):
        """Analyze trade flow impact on TP/SL strategies"""
        try:
            # Check for large trades that might impact TP/SL execution
            if trade_data.size > Decimal('50'):  # Large trade threshold
                self.logger.info(f"🎯 [TPSL] Large trade detected: {trade_data.size} {symbol} at {trade_data.price}")
                
                # Would trigger TP/SL review for affected positions
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error analyzing trade flow impact: {e}")

    async def _monitoring_loop(self):
        """Main monitoring loop for TP/SL engine"""
        while True:
            try:
                # Monitor active TP/SL orders
                await self._monitor_active_orders()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Log status every minute
                if len(self.active_tpsl_orders) > 0:
                    self.logger.debug(f"🎯 [TPSL] Active orders: {len(self.active_tpsl_orders)}")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ [TPSL] Error in monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _monitor_active_orders(self):
        """Monitor active TP/SL orders for execution"""
        try:
            current_time = time.time()
            
            for order_id, order in list(self.active_tpsl_orders.items()):
                # Check for order expiration (1 hour default)
                if current_time - order.created_at > 3600:
                    self.logger.warning(f"🎯 [TPSL] Order {order_id} expired")
                    order.status = TPSLStatus.CANCELLED
                    self.tpsl_history.append(order)
                    del self.active_tpsl_orders[order_id]
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error monitoring active orders: {e}")

    def _update_performance_metrics(self):
        """Update TP/SL performance metrics"""
        try:
            # Calculate metrics from history
            recent_orders = [order for order in self.tpsl_history if order.filled_at and time.time() - order.filled_at < 86400]
            
            if not recent_orders:
                return
            
            total_orders = len(recent_orders)
            successful_orders = len([o for o in recent_orders if o.status == TPSLStatus.FILLED])
            tp_orders = [o for o in recent_orders if o.side == "TP"]
            sl_orders = [o for o in recent_orders if o.side == "SL"]
            
            tp_hit_rate = len([o for o in tp_orders if o.status == TPSLStatus.FILLED]) / len(tp_orders) if tp_orders else 0
            sl_hit_rate = len([o for o in sl_orders if o.status == TPSLStatus.FILLED]) / len(sl_orders) if sl_orders else 0
            
            filled_orders = [o for o in recent_orders if o.slippage_bps is not None]
            avg_slippage = sum(o.slippage_bps for o in filled_orders) / len(filled_orders) if filled_orders else 0
            
            # Store metrics
            self.performance_metrics['daily'] = TPSLPerformance(
                total_orders=total_orders,
                successful_orders=successful_orders,
                tp_hit_rate=tp_hit_rate,
                sl_hit_rate=sl_hit_rate,
                avg_slippage_bps=avg_slippage,
                avg_execution_time_ms=0.0,  # Would calculate from execution data
                profit_factor=0.0,  # Would calculate from PnL data
                sharpe_ratio=0.0,   # Would calculate from returns
                max_adverse_excursion=0.0  # Would track from position data
            )
            
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error updating performance metrics: {e}")

    def get_tpsl_status(self) -> Dict[str, Any]:
        """Get comprehensive TP/SL system status"""
        try:
            return {
                "active_orders": len(self.active_tpsl_orders),
                "total_orders_today": len(self.tpsl_history),
                "performance_metrics": asdict(self.performance_metrics.get('daily', TPSLPerformance(0, 0, 0, 0, 0, 0, 0, 0, 0))),
                "data_integration": {
                    "orderbooks_available": len(self.latest_orderbooks),
                    "depth_analysis_available": len(self.latest_depth_analysis),
                    "liquidity_metrics_available": len(self.latest_liquidity_metrics)
                },
                "engine_status": "active"
            }
        except Exception as e:
            self.logger.error(f"❌ [TPSL] Error getting status: {e}")
            return {"error": str(e)}
