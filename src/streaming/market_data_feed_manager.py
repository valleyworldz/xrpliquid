#!/usr/bin/env python3
"""
📡 MARKET DATA FEED MANAGER
==========================
Orchestrates multiple high-frequency data streams and provides unified market data access.

Features:
- Multi-exchange data aggregation
- Real-time market depth analysis
- Smart routing and failover
- Data quality monitoring
- Latency optimization
- Cross-exchange arbitrage detection
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

from .high_frequency_data_engine import (
    HighFrequencyDataEngine, MarketTick, OrderBookSnapshot, TradeData,
    DataType, ExchangeType, calculate_spread, calculate_mid_price, calculate_order_book_imbalance
)

@dataclass
class MarketDepthAnalysis:
    """Market depth analysis results"""
    symbol: str
    timestamp: float
    bid_depth_5: Decimal  # Total bid volume in top 5 levels
    ask_depth_5: Decimal  # Total ask volume in top 5 levels
    bid_depth_10: Decimal  # Total bid volume in top 10 levels
    ask_depth_10: Decimal  # Total ask volume in top 10 levels
    imbalance_5: float  # Order book imbalance (top 5 levels)
    imbalance_10: float  # Order book imbalance (top 10 levels)
    spread_bps: float  # Spread in basis points
    mid_price: Decimal
    volatility_1min: float  # 1-minute price volatility
    volume_1min: Decimal  # 1-minute trade volume

@dataclass
class CrossExchangeArbitrage:
    """Cross-exchange arbitrage opportunity"""
    symbol: str
    timestamp: float
    buy_exchange: ExchangeType
    sell_exchange: ExchangeType
    buy_price: Decimal
    sell_price: Decimal
    profit_bps: float  # Profit in basis points
    available_size: Decimal  # Maximum arbitrageable size
    confidence: float  # Confidence score (0-1)

@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a symbol"""
    symbol: str
    timestamp: float
    effective_spread: float  # Effective spread in bps
    price_impact_1k: float  # Price impact for $1K trade (bps)
    price_impact_10k: float  # Price impact for $10K trade (bps)
    resilience_score: float  # How quickly book recovers (0-1)
    depth_score: float  # Overall depth quality (0-1)

class MarketDataFeedManager:
    """
    📡 MARKET DATA FEED MANAGER
    Orchestrates multiple data streams and provides unified market analysis
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Data engines
        self.data_engines: Dict[ExchangeType, HighFrequencyDataEngine] = {}
        
        # Market data state
        self.latest_orderbooks: Dict[str, OrderBookSnapshot] = {}  # key: exchange:symbol
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=600))  # 10 minutes at 1-second resolution
        
        # Analysis results
        self.depth_analysis: Dict[str, MarketDepthAnalysis] = {}
        self.arbitrage_opportunities: List[CrossExchangeArbitrage] = []
        self.liquidity_metrics: Dict[str, LiquidityMetrics] = {}
        
        # Performance tracking
        self.analysis_count = 0
        self.last_arbitrage_scan = 0
        self.alert_callbacks: List[Callable] = []
        
        # Configuration
        self.supported_symbols = config.get('symbols', ['XRP'])
        self.supported_exchanges = [ExchangeType.HYPERLIQUID]  # Add more as needed
        self.analysis_interval = config.get('analysis_interval', 1.0)  # seconds
        self.arbitrage_scan_interval = config.get('arbitrage_scan_interval', 5.0)  # seconds
        
        self.logger.info("📡 [FEED_MANAGER] Market Data Feed Manager initialized")

    async def start_feeds(self):
        """Start all market data feeds"""
        try:
            # Initialize data engines for each exchange
            for exchange in self.supported_exchanges:
                engine_config = self.config.get(f'{exchange.value}_config', {})
                engine = HighFrequencyDataEngine(engine_config, self.logger)
                
                # Register callbacks
                engine.register_callback(DataType.TRADE, self._handle_trade_tick)
                engine.register_callback(DataType.ORDER_BOOK, self._handle_orderbook_tick)
                
                self.data_engines[exchange] = engine
                
                # Start streaming
                await engine.start_streaming(self.supported_symbols, [exchange])
            
            # Start analysis tasks
            asyncio.create_task(self._continuous_market_analysis())
            asyncio.create_task(self._continuous_arbitrage_scanning())
            
            self.logger.info(f"📡 [FEED_MANAGER] Started feeds for {len(self.supported_symbols)} symbols on {len(self.supported_exchanges)} exchanges")
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error starting feeds: {e}")

    def _handle_trade_tick(self, tick: MarketTick):
        """Handle incoming trade tick"""
        try:
            if tick.data_type != DataType.TRADE:
                return
            
            key = f"{tick.exchange.value}:{tick.symbol}"
            trade_data = TradeData(**tick.data)
            
            # Store in trade history
            self.trade_history[key].append(trade_data)
            
            # Update price history (1-second resolution)
            current_second = int(tick.timestamp)
            price_key = f"{key}:price"
            
            if not self.price_history[price_key] or int(self.price_history[price_key][-1]['timestamp']) < current_second:
                self.price_history[price_key].append({
                    'timestamp': current_second,
                    'price': trade_data.price,
                    'volume': trade_data.size
                })
            else:
                # Update current second's volume
                self.price_history[price_key][-1]['volume'] += trade_data.size
                self.price_history[price_key][-1]['price'] = trade_data.price  # Latest price
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error handling trade tick: {e}")

    def _handle_orderbook_tick(self, tick: MarketTick):
        """Handle incoming order book tick"""
        try:
            if tick.data_type != DataType.ORDER_BOOK:
                return
            
            key = f"{tick.exchange.value}:{tick.symbol}"
            orderbook = OrderBookSnapshot(**tick.data)
            
            # Store latest order book
            self.latest_orderbooks[key] = orderbook
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error handling orderbook tick: {e}")

    async def _continuous_market_analysis(self):
        """Continuous market depth and liquidity analysis"""
        while True:
            try:
                await asyncio.sleep(self.analysis_interval)
                
                current_time = time.time()
                
                # Analyze each symbol on each exchange
                for exchange in self.supported_exchanges:
                    for symbol in self.supported_symbols:
                        key = f"{exchange.value}:{symbol}"
                        
                        if key in self.latest_orderbooks:
                            analysis = self._analyze_market_depth(symbol, exchange, current_time)
                            if analysis:
                                self.depth_analysis[key] = analysis
                            
                            liquidity = self._analyze_liquidity(symbol, exchange, current_time)
                            if liquidity:
                                self.liquidity_metrics[key] = liquidity
                
                self.analysis_count += 1
                
                if self.analysis_count % 60 == 0:  # Log every minute
                    self.logger.info(f"📡 [ANALYSIS] Completed {self.analysis_count} market analysis cycles")
                
            except Exception as e:
                self.logger.error(f"❌ [FEED_MANAGER] Error in market analysis: {e}")

    def _analyze_market_depth(self, symbol: str, exchange: ExchangeType, timestamp: float) -> Optional[MarketDepthAnalysis]:
        """Analyze market depth for a symbol"""
        try:
            key = f"{exchange.value}:{symbol}"
            orderbook = self.latest_orderbooks.get(key)
            
            if not orderbook or not orderbook.bids or not orderbook.asks:
                return None
            
            # Calculate depth metrics
            bid_depth_5 = sum(level.size for level in orderbook.bids[:5])
            ask_depth_5 = sum(level.size for level in orderbook.asks[:5])
            bid_depth_10 = sum(level.size for level in orderbook.bids[:10])
            ask_depth_10 = sum(level.size for level in orderbook.asks[:10])
            
            # Calculate imbalances
            imbalance_5 = calculate_order_book_imbalance(orderbook, 5)
            imbalance_10 = calculate_order_book_imbalance(orderbook, 10)
            
            # Calculate spread and mid price
            spread = calculate_spread(orderbook)
            mid_price = calculate_mid_price(orderbook)
            
            if not spread or not mid_price:
                return None
            
            spread_bps = float(spread / mid_price) * 10000  # Convert to basis points
            
            # Calculate 1-minute volatility
            volatility_1min = self._calculate_short_term_volatility(symbol, exchange)
            
            # Calculate 1-minute volume
            volume_1min = self._calculate_recent_volume(symbol, exchange, 60)  # 60 seconds
            
            return MarketDepthAnalysis(
                symbol=symbol,
                timestamp=timestamp,
                bid_depth_5=bid_depth_5,
                ask_depth_5=ask_depth_5,
                bid_depth_10=bid_depth_10,
                ask_depth_10=ask_depth_10,
                imbalance_5=imbalance_5,
                imbalance_10=imbalance_10,
                spread_bps=spread_bps,
                mid_price=mid_price,
                volatility_1min=volatility_1min,
                volume_1min=volume_1min
            )
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error analyzing market depth: {e}")
            return None

    def _analyze_liquidity(self, symbol: str, exchange: ExchangeType, timestamp: float) -> Optional[LiquidityMetrics]:
        """Analyze liquidity metrics for a symbol"""
        try:
            key = f"{exchange.value}:{symbol}"
            orderbook = self.latest_orderbooks.get(key)
            
            if not orderbook or not orderbook.bids or not orderbook.asks:
                return None
            
            mid_price = calculate_mid_price(orderbook)
            if not mid_price:
                return None
            
            # Calculate effective spread (considering depth)
            effective_spread = self._calculate_effective_spread(orderbook)
            
            # Calculate price impact for different trade sizes
            price_impact_1k = self._calculate_price_impact(orderbook, Decimal('1000'), mid_price)
            price_impact_10k = self._calculate_price_impact(orderbook, Decimal('10000'), mid_price)
            
            # Calculate resilience score (how quickly order book recovers)
            resilience_score = self._calculate_resilience_score(symbol, exchange)
            
            # Calculate overall depth score
            depth_score = self._calculate_depth_score(orderbook)
            
            return LiquidityMetrics(
                symbol=symbol,
                timestamp=timestamp,
                effective_spread=effective_spread,
                price_impact_1k=price_impact_1k,
                price_impact_10k=price_impact_10k,
                resilience_score=resilience_score,
                depth_score=depth_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error analyzing liquidity: {e}")
            return None

    async def _continuous_arbitrage_scanning(self):
        """Continuously scan for cross-exchange arbitrage opportunities"""
        while True:
            try:
                await asyncio.sleep(self.arbitrage_scan_interval)
                
                current_time = time.time()
                self.last_arbitrage_scan = current_time
                
                # Clear old opportunities
                self.arbitrage_opportunities.clear()
                
                # Scan for arbitrage between exchanges
                if len(self.supported_exchanges) > 1:
                    for symbol in self.supported_symbols:
                        opportunities = self._scan_symbol_arbitrage(symbol, current_time)
                        self.arbitrage_opportunities.extend(opportunities)
                
                # Alert on significant opportunities
                significant_opportunities = [
                    opp for opp in self.arbitrage_opportunities 
                    if opp.profit_bps > 50 and opp.confidence > 0.7  # 50 bps profit, high confidence
                ]
                
                if significant_opportunities:
                    for opportunity in significant_opportunities:
                        await self._alert_arbitrage_opportunity(opportunity)
                
            except Exception as e:
                self.logger.error(f"❌ [FEED_MANAGER] Error in arbitrage scanning: {e}")

    def _scan_symbol_arbitrage(self, symbol: str, timestamp: float) -> List[CrossExchangeArbitrage]:
        """Scan for arbitrage opportunities for a specific symbol"""
        opportunities = []
        
        try:
            # Get order books from all exchanges
            orderbooks = {}
            for exchange in self.supported_exchanges:
                key = f"{exchange.value}:{symbol}"
                if key in self.latest_orderbooks:
                    orderbooks[exchange] = self.latest_orderbooks[key]
            
            if len(orderbooks) < 2:
                return opportunities
            
            # Compare prices between exchanges
            exchanges = list(orderbooks.keys())
            for i, exchange_a in enumerate(exchanges):
                for exchange_b in exchanges[i+1:]:
                    book_a = orderbooks[exchange_a]
                    book_b = orderbooks[exchange_b]
                    
                    if not book_a.bids or not book_a.asks or not book_b.bids or not book_b.asks:
                        continue
                    
                    # Check A->B arbitrage (buy on A, sell on B)
                    buy_price_a = book_a.asks[0].price  # Best ask on A
                    sell_price_b = book_b.bids[0].price  # Best bid on B
                    
                    if sell_price_b > buy_price_a:
                        profit_bps = float((sell_price_b - buy_price_a) / buy_price_a) * 10000
                        available_size = min(book_a.asks[0].size, book_b.bids[0].size)
                        confidence = self._calculate_arbitrage_confidence(book_a, book_b, exchange_a, exchange_b)
                        
                        opportunities.append(CrossExchangeArbitrage(
                            symbol=symbol,
                            timestamp=timestamp,
                            buy_exchange=exchange_a,
                            sell_exchange=exchange_b,
                            buy_price=buy_price_a,
                            sell_price=sell_price_b,
                            profit_bps=profit_bps,
                            available_size=available_size,
                            confidence=confidence
                        ))
                    
                    # Check B->A arbitrage (buy on B, sell on A)
                    buy_price_b = book_b.asks[0].price  # Best ask on B
                    sell_price_a = book_a.bids[0].price  # Best bid on A
                    
                    if sell_price_a > buy_price_b:
                        profit_bps = float((sell_price_a - buy_price_b) / buy_price_b) * 10000
                        available_size = min(book_b.asks[0].size, book_a.bids[0].size)
                        confidence = self._calculate_arbitrage_confidence(book_b, book_a, exchange_b, exchange_a)
                        
                        opportunities.append(CrossExchangeArbitrage(
                            symbol=symbol,
                            timestamp=timestamp,
                            buy_exchange=exchange_b,
                            sell_exchange=exchange_a,
                            buy_price=buy_price_b,
                            sell_price=sell_price_a,
                            profit_bps=profit_bps,
                            available_size=available_size,
                            confidence=confidence
                        ))
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error scanning arbitrage for {symbol}: {e}")
        
        return opportunities

    def _calculate_short_term_volatility(self, symbol: str, exchange: ExchangeType) -> float:
        """Calculate short-term price volatility"""
        try:
            key = f"{exchange.value}:{symbol}:price"
            prices = list(self.price_history[key])
            
            if len(prices) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                prev_price = float(prices[i-1]['price'])
                curr_price = float(prices[i]['price'])
                if prev_price > 0:
                    returns.append((curr_price - prev_price) / prev_price)
            
            if len(returns) < 2:
                return 0.0
            
            # Calculate standard deviation of returns
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5
            
            # Annualize volatility (assuming 1-second intervals)
            return volatility * (365 * 24 * 3600) ** 0.5
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error calculating volatility: {e}")
            return 0.0

    def _calculate_recent_volume(self, symbol: str, exchange: ExchangeType, seconds: int) -> Decimal:
        """Calculate recent trading volume"""
        try:
            key = f"{exchange.value}:{symbol}"
            trades = list(self.trade_history[key])
            
            cutoff_time = time.time() - seconds
            recent_trades = [trade for trade in trades if trade.timestamp >= cutoff_time]
            
            return sum(trade.size for trade in recent_trades)
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error calculating volume: {e}")
            return Decimal('0')

    def _calculate_effective_spread(self, orderbook: OrderBookSnapshot) -> float:
        """Calculate effective spread considering depth"""
        try:
            if not orderbook.bids or not orderbook.asks:
                return 0.0
            
            # Weight spread by available volume
            total_bid_volume = sum(level.size for level in orderbook.bids[:5])
            total_ask_volume = sum(level.size for level in orderbook.asks[:5])
            
            if total_bid_volume == 0 or total_ask_volume == 0:
                return float(calculate_spread(orderbook) or 0)
            
            # Volume-weighted average prices
            weighted_bid = sum(level.price * level.size for level in orderbook.bids[:5]) / total_bid_volume
            weighted_ask = sum(level.price * level.size for level in orderbook.asks[:5]) / total_ask_volume
            
            mid_price = (weighted_bid + weighted_ask) / 2
            spread = weighted_ask - weighted_bid
            
            return float(spread / mid_price) * 10000  # basis points
            
        except Exception:
            return 0.0

    def _calculate_price_impact(self, orderbook: OrderBookSnapshot, trade_size: Decimal, mid_price: Decimal) -> float:
        """Calculate price impact for a given trade size"""
        try:
            # For buy orders, calculate impact on ask side
            cumulative_size = Decimal('0')
            weighted_price = Decimal('0')
            
            for level in orderbook.asks:
                if cumulative_size >= trade_size:
                    break
                
                remaining_size = trade_size - cumulative_size
                level_size = min(level.size, remaining_size)
                
                weighted_price += level.price * level_size
                cumulative_size += level_size
            
            if cumulative_size == 0:
                return 1000.0  # High impact if no liquidity
            
            avg_execution_price = weighted_price / cumulative_size
            price_impact = (avg_execution_price - mid_price) / mid_price
            
            return float(price_impact) * 10000  # basis points
            
        except Exception:
            return 1000.0  # High impact on error

    def _calculate_resilience_score(self, symbol: str, exchange: ExchangeType) -> float:
        """Calculate order book resilience score"""
        try:
            # Simplified resilience calculation
            # In practice, would measure how quickly order book recovers after large trades
            key = f"{exchange.value}:{symbol}"
            orderbook = self.latest_orderbooks.get(key)
            
            if not orderbook:
                return 0.0
            
            # Use depth as proxy for resilience
            total_depth = sum(level.size for level in orderbook.bids[:10]) + sum(level.size for level in orderbook.asks[:10])
            
            # Normalize to 0-1 scale (assuming good resilience at 1000+ size)
            return min(float(total_depth) / 1000.0, 1.0)
            
        except Exception:
            return 0.0

    def _calculate_depth_score(self, orderbook: OrderBookSnapshot) -> float:
        """Calculate overall depth quality score"""
        try:
            if not orderbook.bids or not orderbook.asks:
                return 0.0
            
            # Consider depth, spread, and level distribution
            bid_depth = sum(level.size for level in orderbook.bids[:10])
            ask_depth = sum(level.size for level in orderbook.asks[:10])
            total_depth = bid_depth + ask_depth
            
            spread = calculate_spread(orderbook)
            mid_price = calculate_mid_price(orderbook)
            
            if not spread or not mid_price:
                return 0.0
            
            spread_score = max(0, 1.0 - float(spread / mid_price) * 1000)  # Penalize wide spreads
            depth_score = min(float(total_depth) / 1000.0, 1.0)  # Normalize depth
            
            return (spread_score + depth_score) / 2.0
            
        except Exception:
            return 0.0

    def _calculate_arbitrage_confidence(self, book_buy: OrderBookSnapshot, book_sell: OrderBookSnapshot, 
                                     exchange_buy: ExchangeType, exchange_sell: ExchangeType) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        try:
            confidence = 1.0
            
            # Check order book depth
            buy_depth = book_buy.asks[0].size if book_buy.asks else Decimal('0')
            sell_depth = book_sell.bids[0].size if book_sell.bids else Decimal('0')
            
            if buy_depth < Decimal('10') or sell_depth < Decimal('10'):
                confidence *= 0.5  # Low depth reduces confidence
            
            # Check timestamp freshness (books should be recent)
            current_time = time.time()
            if current_time - book_buy.timestamp > 5 or current_time - book_sell.timestamp > 5:
                confidence *= 0.7  # Stale data reduces confidence
            
            # Additional checks could include:
            # - Exchange connectivity status
            # - Historical execution success rate
            # - Fee consideration
            
            return confidence
            
        except Exception:
            return 0.0

    async def _alert_arbitrage_opportunity(self, opportunity: CrossExchangeArbitrage):
        """Alert about significant arbitrage opportunity"""
        try:
            message = (
                f"🚨 ARBITRAGE OPPORTUNITY: {opportunity.symbol} "
                f"Buy {opportunity.buy_exchange.value} @ {opportunity.buy_price} "
                f"Sell {opportunity.sell_exchange.value} @ {opportunity.sell_price} "
                f"Profit: {opportunity.profit_bps:.1f} bps "
                f"Size: {opportunity.available_size} "
                f"Confidence: {opportunity.confidence:.2f}"
            )
            
            self.logger.warning(message)
            
            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(opportunity)
                except Exception as e:
                    self.logger.error(f"❌ [FEED_MANAGER] Alert callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error alerting arbitrage: {e}")

    def register_alert_callback(self, callback: Callable):
        """Register callback for arbitrage alerts"""
        self.alert_callbacks.append(callback)

    def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market summary for a symbol"""
        try:
            summary = {
                "symbol": symbol,
                "timestamp": time.time(),
                "exchanges": {}
            }
            
            for exchange in self.supported_exchanges:
                key = f"{exchange.value}:{symbol}"
                
                exchange_data = {
                    "connected": key in self.latest_orderbooks,
                    "orderbook": None,
                    "depth_analysis": None,
                    "liquidity_metrics": None,
                    "recent_trades": 0
                }
                
                if key in self.latest_orderbooks:
                    orderbook = self.latest_orderbooks[key]
                    exchange_data["orderbook"] = {
                        "best_bid": float(orderbook.bids[0].price) if orderbook.bids else None,
                        "best_ask": float(orderbook.asks[0].price) if orderbook.asks else None,
                        "spread": float(calculate_spread(orderbook) or 0),
                        "mid_price": float(calculate_mid_price(orderbook) or 0)
                    }
                
                if key in self.depth_analysis:
                    exchange_data["depth_analysis"] = asdict(self.depth_analysis[key])
                
                if key in self.liquidity_metrics:
                    exchange_data["liquidity_metrics"] = asdict(self.liquidity_metrics[key])
                
                exchange_data["recent_trades"] = len(self.trade_history[key])
                
                summary["exchanges"][exchange.value] = exchange_data
            
            # Add arbitrage opportunities
            symbol_arbitrage = [
                asdict(opp) for opp in self.arbitrage_opportunities 
                if opp.symbol == symbol
            ]
            summary["arbitrage_opportunities"] = symbol_arbitrage
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error getting market summary: {e}")
            return {"error": str(e)}

    async def stop_feeds(self):
        """Stop all market data feeds"""
        try:
            for exchange, engine in self.data_engines.items():
                await engine.stop_streaming()
                self.logger.info(f"📡 [FEED_MANAGER] Stopped {exchange.value} feed")
            
            self.logger.info("📡 [FEED_MANAGER] All feeds stopped")
            
        except Exception as e:
            self.logger.error(f"❌ [FEED_MANAGER] Error stopping feeds: {e}")
