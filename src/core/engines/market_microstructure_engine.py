#!/usr/bin/env python3
"""
📊 MARKET MICROSTRUCTURE ANALYSIS ENGINE
=======================================
Advanced order book analysis, volume profiling, and market microstructure
analysis for optimal trade execution and timing.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

# CRITICAL FIX: Add L2 snapshot normalizer for Hyperliquid API changes
def normalise_l2_snapshot(raw):
    """
    Perp: [bids, asks]; Spot: {"levels":[...]}
    Returns normalized {'bids': [...], 'asks': [...]} format
    """
    if isinstance(raw, list) and len(raw) >= 2:
        return {"bids": raw[0], "asks": raw[1]}   # no reversal
    if isinstance(raw, dict) and "levels" in raw:
        lvl = raw["levels"][0]
        return {"bids": [[lvl["bidPx"], lvl.get("bidSz", 0)]],
                "asks": [[lvl["askPx"], lvl.get("askSz", 0)]]}
    # anything else → raise for visibility
    raise TypeError(f"Unrecognised snapshot format: {type(raw)}")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementations
    class KMeans:
        def __init__(self, n_clusters=3, random_state=42):
            self.n_clusters = n_clusters
        def fit_predict(self, X):
            return np.random.randint(0, self.n_clusters, len(X))

@dataclass
class OrderBookLevel:
    """Order book level data"""
    price: float
    size: float
    orders: int
    side: str  # 'bid' or 'ask'

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float
    imbalance: float

@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    price_levels: List[float]
    volumes: List[float]
    poc: float  # Point of Control (highest volume)
    value_area_high: float
    value_area_low: float
    value_area_volume: float

@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics"""
    bid_liquidity: float
    ask_liquidity: float
    total_liquidity: float
    liquidity_imbalance: float
    effective_spread: float
    market_impact: float
    resilience_score: float

@dataclass
class ExecutionSignal:
    """Optimal execution signal"""
    timing_score: float  # 0-1, higher = better timing
    recommended_size: float
    execution_style: str  # 'aggressive', 'passive', 'iceberg'
    urgency: str  # 'low', 'medium', 'high'
    expected_slippage: float
    market_impact_estimate: float
    optimal_chunks: int

class MarketMicrostructureEngine:
    """Advanced market microstructure analysis engine"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        self.max_book_levels = self.config.get('max_book_levels', 20)
        self.volume_profile_bins = self.config.get('volume_profile_bins', 50)
        
        # Data storage
        self.order_book_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=5000)
        self.volume_profiles = {}
        self.liquidity_history = deque(maxlen=500)
        
        # ML components
        self.sklearn_available = SKLEARN_AVAILABLE
        if self.sklearn_available:
            self.impact_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
            self.timing_predictor = RandomForestRegressor(n_estimators=30, random_state=42)
            self.scaler = StandardScaler()
            self.volume_clusterer = KMeans(n_clusters=3, random_state=42)
        
        # Microstructure state
        self.current_book = None
        self.current_liquidity = None
        self.current_volume_profile = None
        
        # Price levels for analysis
        self.support_levels = []
        self.resistance_levels = []
        
        self.logger.info("📊 Market Microstructure Engine initialized")
    
    def analyze_order_book(self, order_book_data: Dict) -> OrderBookSnapshot:
        """Analyze current order book state"""
        try:
            timestamp = datetime.now()
            
            # CRITICAL FIX: Use normalizer to handle all API formats
            try:
                snap = normalise_l2_snapshot(order_book_data)
            except ValueError as e:
                self.logger.warning(f"Order-book format error: {e}")
                return self._empty_book_snapshot(timestamp)
            
            # Parse order book data
            bids = []
            asks = []
            
            # Process bid levels
            for level in snap.get('bids', [])[:self.max_book_levels]:
                if len(level) >= 2:
                    price, size = safe_float(level[0]), safe_float(level[1])
                    orders = int(level[2]) if len(level) > 2 else 1
                    bids.append(OrderBookLevel(price, size, orders, 'bid'))
            
            # Process ask levels
            for level in snap.get('asks', [])[:self.max_book_levels]:
                if len(level) >= 2:
                    price, size = safe_float(level[0]), safe_float(level[1])
                    orders = int(level[2]) if len(level) > 2 else 1
                    asks.append(OrderBookLevel(price, size, orders, 'ask'))
            
            if not bids or not asks:
                return self._empty_book_snapshot(timestamp)
            
            # Calculate metrics
            best_bid = bids[0].price
            best_ask = asks[0].price
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate order book imbalance
            bid_volume = sum(level.size for level in bids[:5])  # Top 5 levels
            ask_volume = sum(level.size for level in asks[:5])
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            snapshot = OrderBookSnapshot(
                timestamp=timestamp,
                bids=bids,
                asks=asks,
                spread=spread,
                mid_price=mid_price,
                imbalance=imbalance
            )
            
            self.current_book = snapshot
            self._update_book_history(snapshot)
            
            self.logger.debug(f"📖 Order book: spread=${spread:.2f}, "
                            f"imbalance={imbalance:.3f}, mid=${mid_price:.2f}")
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"❌ Order book analysis failed: {e}")
            return self._empty_book_snapshot(datetime.now())
    
    def analyze_volume_profile(self, price_data: List[float], volume_data: List[float]) -> VolumeProfile:
        """Analyze volume profile and identify key levels"""
        try:
            if len(price_data) != len(volume_data) or len(price_data) < 10:
                return self._empty_volume_profile()
            
            prices = np.array(price_data)
            volumes = np.array(volume_data)
            
            # Create price bins
            min_price = np.min(prices)
            max_price = np.max(prices)
            price_range = max_price - min_price
            
            if price_range == 0:
                return self._empty_volume_profile()
            
            bin_size = price_range / self.volume_profile_bins
            price_levels = np.arange(min_price, max_price + bin_size, bin_size)
            
            # Aggregate volume by price level
            volume_by_level = np.zeros(len(price_levels) - 1)
            
            for i, (price, volume) in enumerate(zip(prices, volumes)):
                bin_idx = int((price - min_price) / bin_size)
                bin_idx = min(bin_idx, len(volume_by_level) - 1)
                volume_by_level[bin_idx] += volume
            
            # Find Point of Control (POC)
            poc_idx = np.argmax(volume_by_level)
            poc = price_levels[poc_idx] + bin_size / 2
            
            # Calculate Value Area (70% of volume)
            total_volume = np.sum(volume_by_level)
            target_volume = total_volume * 0.7
            
            # Expand from POC to find value area
            value_area_volume = volume_by_level[poc_idx]
            left_idx = right_idx = poc_idx
            
            while value_area_volume < target_volume and (left_idx > 0 or right_idx < len(volume_by_level) - 1):
                left_volume = volume_by_level[left_idx - 1] if left_idx > 0 else 0
                right_volume = volume_by_level[right_idx + 1] if right_idx < len(volume_by_level) - 1 else 0
                
                if left_volume >= right_volume and left_idx > 0:
                    left_idx -= 1
                    value_area_volume += left_volume
                elif right_idx < len(volume_by_level) - 1:
                    right_idx += 1
                    value_area_volume += right_volume
                else:
                    break
            
            value_area_low = price_levels[left_idx]
            value_area_high = price_levels[right_idx + 1]
            
            profile = VolumeProfile(
                price_levels=price_levels[:-1].tolist(),
                volumes=volume_by_level.tolist(),
                poc=poc,
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                value_area_volume=value_area_volume
            )
            
            self.current_volume_profile = profile
            self._identify_key_levels(profile)
            
            self.logger.debug(f"📊 Volume Profile: POC=${poc:.2f}, "
                            f"VA=${value_area_low:.2f}-${value_area_high:.2f}")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"❌ Volume profile analysis failed: {e}")
            return self._empty_volume_profile()
    
    def analyze_liquidity(self, order_book: OrderBookSnapshot) -> LiquidityMetrics:
        """Analyze market liquidity metrics"""
        try:
            if not order_book.bids or not order_book.asks:
                return self._empty_liquidity_metrics()
            
            # Calculate liquidity within spread
            spread_size = order_book.spread
            
            # Bid side liquidity (within 1% of best bid)
            best_bid = order_book.bids[0].price
            bid_threshold = best_bid * 0.99
            bid_liquidity = sum(level.size for level in order_book.bids 
                              if level.price >= bid_threshold)
            
            # Ask side liquidity (within 1% of best ask)
            best_ask = order_book.asks[0].price
            ask_threshold = best_ask * 1.01
            ask_liquidity = sum(level.size for level in order_book.asks 
                              if level.price <= ask_threshold)
            
            total_liquidity = bid_liquidity + ask_liquidity
            liquidity_imbalance = (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0
            
            # Effective spread (weighted by size)
            if len(order_book.bids) > 0 and len(order_book.asks) > 0:
                bid_weighted_price = sum(level.price * level.size for level in order_book.bids[:5])
                bid_total_size = sum(level.size for level in order_book.bids[:5])
                
                ask_weighted_price = sum(level.price * level.size for level in order_book.asks[:5])
                ask_total_size = sum(level.size for level in order_book.asks[:5])
                
                if bid_total_size > 0 and ask_total_size > 0:
                    effective_bid = bid_weighted_price / bid_total_size
                    effective_ask = ask_weighted_price / ask_total_size
                    effective_spread = effective_ask - effective_bid
                else:
                    effective_spread = spread_size
            else:
                effective_spread = spread_size
            
            # Market impact estimation
            market_impact = self._estimate_market_impact(order_book)
            
            # Resilience score (how quickly book recovers)
            resilience_score = self._calculate_resilience_score()
            
            metrics = LiquidityMetrics(
                bid_liquidity=bid_liquidity,
                ask_liquidity=ask_liquidity,
                total_liquidity=total_liquidity,
                liquidity_imbalance=liquidity_imbalance,
                effective_spread=effective_spread,
                market_impact=market_impact,
                resilience_score=resilience_score
            )
            
            self.current_liquidity = metrics
            self._update_liquidity_history(metrics)
            
            self.logger.debug(f"💧 Liquidity: total={total_liquidity:.2f}, "
                            f"imbalance={liquidity_imbalance:.3f}, impact={market_impact:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Liquidity analysis failed: {e}")
            return self._empty_liquidity_metrics()
    
    def get_optimal_execution_signal(self, intended_size: float, side: str, 
                                   urgency: str = 'medium') -> ExecutionSignal:
        """Generate optimal execution strategy"""
        try:
            if not self.current_book or not self.current_liquidity:
                return self._default_execution_signal(intended_size)
            
            # Analyze current market conditions
            timing_score = self._calculate_timing_score()
            
            # Adjust size based on liquidity
            available_liquidity = (self.current_liquidity.bid_liquidity if side == 'sell' 
                                 else self.current_liquidity.ask_liquidity)
            
            # Recommend size (max 20% of available liquidity)
            max_recommended = available_liquidity * 0.2
            recommended_size = min(intended_size, max_recommended)
            
            # Determine execution style
            liquidity_ratio = recommended_size / available_liquidity if available_liquidity > 0 else 1
            spread_ratio = self.current_book.spread / self.current_book.mid_price
            
            if urgency == 'high' or liquidity_ratio < 0.05:
                execution_style = 'aggressive'
                expected_slippage = spread_ratio * 0.8
            elif liquidity_ratio > 0.15 or timing_score < 0.4:
                execution_style = 'iceberg'
                expected_slippage = spread_ratio * 0.3
            else:
                execution_style = 'passive'
                expected_slippage = spread_ratio * 0.1
            
            # Calculate market impact
            market_impact_estimate = self._estimate_execution_impact(recommended_size, side)
            
            # Determine optimal chunking
            if execution_style == 'iceberg':
                chunk_size = available_liquidity * 0.05  # 5% chunks
                optimal_chunks = max(1, int(recommended_size / chunk_size))
            else:
                optimal_chunks = 1
            
            signal = ExecutionSignal(
                timing_score=timing_score,
                recommended_size=recommended_size,
                execution_style=execution_style,
                urgency=urgency,
                expected_slippage=expected_slippage,
                market_impact_estimate=market_impact_estimate,
                optimal_chunks=optimal_chunks
            )
            
            self.logger.info(f"⚡ Execution signal: {execution_style} "
                           f"(size: {recommended_size:.4f}, timing: {timing_score:.3f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Execution signal generation failed: {e}")
            return self._default_execution_signal(intended_size)
    
    def detect_support_resistance(self, price_data: List[float], volume_data: List[float]) -> Dict[str, List[float]]:
        """Detect dynamic support and resistance levels"""
        try:
            if len(price_data) < 50:
                return {'support': [], 'resistance': []}
            
            prices = np.array(price_data)
            volumes = np.array(volume_data) if volume_data else np.ones(len(prices))
            
            # Find local extrema
            support_candidates = []
            resistance_candidates = []
            
            window = 10
            for i in range(window, len(prices) - window):
                # Local minimum (support)
                if prices[i] == np.min(prices[i-window:i+window+1]):
                    weight = volumes[i] if i < len(volumes) else 1
                    support_candidates.append((prices[i], weight))
                
                # Local maximum (resistance)
                if prices[i] == np.max(prices[i-window:i+window+1]):
                    weight = volumes[i] if i < len(volumes) else 1
                    resistance_candidates.append((prices[i], weight))
            
            # Cluster similar levels using ML if available
            if self.sklearn_available and len(support_candidates) > 3:
                support_levels = self._cluster_price_levels([s[0] for s in support_candidates])
                resistance_levels = self._cluster_price_levels([r[0] for r in resistance_candidates])
            else:
                # Simple approach: take most significant levels
                support_candidates.sort(key=lambda x: x[1], reverse=True)
                resistance_candidates.sort(key=lambda x: x[1], reverse=True)
                
                support_levels = [s[0] for s in support_candidates[:5]]
                resistance_levels = [r[0] for r in resistance_candidates[:5]]
            
            # Update stored levels
            self.support_levels = support_levels
            self.resistance_levels = resistance_levels
            
            self.logger.debug(f"🎯 S/R Levels: {len(support_levels)} support, {len(resistance_levels)} resistance")
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            self.logger.error(f"❌ S/R detection failed: {e}")
            return {'support': [], 'resistance': []}
    
    def _cluster_price_levels(self, prices: List[float]) -> List[float]:
        """Cluster similar price levels using ML"""
        if not self.sklearn_available or len(prices) < 3:
            return prices[:5]  # Return top 5
        
        try:
            X = np.array(prices).reshape(-1, 1)
            n_clusters = min(5, len(prices))
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = clusterer.fit_predict(X)
            
            # Get cluster centers
            cluster_centers = []
            for i in range(n_clusters):
                cluster_prices = [prices[j] for j in range(len(prices)) if clusters[j] == i]
                if cluster_prices:
                    cluster_centers.append(np.mean(cluster_prices))
            
            return sorted(cluster_centers)
            
        except Exception as e:
            self.logger.debug(f"Price clustering failed: {e}")
            return prices[:5]
    
    def _calculate_timing_score(self) -> float:
        """Calculate optimal timing score (0-1)"""
        try:
            if not self.current_book or not self.current_liquidity:
                return 0.5
            
            # Factors for good timing
            factors = []
            
            # 1. Spread tightness (tighter = better)
            spread_ratio = self.current_book.spread / self.current_book.mid_price
            spread_score = max(0, 1 - spread_ratio * 1000)  # Normalize
            factors.append(spread_score * 0.3)
            
            # 2. Liquidity availability (more = better)
            liquidity_score = min(1.0, self.current_liquidity.total_liquidity / 1000)
            factors.append(liquidity_score * 0.3)
            
            # 3. Order book balance (balanced = better)
            imbalance_score = 1 - abs(self.current_book.imbalance)
            factors.append(imbalance_score * 0.2)
            
            # 4. Market resilience (higher = better)
            resilience_score = self.current_liquidity.resilience_score
            factors.append(resilience_score * 0.2)
            
            return sum(factors)
            
        except Exception as e:
            self.logger.debug(f"Timing score calculation failed: {e}")
            return 0.5
    
    def _estimate_market_impact(self, order_book: OrderBookSnapshot) -> float:
        """Estimate market impact for typical trade size"""
        try:
            # Simple market impact model
            # Impact = f(spread, liquidity, volatility)
            
            spread_impact = order_book.spread / order_book.mid_price
            
            # Liquidity impact (less liquidity = more impact)
            total_top_liquidity = (sum(level.size for level in order_book.bids[:3]) +
                                 sum(level.size for level in order_book.asks[:3]))
            
            liquidity_impact = 1 / (1 + total_top_liquidity / 100)  # Normalize
            
            # Combine factors
            market_impact = (spread_impact * 0.6 + liquidity_impact * 0.4) * 0.01
            
            return min(0.01, market_impact)  # Cap at 1%
            
        except Exception as e:
            self.logger.debug(f"Market impact estimation failed: {e}")
            return 0.001  # Default 0.1%
    
    def _estimate_execution_impact(self, size: float, side: str) -> float:
        """Estimate impact of specific execution"""
        if not self.current_book or not self.current_liquidity:
            return 0.001
        
        # Get relevant liquidity
        if side == 'buy':
            available_liquidity = self.current_liquidity.ask_liquidity
        else:
            available_liquidity = self.current_liquidity.bid_liquidity
        
        if available_liquidity == 0:
            return 0.01  # High impact if no liquidity
        
        # Impact increases with size ratio
        size_ratio = size / available_liquidity
        base_impact = self.current_liquidity.market_impact
        
        # Non-linear impact function
        execution_impact = base_impact * (1 + size_ratio ** 1.5)
        
        return min(0.05, execution_impact)  # Cap at 5%
    
    def _calculate_resilience_score(self) -> float:
        """Calculate market resilience score"""
        # Simplified resilience calculation
        # In practice, would analyze book recovery after trades
        
        if len(self.order_book_history) < 5:
            return 0.5
        
        # Look at spread stability over time
        recent_spreads = [book.spread for book in list(self.order_book_history)[-5:]]
        spread_stability = 1 - (np.std(recent_spreads) / np.mean(recent_spreads)) if np.mean(recent_spreads) > 0 else 0.5
        
        return max(0, min(1, spread_stability))
    
    def _identify_key_levels(self, volume_profile: VolumeProfile):
        """Identify key support/resistance from volume profile"""
        try:
            # High volume nodes as potential S/R
            volumes = np.array(volume_profile.volumes)
            prices = np.array(volume_profile.price_levels)
            
            # Find volume peaks
            volume_threshold = np.percentile(volumes, 80)  # Top 20% volume
            high_volume_indices = np.where(volumes >= volume_threshold)[0]
            
            key_levels = [prices[i] for i in high_volume_indices]
            
            # Add POC and value area boundaries
            key_levels.extend([
                volume_profile.poc,
                volume_profile.value_area_high,
                volume_profile.value_area_low
            ])
            
            # Update support/resistance based on current price
            current_price = self.current_book.mid_price if self.current_book else 0
            
            if current_price > 0:
                support_levels = [level for level in key_levels if level < current_price]
                resistance_levels = [level for level in key_levels if level > current_price]
                
                self.support_levels = sorted(support_levels, reverse=True)[:5]  # Closest 5
                self.resistance_levels = sorted(resistance_levels)[:5]  # Closest 5
            
        except Exception as e:
            self.logger.debug(f"Key level identification failed: {e}")
    
    def _update_book_history(self, snapshot: OrderBookSnapshot):
        """Update order book history"""
        self.order_book_history.append(snapshot)
    
    def _update_liquidity_history(self, metrics: LiquidityMetrics):
        """Update liquidity history"""
        self.liquidity_history.append({
            'timestamp': datetime.now(),
            'total_liquidity': metrics.total_liquidity,
            'imbalance': metrics.liquidity_imbalance,
            'market_impact': metrics.market_impact
        })
    
    def _empty_book_snapshot(self, timestamp: datetime) -> OrderBookSnapshot:
        """Return empty order book snapshot"""
        return OrderBookSnapshot(
            timestamp=timestamp,
            bids=[],
            asks=[],
            spread=0,
            mid_price=0,
            imbalance=0
        )
    
    def _empty_volume_profile(self) -> VolumeProfile:
        """Return empty volume profile"""
        return VolumeProfile(
            price_levels=[],
            volumes=[],
            poc=0,
            value_area_high=0,
            value_area_low=0,
            value_area_volume=0
        )
    
    def _empty_liquidity_metrics(self) -> LiquidityMetrics:
        """Return empty liquidity metrics"""
        return LiquidityMetrics(
            bid_liquidity=0,
            ask_liquidity=0,
            total_liquidity=0,
            liquidity_imbalance=0,
            effective_spread=0,
            market_impact=0.001,
            resilience_score=0.5
        )
    
    def _default_execution_signal(self, size: float) -> ExecutionSignal:
        """Return default execution signal"""
        return ExecutionSignal(
            timing_score=0.5,
            recommended_size=size,
            execution_style='passive',
            urgency='medium',
            expected_slippage=0.001,
            market_impact_estimate=0.001,
            optimal_chunks=1
        )
    
    def get_microstructure_summary(self) -> Dict[str, Any]:
        """Get comprehensive microstructure summary"""
        summary = {
            'current_book': None,
            'current_liquidity': None,
            'volume_profile': None,
            'key_levels': {
                'support': self.support_levels,
                'resistance': self.resistance_levels
            },
            'market_quality': 'unknown',
            'execution_conditions': 'unknown'
        }
        
        if self.current_book:
            summary['current_book'] = {
                'spread': self.current_book.spread,
                'mid_price': self.current_book.mid_price,
                'imbalance': self.current_book.imbalance,
                'bid_levels': len(self.current_book.bids),
                'ask_levels': len(self.current_book.asks)
            }
        
        if self.current_liquidity:
            summary['current_liquidity'] = {
                'total_liquidity': self.current_liquidity.total_liquidity,
                'imbalance': self.current_liquidity.liquidity_imbalance,
                'effective_spread': self.current_liquidity.effective_spread,
                'market_impact': self.current_liquidity.market_impact,
                'resilience': self.current_liquidity.resilience_score
            }
            
            # Assess market quality
            if (self.current_liquidity.total_liquidity > 1000 and
                self.current_liquidity.effective_spread < 0.001 and
                self.current_liquidity.resilience_score > 0.7):
                summary['market_quality'] = 'excellent'
            elif (self.current_liquidity.total_liquidity > 500 and
                  self.current_liquidity.effective_spread < 0.002):
                summary['market_quality'] = 'good'
            elif self.current_liquidity.total_liquidity > 100:
                summary['market_quality'] = 'fair'
            else:
                summary['market_quality'] = 'poor'
            
            # Assess execution conditions
            timing_score = self._calculate_timing_score()
            if timing_score > 0.8:
                summary['execution_conditions'] = 'excellent'
            elif timing_score > 0.6:
                summary['execution_conditions'] = 'good'
            elif timing_score > 0.4:
                summary['execution_conditions'] = 'fair'
            else:
                summary['execution_conditions'] = 'poor'
        
        if self.current_volume_profile:
            summary['volume_profile'] = {
                'poc': self.current_volume_profile.poc,
                'value_area_high': self.current_volume_profile.value_area_high,
                'value_area_low': self.current_volume_profile.value_area_low,
                'total_levels': len(self.current_volume_profile.price_levels)
            }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize microstructure engine
    engine = MarketMicrostructureEngine()
    
    # Example order book data
    order_book_data = {
        'bids': [
            [45000.0, 1.5, 3],
            [44999.0, 2.1, 5],
            [44998.0, 0.8, 2]
        ],
        'asks': [
            [45001.0, 1.2, 2],
            [45002.0, 1.8, 4],
            [45003.0, 2.5, 6]
        ]
    }
    
    # Example price and volume data
    price_data = [45000 + np.random.normal(0, 50) for _ in range(100)]
    volume_data = [np.random.uniform(0.5, 3.0) for _ in range(100)]
    
    # Analyze microstructure
    book_snapshot = engine.analyze_order_book(order_book_data)
    volume_profile = engine.analyze_volume_profile(price_data, volume_data)
    liquidity_metrics = engine.analyze_liquidity(book_snapshot)
    execution_signal = engine.get_optimal_execution_signal(1.0, 'buy', 'medium')
    
    print(f"Order Book: {book_snapshot}")
    print(f"Liquidity: {liquidity_metrics}")
    print(f"Execution Signal: {execution_signal}")
    print(f"Summary: {engine.get_microstructure_summary()}") 