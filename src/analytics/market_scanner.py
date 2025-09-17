#!/usr/bin/env python3
"""
ADVANCED MARKET SCANNER
=======================
Intelligent token selection based on liquidity, volume, funding rates, and market dynamics
"""

from src.core.utils.decimal_boundary_guard import safe_float
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class TokenMetrics:
    """Token market metrics"""
    symbol: str
    volume_24h: float
    open_interest: float
    funding_rate: float
    price: float
    price_change_24h: float
    spread: float
    volatility: float
    market_cap: float
    score: float = 0.0

class AdvancedMarketScanner:
    def __init__(self, api_client):
        self.api = api_client
        self.min_volume = 5_000_000  # $5M minimum volume
        self.min_oi_ratio = 0.05     # 5% OI to market cap ratio
        self.max_spread = 0.0002     # 0.02% max spread
        self.max_funding_rate = 0.001  # 0.1% max funding rate
        
        # Scoring weights
        self.weights = {
            'volume': 0.3,
            'open_interest': 0.25,
            'funding_rate': 0.2,
            'spread': 0.15,
            'volatility': 0.1
        }
    
    def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch comprehensive market data from HyperLiquid"""
        try:
            # Get all market information
            markets = self.api.info_client.all_mids()
            meta = self.api.info_client.meta()
            
            # Get funding rates
            funding_rates = self.api.info_client.funding_history()
            
            # Get open interest
            open_interest = self.api.info_client.open_interest()
            
            return {
                'markets': markets,
                'meta': meta,
                'funding_rates': funding_rates,
                'open_interest': open_interest
            }
        except Exception as e:
            print(f"❌ Error fetching market data: {e}")
            return {}
    
    def calculate_volatility(self, symbol: str, price_history: List[float]) -> float:
        """Calculate realized volatility"""
        if len(price_history) < 2:
            return 0.0
        
        returns = np.diff(np.log(price_history))
        return np.std(returns) * np.sqrt(24)  # Annualized volatility
    
    def calculate_spread(self, orderbook: Dict) -> float:
        """Calculate bid-ask spread"""
        try:
            if 'levels' in orderbook and len(orderbook['levels']) > 0:
                best_bid = orderbook['levels'][0]['px']
                best_ask = orderbook['levels'][-1]['px']
                return (best_ask - best_bid) / best_bid
            return 0.001  # Default spread
        except:
            return 0.001
    
    def get_orderbook_depth(self, symbol: str) -> Dict:
        """Get order book depth for spread calculation"""
        try:
            orderbook = self.api.info_client.l2_snapshot(symbol)
            return orderbook
        except:
            return {'levels': []}
    
    def analyze_token(self, symbol: str, market_data: Dict) -> Optional[TokenMetrics]:
        """Analyze individual token metrics"""
        try:
            markets = market_data.get('markets', {})
            meta = market_data.get('meta', {})
            funding_rates = market_data.get('funding_rates', {})
            open_interest = market_data.get('open_interest', {})
            
            if symbol not in markets:
                return None
            
            current_price = safe_float(markets[symbol])
            
            # Get token metadata
            token_meta = None
            for asset in meta.get('universe', []):
                if asset.get('name') == symbol:
                    token_meta = asset
                    break
            
            if not token_meta:
                return None
            
            # Calculate metrics
            volume_24h = safe_float(token_meta.get('volume24h', 0))
            market_cap = safe_float(token_meta.get('marketCap', 0))
            oi_value = safe_float(open_interest.get(symbol, 0))
            
            # Get funding rate
            funding_rate = 0.0
            if symbol in funding_rates:
                funding_rate = safe_float(funding_rates[symbol].get('fundingRate', 0))
            
            # Get order book for spread calculation
            orderbook = self.get_orderbook_depth(symbol)
            spread = self.calculate_spread(orderbook)
            
            # Calculate volatility (simplified - would need price history in real implementation)
            volatility = 0.02  # Default volatility
            
            # Calculate OI ratio
            oi_ratio = oi_value / market_cap if market_cap > 0 else 0
            
            # Create token metrics
            token_metrics = TokenMetrics(
                symbol=symbol,
                volume_24h=volume_24h,
                open_interest=oi_value,
                funding_rate=funding_rate,
                price=current_price,
                price_change_24h=0.0,  # Would need historical data
                spread=spread,
                volatility=volatility,
                market_cap=market_cap
            )
            
            return token_metrics
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def filter_tokens(self, tokens: List[TokenMetrics]) -> List[TokenMetrics]:
        """Filter tokens based on minimum criteria"""
        filtered = []
        
        for token in tokens:
            if token is None:
                continue
                
            # Volume filter
            if token.volume_24h < self.min_volume:
                continue
            
            # Open interest filter
            oi_ratio = token.open_interest / token.market_cap if token.market_cap > 0 else 0
            if oi_ratio < self.min_oi_ratio:
                continue
            
            # Spread filter
            if token.spread > self.max_spread:
                continue
            
            # Funding rate filter (avoid extreme funding rates)
            if abs(token.funding_rate) > self.max_funding_rate:
                continue
            
            filtered.append(token)
        
        return filtered
    
    def calculate_composite_score(self, token: TokenMetrics, all_tokens: List[TokenMetrics]) -> float:
        """Calculate composite score for token ranking"""
        if not all_tokens:
            return 0.0
        
        # Normalize metrics
        volumes = [t.volume_24h for t in all_tokens if t.volume_24h > 0]
        ois = [t.open_interest for t in all_tokens if t.open_interest > 0]
        spreads = [t.spread for t in all_tokens if t.spread > 0]
        volatilities = [t.volatility for t in all_tokens if t.volatility > 0]
        
        if not volumes or not ois:
            return 0.0
        
        # Normalized scores
        volume_score = token.volume_24h / max(volumes) if max(volumes) > 0 else 0
        oi_score = token.open_interest / max(ois) if max(ois) > 0 else 0
        spread_score = 1 - (token.spread / max(spreads)) if max(spreads) > 0 else 1
        volatility_score = token.volatility / max(volatilities) if max(volatilities) > 0 else 0
        
        # Funding rate score (prefer negative funding rates)
        funding_score = 1 - abs(token.funding_rate) / self.max_funding_rate
        
        # Composite score
        score = (
            self.weights['volume'] * volume_score +
            self.weights['open_interest'] * oi_score +
            self.weights['funding_rate'] * funding_score +
            self.weights['spread'] * spread_score +
            self.weights['volatility'] * volatility_score
        )
        
        return score
    
    def rank_tokens(self, tokens: List[TokenMetrics]) -> List[TokenMetrics]:
        """Rank tokens by composite score"""
        # Calculate scores
        for token in tokens:
            token.score = self.calculate_composite_score(token, tokens)
        
        # Sort by score (highest first)
        ranked = sorted(tokens, key=lambda x: x.score, reverse=True)
        return ranked
    
    def get_top_tokens(self, count: int = 10) -> List[TokenMetrics]:
        """Get top ranked tokens for trading"""
        print("🔍 Scanning markets for best trading opportunities...")
        
        # Fetch market data
        market_data = self.fetch_market_data()
        if not market_data:
            print("❌ Failed to fetch market data")
            return []
        
        # Get all available symbols
        symbols = list(market_data.get('markets', {}).keys())
        print(f"📊 Analyzing {len(symbols)} tokens...")
        
        # Analyze each token
        tokens = []
        for symbol in symbols:
            token_metrics = self.analyze_token(symbol, market_data)
            if token_metrics:
                tokens.append(token_metrics)
        
        # Filter tokens
        filtered_tokens = self.filter_tokens(tokens)
        print(f"✅ Filtered to {len(filtered_tokens)} qualified tokens")
        
        # Rank tokens
        ranked_tokens = self.rank_tokens(filtered_tokens)
        
        # Return top tokens
        top_tokens = ranked_tokens[:count]
        
        # Print results
        print(f"\n🏆 TOP {len(top_tokens)} TRADING OPPORTUNITIES:")
        print("=" * 80)
        for i, token in enumerate(top_tokens, 1):
            print(f"{i:2d}. {token.symbol:8s} | Score: {token.score:.3f} | "
                  f"Vol: ${token.volume_24h/1e6:.1f}M | "
                  f"OI: ${token.open_interest/1e6:.1f}M | "
                  f"Funding: {token.funding_rate*100:.3f}% | "
                  f"Spread: {token.spread*100:.3f}%")
        
        return top_tokens
    
    def get_trading_recommendations(self, count: int = 5) -> Dict[str, Any]:
        """Get trading recommendations with position sizing"""
        top_tokens = self.get_top_tokens(count * 2)  # Get more tokens for analysis
        
        recommendations = []
        for token in top_tokens[:count]:
            # Calculate position size based on volatility and liquidity
            position_size = self.calculate_position_size(token)
            
            recommendation = {
                'symbol': token.symbol,
                'score': token.score,
                'price': token.price,
                'volume_24h': token.volume_24h,
                'open_interest': token.open_interest,
                'funding_rate': token.funding_rate,
                'spread': token.spread,
                'volatility': token.volatility,
                'position_size': position_size,
                'reasoning': self.get_reasoning(token)
            }
            recommendations.append(recommendation)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'total_tokens_analyzed': len(top_tokens)
        }
    
    def calculate_position_size(self, token: TokenMetrics) -> float:
        """Calculate optimal position size based on volatility and liquidity"""
        # Base position size
        base_size = 0.001
        
        # Adjust for volatility (smaller positions for higher volatility)
        vol_adjustment = 1.0 / (1.0 + token.volatility * 10)
        
        # Adjust for liquidity (larger positions for higher volume)
        liquidity_adjustment = min(token.volume_24h / 10_000_000, 2.0)  # Cap at 2x
        
        # Adjust for spread (smaller positions for wider spreads)
        spread_adjustment = 1.0 / (1.0 + token.spread * 1000)
        
        position_size = base_size * vol_adjustment * liquidity_adjustment * spread_adjustment
        
        # Ensure minimum and maximum sizes
        position_size = max(position_size, 0.0001)  # Minimum
        position_size = min(position_size, 0.01)    # Maximum
        
        return position_size
    
    def get_reasoning(self, token: TokenMetrics) -> str:
        """Get reasoning for token selection"""
        reasons = []
        
        if token.volume_24h > 50_000_000:
            reasons.append("High volume")
        if token.open_interest / token.market_cap > 0.1:
            reasons.append("Strong open interest")
        if token.funding_rate < -0.0005:
            reasons.append("Favorable funding rate")
        if token.spread < 0.0001:
            reasons.append("Tight spreads")
        if token.volatility > 0.03:
            reasons.append("Good volatility")
        
        return ", ".join(reasons) if reasons else "Balanced metrics" 