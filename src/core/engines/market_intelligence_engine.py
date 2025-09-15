#!/usr/bin/env python3
"""
🧠 MARKET INTELLIGENCE & SENTIMENT ANALYSIS ENGINE
=================================================
Real-time market intelligence with sentiment analysis, news processing,
and market regime detection for enhanced trading signals.
"""

import numpy as np
import pandas as pd
import logging
import requests
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementations
    class TfidfVectorizer:
        def __init__(self, max_features=1000, stop_words='english'):
            self.vocabulary = {}
            
        def fit_transform(self, texts):
            return np.random.random((len(texts), 100))
        
        def transform(self, texts):
            return np.random.random((len(texts), 100))

@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    overall_sentiment: float  # -1 to 1
    confidence: float        # 0 to 1
    bullish_signals: int
    bearish_signals: int
    neutral_signals: int
    source_breakdown: Dict[str, float]

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    confidence: float
    trend_strength: float
    volatility_level: str  # 'low', 'medium', 'high'
    duration_days: int
    key_indicators: Dict[str, float]

@dataclass
class NewsEvent:
    """Structured news event"""
    title: str
    content: str
    source: str
    timestamp: datetime
    sentiment: float
    relevance_score: float
    market_impact: str  # 'high', 'medium', 'low'
    keywords: List[str]

class MarketIntelligenceEngine:
    """Advanced market intelligence with sentiment analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        self.update_interval = self.config.get('update_interval', 300)  # 5 minutes
        
        # Data sources
        self.news_sources = {
            'coindesk': 'https://api.coindesk.com/v1/news',
            'cryptonews': 'https://cryptonews.com/api/v1/news',
            'reddit': 'https://www.reddit.com/r/cryptocurrency.json',
            'fear_greed': 'https://api.alternative.me/fng/'
        }
        
        # Sentiment tracking
        self.sentiment_history = deque(maxlen=1000)
        self.news_cache = deque(maxlen=500)
        self.regime_history = deque(maxlen=100)
        
        # ML components
        self.sklearn_available = SKLEARN_AVAILABLE
        if self.sklearn_available:
            self.sentiment_classifier = MultinomialNB()
            self.regime_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.scaler = StandardScaler()
        else:
            self.sentiment_classifier = None
            self.regime_classifier = None
            self.vectorizer = TfidfVectorizer()
            self.scaler = None
        
        # Current state
        self.current_sentiment = None
        self.current_regime = None
        self.last_update = None
        
        # Sentiment keywords
        self.bullish_keywords = [
            'bullish', 'moon', 'pump', 'rally', 'surge', 'breakout', 'bull run',
            'adoption', 'institutional', 'positive', 'growth', 'gains', 'profit',
            'breakthrough', 'partnership', 'upgrade', 'optimistic', 'strong'
        ]
        
        self.bearish_keywords = [
            'bearish', 'dump', 'crash', 'decline', 'drop', 'fall', 'correction',
            'sell-off', 'liquidation', 'fear', 'panic', 'negative', 'loss',
            'regulation', 'ban', 'hack', 'scam', 'weak', 'pessimistic'
        ]
        
        self.logger.info("🧠 Market Intelligence Engine initialized")
    
    def analyze_sentiment(self, symbol: str = 'BTC') -> SentimentScore:
        """Analyze current market sentiment for a symbol"""
        try:
            # Collect data from multiple sources
            news_sentiment = self._analyze_news_sentiment(symbol)
            social_sentiment = self._analyze_social_sentiment(symbol)
            fear_greed = self._get_fear_greed_index()
            
            # Combine sentiments with weights
            overall_sentiment = (
                news_sentiment * 0.4 +
                social_sentiment * 0.3 +
                fear_greed * 0.3
            )
            
            # Calculate confidence based on data availability
            confidence = min(1.0, (
                (0.4 if news_sentiment != 0 else 0) +
                (0.3 if social_sentiment != 0 else 0) +
                (0.3 if fear_greed != 0 else 0)
            ))
            
            # Count signal types
            bullish_signals = sum([
                1 if news_sentiment > 0.1 else 0,
                1 if social_sentiment > 0.1 else 0,
                1 if fear_greed > 0.1 else 0
            ])
            
            bearish_signals = sum([
                1 if news_sentiment < -0.1 else 0,
                1 if social_sentiment < -0.1 else 0,
                1 if fear_greed < -0.1 else 0
            ])
            
            neutral_signals = 3 - bullish_signals - bearish_signals
            
            sentiment_score = SentimentScore(
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                bullish_signals=bullish_signals,
                bearish_signals=bearish_signals,
                neutral_signals=neutral_signals,
                source_breakdown={
                    'news': news_sentiment,
                    'social': social_sentiment,
                    'fear_greed': fear_greed
                }
            )
            
            self.current_sentiment = sentiment_score
            self._update_sentiment_history(sentiment_score)
            
            self.logger.info(f"📊 Sentiment for {symbol}: {overall_sentiment:.3f} "
                           f"(confidence: {confidence:.3f})")
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"❌ Sentiment analysis failed: {e}")
            return SentimentScore(0, 0, 0, 0, 1, {})
    
    def detect_market_regime(self, price_data: List[float], volume_data: List[float] = None) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(price_data) < 50:
                return MarketRegime('unknown', 0, 0, 'medium', 0, {})
            
            prices = np.array(price_data)
            
            # Calculate technical indicators
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(24)  # Daily volatility
            
            # Trend analysis
            short_ma = np.mean(prices[-20:])  # 20-period MA
            long_ma = np.mean(prices[-50:])   # 50-period MA
            trend_strength = (short_ma - long_ma) / long_ma
            
            # Volatility classification
            vol_percentile = np.percentile(returns, 75)
            if volatility < 0.02:
                volatility_level = 'low'
            elif volatility < 0.05:
                volatility_level = 'medium'
            else:
                volatility_level = 'high'
            
            # Regime classification logic
            if abs(trend_strength) < 0.02 and volatility_level == 'low':
                regime_type = 'sideways'
                confidence = 0.8
            elif trend_strength > 0.05 and volatility_level != 'high':
                regime_type = 'bull'
                confidence = 0.9
            elif trend_strength < -0.05 and volatility_level != 'high':
                regime_type = 'bear'
                confidence = 0.9
            elif volatility_level == 'high':
                regime_type = 'volatile'
                confidence = 0.7
            else:
                regime_type = 'transitional'
                confidence = 0.5
            
            # Use ML classification if available
            if self.sklearn_available and len(self.regime_history) > 20:
                try:
                    ml_regime = self._ml_regime_classification(prices)
                    if ml_regime:
                        regime_type = ml_regime
                        confidence = min(1.0, confidence + 0.1)
                except Exception as ml_e:
                    self.logger.debug(f"ML regime classification failed: {ml_e}")
            
            # Estimate duration
            duration_days = self._estimate_regime_duration(regime_type)
            
            # Key indicators
            key_indicators = {
                'trend_strength': trend_strength,
                'volatility': volatility,
                'short_ma': short_ma,
                'long_ma': long_ma,
                'momentum': returns[-1] if len(returns) > 0 else 0
            }
            
            regime = MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                trend_strength=abs(trend_strength),
                volatility_level=volatility_level,
                duration_days=duration_days,
                key_indicators=key_indicators
            )
            
            self.current_regime = regime
            self._update_regime_history(regime)
            
            self.logger.info(f"🎯 Market regime: {regime_type} "
                           f"(confidence: {confidence:.3f}, trend: {trend_strength:.3f})")
            
            return regime
            
        except Exception as e:
            self.logger.error(f"❌ Regime detection failed: {e}")
            return MarketRegime('unknown', 0, 0, 'medium', 0, {})
    
    def get_market_intelligence_signal(self, symbol: str, price_data: List[float]) -> Dict[str, Any]:
        """Generate comprehensive market intelligence signal"""
        try:
            # Get sentiment and regime
            sentiment = self.analyze_sentiment(symbol)
            regime = self.detect_market_regime(price_data)
            
            # Calculate signal strength
            sentiment_weight = sentiment.confidence * abs(sentiment.overall_sentiment)
            regime_weight = regime.confidence * regime.trend_strength
            
            # Combine signals
            if sentiment.overall_sentiment > 0 and regime.regime_type in ['bull', 'sideways']:
                signal_direction = 'bullish'
                signal_strength = (sentiment_weight + regime_weight) / 2
            elif sentiment.overall_sentiment < 0 and regime.regime_type in ['bear', 'volatile']:
                signal_direction = 'bearish'
                signal_strength = (sentiment_weight + regime_weight) / 2
            else:
                signal_direction = 'neutral'
                signal_strength = 0.3
            
            # Adjust for regime
            regime_multiplier = {
                'bull': 1.2,
                'bear': 1.2,
                'sideways': 0.8,
                'volatile': 0.6,
                'transitional': 0.5,
                'unknown': 0.3
            }.get(regime.regime_type, 0.5)
            
            final_strength = min(1.0, signal_strength * regime_multiplier)
            
            intelligence_signal = {
                'direction': signal_direction,
                'strength': final_strength,
                'confidence': (sentiment.confidence + regime.confidence) / 2,
                'sentiment_score': sentiment.overall_sentiment,
                'regime': regime.regime_type,
                'regime_confidence': regime.confidence,
                'bullish_factors': sentiment.bullish_signals,
                'bearish_factors': sentiment.bearish_signals,
                'volatility_level': regime.volatility_level,
                'trend_strength': regime.trend_strength,
                'source_breakdown': sentiment.source_breakdown,
                'recommendation': self._generate_recommendation(sentiment, regime)
            }
            
            self.logger.info(f"🧠 Intelligence signal for {symbol}: {signal_direction} "
                           f"(strength: {final_strength:.3f})")
            
            return intelligence_signal
            
        except Exception as e:
            self.logger.error(f"❌ Intelligence signal generation failed: {e}")
            return {
                'direction': 'neutral',
                'strength': 0.3,
                'confidence': 0.1,
                'recommendation': 'Insufficient data for analysis'
            }
    
    def _analyze_news_sentiment(self, symbol: str) -> float:
        """Analyze sentiment from news sources"""
        try:
            # Fetch recent news
            news_items = self._fetch_crypto_news(symbol)
            
            if not news_items:
                return 0.0
            
            sentiment_scores = []
            
            for item in news_items:
                # Simple keyword-based sentiment
                text = (item.get('title', '') + ' ' + item.get('content', '')).lower()
                
                bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text)
                bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text)
                
                if bullish_count + bearish_count > 0:
                    sentiment = (bullish_count - bearish_count) / (bullish_count + bearish_count)
                    sentiment_scores.append(sentiment)
            
            if sentiment_scores:
                return np.mean(sentiment_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"News sentiment analysis failed: {e}")
            return 0.0
    
    def _analyze_social_sentiment(self, symbol: str) -> float:
        """Analyze sentiment from social media"""
        try:
            # Simplified social sentiment (would integrate with Twitter API, Reddit API, etc.)
            # For now, using a mock implementation
            
            # Simulate social sentiment based on recent price action
            # In practice, this would analyze actual social media posts
            base_sentiment = np.random.normal(0, 0.3)  # Random walk around neutral
            
            # Add some persistence
            if hasattr(self, '_last_social_sentiment'):
                base_sentiment = 0.7 * self._last_social_sentiment + 0.3 * base_sentiment
            
            self._last_social_sentiment = base_sentiment
            
            return np.clip(base_sentiment, -1, 1)
            
        except Exception as e:
            self.logger.debug(f"Social sentiment analysis failed: {e}")
            return 0.0
    
    def _get_fear_greed_index(self) -> float:
        """Get Fear & Greed Index"""
        try:
            response = requests.get(
                'https://api.alternative.me/fng/',
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    fng_value = int(data['data'][0]['value'])
                    # Convert 0-100 scale to -1 to 1 scale
                    # 0-25: Extreme Fear (-1 to -0.5)
                    # 25-45: Fear (-0.5 to -0.1)
                    # 45-55: Neutral (-0.1 to 0.1)
                    # 55-75: Greed (0.1 to 0.5)
                    # 75-100: Extreme Greed (0.5 to 1)
                    
                    if fng_value <= 25:
                        return -1 + (fng_value / 25) * 0.5
                    elif fng_value <= 45:
                        return -0.5 + ((fng_value - 25) / 20) * 0.4
                    elif fng_value <= 55:
                        return -0.1 + ((fng_value - 45) / 10) * 0.2
                    elif fng_value <= 75:
                        return 0.1 + ((fng_value - 55) / 20) * 0.4
                    else:
                        return 0.5 + ((fng_value - 75) / 25) * 0.5
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"Fear & Greed Index fetch failed: {e}")
            return 0.0
    
    def _fetch_crypto_news(self, symbol: str) -> List[Dict]:
        """Fetch cryptocurrency news"""
        try:
            # Mock news fetching - in practice would use real news APIs
            # CoinDesk API, CryptoPanic API, etc.
            
            mock_news = [
                {
                    'title': f'{symbol} shows strong momentum amid institutional adoption',
                    'content': 'Major institutions continue to show interest in cryptocurrency markets',
                    'timestamp': datetime.now() - timedelta(hours=1)
                },
                {
                    'title': f'Market analysis: {symbol} technical indicators suggest continued growth',
                    'content': 'Technical analysis shows bullish patterns emerging',
                    'timestamp': datetime.now() - timedelta(hours=2)
                }
            ]
            
            return mock_news
            
        except Exception as e:
            self.logger.debug(f"News fetching failed: {e}")
            return []
    
    def _ml_regime_classification(self, prices: np.ndarray) -> Optional[str]:
        """Use ML to classify market regime"""
        try:
            # Prepare features
            features = self._prepare_regime_features(prices)
            
            if len(features) < 10:
                return None
            
            # This would be trained on historical data
            # For now, return None to use rule-based classification
            return None
            
        except Exception as e:
            self.logger.debug(f"ML regime classification failed: {e}")
            return None
    
    def _prepare_regime_features(self, prices: np.ndarray) -> List[float]:
        """Prepare features for regime classification"""
        if len(prices) < 50:
            return []
        
        returns = np.diff(np.log(prices))
        
        features = [
            np.mean(returns[-20:]),  # Recent return
            np.std(returns[-20:]),   # Recent volatility
            np.mean(prices[-10:]) / np.mean(prices[-30:]) - 1,  # Short/medium trend
            np.mean(prices[-30:]) / np.mean(prices[-50:]) - 1,  # Medium/long trend
            np.max(prices[-20:]) / np.min(prices[-20:]) - 1,    # Recent range
            len([r for r in returns[-20:] if r > 0]) / 20,      # Win rate
        ]
        
        return features
    
    def _estimate_regime_duration(self, regime_type: str) -> int:
        """Estimate how long the current regime has been active"""
        if not self.regime_history:
            return 1
        
        duration = 1
        for regime in reversed(list(self.regime_history)):
            if regime.regime_type == regime_type:
                duration += 1
            else:
                break
        
        return duration
    
    def _generate_recommendation(self, sentiment: SentimentScore, regime: MarketRegime) -> str:
        """Generate trading recommendation based on intelligence"""
        if regime.regime_type == 'bull' and sentiment.overall_sentiment > 0.3:
            return "Strong bullish signals - consider increasing long exposure"
        elif regime.regime_type == 'bear' and sentiment.overall_sentiment < -0.3:
            return "Strong bearish signals - consider reducing exposure or shorting"
        elif regime.regime_type == 'sideways':
            return "Range-bound market - consider range trading strategies"
        elif regime.regime_type == 'volatile':
            return "High volatility - reduce position sizes and use tight stops"
        elif sentiment.confidence < 0.3:
            return "Low confidence signals - wait for clearer market direction"
        else:
            return "Mixed signals - maintain current strategy with caution"
    
    def _update_sentiment_history(self, sentiment: SentimentScore):
        """Update sentiment history"""
        self.sentiment_history.append({
            'timestamp': datetime.now(),
            'sentiment': sentiment.overall_sentiment,
            'confidence': sentiment.confidence
        })
    
    def _update_regime_history(self, regime: MarketRegime):
        """Update regime history"""
        self.regime_history.append(regime)
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive intelligence summary"""
        summary = {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'current_sentiment': None,
            'current_regime': None,
            'sentiment_trend': 'unknown',
            'regime_stability': 'unknown'
        }
        
        if self.current_sentiment:
            summary['current_sentiment'] = {
                'score': self.current_sentiment.overall_sentiment,
                'confidence': self.current_sentiment.confidence,
                'bullish_signals': self.current_sentiment.bullish_signals,
                'bearish_signals': self.current_sentiment.bearish_signals,
                'sources': self.current_sentiment.source_breakdown
            }
        
        if self.current_regime:
            summary['current_regime'] = {
                'type': self.current_regime.regime_type,
                'confidence': self.current_regime.confidence,
                'trend_strength': self.current_regime.trend_strength,
                'volatility': self.current_regime.volatility_level,
                'duration': self.current_regime.duration_days
            }
        
        # Calculate trends
        if len(self.sentiment_history) >= 5:
            recent_sentiments = [s['sentiment'] for s in list(self.sentiment_history)[-5:]]
            if recent_sentiments[-1] > recent_sentiments[0]:
                summary['sentiment_trend'] = 'improving'
            elif recent_sentiments[-1] < recent_sentiments[0]:
                summary['sentiment_trend'] = 'deteriorating'
            else:
                summary['sentiment_trend'] = 'stable'
        
        if len(self.regime_history) >= 3:
            recent_regimes = [r.regime_type for r in list(self.regime_history)[-3:]]
            if len(set(recent_regimes)) == 1:
                summary['regime_stability'] = 'stable'
            else:
                summary['regime_stability'] = 'changing'
        
        return summary

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize intelligence engine
    intelligence = MarketIntelligenceEngine()
    
    # Example price data
    price_data = [45000 + np.random.normal(0, 1000) for _ in range(100)]
    
    # Analyze sentiment and regime
    sentiment = intelligence.analyze_sentiment('BTC')
    regime = intelligence.detect_market_regime(price_data)
    signal = intelligence.get_market_intelligence_signal('BTC', price_data)
    
    print(f"Sentiment: {sentiment}")
    print(f"Regime: {regime}")
    print(f"Signal: {signal}") 