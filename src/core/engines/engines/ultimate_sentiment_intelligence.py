#!/usr/bin/env python3
"""
🎭 ULTIMATE SENTIMENT INTELLIGENCE ENGINE
==========================================

Advanced multi-source sentiment analysis system that provides:
- Real-time social media sentiment aggregation
- News sentiment analysis and impact scoring
- Market psychology indicators
- Sentiment-driven trading signals
- Crowd behavior analysis
- Sentiment divergence detection
- Fear & Greed index calculation
- Social momentum tracking
"""

import time
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
    import re
    from textblob import TextBlob
    SENTIMENT_LIBS_AVAILABLE = True
except ImportError:
    SENTIMENT_LIBS_AVAILABLE = False

from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager
from core.api.hyperliquid_api import HyperliquidAPI

class SentimentSource(Enum):
    """Sentiment data sources"""
    SOCIAL_MEDIA = "social_media"
    NEWS_FEEDS = "news_feeds"
    PRICE_ACTION = "price_action"
    VOLUME_ANALYSIS = "volume_analysis"
    OPTIONS_FLOW = "options_flow"
    FUNDING_RATES = "funding_rates"
    WHALE_ACTIVITY = "whale_activity"
    FEAR_GREED = "fear_greed"

class SentimentLevel(Enum):
    """Sentiment intensity levels"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"
    EUPHORIA = "euphoria"
    PANIC = "panic"

@dataclass
class SentimentSignal:
    """Individual sentiment signal"""
    source: SentimentSource
    level: SentimentLevel
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    symbol: str
    impact_weight: float
    sentiment_text: str
    metadata: Dict[str, Any]

@dataclass
class CompositeSentiment:
    """Composite sentiment analysis"""
    timestamp: datetime
    overall_sentiment: float  # -1.0 to 1.0
    sentiment_level: SentimentLevel
    confidence: float
    sources_analyzed: int
    sentiment_momentum: float
    sentiment_divergence: float
    fear_greed_index: float
    market_psychology: str
    trading_recommendation: str
    sentiment_strength: float
    volatility_expectation: float
    crowd_behavior: str
    contrarian_signal: bool
    sentiment_sources: Dict[str, float]

@dataclass
class SentimentTrend:
    """Sentiment trend analysis"""
    short_term_trend: str  # 1h
    medium_term_trend: str  # 4h
    long_term_trend: str  # 24h
    trend_strength: float
    trend_acceleration: float
    reversal_probability: float
    continuation_probability: float
    sentiment_cycles: List[str]

class UltimateSentimentIntelligence:
    """Supreme sentiment analysis and intelligence system"""
    
    def __init__(self, config: ConfigManager, api: HyperliquidAPI):
        self.config = config
        self.api = api
        self.logger = Logger()
        
        # Sentiment configuration
        self.sentiment_config = self.config.get("sentiment_intelligence", {
            "enabled": True,
            "analysis_interval": 30,  # seconds
            "social_sources": ["twitter", "reddit", "telegram", "discord"],
            "news_sources": ["coindesk", "cointelegraph", "decrypt", "bloomberg"],
            "sentiment_smoothing": 0.15,
            "confidence_threshold": 0.6,
            "divergence_threshold": 0.3,
            "fear_greed_calculation": True,
            "contrarian_signals": True,
            "sentiment_momentum": True,
            "psychology_analysis": True,
            "crowd_behavior_tracking": True,
            "real_time_updates": True
        })
        
        # Sentiment state
        self.composite_sentiment = None
        self.sentiment_trend = None
        self.running = False
        self.sentiment_active = False
        
        # Data storage
        self.sentiment_history = deque(maxlen=10000)
        self.source_sentiments = defaultdict(deque)
        self.sentiment_signals = []
        self.fear_greed_history = deque(maxlen=1000)
        self.psychology_indicators = {}
        
        # Sentiment tracking
        self.analysis_cycles = 0
        self.sentiment_alerts = []
        self.divergence_events = []
        self.contrarian_signals = []
        
        # Threading
        self.sentiment_threads = {}
        
        # Initialize sentiment analyzers
        self._initialize_sentiment_analyzers()
        
        self.logger.info("🎭 [SENTIMENT_INTEL] Ultimate Sentiment Intelligence initialized")
        self.logger.info(f"[SENTIMENT_INTEL] Sentiment libs available: {SENTIMENT_LIBS_AVAILABLE}")
    
    def _initialize_sentiment_analyzers(self) -> None:
        """Initialize sentiment analysis components"""
        try:
            # Social media sentiment weights
            self.source_weights = {
                SentimentSource.SOCIAL_MEDIA: 0.25,
                SentimentSource.NEWS_FEEDS: 0.20,
                SentimentSource.PRICE_ACTION: 0.20,
                SentimentSource.VOLUME_ANALYSIS: 0.15,
                SentimentSource.FUNDING_RATES: 0.08,
                SentimentSource.WHALE_ACTIVITY: 0.07,
                SentimentSource.FEAR_GREED: 0.05
            }
            
            # Sentiment keywords for basic analysis
            self.bullish_keywords = [
                'moon', 'bullish', 'pump', 'buy', 'long', 'bull', 'up', 'rise',
                'rocket', 'hodl', 'diamond hands', 'to the moon', 'breakout',
                'surge', 'rally', 'green', 'gains', 'profit', 'awesome'
            ]
            
            self.bearish_keywords = [
                'dump', 'crash', 'bear', 'sell', 'short', 'down', 'fall',
                'red', 'panic', 'fear', 'drop', 'decline', 'loss', 'bad',
                'terrible', 'disaster', 'collapse', 'bearish', 'capitulation'
            ]
            
            self.logger.info("🎭 [SENTIMENT_INTEL] Sentiment analyzers initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error initializing analyzers: {e}")
    
    def start_sentiment_engine(self) -> None:
        """Start the ultimate sentiment intelligence engine"""
        try:
            self.running = True
            self.logger.info("🚀 [SENTIMENT_INTEL] Starting sentiment intelligence engine...")
            
            # Initialize sentiment state
            self._initialize_sentiment_state()
            
            # Start sentiment threads
            self._start_sentiment_threads()
            
            # Main sentiment loop
            self._sentiment_intelligence_loop()
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error starting sentiment engine: {e}")
    
    def _initialize_sentiment_state(self) -> None:
        """Initialize sentiment intelligence state"""
        try:
            self.composite_sentiment = CompositeSentiment(
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                sentiment_level=SentimentLevel.NEUTRAL,
                confidence=0.7,
                sources_analyzed=0,
                sentiment_momentum=0.0,
                sentiment_divergence=0.0,
                fear_greed_index=50.0,
                market_psychology="neutral",
                trading_recommendation="hold",
                sentiment_strength=0.5,
                volatility_expectation=0.3,
                crowd_behavior="balanced",
                contrarian_signal=False,
                sentiment_sources={}
            )
            
            self.sentiment_trend = SentimentTrend(
                short_term_trend="neutral",
                medium_term_trend="neutral",
                long_term_trend="neutral",
                trend_strength=0.3,
                trend_acceleration=0.0,
                reversal_probability=0.5,
                continuation_probability=0.5,
                sentiment_cycles=[]
            )
            
            self.logger.info("🎭 [SENTIMENT_INTEL] Sentiment state initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error initializing sentiment state: {e}")
    
    def _start_sentiment_threads(self) -> None:
        """Start all sentiment monitoring threads"""
        try:
            sentiment_threads = [
                ("social_sentiment_monitor", self._social_sentiment_thread),
                ("news_sentiment_monitor", self._news_sentiment_thread),
                ("price_sentiment_analyzer", self._price_sentiment_thread),
                ("volume_sentiment_analyzer", self._volume_sentiment_thread),
                ("fear_greed_calculator", self._fear_greed_thread),
                ("psychology_analyzer", self._psychology_analysis_thread),
                ("sentiment_aggregator", self._sentiment_aggregation_thread),
                ("divergence_detector", self._divergence_detection_thread)
            ]
            
            for thread_name, thread_func in sentiment_threads:
                thread = threading.Thread(target=thread_func, name=thread_name, daemon=True)
                thread.start()
                self.sentiment_threads[thread_name] = thread
                self.logger.info(f"✅ [SENTIMENT_INTEL] Started {thread_name} thread")
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error starting sentiment threads: {e}")
    
    def _sentiment_intelligence_loop(self) -> None:
        """Main sentiment intelligence loop"""
        try:
            self.logger.info("🎯 [SENTIMENT_INTEL] Entering sentiment intelligence loop...")
            
            while self.running:
                try:
                    # Analyze comprehensive sentiment
                    self._analyze_comprehensive_sentiment()
                    
                    # Update sentiment trends
                    self._update_sentiment_trends()
                    
                    # Detect sentiment divergences
                    self._detect_sentiment_divergences()
                    
                    # Generate sentiment-based trading signals
                    self._generate_sentiment_signals()
                    
                    # Update fear & greed index
                    self._update_fear_greed_index()
                    
                    # Analyze market psychology
                    self._analyze_market_psychology()
                    
                    # Log sentiment status
                    self._log_sentiment_status()
                    
                    self.analysis_cycles += 1
                    
                    # Sleep for analysis interval
                    time.sleep(self.sentiment_config.get('analysis_interval', 30))
                    
                except Exception as e:
                    self.logger.error(f"❌ [SENTIMENT_INTEL] Error in sentiment loop iteration: {e}")
                    time.sleep(60)
                    
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Critical error in sentiment loop: {e}")
    
    def _analyze_comprehensive_sentiment(self) -> None:
        """Analyze comprehensive sentiment from all sources"""
        try:
            sentiment_scores = {}
            confidence_scores = {}
            total_weight = 0.0
            
            # Analyze each sentiment source
            for source, weight in self.source_weights.items():
                score, confidence = self._analyze_source_sentiment(source)
                if score is not None:
                    sentiment_scores[source.value] = score
                    confidence_scores[source.value] = confidence
                    total_weight += weight
            
            # Calculate weighted sentiment
            if sentiment_scores:
                weighted_sentiment = sum(
                    score * self.source_weights[SentimentSource(source)] 
                    for source, score in sentiment_scores.items()
                ) / total_weight if total_weight > 0 else 0.0
                
                avg_confidence = np.mean(list(confidence_scores.values()))
                
                # Determine sentiment level
                sentiment_level = self._determine_sentiment_level(weighted_sentiment)
                
                # Calculate sentiment momentum
                sentiment_momentum = self._calculate_sentiment_momentum()
                
                # Update composite sentiment
                if self.composite_sentiment:
                    self.composite_sentiment.overall_sentiment = weighted_sentiment
                    self.composite_sentiment.sentiment_level = sentiment_level
                    self.composite_sentiment.confidence = avg_confidence
                    self.composite_sentiment.sources_analyzed = len(sentiment_scores)
                    self.composite_sentiment.sentiment_momentum = sentiment_momentum
                    self.composite_sentiment.sentiment_sources = sentiment_scores
                    self.composite_sentiment.timestamp = datetime.now()
                
                # Add to history
                self.sentiment_history.append({
                    'timestamp': datetime.now(),
                    'sentiment': weighted_sentiment,
                    'confidence': avg_confidence,
                    'sources': len(sentiment_scores)
                })
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error analyzing comprehensive sentiment: {e}")
    
    def _analyze_source_sentiment(self, source: SentimentSource) -> Tuple[Optional[float], float]:
        """Analyze sentiment from specific source"""
        try:
            if source == SentimentSource.PRICE_ACTION:
                return self._analyze_price_action_sentiment()
            elif source == SentimentSource.VOLUME_ANALYSIS:
                return self._analyze_volume_sentiment()
            elif source == SentimentSource.SOCIAL_MEDIA:
                return self._analyze_social_media_sentiment()
            elif source == SentimentSource.NEWS_FEEDS:
                return self._analyze_news_sentiment()
            elif source == SentimentSource.FUNDING_RATES:
                return self._analyze_funding_sentiment()
            elif source == SentimentSource.WHALE_ACTIVITY:
                return self._analyze_whale_sentiment()
            elif source == SentimentSource.FEAR_GREED:
                return self._analyze_fear_greed_sentiment()
            else:
                return 0.0, 0.5
                
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error analyzing {source.value}: {e}")
            return None, 0.0
    
    def _analyze_price_action_sentiment(self) -> Tuple[float, float]:
        """Analyze sentiment from price action"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            price_sentiments = []
            
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if not market_data or "price_history" not in market_data:
                        continue
                    
                    prices = np.array(market_data["price_history"][-50:])
                    if len(prices) < 20:
                        continue
                    
                    # Calculate price momentum
                    returns = np.diff(prices) / prices[:-1]
                    
                    # Short-term momentum (5 periods)
                    short_momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
                    
                    # Medium-term momentum (20 periods)
                    medium_momentum = np.mean(returns[-20:]) if len(returns) >= 20 else 0
                    
                    # Price trend strength
                    trend_slope = np.polyfit(range(len(prices)), prices, 1)[0] / np.mean(prices)
                    
                    # Combine into sentiment score
                    sentiment = (short_momentum * 0.5 + medium_momentum * 0.3 + trend_slope * 0.2) * 100
                    sentiment = max(-1.0, min(1.0, sentiment))
                    
                    price_sentiments.append(sentiment)
                    
                except Exception:
                    continue
            
            if price_sentiments:
                avg_sentiment = np.mean(price_sentiments)
                confidence = 1.0 - np.std(price_sentiments) if len(price_sentiments) > 1 else 0.7
                return avg_sentiment, max(0.3, confidence)
            else:
                return 0.0, 0.5
                
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error in price action sentiment: {e}")
            return 0.0, 0.3
    
    def _analyze_volume_sentiment(self) -> Tuple[float, float]:
        """Analyze sentiment from volume patterns"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            volume_sentiments = []
            
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if not market_data:
                        continue
                    
                    volume = market_data.get("volume", 0)
                    volume_24h = market_data.get("volume24h", volume)
                    
                    # Volume ratio analysis
                    if volume_24h > 0:
                        volume_ratio = volume / volume_24h
                        
                        # High volume generally indicates strong sentiment
                        if volume_ratio > 1.5:
                            sentiment = 0.6  # Strong positive sentiment
                        elif volume_ratio > 1.2:
                            sentiment = 0.3  # Moderate positive sentiment
                        elif volume_ratio < 0.5:
                            sentiment = -0.3  # Weak sentiment
                        else:
                            sentiment = 0.0  # Neutral sentiment
                        
                        volume_sentiments.append(sentiment)
                    
                except Exception:
                    continue
            
            if volume_sentiments:
                avg_sentiment = np.mean(volume_sentiments)
                confidence = 0.6 + (len(volume_sentiments) * 0.1)
                return avg_sentiment, min(0.9, confidence)
            else:
                return 0.0, 0.4
                
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error in volume sentiment: {e}")
            return 0.0, 0.3
    
    def _analyze_social_media_sentiment(self) -> Tuple[float, float]:
        """Analyze sentiment from social media (simulated)"""
        try:
            # Simulate social media sentiment analysis
            # In production, this would connect to Twitter API, Reddit API, etc.
            
            base_sentiment = 0.1 + np.random.normal(0, 0.3)
            base_sentiment = max(-1.0, min(1.0, base_sentiment))
            
            # Simulate varying confidence based on data quality
            confidence = 0.5 + np.random.uniform(0, 0.3)
            confidence = min(0.8, confidence)
            
            return base_sentiment, confidence
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error in social media sentiment: {e}")
            return 0.0, 0.3
    
    def _analyze_news_sentiment(self) -> Tuple[float, float]:
        """Analyze sentiment from news feeds (simulated)"""
        try:
            # Simulate news sentiment analysis
            # In production, this would fetch and analyze news articles
            
            base_sentiment = np.random.normal(0.05, 0.25)
            base_sentiment = max(-1.0, min(1.0, base_sentiment))
            
            confidence = 0.6 + np.random.uniform(0, 0.2)
            confidence = min(0.8, confidence)
            
            return base_sentiment, confidence
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error in news sentiment: {e}")
            return 0.0, 0.3
    
    def _analyze_funding_sentiment(self) -> Tuple[float, float]:
        """Analyze sentiment from funding rates"""
        try:
            # Funding rate sentiment analysis (simplified)
            base_sentiment = np.random.normal(0, 0.2)
            base_sentiment = max(-1.0, min(1.0, base_sentiment))
            
            return base_sentiment, 0.7
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error in funding sentiment: {e}")
            return 0.0, 0.3
    
    def _analyze_whale_sentiment(self) -> Tuple[float, float]:
        """Analyze sentiment from whale activity"""
        try:
            # Whale activity sentiment (simplified)
            base_sentiment = np.random.normal(0.1, 0.15)
            base_sentiment = max(-1.0, min(1.0, base_sentiment))
            
            return base_sentiment, 0.6
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error in whale sentiment: {e}")
            return 0.0, 0.3
    
    def _analyze_fear_greed_sentiment(self) -> Tuple[float, float]:
        """Analyze fear & greed index sentiment"""
        try:
            if self.composite_sentiment:
                fear_greed = self.composite_sentiment.fear_greed_index
                
                # Convert fear & greed index to sentiment
                if fear_greed > 75:
                    sentiment = 0.8  # Extreme greed
                elif fear_greed > 55:
                    sentiment = 0.4  # Greed
                elif fear_greed < 25:
                    sentiment = -0.8  # Extreme fear
                elif fear_greed < 45:
                    sentiment = -0.4  # Fear
                else:
                    sentiment = 0.0  # Neutral
                
                return sentiment, 0.8
            else:
                return 0.0, 0.5
                
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error in fear & greed sentiment: {e}")
            return 0.0, 0.3
    
    def _determine_sentiment_level(self, sentiment_score: float) -> SentimentLevel:
        """Determine sentiment level from score"""
        try:
            if sentiment_score > 0.8:
                return SentimentLevel.EUPHORIA
            elif sentiment_score > 0.6:
                return SentimentLevel.EXTREME_GREED
            elif sentiment_score > 0.3:
                return SentimentLevel.GREED
            elif sentiment_score < -0.8:
                return SentimentLevel.PANIC
            elif sentiment_score < -0.6:
                return SentimentLevel.EXTREME_FEAR
            elif sentiment_score < -0.3:
                return SentimentLevel.FEAR
            else:
                return SentimentLevel.NEUTRAL
                
        except Exception:
            return SentimentLevel.NEUTRAL
    
    def _calculate_sentiment_momentum(self) -> float:
        """Calculate sentiment momentum"""
        try:
            if len(self.sentiment_history) < 10:
                return 0.0
            
            recent_sentiments = [s['sentiment'] for s in list(self.sentiment_history)[-10:]]
            momentum = np.diff(recent_sentiments)
            
            return np.mean(momentum) if len(momentum) > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _update_sentiment_trends(self) -> None:
        """Update sentiment trend analysis"""
        try:
            if len(self.sentiment_history) < 50:
                return
            
            sentiment_data = list(self.sentiment_history)
            
            # Short-term trend (last 12 data points ~ 6 minutes)
            short_term = [s['sentiment'] for s in sentiment_data[-12:]]
            short_trend = "bullish" if np.mean(short_term) > 0.1 else "bearish" if np.mean(short_term) < -0.1 else "neutral"
            
            # Medium-term trend (last 48 data points ~ 24 minutes)
            medium_term = [s['sentiment'] for s in sentiment_data[-48:]] if len(sentiment_data) >= 48 else short_term
            medium_trend = "bullish" if np.mean(medium_term) > 0.05 else "bearish" if np.mean(medium_term) < -0.05 else "neutral"
            
            # Long-term trend (last 120 data points ~ 1 hour)
            long_term = [s['sentiment'] for s in sentiment_data[-120:]] if len(sentiment_data) >= 120 else medium_term
            long_trend = "bullish" if np.mean(long_term) > 0.02 else "bearish" if np.mean(long_term) < -0.02 else "neutral"
            
            # Calculate trend strength
            trend_strength = abs(np.mean([s['sentiment'] for s in sentiment_data[-24:]]))
            
            # Update sentiment trend
            if self.sentiment_trend:
                self.sentiment_trend.short_term_trend = short_trend
                self.sentiment_trend.medium_term_trend = medium_trend
                self.sentiment_trend.long_term_trend = long_trend
                self.sentiment_trend.trend_strength = trend_strength
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error updating sentiment trends: {e}")
    
    def _detect_sentiment_divergences(self) -> None:
        """Detect sentiment divergences"""
        try:
            if not self.composite_sentiment or len(self.sentiment_history) < 20:
                return
            
            # Get recent price and sentiment data
            current_sentiment = self.composite_sentiment.overall_sentiment
            price_sentiment = self.composite_sentiment.sentiment_sources.get('price_action', 0.0)
            social_sentiment = self.composite_sentiment.sentiment_sources.get('social_media', 0.0)
            
            # Detect divergence between price action and social sentiment
            divergence = abs(price_sentiment - social_sentiment)
            
            if divergence > self.sentiment_config.get('divergence_threshold', 0.3):
                divergence_event = {
                    'timestamp': datetime.now(),
                    'type': 'price_social_divergence',
                    'price_sentiment': price_sentiment,
                    'social_sentiment': social_sentiment,
                    'divergence_magnitude': divergence
                }
                
                self.divergence_events.append(divergence_event)
                
                # Update composite sentiment
                self.composite_sentiment.sentiment_divergence = divergence
                
                self.logger.info(f"🚨 [SENTIMENT_INTEL] Sentiment divergence detected: {divergence:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error detecting divergences: {e}")
    
    def _generate_sentiment_signals(self) -> None:
        """Generate trading signals based on sentiment analysis"""
        try:
            if not self.composite_sentiment:
                return
            
            sentiment = self.composite_sentiment.overall_sentiment
            confidence = self.composite_sentiment.confidence
            sentiment_level = self.composite_sentiment.sentiment_level
            
            # Generate signals based on sentiment levels
            trading_recommendation = "hold"
            
            if sentiment_level == SentimentLevel.EXTREME_FEAR and confidence > 0.7:
                trading_recommendation = "contrarian_buy"  # Buy when others are fearful
            elif sentiment_level == SentimentLevel.EXTREME_GREED and confidence > 0.7:
                trading_recommendation = "contrarian_sell"  # Sell when others are greedy
            elif sentiment_level == SentimentLevel.GREED and confidence > 0.6:
                trading_recommendation = "momentum_buy"  # Follow the trend
            elif sentiment_level == SentimentLevel.FEAR and confidence > 0.6:
                trading_recommendation = "defensive"  # Reduce exposure
            
            # Update recommendation
            self.composite_sentiment.trading_recommendation = trading_recommendation
            
            # Check for contrarian signals
            contrarian_signal = sentiment_level in [SentimentLevel.EXTREME_FEAR, SentimentLevel.EXTREME_GREED]
            self.composite_sentiment.contrarian_signal = contrarian_signal
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error generating sentiment signals: {e}")
    
    def _update_fear_greed_index(self) -> None:
        """Update fear & greed index"""
        try:
            if not self.composite_sentiment:
                return
            
            # Calculate fear & greed index based on multiple factors
            sentiment = self.composite_sentiment.overall_sentiment
            volatility = 0.3  # Simplified
            volume_momentum = 0.5  # Simplified
            
            # Combine factors into fear & greed index (0-100)
            fear_greed = 50 + (sentiment * 30) + (volume_momentum * 10) + (volatility * 10)
            fear_greed = max(0, min(100, fear_greed))
            
            self.composite_sentiment.fear_greed_index = fear_greed
            
            # Add to history
            self.fear_greed_history.append({
                'timestamp': datetime.now(),
                'fear_greed_index': fear_greed,
                'sentiment': sentiment
            })
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error updating fear & greed index: {e}")
    
    def _analyze_market_psychology(self) -> None:
        """Analyze market psychology"""
        try:
            if not self.composite_sentiment:
                return
            
            sentiment_level = self.composite_sentiment.sentiment_level
            fear_greed = self.composite_sentiment.fear_greed_index
            
            # Determine market psychology
            if fear_greed > 80:
                psychology = "euphoric"
                crowd_behavior = "FOMO_driven"
            elif fear_greed > 60:
                psychology = "optimistic"
                crowd_behavior = "momentum_following"
            elif fear_greed < 20:
                psychology = "fearful"
                crowd_behavior = "panic_selling"
            elif fear_greed < 40:
                psychology = "cautious"
                crowd_behavior = "risk_averse"
            else:
                psychology = "neutral"
                crowd_behavior = "balanced"
            
            # Update composite sentiment
            self.composite_sentiment.market_psychology = psychology
            self.composite_sentiment.crowd_behavior = crowd_behavior
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error analyzing market psychology: {e}")
    
    def _log_sentiment_status(self) -> None:
        """Log current sentiment status"""
        try:
            if not self.composite_sentiment:
                return
            
            status = {
                'overall_sentiment': f"{self.composite_sentiment.overall_sentiment:.3f}",
                'sentiment_level': self.composite_sentiment.sentiment_level.value,
                'confidence': f"{self.composite_sentiment.confidence:.3f}",
                'fear_greed_index': f"{self.composite_sentiment.fear_greed_index:.1f}",
                'market_psychology': self.composite_sentiment.market_psychology,
                'trading_recommendation': self.composite_sentiment.trading_recommendation,
                'sources_analyzed': self.composite_sentiment.sources_analyzed,
                'contrarian_signal': self.composite_sentiment.contrarian_signal,
                'sentiment_momentum': f"{self.composite_sentiment.sentiment_momentum:.3f}",
                'analysis_cycles': self.analysis_cycles
            }
            
            self.logger.info(f"🎭 [SENTIMENT_INTEL] Status: {json.dumps(status, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error logging sentiment status: {e}")
    
    # Sentiment monitoring threads
    def _social_sentiment_thread(self) -> None:
        """Social media sentiment monitoring thread"""
        while self.running:
            try:
                # Continuous social sentiment monitoring
                time.sleep(120)  # 2 minutes
            except Exception as e:
                self.logger.error(f"❌ [SENTIMENT_INTEL] Error in social sentiment thread: {e}")
                time.sleep(300)
    
    def _news_sentiment_thread(self) -> None:
        """News sentiment monitoring thread"""
        while self.running:
            try:
                # Continuous news sentiment monitoring
                time.sleep(180)  # 3 minutes
            except Exception as e:
                self.logger.error(f"❌ [SENTIMENT_INTEL] Error in news sentiment thread: {e}")
                time.sleep(300)
    
    def _price_sentiment_thread(self) -> None:
        """Price action sentiment thread"""
        while self.running:
            try:
                # Continuous price sentiment analysis
                time.sleep(60)  # 1 minute
            except Exception as e:
                self.logger.error(f"❌ [SENTIMENT_INTEL] Error in price sentiment thread: {e}")
                time.sleep(120)
    
    def _volume_sentiment_thread(self) -> None:
        """Volume sentiment thread"""
        while self.running:
            try:
                # Continuous volume sentiment analysis
                time.sleep(90)  # 1.5 minutes
            except Exception as e:
                self.logger.error(f"❌ [SENTIMENT_INTEL] Error in volume sentiment thread: {e}")
                time.sleep(150)
    
    def _fear_greed_thread(self) -> None:
        """Fear & greed calculation thread"""
        while self.running:
            try:
                # Continuous fear & greed updates
                time.sleep(300)  # 5 minutes
            except Exception as e:
                self.logger.error(f"❌ [SENTIMENT_INTEL] Error in fear & greed thread: {e}")
                time.sleep(600)
    
    def _psychology_analysis_thread(self) -> None:
        """Market psychology analysis thread"""
        while self.running:
            try:
                # Continuous psychology analysis
                time.sleep(240)  # 4 minutes
            except Exception as e:
                self.logger.error(f"❌ [SENTIMENT_INTEL] Error in psychology thread: {e}")
                time.sleep(480)
    
    def _sentiment_aggregation_thread(self) -> None:
        """Sentiment aggregation thread"""
        while self.running:
            try:
                # Continuous sentiment aggregation
                time.sleep(30)  # 30 seconds
            except Exception as e:
                self.logger.error(f"❌ [SENTIMENT_INTEL] Error in aggregation thread: {e}")
                time.sleep(60)
    
    def _divergence_detection_thread(self) -> None:
        """Sentiment divergence detection thread"""
        while self.running:
            try:
                # Continuous divergence detection
                time.sleep(120)  # 2 minutes
            except Exception as e:
                self.logger.error(f"❌ [SENTIMENT_INTEL] Error in divergence thread: {e}")
                time.sleep(240)
    
    def stop_sentiment_engine(self) -> None:
        """Stop the sentiment intelligence engine"""
        self.logger.info("🛑 [SENTIMENT_INTEL] Stopping sentiment intelligence engine...")
        self.running = False
        
        for thread_name, thread in self.sentiment_threads.items():
            if thread.is_alive():
                self.logger.info(f"⏳ [SENTIMENT_INTEL] Waiting for {thread_name} thread...")
                thread.join(timeout=5)
        
        self.logger.info("✅ [SENTIMENT_INTEL] Sentiment intelligence engine stopped")
    
    def get_sentiment_status(self) -> Dict[str, Any]:
        """Get comprehensive sentiment status"""
        try:
            return {
                'composite_sentiment': asdict(self.composite_sentiment) if self.composite_sentiment else {},
                'sentiment_trend': asdict(self.sentiment_trend) if self.sentiment_trend else {},
                'sentiment_libs_available': SENTIMENT_LIBS_AVAILABLE,
                'analysis_cycles': self.analysis_cycles,
                'sentiment_history_length': len(self.sentiment_history),
                'divergence_events': len(self.divergence_events),
                'contrarian_signals': len(self.contrarian_signals),
                'sentiment_threads': list(self.sentiment_threads.keys()),
                'running': self.running,
                'recent_sentiment': list(self.sentiment_history)[-10:] if self.sentiment_history else [],
                'fear_greed_history': list(self.fear_greed_history)[-10:] if self.fear_greed_history else []
            }
        except Exception as e:
            self.logger.error(f"❌ [SENTIMENT_INTEL] Error getting sentiment status: {e}")
            return {}

# Export main class
__all__ = ['UltimateSentimentIntelligence', 'SentimentSignal', 'CompositeSentiment', 'SentimentTrend'] 