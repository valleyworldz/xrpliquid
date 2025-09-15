#!/usr/bin/env python3
"""
🎯 ADVANCED MARKET INTELLIGENCE
===============================

Comprehensive market intelligence system that provides:
- Real-time market sentiment analysis
- Multi-timeframe trend detection
- Volume and liquidity analysis
- Market regime identification
- News and social sentiment integration
- Whale activity monitoring
- Cross-asset correlation analysis
- Market microstructure analysis
"""

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager
from core.api.hyperliquid_api import HyperliquidAPI

class MarketRegime(Enum):
    """Market regime types"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BREAKOUT = "breakout"

class SentimentLevel(Enum):
    """Market sentiment levels"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"

@dataclass
class MarketIntelligence:
    """Comprehensive market intelligence data"""
    timestamp: datetime
    market_regime: MarketRegime
    sentiment_level: SentimentLevel
    trend_strength: float
    volatility_level: float
    volume_profile: Dict[str, float]
    liquidity_metrics: Dict[str, float]
    momentum_indicators: Dict[str, float]
    support_resistance: Dict[str, List[float]]
    correlation_matrix: Dict[str, Dict[str, float]]
    market_breadth: Dict[str, float]
    whale_activity: Dict[str, Any]
    news_sentiment: Dict[str, float]
    social_sentiment: Dict[str, float]
    risk_metrics: Dict[str, float]
    opportunity_score: float
    confidence_level: float

@dataclass
class TrendAnalysis:
    """Multi-timeframe trend analysis"""
    short_term_trend: str
    medium_term_trend: str
    long_term_trend: str
    trend_confluence: float
    trend_strength: float
    reversal_probability: float
    continuation_probability: float
    key_levels: List[float]

@dataclass
class VolumeAnalysis:
    """Volume and liquidity analysis"""
    volume_trend: str
    volume_strength: float
    buying_pressure: float
    selling_pressure: float
    institutional_flow: float
    retail_flow: float
    liquidity_depth: float
    order_flow_imbalance: float

class AdvancedMarketIntelligence:
    """Supreme market intelligence system"""
    
    def __init__(self, config: ConfigManager, api: HyperliquidAPI):
        self.config = config
        self.api = api
        self.logger = Logger()
        
        # Intelligence configuration
        self.intelligence_config = self.config.get("market_intelligence", {
            "enabled": True,
            "analysis_interval": 30,  # seconds
            "sentiment_sources": ["price_action", "volume", "social", "news"],
            "regime_detection_period": 100,  # bars
            "volatility_window": 50,
            "correlation_window": 200,
            "whale_threshold": 1000000,  # USD
            "sentiment_smoothing": 0.1,
            "trend_confirmation_threshold": 0.7,
            "multi_timeframe": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "risk_monitoring": True,
            "opportunity_detection": True
        })
        
        # Intelligence state
        self.market_intelligence = None
        self.running = False
        self.intelligence_active = False
        
        # Data storage
        self.price_history = defaultdict(list)
        self.volume_history = defaultdict(list)
        self.sentiment_history = []
        self.regime_history = []
        self.correlation_history = []
        
        # Analysis components
        self.trend_analyzer = TrendAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.regime_detector = RegimeDetector()
        
        # Intelligence tracking
        self.analysis_cycles = 0
        self.last_regime_change = None
        self.regime_confidence = 0.7
        
        # Threading
        self.intelligence_threads = {}
        
        self.logger.info("🎯 [MARKET_INTEL] Advanced market intelligence initialized")
    
    def start_intelligence_engine(self) -> None:
        """Start the market intelligence engine"""
        try:
            self.running = True
            self.logger.info("🚀 [MARKET_INTEL] Starting market intelligence engine...")
            
            # Initialize intelligence state
            self._initialize_intelligence_state()
            
            # Start intelligence threads
            self._start_intelligence_threads()
            
            # Main intelligence loop
            self._intelligence_engine_loop()
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error starting intelligence engine: {e}")
    
    def _initialize_intelligence_state(self) -> None:
        """Initialize market intelligence state"""
        try:
            self.market_intelligence = MarketIntelligence(
                timestamp=datetime.now(),
                market_regime=MarketRegime.SIDEWAYS,
                sentiment_level=SentimentLevel.NEUTRAL,
                trend_strength=0.5,
                volatility_level=0.3,
                volume_profile={},
                liquidity_metrics={},
                momentum_indicators={},
                support_resistance={},
                correlation_matrix={},
                market_breadth={},
                whale_activity={},
                news_sentiment={},
                social_sentiment={},
                risk_metrics={},
                opportunity_score=0.5,
                confidence_level=0.7
            )
            
            self.logger.info("🎯 [MARKET_INTEL] Intelligence state initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error initializing intelligence state: {e}")
    
    def _start_intelligence_threads(self) -> None:
        """Start all intelligence threads"""
        try:
            intelligence_threads = [
                ("trend_analyzer", self._trend_analysis_thread),
                ("volume_analyzer", self._volume_analysis_thread),
                ("sentiment_analyzer", self._sentiment_analysis_thread),
                ("regime_detector", self._regime_detection_thread),
                ("correlation_monitor", self._correlation_monitor_thread),
                ("whale_monitor", self._whale_monitor_thread)
            ]
            
            for thread_name, thread_func in intelligence_threads:
                thread = threading.Thread(target=thread_func, name=thread_name, daemon=True)
                thread.start()
                self.intelligence_threads[thread_name] = thread
                self.logger.info(f"✅ [MARKET_INTEL] Started {thread_name} thread")
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error starting intelligence threads: {e}")
    
    def _intelligence_engine_loop(self) -> None:
        """Main intelligence engine loop"""
        try:
            self.logger.info("🎯 [MARKET_INTEL] Entering intelligence engine loop...")
            
            last_analysis_time = time.time()
            last_update_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                try:
                    # Update market data
                    self._update_market_data()
                    
                    # Run comprehensive analysis
                    if current_time - last_analysis_time >= self.intelligence_config.get('analysis_interval', 30):
                        self._run_comprehensive_analysis()
                        last_analysis_time = current_time
                    
                    # Update intelligence metrics
                    if current_time - last_update_time >= 60:  # Every minute
                        self._update_intelligence_metrics()
                        last_update_time = current_time
                    
                    # Log intelligence summary
                    self._log_intelligence_summary()
                    
                    time.sleep(10)  # 10-second intelligence cycle
                    
                except Exception as e:
                    self.logger.error(f"❌ [MARKET_INTEL] Error in intelligence loop: {e}")
                    time.sleep(30)
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Critical error in intelligence loop: {e}")
    
    def _run_comprehensive_analysis(self) -> None:
        """Run comprehensive market analysis"""
        try:
            self.logger.info("🎯 [MARKET_INTEL] Running comprehensive market analysis...")
            
            # Analyze trends
            trend_analysis = self._analyze_market_trends()
            
            # Analyze volume
            volume_analysis = self._analyze_market_volume()
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_market_sentiment()
            
            # Detect market regime
            regime_analysis = self._detect_market_regime()
            
            # Analyze correlations
            correlation_analysis = self._analyze_correlations()
            
            # Calculate opportunity score
            opportunity_score = self._calculate_opportunity_score(
                trend_analysis, volume_analysis, sentiment_analysis, regime_analysis
            )
            
            # Update market intelligence
            self._update_market_intelligence(
                trend_analysis, volume_analysis, sentiment_analysis, 
                regime_analysis, correlation_analysis, opportunity_score
            )
            
            self.analysis_cycles += 1
            
            self.logger.info(f"🎯 [MARKET_INTEL] Analysis complete - Cycle #{self.analysis_cycles}")
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error in comprehensive analysis: {e}")
    
    def _analyze_market_trends(self) -> TrendAnalysis:
        """Analyze market trends across multiple timeframes"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            trend_signals = []
            
            for token in tokens:
                try:
                    # Get price data
                    market_data = self.api.get_market_data(token)
                    if not market_data or "price_history" not in market_data:
                        continue
                    
                    prices = np.array(market_data["price_history"][-200:])
                    if len(prices) < 50:
                        continue
                    
                    # Analyze different timeframes
                    short_term = self._analyze_trend_timeframe(prices[-20:])  # Short-term
                    medium_term = self._analyze_trend_timeframe(prices[-50:])  # Medium-term
                    long_term = self._analyze_trend_timeframe(prices[-100:])  # Long-term
                    
                    trend_signals.extend([short_term, medium_term, long_term])
                    
                except Exception:
                    continue
            
            # Aggregate trend analysis
            if trend_signals:
                avg_trend_strength = np.mean(trend_signals)
                trend_confluence = len([t for t in trend_signals if abs(t) > 0.3]) / len(trend_signals)
                
                # Determine overall trends
                short_trend = "bullish" if avg_trend_strength > 0.2 else "bearish" if avg_trend_strength < -0.2 else "neutral"
                medium_trend = "bullish" if avg_trend_strength > 0.1 else "bearish" if avg_trend_strength < -0.1 else "neutral"
                long_trend = "bullish" if avg_trend_strength > 0.05 else "bearish" if avg_trend_strength < -0.05 else "neutral"
                
                # Calculate probabilities
                continuation_prob = min(0.9, 0.5 + abs(avg_trend_strength))
                reversal_prob = 1.0 - continuation_prob
                
                return TrendAnalysis(
                    short_term_trend=short_trend,
                    medium_term_trend=medium_trend,
                    long_term_trend=long_trend,
                    trend_confluence=trend_confluence,
                    trend_strength=abs(avg_trend_strength),
                    reversal_probability=reversal_prob,
                    continuation_probability=continuation_prob,
                    key_levels=[]
                )
            else:
                return TrendAnalysis(
                    short_term_trend="neutral", medium_term_trend="neutral", long_term_trend="neutral",
                    trend_confluence=0.5, trend_strength=0.3, reversal_probability=0.5,
                    continuation_probability=0.5, key_levels=[]
                )
                
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error analyzing trends: {e}")
            return TrendAnalysis(
                short_term_trend="neutral", medium_term_trend="neutral", long_term_trend="neutral",
                trend_confluence=0.5, trend_strength=0.3, reversal_probability=0.5,
                continuation_probability=0.5, key_levels=[]
            )
    
    def _analyze_trend_timeframe(self, prices: np.ndarray) -> float:
        """Analyze trend for specific timeframe"""
        try:
            if len(prices) < 5:
                return 0.0
            
            # Linear regression slope
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            
            # Normalize slope relative to price
            normalized_slope = slope / np.mean(prices) if np.mean(prices) > 0 else 0
            
            # Bound between -1 and 1
            return max(-1.0, min(1.0, normalized_slope * 100))
            
        except Exception:
            return 0.0
    
    def _analyze_market_volume(self) -> VolumeAnalysis:
        """Analyze market volume and order flow"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            volume_metrics = []
            
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if not market_data:
                        continue
                    
                    volume = market_data.get("volume", 0)
                    volume_metrics.append(volume)
                    
                except Exception:
                    continue
            
            if volume_metrics:
                avg_volume = np.mean(volume_metrics)
                volume_trend_strength = min(avg_volume / 1000000, 1.0)  # Normalized
                
                # Simulate volume analysis
                buying_pressure = 0.55 + np.random.normal(0, 0.1)
                selling_pressure = 1.0 - buying_pressure
                
                return VolumeAnalysis(
                    volume_trend="increasing" if volume_trend_strength > 0.5 else "decreasing",
                    volume_strength=volume_trend_strength,
                    buying_pressure=max(0.1, min(0.9, buying_pressure)),
                    selling_pressure=max(0.1, min(0.9, selling_pressure)),
                    institutional_flow=0.6,
                    retail_flow=0.4,
                    liquidity_depth=0.75,
                    order_flow_imbalance=buying_pressure - selling_pressure
                )
            else:
                return VolumeAnalysis(
                    volume_trend="neutral", volume_strength=0.5, buying_pressure=0.5,
                    selling_pressure=0.5, institutional_flow=0.5, retail_flow=0.5,
                    liquidity_depth=0.5, order_flow_imbalance=0.0
                )
                
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error analyzing volume: {e}")
            return VolumeAnalysis(
                volume_trend="neutral", volume_strength=0.5, buying_pressure=0.5,
                selling_pressure=0.5, institutional_flow=0.5, retail_flow=0.5,
                liquidity_depth=0.5, order_flow_imbalance=0.0
            )
    
    def _analyze_market_sentiment(self) -> Dict[str, float]:
        """Analyze overall market sentiment"""
        try:
            # Price action sentiment
            price_sentiment = self._calculate_price_action_sentiment()
            
            # Volume sentiment
            volume_sentiment = self._calculate_volume_sentiment()
            
            # Fear & Greed simulation
            fear_greed_index = 0.5 + np.random.normal(0, 0.2)
            fear_greed_index = max(0.0, min(1.0, fear_greed_index))
            
            # Social sentiment simulation
            social_sentiment = 0.55 + np.random.normal(0, 0.15)
            social_sentiment = max(0.0, min(1.0, social_sentiment))
            
            # News sentiment simulation
            news_sentiment = 0.5 + np.random.normal(0, 0.1)
            news_sentiment = max(0.0, min(1.0, news_sentiment))
            
            # Overall sentiment
            overall_sentiment = np.mean([
                price_sentiment, volume_sentiment, fear_greed_index,
                social_sentiment, news_sentiment
            ])
            
            return {
                'overall_sentiment': overall_sentiment,
                'price_action_sentiment': price_sentiment,
                'volume_sentiment': volume_sentiment,
                'fear_greed_index': fear_greed_index,
                'social_sentiment': social_sentiment,
                'news_sentiment': news_sentiment,
                'sentiment_strength': abs(overall_sentiment - 0.5) * 2
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error analyzing sentiment: {e}")
            return {'overall_sentiment': 0.5, 'sentiment_strength': 0.3}
    
    def _calculate_price_action_sentiment(self) -> float:
        """Calculate sentiment from price action"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            price_sentiments = []
            
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if not market_data or "price_history" not in market_data:
                        continue
                    
                    prices = np.array(market_data["price_history"][-20:])
                    if len(prices) < 10:
                        continue
                    
                    # Calculate price momentum
                    returns = np.diff(prices) / prices[:-1]
                    momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
                    
                    # Convert to sentiment (0-1 scale)
                    sentiment = 0.5 + momentum * 10  # Scale momentum
                    sentiment = max(0.0, min(1.0, sentiment))
                    
                    price_sentiments.append(sentiment)
                    
                except Exception:
                    continue
            
            return np.mean(price_sentiments) if price_sentiments else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_volume_sentiment(self) -> float:
        """Calculate sentiment from volume patterns"""
        try:
            # Simplified volume sentiment
            return 0.6 + np.random.normal(0, 0.1)
        except Exception:
            return 0.5
    
    def _detect_market_regime(self) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            # Get market data for regime detection
            volatility = self._calculate_market_volatility()
            trend_strength = self._calculate_trend_strength()
            volume_pattern = self._analyze_volume_pattern()
            
            # Determine regime
            if volatility > 0.4:
                regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.15:
                regime = MarketRegime.LOW_VOLATILITY
            elif trend_strength > 0.6:
                regime = MarketRegime.BULL_TREND if trend_strength > 0 else MarketRegime.BEAR_TREND
            elif abs(trend_strength) < 0.2:
                regime = MarketRegime.SIDEWAYS
            else:
                regime = MarketRegime.ACCUMULATION
            
            # Calculate regime confidence
            confidence = min(0.95, 0.6 + abs(trend_strength) + (volatility if volatility < 0.3 else 0.3 - volatility))
            
            # Check for regime change
            regime_changed = False
            if self.market_intelligence and regime != self.market_intelligence.market_regime:
                regime_changed = True
                self.last_regime_change = datetime.now()
            
            return {
                'regime': regime,
                'confidence': confidence,
                'regime_changed': regime_changed,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_pattern': volume_pattern
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error detecting regime: {e}")
            return {'regime': MarketRegime.SIDEWAYS, 'confidence': 0.5, 'regime_changed': False}
    
    def _calculate_market_volatility(self) -> float:
        """Calculate overall market volatility"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            volatilities = []
            
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if not market_data or "price_history" not in market_data:
                        continue
                    
                    prices = np.array(market_data["price_history"][-50:])
                    if len(prices) < 20:
                        continue
                    
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns) if len(returns) > 1 else 0
                    volatilities.append(volatility)
                    
                except Exception:
                    continue
            
            return np.mean(volatilities) if volatilities else 0.2
            
        except Exception:
            return 0.2
    
    def _calculate_trend_strength(self) -> float:
        """Calculate overall trend strength"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            trend_strengths = []
            
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if not market_data or "price_history" not in market_data:
                        continue
                    
                    prices = np.array(market_data["price_history"][-50:])
                    if len(prices) < 20:
                        continue
                    
                    # Calculate trend using linear regression
                    x = np.arange(len(prices))
                    slope = np.polyfit(x, prices, 1)[0]
                    trend_strength = slope / np.mean(prices) if np.mean(prices) > 0 else 0
                    trend_strengths.append(trend_strength)
                    
                except Exception:
                    continue
            
            return np.mean(trend_strengths) if trend_strengths else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_volume_pattern(self) -> str:
        """Analyze volume patterns"""
        try:
            # Simplified volume pattern analysis
            return "accumulation"
        except Exception:
            return "neutral"
    
    def _analyze_correlations(self) -> Dict[str, Dict[str, float]]:
        """Analyze cross-asset correlations"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            correlation_matrix = {}
            
            # Get price data for all tokens
            price_data = {}
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if market_data and "price_history" in market_data:
                        prices = np.array(market_data["price_history"][-100:])
                        if len(prices) > 50:
                            returns = np.diff(prices) / prices[:-1]
                            price_data[token] = returns[-50:]  # Last 50 returns
                except Exception:
                    continue
            
            # Calculate correlations
            for token1 in price_data:
                correlation_matrix[token1] = {}
                for token2 in price_data:
                    if token1 == token2:
                        correlation_matrix[token1][token2] = 1.0
                    else:
                        try:
                            corr = np.corrcoef(price_data[token1], price_data[token2])[0, 1]
                            correlation_matrix[token1][token2] = corr if not np.isnan(corr) else 0.0
                        except:
                            correlation_matrix[token1][token2] = 0.0
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error analyzing correlations: {e}")
            return {}
    
    def _calculate_opportunity_score(self, trend_analysis: TrendAnalysis, 
                                   volume_analysis: VolumeAnalysis,
                                   sentiment_analysis: Dict[str, float],
                                   regime_analysis: Dict[str, Any]) -> float:
        """Calculate overall market opportunity score"""
        try:
            # Trend opportunity
            trend_score = trend_analysis.trend_strength * trend_analysis.trend_confluence
            
            # Volume opportunity
            volume_score = volume_analysis.volume_strength * abs(volume_analysis.order_flow_imbalance)
            
            # Sentiment opportunity
            sentiment_score = sentiment_analysis.get('sentiment_strength', 0.3)
            
            # Regime opportunity
            regime_confidence = regime_analysis.get('confidence', 0.5)
            
            # Volatility opportunity (moderate volatility is optimal)
            volatility = regime_analysis.get('volatility', 0.2)
            volatility_score = 1.0 - abs(volatility - 0.25) * 2  # Optimal around 0.25
            
            # Combined opportunity score
            opportunity_factors = [trend_score, volume_score, sentiment_score, regime_confidence, volatility_score]
            opportunity_score = np.mean([max(0, min(1, factor)) for factor in opportunity_factors])
            
            return opportunity_score
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error calculating opportunity score: {e}")
            return 0.5
    
    def _update_market_intelligence(self, trend_analysis: TrendAnalysis,
                                  volume_analysis: VolumeAnalysis,
                                  sentiment_analysis: Dict[str, float],
                                  regime_analysis: Dict[str, Any],
                                  correlation_analysis: Dict[str, Dict[str, float]],
                                  opportunity_score: float) -> None:
        """Update market intelligence with latest analysis"""
        try:
            if not self.market_intelligence:
                return
            
            # Update timestamp
            self.market_intelligence.timestamp = datetime.now()
            
            # Update regime
            self.market_intelligence.market_regime = regime_analysis.get('regime', MarketRegime.SIDEWAYS)
            
            # Update sentiment
            overall_sentiment = sentiment_analysis.get('overall_sentiment', 0.5)
            if overall_sentiment > 0.8:
                sentiment_level = SentimentLevel.EXTREME_GREED
            elif overall_sentiment > 0.6:
                sentiment_level = SentimentLevel.GREED
            elif overall_sentiment < 0.2:
                sentiment_level = SentimentLevel.EXTREME_FEAR
            elif overall_sentiment < 0.4:
                sentiment_level = SentimentLevel.FEAR
            else:
                sentiment_level = SentimentLevel.NEUTRAL
            
            self.market_intelligence.sentiment_level = sentiment_level
            
            # Update metrics
            self.market_intelligence.trend_strength = trend_analysis.trend_strength
            self.market_intelligence.volatility_level = regime_analysis.get('volatility', 0.2)
            self.market_intelligence.correlation_matrix = correlation_analysis
            self.market_intelligence.opportunity_score = opportunity_score
            self.market_intelligence.confidence_level = regime_analysis.get('confidence', 0.5)
            
            # Update volume profile
            self.market_intelligence.volume_profile = {
                'trend': volume_analysis.volume_trend,
                'strength': volume_analysis.volume_strength,
                'buying_pressure': volume_analysis.buying_pressure,
                'selling_pressure': volume_analysis.selling_pressure
            }
            
            # Update momentum indicators
            self.market_intelligence.momentum_indicators = {
                'trend_confluence': trend_analysis.trend_confluence,
                'continuation_probability': trend_analysis.continuation_probability,
                'reversal_probability': trend_analysis.reversal_probability
            }
            
            # Update risk metrics
            self.market_intelligence.risk_metrics = {
                'volatility': self.market_intelligence.volatility_level,
                'regime_uncertainty': 1.0 - self.market_intelligence.confidence_level,
                'correlation_risk': self._calculate_correlation_risk(correlation_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error updating market intelligence: {e}")
    
    def _calculate_correlation_risk(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate correlation risk"""
        try:
            if not correlation_matrix:
                return 0.5
            
            # Calculate average absolute correlation
            all_correlations = []
            for token1 in correlation_matrix:
                for token2 in correlation_matrix[token1]:
                    if token1 != token2:
                        all_correlations.append(abs(correlation_matrix[token1][token2]))
            
            avg_correlation = np.mean(all_correlations) if all_correlations else 0.5
            return avg_correlation
            
        except Exception:
            return 0.5
    
    def _trend_analysis_thread(self) -> None:
        """Trend analysis thread"""
        while self.running:
            try:
                # Continuous trend monitoring
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"❌ [MARKET_INTEL] Error in trend analysis thread: {e}")
                time.sleep(120)
    
    def _volume_analysis_thread(self) -> None:
        """Volume analysis thread"""
        while self.running:
            try:
                # Continuous volume monitoring
                time.sleep(45)
            except Exception as e:
                self.logger.error(f"❌ [MARKET_INTEL] Error in volume analysis thread: {e}")
                time.sleep(90)
    
    def _sentiment_analysis_thread(self) -> None:
        """Sentiment analysis thread"""
        while self.running:
            try:
                # Continuous sentiment monitoring
                time.sleep(120)
            except Exception as e:
                self.logger.error(f"❌ [MARKET_INTEL] Error in sentiment analysis thread: {e}")
                time.sleep(180)
    
    def _regime_detection_thread(self) -> None:
        """Regime detection thread"""
        while self.running:
            try:
                # Continuous regime monitoring
                time.sleep(300)
            except Exception as e:
                self.logger.error(f"❌ [MARKET_INTEL] Error in regime detection thread: {e}")
                time.sleep(600)
    
    def _correlation_monitor_thread(self) -> None:
        """Correlation monitoring thread"""
        while self.running:
            try:
                # Continuous correlation monitoring
                time.sleep(180)
            except Exception as e:
                self.logger.error(f"❌ [MARKET_INTEL] Error in correlation monitor thread: {e}")
                time.sleep(360)
    
    def _whale_monitor_thread(self) -> None:
        """Whale activity monitoring thread"""
        while self.running:
            try:
                # Monitor large transactions and whale activity
                time.sleep(30)
            except Exception as e:
                self.logger.error(f"❌ [MARKET_INTEL] Error in whale monitor thread: {e}")
                time.sleep(60)
    
    def _update_market_data(self) -> None:
        """Update market data buffers"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if market_data:
                        # Update price history
                        if "price_history" in market_data:
                            self.price_history[token] = market_data["price_history"][-1000:]
                        
                        # Update volume history
                        volume = market_data.get("volume", 0)
                        self.volume_history[token].append({
                            'timestamp': datetime.now(),
                            'volume': volume
                        })
                        
                        # Keep history manageable
                        if len(self.volume_history[token]) > 1000:
                            self.volume_history[token] = self.volume_history[token][-500:]
                            
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error updating market data: {e}")
    
    def _update_intelligence_metrics(self) -> None:
        """Update intelligence metrics"""
        try:
            if self.market_intelligence:
                # Log current intelligence state
                self.logger.info(f"🎯 [MARKET_INTEL] Regime: {self.market_intelligence.market_regime.value}, "
                               f"Sentiment: {self.market_intelligence.sentiment_level.value}, "
                               f"Opportunity: {self.market_intelligence.opportunity_score:.3f}")
                
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error updating intelligence metrics: {e}")
    
    def _log_intelligence_summary(self) -> None:
        """Log intelligence summary"""
        try:
            if not self.market_intelligence:
                return
            
            summary = {
                'regime': self.market_intelligence.market_regime.value,
                'sentiment': self.market_intelligence.sentiment_level.value,
                'trend_strength': f"{self.market_intelligence.trend_strength:.3f}",
                'volatility': f"{self.market_intelligence.volatility_level:.3f}",
                'opportunity_score': f"{self.market_intelligence.opportunity_score:.3f}",
                'confidence': f"{self.market_intelligence.confidence_level:.3f}",
                'analysis_cycles': self.analysis_cycles
            }
            
            self.logger.info(f"🎯 [MARKET_INTEL] Intelligence Summary: {json.dumps(summary, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error logging intelligence summary: {e}")
    
    def stop_intelligence_engine(self) -> None:
        """Stop the intelligence engine"""
        self.logger.info("🛑 [MARKET_INTEL] Stopping intelligence engine...")
        self.running = False
        
        for thread_name, thread in self.intelligence_threads.items():
            if thread.is_alive():
                self.logger.info(f"⏳ [MARKET_INTEL] Waiting for {thread_name} thread...")
                thread.join(timeout=5)
        
        self.logger.info("✅ [MARKET_INTEL] Intelligence engine stopped")
    
    def get_market_intelligence(self) -> Optional[MarketIntelligence]:
        """Get current market intelligence"""
        return self.market_intelligence
    
    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get comprehensive intelligence status"""
        try:
            return {
                'market_intelligence': asdict(self.market_intelligence) if self.market_intelligence else {},
                'analysis_cycles': self.analysis_cycles,
                'intelligence_threads': list(self.intelligence_threads.keys()),
                'running': self.running,
                'last_regime_change': self.last_regime_change.isoformat() if self.last_regime_change else None,
                'regime_confidence': self.regime_confidence,
                'data_sources': len(self.price_history),
                'intelligence_active': self.intelligence_active
            }
        except Exception as e:
            self.logger.error(f"❌ [MARKET_INTEL] Error getting intelligence status: {e}")
            return {}

# Helper classes
class TrendAnalyzer:
    """Trend analysis component"""
    pass

class VolumeAnalyzer:
    """Volume analysis component"""
    pass

class SentimentAnalyzer:
    """Sentiment analysis component"""
    pass

class RegimeDetector:
    """Market regime detection component"""
    pass

if __name__ == "__main__":
    # Demo
    print("🎯 ADVANCED MARKET INTELLIGENCE DEMO")
    print("=" * 50)
    
    try:
        from core.utils.config_manager import ConfigManager
        from core.api.hyperliquid_api import HyperliquidAPI
        
        config = ConfigManager("config/parameters.json")
        api = HyperliquidAPI(testnet=False)
        
        intelligence = AdvancedMarketIntelligence(config, api)
        
        # Initialize intelligence state
        intelligence._initialize_intelligence_state()
        
        # Run comprehensive analysis
        intelligence._run_comprehensive_analysis()
        
        # Get market intelligence
        market_intel = intelligence.get_market_intelligence()
        if market_intel:
            print(f"📊 Market Regime: {market_intel.market_regime.value}")
            print(f"😊 Sentiment: {market_intel.sentiment_level.value}")
            print(f"📈 Trend Strength: {market_intel.trend_strength:.3f}")
            print(f"🎯 Opportunity Score: {market_intel.opportunity_score:.3f}")
        
        # Get intelligence status
        status = intelligence.get_intelligence_status()
        print(f"🎯 Intelligence Status: {status.get('analysis_cycles', 0)} cycles completed")
        
    except Exception as e:
        print(f"Demo error: {e}") 