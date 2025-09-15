"""
📊 CHIEF QUANTITATIVE STRATEGIST
"The data doesn't lie. My models find the edge before the market sees it."

This module implements advanced quantitative trading strategies:
- Mean reversion with statistical edge
- Momentum strategies with regime detection
- Funding rate arbitrage models
- Volatility forecasting
- Multi-factor models
- Kelly criterion position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import logging
from collections import deque
import asyncio
import time

@dataclass
class SignalStrength:
    """Represents signal strength and confidence"""
    signal: str  # 'LONG', 'SHORT', 'HOLD'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    expected_return: float
    risk_score: float
    sharpe_ratio: float
    max_drawdown: float

@dataclass
class RegimeAnalysis:
    """Market regime analysis results"""
    regime: str  # 'TRENDING', 'MEAN_REVERTING', 'VOLATILE', 'QUIET'
    confidence: float
    persistence: float
    expected_duration: int  # minutes
    volatility_forecast: float
    momentum_score: float

@dataclass
class FactorExposure:
    """Factor exposure analysis"""
    momentum: float
    mean_reversion: float
    volatility: float
    funding_rate: float
    volume: float
    correlation: float

class QuantitativeStrategist:
    """
    Chief Quantitative Strategist - Master of Mathematical Models
    
    This class implements sophisticated quantitative strategies:
    1. Mean reversion with statistical edge
    2. Momentum strategies with regime detection
    3. Funding rate arbitrage models
    4. Volatility forecasting
    5. Multi-factor models
    6. Kelly criterion position sizing
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Strategy parameters
        self.lookback_periods = {
            'short': 20,
            'medium': 50,
            'long': 200
        }
        
        # Mean reversion parameters
        self.mean_reversion_config = {
            'z_score_threshold': 2.0,
            'lookback_period': 50,
            'min_volatility': 0.001,
            'max_volatility': 0.05,
            'reversion_speed': 0.1
        }
        
        # Momentum parameters
        self.momentum_config = {
            'short_period': 10,
            'long_period': 30,
            'threshold': 0.02,
            'regime_threshold': 0.6
        }
        
        # Funding rate parameters
        self.funding_config = {
            'min_rate': 0.0001,
            'max_rate': 0.01,
            'arbitrage_threshold': 0.002,
            'hold_time': 3600  # 1 hour
        }
        
        # Volatility parameters
        self.volatility_config = {
            'garch_period': 30,
            'forecast_horizon': 24,  # hours
            'volatility_threshold': 0.02
        }
        
        # Data storage
        self.price_history: deque = deque(maxlen=1000)
        self.volume_history: deque = deque(maxlen=1000)
        self.funding_history: deque = deque(maxlen=100)
        self.volatility_history: deque = deque(maxlen=100)
        
        # Model state
        self.current_regime: Optional[RegimeAnalysis] = None
        self.factor_exposures: Dict[str, FactorExposure] = {}
        self.signal_history: List[SignalStrength] = []
        
        # Performance tracking
        self.strategy_metrics = {
            'total_signals': 0,
            'correct_signals': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize quantitative models"""
        try:
            self.logger.info("Initializing quantitative models...")
            
            # Initialize GARCH model for volatility forecasting
            self.garch_model = None  # Will be initialized when enough data is available
            
            # Initialize regime detection model
            self.regime_model = None  # Will be initialized when enough data is available
            
            # Initialize factor models
            self.factor_models = {
                'momentum': None,
                'mean_reversion': None,
                'volatility': None,
                'funding_rate': None
            }
            
            self.logger.info("Quantitative models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing quantitative models: {e}")
    
    def update_market_data(self, price: float, volume: float, funding_rate: float = 0.0):
        """Update market data for analysis"""
        try:
            self.price_history.append(price)
            self.volume_history.append(volume)
            self.funding_history.append(funding_rate)
            
            # Calculate and store volatility
            if len(self.price_history) > 1:
                returns = np.diff(list(self.price_history)[-20:])
                volatility = np.std(returns) if len(returns) > 1 else 0.0
                self.volatility_history.append(volatility)
            
            # Update models when enough data is available
            if len(self.price_history) >= self.lookback_periods['medium']:
                self._update_models()
                
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _update_models(self):
        """Update quantitative models with new data"""
        try:
            # Update regime detection
            self._update_regime_detection()
            
            # Update volatility forecasting
            self._update_volatility_forecasting()
            
            # Update factor models
            self._update_factor_models()
            
        except Exception as e:
            self.logger.error(f"Error updating models: {e}")
    
    def _update_regime_detection(self):
        """Update market regime detection"""
        try:
            if len(self.price_history) < self.lookback_periods['medium']:
                return
            
            prices = np.array(list(self.price_history))
            volumes = np.array(list(self.volume_history))
            
            # Calculate momentum score
            short_ma = np.mean(prices[-self.momentum_config['short_period']:])
            long_ma = np.mean(prices[-self.momentum_config['long_period']:])
            momentum_score = (short_ma - long_ma) / long_ma
            
            # Calculate volatility
            returns = np.diff(prices[-20:])
            volatility = np.std(returns) if len(returns) > 1 else 0.0
            
            # Calculate mean reversion score
            mean_reversion_score = self._calculate_mean_reversion_score(prices)
            
            # Determine regime
            if abs(momentum_score) > self.momentum_config['threshold']:
                if momentum_score > 0:
                    regime = 'TRENDING_UP'
                else:
                    regime = 'TRENDING_DOWN'
                confidence = min(abs(momentum_score) / self.momentum_config['threshold'], 1.0)
            elif abs(mean_reversion_score) > 0.5:
                regime = 'MEAN_REVERTING'
                confidence = min(abs(mean_reversion_score), 1.0)
            elif volatility > self.volatility_config['volatility_threshold']:
                regime = 'VOLATILE'
                confidence = min(volatility / self.volatility_config['volatility_threshold'], 1.0)
            else:
                regime = 'QUIET'
                confidence = 0.5
            
            # Calculate persistence (how long regime is likely to last)
            persistence = self._calculate_regime_persistence(regime, prices)
            
            # Estimate expected duration
            expected_duration = self._estimate_regime_duration(regime, volatility)
            
            self.current_regime = RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                persistence=persistence,
                expected_duration=expected_duration,
                volatility_forecast=volatility,
                momentum_score=momentum_score
            )
            
        except Exception as e:
            self.logger.error(f"Error updating regime detection: {e}")
    
    def _calculate_mean_reversion_score(self, prices: np.ndarray) -> float:
        """Calculate mean reversion score using Z-score"""
        try:
            if len(prices) < self.mean_reversion_config['lookback_period']:
                return 0.0
            
            recent_prices = prices[-self.mean_reversion_config['lookback_period']:]
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            if std_price == 0:
                return 0.0
            
            current_price = prices[-1]
            z_score = (current_price - mean_price) / std_price
            
            return z_score
            
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion score: {e}")
            return 0.0
    
    def _calculate_regime_persistence(self, regime: str, prices: np.ndarray) -> float:
        """Calculate how persistent the current regime is"""
        try:
            # Simple persistence calculation based on recent price action
            if len(prices) < 10:
                return 0.5
            
            recent_returns = np.diff(prices[-10:])
            
            if regime in ['TRENDING_UP', 'TRENDING_DOWN']:
                # Check consistency of direction
                positive_returns = np.sum(recent_returns > 0)
                consistency = positive_returns / len(recent_returns)
                if regime == 'TRENDING_UP':
                    return consistency
                else:
                    return 1.0 - consistency
            else:
                # For mean reverting and volatile regimes
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating regime persistence: {e}")
            return 0.5
    
    def _estimate_regime_duration(self, regime: str, volatility: float) -> int:
        """Estimate expected duration of current regime in minutes"""
        try:
            base_duration = {
                'TRENDING_UP': 120,  # 2 hours
                'TRENDING_DOWN': 120,  # 2 hours
                'MEAN_REVERTING': 60,  # 1 hour
                'VOLATILE': 30,  # 30 minutes
                'QUIET': 180  # 3 hours
            }
            
            duration = base_duration.get(regime, 60)
            
            # Adjust based on volatility
            if volatility > self.volatility_config['volatility_threshold']:
                duration = int(duration * 0.5)  # Shorter duration in high volatility
            elif volatility < self.volatility_config['volatility_threshold'] * 0.5:
                duration = int(duration * 1.5)  # Longer duration in low volatility
            
            return max(15, min(duration, 480))  # Between 15 minutes and 8 hours
            
        except Exception as e:
            self.logger.error(f"Error estimating regime duration: {e}")
            return 60
    
    def _update_volatility_forecasting(self):
        """Update volatility forecasting model"""
        try:
            if len(self.volatility_history) < self.volatility_config['garch_period']:
                return
            
            # Simple volatility forecasting using exponential smoothing
            volatilities = list(self.volatility_history)
            alpha = 0.1  # Smoothing parameter
            
            # Calculate exponentially smoothed volatility
            smoothed_vol = volatilities[0]
            for vol in volatilities[1:]:
                smoothed_vol = alpha * vol + (1 - alpha) * smoothed_vol
            
            # Store forecast
            self.volatility_forecast = smoothed_vol
            
        except Exception as e:
            self.logger.error(f"Error updating volatility forecasting: {e}")
    
    def _update_factor_models(self):
        """Update factor exposure models"""
        try:
            if len(self.price_history) < self.lookback_periods['medium']:
                return
            
            prices = np.array(list(self.price_history))
            volumes = np.array(list(self.volume_history))
            funding_rates = list(self.funding_history)
            
            # Calculate factor exposures
            momentum_exposure = self._calculate_momentum_exposure(prices)
            mean_reversion_exposure = self._calculate_mean_reversion_exposure(prices)
            volatility_exposure = self._calculate_volatility_exposure(prices)
            funding_exposure = self._calculate_funding_exposure(funding_rates)
            volume_exposure = self._calculate_volume_exposure(volumes)
            
            # Calculate correlation with market
            correlation = self._calculate_market_correlation(prices)
            
            self.factor_exposures['current'] = FactorExposure(
                momentum=momentum_exposure,
                mean_reversion=mean_reversion_exposure,
                volatility=volatility_exposure,
                funding_rate=funding_exposure,
                volume=volume_exposure,
                correlation=correlation
            )
            
        except Exception as e:
            self.logger.error(f"Error updating factor models: {e}")
    
    def _calculate_momentum_exposure(self, prices: np.ndarray) -> float:
        """Calculate momentum factor exposure"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Calculate momentum using multiple timeframes
            short_momentum = (prices[-5] - prices[-10]) / prices[-10]
            medium_momentum = (prices[-10] - prices[-20]) / prices[-20]
            
            # Weighted average
            momentum = 0.7 * short_momentum + 0.3 * medium_momentum
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum exposure: {e}")
            return 0.0
    
    def _calculate_mean_reversion_exposure(self, prices: np.ndarray) -> float:
        """Calculate mean reversion factor exposure"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Calculate mean reversion using Z-score
            recent_prices = prices[-20:]
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            if std_price == 0:
                return 0.0
            
            current_price = prices[-1]
            z_score = (current_price - mean_price) / std_price
            
            # Mean reversion exposure is negative of Z-score
            return -z_score
            
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion exposure: {e}")
            return 0.0
    
    def _calculate_volatility_exposure(self, prices: np.ndarray) -> float:
        """Calculate volatility factor exposure"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Calculate rolling volatility
            returns = np.diff(prices[-20:])
            volatility = np.std(returns) if len(returns) > 1 else 0.0
            
            # Normalize volatility exposure
            return volatility * 100  # Scale for better interpretation
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility exposure: {e}")
            return 0.0
    
    def _calculate_funding_exposure(self, funding_rates: List[float]) -> float:
        """Calculate funding rate factor exposure"""
        try:
            if len(funding_rates) < 5:
                return 0.0
            
            # Calculate average funding rate
            avg_funding = np.mean(funding_rates[-5:])
            
            # Normalize funding exposure
            return avg_funding * 1000  # Scale for better interpretation
            
        except Exception as e:
            self.logger.error(f"Error calculating funding exposure: {e}")
            return 0.0
    
    def _calculate_volume_exposure(self, volumes: np.ndarray) -> float:
        """Calculate volume factor exposure"""
        try:
            if len(volumes) < 20:
                return 0.0
            
            # Calculate volume momentum
            recent_volume = np.mean(volumes[-5:])
            historical_volume = np.mean(volumes[-20:])
            
            if historical_volume == 0:
                return 0.0
            
            volume_exposure = (recent_volume - historical_volume) / historical_volume
            
            return volume_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating volume exposure: {e}")
            return 0.0
    
    def _calculate_market_correlation(self, prices: np.ndarray) -> float:
        """Calculate correlation with market (simplified)"""
        try:
            # This is a simplified correlation calculation
            # In production, you would compare with a market index
            if len(prices) < 20:
                return 0.0
            
            # Calculate price momentum as proxy for market correlation
            returns = np.diff(prices[-20:])
            correlation = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating market correlation: {e}")
            return 0.0
    
    def generate_signal(self) -> SignalStrength:
        """Generate trading signal using quantitative models"""
        try:
            if not self.current_regime or not self.factor_exposures:
                return SignalStrength(
                    signal='HOLD',
                    strength=0.0,
                    confidence=0.0,
                    expected_return=0.0,
                    risk_score=0.5,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0
                )
            
            # Get current factor exposures
            exposures = self.factor_exposures['current']
            regime = self.current_regime
            
            # Calculate signal based on regime and factors
            signal, strength, confidence = self._calculate_signal(regime, exposures)
            
            # Calculate expected return and risk
            expected_return = self._calculate_expected_return(signal, exposures, regime)
            risk_score = self._calculate_risk_score(exposures, regime)
            sharpe_ratio = expected_return / risk_score if risk_score > 0 else 0.0
            max_drawdown = self._calculate_max_drawdown(exposures, regime)
            
            signal_strength = SignalStrength(
                signal=signal,
                strength=strength,
                confidence=confidence,
                expected_return=expected_return,
                risk_score=risk_score,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
            # Store signal for performance tracking
            self.signal_history.append(signal_strength)
            self.strategy_metrics['total_signals'] += 1
            
            return signal_strength
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return SignalStrength(
                signal='HOLD',
                strength=0.0,
                confidence=0.0,
                expected_return=0.0,
                risk_score=0.5,
                sharpe_ratio=0.0,
                max_drawdown=0.0
            )
    
    def _calculate_signal(self, regime: RegimeAnalysis, exposures: FactorExposure) -> Tuple[str, float, float]:
        """Calculate trading signal based on regime and factor exposures"""
        try:
            signal = 'HOLD'
            strength = 0.0
            confidence = 0.0
            
            # Regime-based signal generation
            if regime.regime == 'TRENDING_UP':
                if exposures.momentum > 0.02:  # 2% momentum
                    signal = 'LONG'
                    strength = min(exposures.momentum * 10, 1.0)
                    confidence = regime.confidence
            elif regime.regime == 'TRENDING_DOWN':
                if exposures.momentum < -0.02:  # -2% momentum
                    signal = 'SHORT'
                    strength = min(abs(exposures.momentum) * 10, 1.0)
                    confidence = regime.confidence
            elif regime.regime == 'MEAN_REVERTING':
                if abs(exposures.mean_reversion) > 1.5:  # Z-score > 1.5
                    if exposures.mean_reversion > 0:
                        signal = 'SHORT'  # Overbought, expect reversion down
                    else:
                        signal = 'LONG'   # Oversold, expect reversion up
                    strength = min(abs(exposures.mean_reversion) / 3.0, 1.0)
                    confidence = regime.confidence
            elif regime.regime == 'VOLATILE':
                # In volatile markets, use momentum with higher threshold
                if exposures.momentum > 0.03:  # 3% momentum
                    signal = 'LONG'
                    strength = min(exposures.momentum * 8, 1.0)
                    confidence = regime.confidence * 0.8  # Lower confidence in volatile markets
                elif exposures.momentum < -0.03:  # -3% momentum
                    signal = 'SHORT'
                    strength = min(abs(exposures.momentum) * 8, 1.0)
                    confidence = regime.confidence * 0.8
            
            # Adjust for funding rate
            if exposures.funding_rate > 0.002:  # 0.2% funding rate
                if signal == 'LONG':
                    strength *= 1.2  # Boost long signal in high funding
                    confidence *= 1.1
                elif signal == 'SHORT':
                    strength *= 0.8  # Reduce short signal in high funding
                    confidence *= 0.9
            
            # Adjust for volume
            if exposures.volume > 0.2:  # 20% volume increase
                strength *= 1.1
                confidence *= 1.05
            
            return signal, strength, confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating signal: {e}")
            return 'HOLD', 0.0, 0.0
    
    def _calculate_expected_return(self, signal: str, exposures: FactorExposure, 
                                 regime: RegimeAnalysis) -> float:
        """Calculate expected return for signal"""
        try:
            if signal == 'HOLD':
                return 0.0
            
            # Base expected return
            base_return = 0.01  # 1% base return
            
            # Adjust for regime
            if regime.regime in ['TRENDING_UP', 'TRENDING_DOWN']:
                base_return *= 1.5
            elif regime.regime == 'MEAN_REVERTING':
                base_return *= 1.2
            elif regime.regime == 'VOLATILE':
                base_return *= 1.3
            
            # Adjust for factor exposures
            if signal == 'LONG':
                if exposures.momentum > 0:
                    base_return *= (1 + exposures.momentum * 2)
                if exposures.funding_rate > 0:
                    base_return *= (1 + exposures.funding_rate * 100)
            elif signal == 'SHORT':
                if exposures.momentum < 0:
                    base_return *= (1 + abs(exposures.momentum) * 2)
                if exposures.funding_rate < 0:
                    base_return *= (1 + abs(exposures.funding_rate) * 100)
            
            return min(base_return, 0.05)  # Cap at 5% expected return
            
        except Exception as e:
            self.logger.error(f"Error calculating expected return: {e}")
            return 0.0
    
    def _calculate_risk_score(self, exposures: FactorExposure, regime: RegimeAnalysis) -> float:
        """Calculate risk score for signal"""
        try:
            # Base risk
            base_risk = 0.02  # 2% base risk
            
            # Adjust for volatility
            base_risk *= (1 + exposures.volatility)
            
            # Adjust for regime
            if regime.regime == 'VOLATILE':
                base_risk *= 1.5
            elif regime.regime == 'QUIET':
                base_risk *= 0.8
            
            # Adjust for correlation
            base_risk *= (1 + abs(exposures.correlation) * 0.5)
            
            return min(base_risk, 0.1)  # Cap at 10% risk
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.02
    
    def _calculate_max_drawdown(self, exposures: FactorExposure, regime: RegimeAnalysis) -> float:
        """Calculate expected maximum drawdown"""
        try:
            # Base drawdown
            base_drawdown = 0.05  # 5% base drawdown
            
            # Adjust for volatility
            base_drawdown *= (1 + exposures.volatility * 2)
            
            # Adjust for regime
            if regime.regime == 'VOLATILE':
                base_drawdown *= 1.5
            elif regime.regime == 'QUIET':
                base_drawdown *= 0.8
            
            return min(base_drawdown, 0.2)  # Cap at 20% drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.05
    
    def calculate_kelly_position_size(self, signal: SignalStrength, 
                                    account_value: float) -> float:
        """Calculate optimal position size using Kelly criterion"""
        try:
            if signal.signal == 'HOLD' or signal.expected_return <= 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds, p = probability of win, q = probability of loss
            
            # Estimate win probability from confidence
            win_prob = signal.confidence
            loss_prob = 1 - win_prob
            
            # Estimate odds from expected return
            odds = signal.expected_return / signal.risk_score if signal.risk_score > 0 else 1.0
            
            # Kelly fraction
            kelly_fraction = (odds * win_prob - loss_prob) / odds
            
            # Apply conservative cap
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Calculate position size
            position_size = kelly_fraction * account_value
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {e}")
            return 0.0
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        try:
            if self.strategy_metrics['total_signals'] > 0:
                self.strategy_metrics['win_rate'] = (
                    self.strategy_metrics['correct_signals'] / 
                    self.strategy_metrics['total_signals']
                )
            
            return self.strategy_metrics.copy()
            
        except Exception as e:
            self.logger.error(f"Error getting strategy metrics: {e}")
            return {}
    
    def update_signal_performance(self, signal: SignalStrength, actual_return: float):
        """Update signal performance for learning"""
        try:
            # Determine if signal was correct
            if signal.signal == 'LONG' and actual_return > 0:
                self.strategy_metrics['correct_signals'] += 1
            elif signal.signal == 'SHORT' and actual_return < 0:
                self.strategy_metrics['correct_signals'] += 1
            elif signal.signal == 'HOLD':
                self.strategy_metrics['correct_signals'] += 1
            
            # Update total return
            self.strategy_metrics['total_return'] += actual_return
            
            # Update Sharpe ratio
            if len(self.signal_history) > 1:
                returns = [s.expected_return for s in self.signal_history[-10:]]
                if len(returns) > 1:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        self.strategy_metrics['sharpe_ratio'] = mean_return / std_return
            
            # Update max drawdown
            if actual_return < 0:
                self.strategy_metrics['max_drawdown'] = max(
                    self.strategy_metrics['max_drawdown'],
                    abs(actual_return)
                )
            
        except Exception as e:
            self.logger.error(f"Error updating signal performance: {e}")

