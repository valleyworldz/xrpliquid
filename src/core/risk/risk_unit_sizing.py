"""
🎯 RISK UNIT SIZING SYSTEM
==========================
Advanced risk management with volatility targeting and equity-at-risk sizing
Replaces static position caps with dynamic risk-based position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import math
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger

@dataclass
class RiskUnitConfig:
    """Configuration for risk unit sizing system"""
    
    # Volatility targeting
    target_volatility_percent: float = 2.0        # Target 2% daily volatility
    volatility_lookback_days: int = 30            # 30-day volatility calculation
    volatility_update_frequency_minutes: int = 60  # Update volatility every hour
    min_volatility_percent: float = 0.5           # Minimum volatility floor
    max_volatility_percent: float = 10.0          # Maximum volatility cap
    
    # Equity at risk
    max_equity_at_risk_percent: float = 1.0       # Maximum 1% equity at risk per trade
    base_equity_at_risk_percent: float = 0.5      # Base 0.5% equity at risk per trade
    risk_scaling_factor: float = 1.5              # Risk scaling based on confidence
    max_total_equity_at_risk_percent: float = 5.0 # Maximum 5% total equity at risk
    
    # Position sizing constraints
    min_position_size_usd: float = 25.0           # Minimum position size
    max_position_size_usd: float = 10000.0        # Maximum position size
    position_size_multiplier: float = 0.1         # Max position as % of account value
    
    # Risk management
    max_drawdown_percent: float = 10.0            # Maximum drawdown limit
    emergency_risk_reduction_factor: float = 0.5  # Reduce risk by 50% in emergency
    correlation_risk_limit: float = 0.7           # Maximum correlation between positions
    
    # Kelly Criterion integration
    kelly_multiplier: float = 0.25                # Use 25% of Kelly fraction
    kelly_max_fraction: float = 0.1               # Maximum Kelly fraction
    
    # Market regime adjustments
    regime_risk_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'trending': 1.2,      # Increase risk in trending markets
        'ranging': 0.8,       # Reduce risk in ranging markets
        'volatile': 0.6,      # Reduce risk in volatile markets
        'calm': 1.0,          # Normal risk in calm markets
        'bullish': 1.1,       # Slightly increase risk in bullish markets
        'bearish': 0.9        # Slightly reduce risk in bearish markets
    })

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    
    # Volatility metrics
    current_volatility_percent: float
    target_volatility_percent: float
    volatility_ratio: float  # current/target
    
    # Equity at risk metrics
    equity_at_risk_usd: float
    equity_at_risk_percent: float
    total_equity_at_risk_percent: float
    
    # Position sizing metrics
    position_size_usd: float
    position_size_percent: float
    risk_unit_size: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    max_drawdown_percent: float
    var_95_percent: float  # Value at Risk 95%
    
    # Market regime
    market_regime: str
    regime_risk_multiplier: float
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)

@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    
    symbol: str
    position_size_usd: float
    position_size_percent: float
    equity_at_risk_usd: float
    equity_at_risk_percent: float
    volatility_percent: float
    risk_unit_size: float
    correlation_with_portfolio: float
    beta: float
    timestamp: float = field(default_factory=time.time)

class RiskUnitSizing:
    """
    🎯 RISK UNIT SIZING SYSTEM
    Advanced risk management with volatility targeting and equity-at-risk sizing
    """
    
    def __init__(self, 
                 config: RiskUnitConfig,
                 logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or Logger()
        
        # Risk tracking
        self.current_positions: Dict[str, PositionRisk] = {}
        self.volatility_history: Dict[str, List[Tuple[float, float]]] = {}  # symbol -> [(timestamp, volatility)]
        self.portfolio_metrics: Optional[RiskMetrics] = None
        
        # Performance tracking
        self.total_equity_at_risk_usd = 0.0
        self.max_drawdown_percent = 0.0
        self.emergency_mode = False
        
        # Market regime
        self.current_market_regime = 'calm'
        self.regime_confidence = 0.5
        
        self.logger.info("🎯 [RISK_UNIT] Risk Unit Sizing System initialized")
        self.logger.info(f"📊 [RISK_UNIT] Target volatility: {self.config.target_volatility_percent}%")
        self.logger.info(f"📊 [RISK_UNIT] Max equity at risk: {self.config.max_equity_at_risk_percent}%")
        self.logger.info(f"📊 [RISK_UNIT] Kelly multiplier: {self.config.kelly_multiplier}")
    
    def calculate_volatility(self, 
                           symbol: str,
                           price_history: List[float],
                           lookback_days: Optional[int] = None) -> float:
        """
        Calculate historical volatility for a symbol
        
        Args:
            symbol: Trading symbol
            price_history: List of historical prices
            lookback_days: Number of days to look back (default from config)
        
        Returns:
            Annualized volatility as percentage
        """
        
        if not price_history or len(price_history) < 2:
            return self.config.min_volatility_percent
        
        lookback = lookback_days or self.config.volatility_lookback_days
        
        # Use last N days of data
        recent_prices = price_history[-min(lookback, len(price_history)):]
        
        if len(recent_prices) < 2:
            return self.config.min_volatility_percent
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(recent_prices)):
            daily_return = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(daily_return)
        
        if not returns:
            return self.config.min_volatility_percent
        
        # Calculate standard deviation of returns
        returns_std = np.std(returns)
        
        # Annualize volatility (assuming 365 trading days)
        annualized_volatility = returns_std * math.sqrt(365)
        
        # Convert to percentage and apply bounds
        volatility_percent = annualized_volatility * 100
        volatility_percent = max(self.config.min_volatility_percent, 
                               min(volatility_percent, self.config.max_volatility_percent))
        
        # Store in history
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []
        
        self.volatility_history[symbol].append((time.time(), volatility_percent))
        
        # Keep only recent history
        cutoff_time = time.time() - (self.config.volatility_lookback_days * 24 * 3600)
        self.volatility_history[symbol] = [
            (ts, vol) for ts, vol in self.volatility_history[symbol] 
            if ts > cutoff_time
        ]
        
        return volatility_percent
    
    def calculate_equity_at_risk(self, 
                               symbol: str,
                               position_size_usd: float,
                               volatility_percent: float,
                               confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate equity at risk for a position
        
        Args:
            symbol: Trading symbol
            position_size_usd: Position size in USD
            volatility_percent: Annualized volatility percentage
            confidence_level: Confidence level for VaR calculation
        
        Returns:
            Tuple of (equity_at_risk_usd, equity_at_risk_percent)
        """
        
        # Calculate Value at Risk (VaR)
        # Using normal distribution approximation
        z_score = 1.96 if confidence_level == 0.95 else 2.33  # 95% or 99% confidence
        
        # Daily volatility
        daily_volatility = volatility_percent / math.sqrt(365)
        
        # VaR calculation (1-day)
        var_percent = z_score * daily_volatility
        var_usd = position_size_usd * (var_percent / 100)
        
        # Equity at risk (using VaR as proxy)
        equity_at_risk_usd = var_usd
        equity_at_risk_percent = (equity_at_risk_usd / position_size_usd) * 100 if position_size_usd > 0 else 0
        
        return equity_at_risk_usd, equity_at_risk_percent
    
    def calculate_risk_unit_size(self, 
                               symbol: str,
                               account_value: float,
                               volatility_percent: float,
                               confidence_score: float = 0.5,
                               market_regime: Optional[str] = None) -> float:
        """
        Calculate risk unit size based on volatility targeting and equity at risk
        
        Args:
            symbol: Trading symbol
            account_value: Total account value in USD
            volatility_percent: Current volatility percentage
            confidence_score: Confidence score (0-1)
            market_regime: Current market regime
        
        Returns:
            Risk unit size in USD
        """
        
        # Get market regime
        regime = market_regime or self.current_market_regime
        regime_multiplier = self.config.regime_risk_multipliers.get(regime, 1.0)
        
        # Calculate base equity at risk
        base_equity_at_risk_percent = self.config.base_equity_at_risk_percent
        
        # Scale based on confidence
        confidence_multiplier = 1.0 + (confidence_score - 0.5) * self.config.risk_scaling_factor
        confidence_multiplier = max(0.1, min(2.0, confidence_multiplier))  # Bound between 0.1 and 2.0
        
        # Apply regime adjustment
        regime_adjusted_risk = base_equity_at_risk_percent * regime_multiplier * confidence_multiplier
        
        # Apply emergency mode reduction
        if self.emergency_mode:
            regime_adjusted_risk *= self.config.emergency_risk_reduction_factor
        
        # Calculate equity at risk in USD
        equity_at_risk_usd = account_value * (regime_adjusted_risk / 100)
        
        # Calculate position size based on volatility targeting
        # Position size = (Target Volatility / Current Volatility) * Equity at Risk
        volatility_ratio = self.config.target_volatility_percent / volatility_percent
        volatility_ratio = max(0.1, min(5.0, volatility_ratio))  # Bound between 0.1 and 5.0
        
        # Risk unit size
        risk_unit_size = equity_at_risk_usd * volatility_ratio
        
        # Apply position size constraints
        min_size = self.config.min_position_size_usd
        max_size = min(
            self.config.max_position_size_usd,
            account_value * self.config.position_size_multiplier
        )
        
        risk_unit_size = max(min_size, min(risk_unit_size, max_size))
        
        return risk_unit_size
    
    def calculate_kelly_position_size(self, 
                                    symbol: str,
                                    win_probability: float,
                                    avg_win_percent: float,
                                    avg_loss_percent: float,
                                    account_value: float) -> float:
        """
        Calculate Kelly Criterion position size
        
        Args:
            symbol: Trading symbol
            win_probability: Probability of winning (0-1)
            avg_win_percent: Average win percentage
            avg_loss_percent: Average loss percentage
            account_value: Total account value
        
        Returns:
            Kelly position size in USD
        """
        
        # Kelly Criterion: f* = (bp - q) / b
        # Where: b = avg_win_percent, p = win_probability, q = 1 - win_probability
        
        if avg_win_percent <= 0 or avg_loss_percent <= 0:
            return 0.0
        
        # Calculate Kelly fraction
        kelly_fraction = (win_probability * avg_win_percent - (1 - win_probability) * avg_loss_percent) / avg_win_percent
        
        # Apply Kelly multiplier and bounds
        kelly_fraction *= self.config.kelly_multiplier
        kelly_fraction = max(0, min(kelly_fraction, self.config.kelly_max_fraction))
        
        # Calculate position size
        kelly_position_size = account_value * kelly_fraction
        
        return kelly_position_size
    
    def calculate_optimal_position_size(self, 
                                      symbol: str,
                                      account_value: float,
                                      volatility_percent: float,
                                      confidence_score: float = 0.5,
                                      win_probability: float = 0.5,
                                      avg_win_percent: float = 0.01,
                                      avg_loss_percent: float = 0.01,
                                      market_regime: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal position size using risk unit sizing
        
        Args:
            symbol: Trading symbol
            account_value: Total account value in USD
            volatility_percent: Current volatility percentage
            confidence_score: Confidence score (0-1)
            win_probability: Probability of winning (0-1)
            avg_win_percent: Average win percentage
            avg_loss_percent: Average loss percentage
            market_regime: Current market regime
        
        Returns:
            Tuple of (optimal_position_size_usd, risk_metrics_dict)
        """
        
        # Calculate risk unit size
        risk_unit_size = self.calculate_risk_unit_size(
            symbol, account_value, volatility_percent, confidence_score, market_regime
        )
        
        # Calculate Kelly position size
        kelly_size = self.calculate_kelly_position_size(
            symbol, win_probability, avg_win_percent, avg_loss_percent, account_value
        )
        
        # Use the smaller of the two for conservative sizing
        optimal_position_size = min(risk_unit_size, kelly_size)
        
        # Calculate equity at risk
        equity_at_risk_usd, equity_at_risk_percent = self.calculate_equity_at_risk(
            symbol, optimal_position_size, volatility_percent
        )
        
        # Calculate risk metrics
        risk_metrics = {
            'symbol': symbol,
            'position_size_usd': optimal_position_size,
            'position_size_percent': (optimal_position_size / account_value) * 100,
            'risk_unit_size': risk_unit_size,
            'kelly_size': kelly_size,
            'equity_at_risk_usd': equity_at_risk_usd,
            'equity_at_risk_percent': equity_at_risk_percent,
            'volatility_percent': volatility_percent,
            'volatility_ratio': self.config.target_volatility_percent / volatility_percent,
            'confidence_score': confidence_score,
            'win_probability': win_probability,
            'market_regime': market_regime or self.current_market_regime,
            'regime_multiplier': self.config.regime_risk_multipliers.get(market_regime or self.current_market_regime, 1.0),
            'emergency_mode': self.emergency_mode,
            'timestamp': time.time()
        }
        
        return optimal_position_size, risk_metrics
    
    def update_portfolio_risk(self, 
                            account_value: float,
                            positions: Dict[str, Dict[str, Any]]) -> RiskMetrics:
        """
        Update portfolio-level risk metrics
        
        Args:
            account_value: Total account value in USD
            positions: Dictionary of current positions
        
        Returns:
            Updated RiskMetrics object
        """
        
        # Calculate total equity at risk
        total_equity_at_risk_usd = 0.0
        total_position_value = 0.0
        
        for symbol, position in positions.items():
            position_value = position.get('value_usd', 0.0)
            volatility = position.get('volatility_percent', self.config.min_volatility_percent)
            
            # Calculate equity at risk for this position
            equity_at_risk_usd, _ = self.calculate_equity_at_risk(symbol, position_value, volatility)
            total_equity_at_risk_usd += equity_at_risk_usd
            total_position_value += position_value
        
        # Calculate portfolio volatility (simplified)
        portfolio_volatility = self.config.target_volatility_percent  # Default
        if positions:
            # Weighted average volatility
            total_weight = 0.0
            weighted_vol = 0.0
            
            for symbol, position in positions.items():
                position_value = position.get('value_usd', 0.0)
                volatility = position.get('volatility_percent', self.config.min_volatility_percent)
                
                if total_position_value > 0:
                    weight = position_value / total_position_value
                    weighted_vol += volatility * weight
                    total_weight += weight
            
            if total_weight > 0:
                portfolio_volatility = weighted_vol / total_weight
        
        # Calculate risk metrics
        total_equity_at_risk_percent = (total_equity_at_risk_usd / account_value) * 100 if account_value > 0 else 0
        
        self.portfolio_metrics = RiskMetrics(
            current_volatility_percent=portfolio_volatility,
            target_volatility_percent=self.config.target_volatility_percent,
            volatility_ratio=portfolio_volatility / self.config.target_volatility_percent,
            equity_at_risk_usd=total_equity_at_risk_usd,
            equity_at_risk_percent=total_equity_at_risk_percent,
            total_equity_at_risk_percent=total_equity_at_risk_percent,
            position_size_usd=total_position_value,
            position_size_percent=(total_position_value / account_value) * 100 if account_value > 0 else 0,
            risk_unit_size=total_equity_at_risk_usd,
            sharpe_ratio=0.0,  # Would need historical returns
            max_drawdown_percent=self.max_drawdown_percent,
            var_95_percent=total_equity_at_risk_usd,
            market_regime=self.current_market_regime,
            regime_risk_multiplier=self.config.regime_risk_multipliers.get(self.current_market_regime, 1.0)
        )
        
        # Update total equity at risk
        self.total_equity_at_risk_usd = total_equity_at_risk_usd
        
        # Check for emergency mode
        if total_equity_at_risk_percent > self.config.max_total_equity_at_risk_percent:
            self.emergency_mode = True
            self.logger.warning(f"⚠️ [RISK_UNIT] Emergency mode activated - Total equity at risk: {total_equity_at_risk_percent:.2f}%")
        
        return self.portfolio_metrics
    
    def set_market_regime(self, regime: str, confidence: float = 0.5):
        """Set current market regime"""
        self.current_market_regime = regime
        self.regime_confidence = confidence
        self.logger.info(f"📊 [RISK_UNIT] Market regime set to: {regime} (confidence: {confidence:.2f})")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        return {
            'portfolio_metrics': self.portfolio_metrics.__dict__ if self.portfolio_metrics else {},
            'current_positions': {symbol: pos.__dict__ for symbol, pos in self.current_positions.items()},
            'total_equity_at_risk_usd': self.total_equity_at_risk_usd,
            'max_drawdown_percent': self.max_drawdown_percent,
            'emergency_mode': self.emergency_mode,
            'market_regime': self.current_market_regime,
            'regime_confidence': self.regime_confidence,
            'config': {
                'target_volatility_percent': self.config.target_volatility_percent,
                'max_equity_at_risk_percent': self.config.max_equity_at_risk_percent,
                'max_total_equity_at_risk_percent': self.config.max_total_equity_at_risk_percent,
                'kelly_multiplier': self.config.kelly_multiplier,
                'regime_risk_multipliers': self.config.regime_risk_multipliers
            }
        }
    
    def reset_emergency_mode(self):
        """Reset emergency mode"""
        self.emergency_mode = False
        self.logger.info("✅ [RISK_UNIT] Emergency mode reset")
    
    def update_drawdown(self, current_drawdown_percent: float):
        """Update maximum drawdown"""
        if current_drawdown_percent > self.max_drawdown_percent:
            self.max_drawdown_percent = current_drawdown_percent
            
            # Activate emergency mode if drawdown exceeds limit
            if current_drawdown_percent > self.config.max_drawdown_percent:
                self.emergency_mode = True
                self.logger.warning(f"⚠️ [RISK_UNIT] Emergency mode activated - Drawdown: {current_drawdown_percent:.2f}%")
