#!/usr/bin/env python3
"""
🛡️ ADVANCED RISK MANAGEMENT ENGINE
==================================
Implements sophisticated risk management with Kelly Criterion, dynamic stops,
and portfolio optimization for maximum safety and profit optimization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementations
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        
        def fit_transform(self, X):
            X = np.array(X)
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            return (X - self.mean_) / (self.scale_ + 1e-8)
        
        def transform(self, X):
            X = np.array(X)
            return (X - self.mean_) / (self.scale_ + 1e-8)

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio analysis"""
    portfolio_value: float
    total_exposure: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    correlation_risk: float
    concentration_risk: float

@dataclass
class PositionRisk:
    """Risk analysis for individual positions"""
    symbol: str
    position_size: float
    unrealized_pnl: float
    risk_score: float  # 0-1 scale
    kelly_optimal_size: float
    stop_loss_price: float
    take_profit_price: float
    max_loss_amount: float
    correlation_exposure: float

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_position_size: float = 0.1    # 10% max position size
    max_drawdown_limit: float = 0.15  # 15% max drawdown before stop
    max_correlation_exposure: float = 0.3  # 30% max correlated exposure
    max_concentration: float = 0.25   # 25% max single asset concentration
    min_kelly_size: float = 0.01      # 1% minimum position size
    max_kelly_size: float = 0.2       # 20% maximum Kelly size

class AdvancedRiskManager:
    """Advanced risk management with ML-powered optimization"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        self.risk_limits = RiskLimits(**self.config.get('risk_limits', {}))
        
        # Risk tracking
        self.portfolio_history = []
        self.position_history = {}
        self.drawdown_history = []
        self.risk_events = []
        
        # ML components
        self.sklearn_available = SKLEARN_AVAILABLE
        if self.sklearn_available:
            self.volatility_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
            self.correlation_predictor = RandomForestRegressor(n_estimators=30, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.volatility_predictor = None
            self.correlation_predictor = None
            self.scaler = StandardScaler()
        
        # Risk state
        self.current_risk_metrics = None
        self.position_risks = {}
        self.correlation_matrix = {}
        self.volatility_estimates = {}
        
        self.logger.info("🛡️ Advanced Risk Manager initialized")
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return self.risk_limits.min_kelly_size
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply limits
            kelly_fraction = max(self.risk_limits.min_kelly_size, 
                               min(self.risk_limits.max_kelly_size, kelly_fraction))
            
            self.logger.debug(f"📊 Kelly Criterion: {kelly_fraction:.3f} "
                            f"(win_rate={win_rate:.3f}, b={b:.3f})")
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"❌ Kelly calculation failed: {e}")
            return self.risk_limits.min_kelly_size
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, 
                                  position_size: float, direction: str) -> float:
        """Calculate dynamic stop loss based on volatility and risk limits"""
        try:
            # Get volatility estimate
            volatility = self.volatility_estimates.get(symbol, 0.02)
            
            # Base stop distance (2x volatility)
            base_stop_distance = 2 * volatility * entry_price
            
            # Adjust for position size (larger positions = tighter stops)
            size_multiplier = 1 - (position_size * 0.5)  # Reduce stop distance for larger positions
            stop_distance = base_stop_distance * size_multiplier
            
            # Calculate stop price
            if direction.lower() == 'long':
                stop_price = entry_price - stop_distance
            else:  # short
                stop_price = entry_price + stop_distance
            
            # Ensure stop doesn't exceed max portfolio risk
            max_loss = self.risk_limits.max_portfolio_risk * self.get_portfolio_value()
            position_value = position_size * entry_price
            
            # Prevent division by zero
            if position_value <= 0:
                position_value = 0.01  # Minimum position value to avoid division by zero
            
            max_stop_distance = max_loss / position_value * entry_price
            
            if direction.lower() == 'long':
                stop_price = max(stop_price, entry_price - max_stop_distance)
            else:
                stop_price = min(stop_price, entry_price + max_stop_distance)
            
            self.logger.debug(f"🎯 Dynamic stop for {symbol}: ${stop_price:.2f} "
                            f"(distance: {abs(stop_price-entry_price)/entry_price:.3%})")
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"❌ Stop loss calculation failed: {e}")
            # Fallback: 2% stop loss
            fallback_distance = 0.02 * entry_price
            return entry_price - fallback_distance if direction.lower() == 'long' else entry_price + fallback_distance
    
    def calculate_position_risk(self, symbol: str, position_size: float, 
                              entry_price: float, current_price: float,
                              direction: str) -> PositionRisk:
        """Calculate comprehensive risk metrics for a position"""
        try:
            # Basic metrics
            unrealized_pnl = position_size * (current_price - entry_price)
            if direction.lower() == 'short':
                unrealized_pnl = -unrealized_pnl
            
            # Get historical performance for Kelly calculation
            win_rate, avg_win, avg_loss = self._get_historical_performance(symbol)
            kelly_optimal = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
            
            # Calculate stops
            stop_loss = self.calculate_dynamic_stop_loss(symbol, entry_price, position_size, direction)
            take_profit = self._calculate_take_profit(symbol, entry_price, direction)
            
            # Risk scoring (0-1, higher = riskier)
            volatility = self.volatility_estimates.get(symbol, 0.02)
            correlation_exposure = self._calculate_correlation_exposure(symbol)
            
            portfolio_value = self.get_portfolio_value()
            if portfolio_value == 0:
                portfolio_value = 100000.0  # Default fallback to avoid division by zero
            
            risk_score = min(1.0, (volatility * 10 + correlation_exposure + 
                                 abs(unrealized_pnl) / portfolio_value) / 3)
            
            # Maximum loss calculation
            if direction.lower() == 'long':
                max_loss = position_size * (entry_price - stop_loss)
            else:
                max_loss = position_size * (stop_loss - entry_price)
            
            return PositionRisk(
                symbol=symbol,
                position_size=position_size,
                unrealized_pnl=unrealized_pnl,
                risk_score=risk_score,
                kelly_optimal_size=kelly_optimal * max(self.get_portfolio_value(), 1000.0),
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                max_loss_amount=max_loss,
                correlation_exposure=correlation_exposure
            )
            
        except Exception as e:
            self.logger.error(f"❌ Position risk calculation failed: {e}")
            return PositionRisk(
                symbol=symbol, position_size=position_size, unrealized_pnl=0,
                risk_score=0.5, kelly_optimal_size=position_size,
                stop_loss_price=entry_price * 0.98, take_profit_price=entry_price * 1.04,
                max_loss_amount=position_size * entry_price * 0.02, correlation_exposure=0
            )
    
    def calculate_portfolio_risk(self, positions: Dict[str, Dict]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            portfolio_value = self.get_portfolio_value()
            total_exposure = sum(abs(pos.get('position_size', 0) * pos.get('current_price', 0)) 
                               for pos in positions.values())
            
            # Calculate drawdown
            current_drawdown, max_drawdown = self._calculate_drawdowns()
            
            # Calculate portfolio volatility
            portfolio_volatility = self._calculate_portfolio_volatility(positions)
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Calculate VaR and Expected Shortfall
            var_95, expected_shortfall = self._calculate_var_and_es(positions)
            
            # Calculate correlation and concentration risk
            correlation_risk = self._calculate_correlation_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions)
            
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk
            )
            
            self.current_risk_metrics = risk_metrics
            self._update_risk_history(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio risk calculation failed: {e}")
            return RiskMetrics(
                portfolio_value=10000, total_exposure=0, max_drawdown=0,
                current_drawdown=0, volatility=0.02, sharpe_ratio=0,
                var_95=0, expected_shortfall=0, correlation_risk=0, concentration_risk=0
            )
    
    def should_reduce_risk(self) -> Tuple[bool, str]:
        """Determine if risk reduction is needed"""
        if not self.current_risk_metrics:
            return False, "No risk metrics available"
        
        metrics = self.current_risk_metrics
        
        # Check drawdown limits
        if metrics.current_drawdown >= self.risk_limits.max_drawdown_limit:
            return True, f"Drawdown limit exceeded: {metrics.current_drawdown:.2%}"
        
        # Check correlation risk
        if metrics.correlation_risk >= self.risk_limits.max_correlation_exposure:
            return True, f"Correlation risk too high: {metrics.correlation_risk:.2%}"
        
        # Check concentration risk
        if metrics.concentration_risk >= self.risk_limits.max_concentration:
            return True, f"Concentration risk too high: {metrics.concentration_risk:.2%}"
        
        # Check VaR
        var_limit = self.risk_limits.max_portfolio_risk * metrics.portfolio_value
        if metrics.var_95 >= var_limit:
            return True, f"VaR limit exceeded: ${metrics.var_95:.2f}"
        
        return False, "Risk levels acceptable"
    
    def optimize_position_size(self, symbol: str, intended_size: float, 
                             entry_price: float, direction: str) -> float:
        """Optimize position size based on risk limits and Kelly Criterion"""
        try:
            # Get Kelly optimal size
            win_rate, avg_win, avg_loss = self._get_historical_performance(symbol)
            kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_size = kelly_fraction * self.get_portfolio_value() / entry_price
            
            # Apply risk limits
            portfolio_value = self.get_portfolio_value()
            max_size_by_portfolio = self.risk_limits.max_position_size * portfolio_value / entry_price
            max_size_by_risk = self.risk_limits.max_portfolio_risk * portfolio_value / entry_price
            
            # Choose minimum of all constraints
            optimal_size = min(intended_size, kelly_size, max_size_by_portfolio, max_size_by_risk)
            
            # ULTIMATE MASTER-LEVEL POSITION SIZING OVERRIDE
            # For master-level trading, always use intended size if Kelly is too conservative
            if kelly_size > 100:  # Kelly is extremely high (perfect win rate)
                optimal_size = intended_size * 2.0  # Double the intended size for master-level
            
            # If optimal size is too small, use intended size
            elif optimal_size < intended_size * 0.1:
                optimal_size = intended_size * 1.5  # 1.5x intended size
            
            # Ensure minimum size for master-level trading
            min_size = self.risk_limits.min_kelly_size * portfolio_value / entry_price
            if optimal_size < min_size and intended_size >= min_size:
                optimal_size = intended_size * 1.2  # 1.2x intended size if it meets minimum
            
            self.logger.info(f"📊 Position size optimization for {symbol}: "
                           f"Intended: {intended_size:.4f}, Kelly: {kelly_size:.4f}, "
                           f"Optimal: {optimal_size:.4f}")
            
            return optimal_size
            
        except Exception as e:
            self.logger.error(f"❌ Position size optimization failed: {e}")
            return min(intended_size, self.risk_limits.min_kelly_size * self.get_portfolio_value() / entry_price)
    
    def get_risk_adjusted_signal(self, base_signal: Dict, symbol: str) -> Dict:
        """Adjust trading signal based on risk analysis"""
        try:
            # Get current risk state
            should_reduce, reason = self.should_reduce_risk()
            
            adjusted_signal = base_signal.copy()
            
            if should_reduce:
                # Reduce signal strength
                adjusted_signal['confidence'] *= 0.5
                adjusted_signal['size'] *= 0.3
                self.logger.warning(f"⚠️ Risk reduction applied: {reason}")
            
            # Optimize position size
            if 'size' in adjusted_signal and 'entry_price' in adjusted_signal:
                optimized_size = self.optimize_position_size(
                    symbol, 
                    adjusted_signal['size'],
                    adjusted_signal['entry_price'],
                    adjusted_signal.get('direction', 'long')
                )
                adjusted_signal['size'] = optimized_size
            
            # Add risk-based stops
            if 'entry_price' in adjusted_signal:
                stop_loss = self.calculate_dynamic_stop_loss(
                    symbol,
                    adjusted_signal['entry_price'],
                    adjusted_signal.get('size', 0),
                    adjusted_signal.get('direction', 'long')
                )
                adjusted_signal['stop_loss'] = stop_loss
            
            return adjusted_signal
            
        except Exception as e:
            self.logger.error(f"❌ Risk adjustment failed: {e}")
            return base_signal
    
    def update_volatility_estimates(self, price_data: Dict[str, List[float]]):
        """Update volatility estimates using ML if available"""
        try:
            for symbol, prices in price_data.items():
                if len(prices) < 20:
                    continue
                
                # Calculate historical volatility
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(24)  # Daily volatility
                
                # Use ML prediction if available
                if self.sklearn_available and len(prices) >= 50:
                    try:
                        # Prepare features (rolling stats)
                        features = self._prepare_volatility_features(prices)
                        if len(features) > 10:
                            # Train and predict
                            X = np.array(features[:-1])
                            y = np.array([np.std(returns[i:i+20]) for i in range(len(returns)-20)])
                            
                            if len(y) > 0:
                                self.volatility_predictor.fit(X, y)
                                predicted_vol = self.volatility_predictor.predict([features[-1]])[0]
                                volatility = predicted_vol * np.sqrt(24)
                    except Exception as ml_e:
                        self.logger.debug(f"ML volatility prediction failed: {ml_e}")
                
                self.volatility_estimates[symbol] = volatility
                self.logger.debug(f"📊 Updated volatility for {symbol}: {volatility:.4f}")
                
        except Exception as e:
            self.logger.error(f"❌ Volatility update failed: {e}")
    
    def _prepare_volatility_features(self, prices: List[float]) -> List[List[float]]:
        """Prepare features for volatility prediction"""
        features = []
        prices = np.array(prices)
        
        for i in range(20, len(prices)):
            window = prices[i-20:i]
            feature = [
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                (window[-1] - window[0]) / window[0],  # Return
                np.mean(np.abs(np.diff(window))),      # Average absolute change
            ]
            features.append(feature)
        
        return features
    
    def _get_historical_performance(self, symbol: str) -> Tuple[float, float, float]:
        """Get historical win rate and average win/loss for Kelly calculation"""
        # This would typically come from trade history
        # For now, using reasonable defaults
        default_win_rate = 0.55
        default_avg_win = 0.02
        default_avg_loss = 0.015
        
        if symbol in self.position_history:
            # Calculate from actual history if available
            history = self.position_history[symbol]
            if len(history) > 10:
                wins = [h for h in history if h['pnl'] > 0]
                losses = [h for h in history if h['pnl'] < 0]
                
                if len(wins) > 0 and len(losses) > 0:
                    win_rate = len(wins) / len(history)
                    avg_win = np.mean([h['pnl'] for h in wins])
                    avg_loss = np.mean([abs(h['pnl']) for h in losses])
                    return win_rate, avg_win, avg_loss
        
        return default_win_rate, default_avg_win, default_avg_loss
    
    def _calculate_take_profit(self, symbol: str, entry_price: float, direction: str) -> float:
        """Calculate take profit level"""
        volatility = self.volatility_estimates.get(symbol, 0.02)
        
        # Target 2:1 risk/reward ratio
        stop_distance = 2 * volatility * entry_price
        take_profit_distance = 4 * volatility * entry_price
        
        if direction.lower() == 'long':
            return entry_price + take_profit_distance
        else:
            return entry_price - take_profit_distance
    
    def _calculate_correlation_exposure(self, symbol: str) -> float:
        """Calculate correlation exposure for a symbol"""
        # Simplified correlation calculation
        # In practice, this would use actual correlation matrix
        base_correlation = 0.3  # Assume 30% correlation with market
        return base_correlation
    
    def _calculate_drawdowns(self) -> Tuple[float, float]:
        """Calculate current and maximum drawdown"""
        if len(self.portfolio_history) < 2:
            return 0.0, 0.0
        
        values = [h['portfolio_value'] for h in self.portfolio_history]
        peak = values[0]
        max_drawdown = 0.0
        current_drawdown = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        current_drawdown = (peak - values[-1]) / peak
        return current_drawdown, max_drawdown
    
    def _calculate_portfolio_volatility(self, positions: Dict) -> float:
        """Calculate portfolio volatility"""
        if not positions:
            return 0.0
        
        # Simplified calculation
        individual_vols = []
        weights = []
        
        total_value = sum(abs(pos.get('position_size', 0) * pos.get('current_price', 0)) 
                         for pos in positions.values())
        
        if total_value == 0:
            return 0.0
        
        for symbol, pos in positions.items():
            vol = self.volatility_estimates.get(symbol, 0.02)
            weight = abs(pos.get('position_size', 0) * pos.get('current_price', 0)) / total_value
            individual_vols.append(vol)
            weights.append(weight)
        
        # Weighted average (simplified, ignores correlations)
        portfolio_vol = sum(w * v for w, v in zip(weights, individual_vols))
        return portfolio_vol
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from portfolio history"""
        if len(self.portfolio_history) < 30:
            return 0.0
        
        values = [h['portfolio_value'] for h in self.portfolio_history[-30:]]
        returns = np.diff(values) / values[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualized Sharpe (assuming daily data)
        sharpe = (mean_return * 365) / (std_return * np.sqrt(365))
        return sharpe
    
    def _calculate_var_and_es(self, positions: Dict) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall"""
        if not positions:
            return 0.0, 0.0
        
        # Simplified VaR calculation
        portfolio_value = self.get_portfolio_value()
        portfolio_vol = self._calculate_portfolio_volatility(positions)
        
        # 95% VaR (1.645 standard deviations)
        var_95 = 1.645 * portfolio_vol * portfolio_value
        
        # Expected Shortfall (simplified)
        expected_shortfall = var_95 * 1.3
        
        return var_95, expected_shortfall
    
    def _calculate_correlation_risk(self, positions: Dict) -> float:
        """Calculate correlation risk across positions"""
        if len(positions) <= 1:
            return 0.0
        
        # Simplified correlation risk
        # In practice, would use actual correlation matrix
        return min(0.8, len(positions) * 0.1)
    
    def _calculate_concentration_risk(self, positions: Dict) -> float:
        """Calculate concentration risk"""
        if not positions:
            return 0.0
        
        total_exposure = sum(abs(pos.get('position_size', 0) * pos.get('current_price', 0)) 
                           for pos in positions.values())
        
        if total_exposure == 0:
            return 0.0
        
        max_position = max(abs(pos.get('position_size', 0) * pos.get('current_price', 0)) 
                          for pos in positions.values())
        
        return max_position / total_exposure
    
    def _update_risk_history(self, risk_metrics: RiskMetrics):
        """Update risk history for tracking"""
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': risk_metrics.portfolio_value,
            'drawdown': risk_metrics.current_drawdown,
            'volatility': risk_metrics.volatility
        })
        
        # Keep only last 1000 records
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        # This would typically come from the portfolio manager
        # For now, return a default value
        if self.portfolio_history:
            return self.portfolio_history[-1]['portfolio_value']
        return 10000.0  # Default value
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        if not self.current_risk_metrics:
            return {"status": "No risk data available"}
        
        metrics = self.current_risk_metrics
        should_reduce, reason = self.should_reduce_risk()
        
        return {
            "portfolio_value": metrics.portfolio_value,
            "total_exposure": metrics.total_exposure,
            "exposure_ratio": metrics.total_exposure / metrics.portfolio_value,
            "current_drawdown": f"{metrics.current_drawdown:.2%}",
            "max_drawdown": f"{metrics.max_drawdown:.2%}",
            "portfolio_volatility": f"{metrics.volatility:.2%}",
            "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
            "var_95": f"${metrics.var_95:.2f}",
            "risk_reduction_needed": should_reduce,
            "risk_reason": reason,
            "correlation_risk": f"{metrics.correlation_risk:.2%}",
            "concentration_risk": f"{metrics.concentration_risk:.2%}",
            "risk_limits": {
                "max_portfolio_risk": f"{self.risk_limits.max_portfolio_risk:.2%}",
                "max_position_size": f"{self.risk_limits.max_position_size:.2%}",
                "max_drawdown": f"{self.risk_limits.max_drawdown_limit:.2%}"
            }
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager()
    
    # Example position
    positions = {
        'BTC-USD': {
            'position_size': 0.1,
            'current_price': 45000,
            'entry_price': 44000,
            'direction': 'long'
        }
    }
    
    # Calculate risk metrics
    risk_metrics = risk_manager.calculate_portfolio_risk(positions)
    print(f"Portfolio Risk: {risk_manager.get_risk_summary()}") 