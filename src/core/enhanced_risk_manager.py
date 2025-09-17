#!/usr/bin/env python3
"""
ADVANCED RISK MANAGEMENT & PORTFOLIO OPTIMIZATION v5.0
======================================================

This module implements cutting-edge risk management and portfolio optimization
features to maximize returns while minimizing risk.

Key Features:
- Dynamic risk adjustment based on market volatility
- Portfolio diversification optimization
- Advanced stop-loss and take-profit management
- Correlation-based position sizing
- Maximum drawdown protection
- Kelly Criterion position sizing
- Risk-parity portfolio allocation
"""

try:
    import numpy as np
except ImportError:
    # Fallback for numpy import issues
    print("⚠️ NumPy not available in enhanced_risk_manager, using fallback calculations")
    np = None

try:
    import pandas as pd
except ImportError:
    # Fallback for pandas import issues
    print("⚠️ Pandas not available in enhanced_risk_manager, using fallback calculations")
    pd = None
from src.core.utils.decimal_boundary_guard import safe_float
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import math

class EnhancedRiskManager:
    """Enhanced risk management system for optimal portfolio protection"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.max_portfolio_risk = 0.20  # Maximum 20% portfolio risk
        self.max_single_position = 0.10  # Maximum 10% per position
        self.max_correlation_exposure = 0.30  # Maximum 30% in correlated assets
        self.max_drawdown_limit = 0.15  # Stop trading if 15% drawdown
        
        # CRITICAL UPGRADE: Kelly × CVaR sizing parameters
        self.cvar_confidence = 0.95     # 95% CVaR
        self.kelly_risk_cap = 0.125     # Never size >12.5% equity
        
        # Risk metrics tracking
        self.position_history = {}
        self.correlation_matrix = {}
        self.volatility_cache = {}
        self.drawdown_tracker = {'peak': 0.0, 'current_dd': 0.0}
        
        self.logger.info("[RISK] Enhanced Risk Management v5.0 initialized")
        print("✅ Enhanced Risk Management System ready")
    
    def calculate_portfolio_risk(self, positions: Dict[str, float], 
                                market_data: Dict) -> float:
        """Calculate total portfolio risk considering correlations"""
        try:
            if not positions:
                return 0.0
            
            total_risk = 0.0
            symbols = list(positions.keys())
            
            # Individual position risks
            individual_risks = {}
            for symbol, position_size in positions.items():
                if symbol in market_data:
                    price_data = market_data[symbol].get('prices', [])
                    if len(price_data) >= 20:
                        volatility = self.calculate_volatility(price_data)
                        individual_risks[symbol] = position_size * volatility
                    else:
                        individual_risks[symbol] = position_size * 0.02  # Default 2% risk
            
            # Calculate correlation-adjusted risk
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i <= j:  # Avoid double counting
                        risk1 = individual_risks.get(symbol1, 0)
                        risk2 = individual_risks.get(symbol2, 0)
                        
                        if i == j:
                            # Same asset
                            total_risk += risk1 ** 2
                        else:
                            # Different assets - apply correlation
                            correlation = self.get_correlation(symbol1, symbol2, market_data)
                            total_risk += 2 * risk1 * risk2 * correlation
            
            portfolio_risk = np.sqrt(max(0, total_risk)) if np is not None else math.sqrt(max(0, total_risk))
            
            self.logger.info(f"[RISK] Portfolio risk: {portfolio_risk:.2%}")
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating portfolio risk: {e}")
            return 0.05  # Conservative default
    
    def calculate_volatility(self, price_data: List[float], periods: int = 20) -> float:
        """Calculate annualized volatility"""
        try:
            if len(price_data) < periods:
                return 0.02  # Default 2% volatility
            
            if np is not None:
                prices = np.array(price_data[-periods:])
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                # Fallback calculation
                prices = price_data[-periods:]
                returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
                mean_return = sum(returns) / len(returns) if returns else 0
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns) if returns else 0
                volatility = math.sqrt(variance) * math.sqrt(252)  # Annualized
            
            return safe_float(volatility)
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating volatility: {e}")
            return 0.02
    
    def get_correlation(self, symbol1: str, symbol2: str, 
                       market_data: Dict, periods: int = 50) -> float:
        """Calculate correlation between two assets"""
        try:
            if symbol1 == symbol2:
                return 1.0
            
            # Check cache
            pair_key = f"{min(symbol1, symbol2)}_{max(symbol1, symbol2)}"
            if pair_key in self.correlation_matrix:
                return self.correlation_matrix[pair_key]
            
            # Calculate correlation
            if (symbol1 in market_data and symbol2 in market_data):
                prices1 = market_data[symbol1].get('prices', [])
                prices2 = market_data[symbol2].get('prices', [])
                
                if len(prices1) >= periods and len(prices2) >= periods:
                    if np is not None:
                        p1 = np.array(prices1[-periods:])
                        p2 = np.array(prices2[-periods:])
                        
                        returns1 = np.diff(p1) / p1[:-1]
                        returns2 = np.diff(p2) / p2[:-1]
                        
                        correlation = np.corrcoef(returns1, returns2)[0, 1]
                        if not np.isnan(correlation):
                            self.correlation_matrix[pair_key] = correlation
                            return correlation
                    else:
                        # Fallback correlation calculation
                        p1 = prices1[-periods:]
                        p2 = prices2[-periods:]
                        
                        returns1 = [(p1[i+1] - p1[i]) / p1[i] for i in range(len(p1)-1)]
                        returns2 = [(p2[i+1] - p2[i]) / p2[i] for i in range(len(p2)-1)]
                        
                        if len(returns1) == len(returns2) and len(returns1) > 0:
                            # Simple correlation calculation
                            mean1 = sum(returns1) / len(returns1)
                            mean2 = sum(returns2) / len(returns2)
                            
                            numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(returns1, returns2))
                            denom1 = sum((r1 - mean1) ** 2 for r1 in returns1)
                            denom2 = sum((r2 - mean2) ** 2 for r2 in returns2)
                            
                            if denom1 > 0 and denom2 > 0:
                                correlation = numerator / math.sqrt(denom1 * denom2)
                                correlation = max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
                                self.correlation_matrix[pair_key] = correlation
                                return correlation
            
            # Default correlation for crypto pairs
            crypto_correlations = {
                ('BTC', 'ETH'): 0.7,
                ('BTC', 'SOL'): 0.6,
                ('ETH', 'SOL'): 0.65,
                ('BTC', 'XRP'): 0.5,
                ('ETH', 'XRP'): 0.45
            }
            
            pair = (min(symbol1, symbol2), max(symbol1, symbol2))
            return crypto_correlations.get(pair, 0.3)  # Default 30% correlation
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating correlation: {e}")
            return 0.3
    
    def calculate_kelly_position_size(self, symbol: str, win_rate: float, 
                                    avg_win: float, avg_loss: float, 
                                    account_balance: float) -> float:
        """Calculate optimal position size using Kelly Criterion with CVaR attenuation"""
        try:
            # CRITICAL FIX: Handle edge cases properly
            if win_rate <= 0 or win_rate >= 1:
                return account_balance * 0.02  # Conservative default
            
            if avg_win <= 0:
                return account_balance * 0.01  # Very conservative if no wins
            
            if avg_loss <= 0:
                # If no losses, use conservative approach
                return account_balance * 0.05  # 5% position size
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / abs(avg_loss)  # Use absolute value for safety
            p = win_rate
            q = 1 - win_rate
            
            raw_kelly = max(0, (b * p - q) / b)
            
            # CRITICAL FIX: Ensure minimum Kelly fraction for trading
            if raw_kelly <= 0:
                raw_kelly = 0.05  # Minimum 5% Kelly fraction for trading
            
            # CRITICAL UPGRADE: CVaR attenuation
            tail_risk = self.estimate_cvar(symbol)
            attenuation = 1 / (1 + tail_risk * 10)  # More tail-risk → smaller size
            f_star = min(raw_kelly * attenuation, self.kelly_risk_cap)
            
            # CRITICAL FIX: Ensure minimum position size
            f_star = max(f_star, 0.01)  # Minimum 1% position size
            
            kelly_position = account_balance * f_star
            
            self.logger.info(f"[RISK] Kelly×CVaR position for {symbol}: ${kelly_position:.2f} "
                           f"(raw_kelly: {raw_kelly:.2%}, cvar_att: {attenuation:.2f}, final: {f_star:.2%})")
            
            return kelly_position
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating Kelly position: {e}")
            return account_balance * 0.02
    
    def estimate_cvar(self, symbol: str, lookback: int = 250) -> float:
        """Estimate Conditional Value at Risk (CVaR) for a symbol"""
        try:
            # Get historical returns from cache or calculate
            returns = self.volatility_cache.get(symbol)
            if returns is None or len(returns) < lookback:
                # Fallback to default CVaR for crypto
                return 0.02  # 2% CVaR fallback
            # Calculate CVaR at specified confidence level
            if np is not None:
                sorted_r = np.sort(returns)
                cutoff = int((1 - self.cvar_confidence) * len(sorted_r))
                if cutoff > 0:
                    cvar = abs(sorted_r[:cutoff].mean())
                else:
                    cvar = 0.02  # Default fallback
            else:
                # Fallback calculation
                sorted_r = sorted(returns)
                cutoff = int((1 - self.cvar_confidence) * len(sorted_r))
                if cutoff > 0:
                    cvar = abs(sum(sorted_r[:cutoff]) / len(sorted_r[:cutoff]))
                else:
                    cvar = 0.02  # Default fallback
            self.logger.info(f"[RISK] CVaR for {symbol}: {cvar:.2%} (lookback: {len(returns)})")
            return cvar
            # fallback if main logic does not return
            return 0.02  # Default fallback
        except Exception as e:
            self.logger.error(f"[RISK] Error estimating CVaR for {symbol}: {e}")
            return 0.02  # Conservative fallback
    
    def check_position_limits(self, symbol: str, proposed_size: float, 
                            current_positions: Dict[str, float], 
                            account_balance: float) -> tuple[bool, float, str]:
        """Check if proposed position violates risk limits"""
        try:
            # Check single position limit
            position_pct = proposed_size / account_balance
            if position_pct > self.max_single_position:
                adjusted_size = account_balance * self.max_single_position
                return False, adjusted_size, f"Position too large (max {self.max_single_position:.0%})"
            
            # Check total portfolio exposure
            total_exposure = sum(current_positions.values()) + proposed_size
            portfolio_pct = total_exposure / account_balance
            if portfolio_pct > self.max_portfolio_risk:
                remaining_capacity = (self.max_portfolio_risk * account_balance) - sum(current_positions.values())
                if remaining_capacity > 0:
                    return False, remaining_capacity, f"Portfolio limit reached (max {self.max_portfolio_risk:.0%})"
                else:
                    return False, 0, "Portfolio at maximum risk"
            
            return True, proposed_size, "Position within limits"
            
        except Exception as e:
            self.logger.error(f"[RISK] Error checking position limits: {e}")
            return False, account_balance * 0.01, "Error in position check"
    
    def calculate_correlated_exposure(self, symbol: str, position_size: float, 
                                    current_positions: Dict[str, float]) -> float:
        """Calculate correlated exposure for a symbol"""
        try:
            total_correlated = 0.0
            
            for other_symbol, other_size in current_positions.items():
                if other_symbol != symbol:
                    correlation = self.get_correlation(symbol, other_symbol, {})
                    correlated_exposure = other_size * correlation
                    total_correlated += correlated_exposure
            
            return total_correlated + position_size
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating correlated exposure: {e}")
            return position_size
    
    def update_drawdown_tracker(self, current_balance: float, peak_balance: Optional[float] = None):
        """Update drawdown tracking"""
        try:
            if peak_balance is None:
                peak_balance = self.drawdown_tracker['peak']
            
            # Update peak if current balance is higher
            if current_balance > peak_balance:
                self.drawdown_tracker['peak'] = current_balance
                peak_balance = current_balance
            
            # Calculate current drawdown
            if peak_balance > 0:
                current_dd = (peak_balance - current_balance) / peak_balance
                self.drawdown_tracker['current_dd'] = current_dd
                
                # Check if drawdown limit exceeded
                if current_dd > self.max_drawdown_limit:
                    self.logger.warning(f"[RISK] Maximum drawdown exceeded: {current_dd:.2%}")
                    return False  # Should stop trading
            
            return True
            
        except Exception as e:
            self.logger.error(f"[RISK] Error updating drawdown tracker: {e}")
            return True
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, 
                                  position_type: str, market_data: Dict) -> float:
        """Calculate dynamic stop loss based on volatility"""
        try:
            # Get volatility for this symbol
            volatility = 0.02  # Default 2%
            if symbol in market_data:
                price_data = market_data[symbol].get('prices', [])
                if len(price_data) >= 20:
                    volatility = self.calculate_volatility(price_data)
            
            # Dynamic stop loss based on volatility
            # Higher volatility = wider stop loss
            volatility_multiplier = min(3.0, max(1.0, volatility * 100))  # 1x to 3x
            base_stop_loss = 0.005  # 0.5% base stop loss
            
            dynamic_stop_loss = base_stop_loss * volatility_multiplier
            
            # Cap at reasonable limits
            dynamic_stop_loss = min(dynamic_stop_loss, 0.03)  # Max 3%
            dynamic_stop_loss = max(dynamic_stop_loss, 0.002)  # Min 0.2%
            
            self.logger.info(f"[RISK] Dynamic stop loss for {symbol}: {dynamic_stop_loss:.2%} "
                           f"(volatility: {volatility:.2%})")
            
            return dynamic_stop_loss
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating dynamic stop loss: {e}")
            return 0.01  # Default 1% stop loss
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            return {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_single_position': self.max_single_position,
                'max_correlation_exposure': self.max_correlation_exposure,
                'max_drawdown_limit': self.max_drawdown_limit,
                'current_drawdown': self.drawdown_tracker['current_dd'],
                'peak_balance': self.drawdown_tracker['peak'],
                'correlation_pairs': len(self.correlation_matrix),
                'volatility_cache_size': len(self.volatility_cache),
                'position_history_size': len(self.position_history)
            }
        except Exception as e:
            self.logger.error(f"[RISK] Error getting risk summary: {e}")
            return {}

class PortfolioOptimizer:
    """Portfolio optimization for maximum risk-adjusted returns"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.rebalancing_threshold = 0.05  # 5% threshold for rebalancing
        self.max_positions = 5  # Maximum number of positions
        
        self.logger.info("[PORTFOLIO] Portfolio Optimizer initialized")
    
    def optimize_portfolio_allocation(self, available_symbols: List[str], 
                                    market_data: Dict, account_balance: float,
                                    risk_manager: EnhancedRiskManager) -> Dict[str, float]:
        """Optimize portfolio allocation for maximum risk-adjusted returns"""
        try:
            if not available_symbols:
                return {}
            
            # Calculate volatilities for all symbols
            volatilities = {}
            for symbol in available_symbols:
                if symbol in market_data:
                    price_data = market_data[symbol].get('prices', [])
                    if len(price_data) >= 20:
                        volatilities[symbol] = risk_manager.calculate_volatility(price_data)
                    else:
                        volatilities[symbol] = 0.02  # Default volatility
            
            # Use risk parity allocation
            # Create identity correlation matrix for simplicity
            n_symbols = len(volatilities)
            identity_matrix = [[1.0 if i == j else 0.0 for j in range(n_symbols)] for i in range(n_symbols)]
            
            allocation = self.calculate_risk_parity_weights(
                list(volatilities.keys()), 
                volatilities, 
                identity_matrix  # Identity correlation matrix for simplicity
            )
            
            # Convert to position sizes
            positions = {}
            for symbol, weight in allocation.items():
                positions[symbol] = account_balance * weight
            
            self.logger.info(f"[PORTFOLIO] Optimized allocation: {positions}")
            return positions
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error optimizing allocation: {e}")
            return {}
    
    def calculate_risk_parity_weights(self, symbols: List[str], 
                                    volatilities: Dict[str, float],
                                    correlation_matrix: List[List[float]]) -> Dict[str, float]:
        """Calculate risk parity weights"""
        try:
            if not symbols:
                return {}
            
            # Simple equal risk contribution (simplified risk parity)
            n_symbols = len(symbols)
            equal_weight = 1.0 / n_symbols
            
            weights = {}
            for symbol in symbols:
                weights[symbol] = equal_weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for symbol in weights:
                    weights[symbol] /= total_weight
            
            self.logger.info(f"[PORTFOLIO] Risk parity weights: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error calculating risk parity weights: {e}")
            return {}
    
    def check_rebalancing_needed(self, current_positions: Dict[str, float], 
                               target_allocation: Dict[str, float],
                               account_balance: float) -> bool:
        """Check if portfolio rebalancing is needed"""
        try:
            if not current_positions or not target_allocation:
                return False
            
            total_current = sum(current_positions.values())
            if total_current == 0:
                return False
            
            # Check if any position deviates significantly from target
            for symbol, current_size in current_positions.items():
                if symbol in target_allocation:
                    target_size = target_allocation[symbol]
                    current_pct = current_size / total_current
                    target_pct = target_size / account_balance
                    
                    deviation = abs(current_pct - target_pct)
                    if deviation > self.rebalancing_threshold:
                        self.logger.info(f"[PORTFOLIO] Rebalancing needed: {symbol} deviation {deviation:.2%}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error checking rebalancing: {e}")
            return False 