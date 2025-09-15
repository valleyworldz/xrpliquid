#!/usr/bin/env python3
"""
Optimized Profit Maximizer - Tier 2 Component
=============================================

This module implements advanced profit optimization strategies including
dynamic profit target calculation, compound profit strategies, and
multi-timeframe profit analysis for maximum profitability.

Features:
- Dynamic profit target calculation based on market conditions
- Compound profit strategies with reinvestment logic
- Multi-timeframe profit analysis
- Risk-adjusted profit optimization
- Market regime-aware profit targets
- Volatility-based profit scaling
- Position sizing optimization for profit maximization
"""

import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import logging

class OptimizedProfitMaximizer:
    """Advanced profit optimization system for maximum profitability"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Profit optimization parameters
        self.base_profit_target = 0.02  # 2% base profit target
        self.max_profit_target = 0.15   # 15% maximum profit target
        self.min_profit_target = 0.005  # 0.5% minimum profit target
        
        # Market condition multipliers
        self.volatility_multiplier = 1.5
        self.trend_multiplier = 1.3
        self.volume_multiplier = 1.2
        
        # Compound profit parameters
        self.compound_threshold = 0.05  # 5% profit threshold for compounding
        self.compound_ratio = 0.5       # 50% of profits reinvested
        self.max_compound_cycles = 3    # Maximum compound cycles
        
        # Performance tracking
        self.profit_history = deque(maxlen=1000)
        self.target_history = deque(maxlen=1000)
        self.compound_history = deque(maxlen=100)
        
        # Market regime tracking
        self.regime_profit_multipliers = {
            'trending': 1.5,
            'ranging': 0.8,
            'volatile': 1.3,
            'stable': 0.9
        }
        
        # Time-based profit optimization
        self.time_multipliers = {
            'short_term': 1.2,   # < 1 hour
            'medium_term': 1.0,  # 1-4 hours
            'long_term': 0.8     # > 4 hours
        }
        
        self.logger.info("[PROFIT] Optimized Profit Maximizer initialized")
    
    def calculate_dynamic_profit_target(self, symbol: str, entry_price: float,
                                      market_data: Dict[str, Any], 
                                      position_data: Dict[str, Any]) -> float:
        """Calculate dynamic profit target based on market conditions"""
        try:
            # Base profit target
            profit_target = self.base_profit_target
            
            # Market volatility adjustment
            volatility = market_data.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                profit_target *= self.volatility_multiplier
            elif volatility < 0.01:  # Low volatility
                profit_target *= 0.8
            
            # Trend strength adjustment
            trend_strength = market_data.get('trend_strength', 0.0)
            if abs(trend_strength) > 0.7:  # Strong trend
                profit_target *= self.trend_multiplier
            elif abs(trend_strength) < 0.3:  # Weak trend
                profit_target *= 0.9
            
            # Volume adjustment
            volume_change = market_data.get('volume_change_1h', 0.0)
            if volume_change > 0.5:  # High volume
                profit_target *= self.volume_multiplier
            elif volume_change < -0.3:  # Low volume
                profit_target *= 0.9
            
            # Market regime adjustment
            regime = market_data.get('market_regime', 'stable')
            regime_multiplier = self.regime_profit_multipliers.get(regime, 1.0)
            profit_target *= regime_multiplier
            
            # Position size adjustment
            position_size = position_data.get('position_size', 0.0)
            max_position = position_data.get('max_position', 100.0)
            position_ratio = abs(position_size) / max_position if max_position > 0 else 0.0
            
            if position_ratio > 0.7:  # Large position
                profit_target *= 1.2  # Higher target for larger positions
            elif position_ratio < 0.3:  # Small position
                profit_target *= 0.9  # Lower target for smaller positions
            
            # Time-based adjustment
            position_age = position_data.get('position_age_minutes', 0)
            if position_age < 60:  # Short term
                profit_target *= self.time_multipliers['short_term']
            elif position_age > 240:  # Long term
                profit_target *= self.time_multipliers['long_term']
            else:  # Medium term
                profit_target *= self.time_multipliers['medium_term']
            
            # Technical indicator adjustments
            rsi = market_data.get('rsi', 50.0)
            if rsi < 30:  # Oversold
                profit_target *= 1.1  # Higher target for oversold conditions
            elif rsi > 70:  # Overbought
                profit_target *= 0.9  # Lower target for overbought conditions
            
            # MACD signal adjustment
            macd_signal = market_data.get('macd_signal', 0.0)
            if macd_signal > 0:  # Bullish MACD
                profit_target *= 1.05
            elif macd_signal < 0:  # Bearish MACD
                profit_target *= 0.95
            
            # Bollinger Bands position adjustment
            bb_position = market_data.get('bollinger_position', 0.5)
            if bb_position < 0.2:  # Near lower band
                profit_target *= 1.1
            elif bb_position > 0.8:  # Near upper band
                profit_target *= 0.9
            
            # Ensure profit target is within bounds
            profit_target = max(self.min_profit_target, min(profit_target, self.max_profit_target))
            
            # Store for history
            self.target_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'profit_target': profit_target,
                'entry_price': entry_price,
                'market_conditions': {
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'volume_change': volume_change,
                    'regime': regime,
                    'rsi': rsi,
                    'macd_signal': macd_signal,
                    'bb_position': bb_position
                }
            })
            
            self.logger.info(f"[PROFIT] Dynamic profit target for {symbol}: {profit_target:.2%}")
            
            return profit_target
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error calculating dynamic profit target: {e}")
            return self.base_profit_target
    
    def calculate_compound_profit_strategy(self, current_balance: float, 
                                         initial_balance: float, 
                                         profit_history: List[float]) -> Dict[str, Any]:
        """Calculate compound profit strategy with reinvestment logic"""
        try:
            # Calculate current profit
            current_profit = current_balance - initial_balance
            profit_percentage = current_profit / initial_balance if initial_balance > 0 else 0.0
            
            # Determine if we should compound
            should_compound = profit_percentage >= self.compound_threshold
            
            # Calculate compound amount
            compound_amount = 0.0
            if should_compound:
                compound_amount = current_profit * self.compound_ratio
                
                # Cap compound amount
                max_compound = initial_balance * 0.1  # Max 10% of initial balance
                compound_amount = min(compound_amount, max_compound)
            
            # Calculate compound cycles
            compound_cycles = len([p for p in profit_history if p >= self.compound_threshold])
            remaining_cycles = max(0, self.max_compound_cycles - compound_cycles)
            
            # Calculate projected compound growth
            projected_balance = current_balance
            if should_compound and remaining_cycles > 0:
                # Simple compound growth projection
                growth_rate = 1.0 + (self.base_profit_target * self.compound_ratio)
                projected_balance = current_balance * (growth_rate ** remaining_cycles)
            
            # Calculate compound efficiency
            compound_efficiency = 0.0
            if len(profit_history) > 0:
                avg_profit = sum(profit_history) / len(profit_history)
                compound_efficiency = avg_profit / self.base_profit_target if self.base_profit_target > 0 else 0.0
            
            strategy = {
                'should_compound': should_compound,
                'compound_amount': compound_amount,
                'compound_cycles': compound_cycles,
                'remaining_cycles': remaining_cycles,
                'projected_balance': projected_balance,
                'compound_efficiency': compound_efficiency,
                'profit_percentage': profit_percentage,
                'current_profit': current_profit
            }
            
            # Store compound history
            if should_compound:
                self.compound_history.append({
                    'timestamp': datetime.now(),
                    'compound_amount': compound_amount,
                    'profit_percentage': profit_percentage,
                    'cycles': compound_cycles
                })
            
            self.logger.info(f"[PROFIT] Compound strategy: {should_compound}, "
                           f"amount: ${compound_amount:.2f}, cycles: {compound_cycles}")
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error calculating compound strategy: {e}")
            return {
                'should_compound': False,
                'compound_amount': 0.0,
                'compound_cycles': 0,
                'remaining_cycles': self.max_compound_cycles,
                'projected_balance': current_balance,
                'compound_efficiency': 0.0,
                'profit_percentage': 0.0,
                'current_profit': 0.0
            }
    
    def optimize_position_size_for_profit(self, available_balance: float,
                                        market_data: Dict[str, Any],
                                        risk_tolerance: float = 0.02) -> float:
        """Optimize position size for maximum profit potential"""
        try:
            # Base position size using Kelly Criterion
            win_rate = market_data.get('win_rate', 0.5)
            avg_win = market_data.get('avg_win', 0.02)
            avg_loss = market_data.get('avg_loss', 0.01)
            
            # Kelly formula: f = (bp - q) / b
            if avg_loss > 0:
                b = avg_win / avg_loss
                p = win_rate
                q = 1 - p
                kelly_fraction = max(0, (b * p - q) / b)
            else:
                kelly_fraction = 0.02  # Default 2%
            
            # Adjust for risk tolerance
            kelly_fraction = min(kelly_fraction, risk_tolerance)
            
            # Market condition adjustments
            volatility = market_data.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                kelly_fraction *= 0.8  # Reduce size in high volatility
            elif volatility < 0.01:  # Low volatility
                kelly_fraction *= 1.2  # Increase size in low volatility
            
            # Trend strength adjustment
            trend_strength = market_data.get('trend_strength', 0.0)
            if abs(trend_strength) > 0.7:  # Strong trend
                kelly_fraction *= 1.1  # Increase size in strong trends
            elif abs(trend_strength) < 0.3:  # Weak trend
                kelly_fraction *= 0.9  # Decrease size in weak trends
            
            # Volume adjustment
            volume_change = market_data.get('volume_change_1h', 0.0)
            if volume_change > 0.5:  # High volume
                kelly_fraction *= 1.05  # Slight increase for high volume
            elif volume_change < -0.3:  # Low volume
                kelly_fraction *= 0.95  # Slight decrease for low volume
            
            # Technical indicator adjustments
            rsi = market_data.get('rsi', 50.0)
            if rsi < 30 or rsi > 70:  # Extreme RSI values
                kelly_fraction *= 1.1  # Increase size for extreme conditions
            
            # Ensure minimum and maximum bounds
            kelly_fraction = max(0.01, min(kelly_fraction, 0.2))  # 1% to 20%
            
            # Calculate position size
            position_size = available_balance * kelly_fraction
            
            self.logger.info(f"[PROFIT] Optimized position size: ${position_size:.2f} "
                           f"(Kelly: {kelly_fraction:.2%}, balance: ${available_balance:.2f})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error optimizing position size: {e}")
            return available_balance * 0.02  # Default 2%
    
    def calculate_multi_timeframe_profit_target(self, symbol: str, 
                                              price_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate profit targets for multiple timeframes"""
        try:
            profit_targets = {}
            
            # Calculate profit targets for different timeframes
            timeframes = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '1h': 3600,
                '4h': 14400,
                '1d': 86400
            }
            
            for timeframe, seconds in timeframes.items():
                if timeframe in price_data and len(price_data[timeframe]) >= 2:
                    prices = price_data[timeframe]
                    
                    # Calculate volatility for this timeframe
                    returns = []
                    for i in range(1, len(prices)):
                        if prices[i-1] > 0:
                            returns.append((prices[i] - prices[i-1]) / prices[i-1])
                    
                    if returns:
                        volatility = math.sqrt(sum(r*r for r in returns) / len(returns))
                        
                        # Base profit target based on volatility
                        base_target = volatility * 2  # 2x volatility
                        
                        # Adjust for timeframe
                        if timeframe in ['1m', '5m']:
                            base_target *= 0.8  # Lower targets for short timeframes
                        elif timeframe in ['4h', '1d']:
                            base_target *= 1.2  # Higher targets for long timeframes
                        
                        # Ensure bounds
                        base_target = max(self.min_profit_target, min(base_target, self.max_profit_target))
                        profit_targets[timeframe] = base_target
                    else:
                        profit_targets[timeframe] = self.base_profit_target
                else:
                    profit_targets[timeframe] = self.base_profit_target
            
            self.logger.info(f"[PROFIT] Multi-timeframe targets for {symbol}: {profit_targets}")
            
            return profit_targets
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error calculating multi-timeframe targets: {e}")
            return {tf: self.base_profit_target for tf in ['1m', '5m', '15m', '1h', '4h', '1d']}
    
    def calculate_risk_adjusted_profit_target(self, symbol: str, entry_price: float,
                                            market_data: Dict[str, Any],
                                            position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk-adjusted profit target with stop loss optimization"""
        try:
            # Calculate dynamic profit target
            profit_target = self.calculate_dynamic_profit_target(symbol, entry_price, market_data, position_data)
            
            # Calculate optimal stop loss
            volatility = market_data.get('volatility', 0.02)
            stop_loss = volatility * 1.5  # 1.5x volatility for stop loss
            stop_loss = max(0.005, min(stop_loss, 0.05))  # 0.5% to 5%
            
            # Calculate risk-reward ratio
            risk_reward_ratio = profit_target / stop_loss if stop_loss > 0 else 0.0
            
            # Adjust profit target based on risk-reward ratio
            if risk_reward_ratio < 1.5:  # Poor risk-reward
                profit_target *= 1.2  # Increase target
                stop_loss *= 0.8  # Tighten stop loss
            elif risk_reward_ratio > 3.0:  # Excellent risk-reward
                profit_target *= 0.9  # Slightly reduce target
                stop_loss *= 1.1  # Slightly widen stop loss
            
            # Position size adjustment based on risk
            position_size = position_data.get('position_size', 0.0)
            max_position = position_data.get('max_position', 100.0)
            
            if position_size > max_position * 0.7:  # Large position
                # More conservative targets for large positions
                profit_target *= 0.9
                stop_loss *= 0.8
            elif position_size < max_position * 0.3:  # Small position
                # More aggressive targets for small positions
                profit_target *= 1.1
                stop_loss *= 1.2
            
            # Market regime adjustments
            regime = market_data.get('market_regime', 'stable')
            if regime == 'volatile':
                # More conservative in volatile markets
                profit_target *= 0.9
                stop_loss *= 0.7
            elif regime == 'trending':
                # More aggressive in trending markets
                profit_target *= 1.1
                stop_loss *= 1.1
            
            # Ensure final bounds
            profit_target = max(self.min_profit_target, min(profit_target, self.max_profit_target))
            stop_loss = max(0.005, min(stop_loss, 0.05))
            
            # Recalculate final risk-reward ratio
            final_risk_reward = profit_target / stop_loss if stop_loss > 0 else 0.0
            
            result = {
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'risk_reward_ratio': final_risk_reward,
                'entry_price': entry_price,
                'target_price': entry_price * (1 + profit_target),
                'stop_price': entry_price * (1 - stop_loss),
                'position_size': position_size,
                'market_regime': regime,
                'volatility': volatility
            }
            
            self.logger.info(f"[PROFIT] Risk-adjusted targets for {symbol}: "
                           f"profit={profit_target:.2%}, stop={stop_loss:.2%}, "
                           f"R:R={final_risk_reward:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error calculating risk-adjusted targets: {e}")
            return {
                'profit_target': self.base_profit_target,
                'stop_loss': 0.01,
                'risk_reward_ratio': 2.0,
                'entry_price': entry_price,
                'target_price': entry_price * (1 + self.base_profit_target),
                'stop_price': entry_price * (1 - 0.01),
                'position_size': 0.0,
                'market_regime': 'stable',
                'volatility': 0.02
            }
    
    def record_profit_outcome(self, symbol: str, profit_target: float, 
                            actual_profit: float, trade_duration: float,
                            market_conditions: Dict[str, Any]):
        """Record profit outcome for learning and optimization"""
        try:
            # Calculate profit efficiency
            profit_efficiency = actual_profit / profit_target if profit_target > 0 else 0.0
            
            # Determine if target was achieved
            target_achieved = actual_profit >= profit_target
            
            # Calculate time efficiency
            time_efficiency = 1.0 / (1.0 + trade_duration / 3600)  # Normalize to hours
            
            record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'profit_target': profit_target,
                'actual_profit': actual_profit,
                'profit_efficiency': profit_efficiency,
                'target_achieved': target_achieved,
                'trade_duration': trade_duration,
                'time_efficiency': time_efficiency,
                'market_conditions': market_conditions
            }
            
            self.profit_history.append(record)
            
            self.logger.info(f"[PROFIT] Recorded outcome for {symbol}: "
                           f"target={profit_target:.2%}, actual={actual_profit:.2%}, "
                           f"efficiency={profit_efficiency:.2f}")
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error recording profit outcome: {e}")
    
    def get_profit_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive profit optimization summary"""
        try:
            if not self.profit_history:
                return {
                    'total_trades': 0,
                    'target_achievement_rate': 0.0,
                    'average_profit_efficiency': 0.0,
                    'average_time_efficiency': 0.0,
                    'best_profit_target': 0.0,
                    'worst_profit_target': 0.0,
                    'compound_cycles': 0,
                    'total_compound_amount': 0.0
                }
            
            # Calculate statistics
            total_trades = len(self.profit_history)
            target_achieved = sum(1 for record in self.profit_history if record['target_achieved'])
            target_achievement_rate = target_achieved / total_trades if total_trades > 0 else 0.0
            
            profit_efficiencies = [record['profit_efficiency'] for record in self.profit_history]
            time_efficiencies = [record['time_efficiency'] for record in self.profit_history]
            
            avg_profit_efficiency = sum(profit_efficiencies) / len(profit_efficiencies) if profit_efficiencies else 0.0
            avg_time_efficiency = sum(time_efficiencies) / len(time_efficiencies) if time_efficiencies else 0.0
            
            profit_targets = [record['profit_target'] for record in self.profit_history]
            best_profit_target = max(profit_targets) if profit_targets else 0.0
            worst_profit_target = min(profit_targets) if profit_targets else 0.0
            
            # Compound statistics
            compound_cycles = len(self.compound_history)
            total_compound_amount = sum(record['compound_amount'] for record in self.compound_history)
            
            # Recent performance (last 20 trades)
            recent_trades = list(self.profit_history)[-20:]
            recent_achievement_rate = sum(1 for record in recent_trades if record['target_achieved']) / len(recent_trades) if recent_trades else 0.0
            
            # Market condition analysis
            regime_performance = defaultdict(list)
            for record in self.profit_history:
                regime = record['market_conditions'].get('regime', 'unknown')
                regime_performance[regime].append(record['profit_efficiency'])
            
            regime_avg_efficiency = {}
            for regime, efficiencies in regime_performance.items():
                regime_avg_efficiency[regime] = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
            
            summary = {
                'total_trades': total_trades,
                'target_achievement_rate': target_achievement_rate,
                'recent_achievement_rate': recent_achievement_rate,
                'average_profit_efficiency': avg_profit_efficiency,
                'average_time_efficiency': avg_time_efficiency,
                'best_profit_target': best_profit_target,
                'worst_profit_target': worst_profit_target,
                'compound_cycles': compound_cycles,
                'total_compound_amount': total_compound_amount,
                'regime_performance': dict(regime_avg_efficiency),
                'learning_progress': recent_achievement_rate - target_achievement_rate
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error getting optimization summary: {e}")
            return {}
    
    def get_optimal_profit_strategy(self, symbol: str, market_data: Dict[str, Any],
                                  position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal profit strategy combining all optimization methods"""
        try:
            entry_price = position_data.get('entry_price', market_data.get('current_price', 0.0))
            
            # Calculate all profit targets
            dynamic_target = self.calculate_dynamic_profit_target(symbol, entry_price, market_data, position_data)
            risk_adjusted = self.calculate_risk_adjusted_profit_target(symbol, entry_price, market_data, position_data)
            multi_timeframe = self.calculate_multi_timeframe_profit_target(symbol, {})
            
            # Combine targets using weighted average
            weights = {
                'dynamic': 0.4,
                'risk_adjusted': 0.4,
                'multi_timeframe': 0.2
            }
            
            # Use 1h timeframe for multi-timeframe target
            mtf_target = multi_timeframe.get('1h', self.base_profit_target)
            
            optimal_target = (
                dynamic_target * weights['dynamic'] +
                risk_adjusted['profit_target'] * weights['risk_adjusted'] +
                mtf_target * weights['multi_timeframe']
            )
            
            # Ensure bounds
            optimal_target = max(self.min_profit_target, min(optimal_target, self.max_profit_target))
            
            # Calculate optimal position size
            available_balance = position_data.get('available_balance', 100.0)
            optimal_position_size = self.optimize_position_size_for_profit(available_balance, market_data)
            
            # Get compound strategy
            current_balance = position_data.get('current_balance', available_balance)
            initial_balance = position_data.get('initial_balance', available_balance)
            profit_history = [record['actual_profit'] for record in self.profit_history[-10:]]  # Last 10 trades
            
            compound_strategy = self.calculate_compound_profit_strategy(
                current_balance, initial_balance, profit_history
            )
            
            strategy = {
                'optimal_profit_target': optimal_target,
                'optimal_position_size': optimal_position_size,
                'dynamic_target': dynamic_target,
                'risk_adjusted_target': risk_adjusted['profit_target'],
                'multi_timeframe_target': mtf_target,
                'stop_loss': risk_adjusted['stop_loss'],
                'risk_reward_ratio': risk_adjusted['risk_reward_ratio'],
                'compound_strategy': compound_strategy,
                'target_price': entry_price * (1 + optimal_target),
                'stop_price': entry_price * (1 - risk_adjusted['stop_loss']),
                'confidence_score': self._calculate_strategy_confidence(market_data, position_data)
            }
            
            self.logger.info(f"[PROFIT] Optimal strategy for {symbol}: "
                           f"target={optimal_target:.2%}, size=${optimal_position_size:.2f}, "
                           f"confidence={strategy['confidence_score']:.2f}")
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error getting optimal strategy: {e}")
            return {
                'optimal_profit_target': self.base_profit_target,
                'optimal_position_size': 0.0,
                'dynamic_target': self.base_profit_target,
                'risk_adjusted_target': self.base_profit_target,
                'multi_timeframe_target': self.base_profit_target,
                'stop_loss': 0.01,
                'risk_reward_ratio': 2.0,
                'compound_strategy': {},
                'target_price': 0.0,
                'stop_price': 0.0,
                'confidence_score': 0.0
            }
    
    def _calculate_strategy_confidence(self, market_data: Dict[str, Any], 
                                     position_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the profit strategy"""
        try:
            confidence = 0.5  # Base confidence
            
            # Market condition confidence
            volatility = market_data.get('volatility', 0.02)
            if 0.01 <= volatility <= 0.04:  # Optimal volatility range
                confidence += 0.2
            elif volatility > 0.08:  # Too volatile
                confidence -= 0.1
            
            # Trend strength confidence
            trend_strength = market_data.get('trend_strength', 0.0)
            if abs(trend_strength) > 0.6:  # Strong trend
                confidence += 0.15
            elif abs(trend_strength) < 0.2:  # Weak trend
                confidence -= 0.1
            
            # Technical indicator confidence
            rsi = market_data.get('rsi', 50.0)
            if 30 <= rsi <= 70:  # Normal RSI range
                confidence += 0.1
            elif rsi < 20 or rsi > 80:  # Extreme RSI
                confidence -= 0.05
            
            # Volume confidence
            volume_change = market_data.get('volume_change_1h', 0.0)
            if volume_change > 0:  # Increasing volume
                confidence += 0.1
            elif volume_change < -0.5:  # Decreasing volume
                confidence -= 0.05
            
            # Position size confidence
            position_size = position_data.get('position_size', 0.0)
            max_position = position_data.get('max_position', 100.0)
            if position_size < max_position * 0.5:  # Conservative position
                confidence += 0.1
            elif position_size > max_position * 0.8:  # Large position
                confidence -= 0.1
            
            # Historical performance confidence
            if self.profit_history:
                recent_efficiency = [record['profit_efficiency'] for record in self.profit_history[-5:]]
                avg_recent_efficiency = sum(recent_efficiency) / len(recent_efficiency) if recent_efficiency else 0.0
                
                if avg_recent_efficiency > 1.0:  # Above target performance
                    confidence += 0.1
                elif avg_recent_efficiency < 0.5:  # Below target performance
                    confidence -= 0.1
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"[PROFIT] Error calculating strategy confidence: {e}")
            return 0.5 