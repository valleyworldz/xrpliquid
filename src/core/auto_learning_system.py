#!/usr/bin/env python3
"""
🧠 AUTO-LEARNING & ADAPTATION SYSTEM
Continuously learns from trade patterns and optimizes performance automatically

Features:
- Auto-learning from trade patterns and market behavior
- Auto-optimization of parameters based on performance history
- Auto-detection of optimal trading times and market conditions
- Self-improving algorithms that get better over time
"""

import json
import time
try:
    import numpy as np
except ImportError:
    # Fallback for numpy import issues
    print("⚠️ NumPy not available in auto_learning_system, using fallback calculations")
    np = None
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import statistics
import logging
from pathlib import Path

@dataclass
class TradePattern:
    """Represents a learned trade pattern"""
    timestamp: datetime
    token: str
    market_conditions: Dict[str, float]
    parameters_used: Dict[str, Any]
    outcome: Dict[str, float]  # profit, success_rate, execution_time
    market_regime: str  # bull, bear, sideways, volatile

class AutoLearningSystem:
    """
    🧠 Advanced Auto-Learning & Adaptation System
    
    Continuously learns from trade patterns and automatically optimizes
    trading parameters for maximum performance and profitability.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.trade_patterns = []
        self.learned_parameters = {}
        self.performance_history = []
        self.optimal_conditions = {}
        self.learning_rate = 0.1
        self.min_samples_for_learning = 10
        
        # Auto-optimization tracking
        self.parameter_performance = {
            'position_percentage': {},
            'profit_target': {},
            'timeout_seconds': {},
            'token_preference': {},
            'market_timing': {}
        }
        
        # Market condition learning
        self.market_regimes = {
            'bull': {'volatility': (0.0, 0.02), 'trend': (0.01, 1.0)},
            'bear': {'volatility': (0.0, 0.02), 'trend': (-1.0, -0.01)},
            'sideways': {'volatility': (0.0, 0.01), 'trend': (-0.005, 0.005)},
            'volatile': {'volatility': (0.02, 1.0), 'trend': (-1.0, 1.0)}
        }
        
        # Time-based learning
        self.hourly_performance = {hour: [] for hour in range(24)}
        self.daily_performance = {day: [] for day in range(7)}
        
        # CRITICAL UPGRADE: Enhanced learning capabilities
        self.win_rate_threshold = 0.4  # 40% minimum win rate for learning
        self.profit_threshold = 0.001  # 0.1% minimum profit for positive learning
        self.learning_enabled = True
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(hours=1)  # Optimize every hour
        
        # Load existing learned parameters
        self.load_learned_parameters()
        
        self.logger.info("[LEARNING] Auto-Learning & Adaptation System initialized")
        print("🧠 Auto-Learning & Adaptation System initialized")
        print("✅ Ready to learn from trade patterns and optimize performance")
    
    def record_trade_pattern(self, trade_data: Dict[str, Any]) -> None:
        """Record a trade pattern for learning"""
        try:
            pattern = TradePattern(
                timestamp=datetime.now(),
                token=trade_data.get('token', 'UNKNOWN'),
                market_conditions={
                    'volatility': trade_data.get('volatility', 0.0),
                    'spread': trade_data.get('spread', 0.0),
                    'volume': trade_data.get('volume', 0.0),
                    'price_trend': trade_data.get('price_trend', 0.0)
                },
                parameters_used={
                    'position_percentage': trade_data.get('position_percentage', 0.0),
                    'profit_target': trade_data.get('profit_target', 0.0),
                    'timeout_seconds': trade_data.get('timeout_seconds', 0),
                    'token': trade_data.get('token', 'UNKNOWN')
                },
                outcome={
                    'profit': trade_data.get('profit', 0.0),
                    'success': trade_data.get('success', False),
                    'execution_time': trade_data.get('execution_time', 0.0),
                    'profit_ratio': trade_data.get('profit_ratio', 0.0)
                },
                market_regime=self.detect_market_regime(trade_data)
            )
            
            self.trade_patterns.append(pattern)
            self.update_performance_tracking(pattern)
            
            # CRITICAL UPGRADE: Auto-learn if we have enough samples and enough time has passed
            if (len(self.trade_patterns) >= self.min_samples_for_learning and 
                datetime.now() - self.last_optimization > self.optimization_interval):
                self.auto_optimize_parameters()
            
            self.logger.info(f"[LEARNING] Trade pattern recorded: {pattern.token} | "
                           f"Profit: ${pattern.outcome['profit']:.4f} | Regime: {pattern.market_regime}")
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error recording trade pattern: {e}")
    
    def detect_market_regime(self, trade_data: Dict[str, Any]) -> str:
        """Auto-detect current market regime"""
        try:
            volatility = trade_data.get('volatility', 0.0)
            trend = trade_data.get('price_trend', 0.0)
            
            # High volatility = volatile market
            if volatility > 0.02:
                return 'volatile'
            
            # Low volatility with trend
            if abs(trend) < 0.005:
                return 'sideways'
            elif trend > 0.01:
                return 'bull'
            elif trend < -0.01:
                return 'bear'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"[LEARNING] Error detecting market regime: {e}")
            return 'sideways'
    
    def update_performance_tracking(self, pattern: TradePattern) -> None:
        """Update performance tracking for auto-optimization"""
        try:
            # Track parameter performance
            params = pattern.parameters_used
            outcome = pattern.outcome
            
            # Position percentage performance
            pos_pct = params.get('position_percentage', 0.0)
            pos_key = f"{pos_pct:.1f}"
            if pos_key not in self.parameter_performance['position_percentage']:
                self.parameter_performance['position_percentage'][pos_key] = []
            self.parameter_performance['position_percentage'][pos_key].append(outcome['profit'])
            
            # Profit target performance
            profit_target = params.get('profit_target', 0.0)
            target_key = f"{profit_target:.3f}"
            if target_key not in self.parameter_performance['profit_target']:
                self.parameter_performance['profit_target'][target_key] = []
            self.parameter_performance['profit_target'][target_key].append(outcome['profit'])
            
            # Token performance
            token = params.get('token', 'UNKNOWN')
            if token not in self.parameter_performance['token_preference']:
                self.parameter_performance['token_preference'][token] = []
            self.parameter_performance['token_preference'][token].append(outcome['profit'])
            
            # Time-based performance
            hour = pattern.timestamp.hour
            day = pattern.timestamp.weekday()
            self.hourly_performance[hour].append(outcome['profit'])
            self.daily_performance[day].append(outcome['profit'])
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error updating performance tracking: {e}")
    
    def auto_optimize_parameters(self) -> Dict[str, Any]:
        """Auto-optimize parameters based on learned patterns"""
        try:
            if not self.learning_enabled:
                return {}
            
            optimized_params = {}
            
            # Optimize position percentage
            best_position = self.find_optimal_parameter('position_percentage')
            if best_position:
                optimized_params['position_percentage'] = float(best_position)
            
            # Optimize profit target
            best_profit_target = self.find_optimal_parameter('profit_target')
            if best_profit_target:
                optimized_params['profit_target'] = float(best_profit_target)
            
            # Optimize token preference
            best_token = self.find_optimal_parameter('token_preference')
            if best_token:
                optimized_params['preferred_token'] = best_token
            
            # Optimize trading times
            optimal_hours = self.find_optimal_trading_hours()
            if optimal_hours:
                optimized_params['optimal_hours'] = optimal_hours
            
            # Store learned parameters
            self.learned_parameters.update(optimized_params)
            self.last_optimization = datetime.now()
            
            # Save learned parameters to file
            self.save_learned_parameters()
            
            self.logger.info(f"[LEARNING] AUTO-OPTIMIZATION COMPLETE:")
            for param, value in optimized_params.items():
                self.logger.info(f"   📊 {param}: {value}")
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error in auto-optimization: {e}")
            return {}
    
    def find_optimal_parameter(self, param_type: str) -> Optional[Any]:
        """Find the optimal value for a specific parameter"""
        try:
            if param_type not in self.parameter_performance:
                return None
            
            param_data = self.parameter_performance[param_type]
            if not param_data:
                return None
            
            best_value = None
            best_avg_profit = float('-inf')
            
            for value, profits in param_data.items():
                if len(profits) >= 3:  # Need at least 3 samples
                    avg_profit = statistics.mean(profits)
                    win_rate = len([p for p in profits if p > 0]) / len(profits)
                    
                    # CRITICAL UPGRADE: Consider both profit and win rate
                    if win_rate >= self.win_rate_threshold and avg_profit > best_avg_profit:
                        best_avg_profit = avg_profit
                        best_value = value
            
            return best_value
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error finding optimal parameter: {e}")
            return None
    
    def find_optimal_trading_hours(self) -> List[int]:
        """Find optimal trading hours based on performance"""
        try:
            optimal_hours = []
            
            for hour, profits in self.hourly_performance.items():
                if len(profits) >= 3:  # Need at least 3 samples
                    avg_profit = statistics.mean(profits)
                    win_rate = len([p for p in profits if p > 0]) / len(profits)
                    
                    # CRITICAL UPGRADE: Consider both profit and win rate
                    if win_rate >= self.win_rate_threshold and avg_profit > self.profit_threshold:
                        optimal_hours.append(hour)
            
            # Sort by performance
            optimal_hours.sort(key=lambda h: statistics.mean(self.hourly_performance[h]), reverse=True)
            
            return optimal_hours[:6]  # Return top 6 hours
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error finding optimal trading hours: {e}")
            return []
    
    def get_adaptive_recommendations(self, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive recommendations based on learned patterns"""
        try:
            recommendations = {}
            
            # Get current market regime
            current_regime = self.detect_market_regime(current_conditions)
            
            # Regime-specific recommendations
            if current_regime == 'bull':
                recommendations['position_multiplier'] = 1.2
                recommendations['profit_target_multiplier'] = 1.1
                recommendations['risk_level'] = 'moderate'
            elif current_regime == 'bear':
                recommendations['position_multiplier'] = 0.8
                recommendations['profit_target_multiplier'] = 0.9
                recommendations['risk_level'] = 'conservative'
            elif current_regime == 'volatile':
                recommendations['position_multiplier'] = 0.7
                recommendations['profit_target_multiplier'] = 1.3
                recommendations['risk_level'] = 'aggressive'
            else:  # sideways
                recommendations['position_multiplier'] = 1.0
                recommendations['profit_target_multiplier'] = 1.0
                recommendations['risk_level'] = 'neutral'
            
            # Time-based recommendations
            current_hour = datetime.now().hour
            if current_hour in self.learned_parameters.get('optimal_hours', []):
                recommendations['time_multiplier'] = 1.2
                recommendations['trading_confidence'] = 'high'
            else:
                recommendations['time_multiplier'] = 0.8
                recommendations['trading_confidence'] = 'low'
            
            # Token preference recommendations
            preferred_token = self.learned_parameters.get('preferred_token')
            if preferred_token:
                recommendations['preferred_token'] = preferred_token
            
            # Parameter recommendations
            if 'position_percentage' in self.learned_parameters:
                recommendations['suggested_position_pct'] = self.learned_parameters['position_percentage']
            
            if 'profit_target' in self.learned_parameters:
                recommendations['suggested_profit_target'] = self.learned_parameters['profit_target']
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error getting adaptive recommendations: {e}")
            return {}
    
    def is_optimal_trading_time(self) -> bool:
        """Check if current time is optimal for trading"""
        try:
            current_hour = datetime.now().hour
            optimal_hours = self.learned_parameters.get('optimal_hours', [])
            
            if not optimal_hours:  # No learning yet, allow trading
                return True
            
            return current_hour in optimal_hours
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error checking optimal trading time: {e}")
            return True  # Default to allowing trading
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        try:
            total_patterns = len(self.trade_patterns)
            
            if total_patterns == 0:
                return {
                    'total_patterns': 0,
                    'learning_status': 'no_data',
                    'optimization_count': 0,
                    'last_optimization': None,
                    'learned_parameters': {},
                    'optimal_hours': [],
                    'preferred_tokens': []
                }
            
            # Calculate overall performance
            all_profits = [p.outcome['profit'] for p in self.trade_patterns]
            avg_profit = statistics.mean(all_profits) if all_profits else 0
            win_rate = len([p for p in all_profits if p > 0]) / len(all_profits) if all_profits else 0
            
            # Get top performing tokens
            token_performance = {}
            for pattern in self.trade_patterns:
                token = pattern.token
                if token not in token_performance:
                    token_performance[token] = []
                token_performance[token].append(pattern.outcome['profit'])
            
            top_tokens = []
            for token, profits in token_performance.items():
                if len(profits) >= 3:
                    avg_token_profit = statistics.mean(profits)
                    token_win_rate = len([p for p in profits if p > 0]) / len(profits)
                    if token_win_rate >= self.win_rate_threshold:
                        top_tokens.append({
                            'token': token,
                            'avg_profit': avg_token_profit,
                            'win_rate': token_win_rate,
                            'trades': len(profits)
                        })
            
            # Sort by average profit
            top_tokens.sort(key=lambda x: x['avg_profit'], reverse=True)
            
            return {
                'total_patterns': total_patterns,
                'learning_status': 'active' if self.learning_enabled else 'disabled',
                'optimization_count': len(self.learned_parameters),
                'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
                'overall_performance': {
                    'avg_profit': avg_profit,
                    'win_rate': win_rate,
                    'total_trades': total_patterns
                },
                'learned_parameters': self.learned_parameters,
                'optimal_hours': self.learned_parameters.get('optimal_hours', []),
                'preferred_tokens': [t['token'] for t in top_tokens[:5]],  # Top 5 tokens
                'top_performing_tokens': top_tokens[:5]
            }
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error getting learning summary: {e}")
            return {}
    
    def save_learned_parameters(self) -> None:
        """Save learned parameters to file"""
        try:
            data_dir = Path('data')
            data_dir.mkdir(exist_ok=True)
            
            file_path = data_dir / 'learned_parameters.json'
            
            save_data = {
                'learned_parameters': self.learned_parameters,
                'parameter_performance': self.parameter_performance,
                'hourly_performance': self.hourly_performance,
                'daily_performance': self.daily_performance,
                'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
                'total_patterns': len(self.trade_patterns)
            }
            
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.logger.info(f"[LEARNING] Learned parameters saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error saving learned parameters: {e}")
    
    def load_learned_parameters(self) -> None:
        """Load learned parameters from file"""
        try:
            file_path = Path('data') / 'learned_parameters.json'
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.learned_parameters = data.get('learned_parameters', {})
                self.parameter_performance = data.get('parameter_performance', self.parameter_performance)
                self.hourly_performance = data.get('hourly_performance', self.hourly_performance)
                self.daily_performance = data.get('daily_performance', self.daily_performance)
                
                last_opt_str = data.get('last_optimization')
                if last_opt_str:
                    self.last_optimization = datetime.fromisoformat(last_opt_str)
                
                self.logger.info(f"[LEARNING] Loaded {len(self.learned_parameters)} learned parameters")
                
        except Exception as e:
            self.logger.error(f"[LEARNING] Error loading learned parameters: {e}")
    
    def enable_learning(self) -> None:
        """Enable auto-learning"""
        self.learning_enabled = True
        self.logger.info("[LEARNING] Auto-learning enabled")
    
    def disable_learning(self) -> None:
        """Disable auto-learning"""
        self.learning_enabled = False
        self.logger.info("[LEARNING] Auto-learning disabled")
    
    def reset_learning(self) -> None:
        """Reset all learned parameters"""
        self.trade_patterns = []
        self.learned_parameters = {}
        self.parameter_performance = {
            'position_percentage': {},
            'profit_target': {},
            'timeout_seconds': {},
            'token_preference': {},
            'market_timing': {}
        }
        self.hourly_performance = {hour: [] for hour in range(24)}
        self.daily_performance = {day: [] for day in range(7)}
        
        # Remove saved file
        file_path = Path('data') / 'learned_parameters.json'
        if file_path.exists():
            file_path.unlink()
        
        self.logger.info("[LEARNING] All learned parameters reset")
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get detailed performance insights for optimization"""
        try:
            if not self.trade_patterns:
                return {'status': 'no_data'}
            
            insights = {
                'total_trades': len(self.trade_patterns),
                'recent_performance': {},
                'regime_performance': {},
                'token_performance': {},
                'time_performance': {}
            }
            
            # Recent performance (last 20 trades)
            recent_patterns = self.trade_patterns[-20:]
            recent_profits = [p.outcome['profit'] for p in recent_patterns]
            insights['recent_performance'] = {
                'avg_profit': statistics.mean(recent_profits) if recent_profits else 0,
                'win_rate': len([p for p in recent_profits if p > 0]) / len(recent_profits) if recent_profits else 0,
                'trades': len(recent_patterns)
            }
            
            # Regime performance
            regime_profits = {}
            for pattern in self.trade_patterns:
                regime = pattern.market_regime
                if regime not in regime_profits:
                    regime_profits[regime] = []
                regime_profits[regime].append(pattern.outcome['profit'])
            
            for regime, profits in regime_profits.items():
                insights['regime_performance'][regime] = {
                    'avg_profit': statistics.mean(profits),
                    'win_rate': len([p for p in profits if p > 0]) / len(profits),
                    'trades': len(profits)
                }
            
            # Token performance
            token_profits = {}
            for pattern in self.trade_patterns:
                token = pattern.token
                if token not in token_profits:
                    token_profits[token] = []
                token_profits[token].append(pattern.outcome['profit'])
            
            for token, profits in token_profits.items():
                if len(profits) >= 3:  # Only include tokens with 3+ trades
                    insights['token_performance'][token] = {
                        'avg_profit': statistics.mean(profits),
                        'win_rate': len([p for p in profits if p > 0]) / len(profits),
                        'trades': len(profits)
                    }
            
            # Time performance
            hour_profits = {}
            for pattern in self.trade_patterns:
                hour = pattern.timestamp.hour
                if hour not in hour_profits:
                    hour_profits[hour] = []
                hour_profits[hour].append(pattern.outcome['profit'])
            
            for hour, profits in hour_profits.items():
                if len(profits) >= 3:  # Only include hours with 3+ trades
                    insights['time_performance'][hour] = {
                        'avg_profit': statistics.mean(profits),
                        'win_rate': len([p for p in profits if p > 0]) / len(profits),
                        'trades': len(profits)
                    }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"[LEARNING] Error getting performance insights: {e}")
            return {'status': 'error', 'message': str(e)} 