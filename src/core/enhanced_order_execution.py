#!/usr/bin/env python3
"""
Enhanced Order Execution - Tier 2 Component
==========================================

This module implements advanced order execution strategies including
smart order routing, execution optimization, and intelligent order
management for maximum fill rates and optimal execution.

Features:
- Smart order routing with multiple strategies
- Execution optimization for best fill rates
- Intelligent order splitting and timing
- Market impact minimization
- Dynamic order type selection
- Real-time execution monitoring
- Slippage prediction and management
"""

import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
import logging
import random

class EnhancedOrderExecution:
    """Advanced order execution system for optimal trading performance"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Execution parameters
        self.max_slippage = 0.005  # 0.5% maximum slippage
        self.target_fill_rate = 0.95  # 95% target fill rate
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Order execution strategies
        self.execution_strategies = {
            'aggressive': self._aggressive_execution,
            'passive': self._passive_execution,
            'adaptive': self._adaptive_execution,
            'iceberg': self._iceberg_execution,
            'twap': self._twap_execution,
            'vwap': self._vwap_execution
        }
        
        # Execution tracking
        self.execution_history = deque(maxlen=1000)
        self.fill_rate_history = deque(maxlen=100)
        self.slippage_history = deque(maxlen=100)
        
        # Market impact tracking
        self.market_impact_cache = defaultdict(lambda: {
            'avg_impact': 0.0,
            'impact_count': 0,
            'last_update': None
        })
        
        # Order timing optimization
        self.optimal_timing_cache = defaultdict(lambda: {
            'best_hours': [],
            'worst_hours': [],
            'avg_fill_time': 0.0,
            'fill_count': 0
        })
        
        # Smart routing parameters
        self.routing_weights = {
            'fill_rate': 0.4,
            'slippage': 0.3,
            'speed': 0.2,
            'cost': 0.1
        }
        
        self.logger.info("[EXECUTION] Enhanced Order Execution initialized")
    
    def execute_order(self, symbol: str, side: str, quantity: float, 
                     target_price: float, strategy: str = 'adaptive',
                     market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute order using enhanced execution strategies"""
        try:
            # Validate inputs
            if quantity <= 0 or target_price <= 0:
                raise ValueError("Invalid quantity or target price")
            
            # Select execution strategy
            if strategy not in self.execution_strategies:
                strategy = 'adaptive'
                self.logger.warning(f"[EXECUTION] Unknown strategy '{strategy}', using adaptive")
            
            # Get market data if not provided
            if market_data is None:
                market_data = self._get_default_market_data(symbol)
            
            # Execute using selected strategy
            execution_func = self.execution_strategies[strategy]
            result = execution_func(symbol, side, quantity, target_price, market_data)
            
            # Record execution
            self._record_execution(symbol, side, quantity, target_price, strategy, result)
            
            # Update market impact cache
            self._update_market_impact(symbol, quantity, result.get('actual_price', target_price), target_price)
            
            # Update timing optimization
            self._update_timing_optimization(symbol, result.get('execution_time', 0))
            
            self.logger.info(f"[EXECUTION] Order executed: {symbol} {side} {quantity} @ ${result.get('actual_price', target_price):.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error executing order: {e}")
            return {
                'success': False,
                'error': str(e),
                'filled_quantity': 0.0,
                'actual_price': target_price,
                'slippage': 0.0,
                'execution_time': 0.0,
                'strategy_used': strategy
            }
    
    def _aggressive_execution(self, symbol: str, side: str, quantity: float,
                            target_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggressive execution strategy for immediate fills"""
        try:
            # Calculate aggressive price
            spread = market_data.get('spread', 0.001)
            volatility = market_data.get('volatility', 0.02)
            
            # Aggressive pricing based on side
            if side.upper() == 'BUY':
                aggressive_price = target_price * (1 + spread + volatility * 0.5)
            else:  # SELL
                aggressive_price = target_price * (1 - spread - volatility * 0.5)
            
            # Simulate execution
            execution_time = random.uniform(0.1, 0.5)  # Fast execution
            fill_rate = random.uniform(0.95, 1.0)  # High fill rate
            filled_quantity = quantity * fill_rate
            
            # Calculate slippage
            slippage = abs(aggressive_price - target_price) / target_price
            
            return {
                'success': True,
                'filled_quantity': filled_quantity,
                'actual_price': aggressive_price,
                'slippage': slippage,
                'execution_time': execution_time,
                'fill_rate': fill_rate,
                'strategy_used': 'aggressive',
                'remaining_quantity': quantity - filled_quantity
            }
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error in aggressive execution: {e}")
            return {'success': False, 'error': str(e)}
    
    def _passive_execution(self, symbol: str, side: str, quantity: float,
                          target_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Passive execution strategy for better prices"""
        try:
            # Calculate passive price
            spread = market_data.get('spread', 0.001)
            
            # Passive pricing based on side
            if side.upper() == 'BUY':
                passive_price = target_price * (1 - spread * 0.5)
            else:  # SELL
                passive_price = target_price * (1 + spread * 0.5)
            
            # Simulate execution
            execution_time = random.uniform(1.0, 5.0)  # Slower execution
            fill_rate = random.uniform(0.7, 0.9)  # Lower fill rate
            filled_quantity = quantity * fill_rate
            
            # Calculate slippage (usually negative for passive)
            slippage = (passive_price - target_price) / target_price
            
            return {
                'success': True,
                'filled_quantity': filled_quantity,
                'actual_price': passive_price,
                'slippage': slippage,
                'execution_time': execution_time,
                'fill_rate': fill_rate,
                'strategy_used': 'passive',
                'remaining_quantity': quantity - filled_quantity
            }
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error in passive execution: {e}")
            return {'success': False, 'error': str(e)}
    
    def _adaptive_execution(self, symbol: str, side: str, quantity: float,
                           target_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive execution strategy based on market conditions"""
        try:
            # Analyze market conditions
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 1.0)
            spread = market_data.get('spread', 0.001)
            
            # Determine execution aggressiveness
            if volatility > 0.05:  # High volatility
                aggressiveness = 0.8  # More aggressive
            elif volume < 0.5:  # Low volume
                aggressiveness = 0.6  # Moderately aggressive
            else:
                aggressiveness = 0.4  # Less aggressive
            
            # Calculate adaptive price
            price_adjustment = spread * aggressiveness
            if side.upper() == 'BUY':
                adaptive_price = target_price * (1 + price_adjustment)
            else:  # SELL
                adaptive_price = target_price * (1 - price_adjustment)
            
            # Simulate execution
            execution_time = random.uniform(0.5, 2.0)
            fill_rate = random.uniform(0.8, 0.95)
            filled_quantity = quantity * fill_rate
            
            # Calculate slippage
            slippage = (adaptive_price - target_price) / target_price
            
            return {
                'success': True,
                'filled_quantity': filled_quantity,
                'actual_price': adaptive_price,
                'slippage': slippage,
                'execution_time': execution_time,
                'fill_rate': fill_rate,
                'strategy_used': 'adaptive',
                'aggressiveness': aggressiveness,
                'remaining_quantity': quantity - filled_quantity
            }
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error in adaptive execution: {e}")
            return {'success': False, 'error': str(e)}
    
    def _iceberg_execution(self, symbol: str, side: str, quantity: float,
                          target_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Iceberg execution strategy for large orders"""
        try:
            # Determine optimal chunk size
            avg_volume = market_data.get('avg_volume', 1000.0)
            chunk_size = min(quantity * 0.1, avg_volume * 0.05)  # 10% of order or 5% of avg volume
            chunk_size = max(chunk_size, quantity * 0.05)  # Minimum 5% of order
            
            # Execute in chunks
            total_filled = 0.0
            total_cost = 0.0
            execution_time = 0.0
            
            remaining_quantity = quantity
            while remaining_quantity > 0:
                # Calculate chunk size for this iteration
                current_chunk = min(chunk_size, remaining_quantity)
                
                # Execute chunk with adaptive strategy
                chunk_result = self._adaptive_execution(symbol, side, current_chunk, target_price, market_data)
                
                if chunk_result['success']:
                    total_filled += chunk_result['filled_quantity']
                    total_cost += chunk_result['filled_quantity'] * chunk_result['actual_price']
                    execution_time += chunk_result['execution_time']
                    remaining_quantity -= chunk_result['filled_quantity']
                    
                    # Add delay between chunks
                    time.sleep(random.uniform(0.1, 0.5))
                else:
                    break
            
            # Calculate average price
            avg_price = total_cost / total_filled if total_filled > 0 else target_price
            
            # Calculate overall slippage
            slippage = (avg_price - target_price) / target_price
            
            return {
                'success': total_filled > 0,
                'filled_quantity': total_filled,
                'actual_price': avg_price,
                'slippage': slippage,
                'execution_time': execution_time,
                'fill_rate': total_filled / quantity if quantity > 0 else 0.0,
                'strategy_used': 'iceberg',
                'chunks_executed': quantity / chunk_size if chunk_size > 0 else 0,
                'remaining_quantity': remaining_quantity
            }
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error in iceberg execution: {e}")
            return {'success': False, 'error': str(e)}
    
    def _twap_execution(self, symbol: str, side: str, quantity: float,
                       target_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Time-Weighted Average Price execution strategy"""
        try:
            # Determine execution window
            volatility = market_data.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                window_minutes = 30
            elif volatility > 0.02:  # Medium volatility
                window_minutes = 60
            else:  # Low volatility
                window_minutes = 120
            
            # Calculate time intervals
            interval_minutes = max(5, window_minutes // 10)  # 10 intervals minimum
            intervals = window_minutes // interval_minutes
            
            # Execute over time intervals
            total_filled = 0.0
            total_cost = 0.0
            execution_time = 0.0
            
            quantity_per_interval = quantity / intervals
            
            for i in range(intervals):
                # Execute interval
                interval_result = self._adaptive_execution(
                    symbol, side, quantity_per_interval, target_price, market_data
                )
                
                if interval_result['success']:
                    total_filled += interval_result['filled_quantity']
                    total_cost += interval_result['filled_quantity'] * interval_result['actual_price']
                    execution_time += interval_result['execution_time']
                
                # Wait for next interval
                if i < intervals - 1:
                    time.sleep(interval_minutes * 60)  # Convert to seconds
            
            # Calculate TWAP
            twap_price = total_cost / total_filled if total_filled > 0 else target_price
            
            # Calculate slippage
            slippage = (twap_price - target_price) / target_price
            
            return {
                'success': total_filled > 0,
                'filled_quantity': total_filled,
                'actual_price': twap_price,
                'slippage': slippage,
                'execution_time': execution_time,
                'fill_rate': total_filled / quantity if quantity > 0 else 0.0,
                'strategy_used': 'twap',
                'intervals': intervals,
                'window_minutes': window_minutes,
                'remaining_quantity': quantity - total_filled
            }
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error in TWAP execution: {e}")
            return {'success': False, 'error': str(e)}
    
    def _vwap_execution(self, symbol: str, side: str, quantity: float,
                       target_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Volume-Weighted Average Price execution strategy"""
        try:
            # Get volume profile
            volume_profile = market_data.get('volume_profile', {})
            if not volume_profile:
                # Fallback to adaptive execution
                return self._adaptive_execution(symbol, side, quantity, target_price, market_data)
            
            # Calculate VWAP
            total_volume = sum(volume_profile.values())
            vwap_price = sum(price * volume for price, volume in volume_profile.items()) / total_volume
            
            # Execute with VWAP target
            result = self._adaptive_execution(symbol, side, quantity, vwap_price, market_data)
            
            # Update strategy info
            result['strategy_used'] = 'vwap'
            result['vwap_price'] = vwap_price
            
            return result
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error in VWAP execution: {e}")
            return {'success': False, 'error': str(e)}
    
    def select_optimal_strategy(self, symbol: str, side: str, quantity: float,
                              target_price: float, market_data: Dict[str, Any]) -> str:
        """Select optimal execution strategy based on order characteristics and market conditions"""
        try:
            # Analyze order characteristics
            order_value = quantity * target_price
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 1.0)
            spread = market_data.get('spread', 0.001)
            
            # Strategy selection logic
            if order_value > 10000:  # Large order
                if volatility > 0.05:  # High volatility
                    return 'iceberg'
                else:
                    return 'twap'
            elif order_value > 1000:  # Medium order
                if spread > 0.002:  # Wide spread
                    return 'passive'
                else:
                    return 'adaptive'
            else:  # Small order
                if volatility > 0.03:  # High volatility
                    return 'aggressive'
                else:
                    return 'adaptive'
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error selecting strategy: {e}")
            return 'adaptive'
    
    def predict_slippage(self, symbol: str, side: str, quantity: float,
                        target_price: float, market_data: Dict[str, Any]) -> float:
        """Predict expected slippage for an order"""
        try:
            # Get historical slippage data
            symbol_slippage = [record['slippage'] for record in self.execution_history 
                             if record['symbol'] == symbol and record['side'] == side]
            
            if symbol_slippage:
                avg_slippage = sum(symbol_slippage) / len(symbol_slippage)
            else:
                avg_slippage = 0.001  # Default 0.1%
            
            # Adjust for order size
            order_value = quantity * target_price
            size_multiplier = min(2.0, max(0.5, order_value / 1000))  # Scale based on order size
            
            # Adjust for market conditions
            volatility = market_data.get('volatility', 0.02)
            volatility_multiplier = 1.0 + (volatility * 10)  # Higher volatility = more slippage
            
            # Adjust for spread
            spread = market_data.get('spread', 0.001)
            spread_multiplier = 1.0 + (spread * 100)  # Wider spread = more slippage
            
            # Calculate predicted slippage
            predicted_slippage = avg_slippage * size_multiplier * volatility_multiplier * spread_multiplier
            
            # Ensure reasonable bounds
            predicted_slippage = max(0.0, min(predicted_slippage, self.max_slippage))
            
            return predicted_slippage
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error predicting slippage: {e}")
            return 0.001  # Default 0.1%
    
    def optimize_order_timing(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize order timing based on historical data"""
        try:
            # Get timing data for symbol
            timing_data = self.optimal_timing_cache[symbol]
            
            current_hour = datetime.now().hour
            
            # Check if current time is optimal
            is_optimal_time = current_hour in timing_data['best_hours']
            is_worst_time = current_hour in timing_data['worst_hours']
            
            # Calculate timing score
            if is_optimal_time:
                timing_score = 1.0
            elif is_worst_time:
                timing_score = 0.3
            else:
                timing_score = 0.7
            
            # Get recommended delay
            recommended_delay = 0
            if is_worst_time:
                # Wait for better time
                next_best_hour = min([h for h in timing_data['best_hours'] if h > current_hour], default=current_hour + 1)
                recommended_delay = (next_best_hour - current_hour) * 3600  # Convert to seconds
            
            return {
                'is_optimal_time': is_optimal_time,
                'timing_score': timing_score,
                'recommended_delay': recommended_delay,
                'best_hours': timing_data['best_hours'],
                'worst_hours': timing_data['worst_hours'],
                'avg_fill_time': timing_data['avg_fill_time']
            }
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error optimizing order timing: {e}")
            return {
                'is_optimal_time': True,
                'timing_score': 0.7,
                'recommended_delay': 0,
                'best_hours': [],
                'worst_hours': [],
                'avg_fill_time': 0.0
            }
    
    def _record_execution(self, symbol: str, side: str, quantity: float,
                         target_price: float, strategy: str, result: Dict[str, Any]):
        """Record execution result for analysis"""
        try:
            record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'target_price': target_price,
                'strategy': strategy,
                'success': result.get('success', False),
                'filled_quantity': result.get('filled_quantity', 0.0),
                'actual_price': result.get('actual_price', target_price),
                'slippage': result.get('slippage', 0.0),
                'execution_time': result.get('execution_time', 0.0),
                'fill_rate': result.get('fill_rate', 0.0)
            }
            
            self.execution_history.append(record)
            
            # Update fill rate history
            if result.get('success', False):
                self.fill_rate_history.append(result.get('fill_rate', 0.0))
                self.slippage_history.append(result.get('slippage', 0.0))
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error recording execution: {e}")
    
    def _update_market_impact(self, symbol: str, quantity: float, actual_price: float, target_price: float):
        """Update market impact tracking"""
        try:
            impact = abs(actual_price - target_price) / target_price
            
            cache = self.market_impact_cache[symbol]
            cache['avg_impact'] = (cache['avg_impact'] * cache['impact_count'] + impact) / (cache['impact_count'] + 1)
            cache['impact_count'] += 1
            cache['last_update'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error updating market impact: {e}")
    
    def _update_timing_optimization(self, symbol: str, execution_time: float):
        """Update timing optimization data"""
        try:
            cache = self.optimal_timing_cache[symbol]
            current_hour = datetime.now().hour
            
            # Update average fill time
            cache['avg_fill_time'] = (cache['avg_fill_time'] * cache['fill_count'] + execution_time) / (cache['fill_count'] + 1)
            cache['fill_count'] += 1
            
            # Update best/worst hours based on fill time
            if execution_time < cache['avg_fill_time'] * 0.8:  # Fast execution
                if current_hour not in cache['best_hours']:
                    cache['best_hours'].append(current_hour)
            elif execution_time > cache['avg_fill_time'] * 1.2:  # Slow execution
                if current_hour not in cache['worst_hours']:
                    cache['worst_hours'].append(current_hour)
            
            # Keep only top 3 best/worst hours
            cache['best_hours'] = sorted(cache['best_hours'])[:3]
            cache['worst_hours'] = sorted(cache['worst_hours'])[:3]
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error updating timing optimization: {e}")
    
    def _get_default_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get default market data for execution"""
        return {
            'volatility': 0.02,
            'volume': 1.0,
            'spread': 0.001,
            'avg_volume': 1000.0,
            'volume_profile': {}
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution performance summary"""
        try:
            if not self.execution_history:
                return {
                    'total_orders': 0,
                    'success_rate': 0.0,
                    'avg_fill_rate': 0.0,
                    'avg_slippage': 0.0,
                    'avg_execution_time': 0.0,
                    'strategy_performance': {},
                    'market_impact_summary': {}
                }
            
            # Calculate overall statistics
            total_orders = len(self.execution_history)
            successful_orders = sum(1 for record in self.execution_history if record['success'])
            success_rate = successful_orders / total_orders if total_orders > 0 else 0.0
            
            # Calculate averages
            avg_fill_rate = sum(self.fill_rate_history) / len(self.fill_rate_history) if self.fill_rate_history else 0.0
            avg_slippage = sum(self.slippage_history) / len(self.slippage_history) if self.slippage_history else 0.0
            
            execution_times = [record['execution_time'] for record in self.execution_history if record['success']]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
            
            # Strategy performance
            strategy_performance = defaultdict(lambda: {
                'count': 0,
                'success_rate': 0.0,
                'avg_fill_rate': 0.0,
                'avg_slippage': 0.0
            })
            
            for record in self.execution_history:
                strategy = record['strategy']
                perf = strategy_performance[strategy]
                perf['count'] += 1
                
                if record['success']:
                    perf['success_rate'] += 1
                    perf['avg_fill_rate'] += record['fill_rate']
                    perf['avg_slippage'] += record['slippage']
            
            # Normalize strategy performance
            for strategy, perf in strategy_performance.items():
                if perf['count'] > 0:
                    perf['success_rate'] /= perf['count']
                    perf['avg_fill_rate'] /= perf['count']
                    perf['avg_slippage'] /= perf['count']
            
            # Market impact summary
            market_impact_summary = {}
            for symbol, impact_data in self.market_impact_cache.items():
                market_impact_summary[symbol] = {
                    'avg_impact': impact_data['avg_impact'],
                    'impact_count': impact_data['impact_count'],
                    'last_update': impact_data['last_update']
                }
            
            return {
                'total_orders': total_orders,
                'success_rate': success_rate,
                'avg_fill_rate': avg_fill_rate,
                'avg_slippage': avg_slippage,
                'avg_execution_time': avg_execution_time,
                'strategy_performance': dict(strategy_performance),
                'market_impact_summary': market_impact_summary
            }
            
        except Exception as e:
            self.logger.error(f"[EXECUTION] Error getting execution summary: {e}")
            return {} 