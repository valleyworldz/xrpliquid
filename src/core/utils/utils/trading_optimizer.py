#!/usr/bin/env python3
"""
Trading Optimization Module
Optimizes trading strategy based on performance data
"""

import time
import logging
from typing import Dict, List, Optional
import numpy as np

class TradingOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.performance_history = []
        self.strategy_metrics = {}
        self.logger = logging.getLogger(__name__)
        
    def should_trade(self, market_data: Dict, signal_strength: float) -> bool:
        """Determine if we should trade based on market conditions and signal"""
        
        # Check confidence threshold
        min_confidence = self.config.get('entry_confidence_threshold', 0.75)
        if signal_strength < min_confidence:
            self.logger.info(f"Signal strength {signal_strength:.2f} below threshold {min_confidence}")
            return False
        
        # Check market conditions
        if not self._check_market_conditions(market_data):
            return False
        
        # Check trading hours (avoid 24/7 trading)
        current_hour = time.localtime().tm_hour
        if current_hour < 6 or current_hour > 22:  # Trade 6 AM - 10 PM
            self.logger.info(f"Outside trading hours: {current_hour}:00")
            return False
        
        return True
    
    def optimize_position_size(self, base_size: float, market_volatility: float, signal_strength: float) -> float:
        """Optimize position size based on market conditions"""
        
        # Adjust for volatility
        volatility_factor = 1.0 / (1.0 + market_volatility)
        
        # Adjust for signal strength
        signal_factor = signal_strength
        
        # Combine factors
        optimized_size = base_size * volatility_factor * signal_factor
        
        # Ensure within limits
        min_size = self.config.get('min_position_size', 500)
        max_size = self.config.get('max_position_size', 2000)
        
        return max(min_size, min(max_size, optimized_size))
    
    def get_optimal_entry_timing(self, market_data: Dict) -> Dict:
        """Determine optimal entry timing"""
        
        # Analyze market microstructure
        volume = market_data.get('volume_24h', 0)
        spread = market_data.get('spread_percent', 0)
        volatility = market_data.get('volatility', 0)
        
        # Prefer high volume, low spread, moderate volatility
        volume_score = min(1.0, volume / 10000000)  # Normalize to 10M
        spread_score = max(0, 1.0 - spread / 0.1)   # Prefer spreads < 0.1%
        volatility_score = max(0, 1.0 - abs(volatility - 0.05) / 0.05)  # Prefer 5% volatility
        
        timing_score = (volume_score + spread_score + volatility_score) / 3
        
        return {
            'should_enter': timing_score > 0.7,
            'timing_score': timing_score,
            'recommended_size_multiplier': timing_score
        }
    
    def update_strategy_performance(self, trade_result: Dict):
        """Update strategy performance metrics"""
        
        self.performance_history.append(trade_result)
        
        # Calculate rolling metrics
        if len(self.performance_history) >= 10:
            recent_trades = self.performance_history[-10:]
            
            win_rate = sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / len(recent_trades)
            avg_profit = np.mean([t.get('pnl', 0) for t in recent_trades if t.get('pnl', 0) > 0])
            avg_loss = np.mean([t.get('pnl', 0) for t in recent_trades if t.get('pnl', 0) < 0])
            
            self.strategy_metrics = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            }
            
            self.logger.info(f"Strategy metrics - Win rate: {win_rate:.2f}, Profit factor: {self.strategy_metrics['profit_factor']:.2f}")
    
    def should_adjust_strategy(self) -> Dict:
        """Determine if strategy should be adjusted"""
        
        if not self.strategy_metrics:
            return {'should_adjust': False, 'reason': 'Insufficient data'}
        
        win_rate = self.strategy_metrics.get('win_rate', 0)
        profit_factor = self.strategy_metrics.get('profit_factor', 0)
        
        adjustments = []
        
        # Adjust for low win rate
        if win_rate < 0.5:
            adjustments.append('increase_confidence_threshold')
        
        # Adjust for poor profit factor
        if profit_factor < 1.5:
            adjustments.append('improve_risk_reward')
        
        # Adjust for high win rate but low profits
        if win_rate > 0.7 and profit_factor < 1.2:
            adjustments.append('increase_position_sizes')
        
        return {
            'should_adjust': len(adjustments) > 0,
            'adjustments': adjustments,
            'current_metrics': self.strategy_metrics
        }
    
    def _check_market_conditions(self, market_data: Dict) -> bool:
        """Check if market conditions are suitable for trading"""
        
        volume = market_data.get('volume_24h', 0)
        spread = market_data.get('spread_percent', 0)
        volatility = market_data.get('volatility', 0)
        
        min_volume = self.config.get('min_volume', 5000000)
        max_spread = self.config.get('max_spread_percent', 0.05)
        min_volatility = self.config.get('min_volatility', 0.01)
        max_volatility = self.config.get('max_volatility', 0.10)
        
        if volume < min_volume:
            self.logger.info(f"Volume ${volume:,.0f} below minimum ${min_volume:,.0f}")
            return False
        
        if spread > max_spread:
            self.logger.info(f"Spread {spread:.4f} above maximum {max_spread:.4f}")
            return False
        
        if volatility < min_volatility or volatility > max_volatility:
            self.logger.info(f"Volatility {volatility:.4f} outside range [{min_volatility:.4f}, {max_volatility:.4f}]")
            return False
        
        return True
