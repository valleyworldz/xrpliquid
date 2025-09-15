#!/usr/bin/env python3
"""
Hyperliquid Fee Optimizer Module
Advanced fee optimization for Hyperliquid trading with VIP tiers and staking discounts
"""

import logging
import time
from typing import Dict, Tuple, Optional
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import configuration safely
try:
    from src.core.config import config
except ImportError:
    # Fallback configuration if modular config not available
    class FallbackConfig:
        """Fallback configuration for fee optimizer"""
        VIP_TIER_THRESHOLDS = {
            0: 0, 1: 5e6, 2: 25e6, 3: 100e6, 4: 500e6, 5: 2e9, 6: 7e9
        }
        FEE_RATES = {
            0: (0.045, 0.015), 1: (0.040, 0.012), 2: (0.035, 0.008),
            3: (0.030, 0.004), 4: (0.028, 0.000), 5: (0.026, 0.000), 6: (0.024, 0.000)
        }
        STAKING_DISCOUNTS = {
            'wood': 0.05, 'bronze': 0.10, 'silver': 0.15,
            'gold': 0.20, 'platinum': 0.30, 'diamond': 0.40
        }
        TAKER_FEE = 0.00045
        MAKER_FEE = 0.00015
        ROUND_TRIP_TAKER = 0.0009
        ROUND_TRIP_MAKER = 0.0003
        DEFAULT_SLIPPAGE = 0.0005
        FEE_BUFFER = 0.0003
        PROFIT_TARGET_PCT = 0.050
        STOP_LOSS_PCT = 0.003
        MIN_SIGNAL_STRENGTH = 0.95
        MAX_TRADES_PER_HOUR = 3
        MIN_TRADE_INTERVAL = 600
    
    config = FallbackConfig()


class HyperliquidFeeOptimizer:
    """Advanced fee optimization for Hyperliquid trading"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # VIP Tier thresholds (14-day weighted volume)
        self.vip_tiers = config.VIP_TIER_THRESHOLDS
        
        # Fee rates by VIP tier (perps taker/maker)
        self.fee_rates = config.FEE_RATES
        
        # HYPE staking discounts
        self.staking_discounts = config.STAKING_DISCOUNTS
        
        # Track volume for VIP tier calculation
        self.rolling_volume = 0.0
        self.volume_history = []
        
    def calculate_effective_fee(self, order_type="maker", current_tier=0, staking_tier="wood"):
        """Calculate effective fee after all discounts"""
        base_taker, base_maker = self.fee_rates[current_tier]
        base_fee = base_taker if order_type == "taker" else base_maker
        
        # Apply staking discount
        staking_discount = self.staking_discounts.get(staking_tier, 0.0)
        effective_fee = base_fee * (1 - staking_discount)
        
        return effective_fee
    
    def should_use_maker_order(self, urgency="low"):
        """Determine if we should use maker orders for fee optimization"""
        if urgency == "high":
            return False  # Use taker for urgent orders
        
        # Always prefer maker orders for fee optimization
        return True
    
    def get_optimal_order_type(self, signal_strength, market_volatility):
        """Get optimal order type based on market conditions with enhanced fee optimization"""
        # CRITICAL FIX: Enhanced maker order preference for 66% fee reduction
        # Base tier: 0.045% taker → 0.015% maker = 66% savings
        
        # Strong signal + low volatility = perfect for maker orders
        if signal_strength > 0.8 and market_volatility < 0.03:
            return "post_only"  # Force maker for maximum fee savings
        
        # Medium signal + moderate volatility = prefer maker
        elif signal_strength > 0.6 and market_volatility < 0.05:
            return "post_only"  # Still prefer maker for fee optimization
        
        # Weak signal or high volatility = use market for execution certainty
        else:
            return "market"  # Only use taker when necessary
    
    def check_minimum_profit_requirement(self, entry_price, target_price, position_size, order_type="maker"):
        """Check if potential trade meets minimum profit requirements after fees"""
        try:
            # Calculate gross profit
            price_diff = abs(target_price - entry_price)
            gross_profit_pct = price_diff / entry_price
            
            # Get effective fees
            effective_fee = self.calculate_effective_fee(order_type)
            total_fees = effective_fee * 2  # Round trip
            
            # Calculate net profit
            net_profit_pct = gross_profit_pct - total_fees
            
            # Check against minimum profitable move
            min_profitable = self.min_profitable_move(order_type == "maker")
            
            # Add safety buffer
            safety_buffer = 0.0005  # 0.05%
            required_profit = min_profitable + safety_buffer
            
            meets_requirement = net_profit_pct >= required_profit
            
            if self.logger:
                self.logger.info(f"Profit check: {gross_profit_pct:.4f}% gross, {net_profit_pct:.4f}% net, "
                               f"required: {required_profit:.4f}%, meets: {meets_requirement}")
            
            return meets_requirement, net_profit_pct
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking minimum profit: {e}")
            return False, 0.0
    
    def check_trading_frequency_limit(self, last_trade_time=None):
        """Enhanced trading frequency control with hourly limits"""
        try:
            if last_trade_time is None:
                return True
            
            current_time = time.time()
            time_since_last = current_time - last_trade_time
            
            # CRITICAL FIX: 5-minute minimum interval between trades
            min_interval = config.MIN_TRADE_INTERVAL  # 300 seconds
            if time_since_last < min_interval:
                return False
            
            # CRITICAL FIX: Track hourly trade count for 6 trades/hour limit
            if not hasattr(self, '_hourly_trades'):
                self._hourly_trades = []
            
            # Clean old trades (older than 1 hour)
            one_hour_ago = current_time - 3600
            self._hourly_trades = [t for t in self._hourly_trades if t > one_hour_ago]
            
            # Check if we've hit the hourly limit
            if len(self._hourly_trades) >= config.MAX_TRADES_PER_HOUR:
                return False
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking trading frequency: {e}")
            return True  # Allow trading if check fails
    
    def record_trade_time(self):
        """Record trade time for frequency tracking"""
        try:
            current_time = time.time()
            if not hasattr(self, '_hourly_trades'):
                self._hourly_trades = []
            
            self._hourly_trades.append(current_time)
            
            if self.logger:
                self.logger.info(f"📊 Trade recorded. Hourly count: {len(self._hourly_trades)}/{config.MAX_TRADES_PER_HOUR}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error recording trade time: {e}")
    
    def check_signal_strength_gate(self, signal_confidence):
        """Check if signal meets minimum strength requirement"""
        try:
            min_strength = 0.80  # Updated from 0.30 to 0.80 for better signal quality
            meets_gate = signal_confidence >= min_strength
            
            if self.logger and not meets_gate:
                self.logger.info(f"🚫 Signal strength {signal_confidence:.2f} below minimum {min_strength}")
            
            return meets_gate
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking signal strength: {e}")
            return True  # Allow trading if check fails
    
    def min_profitable_move(self, is_maker: bool = True) -> float:
        """Calculate minimum profitable move after fees"""
        try:
            if is_maker:
                return config.ROUND_TRIP_MAKER + config.DEFAULT_SLIPPAGE + config.FEE_BUFFER
            else:
                return config.ROUND_TRIP_TAKER + config.DEFAULT_SLIPPAGE + config.FEE_BUFFER
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating min profitable move: {e}")
            return 0.002  # 0.2% fallback
    
    def _calculate_fee_savings(self, order_type):
        """Calculate fee savings compared to taker orders"""
        try:
            if order_type in ["post_only", "limit"]:
                # Maker order: 0.015% vs taker 0.045% = 66% savings
                taker_fee = config.TAKER_FEE
                maker_fee = config.MAKER_FEE
                savings = (taker_fee - maker_fee) / taker_fee
                return savings
            else:
                return 0.0  # No savings for market orders
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating fee savings: {e}")
            return 0.0

    # Example fee table (can be replaced with config)
    FEE_TABLE = {
        0: {'maker': 0.00015, 'taker': 0.00045},
        1: {'maker': 0.00012, 'taker': 0.00040},
        2: {'maker': 0.00008, 'taker': 0.00035},
        3: {'maker': 0.00004, 'taker': 0.00030},
        4: {'maker': 0.00000, 'taker': 0.00028},
        5: {'maker': 0.00000, 'taker': 0.00026},
        6: {'maker': 0.00000, 'taker': 0.00024},
    }
    VIP_THRESHOLDS = [0, 5e6, 25e6, 100e6, 500e6, 2e9, 7e9]

    def calculate_fees(self, order_value, is_maker=True, vip_tier=0):
        """Calculate trading fees for a given order."""
        tier = min(max(int(vip_tier), 0), 6)
        fee_rate = self.FEE_TABLE[tier]['maker' if is_maker else 'taker']
        return order_value * fee_rate

    def get_vip_tier(self, volume):
        """Determine VIP tier based on trading volume."""
        for i in reversed(range(len(self.VIP_THRESHOLDS))):
            if volume >= self.VIP_THRESHOLDS[i]:
                return i
        return 0

    def adjust_parameters_for_regime(self, market_regime):
        """Adjust trading parameters based on market regime"""
        regime_params = {
            "TRENDING": {
                "profit_target": config.PROFIT_TARGET_PCT,  # 3.5% in trending markets
                "stop_loss": config.STOP_LOSS_PCT,      # 2.5% stop loss
                "confidence_boost": 1.2,  # 20% confidence boost
                "position_size_boost": 1.3  # 30% larger positions
            },
            "RANGING": {
                "profit_target": config.PROFIT_TARGET_PCT,  # 3.5% in ranging markets
                "stop_loss": config.STOP_LOSS_PCT,      # 2.5% stop loss
                "confidence_boost": 0.9,  # 10% confidence reduction
                "position_size_boost": 0.8  # 20% smaller positions
            },
            "VOLATILE": {
                "profit_target": config.PROFIT_TARGET_PCT + 0.02,  # 5.5% in volatile markets
                "stop_loss": config.STOP_LOSS_PCT + 0.01,      # 3.5% stop loss
                "confidence_boost": 0.8,  # 20% confidence reduction
                "position_size_boost": 0.7  # 30% smaller positions
            },
            "MIXED": {
                "profit_target": config.PROFIT_TARGET_PCT,  # 3.5% default
                "stop_loss": config.STOP_LOSS_PCT,      # 2.5% default
                "confidence_boost": 1.0,  # No change
                "position_size_boost": 1.0  # No change
            }
        }
        
        return regime_params.get(market_regime, regime_params["MIXED"]) 