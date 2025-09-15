#!/usr/bin/env python3
"""
Risk Management and Position Sizing for XRP Trading Bot
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Risk parameters
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.max_daily_loss = 0.1       # 10% max daily loss
        self.max_consecutive_losses = 3  # Max consecutive losses
        self.max_daily_trades = 50      # Max trades per day
        self.min_position_size = 10.0   # Minimum XRP position
        self.max_position_size = 100.0  # Maximum XRP position
        
        # Margin safety
        self.margin_ratio_threshold = 1.2  # Minimum margin ratio
        self.collateral_safety_buffer = 0.8  # 80% safety buffer
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.last_daily_reset = time.time()
        
        # Position tracking
        self.current_position = None
        self.initial_capital = None
        self.peak_capital = None
        
    def calculate_position_size(self, current_price: float, free_collateral: float, 
                              confidence: float, signal_type: str) -> float:
        """
        Calculate safe position size based on risk parameters
        
        Args:
            current_price: Current XRP price
            free_collateral: Available collateral
            confidence: Signal confidence (0-1)
            signal_type: Signal type (LONG/SHORT)
            
        Returns:
            Position size in XRP
        """
        try:
            # Recompute current collateral for accuracy
            updated_collateral = self._update_collateral(free_collateral)
            
            # Check if we can afford minimum position
            min_position_cost = current_price * self.min_position_size
            if min_position_cost > updated_collateral * self.collateral_safety_buffer:
                self.logger.warning(f"Cannot afford minimum position: ${min_position_cost:.2f} > ${updated_collateral * self.collateral_safety_buffer:.2f}")
                return 0.0
            
            # Base position sizing
            risk_amount = updated_collateral * self.max_risk_per_trade
            confidence_multiplier = min(confidence * 2, 1.0)
            
            # Calculate position size based on risk
            position_value = risk_amount * confidence_multiplier
            position_size = position_value / current_price
            
            # Apply size constraints
            position_size = max(position_size, self.min_position_size)
            position_size = min(position_size, self.max_position_size)
            
            # Check maximum position value (50% of collateral)
            max_position_value = updated_collateral * 0.5
            max_position_size = max_position_value / current_price
            position_size = min(position_size, max_position_size)
            
            # Final validation
            final_value = position_size * current_price
            if final_value > updated_collateral * self.collateral_safety_buffer:
                self.logger.warning(f"Position too large: ${final_value:.2f} > ${updated_collateral * self.collateral_safety_buffer:.2f}")
                return 0.0
            
            self.logger.info(f"Position size calculated: {position_size:.1f} XRP (${final_value:.2f})")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _update_collateral(self, current_collateral: float) -> float:
        """Update collateral tracking"""
        try:
            # Track initial capital if not set
            if self.initial_capital is None:
                self.initial_capital = current_collateral
                self.peak_capital = current_collateral
                self.logger.info(f"Initial capital tracked: ${current_collateral:.2f}")
            
            # Update peak capital
            if current_collateral > self.peak_capital:
                self.peak_capital = current_collateral
            
            return current_collateral
            
        except Exception as e:
            self.logger.error(f"Error updating collateral: {e}")
            return current_collateral
    
    def check_risk_limits(self, trade_value: float, signal_confidence: float) -> bool:
        """
        Check if trade meets risk limits
        
        Args:
            trade_value: Value of proposed trade
            signal_confidence: Confidence of trading signal
            
        Returns:
            True if trade is allowed, False otherwise
        """
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.initial_capital * self.max_daily_loss:
                self.logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
                return False
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
                return False
            
            # Check daily trade limit
            if len(self.daily_trades) >= self.max_daily_trades:
                self.logger.warning(f"Daily trade limit reached: {len(self.daily_trades)}")
                return False
            
            # Check minimum confidence
            if signal_confidence < 0.6:
                self.logger.warning(f"Signal confidence too low: {signal_confidence:.2f}")
                return False
            
            # Check drawdown limit
            if self.initial_capital and self.peak_capital:
                drawdown = (self.peak_capital - self.daily_pnl) / self.peak_capital
                if drawdown > 0.15:  # 15% drawdown limit
                    self.logger.warning(f"Drawdown limit reached: {drawdown:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def check_margin_ratio(self, margin_ratio: float, free_collateral: float) -> bool:
        """
        Check margin ratio safety
        
        Args:
            margin_ratio: Current margin ratio
            free_collateral: Available collateral
            
        Returns:
            True if margin is safe, False otherwise
        """
        try:
            # Emergency stop for critically low margin
            if margin_ratio < 1.0:
                self.logger.warning(f"CRITICAL: Margin ratio {margin_ratio:.2f}x - Emergency stop")
                return False
            
            # Check if we can afford minimum position
            if free_collateral < 10.0:  # Minimum $10 collateral
                self.logger.warning(f"Insufficient collateral: ${free_collateral:.2f}")
                return False
            
            # Standard margin check
            if margin_ratio < self.margin_ratio_threshold:
                self.logger.warning(f"Margin ratio too low: {margin_ratio:.2f}x")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking margin ratio: {e}")
            return False
    
    def update_trade_result(self, trade_pnl: float, trade_size: float, 
                          trade_type: str, success: bool) -> None:
        """
        Update risk tracking with trade result
        
        Args:
            trade_pnl: Profit/loss from trade
            trade_size: Size of trade
            trade_type: Type of trade (LONG/SHORT)
            success: Whether trade was successful
        """
        try:
            # Update daily PnL
            self.daily_pnl += trade_pnl
            
            # Track trade
            trade_record = {
                'timestamp': datetime.now(),
                'pnl': trade_pnl,
                'size': trade_size,
                'type': trade_type,
                'success': success
            }
            self.daily_trades.append(trade_record)
            
            # Update consecutive losses
            if trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                self.winning_trades += 1
            
            self.total_trades += 1
            
            # Log trade result
            self.logger.info(f"Trade result: ${trade_pnl:.2f} ({trade_type} {trade_size:.1f} XRP)")
            self.logger.info(f"Daily PnL: ${self.daily_pnl:.2f}, Consecutive losses: {self.consecutive_losses}")
            
        except Exception as e:
            self.logger.error(f"Error updating trade result: {e}")
    
    def get_daily_pnl(self) -> float:
        """Get current daily PnL"""
        try:
            # Reset daily PnL if 24 hours have passed
            current_time = time.time()
            if current_time - self.last_daily_reset > 86400:  # 24 hours
                self.daily_pnl = 0.0
                self.daily_trades = []
                self.last_daily_reset = current_time
                self.logger.info("Daily PnL reset")
            
            return self.daily_pnl
            
        except Exception as e:
            self.logger.error(f"Error getting daily PnL: {e}")
            return 0.0
    
    def get_win_rate(self) -> float:
        """Get current win rate"""
        try:
            if self.total_trades == 0:
                return 0.0
            
            return self.winning_trades / self.total_trades
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def get_drawdown(self) -> float:
        """Get current drawdown percentage"""
        try:
            if not self.initial_capital or not self.peak_capital:
                return 0.0
            
            current_value = self.initial_capital + self.daily_pnl
            drawdown = (self.peak_capital - current_value) / self.peak_capital
            return max(drawdown, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    def adapt_risk_parameters(self) -> None:
        """Adapt risk parameters based on performance"""
        try:
            win_rate = self.get_win_rate()
            drawdown = self.get_drawdown()
            
            # Adjust risk based on performance
            if win_rate < 0.4:  # Low win rate
                self.max_risk_per_trade *= 0.9  # Reduce risk
                self.logger.info(f"Adapting - Low win rate: confidence={win_rate:.2f}, risk={self.max_risk_per_trade:.3f}")
            
            elif win_rate > 0.6:  # High win rate
                self.max_risk_per_trade *= 1.05  # Slightly increase risk
                self.logger.info(f"Adapting - High win rate: confidence={win_rate:.2f}, risk={self.max_risk_per_trade:.3f}")
            
            # Adjust for drawdown
            if drawdown > 0.1:  # High drawdown
                self.max_risk_per_trade *= 0.8  # Reduce risk significantly
                self.logger.info(f"Adapting - High drawdown: {drawdown:.2%}, risk={self.max_risk_per_trade:.3f}")
            
            # Ensure risk stays within bounds
            self.max_risk_per_trade = max(self.max_risk_per_trade, 0.005)  # Min 0.5%
            self.max_risk_per_trade = min(self.max_risk_per_trade, 0.05)   # Max 5%
            
        except Exception as e:
            self.logger.error(f"Error adapting risk parameters: {e}")
    
    def calculate_compound_position_size(self, base_size: float, 
                                       compound_factor: float = 1.1) -> float:
        """
        Calculate compound position size based on performance
        
        Args:
            base_size: Base position size
            compound_factor: Compound factor for winning streaks
            
        Returns:
            Compound position size
        """
        try:
            win_rate = self.get_win_rate()
            
            # Apply compound factor based on win rate
            if win_rate > 0.6 and self.consecutive_losses == 0:
                compound_size = base_size * compound_factor
            else:
                compound_size = base_size
            
            # Apply size constraints
            compound_size = max(compound_size, self.min_position_size)
            compound_size = min(compound_size, self.max_position_size)
            
            self.logger.info(f"Compound sizing: base={base_size}, factor={compound_factor:.2f}, final={compound_size:.1f} XRP")
            
            return compound_size
            
        except Exception as e:
            self.logger.error(f"Error calculating compound position size: {e}")
            return base_size
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            return {
                'daily_pnl': self.daily_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': self.get_win_rate(),
                'consecutive_losses': self.consecutive_losses,
                'drawdown': self.get_drawdown(),
                'initial_capital': self.initial_capital,
                'peak_capital': self.peak_capital,
                'current_capital': self.initial_capital + self.daily_pnl if self.initial_capital else None,
                'risk_per_trade': self.max_risk_per_trade,
                'daily_trades_count': len(self.daily_trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}

# Convenience functions for backward compatibility
def calculate_position_size(current_price: float, free_collateral: float, 
                          confidence: float, signal_type: str,
                          logger: Optional[logging.Logger] = None) -> float:
    """Calculate position size (convenience function)"""
    risk_manager = RiskManager(logger)
    return risk_manager.calculate_position_size(current_price, free_collateral, confidence, signal_type)

def check_risk_limits(trade_value: float, signal_confidence: float,
                     logger: Optional[logging.Logger] = None) -> bool:
    """Check risk limits (convenience function)"""
    risk_manager = RiskManager(logger)
    return risk_manager.check_risk_limits(trade_value, signal_confidence)


