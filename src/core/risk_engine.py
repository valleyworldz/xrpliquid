#!/usr/bin/env python3
"""
Risk Management Engine
=====================

This module centralizes all risk checks and returns typed RiskDecision enums.
The trading loop just pattern-matches on the results.
"""

import time
import logging
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.core.config import TradingConfig
from src.core.state import RuntimeState


class RiskDecision(Enum):
    """Risk assessment decisions"""
    OK = "ok"
    COOL_DOWN = "cool_down"
    LIMIT_HIT = "limit_hit"
    BAD_RR = "bad_risk_reward"
    BAD_FUNDING = "bad_funding"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    BAD_ATR = "bad_atr"
    POSITION_TOO_SMALL = "position_too_small"
    POSITION_TOO_LARGE = "position_too_large"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    HOLD_TIME_VIOLATION = "hold_time_violation"


@dataclass
class RiskAssessment:
    """Risk assessment result"""
    decision: RiskDecision
    details: str
    metadata: Optional[Dict[str, Any]] = None


class RiskEngine:
    """Centralized risk management engine"""
    
    def __init__(self, config: TradingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
    def assess_trade_risk(self, state: RuntimeState, signal: Dict[str, Any], 
                         entry_price: float, tp_price: float, sl_price: float,
                         position_size: int, free_collateral: float,
                         atr: float, funding_rate: Optional[float] = None) -> RiskAssessment:
        """Comprehensive risk assessment for a potential trade"""
        
        # Check cooldown
        cooldown_check = self._check_cooldown(state)
        if cooldown_check.decision != RiskDecision.OK:
            return cooldown_check
            
        # Check daily limits
        daily_check = self._check_daily_limits(state)
        if daily_check.decision != RiskDecision.OK:
            return daily_check
            
        # Check consecutive losses
        consecutive_check = self._check_consecutive_losses(state)
        if consecutive_check.decision != RiskDecision.OK:
            return consecutive_check
            
        # Check margin requirements
        margin_check = self._check_margin_requirements(free_collateral, position_size, entry_price)
        if margin_check.decision != RiskDecision.OK:
            return margin_check
            
        # Check risk/reward ratio
        rr_check = self._check_risk_reward(entry_price, tp_price, sl_price)
        if rr_check.decision != RiskDecision.OK:
            return rr_check
            
        # Check ATR validity
        atr_check = self._check_atr_validity(atr, entry_price)
        if atr_check.decision != RiskDecision.OK:
            return atr_check
            
        # Check position size
        size_check = self._check_position_size(position_size, entry_price)
        if size_check.decision != RiskDecision.OK:
            return size_check
            
        # Check funding rate
        if funding_rate is not None:
            funding_check = self._check_funding_risk(funding_rate, signal.get('side', 'long'))
            if funding_check.decision != RiskDecision.OK:
                return funding_check
                
        # All checks passed
        return RiskAssessment(
            decision=RiskDecision.OK,
            details="All risk checks passed",
            metadata={
                'risk_reward_ratio': abs(tp_price - entry_price) / abs(sl_price - entry_price),
                'position_value': position_size * entry_price,
                'margin_ratio': (free_collateral * entry_price) / (position_size * entry_price) if position_size > 0 else float('inf')
            }
        )
        
    def _check_cooldown(self, state: RuntimeState) -> RiskAssessment:
        """Check if enough time has passed since last trade"""
        if state.last_trade_time is None:
            return RiskAssessment(RiskDecision.OK, "No previous trade")
            
        time_since_last = time.time() - state.last_trade_time
        min_interval = getattr(self.config, 'MIN_TRADE_INTERVAL', 300)  # 5 minutes default
        
        if time_since_last < min_interval:
            remaining = min_interval - time_since_last
            return RiskAssessment(
                RiskDecision.COOL_DOWN,
                f"Trade cooldown active, {remaining:.0f}s remaining",
                {'remaining_seconds': remaining}
            )
            
        return RiskAssessment(RiskDecision.OK, "Cooldown period passed")
        
    def _check_daily_limits(self, state: RuntimeState) -> RiskAssessment:
        """Check daily trading limits"""
        max_daily_trades = getattr(self.config, 'MAX_TRADES_PER_HOUR', 3) * 24  # Convert hourly to daily
        max_daily_loss = getattr(self.config, 'MAX_DRAWDOWN_PCT', 0.03)
        
        # Check trade count
        if state.daily_trades >= max_daily_trades:
            return RiskAssessment(
                RiskDecision.LIMIT_HIT,
                f"Daily trade limit reached ({state.daily_trades}/{max_daily_trades})"
            )
            
        # Check daily loss
        if state.daily_pnl < 0 and abs(state.daily_pnl) > max_daily_loss:
            return RiskAssessment(
                RiskDecision.DAILY_LOSS_LIMIT,
                f"Daily loss limit exceeded ({state.daily_pnl:.2%})"
            )
            
        return RiskAssessment(RiskDecision.OK, "Daily limits OK")
        
    def _check_consecutive_losses(self, state: RuntimeState) -> RiskAssessment:
        """Check consecutive loss limit"""
        max_consecutive = getattr(self.config, 'MAX_CONSECUTIVE_LOSSES', 3)
        
        if state.consecutive_losses >= max_consecutive:
            return RiskAssessment(
                RiskDecision.CONSECUTIVE_LOSSES,
                f"Consecutive loss limit reached ({state.consecutive_losses})"
            )
            
        return RiskAssessment(RiskDecision.OK, "Consecutive losses OK")
        
    def _check_margin_requirements(self, free_collateral: float, position_size: int, 
                                  current_price: float) -> RiskAssessment:
        """Check margin requirements"""
        position_value = position_size * current_price
        margin_ratio = free_collateral / position_value if position_value > 0 else float('inf')
        min_margin_ratio = 2.0  # 2:1 margin ratio minimum
        
        if margin_ratio < min_margin_ratio:
            return RiskAssessment(
                RiskDecision.INSUFFICIENT_MARGIN,
                f"Insufficient margin ratio ({margin_ratio:.2f} < {min_margin_ratio})",
                {'margin_ratio': margin_ratio, 'required': min_margin_ratio}
            )
            
        return RiskAssessment(RiskDecision.OK, "Margin requirements met")
        
    def _check_risk_reward(self, entry_price: float, tp_price: float, sl_price: float) -> RiskAssessment:
        """Check risk/reward ratio"""
        if entry_price == 0:
            return RiskAssessment(RiskDecision.BAD_RR, "Invalid entry price")
            
        # Calculate R:R ratio
        if entry_price > tp_price:  # Short position
            reward = entry_price - tp_price
            risk = sl_price - entry_price
        else:  # Long position
            reward = tp_price - entry_price
            risk = entry_price - sl_price
            
        if risk <= 0:
            return RiskAssessment(RiskDecision.BAD_RR, "Invalid stop loss")
            
        rr_ratio = reward / risk
        min_rr = 2.0  # Minimum 2:1 R:R ratio
        
        if rr_ratio < min_rr:
            return RiskAssessment(
                RiskDecision.BAD_RR,
                f"Risk/reward ratio too low ({rr_ratio:.2f} < {min_rr})",
                {'rr_ratio': rr_ratio, 'min_required': min_rr}
            )
            
        return RiskAssessment(RiskDecision.OK, f"R:R ratio acceptable ({rr_ratio:.2f})")
        
    def _check_atr_validity(self, atr: float, entry_price: float) -> RiskAssessment:
        """Check if ATR is within valid range"""
        if atr <= 0:
            return RiskAssessment(RiskDecision.BAD_ATR, "Invalid ATR value")
            
        atr_pct = atr / entry_price if entry_price > 0 else 0
        min_atr = getattr(self.config, 'MIN_ATR_FOR_ENTRY', 0.001)  # 0.1%
        max_atr = getattr(self.config, 'MAX_ATR_FOR_ENTRY', 0.005)  # 0.5%
        
        if atr_pct < min_atr:
            return RiskAssessment(
                RiskDecision.BAD_ATR,
                f"ATR too small ({atr_pct:.3%} < {min_atr:.3%})"
            )
            
        if atr_pct > max_atr:
            return RiskAssessment(
                RiskDecision.BAD_ATR,
                f"ATR too large ({atr_pct:.3%} > {max_atr:.3%})"
            )
            
        return RiskAssessment(RiskDecision.OK, f"ATR within range ({atr_pct:.3%})")
        
    def _check_position_size(self, position_size: int, current_price: float) -> RiskAssessment:
        """Check position size limits"""
        position_value = position_size * current_price
        min_notional = getattr(self.config, 'MIN_NOTIONAL', 10.0)
        max_position_pct = getattr(self.config, 'MAX_POSITION_SIZE_PCT', 0.25)
        
        # Check minimum size
        if position_value < min_notional:
            return RiskAssessment(
                RiskDecision.POSITION_TOO_SMALL,
                f"Position too small (${position_value:.2f} < ${min_notional})"
            )
            
        # Check maximum size (would need account value for full check)
        # This is a simplified check - full check requires account info
        if position_size > 1000:  # Arbitrary large number
            return RiskAssessment(
                RiskDecision.POSITION_TOO_LARGE,
                f"Position too large ({position_size} units)"
            )
            
        return RiskAssessment(RiskDecision.OK, "Position size acceptable")
        
    def _check_funding_risk(self, funding_rate: float, signal_type: str) -> RiskAssessment:
        """Check funding rate risk"""
        if funding_rate is None:
            return RiskAssessment(RiskDecision.OK, "Funding rate unknown - skipping check")
            
        threshold = getattr(self.config, 'funding_rate_threshold', 0.0001)
        
        # For long positions, negative funding is bad
        # For short positions, positive funding is bad
        if signal_type.lower() == 'long' and funding_rate < -threshold:
            return RiskAssessment(
                RiskDecision.BAD_FUNDING,
                f"Adverse funding rate for long ({funding_rate:.4%})"
            )
        elif signal_type.lower() == 'short' and funding_rate > threshold:
            return RiskAssessment(
                RiskDecision.BAD_FUNDING,
                f"Adverse funding rate for short ({funding_rate:.4%})"
            )
            
        return RiskAssessment(RiskDecision.OK, f"Funding rate acceptable ({funding_rate:.4%})")
        
    def calculate_position_size(self, free_collateral: float, entry_price: float, 
                               sl_price: float, confidence: float) -> int:
        """Calculate position size based on risk parameters"""
        if entry_price <= 0 or sl_price <= 0:
            return 0
            
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit <= 0:
            return 0
            
        # Calculate risk amount (2% of collateral)
        risk_amount = free_collateral * getattr(self.config, 'RISK_PER_TRADE', 0.02)
        
        # Apply confidence scaling
        risk_amount *= confidence
        
        # Calculate position size
        position_size = int(risk_amount / risk_per_unit)
        
        # Apply minimum size
        min_size = int(getattr(self.config, 'MIN_POSITION_SIZE', 10.0))
        position_size = max(position_size, min_size)
        
        return position_size
        
    def calculate_fee_impact(self, entry_price: float, position_size: int, 
                           est_fee: Optional[float] = None) -> float:
        """Calculate fee impact on trade"""
        if est_fee is None:
            est_fee = getattr(self.config, 'TAKER_FEE', 0.00045)
            
        position_value = position_size * entry_price
        fee_cost = position_value * est_fee
        
        return fee_cost
        
    def should_skip_by_funding(self, funding_rate: Optional[float], signal_type: str) -> bool:
        """Determine if trade should be skipped due to funding rate"""
        if funding_rate is None:
            return False  # Skip if we can't determine funding rate
            
        threshold = getattr(self.config, 'funding_rate_threshold', 0.0001)
        
        if signal_type.lower() == 'long' and funding_rate < -threshold:
            return True
        elif signal_type.lower() == 'short' and funding_rate > threshold:
            return True
            
        return False 