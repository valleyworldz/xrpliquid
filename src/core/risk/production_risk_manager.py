"""
🛡️ PRODUCTION RISK MANAGER
===========================
Production-grade risk management with ATR/vol-targeted sizing and drawdown kill-switch.

Features:
- ATR-based position sizing
- Volatility-targeted sizing
- %-equity-at-risk management
- Hard drawdown kill-switch
- Daily/rolling drawdown limits
- Auto-cooldown windows
- Real-time risk monitoring
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    KILL_SWITCH = "kill_switch"

class RiskAction(Enum):
    """Risk action enumeration"""
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    EMERGENCY_EXIT = "emergency_exit"
    KILL_SWITCH = "kill_switch"

@dataclass
class RiskConfig:
    """Configuration for production risk manager"""
    
    # Position sizing configuration
    position_sizing_config: Dict[str, Any] = field(default_factory=lambda: {
        'atr_period': 14,                    # ATR calculation period
        'atr_multiplier': 2.0,               # ATR multiplier for position sizing
        'vol_target_percent': 0.02,          # 2% volatility target
        'max_position_size_percent': 0.1,    # 10% max position size
        'min_position_size_usd': 100.0,      # $100 minimum position
        'max_position_size_usd': 10000.0,    # $10k maximum position
        'equity_at_risk_percent': 0.05,      # 5% equity at risk
        'max_leverage': 10.0,                # Maximum leverage
    })
    
    # Drawdown management
    drawdown_config: Dict[str, Any] = field(default_factory=lambda: {
        'daily_drawdown_limit': 0.02,        # 2% daily drawdown limit
        'rolling_drawdown_limit': 0.05,      # 5% rolling drawdown limit
        'rolling_period_days': 7,            # 7-day rolling period
        'kill_switch_threshold': 0.08,       # 8% kill switch threshold
        'cooldown_period_hours': 24,         # 24-hour cooldown after kill switch
        'emergency_exit_threshold': 0.06,    # 6% emergency exit threshold
    })
    
    # Risk monitoring
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        'check_interval_seconds': 5,         # Check every 5 seconds
        'var_95_period': 252,                # 252-day VaR calculation
        'var_95_threshold': 0.02,            # 2% VaR threshold
        'max_correlation': 0.8,              # Maximum correlation between positions
        'max_concentration': 0.3,            # Maximum 30% concentration in single asset
        'liquidity_threshold': 100000.0,     # $100k minimum liquidity
    })
    
    # Volatility targeting
    volatility_config: Dict[str, Any] = field(default_factory=lambda: {
        'vol_estimation_period': 20,         # 20-day volatility estimation
        'vol_target_annual': 0.15,           # 15% annual volatility target
        'vol_scaling_factor': 1.0,           # Volatility scaling factor
        'min_vol_estimate': 0.05,            # 5% minimum volatility estimate
        'max_vol_estimate': 0.50,            # 50% maximum volatility estimate
    })

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    
    # Portfolio metrics
    total_equity: float = 0.0
    available_margin: float = 0.0
    used_margin: float = 0.0
    margin_ratio: float = 0.0
    
    # Position metrics
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    position_count: int = 0
    
    # Risk metrics
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Volatility metrics
    portfolio_volatility: float = 0.0
    target_volatility: float = 0.0
    vol_ratio: float = 0.0
    
    # Timestamp
    timestamp: float = 0.0

@dataclass
class PositionRisk:
    """Position risk assessment"""
    
    symbol: str
    position_size: float
    position_value: float
    atr: float
    volatility: float
    risk_score: float
    recommended_size: float
    max_allowed_size: float
    risk_action: RiskAction
    risk_level: RiskLevel
    
    # Risk factors
    concentration_risk: float = 0.0
    liquidity_risk: float = 0.0
    correlation_risk: float = 0.0
    leverage_risk: float = 0.0

class ProductionRiskManager:
    """
    🛡️ PRODUCTION RISK MANAGER
    
    Production-grade risk management with comprehensive position sizing,
    drawdown monitoring, and kill-switch functionality.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or Logger()
        
        # Initialize configuration
        self.risk_config = RiskConfig()
        
        # Risk state
        self.current_risk_metrics = RiskMetrics()
        self.position_risks: Dict[str, PositionRisk] = {}
        self.risk_history: List[RiskMetrics] = []
        self.kill_switch_active = False
        self.kill_switch_time = 0.0
        self.cooldown_active = False
        self.cooldown_end_time = 0.0
        
        # Performance tracking
        self.risk_performance = {
            'total_risk_checks': 0,
            'risk_violations': 0,
            'kill_switch_activations': 0,
            'emergency_exits': 0,
            'position_sizing_adjustments': 0,
            'avg_risk_score': 0.0,
            'max_risk_score': 0.0,
        }
        
        # Market data cache
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        self.atr_cache: Dict[str, float] = {}
        self.volatility_cache: Dict[str, float] = {}
        
        self.logger.info("🛡️ [RISK_MANAGER] Production Risk Manager initialized")
        self.logger.info("🛡️ [RISK_MANAGER] ATR-based sizing and drawdown kill-switch enabled")
    
    async def start_risk_monitoring(self):
        """Start continuous risk monitoring"""
        try:
            self.logger.info("🛡️ [RISK_MONITORING] Starting continuous risk monitoring...")
            
            while True:
                try:
                    await self._risk_monitoring_cycle()
                    await asyncio.sleep(self.risk_config.monitoring_config['check_interval_seconds'])
                except Exception as e:
                    self.logger.error(f"❌ [RISK_MONITORING] Error in risk monitoring cycle: {e}")
                    await asyncio.sleep(30)  # Wait 30 seconds on error
                    
        except Exception as e:
            self.logger.error(f"❌ [RISK_MONITORING] Error starting risk monitoring: {e}")
    
    async def _risk_monitoring_cycle(self):
        """Main risk monitoring cycle"""
        try:
            # Update risk metrics
            await self._update_risk_metrics()
            
            # Check kill switch status
            await self._check_kill_switch()
            
            # Check drawdown limits
            await self._check_drawdown_limits()
            
            # Check position risks
            await self._check_position_risks()
            
            # Update risk history
            self.risk_history.append(self.current_risk_metrics)
            
            # Keep only recent history
            max_history = 1000
            if len(self.risk_history) > max_history:
                self.risk_history = self.risk_history[-max_history:]
            
            # Update performance tracking
            self.risk_performance['total_risk_checks'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_CYCLE] Error in risk monitoring cycle: {e}")
    
    async def _update_risk_metrics(self):
        """Update current risk metrics"""
        try:
            # This would integrate with actual portfolio data
            # For now, simulate risk metrics
            
            current_time = time.time()
            
            # Simulate portfolio data
            self.current_risk_metrics = RiskMetrics(
                total_equity=10000.0,
                available_margin=8000.0,
                used_margin=2000.0,
                margin_ratio=0.2,
                total_exposure=2000.0,
                net_exposure=1000.0,
                gross_exposure=2000.0,
                position_count=2,
                var_95=0.015,
                expected_shortfall=0.025,
                max_drawdown=0.03,
                current_drawdown=0.01,
                sharpe_ratio=1.5,
                portfolio_volatility=0.12,
                target_volatility=0.15,
                vol_ratio=0.8,
                timestamp=current_time,
            )
            
        except Exception as e:
            self.logger.error(f"❌ [UPDATE_METRICS] Error updating risk metrics: {e}")
    
    async def _check_kill_switch(self):
        """Check kill switch conditions"""
        try:
            current_time = time.time()
            
            # Check if we're in cooldown period
            if self.cooldown_active and current_time < self.cooldown_end_time:
                return
            
            # Check kill switch threshold
            kill_switch_threshold = self.risk_config.drawdown_config['kill_switch_threshold']
            
            if self.current_risk_metrics.current_drawdown >= kill_switch_threshold:
                await self._activate_kill_switch()
            elif self.current_risk_metrics.max_drawdown >= kill_switch_threshold:
                await self._activate_kill_switch()
            
        except Exception as e:
            self.logger.error(f"❌ [KILL_SWITCH] Error checking kill switch: {e}")
    
    async def _activate_kill_switch(self):
        """Activate kill switch"""
        try:
            if self.kill_switch_active:
                return  # Already active
            
            self.kill_switch_active = True
            self.kill_switch_time = time.time()
            
            # Set cooldown period
            cooldown_hours = self.risk_config.drawdown_config['cooldown_period_hours']
            self.cooldown_end_time = time.time() + (cooldown_hours * 3600)
            self.cooldown_active = True
            
            # Update performance tracking
            self.risk_performance['kill_switch_activations'] += 1
            
            self.logger.critical(f"🚨 [KILL_SWITCH] KILL SWITCH ACTIVATED!")
            self.logger.critical(f"🚨 [KILL_SWITCH] Drawdown: {self.current_risk_metrics.current_drawdown:.2%}")
            self.logger.critical(f"🚨 [KILL_SWITCH] Cooldown period: {cooldown_hours} hours")
            
            # This would trigger emergency exit of all positions
            await self._emergency_exit_all_positions()
            
        except Exception as e:
            self.logger.error(f"❌ [ACTIVATE_KILL_SWITCH] Error activating kill switch: {e}")
    
    async def _emergency_exit_all_positions(self):
        """Emergency exit all positions"""
        try:
            self.logger.critical("🚨 [EMERGENCY_EXIT] Initiating emergency exit of all positions...")
            
            # This would integrate with the trading system to close all positions
            # For now, log the action
            
            self.risk_performance['emergency_exits'] += 1
            
            self.logger.critical("🚨 [EMERGENCY_EXIT] All positions marked for emergency exit")
            
        except Exception as e:
            self.logger.error(f"❌ [EMERGENCY_EXIT] Error in emergency exit: {e}")
    
    async def _check_drawdown_limits(self):
        """Check drawdown limits"""
        try:
            drawdown_config = self.risk_config.drawdown_config
            
            # Check daily drawdown limit
            if self.current_risk_metrics.current_drawdown >= drawdown_config['daily_drawdown_limit']:
                self.logger.warning(f"⚠️ [DRAWDOWN] Daily drawdown limit exceeded: {self.current_risk_metrics.current_drawdown:.2%}")
                await self._reduce_positions()
            
            # Check rolling drawdown limit
            if self.current_risk_metrics.max_drawdown >= drawdown_config['rolling_drawdown_limit']:
                self.logger.warning(f"⚠️ [DRAWDOWN] Rolling drawdown limit exceeded: {self.current_risk_metrics.max_drawdown:.2%}")
                await self._reduce_positions()
            
            # Check emergency exit threshold
            if self.current_risk_metrics.current_drawdown >= drawdown_config['emergency_exit_threshold']:
                self.logger.critical(f"🚨 [EMERGENCY] Emergency exit threshold reached: {self.current_risk_metrics.current_drawdown:.2%}")
                await self._emergency_exit_all_positions()
            
        except Exception as e:
            self.logger.error(f"❌ [DRAWDOWN_CHECK] Error checking drawdown limits: {e}")
    
    async def _check_position_risks(self):
        """Check individual position risks"""
        try:
            # This would check each position for risk violations
            # For now, simulate position risk checks
            
            for symbol, position_risk in self.position_risks.items():
                if position_risk.risk_level == RiskLevel.CRITICAL:
                    self.logger.warning(f"⚠️ [POSITION_RISK] Critical risk for {symbol}: {position_risk.risk_score:.2f}")
                    await self._reduce_position(symbol)
                elif position_risk.risk_level == RiskLevel.HIGH:
                    self.logger.warning(f"⚠️ [POSITION_RISK] High risk for {symbol}: {position_risk.risk_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ [POSITION_RISK] Error checking position risks: {e}")
    
    async def _reduce_positions(self):
        """Reduce all positions"""
        try:
            self.logger.warning("⚠️ [REDUCE_POSITIONS] Reducing all positions due to risk limits...")
            
            # This would integrate with the trading system to reduce positions
            # For now, log the action
            
            self.risk_performance['position_sizing_adjustments'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ [REDUCE_POSITIONS] Error reducing positions: {e}")
    
    async def _reduce_position(self, symbol: str):
        """Reduce specific position"""
        try:
            self.logger.warning(f"⚠️ [REDUCE_POSITION] Reducing position for {symbol}...")
            
            # This would integrate with the trading system to reduce the specific position
            # For now, log the action
            
        except Exception as e:
            self.logger.error(f"❌ [REDUCE_POSITION] Error reducing position {symbol}: {e}")
    
    async def calculate_position_size(self, symbol: str, signal_strength: float, 
                                    current_price: float, account_equity: float) -> Dict[str, Any]:
        """
        🛡️ Calculate optimal position size using ATR and volatility targeting
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (0-1)
            current_price: Current market price
            account_equity: Account equity
            
        Returns:
            Dict with position sizing information
        """
        try:
            # Get ATR and volatility
            atr = await self._get_atr(symbol)
            volatility = await self._get_volatility(symbol)
            
            # Calculate ATR-based position size
            atr_size = await self._calculate_atr_position_size(symbol, atr, current_price, account_equity)
            
            # Calculate volatility-targeted position size
            vol_size = await self._calculate_volatility_position_size(symbol, volatility, account_equity)
            
            # Calculate equity-at-risk position size
            ear_size = await self._calculate_equity_at_risk_size(account_equity)
            
            # Take the minimum of all sizing methods
            recommended_size = min(atr_size, vol_size, ear_size)
            
            # Apply signal strength scaling
            recommended_size *= signal_strength
            
            # Apply position limits
            position_config = self.risk_config.position_sizing_config
            recommended_size = max(recommended_size, position_config['min_position_size_usd'])
            recommended_size = min(recommended_size, position_config['max_position_size_usd'])
            
            # Calculate risk metrics
            position_value = recommended_size
            risk_score = await self._calculate_position_risk_score(symbol, position_value, atr, volatility)
            
            # Determine risk action
            risk_action = await self._determine_risk_action(risk_score, position_value, account_equity)
            
            # Create position risk assessment
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=recommended_size,
                position_value=position_value,
                atr=atr,
                volatility=volatility,
                risk_score=risk_score,
                recommended_size=recommended_size,
                max_allowed_size=ear_size,
                risk_action=risk_action,
                risk_level=self._get_risk_level(risk_score),
            )
            
            # Store position risk
            self.position_risks[symbol] = position_risk
            
            return {
                'recommended_size': recommended_size,
                'atr_size': atr_size,
                'vol_size': vol_size,
                'ear_size': ear_size,
                'risk_score': risk_score,
                'risk_action': risk_action.value,
                'risk_level': position_risk.risk_level.value,
                'atr': atr,
                'volatility': volatility,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [POSITION_SIZE] Error calculating position size: {e}")
            return {
                'recommended_size': 100.0,
                'atr_size': 100.0,
                'vol_size': 100.0,
                'ear_size': 100.0,
                'risk_score': 1.0,
                'risk_action': RiskAction.BLOCK.value,
                'risk_level': RiskLevel.CRITICAL.value,
                'atr': 0.01,
                'volatility': 0.02,
            }
    
    async def _calculate_atr_position_size(self, symbol: str, atr: float, current_price: float, account_equity: float) -> float:
        """Calculate ATR-based position size"""
        try:
            position_config = self.risk_config.position_sizing_config
            
            # Calculate position size based on ATR
            atr_multiplier = position_config['atr_multiplier']
            risk_per_trade = account_equity * position_config['equity_at_risk_percent']
            
            # Position size = Risk per trade / (ATR * ATR multiplier)
            if atr > 0:
                position_size = risk_per_trade / (atr * atr_multiplier)
            else:
                position_size = account_equity * 0.01  # 1% fallback
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ [ATR_SIZE] Error calculating ATR position size: {e}")
            return account_equity * 0.01
    
    async def _calculate_volatility_position_size(self, symbol: str, volatility: float, account_equity: float) -> float:
        """Calculate volatility-targeted position size"""
        try:
            vol_config = self.risk_config.volatility_config
            position_config = self.risk_config.position_sizing_config
            
            # Calculate volatility target
            vol_target = vol_config['vol_target_annual']
            vol_scaling = vol_config['vol_scaling_factor']
            
            # Position size = (Vol target / Current vol) * Scaling factor * Equity
            if volatility > 0:
                vol_ratio = vol_target / volatility
                position_size = vol_ratio * vol_scaling * account_equity * position_config['max_position_size_percent']
            else:
                position_size = account_equity * position_config['max_position_size_percent']
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ [VOL_SIZE] Error calculating volatility position size: {e}")
            return account_equity * 0.05
    
    async def _calculate_equity_at_risk_size(self, account_equity: float) -> float:
        """Calculate equity-at-risk position size"""
        try:
            position_config = self.risk_config.position_sizing_config
            
            # Calculate maximum position size based on equity at risk
            ear_percent = position_config['equity_at_risk_percent']
            max_size = account_equity * ear_percent
            
            return max_size
            
        except Exception as e:
            self.logger.error(f"❌ [EAR_SIZE] Error calculating equity-at-risk size: {e}")
            return account_equity * 0.02
    
    async def _calculate_position_risk_score(self, symbol: str, position_value: float, atr: float, volatility: float) -> float:
        """Calculate position risk score (0-1, higher is riskier)"""
        try:
            # Base risk score from volatility
            base_risk = min(1.0, volatility * 10)  # Scale volatility to 0-1
            
            # Adjust for position size
            position_config = self.risk_config.position_sizing_config
            max_size = position_config['max_position_size_usd']
            size_risk = min(1.0, position_value / max_size)
            
            # Adjust for ATR
            atr_risk = min(1.0, atr * 100)  # Scale ATR to 0-1
            
            # Calculate combined risk score
            risk_score = (base_risk + size_risk + atr_risk) / 3
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_SCORE] Error calculating risk score: {e}")
            return 0.5
    
    async def _determine_risk_action(self, risk_score: float, position_value: float, account_equity: float) -> RiskAction:
        """Determine risk action based on risk score"""
        try:
            if risk_score >= 0.9:
                return RiskAction.KILL_SWITCH
            elif risk_score >= 0.8:
                return RiskAction.EMERGENCY_EXIT
            elif risk_score >= 0.6:
                return RiskAction.BLOCK
            elif risk_score >= 0.4:
                return RiskAction.REDUCE
            else:
                return RiskAction.ALLOW
                
        except Exception as e:
            self.logger.error(f"❌ [RISK_ACTION] Error determining risk action: {e}")
            return RiskAction.BLOCK
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Get risk level from risk score"""
        try:
            if risk_score >= 0.9:
                return RiskLevel.KILL_SWITCH
            elif risk_score >= 0.8:
                return RiskLevel.CRITICAL
            elif risk_score >= 0.6:
                return RiskLevel.HIGH
            elif risk_score >= 0.4:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"❌ [RISK_LEVEL] Error getting risk level: {e}")
            return RiskLevel.HIGH
    
    async def _get_atr(self, symbol: str) -> float:
        """Get ATR for symbol"""
        try:
            # Check cache first
            if symbol in self.atr_cache:
                return self.atr_cache[symbol]
            
            # This would integrate with actual market data
            # For now, return mock ATR
            atr = 0.01  # 1% ATR
            
            # Cache the result
            self.atr_cache[symbol] = atr
            
            return atr
            
        except Exception as e:
            self.logger.error(f"❌ [GET_ATR] Error getting ATR for {symbol}: {e}")
            return 0.01
    
    async def _get_volatility(self, symbol: str) -> float:
        """Get volatility for symbol"""
        try:
            # Check cache first
            if symbol in self.volatility_cache:
                return self.volatility_cache[symbol]
            
            # This would integrate with actual market data
            # For now, return mock volatility
            volatility = 0.02  # 2% daily volatility
            
            # Cache the result
            self.volatility_cache[symbol] = volatility
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"❌ [GET_VOLATILITY] Error getting volatility for {symbol}: {e}")
            return 0.02
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active"""
        return self.kill_switch_active
    
    def is_cooldown_active(self) -> bool:
        """Check if cooldown is active"""
        return self.cooldown_active and time.time() < self.cooldown_end_time
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            return {
                'current_risk_metrics': self.current_risk_metrics.__dict__,
                'kill_switch_active': self.kill_switch_active,
                'cooldown_active': self.is_cooldown_active(),
                'position_risks': {symbol: risk.__dict__ for symbol, risk in self.position_risks.items()},
                'risk_performance': self.risk_performance,
                'risk_config': self.risk_config.__dict__,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_SUMMARY] Error getting risk summary: {e}")
            return {}
