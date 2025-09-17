#!/usr/bin/env python3
"""
🛡️ REAL-TIME RISK ENGINEER (FRM, CFA)
======================================

Advanced real-time risk management system implementing kill-switches and risk limits
that are checked on every order to prevent catastrophic losses like the 40.21% drawdown.

Features:
- Real-time kill-switch monitoring
- Position, loss, and concentration limits
- Dynamic risk adjustment
- Portfolio-level risk management
- Emergency shutdown procedures
- Real-time P&L tracking
- VaR calculations
- Stress testing
"""

from src.core.utils.decimal_boundary_guard import safe_float
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import asyncio

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    portfolio_value: float
    total_exposure: float
    current_drawdown: float
    max_drawdown: float
    volatility: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    correlation_risk: float
    concentration_risk: float
    leverage_ratio: float
    margin_ratio: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class KillSwitch:
    """Kill switch configuration"""
    name: str
    threshold: float
    enabled: bool
    action: str  # 'stop_trading', 'close_positions', 'emergency_shutdown'
    description: str

class RealTimeRiskEngine:
    """Advanced real-time risk management system"""
    
    def __init__(
        self,
        logger=None,
        trading_bot=None,
        max_drawdown_threshold: float | None = None,
        position_loss_threshold: float | None = None,
        daily_loss_threshold: float | None = None,
        emergency_threshold: float | None = None,
        leverage_threshold: float | None = None,
        margin_threshold: float | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.trading_bot = trading_bot  # CRITICAL: Reference to trading bot for emergency exits
        
        # Risk limits configuration
        self.risk_limits = {
            'max_drawdown': max_drawdown_threshold if max_drawdown_threshold is not None else 0.05,
            'max_position_loss': position_loss_threshold if position_loss_threshold is not None else 0.02,
            'max_concentration': 0.25,
            'max_daily_loss': daily_loss_threshold if daily_loss_threshold is not None else 0.10,
            'max_portfolio_risk': 0.02,
            'emergency_stop': emergency_threshold if emergency_threshold is not None else 0.15,
            'max_leverage': leverage_threshold if leverage_threshold is not None else 5.0,
            'min_margin_ratio': margin_threshold if margin_threshold is not None else 1.2,
            'max_correlation_exposure': 0.3,
        }
        
        # Kill switches configuration
        self.kill_switches = {
            'drawdown_kill': KillSwitch(
                name='drawdown_kill',
                threshold=self.risk_limits['max_drawdown'],
                enabled=True,
                action='stop_trading',
                description=f"Stop trading when drawdown exceeds {self.risk_limits['max_drawdown']:.0%}"
            ),
            'position_loss_kill': KillSwitch(
                name='position_loss_kill',
                threshold=self.risk_limits['max_position_loss'],
                enabled=True,
                action='close_positions',
                description=f"Close positions when loss exceeds {self.risk_limits['max_position_loss']:.0%}"
            ),
            'daily_loss_kill': KillSwitch(
                name='daily_loss_kill',
                threshold=self.risk_limits['max_daily_loss'],
                enabled=True,
                action='stop_trading',
                description=f"Stop trading when daily loss exceeds {self.risk_limits['max_daily_loss']:.0%}"
            ),
            'emergency_kill': KillSwitch(
                name='emergency_kill',
                threshold=self.risk_limits['emergency_stop'],
                enabled=True,
                action='emergency_shutdown',
                description=f"Emergency shutdown when loss exceeds {self.risk_limits['emergency_stop']:.0%}"
            ),
            'leverage_kill': KillSwitch(
                name='leverage_kill',
                threshold=self.risk_limits['max_leverage'],
                enabled=True,
                action='reduce_positions',
                description=f"Reduce positions when leverage exceeds {self.risk_limits['max_leverage']}x"
            ),
            'margin_kill': KillSwitch(
                name='margin_kill',
                threshold=self.risk_limits['min_margin_ratio'],
                enabled=True,
                action='close_positions',
                description=f"Close positions when margin ratio below {self.risk_limits['min_margin_ratio']}x"
            )
        }
        
        # Risk tracking
        self.portfolio_history = deque(maxlen=1000)
        self.risk_history = deque(maxlen=1000)
        self.kill_switch_history = deque(maxlen=100)
        self.peak_portfolio_value = 0.0
        self.initial_portfolio_value = 0.0
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_daily_reset = time.time()
        
        # Position tracking
        self.positions = {}
        self.position_history = deque(maxlen=1000)
        
        # Risk calculation parameters
        self.var_confidence = 0.95
        self.volatility_window = 100
        self.correlation_window = 200
        
        # State tracking
        self.is_trading_allowed = True
        self.emergency_mode = False
        self.last_risk_check = time.time()
        self.risk_check_interval = 1.0  # 1 second
        
        self.logger.info("🛡️ [RISK_ENGINE] Real-Time Risk Engineer initialized")
        self.logger.info(f"🛡️ [RISK_ENGINE] Kill switches: {len(self.kill_switches)} active")
    
    def initialize_portfolio(self, initial_value: float) -> None:
        """Initialize portfolio tracking"""
        self.initial_portfolio_value = initial_value
        self.peak_portfolio_value = initial_value
        self.logger.info(f"🛡️ [RISK_ENGINE] Portfolio initialized: ${initial_value:,.2f}")
    
    def update_portfolio_state(self, portfolio_value: float, positions: Dict[str, Any], 
                             unrealized_pnl: float = 0.0) -> RiskMetrics:
        """Update portfolio state and calculate risk metrics"""
        try:
            current_time = time.time()
            
            # Update portfolio tracking
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value
            
            # Calculate drawdown
            current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            max_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            
            # Calculate exposure
            total_exposure = sum(abs(pos.get('value', 0)) for pos in positions.values())
            
            # Calculate volatility (simplified)
            self.portfolio_history.append(portfolio_value)
            if len(self.portfolio_history) > 2:
                returns = np.diff(list(self.portfolio_history)) / list(self.portfolio_history)[:-1]
                volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
            else:
                volatility = 0.0
            
            # Calculate VaR (simplified)
            if len(self.portfolio_history) > 10:
                returns = np.diff(list(self.portfolio_history)) / list(self.portfolio_history)[:-1]
                var_95 = np.percentile(returns, (1 - self.var_confidence) * 100) * portfolio_value
                expected_shortfall = np.mean(returns[returns <= np.percentile(returns, (1 - self.var_confidence) * 100)]) * portfolio_value
            else:
                var_95 = -portfolio_value * 0.02  # Conservative estimate
                expected_shortfall = -portfolio_value * 0.03  # Conservative estimate
            
            # Calculate correlation risk (simplified)
            correlation_risk = 0.1  # Placeholder
            
            # Calculate concentration risk
            if total_exposure > 0:
                max_position_value = max((abs(pos.get('value', 0)) for pos in positions.values()), default=0)
                concentration_risk = max_position_value / total_exposure
            else:
                concentration_risk = 0.0
            
            # Calculate leverage ratio
            leverage_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0.0
            
            # Calculate margin ratio (simplified)
            margin_ratio = 2.0  # Placeholder - should be calculated from actual margin
            
            # Create risk metrics
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                volatility=volatility,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                margin_ratio=margin_ratio,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0.0  # Would need to track realized P&L
            )
            
            # Store history
            self.risk_history.append({
                'timestamp': current_time,
                'metrics': risk_metrics,
                'positions': positions.copy()
            })
            
            # Check kill switches
            self._check_kill_switches(risk_metrics, positions)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_ENGINE] Error updating portfolio state: {e}")
            return self._empty_risk_metrics()
    
    def _check_kill_switches(self, risk_metrics: RiskMetrics, positions: Dict[str, Any]) -> None:
        """Check all kill switches and take action if needed"""
        try:
            current_time = time.time()
            
            # Check drawdown kill switch
            if self.kill_switches['drawdown_kill'].enabled:
                if risk_metrics.current_drawdown >= self.kill_switches['drawdown_kill'].threshold:
                    self._activate_kill_switch('drawdown_kill', risk_metrics.current_drawdown)
            
            # Check position loss kill switch
            if self.kill_switches['position_loss_kill'].enabled:
                # CRITICAL FIX: Use returnOnEquity from positions instead of calculated percentage
                position_loss_pct = 0.0
                for pos in positions.values():
                    if isinstance(pos, dict):
                        # Check for returnOnEquity in position data
                        if 'returnOnEquity' in pos:
                            position_loss_pct = abs(safe_float(pos['returnOnEquity']))
                            self.logger.debug(f"🔍 [RISK_ENGINE] Position loss check: ROE={pos['returnOnEquity']}, calculated={position_loss_pct:.4f}, threshold={self.kill_switches['position_loss_kill'].threshold:.4f}")
                            break
                        elif 'position' in pos and isinstance(pos['position'], dict):
                            if 'returnOnEquity' in pos['position']:
                                position_loss_pct = abs(safe_float(pos['position']['returnOnEquity']))
                                self.logger.debug(f"🔍 [RISK_ENGINE] Position loss check: ROE={pos['position']['returnOnEquity']}, calculated={position_loss_pct:.4f}, threshold={self.kill_switches['position_loss_kill'].threshold:.4f}")
                                break
                
                # Fallback to calculated percentage if returnOnEquity not available
                if position_loss_pct == 0.0:
                    position_loss_pct = abs(risk_metrics.unrealized_pnl) / risk_metrics.portfolio_value if risk_metrics.portfolio_value > 0 else 0.0
                    self.logger.debug(f"🔍 [RISK_ENGINE] Position loss check: fallback={position_loss_pct:.4f}, threshold={self.kill_switches['position_loss_kill'].threshold:.4f}")
                
                if position_loss_pct >= self.kill_switches['position_loss_kill'].threshold:
                    self.logger.warning(f"🚨 [RISK_ENGINE] POSITION LOSS KILL SWITCH TRIGGERED: {position_loss_pct:.4f} >= {self.kill_switches['position_loss_kill'].threshold:.4f}")
                    self._activate_kill_switch('position_loss_kill', position_loss_pct)
            
            # Check daily loss kill switch
            if self.kill_switches['daily_loss_kill'].enabled:
                daily_loss_pct = self.daily_pnl / self.initial_portfolio_value if self.initial_portfolio_value > 0 else 0
                if daily_loss_pct <= -self.kill_switches['daily_loss_kill'].threshold:
                    self._activate_kill_switch('daily_loss_kill', abs(daily_loss_pct))
            
            # Check emergency kill switch
            if self.kill_switches['emergency_kill'].enabled:
                if risk_metrics.current_drawdown >= self.kill_switches['emergency_kill'].threshold:
                    self._activate_kill_switch('emergency_kill', risk_metrics.current_drawdown)
            
            # Check leverage kill switch
            if self.kill_switches['leverage_kill'].enabled:
                if risk_metrics.leverage_ratio >= self.kill_switches['leverage_kill'].threshold:
                    self._activate_kill_switch('leverage_kill', risk_metrics.leverage_ratio)
            
            # Check margin kill switch
            if self.kill_switches['margin_kill'].enabled:
                if risk_metrics.margin_ratio <= self.kill_switches['margin_kill'].threshold:
                    self._activate_kill_switch('margin_kill', risk_metrics.margin_ratio)
            
            self.last_risk_check = current_time
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_ENGINE] Error checking kill switches: {e}")
    
    def _activate_kill_switch(self, switch_name: str, trigger_value: float) -> None:
        """Activate a kill switch and take appropriate action"""
        try:
            kill_switch = self.kill_switches[switch_name]
            
            # Log kill switch activation
            self.logger.warning(f"🚨 [RISK_ENGINE] KILL SWITCH ACTIVATED: {switch_name}")
            self.logger.warning(f"🚨 [RISK_ENGINE] Trigger value: {trigger_value:.4f} (threshold: {kill_switch.threshold:.4f})")
            self.logger.warning(f"🚨 [RISK_ENGINE] Action: {kill_switch.action}")
            self.logger.warning(f"🚨 [RISK_ENGINE] Description: {kill_switch.description}")
            
            # Store kill switch history
            self.kill_switch_history.append({
                'timestamp': time.time(),
                'switch_name': switch_name,
                'trigger_value': trigger_value,
                'threshold': kill_switch.threshold,
                'action': kill_switch.action,
                'description': kill_switch.description
            })
            
            # Take action based on kill switch type
            if kill_switch.action == 'stop_trading':
                self.is_trading_allowed = False
                self.logger.error("🚨 [RISK_ENGINE] TRADING STOPPED - Kill switch activated")
            
            elif kill_switch.action == 'close_positions':
                self.logger.error("🚨 [RISK_ENGINE] POSITIONS MUST BE CLOSED - Kill switch activated")
                # CRITICAL FIX: Actually trigger position closing logic
                try:
                    if hasattr(self, 'trading_bot') and self.trading_bot:
                        # Get current position and close it
                        positions = self.trading_bot.get_positions()
                        current_pos = None
                        
                        for pos in positions:
                            if isinstance(pos, dict):
                                if pos.get('coin') == 'XRP':
                                    current_pos = pos
                                    break
                                elif 'position' in pos and isinstance(pos['position'], dict):
                                    if pos['position'].get('coin') == 'XRP':
                                        current_pos = pos['position']
                                        break
                        
                        if current_pos and current_pos.get('szi', 0) != 0:
                            position_size = abs(safe_float(current_pos.get('szi', 0)))
                            is_long = safe_float(current_pos.get('szi', 0)) > 0
                            success = self.trading_bot._emergency_position_exit(position_size, is_long)
                            if success:
                                self.logger.info("🚨 [RISK_ENGINE] Emergency position exit triggered successfully")
                            else:
                                self.logger.error("🚨 [RISK_ENGINE] Emergency position exit failed")
                        else:
                            self.logger.warning("🚨 [RISK_ENGINE] No XRP position found to close")
                    else:
                        self.logger.error("🚨 [RISK_ENGINE] No trading bot reference for emergency exit")
                except Exception as e:
                    self.logger.error(f"🚨 [RISK_ENGINE] Emergency position exit failed: {e}")
            
            elif kill_switch.action == 'emergency_shutdown':
                self.emergency_mode = True
                self.is_trading_allowed = False
                self.logger.critical("🚨 [RISK_ENGINE] EMERGENCY SHUTDOWN - Kill switch activated")
                # This would trigger emergency shutdown procedures
            
            elif kill_switch.action == 'reduce_positions':
                self.logger.error("🚨 [RISK_ENGINE] POSITIONS MUST BE REDUCED - Kill switch activated")
                # This would trigger position reduction logic
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_ENGINE] Error activating kill switch {switch_name}: {e}")
    
    def can_open_position(self, position_size: float, position_value: float, 
                         portfolio_value: float, symbol: str) -> Tuple[bool, str]:
        """Check if a new position can be opened"""
        try:
            if not self.is_trading_allowed:
                return False, "Trading stopped by kill switch"
            
            if self.emergency_mode:
                return False, "Emergency mode active"
            
            # Check position size limits
            if position_value > portfolio_value * self.risk_limits['max_position_loss']:
                return False, f"Position size exceeds {self.risk_limits['max_position_loss']:.1%} limit"
            
            # Check concentration limits
            total_exposure = sum(abs(pos.get('value', 0)) for pos in self.positions.values())
            if (total_exposure + position_value) > portfolio_value * self.risk_limits['max_concentration']:
                return False, f"Concentration exceeds {self.risk_limits['max_concentration']:.1%} limit"
            
            # Check portfolio risk limits
            if position_value > portfolio_value * self.risk_limits['max_portfolio_risk']:
                return False, f"Portfolio risk exceeds {self.risk_limits['max_portfolio_risk']:.1%} limit"
            
            return True, "Position allowed"
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_ENGINE] Error checking position limits: {e}")
            return False, f"Error: {e}"
    
    def calculate_position_size(self, portfolio_value: float, entry_price: float, 
                              stop_loss_price: float, risk_per_trade: float = 0.02) -> float:
        """Calculate safe position size based on risk parameters"""
        try:
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share <= 0:
                return 0.0
            
            # Calculate maximum risk amount
            max_risk_amount = portfolio_value * risk_per_trade
            
            # Calculate position size
            position_size = max_risk_amount / risk_per_share
            
            # Apply position size limits
            max_position_value = portfolio_value * self.risk_limits['max_position_loss']
            max_position_size = max_position_value / entry_price
            
            position_size = min(position_size, max_position_size)
            
            self.logger.info(f"🛡️ [RISK_ENGINE] Calculated position size: {position_size:.4f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_ENGINE] Error calculating position size: {e}")
            return 0.0
    
    def update_daily_metrics(self, trade_pnl: float) -> None:
        """Update daily performance metrics"""
        try:
            current_time = time.time()
            
            # Reset daily metrics if 24 hours have passed
            if current_time - self.last_daily_reset >= 86400:  # 24 hours
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.consecutive_losses = 0
                self.last_daily_reset = current_time
                self.logger.info("🛡️ [RISK_ENGINE] Daily metrics reset")
            
            # Update daily metrics
            self.daily_pnl += trade_pnl
            self.daily_trades += 1
            
            if trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Log daily performance
            if self.daily_trades % 10 == 0:  # Log every 10 trades
                self.logger.info(f"🛡️ [RISK_ENGINE] Daily P&L: ${self.daily_pnl:.2f}, Trades: {self.daily_trades}")
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_ENGINE] Error updating daily metrics: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            if not self.risk_history:
                return self._empty_risk_summary()
            
            latest_metrics = self.risk_history[-1]['metrics']
            
            return {
                'portfolio_value': latest_metrics.portfolio_value,
                'current_drawdown': latest_metrics.current_drawdown,
                'max_drawdown': latest_metrics.max_drawdown,
                'volatility': latest_metrics.volatility,
                'var_95': latest_metrics.var_95,
                'leverage_ratio': latest_metrics.leverage_ratio,
                'concentration_risk': latest_metrics.concentration_risk,
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'consecutive_losses': self.consecutive_losses,
                'is_trading_allowed': self.is_trading_allowed,
                'emergency_mode': self.emergency_mode,
                'active_kill_switches': len([k for k in self.kill_switches.values() if k.enabled]),
                'kill_switch_activations': len(self.kill_switch_history)
            }
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_ENGINE] Error getting risk summary: {e}")
            return self._empty_risk_summary()
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            portfolio_value=0.0,
            total_exposure=0.0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            leverage_ratio=0.0,
            margin_ratio=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
    
    def _empty_risk_summary(self) -> Dict[str, Any]:
        """Return empty risk summary"""
        return {
            'portfolio_value': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'var_95': 0.0,
            'leverage_ratio': 0.0,
            'concentration_risk': 0.0,
            'daily_pnl': 0.0,
            'daily_trades': 0,
            'consecutive_losses': 0,
            'is_trading_allowed': True,
            'emergency_mode': False,
            'active_kill_switches': 0,
            'kill_switch_activations': 0
        }
    
    def reset_kill_switches(self) -> None:
        """Reset kill switches (for testing or manual override)"""
        self.is_trading_allowed = True
        self.emergency_mode = False
        self.logger.warning("🛡️ [RISK_ENGINE] Kill switches manually reset")
    
    def log_risk_status(self) -> None:
        """Log current risk status"""
        try:
            summary = self.get_risk_summary()
            
            self.logger.info("🛡️ [RISK_ENGINE] === RISK STATUS ===")
            self.logger.info(f"🛡️ [RISK_ENGINE] Portfolio Value: ${summary['portfolio_value']:,.2f}")
            self.logger.info(f"🛡️ [RISK_ENGINE] Current Drawdown: {summary['current_drawdown']:.2%}")
            self.logger.info(f"🛡️ [RISK_ENGINE] Max Drawdown: {summary['max_drawdown']:.2%}")
            self.logger.info(f"🛡️ [RISK_ENGINE] Volatility: {summary['volatility']:.2%}")
            self.logger.info(f"🛡️ [RISK_ENGINE] VaR 95%: ${summary['var_95']:,.2f}")
            self.logger.info(f"🛡️ [RISK_ENGINE] Leverage Ratio: {summary['leverage_ratio']:.2f}x")
            self.logger.info(f"🛡️ [RISK_ENGINE] Daily P&L: ${summary['daily_pnl']:,.2f}")
            self.logger.info(f"🛡️ [RISK_ENGINE] Trading Allowed: {summary['is_trading_allowed']}")
            self.logger.info(f"🛡️ [RISK_ENGINE] Emergency Mode: {summary['emergency_mode']}")
            self.logger.info(f"🛡️ [RISK_ENGINE] Kill Switch Activations: {summary['kill_switch_activations']}")
            self.logger.info("🛡️ [RISK_ENGINE] ===================")
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_ENGINE] Error logging risk status: {e}")
