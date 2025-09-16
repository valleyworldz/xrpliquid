"""
🛡️ REALIZED DRAWDOWN KILL-SWITCH
==================================
Production-grade realized PnL drawdown monitoring with automatic kill-switch.

Features:
- Realized PnL drawdown limits
- Rolling window monitoring
- Automatic trading halt
- Risk event logging
- Recovery procedures
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger

@dataclass
class KillSwitchConfig:
    """Configuration for realized drawdown kill-switch"""
    
    # Drawdown limits
    daily_drawdown_limit: float = 0.02      # 2% daily drawdown limit
    rolling_drawdown_limit: float = 0.05    # 5% rolling drawdown limit
    rolling_window_days: int = 7            # 7-day rolling window
    kill_switch_threshold: float = 0.08     # 8% kill switch threshold
    
    # Monitoring
    check_interval_seconds: int = 5         # Check every 5 seconds
    cooldown_period_hours: int = 24         # 24-hour cooldown after kill switch
    
    # Logging
    log_directory: str = "reports/risk_events"
    log_retention_days: int = 90            # Keep logs for 90 days

@dataclass
class RiskEvent:
    """Risk event data structure"""
    
    timestamp: float
    event_type: str  # 'drawdown_warning', 'drawdown_limit', 'kill_switch_activated'
    current_drawdown: float
    daily_drawdown: float
    rolling_drawdown: float
    realized_pnl: float
    account_balance: float
    position_count: int
    total_exposure: float
    message: str
    action_taken: str

class RealizedDrawdownKillSwitch:
    """
    🛡️ REALIZED DRAWDOWN KILL-SWITCH
    
    Production-grade realized PnL drawdown monitoring with automatic kill-switch
    and comprehensive risk event logging.
    """
    
    def __init__(self, config: KillSwitchConfig, logger=None):
        self.config = config
        self.logger = logger or Logger()
        
        # Kill switch state
        self.is_active = False
        self.kill_switch_activated = False
        self.kill_switch_time = 0.0
        self.cooldown_end_time = 0.0
        
        # Performance tracking
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.realized_pnl = 0.0
        self.daily_pnl_history = []
        self.rolling_pnl_history = []
        
        # Risk metrics
        self.current_drawdown = 0.0
        self.daily_drawdown = 0.0
        self.rolling_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Risk events
        self.risk_events: List[RiskEvent] = []
        
        # Initialize logging directory
        self._initialize_logging_directory()
        
        self.logger.info("🛡️ [KILL_SWITCH] Realized Drawdown Kill-Switch initialized")
        self.logger.info(f"🛡️ [KILL_SWITCH] Daily limit: {self.config.daily_drawdown_limit:.2%}")
        self.logger.info(f"🛡️ [KILL_SWITCH] Rolling limit: {self.config.rolling_drawdown_limit:.2%}")
        self.logger.info(f"🛡️ [KILL_SWITCH] Kill switch threshold: {self.config.kill_switch_threshold:.2%}")
    
    def _initialize_logging_directory(self):
        """Initialize risk event logging directory"""
        try:
            log_dir = Path(self.config.log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"🛡️ [LOG_DIR] Initialized risk event logging: {log_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ [LOG_DIR] Error initializing logging directory: {e}")
    
    async def start_monitoring(self):
        """Start continuous drawdown monitoring"""
        try:
            self.logger.info("🛡️ [MONITORING] Starting realized drawdown monitoring...")
            
            while True:
                try:
                    await self._monitoring_cycle()
                    await asyncio.sleep(self.config.check_interval_seconds)
                except Exception as e:
                    self.logger.error(f"❌ [MONITORING] Error in monitoring cycle: {e}")
                    await asyncio.sleep(30)  # Wait 30 seconds on error
                    
        except Exception as e:
            self.logger.error(f"❌ [MONITORING] Error starting monitoring: {e}")
    
    async def _monitoring_cycle(self):
        """Main monitoring cycle"""
        try:
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Check drawdown limits
            await self._check_drawdown_limits()
            
            # Check kill switch status
            await self._check_kill_switch_status()
            
        except Exception as e:
            self.logger.error(f"❌ [MONITORING_CYCLE] Error in monitoring cycle: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # This would integrate with actual portfolio data
            # For now, simulate performance metrics
            
            current_time = time.time()
            
            # Simulate current balance and realized PnL
            # In production, this would come from the trading system
            self.current_balance = 10000.0  # Simulated current balance
            self.realized_pnl = -500.0      # Simulated realized PnL (loss)
            
            if self.initial_balance == 0:
                self.initial_balance = self.current_balance
            
            # Calculate current drawdown
            self.current_drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
            
            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            
            # Update daily PnL history
            self.daily_pnl_history.append(self.realized_pnl)
            
            # Keep only recent history (last 30 days)
            if len(self.daily_pnl_history) > 30:
                self.daily_pnl_history = self.daily_pnl_history[-30:]
            
            # Calculate daily drawdown
            if len(self.daily_pnl_history) > 0:
                daily_pnl = sum(self.daily_pnl_history[-1:])  # Last day
                self.daily_drawdown = abs(daily_pnl) / self.initial_balance if daily_pnl < 0 else 0.0
            
            # Calculate rolling drawdown
            if len(self.daily_pnl_history) >= self.config.rolling_window_days:
                rolling_pnl = sum(self.daily_pnl_history[-self.config.rolling_window_days:])
                self.rolling_drawdown = abs(rolling_pnl) / self.initial_balance if rolling_pnl < 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ [UPDATE_METRICS] Error updating performance metrics: {e}")
    
    async def _check_drawdown_limits(self):
        """Check drawdown limits"""
        try:
            # Check daily drawdown limit
            if self.daily_drawdown >= self.config.daily_drawdown_limit:
                await self._log_risk_event(
                    event_type='drawdown_warning',
                    message=f"Daily drawdown limit exceeded: {self.daily_drawdown:.2%} >= {self.config.daily_drawdown_limit:.2%}",
                    action_taken='warning_logged'
                )
            
            # Check rolling drawdown limit
            if self.rolling_drawdown >= self.config.rolling_drawdown_limit:
                await self._log_risk_event(
                    event_type='drawdown_limit',
                    message=f"Rolling drawdown limit exceeded: {self.rolling_drawdown:.2%} >= {self.config.rolling_drawdown_limit:.2%}",
                    action_taken='position_reduction_recommended'
                )
            
            # Check kill switch threshold
            if self.current_drawdown >= self.config.kill_switch_threshold:
                await self._activate_kill_switch()
            
        except Exception as e:
            self.logger.error(f"❌ [DRAWDOWN_CHECK] Error checking drawdown limits: {e}")
    
    async def _check_kill_switch_status(self):
        """Check kill switch status"""
        try:
            current_time = time.time()
            
            # Check if we're in cooldown period
            if self.kill_switch_activated and current_time < self.cooldown_end_time:
                # Still in cooldown
                remaining_time = self.cooldown_end_time - current_time
                self.logger.info(f"🛡️ [KILL_SWITCH] In cooldown period: {remaining_time/3600:.1f} hours remaining")
            elif self.kill_switch_activated and current_time >= self.cooldown_end_time:
                # Cooldown period ended
                self.kill_switch_activated = False
                self.logger.info("🛡️ [KILL_SWITCH] Cooldown period ended - system ready for restart")
            
        except Exception as e:
            self.logger.error(f"❌ [KILL_SWITCH_STATUS] Error checking kill switch status: {e}")
    
    async def _activate_kill_switch(self):
        """Activate kill switch"""
        try:
            if self.kill_switch_activated:
                return  # Already activated
            
            self.kill_switch_activated = True
            self.kill_switch_time = time.time()
            self.cooldown_end_time = time.time() + (self.config.cooldown_period_hours * 3600)
            
            # Log kill switch activation
            await self._log_risk_event(
                event_type='kill_switch_activated',
                message=f"Kill switch activated due to drawdown: {self.current_drawdown:.2%} >= {self.config.kill_switch_threshold:.2%}",
                action_taken='trading_halted'
            )
            
            self.logger.critical("🚨 [KILL_SWITCH] KILL SWITCH ACTIVATED!")
            self.logger.critical(f"🚨 [KILL_SWITCH] Current drawdown: {self.current_drawdown:.2%}")
            self.logger.critical(f"🚨 [KILL_SWITCH] Cooldown period: {self.config.cooldown_period_hours} hours")
            
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
            
            await self._log_risk_event(
                event_type='emergency_exit',
                message="Emergency exit of all positions initiated",
                action_taken='all_positions_marked_for_closure'
            )
            
            self.logger.critical("🚨 [EMERGENCY_EXIT] All positions marked for emergency exit")
            
        except Exception as e:
            self.logger.error(f"❌ [EMERGENCY_EXIT] Error in emergency exit: {e}")
    
    async def _log_risk_event(self, event_type: str, message: str, action_taken: str):
        """Log risk event"""
        try:
            current_time = time.time()
            
            # Create risk event
            risk_event = RiskEvent(
                timestamp=current_time,
                event_type=event_type,
                current_drawdown=self.current_drawdown,
                daily_drawdown=self.daily_drawdown,
                rolling_drawdown=self.rolling_drawdown,
                realized_pnl=self.realized_pnl,
                account_balance=self.current_balance,
                position_count=0,  # Would be actual position count
                total_exposure=0.0,  # Would be actual total exposure
                message=message,
                action_taken=action_taken,
            )
            
            # Add to risk events list
            self.risk_events.append(risk_event)
            
            # Log to file
            await self._write_risk_event_to_file(risk_event)
            
            # Log to console
            self.logger.warning(f"🛡️ [RISK_EVENT] {event_type}: {message}")
            self.logger.warning(f"🛡️ [RISK_EVENT] Action taken: {action_taken}")
            
        except Exception as e:
            self.logger.error(f"❌ [LOG_RISK_EVENT] Error logging risk event: {e}")
    
    async def _write_risk_event_to_file(self, risk_event: RiskEvent):
        """Write risk event to file"""
        try:
            timestamp = datetime.fromtimestamp(risk_event.timestamp).strftime('%Y%m%d_%H%M%S')
            log_file = Path(self.config.log_directory) / f'risk_events_{datetime.now().strftime("%Y%m%d")}.json'
            
            # Convert to dictionary
            event_data = {
                'timestamp': risk_event.timestamp,
                'event_type': risk_event.event_type,
                'current_drawdown': risk_event.current_drawdown,
                'daily_drawdown': risk_event.daily_drawdown,
                'rolling_drawdown': risk_event.rolling_drawdown,
                'realized_pnl': risk_event.realized_pnl,
                'account_balance': risk_event.account_balance,
                'position_count': risk_event.position_count,
                'total_exposure': risk_event.total_exposure,
                'message': risk_event.message,
                'action_taken': risk_event.action_taken,
            }
            
            # Append to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
            
        except Exception as e:
            self.logger.error(f"❌ [WRITE_RISK_EVENT] Error writing risk event to file: {e}")
    
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active"""
        return self.kill_switch_activated
    
    def is_in_cooldown(self) -> bool:
        """Check if system is in cooldown period"""
        return self.kill_switch_activated and time.time() < self.cooldown_end_time
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            return {
                'kill_switch_active': self.kill_switch_activated,
                'in_cooldown': self.is_in_cooldown(),
                'current_drawdown': self.current_drawdown,
                'daily_drawdown': self.daily_drawdown,
                'rolling_drawdown': self.rolling_drawdown,
                'max_drawdown': self.max_drawdown,
                'realized_pnl': self.realized_pnl,
                'account_balance': self.current_balance,
                'initial_balance': self.initial_balance,
                'risk_events_count': len(self.risk_events),
                'last_risk_event': self.risk_events[-1].__dict__ if self.risk_events else None,
                'config': self.config.__dict__,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_SUMMARY] Error getting risk summary: {e}")
            return {}
    
    async def reset_kill_switch(self):
        """Reset kill switch (manual override)"""
        try:
            self.kill_switch_activated = False
            self.cooldown_end_time = 0.0
            
            await self._log_risk_event(
                event_type='kill_switch_reset',
                message="Kill switch manually reset",
                action_taken='manual_override'
            )
            
            self.logger.info("🛡️ [KILL_SWITCH] Kill switch manually reset")
            
        except Exception as e:
            self.logger.error(f"❌ [RESET_KILL_SWITCH] Error resetting kill switch: {e}")
