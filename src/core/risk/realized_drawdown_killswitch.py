"""
🛡️ REALIZED DRAWDOWN KILL-SWITCH
=================================
Daily DD kill tied to realized PnL with trip logs
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    daily_drawdown_limit: float = 0.02  # 2% daily limit
    rolling_drawdown_limit: float = 0.05  # 5% rolling limit
    kill_switch_threshold: float = 0.08  # 8% kill switch
    cooldown_hours: int = 24  # 24 hour cooldown

class RealizedDrawdownKillSwitch:
    """Realized PnL drawdown kill-switch"""
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.daily_pnl = 0.0
        self.rolling_pnl = 0.0
        self.peak_equity = 0.0
        self.kill_switch_triggered = False
        self.last_reset = datetime.now()
        self.trip_logs = []
        
        logger.info("🛡️ [KILL_SWITCH] Realized drawdown kill-switch initialized")
        logger.info(f"🛡️ [KILL_SWITCH] Daily limit: {limits.daily_drawdown_limit:.2%}")
        logger.info(f"🛡️ [KILL_SWITCH] Rolling limit: {limits.rolling_drawdown_limit:.2%}")
        logger.info(f"🛡️ [KILL_SWITCH] Kill switch threshold: {limits.kill_switch_threshold:.2%}")
    
    def update_pnl(self, realized_pnl: float, current_equity: float):
        """Update PnL and check limits"""
        self.daily_pnl += realized_pnl
        self.rolling_pnl += realized_pnl
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Check daily limit
        daily_drawdown = abs(self.daily_pnl) / self.peak_equity if self.peak_equity > 0 else 0
        if daily_drawdown > self.limits.daily_drawdown_limit:
            self._log_trip("daily_drawdown_limit", daily_drawdown, "warning_logged")
        
        # Check rolling limit
        rolling_drawdown = abs(self.rolling_pnl) / self.peak_equity if self.peak_equity > 0 else 0
        if rolling_drawdown > self.limits.rolling_drawdown_limit:
            self._log_trip("rolling_drawdown_limit", rolling_drawdown, "position_reduction_recommended")
        
        # Check kill switch
        if rolling_drawdown > self.limits.kill_switch_threshold:
            self._trigger_kill_switch(rolling_drawdown)
    
    def _trigger_kill_switch(self, drawdown: float):
        """Trigger kill switch"""
        if not self.kill_switch_triggered:
            self.kill_switch_triggered = True
            self._log_trip("kill_switch_triggered", drawdown, "trading_halted")
            logger.critical(f"🛡️ [KILL_SWITCH] KILL SWITCH TRIGGERED: {drawdown:.2%} drawdown")
    
    def _log_trip(self, event_type: str, drawdown: float, action: str):
        """Log risk event"""
        trip_log = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'drawdown': drawdown,
            'action': action,
            'daily_pnl': self.daily_pnl,
            'rolling_pnl': self.rolling_pnl,
            'peak_equity': self.peak_equity
        }
        
        self.trip_logs.append(trip_log)
        logger.warning(f"🛡️ [RISK_EVENT] {event_type}: {drawdown:.2%} drawdown")
        logger.warning(f"🛡️ [RISK_EVENT] Action taken: {action}")
    
    def reset_daily(self):
        """Reset daily PnL"""
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()
        logger.info("🛡️ [KILL_SWITCH] Daily PnL reset")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if self.kill_switch_triggered:
            # Check cooldown
            if datetime.now() - self.last_reset > timedelta(hours=self.limits.cooldown_hours):
                self.kill_switch_triggered = False
                logger.info("🛡️ [KILL_SWITCH] Cooldown period ended, trading resumed")
                return True
            return False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get kill-switch status"""
        return {
            'kill_switch_triggered': self.kill_switch_triggered,
            'daily_pnl': self.daily_pnl,
            'rolling_pnl': self.rolling_pnl,
            'peak_equity': self.peak_equity,
            'daily_drawdown': abs(self.daily_pnl) / self.peak_equity if self.peak_equity > 0 else 0,
            'rolling_drawdown': abs(self.rolling_pnl) / self.peak_equity if self.peak_equity > 0 else 0,
            'can_trade': self.can_trade(),
            'trip_count': len(self.trip_logs)
        }
    
    def save_trip_logs(self, filepath: str):
        """Save trip logs to JSON"""
        logs = {
            'timestamp': datetime.now().isoformat(),
            'status': self.get_status(),
            'trip_logs': self.trip_logs
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"🛡️ [KILL_SWITCH] Saved trip logs: {filepath}")

# Global kill-switch instance
risk_limits = RiskLimits()
kill_switch = RealizedDrawdownKillSwitch(risk_limits)