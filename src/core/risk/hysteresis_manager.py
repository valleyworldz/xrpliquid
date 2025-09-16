"""
Hysteresis Manager - Prevents Kill-Switch Flapping
Implements staged re-enable rules and cooldown windows for risk controls.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class RiskState(Enum):
    """Risk state enumeration."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    KILLED = "killed"
    COOLDOWN = "cooldown"


@dataclass
class HysteresisRule:
    """Hysteresis rule configuration."""
    trigger_threshold: float
    reset_threshold: float
    cooldown_seconds: int
    max_trigger_count: int
    escalation_delay_seconds: int


class HysteresisManager:
    """Manages hysteresis for risk controls to prevent flapping."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.risk_dir = self.reports_dir / "risk"
        self.risk_dir.mkdir(parents=True, exist_ok=True)
        
        # Hysteresis rules
        self.rules = self._define_hysteresis_rules()
        
        # State tracking
        self.current_states: Dict[str, RiskState] = {}
        self.trigger_counts: Dict[str, int] = {}
        self.last_trigger_times: Dict[str, datetime] = {}
        self.cooldown_until: Dict[str, datetime] = {}
        
        # Load existing state
        self._load_state()
    
    def _define_hysteresis_rules(self) -> Dict[str, HysteresisRule]:
        """Define hysteresis rules for different risk metrics."""
        
        rules = {
            "daily_drawdown": HysteresisRule(
                trigger_threshold=0.05,  # 5% drawdown triggers kill
                reset_threshold=0.02,    # Must drop to 2% to reset
                cooldown_seconds=3600,   # 1 hour cooldown
                max_trigger_count=3,     # Max 3 triggers per day
                escalation_delay_seconds=300  # 5 min delay before escalation
            ),
            
            "position_size": HysteresisRule(
                trigger_threshold=0.15,  # 15% position size triggers warning
                reset_threshold=0.10,    # Must drop to 10% to reset
                cooldown_seconds=1800,   # 30 min cooldown
                max_trigger_count=5,     # Max 5 triggers per day
                escalation_delay_seconds=60  # 1 min delay
            ),
            
            "volatility": HysteresisRule(
                trigger_threshold=0.30,  # 30% volatility triggers warning
                reset_threshold=0.20,    # Must drop to 20% to reset
                cooldown_seconds=7200,   # 2 hour cooldown
                max_trigger_count=2,     # Max 2 triggers per day
                escalation_delay_seconds=600  # 10 min delay
            ),
            
            "api_errors": HysteresisRule(
                trigger_threshold=0.10,  # 10% error rate triggers warning
                reset_threshold=0.05,    # Must drop to 5% to reset
                cooldown_seconds=900,    # 15 min cooldown
                max_trigger_count=10,    # Max 10 triggers per day
                escalation_delay_seconds=30  # 30 sec delay
            )
        }
        
        return rules
    
    def check_risk_metric(self, 
                         metric_name: str, 
                         current_value: float,
                         timestamp: datetime = None) -> Dict[str, Any]:
        """Check risk metric against hysteresis rules."""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if metric_name not in self.rules:
            return {
                "metric": metric_name,
                "status": "unknown",
                "message": f"No hysteresis rule defined for {metric_name}",
                "timestamp": timestamp.isoformat()
            }
        
        rule = self.rules[metric_name]
        current_state = self.current_states.get(metric_name, RiskState.NORMAL)
        
        # Check if in cooldown
        if metric_name in self.cooldown_until:
            if timestamp < self.cooldown_until[metric_name]:
                remaining_cooldown = (self.cooldown_until[metric_name] - timestamp).total_seconds()
                return {
                    "metric": metric_name,
                    "status": "cooldown",
                    "message": f"In cooldown for {remaining_cooldown:.0f} seconds",
                    "current_value": current_value,
                    "current_state": current_state.value,
                    "timestamp": timestamp.isoformat()
                }
        
        # Check trigger conditions
        should_trigger = current_value >= rule.trigger_threshold
        should_reset = current_value <= rule.reset_threshold
        
        # Determine new state
        new_state = current_state
        action_taken = None
        
        if should_trigger and current_state != RiskState.KILLED:
            # Check if we can trigger (not in cooldown and under max count)
            trigger_count = self.trigger_counts.get(metric_name, 0)
            
            if trigger_count < rule.max_trigger_count:
                # Check escalation delay
                last_trigger = self.last_trigger_times.get(metric_name)
                if last_trigger is None or (timestamp - last_trigger).total_seconds() >= rule.escalation_delay_seconds:
                    # Trigger escalation
                    if current_state == RiskState.NORMAL:
                        new_state = RiskState.WARNING
                        action_taken = "escalated_to_warning"
                    elif current_state == RiskState.WARNING:
                        new_state = RiskState.CRITICAL
                        action_taken = "escalated_to_critical"
                    elif current_state == RiskState.CRITICAL:
                        new_state = RiskState.KILLED
                        action_taken = "killed_trading"
                    
                    # Update tracking
                    self.trigger_counts[metric_name] = trigger_count + 1
                    self.last_trigger_times[metric_name] = timestamp
                    
                    # Set cooldown
                    self.cooldown_until[metric_name] = timestamp + timedelta(seconds=rule.cooldown_seconds)
        
        elif should_reset and current_state != RiskState.NORMAL:
            # Reset to normal state
            new_state = RiskState.NORMAL
            action_taken = "reset_to_normal"
            
            # Clear cooldown
            if metric_name in self.cooldown_until:
                del self.cooldown_until[metric_name]
        
        # Update state
        self.current_states[metric_name] = new_state
        
        # Save state
        self._save_state()
        
        # Log action if taken
        if action_taken:
            self._log_action(metric_name, action_taken, current_value, new_state, timestamp)
        
        return {
            "metric": metric_name,
            "status": new_state.value,
            "action_taken": action_taken,
            "current_value": current_value,
            "trigger_threshold": rule.trigger_threshold,
            "reset_threshold": rule.reset_threshold,
            "trigger_count": self.trigger_counts.get(metric_name, 0),
            "max_trigger_count": rule.max_trigger_count,
            "cooldown_until": self.cooldown_until.get(metric_name).isoformat() if metric_name in self.cooldown_until else None,
            "timestamp": timestamp.isoformat()
        }
    
    def force_reset(self, metric_name: str, reason: str = "manual_reset") -> Dict[str, Any]:
        """Force reset a metric to normal state."""
        
        timestamp = datetime.now(timezone.utc)
        
        if metric_name in self.current_states:
            old_state = self.current_states[metric_name]
            self.current_states[metric_name] = RiskState.NORMAL
            
            # Clear cooldown
            if metric_name in self.cooldown_until:
                del self.cooldown_until[metric_name]
            
            # Reset trigger count
            self.trigger_counts[metric_name] = 0
            
            # Save state
            self._save_state()
            
            # Log action
            self._log_action(metric_name, f"force_reset_{reason}", 0.0, RiskState.NORMAL, timestamp)
            
            return {
                "metric": metric_name,
                "status": "reset",
                "old_state": old_state.value,
                "new_state": RiskState.NORMAL.value,
                "reason": reason,
                "timestamp": timestamp.isoformat()
            }
        
        return {
            "metric": metric_name,
            "status": "not_found",
            "message": f"No state found for metric {metric_name}",
            "timestamp": timestamp.isoformat()
        }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of all risk states."""
        
        timestamp = datetime.now(timezone.utc)
        
        summary = {
            "timestamp": timestamp.isoformat(),
            "total_metrics": len(self.rules),
            "active_metrics": len(self.current_states),
            "killed_metrics": 0,
            "warning_metrics": 0,
            "critical_metrics": 0,
            "cooldown_metrics": 0,
            "metric_states": {}
        }
        
        for metric_name, state in self.current_states.items():
            summary["metric_states"][metric_name] = {
                "state": state.value,
                "trigger_count": self.trigger_counts.get(metric_name, 0),
                "max_trigger_count": self.rules[metric_name].max_trigger_count,
                "cooldown_until": self.cooldown_until.get(metric_name).isoformat() if metric_name in self.cooldown_until else None,
                "in_cooldown": metric_name in self.cooldown_until and timestamp < self.cooldown_until[metric_name]
            }
            
            # Count states
            if state == RiskState.KILLED:
                summary["killed_metrics"] += 1
            elif state == RiskState.WARNING:
                summary["warning_metrics"] += 1
            elif state == RiskState.CRITICAL:
                summary["critical_metrics"] += 1
            
            if metric_name in self.cooldown_until and timestamp < self.cooldown_until[metric_name]:
                summary["cooldown_metrics"] += 1
        
        return summary
    
    def _log_action(self, 
                   metric_name: str, 
                   action: str, 
                   value: float, 
                   new_state: RiskState, 
                   timestamp: datetime):
        """Log risk action for audit trail."""
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "metric": metric_name,
            "action": action,
            "value": value,
            "new_state": new_state.value,
            "trigger_count": self.trigger_counts.get(metric_name, 0)
        }
        
        # Save to risk log
        risk_log_file = self.risk_dir / "hysteresis_actions.jsonl"
        with open(risk_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _save_state(self):
        """Save current state to disk."""
        
        state_data = {
            "current_states": {k: v.value for k, v in self.current_states.items()},
            "trigger_counts": self.trigger_counts,
            "last_trigger_times": {k: v.isoformat() for k, v in self.last_trigger_times.items()},
            "cooldown_until": {k: v.isoformat() for k, v in self.cooldown_until.items()},
            "last_saved": datetime.now(timezone.utc).isoformat()
        }
        
        state_file = self.risk_dir / "hysteresis_state.json"
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def _load_state(self):
        """Load state from disk."""
        
        state_file = self.risk_dir / "hysteresis_state.json"
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Load states
            self.current_states = {
                k: RiskState(v) for k, v in state_data.get("current_states", {}).items()
            }
            
            # Load trigger counts
            self.trigger_counts = state_data.get("trigger_counts", {})
            
            # Load trigger times
            self.last_trigger_times = {
                k: datetime.fromisoformat(v) for k, v in state_data.get("last_trigger_times", {}).items()
            }
            
            # Load cooldown times
            self.cooldown_until = {
                k: datetime.fromisoformat(v) for k, v in state_data.get("cooldown_until", {}).items()
            }
            
        except Exception as e:
            print(f"Warning: Could not load hysteresis state: {e}")


def main():
    """Test hysteresis manager functionality."""
    manager = HysteresisManager()
    
    # Test daily drawdown
    result1 = manager.check_risk_metric("daily_drawdown", 0.03)  # Normal
    print(f"✅ Drawdown 3%: {result1['status']}")
    
    result2 = manager.check_risk_metric("daily_drawdown", 0.06)  # Should trigger warning
    print(f"✅ Drawdown 6%: {result2['status']}, Action: {result2['action_taken']}")
    
    result3 = manager.check_risk_metric("daily_drawdown", 0.08)  # Should escalate to critical
    print(f"✅ Drawdown 8%: {result3['status']}, Action: {result3['action_taken']}")
    
    # Test reset
    result4 = manager.check_risk_metric("daily_drawdown", 0.01)  # Should reset
    print(f"✅ Drawdown 1%: {result4['status']}, Action: {result4['action_taken']}")
    
    # Get summary
    summary = manager.get_risk_summary()
    print(f"✅ Risk summary: {summary['killed_metrics']} killed, {summary['warning_metrics']} warning")
    
    print("✅ Hysteresis manager testing completed")


if __name__ == "__main__":
    main()
