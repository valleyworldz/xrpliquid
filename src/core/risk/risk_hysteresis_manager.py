"""
Risk Hysteresis Manager
Implements cool-downs and staged re-enable to avoid kill-switch flapping.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskHysteresisManager:
    """Manages risk hysteresis to prevent kill-switch flapping."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Hysteresis state
        self.kill_switch_active = False
        self.kill_switch_triggered_at = None
        self.cooldown_end_time = None
        self.current_recovery_stage = 0
        self.position_size_multiplier = 1.0
        
        # Load configuration
        self.config = self.load_hysteresis_config()
        
        # Load state from disk
        self.load_hysteresis_state()
    
    def load_hysteresis_config(self) -> Dict:
        """Load hysteresis configuration."""
        config_file = self.repo_root / "config" / "sizing_by_regime.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                return config_data.get('hysteresis_settings', {})
        else:
            # Default configuration
            return {
                'kill_switch_cooldown_hours': 24,
                'reduction_threshold': 0.03,
                'recovery_threshold': 0.01,
                'staged_recovery': True,
                'recovery_stages': [0.25, 0.5, 0.75, 1.0]
            }
    
    def load_hysteresis_state(self):
        """Load hysteresis state from disk."""
        state_file = self.reports_dir / "risk" / "hysteresis_state.json"
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                self.kill_switch_active = state_data.get('kill_switch_active', False)
                self.kill_switch_triggered_at = state_data.get('kill_switch_triggered_at')
                self.cooldown_end_time = state_data.get('cooldown_end_time')
                self.current_recovery_stage = state_data.get('current_recovery_stage', 0)
                self.position_size_multiplier = state_data.get('position_size_multiplier', 1.0)
            
            if self.kill_switch_triggered_at:
                self.kill_switch_triggered_at = datetime.fromisoformat(self.kill_switch_triggered_at)
            if self.cooldown_end_time:
                self.cooldown_end_time = datetime.fromisoformat(self.cooldown_end_time)
            
            logger.info(f"📂 Loaded hysteresis state: active={self.kill_switch_active}, stage={self.current_recovery_stage}")
    
    def save_hysteresis_state(self):
        """Save hysteresis state to disk."""
        risk_dir = self.reports_dir / "risk"
        risk_dir.mkdir(exist_ok=True)
        
        state_data = {
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_triggered_at': self.kill_switch_triggered_at.isoformat() if self.kill_switch_triggered_at else None,
            'cooldown_end_time': self.cooldown_end_time.isoformat() if self.cooldown_end_time else None,
            'current_recovery_stage': self.current_recovery_stage,
            'position_size_multiplier': self.position_size_multiplier,
            'last_updated': datetime.now().isoformat()
        }
        
        state_file = risk_dir / "hysteresis_state.json"
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"💾 Hysteresis state saved: {state_file}")
    
    def check_kill_switch_trigger(self, current_drawdown: float, realized_pnl: float) -> bool:
        """Check if kill switch should be triggered."""
        reduction_threshold = self.config.get('reduction_threshold', 0.03)
        
        # Trigger kill switch if drawdown exceeds threshold
        if current_drawdown > reduction_threshold:
            if not self.kill_switch_active:
                self.trigger_kill_switch(current_drawdown, realized_pnl)
            return True
        
        return False
    
    def trigger_kill_switch(self, drawdown: float, realized_pnl: float):
        """Trigger the kill switch with cooldown."""
        self.kill_switch_active = True
        self.kill_switch_triggered_at = datetime.now()
        
        # Set cooldown period
        cooldown_hours = self.config.get('kill_switch_cooldown_hours', 24)
        self.cooldown_end_time = self.kill_switch_triggered_at + timedelta(hours=cooldown_hours)
        
        # Reset recovery stage
        self.current_recovery_stage = 0
        self.position_size_multiplier = 0.0
        
        # Log kill switch event
        self.log_kill_switch_event(drawdown, realized_pnl)
        
        logger.warning(f"🚨 KILL SWITCH TRIGGERED: Drawdown {drawdown:.2%}, PnL ${realized_pnl:.2f}")
        logger.info(f"⏰ Cooldown until: {self.cooldown_end_time}")
    
    def check_recovery_conditions(self, current_drawdown: float, realized_pnl: float) -> bool:
        """Check if recovery conditions are met."""
        if not self.kill_switch_active:
            return False
        
        recovery_threshold = self.config.get('recovery_threshold', 0.01)
        
        # Check if cooldown period has passed
        if datetime.now() < self.cooldown_end_time:
            return False
        
        # Check if drawdown has recovered below threshold
        if current_drawdown <= recovery_threshold:
            return True
        
        return False
    
    def advance_recovery_stage(self, current_drawdown: float, realized_pnl: float):
        """Advance to next recovery stage."""
        if not self.kill_switch_active:
            return
        
        recovery_stages = self.config.get('recovery_stages', [0.25, 0.5, 0.75, 1.0])
        
        # Advance to next stage
        self.current_recovery_stage += 1
        
        if self.current_recovery_stage >= len(recovery_stages):
            # Full recovery
            self.complete_recovery()
        else:
            # Partial recovery
            self.position_size_multiplier = recovery_stages[self.current_recovery_stage]
            self.log_recovery_stage_advancement(current_drawdown, realized_pnl)
            
            logger.info(f"📈 Recovery stage {self.current_recovery_stage}: {self.position_size_multiplier:.0%} position size")
    
    def complete_recovery(self):
        """Complete the recovery process."""
        self.kill_switch_active = False
        self.kill_switch_triggered_at = None
        self.cooldown_end_time = None
        self.current_recovery_stage = 0
        self.position_size_multiplier = 1.0
        
        self.log_recovery_completion()
        logger.info("✅ RECOVERY COMPLETE: Full trading capacity restored")
    
    def get_position_size_multiplier(self) -> float:
        """Get current position size multiplier."""
        return self.position_size_multiplier
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        if not self.kill_switch_active:
            return True
        
        # Check if cooldown period has passed
        if datetime.now() >= self.cooldown_end_time:
            return True
        
        return False
    
    def get_hysteresis_status(self) -> Dict:
        """Get current hysteresis status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_triggered_at': self.kill_switch_triggered_at.isoformat() if self.kill_switch_triggered_at else None,
            'cooldown_end_time': self.cooldown_end_time.isoformat() if self.cooldown_end_time else None,
            'current_recovery_stage': self.current_recovery_stage,
            'position_size_multiplier': self.position_size_multiplier,
            'trading_allowed': self.is_trading_allowed(),
            'time_until_cooldown_end': (
                (self.cooldown_end_time - datetime.now()).total_seconds() / 3600
                if self.cooldown_end_time and datetime.now() < self.cooldown_end_time
                else 0
            )
        }
    
    def log_kill_switch_event(self, drawdown: float, realized_pnl: float):
        """Log kill switch event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'kill_switch_triggered',
            'drawdown': drawdown,
            'realized_pnl': realized_pnl,
            'cooldown_hours': self.config.get('kill_switch_cooldown_hours', 24)
        }
        
        self.save_risk_event(event)
    
    def log_recovery_stage_advancement(self, drawdown: float, realized_pnl: float):
        """Log recovery stage advancement."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'recovery_stage_advancement',
            'stage': self.current_recovery_stage,
            'position_size_multiplier': self.position_size_multiplier,
            'drawdown': drawdown,
            'realized_pnl': realized_pnl
        }
        
        self.save_risk_event(event)
    
    def log_recovery_completion(self):
        """Log recovery completion."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'recovery_complete',
            'total_recovery_time_hours': (
                (datetime.now() - self.kill_switch_triggered_at).total_seconds() / 3600
                if self.kill_switch_triggered_at
                else 0
            )
        }
        
        self.save_risk_event(event)
    
    def save_risk_event(self, event: Dict):
        """Save risk event to log."""
        events_dir = self.reports_dir / "risk_events"
        events_dir.mkdir(exist_ok=True)
        
        event_file = events_dir / "risk_events.jsonl"
        with open(event_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        logger.info(f"📝 Risk event logged: {event['event_type']}")


def main():
    """Main function to demonstrate risk hysteresis management."""
    manager = RiskHysteresisManager()
    
    # Simulate some risk events
    print("🧪 Testing risk hysteresis management...")
    
    # Test kill switch trigger
    print(f"Initial status: {manager.get_hysteresis_status()}")
    
    # Trigger kill switch
    manager.check_kill_switch_trigger(0.05, -1000)  # 5% drawdown
    print(f"After kill switch: {manager.get_hysteresis_status()}")
    
    # Test recovery
    manager.check_recovery_conditions(0.005, -100)  # 0.5% drawdown
    if manager.check_recovery_conditions(0.005, -100):
        manager.advance_recovery_stage(0.005, -100)
        print(f"After recovery stage: {manager.get_hysteresis_status()}")
    
    # Save state
    manager.save_hysteresis_state()
    
    print("✅ Risk hysteresis management demonstration completed")


if __name__ == "__main__":
    main()
