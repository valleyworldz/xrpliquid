import time
import logging

class EmergencyStopHandler:
    def __init__(self, config):
        self.config = config
        self.stop_triggered = False
        self.stop_reason = None
        self.last_check_time = 0
        self.check_interval = config.get('emergency', {}).get('check_interval', 10)
        self.logger = logging.getLogger('EmergencyStopHandler')

    def check_emergency_conditions(self, system_status):
        """Check for emergency stop conditions"""
        now = time.time()
        if now - self.last_check_time < self.check_interval:
            return False
        self.last_check_time = now

        # Example: Check for critical errors or drawdown
        if system_status.get('critical_error', False):
            self.stop_triggered = True
            self.stop_reason = 'Critical error detected'
            self.logger.error('Emergency stop triggered: Critical error')
            return True
        if system_status.get('max_drawdown_exceeded', False):
            self.stop_triggered = True
            self.stop_reason = 'Max drawdown exceeded'
            self.logger.error('Emergency stop triggered: Max drawdown')
            return True
        return False

    def trigger_emergency_stop(self, reason):
        self.stop_triggered = True
        self.stop_reason = reason
        self.logger.error(f'Emergency stop triggered: {reason}')

    def reset(self):
        self.stop_triggered = False
        self.stop_reason = None 