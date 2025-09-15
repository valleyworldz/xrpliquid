import time
import os
import sys
import json
from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager

class EmergencyHandler:
    def __init__(self):
        self.logger = Logger()
        self.config_manager = ConfigManager()
        self.last_success_timestamp = time.time()
        self.idle_threshold_seconds = 300 # 5 minutes of inactivity
        self.load_kill_switch_status()

    def load_kill_switch_status(self):
        try:
            # Use the new consolidated config manager
            self.kill_switch_active = self.config_manager.get("emergency.kill_switch_active", False)
            self.logger.info(f"Kill switch status loaded: {self.kill_switch_active}")
        except Exception as e:
            self.logger.error(f"Error loading kill switch status: {e}")
            self.kill_switch_active = False

    def update_heartbeat(self):
        """
        Updates the last successful activity timestamp.
        This should be called by critical services upon successful operation.
        """
        self.last_success_timestamp = time.time()

    def check_idle_status(self):
        """
        Checks if the system has been idle for too long.
        """
        time_since_last_success = time.time() - self.last_success_timestamp
        if time_since_last_success > self.idle_threshold_seconds:
            reason = f"System idle for {time_since_last_success:.2f} seconds, exceeding threshold of {self.idle_threshold_seconds} seconds."
            self.logger.error(reason, send_alert=True)
            self.trigger_auto_restart(reason)
            return True
        return False

    def is_kill_switch_active(self):
        """
        Checks if the kill switch is active. If active, the bot should cease trading.
        """
        self.load_kill_switch_status() # Reload to get latest status
        if self.kill_switch_active:
            self.logger.warning("Kill switch is active. Trading operations are suspended.", send_alert=True)
        return self.kill_switch_active

    def handle_emergency(self, reason):
        """
        Executes predefined emergency procedures.

        This typically involves:
        - Notifying relevant personnel.
        - Initiating system shutdown or safe mode.
        - Logging the incident for post-mortem analysis.

        Args:
            reason (str): A description of why the emergency handler was triggered.
        """
        self.logger.error(f"Emergency handled: {reason}. Initiating shutdown procedures...", send_alert=True)
        # In a real system, this would call the RiskManagement's emergency functions
        # from core.engines.risk_management import RiskManagement
        # risk_manager = RiskManagement()
        # risk_manager.emergency_handler(reason)
        # Further actions like sending notifications, saving state, etc.

    def trigger_auto_restart(self, reason):
        """
        Triggers an auto-restart of the bot.
        In a production environment, this would involve a process manager (e.g., systemd, Kubernetes) or a dedicated restart script.
        For this example, we'll log the intent and exit, relying on an external process manager.
        """
        self.logger.critical(f"Attempting to auto-restart bot due to: {reason}. Exiting process.")
        # In a real deployment, a process manager (e.g., systemd, supervisord, Kubernetes) would be configured
        # to restart this script automatically upon exit.
        sys.exit(1) # Exit with a non-zero status to indicate an error

    def set_kill_switch(self, status: bool):
        """
        Sets the status of the kill switch and saves it to the consolidated config.
        """
        try:
            self.config_manager.set("emergency.kill_switch_active", status)
            self.kill_switch_active = status
            self.logger.info(f"Kill switch status set to {status} and saved to config")
        except Exception as e:
            self.logger.error(f"Error setting kill switch status: {e}")



