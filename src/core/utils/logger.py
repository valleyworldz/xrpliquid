import logging
import json
import requests
import os
import datetime
from src.core.utils.config_manager import ConfigManager

EMOJI_REPLACEMENTS = {
    '✅': '[OK]', '🚀': '[RUN]', '❌': '[FAIL]', '🪙': '[TOKEN]', '💰': '[MONEY]'
}

def setup_logger(name="hypeliquidOG", level=logging.INFO):
    """
    Setup and return a logger instance.
    
    Args:
        name (str): The name of the logger
        level: The logging level
        
    Returns:
        Logger: Configured logger instance
    """
    return Logger(name)

class Logger:
    def __init__(self, name="hypeliquidOG"):
        """
        Initializes the logger.

        Args:
            name (str): The name of the logger, typically the module or application name.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers to prevent duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create file handler
        fh = logging.FileHandler('logs/trading.log', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Add formatter to both handlers
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

        self.websocket_client = None # Placeholder for a WebSocket client instance
        
        # Initialize config manager for Telegram settings
        try:
            self.config_manager = ConfigManager()
            telegram_config = self.config_manager.get_telegram_config()
            self.telegram_bot_token = telegram_config.get("bot_token", "")
            self.telegram_chat_id = telegram_config.get("chat_id", "")
            self.telegram_enabled = telegram_config.get("enabled", False)
        except Exception as e:
            self.logger.warning(f"Could not load Telegram config: {e}")
            self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            self.telegram_enabled = False

        if not self.telegram_enabled or not self.telegram_bot_token or not self.telegram_chat_id:
            self.logger.warning("Telegram bot token or chat ID not configured. Cannot send alert.")

    def set_websocket_client(self, ws_client):
        """
        Set the WebSocket client for real-time logging
        """
        self.websocket_client = ws_client

    def _send_to_websocket(self, level, message, *args, **kwargs):
        """
        Sends log message to the connected WebSocket client.
        """
        if self.websocket_client and self.websocket_client.connected_event.is_set():
            try:
                # Format the message if args are provided
                if args:
                    formatted_message = message % args
                else:
                    formatted_message = message

                # Replace emojis
                for emoji, replacement in EMOJI_REPLACEMENTS.items():
                    formatted_message = formatted_message.replace(emoji, replacement)

                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "level": logging.getLevelName(level),
                    "message": formatted_message
                }
                # Assuming the websocket_client has a ws attribute with a send method
                if hasattr(self.websocket_client, 'ws') and hasattr(self.websocket_client.ws, 'send'):
                    self.websocket_client.ws.send(json.dumps(log_entry))
                else:
                    print("WebSocket client or its send method not properly configured.")
            except Exception as e:
                print(f"Error sending log to WebSocket: {e}")

    def _send_telegram_alert(self, message):
        """
        Sends alerts to Telegram.
        """
        if self.telegram_bot_token and self.telegram_chat_id:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            # Replace emojis in Telegram message
            for emoji, replacement in EMOJI_REPLACEMENTS.items():
                message = message.replace(emoji, replacement)
            payload = {"chat_id": self.telegram_chat_id, "text": message}
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                # print("Telegram alert sent successfully.")
            except requests.exceptions.RequestException as e:
                print(f"Error sending Telegram alert: {e}")
        else:
            print("Telegram bot token or chat ID not configured. Cannot send alert.")

    def info(self, message, *args, **kwargs):
        """
        Logs an informational message.
        """
        self.logger.info(message, *args, **kwargs)
        self._send_to_websocket(logging.INFO, message, *args, **kwargs)
        if kwargs.get("send_alert", False):
            self._send_telegram_alert(f"INFO: {message}")

    def error(self, message, *args, **kwargs):
        """
        Logs an error message.
        """
        self.logger.error(message, *args, **kwargs)
        self._send_to_websocket(logging.ERROR, message, *args, **kwargs)
        if kwargs.get("send_alert", True):
            self._send_telegram_alert(f"ERROR: {message}")

    def warning(self, message, *args, **kwargs):
        """
        Logs a warning message.
        """
        self.logger.warning(message, *args, **kwargs)
        self._send_to_websocket(logging.WARNING, message, *args, **kwargs)
        if kwargs.get("send_alert", False):
            self._send_telegram_alert(f"WARNING: {message}")

    def debug(self, message, *args, **kwargs):
        """
        Logs a debug message.
        """
        self.logger.debug(message, *args, **kwargs)
        self._send_to_websocket(logging.DEBUG, message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        """
        Logs a critical message.
        """
        self.logger.critical(message, *args, **kwargs)
        self._send_to_websocket(logging.CRITICAL, message, *args, **kwargs)
        if kwargs.get("send_alert", True):
            self._send_telegram_alert(f"CRITICAL: {message}")



