#!/usr/bin/env python3
"""
🔧 CENTRALIZED CONFIGURATION MANAGER
====================================

Unified configuration system for HyperliquidOG trading bot.
Consolidates all config files into a single, validated structure.

Features:
- Single source of truth for all configuration
- Environment variable support
- Validation and schema checking
- Default value management
- Secure credential handling
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Try to import dotenv, but handle gracefully if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(file_path):
        """Dummy function when dotenv is not available"""
        pass

class ConfigManager:
    """
    Centralized configuration manager for HyperliquidOG trading bot
    """
    
    def __init__(self, config_path: str = "config/parameters.json"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        self._load_env_vars()
        
        # Load and validate configuration
        self._load_config()
        
    def _load_env_vars(self):
        """Load environment variables from .env file"""
        # Try multiple possible .env locations
        env_files = [
            ".env",
            "configs/secrets.env",
            "credentials/.env"
        ]
        
        for env_file in env_files:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                self.logger.info(f"Loaded environment variables from {env_file}")
                break
        else:
            self.logger.warning("No .env file found, using system environment variables")
            
    def _load_config(self):
        """Load and merge all configuration files"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config from {self.config_path}: {e}")
                self._create_default_config()
        else:
            self.logger.info(f"Config file not found, creating default: {self.config_path}")
            self._create_default_config()
            
    def _create_default_config(self):
        """Create default configuration structure"""
        self.config = {
            "hyperliquid": {
                "api_url": "https://api.hyperliquid.xyz",
                "testnet": False,
                "timeout": 30,
                "websocket_url": "wss://api.hyperliquid.xyz/ws"
            },
            "trading": {
                "default_token": "DOGE",
                "default_strategy": "rl_ai",
                "aggressive_35_mode": False,
                "max_iterations": 100,
                "min_trade_interval": 10,
                "position_percentage": 0.1,
                "default_leverage": 10,
                "max_risk_per_trade": 0.01,
                "slippage_tolerance": 0.0005,
                "order_type": "market"
            },
            "risk_management": {
                "max_position_size": 0.5,
                "stop_loss_percentage": 0.05,
                "take_profit_percentage": 0.02,
                "max_consecutive_errors": 5,
                "max_trades_per_minute": 5,
                "max_trades_per_hour": 50,
                "emergency_stop_on_loss": 0.1
            },
            "strategies": {
                "scalping": {
                    "enabled": True,
                    "params": {
                        "grid_size": 0.0005,
                        "num_grids": 20,
                        "order_quantity": 0.001
                    }
                },
                "mean_reversion": {
                    "enabled": False,
                    "params": {
                        "entry_deviation": 0.0001,
                        "exit_deviation": 0.0002,
                        "max_position_size": 0.01,
                        "standard_deviations": 2
                    }
                },
                "grid_trading": {
                    "enabled": True,
                    "params": {
                        "grid_size": 0.001,
                        "num_grids": 10,
                        "order_quantity": 0.002
                    }
                },
                "rl_ai": {
                    "enabled": True,
                    "params": {
                        "learning_rate": 0.001,
                        "exploration_rate": 0.1,
                        "memory_size": 10000
                    }
                }
            },
            "monitoring": {
                "update_interval": 5,
                "log_level": "INFO",
                "show_progress": True,
                "file_rotation": True,
                "max_file_size": "10MB",
                "backup_count": 5
            },
            "telegram": {
                "enabled": False,
                "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                "chat_id": os.getenv("TELEGRAM_CHAT_ID", "")
            },
            "emergency": {
                "kill_switch_active": False,
                "auto_restart": True,
                "max_restart_attempts": 3
            }
        }
        
        # Save default config
        self._save_config()
        
    def _save_config(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'trading.default_token')"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set the value
        config[keys[-1]] = value
        
        # Save the updated config
        self._save_config()
        
    def validate_required_keys(self) -> bool:
        """Validate that all required configuration keys are present"""
        required_keys = [
            "hyperliquid.api_url",
            "trading.default_token",
            "trading.default_strategy"
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
                
        if missing_keys:
            self.logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
            
        return True
        
    def get_telegram_config(self) -> Dict[str, Any]:
        """Get Telegram configuration with validation"""
        telegram_config = self.get("telegram", {})
        
        if not telegram_config.get("enabled", False):
            return {"enabled": False}
            
        bot_token = telegram_config.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = telegram_config.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            self.logger.warning("Telegram bot token or chat ID not found. Telegram alerts will be disabled.")
            return {"enabled": False}
            
        return {
            "enabled": True,
            "bot_token": bot_token,
            "chat_id": chat_id
        }
        
    def get_hyperliquid_credentials(self) -> Dict[str, str]:
        """Get Hyperliquid API credentials from environment"""
        private_key = os.getenv("HL_PRIVATE_KEY")
        
        if not private_key:
            raise ValueError("HL_PRIVATE_KEY environment variable is required")
            
        return {
            "private_key": private_key
        }
        
    def reload(self):
        """Reload configuration from file"""
        self._load_env_vars()
        self._load_config()
        
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data"""
        return self.config.copy()
        
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration (excluding sensitive data)"""
        export_config = self.config.copy()
        
        # Remove sensitive data
        if "telegram" in export_config:
            export_config["telegram"]["bot_token"] = "***" if export_config["telegram"].get("bot_token") else ""
            
        return export_config

# Global config instance
_config_instance: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance 