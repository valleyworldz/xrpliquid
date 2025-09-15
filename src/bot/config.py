#!/usr/bin/env python3
"""
Configuration Management for XRP Trading Bot
"""

from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os

class TradingConfig(BaseSettings):
    """Trading configuration with validation"""
    
    # Core trading parameters
    PROFIT_TARGET_PCT: float = Field(default=0.035, ge=0.001, le=0.5, description="Take profit percentage")
    STOP_LOSS_PCT: float = Field(default=0.025, ge=0.001, le=0.5, description="Stop loss percentage")
    MIN_POSITION_SIZE: float = Field(default=10.0, ge=1.0, le=1000.0, description="Minimum position size in XRP")
    PRICE_PRECISION: int = Field(default=4, ge=1, le=8, description="Price precision")
    MIN_ORDER_VALUE: float = Field(default=10.0, ge=1.0, description="Minimum order value in USD")
    TRAILING_DISTANCE: float = Field(default=0.015, ge=0.001, le=0.1, description="Trailing stop distance")
    CONFIDENCE_THRESHOLD: float = Field(default=0.95, ge=0.5, le=1.0, description="Minimum confidence threshold")
    VOLUME_THRESHOLD: float = Field(default=0.8, ge=0.1, le=1.0, description="Volume threshold")
    
    # Risk management
    MAX_DAILY_LOSS: float = Field(default=0.1, ge=0.01, le=0.5, description="Maximum daily loss percentage")
    MAX_CONSECUTIVE_LOSSES: int = Field(default=3, ge=1, le=10, description="Maximum consecutive losses")
    MAX_DAILY_TRADES: int = Field(default=50, ge=1, le=1000, description="Maximum trades per day")
    RISK_PER_TRADE: float = Field(default=0.02, ge=0.001, le=0.1, description="Risk per trade percentage")
    
    # Position sizing
    BASE_POSITION_SIZE: float = Field(default=5.0, ge=1.0, le=100.0, description="Base position size in XRP")
    MAX_POSITION_SIZE: float = Field(default=10.0, ge=1.0, le=1000.0, description="Maximum position size in XRP")
    COMPOUND_FACTOR: float = Field(default=1.1, ge=1.0, le=2.0, description="Position size compound factor")
    
    # Technical indicators
    MACD_FAST: int = Field(default=12, ge=5, le=50, description="MACD fast period")
    MACD_SLOW: int = Field(default=26, ge=10, le=100, description="MACD slow period")
    MACD_SIGNAL: int = Field(default=9, ge=5, le=50, description="MACD signal period")
    RSI_PERIOD: int = Field(default=14, ge=5, le=50, description="RSI period")
    ATR_PERIOD: int = Field(default=14, ge=5, le=50, description="ATR period")
    
    # Execution settings
    CYCLE_INTERVAL: int = Field(default=2, ge=1, le=60, description="Trading cycle interval in seconds")
    POST_TRADE_COOLDOWN: int = Field(default=300, ge=0, le=3600, description="Post-trade cooldown in seconds")
    EMERGENCY_STOP_LOSS: float = Field(default=0.005, ge=0.001, le=0.1, description="Emergency stop loss percentage")
    
    # TP/SL settings
    GROK_TP_TIERS: List[float] = Field(default=[0.01, 0.02, 0.03], description="Take profit tiers")
    GROK_TP_SIZES: List[float] = Field(default=[0.3, 0.3, 0.4], description="Take profit sizes")
    GROK_BREAKEVEN_ACTIVATION: float = Field(default=0.01, ge=0.001, le=0.1, description="Breakeven activation")
    GROK_TRAILING_ACTIVATION: float = Field(default=0.005, ge=0.001, le=0.1, description="Trailing activation")
    GROK_ATR_MULTIPLIER: float = Field(default=2.0, ge=0.5, le=5.0, description="ATR multiplier for stops")
    
    # API settings
    API_TIMEOUT: int = Field(default=30, ge=5, le=120, description="API timeout in seconds")
    WS_MAX_RECONNECT_ATTEMPTS: int = Field(default=5, ge=1, le=20, description="WebSocket max reconnect attempts")
    WS_PING_INTERVAL: int = Field(default=30, ge=10, le=120, description="WebSocket ping interval")
    
    # Logging and monitoring
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$", description="Log level")
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=8000, ge=1000, le=65535, description="Metrics server port")
    
    # Email alerts
    EMAIL_ALERTS: bool = Field(default=False, description="Enable email alerts")
    EMAIL_SENDER: Optional[str] = Field(default=None, description="Email sender address")
    EMAIL_PASSWORD: Optional[str] = Field(default=None, description="Email password")
    EMAIL_RECEIVER: Optional[str] = Field(default=None, description="Email receiver address")
    
    # Feature flags
    ENABLE_COMPOUNDING: bool = Field(default=True, description="Enable position size compounding")
    ENABLE_FUNDING_FILTERS: bool = Field(default=True, description="Enable funding rate filters")
    ENABLE_VOLATILITY_FILTERS: bool = Field(default=True, description="Enable volatility filters")
    ENABLE_ATR_STOPS: bool = Field(default=True, description="Enable ATR-based stops")
    
    @validator('GROK_TP_TIERS', 'GROK_TP_SIZES')
    def validate_tp_config(cls, v, values):
        """Validate TP configuration"""
        if 'GROK_TP_TIERS' in values and 'GROK_TP_SIZES' in values:
            if len(values['GROK_TP_TIERS']) != len(values['GROK_TP_SIZES']):
                raise ValueError("TP tiers and sizes must have the same length")
            if sum(values['GROK_TP_SIZES']) != 1.0:
                raise ValueError("TP sizes must sum to 1.0")
        return v
    
    @validator('EMAIL_SENDER', 'EMAIL_PASSWORD', 'EMAIL_RECEIVER')
    def validate_email_config(cls, v, values):
        """Validate email configuration"""
        if values.get('EMAIL_ALERTS', False):
            if not v:
                raise ValueError("Email configuration required when EMAIL_ALERTS is True")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class BotConfig:
    """Bot configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration"""
        self.config = TradingConfig()
        self.config_file = config_file
        
        # Load from file if specified
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from file"""
        try:
            # This would load from JSON/YAML file
            # For now, we use environment variables
            pass
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        setattr(self.config, key, value)
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            self.config.validate()
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.dict()
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to config"""
        return getattr(self.config, name)

# Global configuration instance
config = BotConfig()

# Constants (moved from main file)
MIN_NOTIONAL = 10.0
HYPER_OPTIMIZED_MODE = True
WS_ERROR_SUPPRESSION_TIME = 60

# Legacy compatibility
def get_config():
    """Get configuration for legacy compatibility"""
    return config 