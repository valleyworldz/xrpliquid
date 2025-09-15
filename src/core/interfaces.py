#!/usr/bin/env python3
"""
Base Interfaces and Abstract Classes for XRP Trading Bot
=======================================================

This module defines the core interfaces and abstract base classes that all
trading bot components must implement. This ensures consistency and enables
easy testing and mocking.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging


class ILogger(ABC):
    """Abstract interface for logging functionality"""
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Log info message"""
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Log warning message"""
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Log error message"""
        pass
    
    @abstractmethod
    def debug(self, message: str) -> None:
        """Log debug message"""
        pass


class IAPIClient(ABC):
    """Abstract interface for API client functionality"""
    
    @abstractmethod
    async def get_account_status(self) -> Optional[Dict[str, Any]]:
        """Get current account status"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str = "XRP") -> Optional[float]:
        """Get current price for symbol"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, is_buy: bool, size: float, 
                         price: float, order_type: str = "auto", 
                         urgency: str = "auto", is_exit: bool = False) -> Dict[str, Any]:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        pass


class IWebSocketManager(ABC):
    """Abstract interface for WebSocket management"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to WebSocket"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        pass
    
    @abstractmethod
    async def subscribe_to_ticker(self, symbol: str) -> bool:
        """Subscribe to ticker updates"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        pass
    
    @abstractmethod
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last price from WebSocket"""
        pass


class IRiskManager(ABC):
    """Abstract interface for risk management"""
    
    @abstractmethod
    def calculate_position_size(self, price: float, free_collateral: float, 
                               signal_confidence: float = 0.5) -> float:
        """Calculate position size based on risk parameters"""
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, entry_price: float, volatility: float, 
                           position_type: str = "LONG") -> float:
        """Calculate stop loss price"""
        pass
    
    @abstractmethod
    def should_reduce_risk(self, current_drawdown: float, volatility: float) -> bool:
        """Determine if risk should be reduced"""
        pass
    
    @abstractmethod
    def check_liquidation_risk(self, account_data: Dict[str, Any]) -> bool:
        """Check if account is at risk of liquidation"""
        pass


class IPatternAnalyzer(ABC):
    """Abstract interface for pattern analysis"""
    
    @abstractmethod
    def analyze_patterns(self, price_history: List[float], 
                        volume_history: Optional[List[float]] = None) -> Dict[str, Any]:
        """Analyze price patterns"""
        pass
    
    @abstractmethod
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        pass
    
    @abstractmethod
    def calculate_momentum(self, prices: List[float], period: int = 5) -> float:
        """Calculate momentum"""
        pass
    
    @abstractmethod
    def calculate_volatility(self, prices: List[float], period: int = 20) -> float:
        """Calculate volatility"""
        pass
    
    @abstractmethod
    def detect_market_regime(self, price_history: List[float], 
                           volume_history: Optional[List[float]] = None) -> str:
        """Detect market regime"""
        pass


class IPerformanceTracker(ABC):
    """Abstract interface for performance tracking"""
    
    @abstractmethod
    def record_trade(self, symbol: str, entry_price: float, exit_price: float,
                    size: float, entry_time: datetime, exit_time: datetime,
                    pnl: float, **additional_data: Any) -> None:
        """Record a completed trade"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        pass
    
    @abstractmethod
    def print_performance_summary(self) -> None:
        """Print performance summary"""
        pass
    
    @abstractmethod
    def get_win_rate(self) -> float:
        """Get win rate percentage"""
        pass
    
    @abstractmethod
    def get_total_pnl(self) -> float:
        """Get total PnL"""
        pass


class IFeeOptimizer(ABC):
    """Abstract interface for fee optimization"""
    
    @abstractmethod
    def calculate_effective_fee(self, order_type: str = "maker", 
                               current_tier: int = 0, staking_tier: str = "wood") -> float:
        """Calculate effective fee rate"""
        pass
    
    @abstractmethod
    def should_use_maker_order(self, urgency: str = "low") -> bool:
        """Determine if maker order should be used"""
        pass
    
    @abstractmethod
    def get_optimal_order_type(self, signal_strength: float, 
                              market_volatility: float) -> str:
        """Get optimal order type"""
        pass
    
    @abstractmethod
    def check_minimum_profit_requirement(self, entry_price: float, 
                                       target_price: float, position_size: float,
                                       order_type: str = "maker") -> bool:
        """Check if trade meets minimum profit requirement"""
        pass


class IOrderManager(ABC):
    """Abstract interface for order management"""
    
    @abstractmethod
    async def place_order(self, symbol: str, is_buy: bool, size: float,
                         price: float, order_type: str = "auto",
                         urgency: str = "auto", is_exit: bool = False) -> Dict[str, Any]:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    def validate_order_parameters(self, symbol: str, quantity: float,
                                price: float, is_buy: bool) -> bool:
        """Validate order parameters"""
        pass
    
    @abstractmethod
    def meets_min_notional(self, symbol: str, quantity: float,
                          price: Optional[float] = None) -> bool:
        """Check if order meets minimum notional"""
        pass
    
    @abstractmethod
    def align_price_to_tick(self, price: float, tick_size: float) -> float:
        """Align price to tick size"""
        pass


class IPositionManager(ABC):
    """Abstract interface for position management"""
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for symbol"""
        pass
    
    @abstractmethod
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions"""
        pass
    
    @abstractmethod
    def calculate_position_pnl(self, position: Dict[str, Any], 
                             current_price: float) -> float:
        """Calculate position PnL"""
        pass
    
    @abstractmethod
    def should_close_position(self, position: Dict[str, Any], 
                            current_price: float, profit_target: float,
                            stop_loss: float) -> Tuple[bool, str]:
        """Determine if position should be closed"""
        pass
    
    @abstractmethod
    def update_trailing_stop(self, position: Dict[str, Any], 
                           current_price: float) -> Optional[float]:
        """Update trailing stop for position"""
        pass


class ISignalGenerator(ABC):
    """Abstract interface for signal generation"""
    
    @abstractmethod
    def generate_signals(self, current_price: float, price_history: List[float],
                        volume_history: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def get_momentum_signals(self, current_price: float) -> List[Dict[str, Any]]:
        """Get momentum-based signals"""
        pass
    
    @abstractmethod
    def get_regression_signal(self, current_price: float, 
                            price_history: List[float]) -> Dict[str, Any]:
        """Get regression-based signal"""
        pass
    
    @abstractmethod
    def get_book_imbalance_signal(self, current_price: float) -> Dict[str, Any]:
        """Get order book imbalance signal"""
        pass


class IStrategyExecutor(ABC):
    """Abstract interface for strategy execution"""
    
    @abstractmethod
    async def execute_strategy(self, strategy_name: str, signals: List[Dict[str, Any]],
                             current_price: float, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading strategy"""
        pass
    
    @abstractmethod
    async def execute_grid_trading(self, current_price: float, 
                                 vol_regime: str) -> Dict[str, Any]:
        """Execute grid trading strategy"""
        pass
    
    @abstractmethod
    async def execute_market_making(self, current_price: float) -> Dict[str, Any]:
        """Execute market making strategy"""
        pass
    
    @abstractmethod
    async def execute_funding_arb(self) -> Dict[str, Any]:
        """Execute funding arbitrage strategy"""
        pass


class ITradingBot(ABC):
    """Abstract interface for the main trading bot"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the trading bot"""
        pass
    
    @abstractmethod
    async def run(self) -> None:
        """Run the trading bot"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the trading bot"""
        pass
    
    @abstractmethod
    async def execute_trading_cycle(self) -> None:
        """Execute one trading cycle"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        pass


class BaseComponent(ABC):
    """Base class for all trading bot components"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)


class BaseAPIClient(BaseComponent, IAPIClient):
    """Base class for API clients"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.connected = False
    
    @abstractmethod
    async def _connect(self) -> bool:
        """Internal connection method"""
        pass
    
    @abstractmethod
    async def _disconnect(self) -> None:
        """Internal disconnection method"""
        pass


class BaseRiskManager(BaseComponent, IRiskManager):
    """Base class for risk managers"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.max_drawdown = 0.05  # 5% default max drawdown


class BasePatternAnalyzer(BaseComponent, IPatternAnalyzer):
    """Base class for pattern analyzers"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.min_data_points = 10


class BasePerformanceTracker(BaseComponent, IPerformanceTracker):
    """Base class for performance trackers"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.trades = []
        self.total_pnl = 0.0
        self.winning_trades = 0
        self.total_trades = 0


# Type aliases for better code readability
Logger = Union[logging.Logger, ILogger]
APIClient = Union[BaseAPIClient, IAPIClient]
RiskManager = Union[BaseRiskManager, IRiskManager]
PatternAnalyzer = Union[BasePatternAnalyzer, IPatternAnalyzer]
PerformanceTracker = Union[BasePerformanceTracker, IPerformanceTracker]
FeeOptimizer = IFeeOptimizer
OrderManager = IOrderManager
PositionManager = IPositionManager
SignalGenerator = ISignalGenerator
StrategyExecutor = IStrategyExecutor
TradingBot = ITradingBot 