"""
🛡️ RISK OVERSIGHT OFFICER
"I am the circuit breaker. My purpose is survival above all else."

This module implements comprehensive risk management:
- Circuit breakers and kill switches
- Position sizing and exposure limits
- Drawdown monitoring and protection
- Volatility-based risk adjustment
- Correlation risk management
- Real-time risk monitoring
- Emergency shutdown procedures
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import json
from datetime import datetime, timedelta
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CircuitBreakerType(Enum):
    """Circuit breaker type enumeration"""
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    POSITION_SIZE = "position_size"
    EXPOSURE = "exposure"
    LIQUIDATION = "liquidation"

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.10  # 10% max drawdown
    max_position_size: float = 0.20  # 20% max position size
    max_total_exposure: float = 0.50  # 50% max total exposure
    max_volatility: float = 0.30  # 30% max volatility
    max_correlation: float = 0.80  # 80% max correlation
    max_leverage: float = 3.0  # 3x max leverage
    min_liquidity: float = 1000.0  # $1000 min liquidity
    max_slippage: float = 0.01  # 1% max slippage
    max_order_size: float = 10000.0  # $10k max order size

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    total_exposure: float = 0.0
    portfolio_volatility: float = 0.0
    max_correlation: float = 0.0
    liquidity_ratio: float = 0.0
    leverage_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    expected_shortfall: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

@dataclass
class CircuitBreaker:
    """Circuit breaker data structure"""
    breaker_type: CircuitBreakerType
    threshold: float
    current_value: float
    triggered: bool = False
    triggered_at: Optional[float] = None
    cooldown_period: float = 300.0  # 5 minutes
    auto_reset: bool = True

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    alert_type: str
    severity: RiskLevel
    message: str
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False

class RiskOversightOfficer:
    """
    Risk Oversight Officer - Master of Survival and Protection
    
    This class implements comprehensive risk management:
    1. Circuit breakers and kill switches
    2. Position sizing and exposure limits
    3. Drawdown monitoring and protection
    4. Volatility-based risk adjustment
    5. Correlation risk management
    6. Real-time risk monitoring
    7. Emergency shutdown procedures
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Risk configuration
        self.risk_limits = RiskLimits()
        self.risk_config = {
            'monitoring_interval': 1.0,  # seconds
            'alert_cooldown': 60.0,  # seconds
            'max_alerts': 100,
            'auto_shutdown': True,
            'emergency_stop_loss': 0.15,  # 15% emergency stop
            'position_sizing_method': 'kelly',  # 'kelly', 'fixed', 'volatility'
            'correlation_window': 30,  # days
            'volatility_window': 20,  # days
            'drawdown_window': 252  # trading days
        }
        
        # Risk state
        self.risk_metrics = RiskMetrics()
        self.circuit_breakers: Dict[CircuitBreakerType, CircuitBreaker] = {}
        self.risk_alerts: deque = deque(maxlen=self.risk_config['max_alerts'])
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.portfolio_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.risk_metrics_history: deque = deque(maxlen=1000)
        
        # Monitoring
        self.monitoring_thread = None
        self.running = False
        
        # Callbacks
        self.risk_callbacks: Dict[str, List[Callable]] = {
            'on_risk_alert': [],
            'on_circuit_breaker_triggered': [],
            'on_emergency_shutdown': [],
            'on_risk_limits_exceeded': []
        }
        
        # Initialize risk management
        self._initialize_risk_management()
    
    def _initialize_risk_management(self):
        """Initialize risk management system"""
        try:
            self.logger.info("Initializing risk oversight officer...")
            
            # Initialize circuit breakers
            self._initialize_circuit_breakers()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._risk_monitoring_loop,
                daemon=True,
                name="risk_monitor"
            )
            self.monitoring_thread.start()
            
            self.running = True
            self.logger.info("Risk oversight officer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing risk management: {e}")
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers"""
        try:
            self.circuit_breakers = {
                CircuitBreakerType.DAILY_LOSS: CircuitBreaker(
                    breaker_type=CircuitBreakerType.DAILY_LOSS,
                    threshold=self.risk_limits.max_daily_loss,
                    current_value=0.0
                ),
                CircuitBreakerType.DRAWDOWN: CircuitBreaker(
                    breaker_type=CircuitBreakerType.DRAWDOWN,
                    threshold=self.risk_limits.max_drawdown,
                    current_value=0.0
                ),
                CircuitBreakerType.VOLATILITY: CircuitBreaker(
                    breaker_type=CircuitBreakerType.VOLATILITY,
                    threshold=self.risk_limits.max_volatility,
                    current_value=0.0
                ),
                CircuitBreakerType.CORRELATION: CircuitBreaker(
                    breaker_type=CircuitBreakerType.CORRELATION,
                    threshold=self.risk_limits.max_correlation,
                    current_value=0.0
                ),
                CircuitBreakerType.POSITION_SIZE: CircuitBreaker(
                    breaker_type=CircuitBreakerType.POSITION_SIZE,
                    threshold=self.risk_limits.max_position_size,
                    current_value=0.0
                ),
                CircuitBreakerType.EXPOSURE: CircuitBreaker(
                    breaker_type=CircuitBreakerType.EXPOSURE,
                    threshold=self.risk_limits.max_total_exposure,
                    current_value=0.0
                ),
                CircuitBreakerType.LIQUIDATION: CircuitBreaker(
                    breaker_type=CircuitBreakerType.LIQUIDATION,
                    threshold=0.80,  # 80% of liquidation threshold
                    current_value=0.0
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing circuit breakers: {e}")
    
    def _risk_monitoring_loop(self):
        """Main risk monitoring loop"""
        try:
            while self.running:
                try:
                    # Update risk metrics
                    asyncio.run(self._update_risk_metrics())
                    
                    # Check circuit breakers
                    self._check_circuit_breakers()
                    
                    # Check risk limits
                    self._check_risk_limits()
                    
                    # Sleep for monitoring interval
                    time.sleep(self.risk_config['monitoring_interval'])
                    
                except Exception as e:
                    self.logger.error(f"Error in risk monitoring loop: {e}")
                    time.sleep(5.0)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in risk monitoring loop: {e}")
    
    async def _update_risk_metrics(self):
        """Update risk metrics"""
        try:
            # Update portfolio value
            current_equity = await self._get_current_equity()
            
            # Update peak equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            # Calculate drawdown
            if self.peak_equity > 0:
                self.risk_metrics.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
                self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, self.risk_metrics.current_drawdown)
            
            # Update daily PnL
            if self.daily_start_equity > 0:
                self.risk_metrics.daily_pnl = (current_equity - self.daily_start_equity) / self.daily_start_equity
            
            # Update total exposure
            self.risk_metrics.total_exposure = await self._calculate_total_exposure()
            
            # Update portfolio volatility
            self.risk_metrics.portfolio_volatility = await self._calculate_portfolio_volatility()
            
            # Update correlation
            self.risk_metrics.max_correlation = await self._calculate_max_correlation()
            
            # Update liquidity ratio
            self.risk_metrics.liquidity_ratio = await self._calculate_liquidity_ratio()
            
            # Update leverage ratio
            self.risk_metrics.leverage_ratio = await self._calculate_leverage_ratio()
            
            # Calculate VaR and Expected Shortfall
            self.risk_metrics.var_95 = await self._calculate_var_95()
            self.risk_metrics.expected_shortfall = await self._calculate_expected_shortfall()
            
            # Calculate Sharpe ratio
            self.risk_metrics.sharpe_ratio = await self._calculate_sharpe_ratio()
            
            # Calculate Calmar ratio
            if self.risk_metrics.max_drawdown > 0:
                self.risk_metrics.calmar_ratio = self.risk_metrics.sharpe_ratio / self.risk_metrics.max_drawdown
            
            # Store metrics history
            self.risk_metrics_history.append(self.risk_metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    async def _get_current_equity(self) -> float:
        """Get current portfolio equity"""
        try:
            # This would get actual equity from the exchange
            # For now, return a mock value
            return 10000.0  # $10,000
            
        except Exception as e:
            self.logger.error(f"Error getting current equity: {e}")
            return 0.0
    
    async def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        try:
            total_exposure = 0.0
            
            for symbol, position in self.positions.items():
                position_value = abs(position.get('size', 0.0) * position.get('price', 0.0))
                total_exposure += position_value
            
            return total_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating total exposure: {e}")
            return 0.0
    
    async def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        try:
            if len(self.portfolio_history) < 20:
                return 0.0
            
            # Get recent portfolio values
            portfolio_values = [entry['equity'] for entry in list(self.portfolio_history)[-20:]]
            
            # Calculate returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Calculate volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)  # 252 trading days
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    async def _calculate_max_correlation(self) -> float:
        """Calculate maximum correlation between positions"""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            # Get price histories for all positions
            price_histories = {}
            for symbol in self.positions.keys():
                price_histories[symbol] = await self._get_price_history(symbol, 30)
            
            # Calculate correlations
            max_correlation = 0.0
            symbols = list(price_histories.keys())
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    if len(price_histories[symbol1]) > 10 and len(price_histories[symbol2]) > 10:
                        # Calculate returns
                        returns1 = np.diff(price_histories[symbol1]) / price_histories[symbol1][:-1]
                        returns2 = np.diff(price_histories[symbol2]) / price_histories[symbol2][:-1]
                        
                        # Calculate correlation
                        if len(returns1) == len(returns2) and len(returns1) > 1:
                            correlation = np.corrcoef(returns1, returns2)[0, 1]
                            if not np.isnan(correlation):
                                max_correlation = max(max_correlation, abs(correlation))
            
            return max_correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating max correlation: {e}")
            return 0.0
    
    async def _calculate_liquidity_ratio(self) -> float:
        """Calculate liquidity ratio"""
        try:
            # This would calculate actual liquidity ratio
            # For now, return a mock value
            return 0.8  # 80% liquidity ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity ratio: {e}")
            return 0.0
    
    async def _calculate_leverage_ratio(self) -> float:
        """Calculate leverage ratio"""
        try:
            # This would calculate actual leverage ratio
            # For now, return a mock value
            return 1.5  # 1.5x leverage
            
        except Exception as e:
            self.logger.error(f"Error calculating leverage ratio: {e}")
            return 0.0
    
    async def _calculate_var_95(self) -> float:
        """Calculate Value at Risk at 95% confidence level"""
        try:
            if len(self.portfolio_history) < 20:
                return 0.0
            
            # Get recent portfolio values
            portfolio_values = [entry['equity'] for entry in list(self.portfolio_history)[-20:]]
            
            # Calculate returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Calculate VaR (95% confidence)
            var_95 = np.percentile(returns, 5)  # 5th percentile
            
            return abs(var_95)
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    async def _calculate_expected_shortfall(self) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(self.portfolio_history) < 20:
                return 0.0
            
            # Get recent portfolio values
            portfolio_values = [entry['equity'] for entry in list(self.portfolio_history)[-20:]]
            
            # Calculate returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Calculate Expected Shortfall
            var_95 = np.percentile(returns, 5)
            tail_returns = returns[returns <= var_95]
            
            if len(tail_returns) > 0:
                expected_shortfall = np.mean(tail_returns)
                return abs(expected_shortfall)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0
    
    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.portfolio_history) < 20:
                return 0.0
            
            # Get recent portfolio values
            portfolio_values = [entry['equity'] for entry in list(self.portfolio_history)[-20:]]
            
            # Calculate returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            if len(returns) > 1:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                if std_return > 0:
                    # Annualized Sharpe ratio
                    sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252))
                    return sharpe_ratio
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    async def _get_price_history(self, symbol: str, days: int) -> List[float]:
        """Get price history for symbol"""
        try:
            # This would get actual price history
            # For now, return mock data
            return [0.5 + 0.1 * np.sin(i) for i in range(days)]
            
        except Exception as e:
            self.logger.error(f"Error getting price history for {symbol}: {e}")
            return []
    
    def _check_circuit_breakers(self):
        """Check all circuit breakers"""
        try:
            for breaker_type, breaker in self.circuit_breakers.items():
                # Update current value
                if breaker_type == CircuitBreakerType.DAILY_LOSS:
                    breaker.current_value = abs(self.risk_metrics.daily_pnl)
                elif breaker_type == CircuitBreakerType.DRAWDOWN:
                    breaker.current_value = self.risk_metrics.current_drawdown
                elif breaker_type == CircuitBreakerType.VOLATILITY:
                    breaker.current_value = self.risk_metrics.portfolio_volatility
                elif breaker_type == CircuitBreakerType.CORRELATION:
                    breaker.current_value = self.risk_metrics.max_correlation
                elif breaker_type == CircuitBreakerType.POSITION_SIZE:
                    breaker.current_value = self._get_max_position_size()
                elif breaker_type == CircuitBreakerType.EXPOSURE:
                    breaker.current_value = self.risk_metrics.total_exposure / self.peak_equity if self.peak_equity > 0 else 0
                elif breaker_type == CircuitBreakerType.LIQUIDATION:
                    breaker.current_value = await self._get_liquidation_ratio()
                
                # Check if threshold is exceeded
                if breaker.current_value > breaker.threshold and not breaker.triggered:
                    self._trigger_circuit_breaker(breaker)
                elif breaker.current_value <= breaker.threshold and breaker.triggered:
                    self._reset_circuit_breaker(breaker)
                    
        except Exception as e:
            self.logger.error(f"Error checking circuit breakers: {e}")
    
    def _get_max_position_size(self) -> float:
        """Get maximum position size as percentage of portfolio"""
        try:
            if self.peak_equity == 0:
                return 0.0
            
            max_size = 0.0
            for symbol, position in self.positions.items():
                position_value = abs(position.get('size', 0.0) * position.get('price', 0.0))
                position_ratio = position_value / self.peak_equity
                max_size = max(max_size, position_ratio)
            
            return max_size
            
        except Exception as e:
            self.logger.error(f"Error getting max position size: {e}")
            return 0.0
    
    async def _get_liquidation_ratio(self) -> float:
        """Get liquidation ratio"""
        try:
            # This would get actual liquidation ratio from exchange
            # For now, return a mock value
            return 0.3  # 30% liquidation ratio
            
        except Exception as e:
            self.logger.error(f"Error getting liquidation ratio: {e}")
            return 0.0
    
    def _trigger_circuit_breaker(self, breaker: CircuitBreaker):
        """Trigger circuit breaker"""
        try:
            breaker.triggered = True
            breaker.triggered_at = time.time()
            
            # Create risk alert
            alert = RiskAlert(
                alert_id=f"cb_{breaker.breaker_type.value}_{int(time.time())}",
                alert_type="circuit_breaker",
                severity=RiskLevel.CRITICAL,
                message=f"Circuit breaker triggered: {breaker.breaker_type.value} exceeded threshold {breaker.threshold:.4f} (current: {breaker.current_value:.4f})",
                timestamp=time.time()
            )
            
            self.risk_alerts.append(alert)
            
            # Trigger callbacks
            self._trigger_callbacks('on_circuit_breaker_triggered', breaker)
            self._trigger_callbacks('on_risk_alert', alert)
            
            # Emergency shutdown if critical
            if breaker.breaker_type in [CircuitBreakerType.DAILY_LOSS, CircuitBreakerType.DRAWDOWN, CircuitBreakerType.LIQUIDATION]:
                self._emergency_shutdown(f"Critical circuit breaker triggered: {breaker.breaker_type.value}")
            
            self.logger.critical(f"Circuit breaker triggered: {breaker.breaker_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error triggering circuit breaker: {e}")
    
    def _reset_circuit_breaker(self, breaker: CircuitBreaker):
        """Reset circuit breaker"""
        try:
            if breaker.auto_reset:
                breaker.triggered = False
                breaker.triggered_at = None
                
                self.logger.info(f"Circuit breaker reset: {breaker.breaker_type.value}")
                
        except Exception as e:
            self.logger.error(f"Error resetting circuit breaker: {e}")
    
    def _check_risk_limits(self):
        """Check risk limits"""
        try:
            # Check daily loss limit
            if abs(self.risk_metrics.daily_pnl) > self.risk_limits.max_daily_loss:
                self._create_risk_alert(
                    "daily_loss_exceeded",
                    RiskLevel.HIGH,
                    f"Daily loss limit exceeded: {self.risk_metrics.daily_pnl:.4f} > {self.risk_limits.max_daily_loss:.4f}"
                )
            
            # Check drawdown limit
            if self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown:
                self._create_risk_alert(
                    "drawdown_exceeded",
                    RiskLevel.HIGH,
                    f"Drawdown limit exceeded: {self.risk_metrics.current_drawdown:.4f} > {self.risk_limits.max_drawdown:.4f}"
                )
            
            # Check volatility limit
            if self.risk_metrics.portfolio_volatility > self.risk_limits.max_volatility:
                self._create_risk_alert(
                    "volatility_exceeded",
                    RiskLevel.MEDIUM,
                    f"Volatility limit exceeded: {self.risk_metrics.portfolio_volatility:.4f} > {self.risk_limits.max_volatility:.4f}"
                )
            
            # Check correlation limit
            if self.risk_metrics.max_correlation > self.risk_limits.max_correlation:
                self._create_risk_alert(
                    "correlation_exceeded",
                    RiskLevel.MEDIUM,
                    f"Correlation limit exceeded: {self.risk_metrics.max_correlation:.4f} > {self.risk_limits.max_correlation:.4f}"
                )
            
            # Check leverage limit
            if self.risk_metrics.leverage_ratio > self.risk_limits.max_leverage:
                self._create_risk_alert(
                    "leverage_exceeded",
                    RiskLevel.HIGH,
                    f"Leverage limit exceeded: {self.risk_metrics.leverage_ratio:.4f} > {self.risk_limits.max_leverage:.4f}"
                )
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    def _create_risk_alert(self, alert_type: str, severity: RiskLevel, message: str):
        """Create risk alert"""
        try:
            alert = RiskAlert(
                alert_id=f"{alert_type}_{int(time.time())}",
                alert_type=alert_type,
                severity=severity,
                message=message,
                timestamp=time.time()
            )
            
            self.risk_alerts.append(alert)
            self._trigger_callbacks('on_risk_alert', alert)
            
            self.logger.warning(f"Risk alert: {message}")
            
        except Exception as e:
            self.logger.error(f"Error creating risk alert: {e}")
    
    def _emergency_shutdown(self, reason: str):
        """Emergency shutdown procedure"""
        try:
            self.logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
            
            # Create emergency alert
            alert = RiskAlert(
                alert_id=f"emergency_shutdown_{int(time.time())}",
                alert_type="emergency_shutdown",
                severity=RiskLevel.CRITICAL,
                message=f"Emergency shutdown initiated: {reason}",
                timestamp=time.time()
            )
            
            self.risk_alerts.append(alert)
            
            # Trigger callbacks
            self._trigger_callbacks('on_emergency_shutdown', alert)
            
            # Stop all trading
            self.running = False
            
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger event callbacks"""
        try:
            if event in self.risk_callbacks:
                for callback in self.risk_callbacks[event]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in risk callback for {event}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error triggering risk callbacks: {e}")
    
    def calculate_position_size(self, symbol: str, price: float, 
                              account_value: float, confidence: float) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            # Get current risk metrics
            current_volatility = self.risk_metrics.portfolio_volatility
            current_drawdown = self.risk_metrics.current_drawdown
            
            # Base position size
            if self.risk_config['position_sizing_method'] == 'kelly':
                # Kelly criterion
                base_size = confidence * account_value * 0.25  # 25% of Kelly
            elif self.risk_config['position_sizing_method'] == 'volatility':
                # Volatility-based sizing
                target_volatility = 0.02  # 2% target volatility
                base_size = (target_volatility * account_value) / (price * current_volatility) if current_volatility > 0 else 0
            else:
                # Fixed percentage
                base_size = account_value * 0.05  # 5% of account
            
            # Apply risk adjustments
            # Reduce size if drawdown is high
            if current_drawdown > 0.05:  # 5% drawdown
                drawdown_factor = 1.0 - (current_drawdown - 0.05) * 2  # Reduce by 2x excess drawdown
                base_size *= max(0.1, drawdown_factor)  # Minimum 10% of original size
            
            # Reduce size if volatility is high
            if current_volatility > 0.20:  # 20% volatility
                volatility_factor = 0.20 / current_volatility
                base_size *= volatility_factor
            
            # Apply position size limits
            max_position_value = account_value * self.risk_limits.max_position_size
            max_position_size = max_position_value / price
            
            # Apply order size limits
            max_order_size = self.risk_limits.max_order_size / price
            
            # Final position size
            position_size = min(base_size, max_position_size, max_order_size)
            
            # Ensure minimum size
            min_size = 1.0
            position_size = max(position_size, min_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def can_trade(self, symbol: str, side: str, size: float, price: float) -> bool:
        """Check if trade is allowed based on risk management"""
        try:
            # Check if any circuit breakers are triggered
            for breaker in self.circuit_breakers.values():
                if breaker.triggered:
                    self.logger.warning(f"Trade blocked: Circuit breaker {breaker.breaker_type.value} is triggered")
                    return False
            
            # Check position size limits
            position_value = size * price
            if position_value > self.risk_limits.max_order_size:
                self.logger.warning(f"Trade blocked: Order size {position_value} exceeds limit {self.risk_limits.max_order_size}")
                return False
            
            # Check total exposure limits
            current_exposure = self.risk_metrics.total_exposure
            if current_exposure + position_value > self.peak_equity * self.risk_limits.max_total_exposure:
                self.logger.warning(f"Trade blocked: Total exposure would exceed limit")
                return False
            
            # Check daily loss limits
            if abs(self.risk_metrics.daily_pnl) > self.risk_limits.max_daily_loss:
                self.logger.warning(f"Trade blocked: Daily loss limit exceeded")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking if trade is allowed: {e}")
            return False
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position information"""
        try:
            self.positions[symbol] = position_data
            
        except Exception as e:
            self.logger.error(f"Error updating position for {symbol}: {e}")
    
    def update_portfolio_value(self, equity: float):
        """Update portfolio value"""
        try:
            # Store portfolio history
            self.portfolio_history.append({
                'equity': equity,
                'timestamp': time.time()
            })
            
            # Update daily start equity if it's a new day
            current_time = time.time()
            if not hasattr(self, 'last_daily_reset') or current_time - self.last_daily_reset > 86400:  # 24 hours
                self.daily_start_equity = equity
                self.last_daily_reset = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add risk event callback"""
        if event in self.risk_callbacks:
            self.risk_callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove risk event callback"""
        if event in self.risk_callbacks and callback in self.risk_callbacks[event]:
            self.risk_callbacks[event].remove(callback)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        return self.risk_metrics
    
    def get_circuit_breakers(self) -> Dict[CircuitBreakerType, CircuitBreaker]:
        """Get circuit breaker status"""
        return self.circuit_breakers.copy()
    
    def get_risk_alerts(self) -> List[RiskAlert]:
        """Get recent risk alerts"""
        return list(self.risk_alerts)
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        return self.positions.copy()
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge risk alert"""
        try:
            for alert in self.risk_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    break
                    
        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve risk alert"""
        try:
            for alert in self.risk_alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    break
                    
        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {e}")
    
    def shutdown(self):
        """Shutdown risk management system"""
        try:
            self.running = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            self.logger.info("Risk oversight officer shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during risk management shutdown: {e}")

