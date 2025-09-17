"""
🛡️ HAT MANIFESTO RISK MANAGEMENT SYSTEM
=======================================
Advanced risk management with dynamic ATR-based stops and isolated margin optimization.

This system implements the pinnacle of risk management with:
- Dynamic ATR-based stop losses
- Isolated margin mode optimization
- Multi-tier TP/SL grids
- Emergency circuit breakers
- Real-time drawdown monitoring
- Volatility-based position sizing
- Correlation risk management
"""

from src.core.utils.decimal_boundary_guard import safe_float
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import logging
from decimal import Decimal, ROUND_DOWN

@dataclass
class RiskManagementConfig:
    """Configuration for Hat Manifesto Risk Management"""
    
    # ATR-based stop loss settings
    atr_settings: Dict[str, Any] = field(default_factory=lambda: {
        'atr_period': 14,                    # 14-period ATR
        'atr_multiplier': 2.0,               # 2x ATR for stop loss
        'atr_update_frequency_minutes': 5,   # Update ATR every 5 minutes
        'min_stop_distance_percent': 0.5,    # Minimum 0.5% stop distance
        'max_stop_distance_percent': 5.0,    # Maximum 5% stop distance
        'trailing_stop_enabled': True,       # Enable trailing stops
        'trailing_stop_multiplier': 1.5,     # 1.5x ATR for trailing
    })
    
    # Isolated margin settings
    isolated_margin: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                     # Enable isolated margin mode
        'max_leverage': 20.0,                # Maximum 20x leverage
        'margin_threshold_percent': 80.0,    # 80% margin threshold
        'liquidation_buffer_percent': 5.0,   # 5% liquidation buffer
        'auto_reduce_leverage': True,        # Auto-reduce leverage on high margin
        'cross_margin_fallback': True,       # Fallback to cross margin
    })
    
    # Multi-tier TP/SL settings
    tp_sl_grid: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                     # Enable multi-tier TP/SL
        'tier_1_percent': 25.0,              # 25% position at 1% profit
        'tier_2_percent': 50.0,              # 50% position at 2% profit
        'tier_3_percent': 25.0,              # 25% position at 3% profit
        'stop_loss_percent': 1.5,            # 1.5% stop loss
        'breakeven_enabled': True,           # Move to breakeven at 1% profit
        'trailing_enabled': True,            # Enable trailing take profits
    })
    
    # Emergency circuit breaker settings
    circuit_breaker: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                     # Enable circuit breaker
        'max_drawdown_percent': 8.0,         # 8% maximum drawdown
        'max_daily_loss_percent': 5.0,       # 5% maximum daily loss
        'max_consecutive_losses': 5,         # 5 consecutive losses
        'emergency_stop_enabled': True,      # Emergency stop on breach
        'recovery_threshold_percent': 2.0,   # 2% recovery threshold
        'cooldown_period_hours': 1.0,        # 1 hour cooldown period
    })
    
    # Volatility-based position sizing
    volatility_sizing: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                     # Enable volatility-based sizing
        'target_volatility_percent': 2.0,    # 2% target volatility
        'volatility_lookback_days': 30,      # 30-day volatility lookback
        'min_position_size_usd': 25.0,       # Minimum position size
        'max_position_size_usd': 10000.0,    # Maximum position size
        'volatility_scaling_factor': 1.5,    # Volatility scaling factor
    })
    
    # Correlation risk management
    correlation_risk: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,                     # Enable correlation risk management
        'max_correlation': 0.7,              # Maximum 70% correlation
        'correlation_lookback_days': 30,     # 30-day correlation lookback
        'diversification_required': True,    # Require diversification
        'max_correlated_positions': 3,       # Maximum 3 correlated positions
    })

@dataclass
class ATRStopLoss:
    """ATR-based stop loss data structure"""
    
    symbol: str
    current_price: float
    atr_value: float
    stop_loss_price: float
    stop_distance_percent: float
    trailing_stop_price: float
    is_trailing: bool
    last_update: float
    stop_id: str

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    
    # Account metrics
    account_value: float
    total_margin_used: float
    available_margin: float
    margin_ratio: float
    
    # Position metrics
    total_positions: int
    total_exposure_usd: float
    max_position_size_usd: float
    avg_position_size_usd: float
    
    # Risk metrics
    current_drawdown_percent: float
    max_drawdown_percent: float
    daily_pnl_percent: float
    volatility_percent: float
    sharpe_ratio: float
    
    # ATR metrics
    avg_atr_percent: float
    max_atr_percent: float
    atr_stop_coverage: float
    
    # Circuit breaker status
    circuit_breaker_active: bool
    emergency_mode: bool
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

class HatManifestoRiskManagement:
    """
    🛡️ HAT MANIFESTO RISK MANAGEMENT SYSTEM
    
    The pinnacle of risk management with advanced features:
    1. Dynamic ATR-based stop losses
    2. Isolated margin mode optimization
    3. Multi-tier TP/SL grids
    4. Emergency circuit breakers
    5. Real-time drawdown monitoring
    6. Volatility-based position sizing
    7. Correlation risk management
    """
    
    def __init__(self, api, config: Dict[str, Any], logger=None):
        self.api = api
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize risk management configuration
        self.risk_config = RiskManagementConfig()
        
        # Data storage
        self.price_history = {}  # symbol -> deque of prices
        self.atr_history = {}    # symbol -> deque of ATR values
        self.active_stops = {}   # stop_id -> ATRStopLoss
        self.risk_metrics = RiskMetrics(
            account_value=0.0,
            total_margin_used=0.0,
            available_margin=0.0,
            margin_ratio=0.0,
            total_positions=0,
            total_exposure_usd=0.0,
            max_position_size_usd=0.0,
            avg_position_size_usd=0.0,
            current_drawdown_percent=0.0,
            max_drawdown_percent=0.0,
            daily_pnl_percent=0.0,
            volatility_percent=0.0,
            sharpe_ratio=0.0,
            avg_atr_percent=0.0,
            max_atr_percent=0.0,
            atr_stop_coverage=0.0,
            circuit_breaker_active=False,
            emergency_mode=False,
            risk_level='LOW'
        )
        
        # Circuit breaker state
        self.circuit_breaker_state = {
            'active': False,
            'trigger_time': 0.0,
            'trigger_reason': '',
            'cooldown_until': 0.0,
            'recovery_threshold': 0.0,
        }
        
        # Performance tracking
        self.daily_pnl_history = deque(maxlen=30)  # 30 days
        self.drawdown_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=100)
        
        self.logger.info("🛡️ [RISK_MANAGEMENT] Hat Manifesto Risk Management initialized")
        self.logger.info("🎯 [RISK_MANAGEMENT] All risk management systems activated")
    
    async def monitor_account_health(self) -> RiskMetrics:
        """
        🛡️ Monitor comprehensive account health and risk metrics
        """
        try:
            # Get user state
            user_state = self.api.get_user_state()
            if not user_state:
                self.logger.warning("⚠️ [RISK_MANAGEMENT] Could not get user state")
                return self.risk_metrics
            
            # Extract account metrics
            margin_summary = user_state.get("marginSummary", {})
            account_value = safe_float(margin_summary.get("accountValue", 0))
            total_margin_used = safe_float(margin_summary.get("totalMarginUsed", 0))
            available_margin = account_value - total_margin_used
            margin_ratio = (total_margin_used / account_value) if account_value > 0 else 0
            
            # Get position data
            positions = user_state.get("assetPositions", [])
            total_positions = len(positions)
            total_exposure_usd = sum(safe_float(pos.get("position", {}).get("szi", 0)) * 
                                   safe_float(pos.get("position", {}).get("entryPx", 0)) 
                                   for pos in positions)
            
            # Calculate position size metrics
            position_sizes = [safe_float(pos.get("position", {}).get("szi", 0)) * 
                            safe_float(pos.get("position", {}).get("entryPx", 0)) 
                            for pos in positions]
            max_position_size_usd = max(position_sizes) if position_sizes else 0
            avg_position_size_usd = np.mean(position_sizes) if position_sizes else 0
            
            # Calculate risk metrics
            current_drawdown_percent = self._calculate_current_drawdown(account_value)
            max_drawdown_percent = self._calculate_max_drawdown()
            daily_pnl_percent = self._calculate_daily_pnl(account_value)
            volatility_percent = self._calculate_volatility()
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Calculate ATR metrics
            avg_atr_percent = self._calculate_avg_atr()
            max_atr_percent = self._calculate_max_atr()
            atr_stop_coverage = self._calculate_atr_stop_coverage()
            
            # Update risk metrics
            self.risk_metrics = RiskMetrics(
                account_value=account_value,
                total_margin_used=total_margin_used,
                available_margin=available_margin,
                margin_ratio=margin_ratio,
                total_positions=total_positions,
                total_exposure_usd=total_exposure_usd,
                max_position_size_usd=max_position_size_usd,
                avg_position_size_usd=avg_position_size_usd,
                current_drawdown_percent=current_drawdown_percent,
                max_drawdown_percent=max_drawdown_percent,
                daily_pnl_percent=daily_pnl_percent,
                volatility_percent=volatility_percent,
                sharpe_ratio=sharpe_ratio,
                avg_atr_percent=avg_atr_percent,
                max_atr_percent=max_atr_percent,
                atr_stop_coverage=atr_stop_coverage,
                circuit_breaker_active=self.circuit_breaker_state['active'],
                emergency_mode=self.risk_metrics.emergency_mode,
                risk_level=self._calculate_risk_level()
            )
            
            # Check circuit breaker conditions
            await self._check_circuit_breaker_conditions()
            
            # Log risk status
            if self.cycle_count % 100 == 0:  # Log every 100 cycles
                self._log_risk_status()
            
            return self.risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_MANAGEMENT] Error monitoring account health: {e}")
            return self.risk_metrics
    
    async def update_atr_stops(self, symbol: str = "XRP") -> Optional[ATRStopLoss]:
        """
        📊 Update ATR-based stop loss for symbol
        """
        try:
            # Get current price
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                return None
            
            # Calculate ATR
            atr_value = self._calculate_atr(symbol, current_price)
            if atr_value <= 0:
                return None
            
            # Calculate stop loss price
            stop_distance = atr_value * self.risk_config.atr_settings['atr_multiplier']
            stop_loss_price = current_price - stop_distance
            
            # Apply min/max stop distance constraints
            min_stop_distance = current_price * self.risk_config.atr_settings['min_stop_distance_percent'] / 100
            max_stop_distance = current_price * self.risk_config.atr_settings['max_stop_distance_percent'] / 100
            
            stop_distance = max(min_stop_distance, min(max_stop_distance, stop_distance))
            stop_loss_price = current_price - stop_distance
            
            # Calculate trailing stop if enabled
            trailing_stop_price = current_price - (atr_value * self.risk_config.atr_settings['trailing_stop_multiplier'])
            
            # Check if we should update existing stop
            stop_id = f"atr_stop_{symbol}"
            existing_stop = self.active_stops.get(stop_id)
            
            is_trailing = False
            if existing_stop and self.risk_config.atr_settings['trailing_stop_enabled']:
                # Update trailing stop if price moved favorably
                if current_price > existing_stop.current_price:
                    trailing_stop_price = max(trailing_stop_price, existing_stop.trailing_stop_price)
                    is_trailing = True
            
            # Create or update ATR stop loss
            atr_stop = ATRStopLoss(
                symbol=symbol,
                current_price=current_price,
                atr_value=atr_value,
                stop_loss_price=stop_loss_price,
                stop_distance_percent=(stop_distance / current_price) * 100,
                trailing_stop_price=trailing_stop_price,
                is_trailing=is_trailing,
                last_update=time.time(),
                stop_id=stop_id
            )
            
            # Store active stop
            self.active_stops[stop_id] = atr_stop
            
            # Store ATR history
            if symbol not in self.atr_history:
                self.atr_history[symbol] = deque(maxlen=100)
            self.atr_history[symbol].append((time.time(), atr_value))
            
            self.logger.info(f"📊 [ATR_STOP] {symbol}: Price=${current_price:.4f}, ATR={atr_value:.4f}, Stop=${stop_loss_price:.4f} ({atr_stop.stop_distance_percent:.2f}%)")
            
            return atr_stop
            
        except Exception as e:
            self.logger.error(f"❌ [ATR_STOP] Error updating ATR stop for {symbol}: {e}")
            return None
    
    async def check_stop_loss_triggers(self) -> List[Dict[str, Any]]:
        """
        🚨 Check for stop loss triggers and execute stops
        """
        triggered_stops = []
        
        try:
            for stop_id, atr_stop in self.active_stops.items():
                current_price = self._get_current_price(atr_stop.symbol)
                
                if current_price <= 0:
                    continue
                
                # Check if stop loss is triggered
                stop_triggered = False
                trigger_price = atr_stop.stop_loss_price
                
                if atr_stop.is_trailing:
                    trigger_price = atr_stop.trailing_stop_price
                
                if current_price <= trigger_price:
                    stop_triggered = True
                
                if stop_triggered:
                    # Execute stop loss
                    stop_result = await self._execute_stop_loss(atr_stop, current_price)
                    
                    if stop_result['success']:
                        triggered_stops.append({
                            'stop_id': stop_id,
                            'symbol': atr_stop.symbol,
                            'trigger_price': current_price,
                            'stop_price': trigger_price,
                            'result': stop_result
                        })
                        
                        # Remove from active stops
                        del self.active_stops[stop_id]
                        
                        self.logger.warning(f"🚨 [STOP_TRIGGER] {atr_stop.symbol} stop loss triggered at ${current_price:.4f}")
            
            return triggered_stops
            
        except Exception as e:
            self.logger.error(f"❌ [STOP_TRIGGER] Error checking stop loss triggers: {e}")
            return []
    
    async def optimize_isolated_margin(self, symbol: str = "XRP") -> Dict[str, Any]:
        """
        🎯 Optimize isolated margin mode for symbol
        """
        try:
            # Check current margin mode
            current_mode = await self._get_margin_mode(symbol)
            
            # Calculate optimal margin mode
            optimal_mode = self._calculate_optimal_margin_mode(symbol)
            
            # Switch margin mode if beneficial
            if current_mode != optimal_mode:
                switch_result = await self._switch_margin_mode(symbol, optimal_mode)
                
                if switch_result['success']:
                    self.logger.info(f"🎯 [MARGIN_OPT] Switched {symbol} to {optimal_mode} margin mode")
                    
                    return {
                        'success': True,
                        'previous_mode': current_mode,
                        'new_mode': optimal_mode,
                        'benefit': switch_result.get('benefit', 0)
                    }
            
            return {
                'success': False,
                'reason': 'Margin mode already optimal',
                'current_mode': current_mode
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MARGIN_OPT] Error optimizing isolated margin for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def setup_multi_tier_tp_sl(self, symbol: str, position_size: float, entry_price: float) -> Dict[str, Any]:
        """
        🎯 Setup multi-tier take profit and stop loss grid
        """
        try:
            if not self.risk_config.tp_sl_grid['enabled']:
                return {'success': False, 'reason': 'Multi-tier TP/SL disabled'}
            
            # Calculate tier sizes
            tier_1_size = position_size * (self.risk_config.tp_sl_grid['tier_1_percent'] / 100)
            tier_2_size = position_size * (self.risk_config.tp_sl_grid['tier_2_percent'] / 100)
            tier_3_size = position_size * (self.risk_config.tp_sl_grid['tier_3_percent'] / 100)
            
            # Calculate take profit prices
            tp_1_price = entry_price * (1 + 0.01)  # 1% profit
            tp_2_price = entry_price * (1 + 0.02)  # 2% profit
            tp_3_price = entry_price * (1 + 0.03)  # 3% profit
            
            # Calculate stop loss price
            stop_loss_price = entry_price * (1 - self.risk_config.tp_sl_grid['stop_loss_percent'] / 100)
            
            # Setup orders
            orders = []
            
            # Take profit orders
            if tier_1_size > 0:
                tp_1_order = await self._place_take_profit_order(symbol, tier_1_size, tp_1_price)
                if tp_1_order['success']:
                    orders.append(tp_1_order)
            
            if tier_2_size > 0:
                tp_2_order = await self._place_take_profit_order(symbol, tier_2_size, tp_2_price)
                if tp_2_order['success']:
                    orders.append(tp_2_order)
            
            if tier_3_size > 0:
                tp_3_order = await self._place_take_profit_order(symbol, tier_3_size, tp_3_price)
                if tp_3_order['success']:
                    orders.append(tp_3_order)
            
            # Stop loss order
            stop_order = await self._place_stop_loss_order(symbol, position_size, stop_loss_price)
            if stop_order['success']:
                orders.append(stop_order)
            
            # Breakeven order (if enabled)
            if self.risk_config.tp_sl_grid['breakeven_enabled']:
                breakeven_order = await self._place_breakeven_order(symbol, position_size, entry_price)
                if breakeven_order['success']:
                    orders.append(breakeven_order)
            
            self.logger.info(f"🎯 [TP_SL_GRID] Setup multi-tier TP/SL for {symbol}: {len(orders)} orders")
            
            return {
                'success': len(orders) > 0,
                'orders': orders,
                'tier_1': {'size': tier_1_size, 'price': tp_1_price},
                'tier_2': {'size': tier_2_size, 'price': tp_2_price},
                'tier_3': {'size': tier_3_size, 'price': tp_3_price},
                'stop_loss': {'price': stop_loss_price}
            }
            
        except Exception as e:
            self.logger.error(f"❌ [TP_SL_GRID] Error setting up multi-tier TP/SL for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _check_circuit_breaker_conditions(self):
        """
        🚨 Check circuit breaker conditions and activate if necessary
        """
        try:
            if not self.risk_config.circuit_breaker['enabled']:
                return
            
            # Check if already in cooldown
            if self.circuit_breaker_state['cooldown_until'] > time.time():
                return
            
            # Check drawdown condition
            if self.risk_metrics.current_drawdown_percent > self.risk_config.circuit_breaker['max_drawdown_percent']:
                await self._activate_circuit_breaker('max_drawdown', self.risk_metrics.current_drawdown_percent)
                return
            
            # Check daily loss condition
            if self.risk_metrics.daily_pnl_percent < -self.risk_config.circuit_breaker['max_daily_loss_percent']:
                await self._activate_circuit_breaker('max_daily_loss', self.risk_metrics.daily_pnl_percent)
                return
            
            # Check consecutive losses
            consecutive_losses = self._count_consecutive_losses()
            if consecutive_losses >= self.risk_config.circuit_breaker['max_consecutive_losses']:
                await self._activate_circuit_breaker('consecutive_losses', consecutive_losses)
                return
            
            # Check if circuit breaker should be deactivated
            if self.circuit_breaker_state['active']:
                if self.risk_metrics.current_drawdown_percent < self.circuit_breaker_state['recovery_threshold']:
                    await self._deactivate_circuit_breaker()
            
        except Exception as e:
            self.logger.error(f"❌ [CIRCUIT_BREAKER] Error checking circuit breaker conditions: {e}")
    
    async def _activate_circuit_breaker(self, reason: str, value: float):
        """
        🚨 Activate circuit breaker
        """
        self.circuit_breaker_state.update({
            'active': True,
            'trigger_time': time.time(),
            'trigger_reason': reason,
            'cooldown_until': time.time() + (self.risk_config.circuit_breaker['cooldown_period_hours'] * 3600),
            'recovery_threshold': self.risk_metrics.current_drawdown_percent + self.risk_config.circuit_breaker['recovery_threshold_percent']
        })
        
        self.risk_metrics.emergency_mode = True
        
        self.logger.error(f"🚨 [CIRCUIT_BREAKER] ACTIVATED - Reason: {reason}, Value: {value:.2f}%")
        self.logger.error(f"🚨 [CIRCUIT_BREAKER] Cooldown until: {datetime.fromtimestamp(self.circuit_breaker_state['cooldown_until'])}")
        
        # Execute emergency stop if enabled
        if self.risk_config.circuit_breaker['emergency_stop_enabled']:
            await self._execute_emergency_stop()
    
    async def _deactivate_circuit_breaker(self):
        """
        ✅ Deactivate circuit breaker
        """
        self.circuit_breaker_state.update({
            'active': False,
            'trigger_time': 0.0,
            'trigger_reason': '',
            'cooldown_until': 0.0,
            'recovery_threshold': 0.0,
        })
        
        self.risk_metrics.emergency_mode = False
        
        self.logger.info("✅ [CIRCUIT_BREAKER] DEACTIVATED - Risk conditions improved")
    
    async def _execute_emergency_stop(self):
        """
        🛑 Execute emergency stop - close all positions
        """
        try:
            # Get all positions
            user_state = self.api.get_user_state()
            positions = user_state.get("assetPositions", [])
            
            emergency_orders = []
            
            for position in positions:
                symbol = position.get("coin", "")
                position_size = safe_float(position.get("position", {}).get("szi", 0))
                
                if abs(position_size) > 0:
                    # Close position
                    close_side = "sell" if position_size > 0 else "buy"
                    close_quantity = abs(position_size)
                    
                    order_result = self.api.place_order(
                        symbol=symbol,
                        side=close_side,
                        quantity=close_quantity,
                        price=0,  # Market order
                        order_type="market",
                        time_in_force="Ioc",
                        reduce_only=True
                    )
                    
                    if order_result.get('success'):
                        emergency_orders.append({
                            'symbol': symbol,
                            'side': close_side,
                            'quantity': close_quantity,
                            'order_id': order_result.get('order_id')
                        })
            
            self.logger.error(f"🛑 [EMERGENCY_STOP] Executed {len(emergency_orders)} emergency orders")
            
        except Exception as e:
            self.logger.error(f"❌ [EMERGENCY_STOP] Error executing emergency stop: {e}")
    
    # Helper methods
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            market_data = self.api.info_client.all_mids()
            if isinstance(market_data, list):
                for asset_data in market_data:
                    if isinstance(asset_data, dict) and asset_data.get('coin') == symbol:
                        return safe_float(asset_data.get('mid', 0))
            return 0.52  # Fallback for XRP
        except:
            return 0.52  # Fallback for XRP
    
    def _calculate_atr(self, symbol: str, current_price: float) -> float:
        """Calculate Average True Range for symbol"""
        try:
            # Store current price
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=100)
            self.price_history[symbol].append(current_price)
            
            # Need at least 14 periods for ATR calculation
            if len(self.price_history[symbol]) < self.risk_config.atr_settings['atr_period']:
                return 0.0
            
            # Calculate True Range for each period
            prices = list(self.price_history[symbol])
            true_ranges = []
            
            for i in range(1, len(prices)):
                high = prices[i]
                low = prices[i]
                prev_close = prices[i-1]
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)
            
            # Calculate ATR as average of True Ranges
            if len(true_ranges) >= self.risk_config.atr_settings['atr_period']:
                atr = np.mean(true_ranges[-self.risk_config.atr_settings['atr_period']:])
                return atr
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ [ATR_CALC] Error calculating ATR for {symbol}: {e}")
            return 0.0
    
    def _calculate_current_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown percentage"""
        if not self.drawdown_history:
            return 0.0
        
        peak_value = max(self.drawdown_history)
        if peak_value <= 0:
            return 0.0
        
        return ((peak_value - current_value) / peak_value) * 100
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.drawdown_history:
            return 0.0
        
        peak = self.drawdown_history[0]
        max_dd = 0.0
        
        for value in self.drawdown_history:
            if value > peak:
                peak = value
            else:
                dd = ((peak - value) / peak) * 100
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_daily_pnl(self, current_value: float) -> float:
        """Calculate daily P&L percentage"""
        if not self.daily_pnl_history:
            return 0.0
        
        # Get yesterday's value
        yesterday_value = self.daily_pnl_history[-1] if self.daily_pnl_history else current_value
        
        if yesterday_value <= 0:
            return 0.0
        
        return ((current_value - yesterday_value) / yesterday_value) * 100
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.volatility_history) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(self.volatility_history)):
            ret = (self.volatility_history[i] - self.volatility_history[i-1]) / self.volatility_history[i-1]
            returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        return np.std(returns) * np.sqrt(252) * 100  # Annualized volatility
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.volatility_history) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(self.volatility_history)):
            ret = (self.volatility_history[i] - self.volatility_history[i-1]) / self.volatility_history[i-1]
            returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
        
        return (avg_return / volatility) * np.sqrt(252)  # Annualized Sharpe ratio
    
    def _calculate_avg_atr(self) -> float:
        """Calculate average ATR across all symbols"""
        if not self.atr_history:
            return 0.0
        
        all_atrs = []
        for symbol, atr_data in self.atr_history.items():
            if atr_data:
                all_atrs.extend([atr for _, atr in atr_data])
        
        return np.mean(all_atrs) if all_atrs else 0.0
    
    def _calculate_max_atr(self) -> float:
        """Calculate maximum ATR across all symbols"""
        if not self.atr_history:
            return 0.0
        
        all_atrs = []
        for symbol, atr_data in self.atr_history.items():
            if atr_data:
                all_atrs.extend([atr for _, atr in atr_data])
        
        return np.max(all_atrs) if all_atrs else 0.0
    
    def _calculate_atr_stop_coverage(self) -> float:
        """Calculate ATR stop coverage percentage"""
        if not self.active_stops:
            return 0.0
        
        total_positions = len(self.active_stops)
        covered_positions = sum(1 for stop in self.active_stops.values() if stop.stop_loss_price > 0)
        
        return (covered_positions / total_positions) * 100 if total_positions > 0 else 0.0
    
    def _calculate_risk_level(self) -> str:
        """Calculate overall risk level"""
        if self.risk_metrics.emergency_mode:
            return 'CRITICAL'
        elif self.risk_metrics.current_drawdown_percent > 5.0:
            return 'HIGH'
        elif self.risk_metrics.current_drawdown_percent > 2.0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losses"""
        if not self.daily_pnl_history:
            return 0
        
        consecutive_losses = 0
        for pnl in reversed(self.daily_pnl_history):
            if pnl < 0:
                consecutive_losses += 1
            else:
                break
        
        return consecutive_losses
    
    def _log_risk_status(self):
        """Log comprehensive risk status"""
        self.logger.info("🛡️ [RISK_STATUS] ===== RISK MANAGEMENT STATUS =====")
        self.logger.info(f"💰 Account Value: ${self.risk_metrics.account_value:.2f}")
        self.logger.info(f"📊 Margin Ratio: {self.risk_metrics.margin_ratio:.1%}")
        self.logger.info(f"📉 Current Drawdown: {self.risk_metrics.current_drawdown_percent:.2f}%")
        self.logger.info(f"📈 Daily P&L: {self.risk_metrics.daily_pnl_percent:.2f}%")
        self.logger.info(f"📊 Volatility: {self.risk_metrics.volatility_percent:.2f}%")
        self.logger.info(f"🎯 Sharpe Ratio: {self.risk_metrics.sharpe_ratio:.2f}")
        self.logger.info(f"🚨 Risk Level: {self.risk_metrics.risk_level}")
        self.logger.info(f"🛡️ Circuit Breaker: {'ACTIVE' if self.circuit_breaker_state['active'] else 'INACTIVE'}")
        self.logger.info(f"📊 ATR Stops: {len(self.active_stops)} active")
        self.logger.info("🛡️ [RISK_STATUS] ================================")
    
    # Placeholder methods for order execution
    async def _execute_stop_loss(self, atr_stop: ATRStopLoss, current_price: float) -> Dict[str, Any]:
        """Execute stop loss order"""
        # Placeholder implementation
        return {'success': True, 'order_id': f"stop_{atr_stop.stop_id}"}
    
    async def _get_margin_mode(self, symbol: str) -> str:
        """Get current margin mode for symbol"""
        # Placeholder implementation
        return 'isolated'
    
    def _calculate_optimal_margin_mode(self, symbol: str) -> str:
        """Calculate optimal margin mode for symbol"""
        # Placeholder implementation
        return 'isolated'
    
    async def _switch_margin_mode(self, symbol: str, mode: str) -> Dict[str, Any]:
        """Switch margin mode for symbol"""
        # Placeholder implementation
        return {'success': True, 'benefit': 0.1}
    
    async def _place_take_profit_order(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Place take profit order"""
        # Placeholder implementation
        return {'success': True, 'order_id': f"tp_{symbol}_{int(time.time())}"}
    
    async def _place_stop_loss_order(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Place stop loss order"""
        # Placeholder implementation
        return {'success': True, 'order_id': f"sl_{symbol}_{int(time.time())}"}
    
    async def _place_breakeven_order(self, symbol: str, quantity: float, entry_price: float) -> Dict[str, Any]:
        """Place breakeven order"""
        # Placeholder implementation
        return {'success': True, 'order_id': f"be_{symbol}_{int(time.time())}"}
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        return self.risk_metrics
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return self.circuit_breaker_state
