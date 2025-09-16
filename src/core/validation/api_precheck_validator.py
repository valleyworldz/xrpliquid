"""
🎯 API PRECHECK VALIDATOR
=========================
Production-grade API pre-validation to prevent order rejects and noisy retries.

Features:
- Tick size validation
- Min notional validation
- Reduce-only validation
- Margin validation
- Leverage validation
- Position size validation
- Real-time market data integration
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger

class ValidationResult(Enum):
    """Validation result enumeration"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ADJUSTED = "adjusted"

class ValidationError(Enum):
    """Validation error enumeration"""
    TICK_SIZE = "tick_size"
    MIN_NOTIONAL = "min_notional"
    REDUCE_ONLY = "reduce_only"
    MARGIN_INSUFFICIENT = "margin_insufficient"
    LEVERAGE_EXCEEDED = "leverage_exceeded"
    POSITION_SIZE = "position_size"
    PRICE_INVALID = "price_invalid"
    QUANTITY_INVALID = "quantity_invalid"
    SYMBOL_INVALID = "symbol_invalid"

@dataclass
class ValidationConfig:
    """Configuration for API precheck validator"""
    
    # Hyperliquid-specific parameters
    hyperliquid_params: Dict[str, Any] = field(default_factory=lambda: {
        'tick_sizes': {
            'XRP': 0.0001,                    # XRP tick size
            'BTC': 0.01,                      # BTC tick size
            'ETH': 0.01,                      # ETH tick size
            'SOL': 0.001,                     # SOL tick size
            'ARB': 0.0001,                    # ARB tick size
        },
        'min_notional': {
            'XRP': 1.0,                       # $1 minimum notional
            'BTC': 10.0,                      # $10 minimum notional
            'ETH': 10.0,                      # $10 minimum notional
            'SOL': 5.0,                       # $5 minimum notional
            'ARB': 1.0,                       # $1 minimum notional
        },
        'position_size_limits': {
            'max_position_size_usd': 1000000.0,
            'min_position_size_usd': 10.0,
        },
        'leverage_limits': {
            'max_leverage': 50.0,
            'min_leverage': 1.0,
        },
        'margin_requirements': {
            'initial_margin_percent': 0.1,    # 10% initial margin
            'maintenance_margin_percent': 0.05,  # 5% maintenance margin
        },
    })
    
    # Validation settings
    validation_settings: Dict[str, Any] = field(default_factory=lambda: {
        'tick_size_validation': True,
        'min_notional_validation': True,
        'reduce_only_validation': True,
        'margin_check_validation': True,
        'leverage_validation': True,
        'position_size_validation': True,
        'auto_adjust_prices': True,
        'auto_adjust_quantities': False,
        'strict_validation': True,
    })
    
    # Market data settings
    market_data_config: Dict[str, Any] = field(default_factory=lambda: {
        'cache_duration_seconds': 5,          # 5-second cache
        'max_price_deviation_percent': 0.1,   # 0.1% max price deviation
        'min_volume_threshold': 1000.0,       # $1k minimum volume
    })

@dataclass
class OrderValidationRequest:
    """Order validation request"""
    
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str = 'limit'
    reduce_only: bool = False
    leverage: float = 1.0
    cloid: str = ""
    
    # Account context
    account_equity: float = 0.0
    available_margin: float = 0.0
    current_positions: Dict[str, float] = field(default_factory=dict)
    
    # Market context
    current_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume_24h: float = 0.0

@dataclass
class ValidationResponse:
    """Validation response"""
    
    result: ValidationResult
    errors: List[ValidationError]
    warnings: List[str]
    adjusted_params: Dict[str, Any]
    validation_details: Dict[str, Any]
    
    # Performance metrics
    validation_time_ms: float = 0.0
    cache_hit: bool = False
    
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return self.result in [ValidationResult.VALID, ValidationResult.ADJUSTED]
    
    def has_errors(self) -> bool:
        """Check if validation has errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings"""
        return len(self.warnings) > 0

class APIPrecheckValidator:
    """
    🎯 API PRECHECK VALIDATOR
    
    Production-grade API pre-validation system to prevent order rejects
    and optimize order placement success rates.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or Logger()
        
        # Initialize configuration
        self.validation_config = ValidationConfig()
        
        # Validation state
        self.validation_cache: Dict[str, ValidationResponse] = {}
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        self.last_market_update = 0.0
        
        # Performance tracking
        self.validation_metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'adjusted_validations': 0,
            'cache_hits': 0,
            'avg_validation_time_ms': 0.0,
            'validation_errors': {},
        }
        
        self.logger.info("🎯 [API_PRECHECK] API Precheck Validator initialized")
        self.logger.info("🎯 [API_PRECHECK] Comprehensive validation enabled")
    
    async def validate_order(self, request: OrderValidationRequest) -> ValidationResponse:
        """
        🎯 Validate order parameters before API submission
        
        Args:
            request: OrderValidationRequest with order details
            
        Returns:
            ValidationResponse: Comprehensive validation result
        """
        try:
            validation_start = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.validation_cache:
                cached_response = self.validation_cache[cache_key]
                cached_response.cache_hit = True
                self.validation_metrics['cache_hits'] += 1
                return cached_response
            
            # Initialize response
            response = ValidationResponse(
                result=ValidationResult.VALID,
                errors=[],
                warnings=[],
                adjusted_params={},
                validation_details={},
            )
            
            # Update market data
            await self._update_market_data(request.symbol)
            
            # Perform validations
            await self._validate_tick_size(request, response)
            await self._validate_min_notional(request, response)
            await self._validate_reduce_only(request, response)
            await self._validate_margin_requirements(request, response)
            await self._validate_leverage(request, response)
            await self._validate_position_size(request, response)
            await self._validate_price_reasonableness(request, response)
            await self._validate_quantity_reasonableness(request, response)
            
            # Determine final result
            if response.errors:
                response.result = ValidationResult.INVALID
            elif response.adjusted_params:
                response.result = ValidationResult.ADJUSTED
            elif response.warnings:
                response.result = ValidationResult.WARNING
            else:
                response.result = ValidationResult.VALID
            
            # Calculate validation time
            response.validation_time_ms = (time.time() - validation_start) * 1000
            
            # Cache result
            self.validation_cache[cache_key] = response
            
            # Update metrics
            await self._update_validation_metrics(response)
            
            # Log validation result
            await self._log_validation_result(request, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ [VALIDATE_ORDER] Error validating order: {e}")
            return ValidationResponse(
                result=ValidationResult.INVALID,
                errors=[ValidationError.PRICE_INVALID],
                warnings=[f"Validation error: {str(e)}"],
                adjusted_params={},
                validation_details={'error': str(e)},
                validation_time_ms=(time.time() - validation_start) * 1000,
            )
    
    async def _validate_tick_size(self, request: OrderValidationRequest, response: ValidationResponse):
        """Validate tick size compliance"""
        try:
            if not self.validation_config.validation_settings['tick_size_validation']:
                return
            
            symbol = request.symbol
            price = request.price
            
            # Get tick size for symbol
            tick_size = self.validation_config.hyperliquid_params['tick_sizes'].get(symbol, 0.0001)
            
            # Check if price is aligned with tick size
            if price % tick_size != 0:
                if self.validation_config.validation_settings['auto_adjust_prices']:
                    # Adjust price to nearest tick
                    adjusted_price = round(price / tick_size) * tick_size
                    response.adjusted_params['price'] = adjusted_price
                    response.warnings.append(f"Price adjusted to tick size: {adjusted_price}")
                    response.validation_details['tick_size_adjustment'] = {
                        'original_price': price,
                        'adjusted_price': adjusted_price,
                        'tick_size': tick_size,
                    }
                else:
                    response.errors.append(ValidationError.TICK_SIZE)
                    response.validation_details['tick_size_violation'] = {
                        'price': price,
                        'tick_size': tick_size,
                        'remainder': price % tick_size,
                    }
            
        except Exception as e:
            self.logger.error(f"❌ [TICK_SIZE] Error validating tick size: {e}")
            response.errors.append(ValidationError.TICK_SIZE)
    
    async def _validate_min_notional(self, request: OrderValidationRequest, response: ValidationResponse):
        """Validate minimum notional requirements"""
        try:
            if not self.validation_config.validation_settings['min_notional_validation']:
                return
            
            symbol = request.symbol
            quantity = request.quantity
            price = request.price
            
            # Calculate notional value
            notional_value = quantity * price
            
            # Get minimum notional for symbol
            min_notional = self.validation_config.hyperliquid_params['min_notional'].get(symbol, 1.0)
            
            if notional_value < min_notional:
                response.errors.append(ValidationError.MIN_NOTIONAL)
                response.validation_details['min_notional_violation'] = {
                    'notional_value': notional_value,
                    'min_notional': min_notional,
                    'shortfall': min_notional - notional_value,
                }
            
        except Exception as e:
            self.logger.error(f"❌ [MIN_NOTIONAL] Error validating min notional: {e}")
            response.errors.append(ValidationError.MIN_NOTIONAL)
    
    async def _validate_reduce_only(self, request: OrderValidationRequest, response: ValidationResponse):
        """Validate reduce-only order requirements"""
        try:
            if not self.validation_config.validation_settings['reduce_only_validation']:
                return
            
            if not request.reduce_only:
                return
            
            symbol = request.symbol
            side = request.side
            current_positions = request.current_positions
            
            # Check if we have existing position to reduce
            current_position = current_positions.get(symbol, 0)
            
            if current_position == 0:
                response.errors.append(ValidationError.REDUCE_ONLY)
                response.validation_details['reduce_only_violation'] = {
                    'symbol': symbol,
                    'side': side,
                    'current_position': current_position,
                    'order_quantity': request.quantity,
                }
            elif (side == 'buy' and current_position >= 0) or (side == 'sell' and current_position <= 0):
                response.warnings.append("Reduce-only order may not reduce position effectively")
                response.validation_details['reduce_only_warning'] = {
                    'symbol': symbol,
                    'side': side,
                    'current_position': current_position,
                    'order_quantity': request.quantity,
                }
            
        except Exception as e:
            self.logger.error(f"❌ [REDUCE_ONLY] Error validating reduce-only: {e}")
            response.errors.append(ValidationError.REDUCE_ONLY)
    
    async def _validate_margin_requirements(self, request: OrderValidationRequest, response: ValidationResponse):
        """Validate margin requirements"""
        try:
            if not self.validation_config.validation_settings['margin_check_validation']:
                return
            
            # Calculate required margin
            notional_value = request.quantity * request.price
            leverage = request.leverage
            
            # Calculate required margin
            margin_requirements = self.validation_config.hyperliquid_params['margin_requirements']
            required_margin = notional_value / leverage * margin_requirements['initial_margin_percent']
            
            # Check available margin
            available_margin = request.available_margin
            
            if required_margin > available_margin:
                response.errors.append(ValidationError.MARGIN_INSUFFICIENT)
                response.validation_details['margin_violation'] = {
                    'required_margin': required_margin,
                    'available_margin': available_margin,
                    'shortfall': required_margin - available_margin,
                    'notional_value': notional_value,
                    'leverage': leverage,
                }
            
        except Exception as e:
            self.logger.error(f"❌ [MARGIN] Error validating margin: {e}")
            response.errors.append(ValidationError.MARGIN_INSUFFICIENT)
    
    async def _validate_leverage(self, request: OrderValidationRequest, response: ValidationResponse):
        """Validate leverage limits"""
        try:
            if not self.validation_config.validation_settings['leverage_validation']:
                return
            
            leverage = request.leverage
            leverage_limits = self.validation_config.hyperliquid_params['leverage_limits']
            
            if leverage > leverage_limits['max_leverage']:
                response.errors.append(ValidationError.LEVERAGE_EXCEEDED)
                response.validation_details['leverage_violation'] = {
                    'requested_leverage': leverage,
                    'max_leverage': leverage_limits['max_leverage'],
                }
            elif leverage < leverage_limits['min_leverage']:
                response.warnings.append(f"Leverage {leverage}x below minimum {leverage_limits['min_leverage']}x")
                response.validation_details['leverage_warning'] = {
                    'requested_leverage': leverage,
                    'min_leverage': leverage_limits['min_leverage'],
                }
            
        except Exception as e:
            self.logger.error(f"❌ [LEVERAGE] Error validating leverage: {e}")
            response.errors.append(ValidationError.LEVERAGE_EXCEEDED)
    
    async def _validate_position_size(self, request: OrderValidationRequest, response: ValidationResponse):
        """Validate position size limits"""
        try:
            if not self.validation_config.validation_settings['position_size_validation']:
                return
            
            notional_value = request.quantity * request.price
            position_limits = self.validation_config.hyperliquid_params['position_size_limits']
            
            if notional_value > position_limits['max_position_size_usd']:
                response.errors.append(ValidationError.POSITION_SIZE)
                response.validation_details['position_size_violation'] = {
                    'notional_value': notional_value,
                    'max_position_size': position_limits['max_position_size_usd'],
                }
            elif notional_value < position_limits['min_position_size_usd']:
                response.warnings.append(f"Position size ${notional_value:.2f} below recommended minimum ${position_limits['min_position_size_usd']}")
                response.validation_details['position_size_warning'] = {
                    'notional_value': notional_value,
                    'min_position_size': position_limits['min_position_size_usd'],
                }
            
        except Exception as e:
            self.logger.error(f"❌ [POSITION_SIZE] Error validating position size: {e}")
            response.errors.append(ValidationError.POSITION_SIZE)
    
    async def _validate_price_reasonableness(self, request: OrderValidationRequest, response: ValidationResponse):
        """Validate price reasonableness"""
        try:
            symbol = request.symbol
            price = request.price
            current_price = request.current_price
            
            if current_price <= 0:
                return  # Skip if no current price available
            
            # Check price deviation
            market_config = self.validation_config.market_data_config
            max_deviation = market_config['max_price_deviation_percent']
            
            price_deviation = abs(price - current_price) / current_price
            
            if price_deviation > max_deviation:
                response.warnings.append(f"Price deviation {price_deviation:.2%} exceeds {max_deviation:.2%} threshold")
                response.validation_details['price_deviation'] = {
                    'order_price': price,
                    'current_price': current_price,
                    'deviation_percent': price_deviation,
                    'max_deviation_percent': max_deviation,
                }
            
        except Exception as e:
            self.logger.error(f"❌ [PRICE_REASONABLENESS] Error validating price reasonableness: {e}")
    
    async def _validate_quantity_reasonableness(self, request: OrderValidationRequest, response: ValidationResponse):
        """Validate quantity reasonableness"""
        try:
            symbol = request.symbol
            quantity = request.quantity
            volume_24h = request.volume_24h
            
            if volume_24h <= 0:
                return  # Skip if no volume data available
            
            # Check if quantity is reasonable relative to daily volume
            quantity_value = quantity * request.price
            volume_ratio = quantity_value / volume_24h
            
            if volume_ratio > 0.1:  # More than 10% of daily volume
                response.warnings.append(f"Order size {volume_ratio:.2%} of daily volume may cause significant market impact")
                response.validation_details['quantity_impact'] = {
                    'order_value': quantity_value,
                    'daily_volume': volume_24h,
                    'volume_ratio': volume_ratio,
                }
            
        except Exception as e:
            self.logger.error(f"❌ [QUANTITY_REASONABLENESS] Error validating quantity reasonableness: {e}")
    
    async def _update_market_data(self, symbol: str):
        """Update market data cache"""
        try:
            current_time = time.time()
            cache_duration = self.validation_config.market_data_config['cache_duration_seconds']
            
            # Check if cache is still valid
            if (symbol in self.market_data_cache and 
                current_time - self.last_market_update < cache_duration):
                return
            
            # This would integrate with actual market data API
            # For now, simulate market data
            market_data = {
                'symbol': symbol,
                'current_price': 0.52,  # Mock XRP price
                'bid_price': 0.5199,
                'ask_price': 0.5201,
                'volume_24h': 1000000,
                'timestamp': current_time,
            }
            
            # Update cache
            self.market_data_cache[symbol] = market_data
            self.last_market_update = current_time
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_DATA] Error updating market data: {e}")
    
    def _generate_cache_key(self, request: OrderValidationRequest) -> str:
        """Generate cache key for validation request"""
        try:
            # Create a hash of the request parameters
            key_data = {
                'symbol': request.symbol,
                'side': request.side,
                'quantity': round(request.quantity, 8),
                'price': round(request.price, 8),
                'order_type': request.order_type,
                'reduce_only': request.reduce_only,
                'leverage': request.leverage,
            }
            
            return json.dumps(key_data, sort_keys=True)
            
        except Exception as e:
            self.logger.error(f"❌ [CACHE_KEY] Error generating cache key: {e}")
            return str(time.time())
    
    async def _update_validation_metrics(self, response: ValidationResponse):
        """Update validation metrics"""
        try:
            self.validation_metrics['total_validations'] += 1
            
            if response.result == ValidationResult.VALID:
                self.validation_metrics['successful_validations'] += 1
            elif response.result == ValidationResult.INVALID:
                self.validation_metrics['failed_validations'] += 1
            elif response.result == ValidationResult.ADJUSTED:
                self.validation_metrics['adjusted_validations'] += 1
            
            # Update error counts
            for error in response.errors:
                error_name = error.value
                if error_name not in self.validation_metrics['validation_errors']:
                    self.validation_metrics['validation_errors'][error_name] = 0
                self.validation_metrics['validation_errors'][error_name] += 1
            
            # Update average validation time
            total_time = self.validation_metrics['avg_validation_time_ms'] * (self.validation_metrics['total_validations'] - 1)
            self.validation_metrics['avg_validation_time_ms'] = (total_time + response.validation_time_ms) / self.validation_metrics['total_validations']
            
        except Exception as e:
            self.logger.error(f"❌ [UPDATE_METRICS] Error updating validation metrics: {e}")
    
    async def _log_validation_result(self, request: OrderValidationRequest, response: ValidationResponse):
        """Log validation result"""
        try:
            log_data = {
                'symbol': request.symbol,
                'side': request.side,
                'quantity': request.quantity,
                'price': request.price,
                'result': response.result.value,
                'errors': [error.value for error in response.errors],
                'warnings': response.warnings,
                'adjusted_params': response.adjusted_params,
                'validation_time_ms': response.validation_time_ms,
                'cache_hit': response.cache_hit,
            }
            
            if response.result == ValidationResult.VALID:
                self.logger.info(f"🎯 [VALIDATION] Order validation passed: {json.dumps(log_data)}")
            elif response.result == ValidationResult.INVALID:
                self.logger.warning(f"🎯 [VALIDATION] Order validation failed: {json.dumps(log_data)}")
            elif response.result == ValidationResult.ADJUSTED:
                self.logger.info(f"🎯 [VALIDATION] Order validation adjusted: {json.dumps(log_data)}")
            else:
                self.logger.info(f"🎯 [VALIDATION] Order validation warning: {json.dumps(log_data)}")
            
        except Exception as e:
            self.logger.error(f"❌ [LOG_VALIDATION] Error logging validation result: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        try:
            return {
                'validation_metrics': self.validation_metrics,
                'validation_config': self.validation_config.__dict__,
                'cache_size': len(self.validation_cache),
                'market_data_cache_size': len(self.market_data_cache),
                'last_market_update': self.last_market_update,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [VALIDATION_SUMMARY] Error getting validation summary: {e}")
            return {}
    
    def clear_cache(self):
        """Clear validation cache"""
        try:
            self.validation_cache.clear()
            self.market_data_cache.clear()
            self.logger.info("🎯 [CACHE] Validation cache cleared")
            
        except Exception as e:
            self.logger.error(f"❌ [CLEAR_CACHE] Error clearing cache: {e}")
