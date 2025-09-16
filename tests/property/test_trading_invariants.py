"""
🎯 PROPERTY TESTS FOR TRADING INVARIANTS
========================================
Property-based tests to ensure trading system invariants are never violated.

Invariants:
- Never exceed margin %
- Never cross long+short unless hedged
- Notional >= min
- Price % tick == 0
- Position size limits
- Risk limits
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, example
from typing import Dict, Any, List, Tuple
import time
import random

# Add src to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.validation.api_precheck_validator import APIPrecheckValidator, OrderValidationRequest
from src.core.risk.production_risk_manager import ProductionRiskManager
from src.core.execution.maker_first_router import MakerFirstRouter, OrderRequest

class TestTradingInvariants:
    """Property tests for trading system invariants"""
    
    @pytest.fixture
    def validator(self):
        """Create API precheck validator"""
        return APIPrecheckValidator({})
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager"""
        return ProductionRiskManager({})
    
    @pytest.fixture
    def router(self):
        """Create maker-first router"""
        return MakerFirstRouter({})
    
    @given(
        symbol=st.sampled_from(['XRP', 'BTC', 'ETH', 'SOL', 'ARB']),
        price=st.floats(min_value=0.001, max_value=100000.0),
        quantity=st.floats(min_value=0.001, max_value=10000.0),
        leverage=st.floats(min_value=1.0, max_value=50.0),
        account_equity=st.floats(min_value=1000.0, max_value=1000000.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_margin_never_exceeded(self, validator, symbol, price, quantity, leverage, account_equity):
        """Property: Never exceed margin percentage"""
        
        # Create order request
        request = OrderValidationRequest(
            symbol=symbol,
            side='buy',
            quantity=quantity,
            price=price,
            leverage=leverage,
            account_equity=account_equity,
            available_margin=account_equity * 0.8,  # 80% available margin
            current_positions={}
        )
        
        # Validate order
        response = validator.validate_order(request)
        
        if response.is_valid():
            # Calculate required margin
            notional_value = quantity * price
            required_margin = notional_value / leverage * 0.1  # 10% initial margin
            
            # Property: Required margin should not exceed available margin
            assert required_margin <= request.available_margin, \
                f"Margin exceeded: required={required_margin}, available={request.available_margin}"
    
    @given(
        symbol=st.sampled_from(['XRP', 'BTC', 'ETH']),
        long_quantity=st.floats(min_value=0.0, max_value=1000.0),
        short_quantity=st.floats(min_value=0.0, max_value=1000.0),
        price=st.floats(min_value=0.1, max_value=100.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_long_short_limits(self, validator, symbol, long_quantity, short_quantity, price):
        """Property: Never cross long+short unless hedged"""
        
        # Create positions
        current_positions = {symbol: long_quantity - short_quantity}
        
        # Create order request
        request = OrderValidationRequest(
            symbol=symbol,
            side='buy' if short_quantity > 0 else 'sell',
            quantity=abs(long_quantity - short_quantity) + 0.1,  # Slightly more than net position
            price=price,
            reduce_only=True,
            current_positions=current_positions
        )
        
        # Validate order
        response = validator.validate_order(request)
        
        if response.is_valid():
            # Property: Reduce-only orders should not increase position beyond current
            new_position = current_positions[symbol] + (request.quantity if request.side == 'buy' else -request.quantity)
            
            # For reduce-only orders, new position should be closer to zero
            if request.side == 'buy' and current_positions[symbol] < 0:
                assert new_position <= current_positions[symbol], "Reduce-only buy should not increase short position"
            elif request.side == 'sell' and current_positions[symbol] > 0:
                assert new_position >= current_positions[symbol], "Reduce-only sell should not increase long position"
    
    @given(
        symbol=st.sampled_from(['XRP', 'BTC', 'ETH', 'SOL', 'ARB']),
        price=st.floats(min_value=0.001, max_value=100000.0),
        quantity=st.floats(min_value=0.001, max_value=10000.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_notional_always_above_minimum(self, validator, symbol, price, quantity):
        """Property: Notional value always above minimum"""
        
        # Create order request
        request = OrderValidationRequest(
            symbol=symbol,
            side='buy',
            quantity=quantity,
            price=price,
            account_equity=10000.0,
            available_margin=8000.0,
            current_positions={}
        )
        
        # Validate order
        response = validator.validate_order(request)
        
        if response.is_valid():
            # Calculate notional value
            notional_value = quantity * price
            
            # Get minimum notional for symbol
            min_notional = validator.validation_config.hyperliquid_params['min_notional'].get(symbol, 1.0)
            
            # Property: Notional value should be at least minimum
            assert notional_value >= min_notional, \
                f"Notional below minimum: {notional_value} < {min_notional}"
    
    @given(
        symbol=st.sampled_from(['XRP', 'BTC', 'ETH', 'SOL', 'ARB']),
        price=st.floats(min_value=0.001, max_value=100000.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_price_always_tick_aligned(self, validator, symbol, price):
        """Property: Price always aligned to tick size"""
        
        # Create order request
        request = OrderValidationRequest(
            symbol=symbol,
            side='buy',
            quantity=1.0,
            price=price,
            account_equity=10000.0,
            available_margin=8000.0,
            current_positions={}
        )
        
        # Validate order
        response = validator.validate_order(request)
        
        if response.is_valid():
            # Get tick size for symbol
            tick_size = validator.validation_config.hyperliquid_params['tick_sizes'].get(symbol, 0.0001)
            
            # Property: Price should be aligned to tick size
            assert price % tick_size == 0, \
                f"Price not tick-aligned: {price} % {tick_size} = {price % tick_size}"
    
    @given(
        symbol=st.sampled_from(['XRP', 'BTC', 'ETH']),
        quantity=st.floats(min_value=0.001, max_value=10000.0),
        price=st.floats(min_value=0.1, max_value=100.0),
        leverage=st.floats(min_value=1.0, max_value=50.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_position_size_limits(self, validator, symbol, quantity, price, leverage):
        """Property: Position size within limits"""
        
        # Create order request
        request = OrderValidationRequest(
            symbol=symbol,
            side='buy',
            quantity=quantity,
            price=price,
            leverage=leverage,
            account_equity=10000.0,
            available_margin=8000.0,
            current_positions={}
        )
        
        # Validate order
        response = validator.validate_order(request)
        
        if response.is_valid():
            # Calculate position value
            position_value = quantity * price
            
            # Get position limits
            position_limits = validator.validation_config.hyperliquid_params['position_size_limits']
            
            # Property: Position size should be within limits
            assert position_value >= position_limits['min_position_size_usd'], \
                f"Position too small: {position_value} < {position_limits['min_position_size_usd']}"
            
            assert position_value <= position_limits['max_position_size_usd'], \
                f"Position too large: {position_value} > {position_limits['max_position_size_usd']}"
    
    @given(
        leverage=st.floats(min_value=1.0, max_value=100.0)
    )
    @settings(max_examples=50, deadline=1000)
    def test_leverage_limits(self, validator, leverage):
        """Property: Leverage within limits"""
        
        # Create order request
        request = OrderValidationRequest(
            symbol='XRP',
            side='buy',
            quantity=1.0,
            price=0.5,
            leverage=leverage,
            account_equity=10000.0,
            available_margin=8000.0,
            current_positions={}
        )
        
        # Validate order
        response = validator.validate_order(request)
        
        if response.is_valid():
            # Get leverage limits
            leverage_limits = validator.validation_config.hyperliquid_params['leverage_limits']
            
            # Property: Leverage should be within limits
            assert leverage >= leverage_limits['min_leverage'], \
                f"Leverage too low: {leverage} < {leverage_limits['min_leverage']}"
            
            assert leverage <= leverage_limits['max_leverage'], \
                f"Leverage too high: {leverage} > {leverage_limits['max_leverage']}"
    
    @given(
        account_equity=st.floats(min_value=1000.0, max_value=1000000.0),
        position_value=st.floats(min_value=100.0, max_value=50000.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_equity_at_risk_limits(self, risk_manager, account_equity, position_value):
        """Property: Equity at risk within limits"""
        
        # Calculate position size using risk manager
        position_size = risk_manager._calculate_equity_at_risk_size(account_equity)
        
        # Property: Position size should not exceed equity at risk limit
        equity_at_risk_percent = 0.05  # 5% equity at risk
        max_equity_at_risk = account_equity * equity_at_risk_percent
        
        assert position_size <= max_equity_at_risk, \
            f"Position size exceeds equity at risk: {position_size} > {max_equity_at_risk}"
    
    @given(
        volatility=st.floats(min_value=0.01, max_value=0.5),
        account_equity=st.floats(min_value=1000.0, max_value=1000000.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_volatility_targeting(self, risk_manager, volatility, account_equity):
        """Property: Volatility targeting within bounds"""
        
        # Calculate position size using volatility targeting
        position_size = risk_manager._calculate_volatility_position_size('XRP', volatility, account_equity)
        
        # Property: Position size should be reasonable relative to volatility
        vol_target = 0.15  # 15% annual volatility target
        expected_size = (vol_target / volatility) * account_equity * 0.1  # 10% max position
        
        # Allow for some variance due to scaling factors
        assert position_size <= expected_size * 2, \
            f"Position size too large for volatility: {position_size} > {expected_size * 2}"
    
    @given(
        atr=st.floats(min_value=0.001, max_value=0.1),
        account_equity=st.floats(min_value=1000.0, max_value=1000000.0),
        price=st.floats(min_value=0.1, max_value=100.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_atr_position_sizing(self, risk_manager, atr, account_equity, price):
        """Property: ATR-based position sizing"""
        
        # Calculate position size using ATR
        position_size = risk_manager._calculate_atr_position_size('XRP', atr, price, account_equity)
        
        # Property: Position size should be inversely related to ATR
        if atr > 0:
            # Higher ATR should result in smaller position size
            risk_per_trade = account_equity * 0.05  # 5% equity at risk
            expected_size = risk_per_trade / (atr * 2.0)  # 2x ATR multiplier
            
            # Allow for some variance
            assert position_size <= expected_size * 1.5, \
                f"Position size too large for ATR: {position_size} > {expected_size * 1.5}"
    
    @given(
        signal_strength=st.floats(min_value=0.0, max_value=1.0),
        account_equity=st.floats(min_value=1000.0, max_value=1000000.0)
    )
    @settings(max_examples=100, deadline=1000)
    def test_signal_strength_scaling(self, risk_manager, signal_strength, account_equity):
        """Property: Signal strength scaling"""
        
        # Calculate position size with different signal strengths
        position_size_1 = risk_manager.calculate_position_size('XRP', signal_strength, 0.5, account_equity)
        position_size_2 = risk_manager.calculate_position_size('XRP', 1.0, 0.5, account_equity)
        
        # Property: Higher signal strength should result in larger position size
        if signal_strength > 0:
            assert position_size_1['recommended_size'] <= position_size_2['recommended_size'], \
                f"Signal strength scaling failed: {position_size_1['recommended_size']} > {position_size_2['recommended_size']}"
    
    @given(
        order_requests=st.lists(
            st.builds(
                OrderRequest,
                symbol=st.sampled_from(['XRP', 'BTC', 'ETH']),
                side=st.sampled_from(['buy', 'sell']),
                quantity=st.floats(min_value=0.001, max_value=1000.0),
                price=st.floats(min_value=0.1, max_value=100.0),
                urgency=st.floats(min_value=0.0, max_value=1.0)
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=50, deadline=2000)
    def test_order_routing_consistency(self, router, order_requests):
        """Property: Order routing consistency"""
        
        for request in order_requests:
            # Route order
            result = router.route_order(request)
            
            # Property: Order should be routed successfully
            assert result is not None, "Order routing failed"
            
            # Property: Result should have required fields
            assert hasattr(result, 'order_id'), "Result missing order_id"
            assert hasattr(result, 'symbol'), "Result missing symbol"
            assert hasattr(result, 'side'), "Result missing side"
            assert hasattr(result, 'quantity'), "Result missing quantity"
            assert hasattr(result, 'price'), "Result missing price"
            
            # Property: Result should match request
            assert result.symbol == request.symbol, "Symbol mismatch"
            assert result.side == request.side, "Side mismatch"
            assert result.quantity == request.quantity, "Quantity mismatch"
    
    @given(
        risk_scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100)
    )
    @settings(max_examples=50, deadline=1000)
    def test_risk_score_consistency(self, risk_manager, risk_scores):
        """Property: Risk score consistency"""
        
        for risk_score in risk_scores:
            # Get risk level
            risk_level = risk_manager._get_risk_level(risk_score)
            
            # Property: Risk level should be consistent with risk score
            if risk_score >= 0.9:
                assert risk_level.value in ['kill_switch', 'critical'], \
                    f"High risk score {risk_score} should map to critical level"
            elif risk_score >= 0.6:
                assert risk_level.value in ['high', 'critical'], \
                    f"Medium-high risk score {risk_score} should map to high level"
            elif risk_score >= 0.4:
                assert risk_level.value in ['medium', 'high'], \
                    f"Medium risk score {risk_score} should map to medium level"
            else:
                assert risk_level.value in ['low', 'medium'], \
                    f"Low risk score {risk_score} should map to low level"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
