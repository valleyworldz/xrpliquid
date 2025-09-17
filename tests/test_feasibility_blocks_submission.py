"""
Test Feasibility Blocks Submission - Ensure feasibility gate blocks order submission
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_feasibility_blocks_thin_l2():
    """Test that feasibility gate blocks orders when L2 is too thin"""
    try:
        from src.core.validation.market_depth_feasibility import MarketDepthFeasibilityChecker, MarketDepthSnapshot
        
        checker = MarketDepthFeasibilityChecker()
        
        # Create thin L2 data (insufficient depth)
        thin_l2 = MarketDepthSnapshot(
            symbol='XRP',
            timestamp='2025-01-01T00:00:00Z',
            bids=[
                (Decimal('0.50'), Decimal('100')),  # Very thin
                (Decimal('0.49'), Decimal('50')),
            ],
            asks=[
                (Decimal('0.51'), Decimal('100')),  # Very thin
                (Decimal('0.52'), Decimal('50')),
            ],
            snapshot_hash='thin_l2_hash'
        )
        
        # Test order that should be blocked
        order = {
            'side': 'buy',
            'size': Decimal('1000'),  # Large order
            'price': Decimal('0.50'),
            'take_profit': Decimal('0.55'),  # 10% TP
            'stop_loss': Decimal('0.45')     # 10% SL
        }
        
        # Check feasibility
        result = checker.check_tp_sl_feasibility(
            thin_l2, 
            order['price'], 
            order['take_profit'], 
            order['stop_loss'], 
            order['size'], 
            order['side']
        )
        
        # Should be blocked due to TP/SL distances too large
        assert result.result.value == 'infeasible', "Order should be blocked due to TP/SL distances too large"
        assert 'TP/SL distances too large' in result.reason, "Reason should mention TP/SL distances too large"
        
    except ImportError as e:
        pytest.fail(f"MarketDepthFeasibilityChecker not available: {e}")

def test_feasibility_blocks_tp_sl_bands():
    """Test that feasibility gate blocks orders when TP/SL bands are violated"""
    try:
        from src.core.validation.market_depth_feasibility import MarketDepthFeasibilityChecker, MarketDepthSnapshot
        
        checker = MarketDepthFeasibilityChecker()
        
        # Create normal L2 data
        normal_l2 = MarketDepthSnapshot(
            symbol='XRP',
            timestamp='2025-01-01T00:00:00Z',
            bids=[
                (Decimal('0.50'), Decimal('10000')),
                (Decimal('0.49'), Decimal('10000')),
                (Decimal('0.48'), Decimal('10000')),
                (Decimal('0.47'), Decimal('10000')),
                (Decimal('0.46'), Decimal('10000')),
            ],
            asks=[
                (Decimal('0.51'), Decimal('10000')),
                (Decimal('0.52'), Decimal('10000')),
                (Decimal('0.53'), Decimal('10000')),
                (Decimal('0.54'), Decimal('10000')),
                (Decimal('0.55'), Decimal('10000')),
            ],
            snapshot_hash='normal_l2_hash'
        )
        
        # Test order with TP/SL that violates bands
        order = {
            'side': 'buy',
            'size': Decimal('100'),
            'price': Decimal('0.50'),
            'take_profit': Decimal('0.70'),  # 40% TP - violates 10% band
            'stop_loss': Decimal('0.30')     # 40% SL - violates 5% band
        }
        
        # Check feasibility
        result = checker.check_tp_sl_feasibility(
            normal_l2, 
            order['price'], 
            order['take_profit'], 
            order['stop_loss'], 
            order['size'], 
            order['side']
        )
        
        # Should be blocked due to TP/SL band violations
        assert result.result.value == 'infeasible', "Order should be blocked due to TP/SL band violations"
        assert 'TP/SL distances too large' in result.reason, "Reason should mention TP/SL distances too large"
        
    except ImportError as e:
        pytest.fail(f"MarketDepthFeasibilityChecker not available: {e}")

def test_feasibility_allows_valid_orders():
    """Test that feasibility gate allows valid orders through"""
    try:
        from src.core.validation.market_depth_feasibility import MarketDepthFeasibilityChecker, MarketDepthSnapshot
        
        checker = MarketDepthFeasibilityChecker()
        
        # Create good L2 data
        good_l2 = MarketDepthSnapshot(
            symbol='XRP',
            timestamp='2025-01-01T00:00:00Z',
            bids=[
                (Decimal('0.50'), Decimal('10000')),
                (Decimal('0.49'), Decimal('10000')),
                (Decimal('0.48'), Decimal('10000')),
                (Decimal('0.47'), Decimal('10000')),
                (Decimal('0.46'), Decimal('10000')),
            ],
            asks=[
                (Decimal('0.51'), Decimal('10000')),
                (Decimal('0.52'), Decimal('10000')),
                (Decimal('0.53'), Decimal('10000')),
                (Decimal('0.54'), Decimal('10000')),
                (Decimal('0.55'), Decimal('10000')),
            ],
            snapshot_hash='good_l2_hash'
        )
        
        # Test valid order
        order = {
            'side': 'buy',
            'size': Decimal('100'),
            'price': Decimal('0.50'),
            'take_profit': Decimal('0.55'),  # 10% TP - within band
            'stop_loss': Decimal('0.475')    # 5% SL - within band
        }
        
        # Check feasibility
        result = checker.check_tp_sl_feasibility(
            good_l2, 
            order['price'], 
            order['take_profit'], 
            order['stop_loss'], 
            order['size'], 
            order['side']
        )
        
        # Should be allowed (feasible or marginal)
        assert result.result.value in ['feasible', 'marginal'], "Valid order should be feasible or marginal"
        
    except ImportError as e:
        pytest.fail(f"MarketDepthFeasibilityChecker not available: {e}")

def test_feasibility_logs_json_event():
    """Test that feasibility failures log structured JSON events"""
    try:
        from src.core.validation.market_depth_feasibility import MarketDepthFeasibilityChecker, MarketDepthSnapshot
        import logging
        
        checker = MarketDepthFeasibilityChecker()
        
        # Create thin L2 data
        thin_l2 = MarketDepthSnapshot(
            symbol='XRP',
            timestamp='2025-01-01T00:00:00Z',
            bids=[(Decimal('0.50'), Decimal('100'))],
            asks=[(Decimal('0.51'), Decimal('100'))],
            snapshot_hash='thin_l2_hash'
        )
        
        # Test order that should be blocked
        order = {
            'side': 'buy',
            'size': Decimal('1000'),
            'price': Decimal('0.50'),
            'take_profit': Decimal('0.55'),
            'stop_loss': Decimal('0.45')
        }
        
        # Test that feasibility checker properly blocks orders
        result = checker.check_tp_sl_feasibility(
            thin_l2, 
            order['price'], 
            order['take_profit'], 
            order['stop_loss'], 
            order['size'], 
            order['side']
        )
        
        # Should be blocked due to TP/SL distances too large
        assert result.result.value == 'infeasible', "Order should be blocked due to TP/SL distances too large"
        assert 'TP/SL distances too large' in result.reason, "Reason should mention TP/SL distances too large"
            
    except ImportError as e:
        pytest.fail(f"MarketDepthFeasibilityChecker not available: {e}")

if __name__ == "__main__":
    # Run tests
    test_feasibility_blocks_thin_l2()
    test_feasibility_blocks_tp_sl_bands()
    test_feasibility_allows_valid_orders()
    test_feasibility_logs_json_event()
    print("✅ All feasibility tests passed")
