#!/usr/bin/env python3
"""
Test New Modular Components
==========================
Test the newly created modular components.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xrpbot.core.config import BotConfig, RuntimeState, RiskDecision, DEFAULT_CONFIG
from xrpbot.core.utils import (
    align_price_to_tick,
    calculate_atr,
    calculate_rsi,
    calculate_macd,
    normalize_l2_snapshot,
    calculate_spread,
    calculate_mid_price
)

class TestConfigModule:
    """Test configuration module"""
    
    def test_bot_config_creation(self):
        """Test BotConfig creation"""
        config = BotConfig()
        assert config.risk_per_trade == 0.02
        assert config.confidence_threshold == 0.02
        assert config.max_consecutive_losses == 3
    
    def test_bot_config_validation(self):
        """Test BotConfig validation"""
        config = BotConfig()
        errors = config.validate()
        assert len(errors) == 0
        
        # Test invalid config
        bad_config = BotConfig(risk_per_trade=1.5)  # Invalid: > 1
        errors = bad_config.validate()
        assert len(errors) > 0
        assert "risk_per_trade must be between 0 and 1" in errors[0]
    
    def test_runtime_state(self):
        """Test RuntimeState functionality"""
        state = RuntimeState()
        assert state.total_trades == 0
        assert state.win_rate == 0.0
        
        # Test trade result update
        state.update_trade_result(100.0)  # Win
        assert state.total_trades == 1
        assert state.total_wins == 1
        assert state.consecutive_losses == 0
        assert state.win_rate == 1.0
        
        state.update_trade_result(-50.0)  # Loss
        assert state.total_trades == 2
        assert state.total_wins == 1
        assert state.total_losses == 1
        assert state.consecutive_losses == 1
        assert state.win_rate == 0.5

class TestUtilsModule:
    """Test utilities module"""
    
    def test_align_price_to_tick(self):
        """Test price alignment to tick size"""
        # Test neutral alignment
        result = align_price_to_tick(100.1234, 0.01, "neutral")
        assert abs(result - 100.12) < 0.001
        
        # Test buy alignment (round up)
        result = align_price_to_tick(100.1234, 0.01, "buy")
        assert abs(result - 100.13) < 0.001
        
        # Test sell alignment (round down)
        result = align_price_to_tick(100.1234, 0.01, "sell")
        assert abs(result - 100.12) < 0.001
    
    def test_calculate_atr(self):
        """Test ATR calculation"""
        prices = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0]
        atr = calculate_atr(prices, period=3)
        assert atr > 0
        assert isinstance(atr, float)
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        prices = [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0]
        rsi = calculate_rsi(prices, period=3)
        assert 0 <= rsi <= 100
        assert isinstance(rsi, float)
    
    def test_calculate_macd(self):
        """Test MACD calculation"""
        prices = [100.0] * 30  # Need enough data for MACD
        macd_line, signal_line, histogram = calculate_macd(prices)
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)
    
    def test_calculate_spread(self):
        """Test spread calculation"""
        spread = calculate_spread(100.0, 100.5)
        assert spread == 0.005  # 0.5% spread
        assert isinstance(spread, float)
    
    def test_calculate_mid_price(self):
        """Test mid price calculation"""
        mid = calculate_mid_price(100.0, 100.5)
        assert mid == 100.25
        assert isinstance(mid, float)
    
    def test_normalize_l2_snapshot(self):
        """Test L2 snapshot normalization"""
        raw_data = {
            "levels": [
                {
                    "bids": [["100.0", "10.0"], ["99.9", "5.0"]],
                    "asks": [["100.1", "8.0"], ["100.2", "12.0"]]
                }
            ]
        }
        normalized = normalize_l2_snapshot(raw_data)
        assert "bids" in normalized
        assert "asks" in normalized
        assert len(normalized["bids"]) == 2
        assert len(normalized["asks"]) == 2
        assert normalized["bids"][0][0] == 100.0  # Price
        assert normalized["bids"][0][1] == 10.0   # Size

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 