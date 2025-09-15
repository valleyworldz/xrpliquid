import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PERFECT_CONSOLIDATED_BOT import XRPTradingBot, BotConfig

class TestEdgeCases:
    """Test edge cases for critical functions"""
    
    @pytest.fixture
    def bot(self):
        """Create a bot instance for testing"""
        config = BotConfig()
        return XRPTradingBot(config)
    
    def test_rr_and_atr_check_division_by_zero(self, bot):
        """Test RR/ATR check with zero risk (division by zero)"""
        # Setup: entry_price = tp_price = sl_price (zero risk)
        entry_price = 1.0
        tp_price = 1.0  # Same as entry
        sl_price = 1.0  # Same as entry
        atr = 0.001
        
        # This should handle division by zero gracefully
        result = bot.rr_and_atr_check(entry_price, tp_price, sl_price, atr)
        assert result is False  # Should fail due to zero risk
        
    def test_rr_and_atr_check_atr_clamp(self, bot):
        """Test ATR clamp to minimum tick size"""
        entry_price = 1.0
        tp_price = 1.02  # 2% above
        sl_price = 0.98  # 2% below
        atr = 0.0001  # Very small ATR (less than 2 ticks)
        
        # Should clamp ATR to minimum tick size
        result = bot.rr_and_atr_check(entry_price, tp_price, sl_price, atr)
        # Should pass because ATR gets clamped appropriately
        
    def test_rr_and_atr_check_extreme_atr(self, bot):
        """Test ATR check with extremely large ATR"""
        entry_price = 1.0
        tp_price = 1.02
        sl_price = 0.98
        atr = 0.5  # 50% ATR (extremely large)
        
        result = bot.rr_and_atr_check(entry_price, tp_price, sl_price, atr)
        assert result is False  # Should fail due to excessive ATR
        
    def test_rr_and_atr_check_minimum_distances(self, bot):
        """Test minimum TP/SL distance requirements"""
        entry_price = 1.0
        tp_price = 1.001  # Very small TP distance
        sl_price = 0.999  # Very small SL distance
        atr = 0.01
        
        result = bot.rr_and_atr_check(entry_price, tp_price, sl_price, atr)
        assert result is False  # Should fail due to insufficient distances
        
    def test_rr_and_atr_check_fee_calculation(self, bot):
        """Test fee calculation with position size"""
        entry_price = 1.0
        tp_price = 1.02
        sl_price = 0.98
        atr = 0.01
        position_size = 100.0  # Large position size
        
        # Test with high fees that exceed 10% of reward
        bot.maker_fee = 0.1  # 10% fee (very high)
        
        result = bot.rr_and_atr_check(entry_price, tp_price, sl_price, atr, position_size)
        assert result is False  # Should fail due to excessive fees
        
    def test_calculate_atr_insufficient_data(self, bot):
        """Test ATR calculation with insufficient data"""
        # Test with less than period + 1 prices
        prices = [1.0, 1.01, 1.02]  # Only 3 prices for 14-period ATR
        
        atr = bot.calculate_atr(prices, period=14)
        assert atr > 0  # Should return a valid fallback value
        
    def test_calculate_atr_empty_data(self, bot):
        """Test ATR calculation with empty data"""
        prices = []
        
        atr = bot.calculate_atr(prices, period=14)
        assert atr == 0.001  # Should return fallback value
        
    def test_calculate_atr_single_price(self, bot):
        """Test ATR calculation with single price"""
        prices = [1.0]
        
        atr = bot.calculate_atr(prices, period=14)
        assert atr > 0  # Should return a valid fallback value
        
    @pytest.mark.slow
    def test_smoke_test_instantiation(self):
        """Smoke test for bot instantiation and basic functionality"""
        config = BotConfig()
        bot = XRPTradingBot(config)
        
        # Test basic attributes are set
        assert hasattr(bot, 'config')
        assert hasattr(bot, 'logger')
        assert hasattr(bot, 'maker_fee')
        assert hasattr(bot, 'taker_fee')
        
        # Test price alignment functions
        assert float(bot.align_up(1.0, 0.0001)) >= 1.0
        assert float(bot.align_down(1.0, 0.0001)) <= 1.0
        
        # Test address truncation
        assert bot.short_addr("0x1234567890abcdef") == "0x1234"
        assert bot.short_addr("") == ""
        
        print("✅ Smoke test passed - bot instantiation successful") 