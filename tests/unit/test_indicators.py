import pytest
import math

# Import the relevant functions/classes from your bot
from PERFECT_CONSOLIDATED_BOT import XRPTradingBot, BotConfig

@pytest.fixture
def bot():
    return XRPTradingBot(BotConfig())

# --- ATR Calculation Tests ---
def test_atr_simple_close_to_close(bot):
    # Simple increasing prices
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    atr = bot.calculate_atr(prices, period=3)
    # All diffs are 1, so ATR should be 1
    assert math.isclose(atr, 1.0, rel_tol=1e-6)

def test_atr_with_flat_prices(bot):
    prices = [5, 5, 5, 5, 5, 5, 5]
    atr = bot.calculate_atr(prices, period=3)
    assert math.isclose(atr, 0.0, abs_tol=1e-8)

def test_atr_with_spike(bot):
    prices = [10, 10, 10, 50, 10, 10]
    atr = bot.calculate_atr(prices, period=3)
    # Should reflect the spike
    assert atr > 10

# --- RR/ATR Check Tests ---
def test_rr_and_atr_check_good(bot):
    entry = 10
    tp = 12
    sl = 9
    atr = 0.5
    # Should pass RR and ATR checks
    result = bot.rr_and_atr_check(entry, tp, sl, atr)
    assert result is True

def test_rr_and_atr_check_bad_rr(bot):
    entry = 10
    tp = 10.5
    sl = 9.8
    atr = 0.5
    # Should fail RR check
    result = bot.rr_and_atr_check(entry, tp, sl, atr)
    assert result is False

def test_rr_and_atr_check_bad_atr(bot):
    entry = 10
    tp = 10.1
    sl = 9.9
    atr = 0.0001
    # Should fail ATR check (distances too small)
    result = bot.rr_and_atr_check(entry, tp, sl, atr)
    assert result is False

# --- TP/SL Placement Builder (if pure function exists) ---
# If you have a function like bot.calculate_dynamic_tpsl, add tests for it here

def test_calculate_dynamic_tpsl(bot):
    entry = 10
    side = "BUY"
    # Use a simple price history for ATR
    bot.price_history.extend([10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15])
    result = bot.calculate_dynamic_tpsl(entry, side)
    assert isinstance(result, dict) or isinstance(result, tuple)
    # Should return reasonable TP/SL values
    # (You can add more specific checks based on your implementation) 