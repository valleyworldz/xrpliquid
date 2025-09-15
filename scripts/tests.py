import pytest

from newbotcode import XRPTradingBot


def test_atr_minimum_when_short_series():
    bot = XRPTradingBot()
    prices = [1.0, 1.001]  # fewer than period+1
    atr = bot.calculate_atr(prices, period=14)
    assert atr > 0


def test_atr_vector_like():
    bot = XRPTradingBot()
    prices = [1.0, 1.01, 1.02, 1.015, 1.03, 1.025, 1.05, 1.045, 1.06, 1.07, 1.06, 1.065, 1.07, 1.08, 1.09]
    atr = bot.calculate_atr(prices, period=14)
    assert atr > 0


