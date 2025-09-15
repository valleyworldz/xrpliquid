import pytest

from newbotcode import XRPTradingBot


def test_simulate_fill_prob_bounds(monkeypatch):
	bot = XRPTradingBot()
	# mock resilient_info.l2_snapshot to supply a simple book
	class _Info:
		def l2_snapshot(self, sym):
			return {"bids": [(0.9990, 1000)], "asks": [(1.0010, 1000)]}
	bot.resilient_info = _Info()
	prob_buy = bot.simulate_limit_fill_probability(1.0005, True)
	prob_sell = bot.simulate_limit_fill_probability(1.0005, False)
	assert 0.01 <= prob_buy <= 0.95
	assert 0.01 <= prob_sell <= 0.95


def test_var_es_nonnegative():
	bot = XRPTradingBot()
	prices = [1.0 + 0.001 * i for i in range(200)]
	var99, es99 = bot.compute_var_es(prices, alpha=0.99)
	assert var99 >= 0.0
	assert es99 >= 0.0

