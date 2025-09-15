from hypothesis import given, strategies as st

from newbotcode import XRPTradingBot


@given(price=st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False))
def test_align_price_fuzz(price):
	bot = XRPTradingBot()
	tick = 0.0001
	p_up = bot._align_price_to_tick(price, tick, "up")
	p_dn = bot._align_price_to_tick(price, tick, "down")
	# Both should be multiples of tick to 4 decimals
	assert abs((float(p_up) * 10000) - round(float(p_up) * 10000)) < 1e-6
	assert abs((float(p_dn) * 10000) - round(float(p_dn) * 10000)) < 1e-6


@given(order_size=st.integers(min_value=1, max_value=10000))
def test_fill_prob_fuzz(order_size):
	bot = XRPTradingBot()
	# Stub L2 snapshot
	class _Info:
		def l2_snapshot(self, sym):
			return {"bids": [(0.999, 1000)], "asks": [(1.001, 1000)]}
	bot.resilient_info = _Info()
	prob = bot.simulate_limit_fill_probability(1.0005, True, order_size=order_size)
	assert 0.01 <= prob <= 0.95

