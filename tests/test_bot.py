import asyncio
import math
import pytest

# Import the bot; if environment lacks optional deps, skip tests gracefully
try:
	from newbotcode import XRPTradingBot
except Exception as e:  # pragma: no cover
	pytest.skip(f"newbotcode import failed: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def bot():
	# Instantiate without running any network setup
	return XRPTradingBot()


def test_band_tp_sl_clamps(bot):
	entry = 3.0
	atr = 0.01  # 1 cent
	# Place TP/SL far beyond 6x ATR band (0.06)
	tp_far = 3.20
	sl_far = 2.70
	tp_c, sl_c = bot._band_tp_sl(entry, atr, tp_far, sl_far, is_long=True)
	max_dist = 6.0 * atr
	assert tp_c <= entry + max_dist + 1e-12
	assert sl_c >= entry - max_dist - 1e-12


def test_update_realized_pnl_aggregate(bot):
	bot.realized_pnl_total = 0.0
	bot.update_realized_pnl_aggregate(5.0, funding_since_open=1.0)
	assert math.isclose(bot.realized_pnl_total, 4.0, rel_tol=1e-9)


@pytest.mark.asyncio
async def test_place_market_order_none_guard(bot, monkeypatch):
	# Force current price fetch to return None → market order returns None (no crash)
	async def fake_price(symbol: str):
		return None
	monkeypatch.setattr(bot, "get_current_price_async", fake_price, raising=True)
	res = await bot.place_market_order("XRP", "BUY", 1.0, reduce_only=True)
	assert res is None
