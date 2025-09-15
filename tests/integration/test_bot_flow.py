import asyncio
import types

from newbotcode import XRPTradingBot


async def _dummy_guardian(*args, **kwargs):
	return True


def test_end_to_end_flow(monkeypatch):
	bot = XRPTradingBot()
	# minimal stubs
	bot.logger.disabled = True
	bot.price_history = [0.5 + 0.0001 * i for i in range(400)]
	class _Lock:
		def __enter__(self): return None
		def __exit__(self, a,b,c): return False
	bot._price_history_lock = _Lock()
	monkeypatch.setattr(bot, 'get_current_price', lambda symbol="XRP": 0.5)
	monkeypatch.setattr(bot, 'get_account_status', lambda: {"freeCollateral": 100, "account_value": 100})
	monkeypatch.setattr(bot, 'get_recent_price_data', lambda: [0.5 for _ in range(50)])
	monkeypatch.setattr(bot, 'place_native_tpsl_pair', lambda **kw: {"tp_oid": 1, "sl_oid": 2})
	monkeypatch.setattr(bot, 'execute_synthetic_exit', lambda *a, **kw: {"success": True})
	monkeypatch.setattr(bot, 'activate_offchain_guardian', lambda *a, **kw: True)
	# mock L2 for fill prob
	class _Info:
		def l2_snapshot(self, sym):
			return {"bids": [(0.4999, 1000)], "asks": [(0.5001, 1000)]}
	bot.resilient_info = _Info()
	# run position sizing and tpsl preview
	size = bot.calculate_position_size_with_risk(0.6)
	assert size > 0
	entry = 0.5
	tp, sl, atr = bot.calculate_dynamic_tpsl(entry, "BUY")
	assert tp is not None and sl is not None and atr is not None
	# simulate mirroring
	asyncio.get_event_loop().run_until_complete(bot._mirror_tp_limits(tp, None, size, True))

