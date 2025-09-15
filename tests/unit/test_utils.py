import math
import pytest

from newbotcode import XRPTradingBot


class DummyConfig:
	atr_scaled_position_enabled = False
	drawdown_size_throttle_enabled = False
	target_atr = 0.01
	atr_scaling_factor = 1.0
	atr_granular_scaling = False
	atr_lookback_period = 14


def test_align_price_rounding():
	bot = XRPTradingBot()
	tick = 0.0001
	# Upwards alignment
	assert str(bot._align_price_to_tick(0.51231, tick, "up")) == "0.5124"
	# Downwards alignment
	assert str(bot._align_price_to_tick(0.51239, tick, "down")) == "0.5123"


def test_es_limiter_caps_size(monkeypatch):
	bot = XRPTradingBot()
	bot.config = DummyConfig()
	bot.price_history = [0.5 + i * 0.0001 for i in range(300)]
	# simple context manager stub for lock usage
	class _NL:
		def __enter__(self):
			return None
		def __exit__(self, exc_type, exc, tb):
			return False
	bot._price_history_lock = _NL()
	# mocks
	monkeypatch.setattr(bot, 'get_current_price', lambda: 0.5)
	monkeypatch.setattr(bot, 'get_account_status', lambda: {"freeCollateral": 100, "account_value": 100})
	monkeypatch.setattr(bot, 'compute_var_es', lambda prices, alpha=0.99: (0.05, 0.05))
	monkeypatch.setattr(bot, 'get_recent_price_data', lambda: [0.5 for _ in range(20)])
	monkeypatch.setattr(bot, 'calculate_position_size', lambda px, eq, conf: 1000)
	size = bot.calculate_position_size_with_risk(0.8)
	assert size > 0
	assert size <= 1000

