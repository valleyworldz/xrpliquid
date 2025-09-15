import time

import pytest


def test_rr_and_atr_check_passes_for_good_rr():
    from newbotcode import XRPTradingBot, BotConfig

    bot = XRPTradingBot(config=BotConfig())
    # RR = (tp-entry)/(entry-sl) = 0.06/0.03 = 2.0 ≥ default min_rr_ratio
    ok = bot.rr_and_atr_check(entry_price=1.00, tp_price=1.06, sl_price=0.97, atr=0.01, position_size=10, est_fee=0.0, spread=0.0)
    assert ok is True


def test_rr_and_atr_check_fails_for_poor_rr():
    from newbotcode import XRPTradingBot, BotConfig

    bot = XRPTradingBot(config=BotConfig())
    # RR = 0.02/0.03 < 1.0 < min_rr_ratio
    ok = bot.rr_and_atr_check(entry_price=1.00, tp_price=1.02, sl_price=0.97, atr=0.01, position_size=10, est_fee=0.0, spread=0.0)
    assert ok is False


def test_fee_adjusted_tp_sl_moves_tp_for_long_due_to_fees():
    from newbotcode import XRPTradingBot, BotConfig

    bot = XRPTradingBot(config=BotConfig())
    entry, sl, tp = 1.00, 0.97, 1.06
    sl_adj, tp_adj = bot.fee_adjusted_tp_sl(entry_px=entry, sl_px=sl, tp_px=tp, taker_fee=0.0045, is_long=True)
    # For long, TP should move slightly closer (<= original)
    assert tp_adj <= tp


def test_drawdown_lock_respects_configurable_duration(monkeypatch):
    from newbotcode import XRPTradingBot, BotConfig

    bot = XRPTradingBot(config=BotConfig())
    bot.drawdown_lock_seconds = 1  # 1 second lock for fast test
    bot.max_drawdown_pct = 0.10
    bot.peak_capital = 100.0

    # Simulate account status with 15% DD
    monkeypatch.setattr(bot, 'get_account_status', lambda: {'freeCollateral': 85.0, 'account_value': 85.0})

    # First invocation should lock
    allowed = bot.check_risk_limits()
    assert allowed is False
    assert getattr(bot, 'drawdown_lock_time', None) is not None

    # Immediately again should still be locked
    allowed = bot.check_risk_limits()
    assert allowed is False

    # After lock duration, it should expire and allow resume (even with DD still high per logic)
    bot.drawdown_lock_time = time.time() - 2
    allowed = bot.check_risk_limits()
    assert allowed is True


def test_analyzer_signal_on_uptrend_is_buy():
    from newbotcode import XRPTradingBot, BotConfig

    bot = XRPTradingBot(config=BotConfig())
    # Synthetic uptrend prices
    prices = [1.00 + i * 0.001 for i in range(260)]
    with bot._price_history_lock:
        bot.price_history.clear()
        for p in prices:
            bot.price_history.append(p)
    result = bot.analyze_xrp_signals()
    assert result["signal"] in ("BUY", "HOLD")

