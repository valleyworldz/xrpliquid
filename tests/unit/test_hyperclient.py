import sys
import os
import pytest
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class DummyLogger:
    def __init__(self):
        self.logs = []
    def info(self, msg):
        self.logs.append(msg)
    def warning(self, msg):
        self.logs.append(msg)
    def error(self, msg):
        self.logs.append(msg)


def test_hyperclient_import():
    import src.bot.hyperclient
    assert True

def test_place_order_maker_then_taker(monkeypatch):
    from src.bot.hyperclient import HyperliquidClient
    logger = DummyLogger()
    client = HyperliquidClient(exchange=None, logger=logger)
    # Should fallback to taker after 3s
    result = client.place_order('XRP', True, 10, 1.0)
    assert result['status'] == 'filled'
    assert any('falling back to IOC' in log for log in logger.logs)


def test_place_order_maker_fill(monkeypatch):
    from src.bot.hyperclient import HyperliquidClient
    logger = DummyLogger()
    class DummyExchange:
        def order(self, symbol, is_buy, size, price, tif, post_only):
            return {'status': 'filled', 'filled': size, 'order_id': 'maker', 'post_only': post_only}
    client = HyperliquidClient(exchange=DummyExchange(), logger=logger)
    result = client.place_order('XRP', True, 10, 1.0)
    assert result['status'] == 'filled'
    assert any('Maker rebate earned' in log for log in logger.logs) 