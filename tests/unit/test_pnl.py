import pytest
import time

class DummyLogger:
    def __init__(self):
        self.logs = []
    def info(self, msg):
        self.logs.append(msg)
    def warning(self, msg):
        self.logs.append(msg)
    def error(self, msg):
        self.logs.append(msg)

class DummyBot:
    def __init__(self):
        self.logger = DummyLogger()
        self.fills_ledger = []
        self._seen_fills = set()
        self.daily_pnl = 0
    def refresh_fills(self, fills):
        for f in fills:
            oid = f.get("oid")
            if oid in self._seen_fills:
                continue
            self._seen_fills.add(oid)
            side = 1 if f["side"].lower() == "buy" else -1
            px = float(f["avgPx"])
            sz = float(f["totalSz"]) * side
            fee = float(f.get("fee", 0))
            ts = float(f.get("timestamp", time.time()*1000))
            self.fills_ledger.append({"ts": ts, "px": px, "sz": sz, "fee": fee})
        self._recalc_realized_pnl()
    def _recalc_realized_pnl(self):
        pos = 0.0
        cost = 0.0
        realized = 0.0
        for r in self.fills_ledger:
            if pos == 0:
                pos = r["sz"]
                cost = r["px"] * r["sz"]
            else:
                same_dir = (pos > 0 and r["sz"] > 0) or (pos < 0 and r["sz"] < 0)
                if same_dir:
                    pos += r["sz"]
                    cost += r["px"] * r["sz"]
                else:
                    closed = min(abs(pos), abs(r["sz"])) * (1 if pos > 0 else -1)
                    realized += (r["px"] - cost/pos) * closed * (1 if pos > 0 else -1)
                    pos += r["sz"]
                    cost += r["px"] * r["sz"]
            realized -= r["fee"]
        self.daily_pnl = realized
    def calculate_position_size(self, price, free_collateral, min_xrp=10, min_notional=10):
        size = free_collateral * 0.02 / price
        if size * price < min_notional or size < min_xrp:
            return 0
        return int(size)

def test_pnl_round_trip():
    bot = DummyBot()
    fills = [
        {"oid": "1", "side": "buy", "avgPx": 100, "totalSz": 10, "fee": 0.1},
        {"oid": "2", "side": "sell", "avgPx": 110, "totalSz": 10, "fee": 0.1},
    ]
    bot.refresh_fills(fills)
    assert abs(bot.daily_pnl - (10*(110-100) - 0.2)) < 1e-6

def test_pnl_partial_close():
    bot = DummyBot()
    fills = [
        {"oid": "1", "side": "buy", "avgPx": 100, "totalSz": 10, "fee": 0.1},
        {"oid": "2", "side": "sell", "avgPx": 110, "totalSz": 5, "fee": 0.05},
        {"oid": "3", "side": "sell", "avgPx": 120, "totalSz": 5, "fee": 0.05},
    ]
    bot.refresh_fills(fills)
    # First close: 5*(110-100), second: 5*(120-100), total - fees
    assert abs(bot.daily_pnl - (5*10 + 5*20 - 0.2)) < 1e-6

def test_skip_trade_when_not_affordable():
    bot = DummyBot()
    # Too little collateral
    assert bot.calculate_position_size(100, 5) == 0
    # Notional too small
    assert bot.calculate_position_size(0.5, 5) == 0
    # Just enough
    assert bot.calculate_position_size(10, 1000) >= 10

def test_position_tracking_after_fills():
    bot = DummyBot()
    fills = [
        {"oid": "1", "side": "buy", "avgPx": 100, "totalSz": 10, "fee": 0.1},
        {"oid": "2", "side": "sell", "avgPx": 110, "totalSz": 5, "fee": 0.05},
    ]
    bot.refresh_fills(fills)
    # Should have 5 XRP open
    pos = 0
    for r in bot.fills_ledger:
        pos += r["sz"]
    assert abs(pos - 5) < 1e-6 