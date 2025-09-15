import sys
import os
import csv
import argparse
import logging
from collections import deque

# Ensure src is in sys.path regardless of CWD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bot.engine import XRPTradingBot
from bot.patterns import AdvancedPatternAnalyzer

class DummyAPIClient:
    def __init__(self, prices):
        self.prices = prices
        self.index = 0
    def get_current_price(self, symbol="XRP"):
        if self.index < len(self.prices):
            return self.prices[self.index]
        return self.prices[-1]
    def next(self):
        self.index += 1

# Simple backtest runner
class BacktestRunner:
    def __init__(self, price_history):
        self.price_history = price_history
        self.bot = XRPTradingBot.__new__(XRPTradingBot)
        self.bot.logger = logging.getLogger("backtest")
        self.bot.xrp_price_history = deque(maxlen=200)
        self.bot.pattern_analyzer = AdvancedPatternAnalyzer(self.bot.logger)
        self.bot.get_current_price = lambda symbol="XRP": self.price_history[self.current_index]
        self.bot.get_volume_data = lambda symbol="XRP": {"volume_24h": 2_000_000}
        self.bot.get_account_status = lambda: {"withdrawable": 10000, "xrp_position": 0}
        self.bot.place_order = lambda *a, **kw: {"status": "filled", "filled": 100, "order_id": "sim"}
        self.bot.place_native_tpsl_pair = lambda **kw: {"tp_oid": 1, "sl_oid": 2}
        self.bot.calculate_atr_enhanced = lambda prices: 0.02
        self.bot.has_open_position = lambda: False
        self.bot.check_post_trade_cooldown_patch6 = lambda: True
        self.bot.should_skip_trade_by_funding_enhanced = lambda signal: False
        self.bot.check_margin_ratio = lambda: True
        self.bot.check_hold_time_constraints_enhanced = lambda: True
        self.bot.update_trade_timestamp = lambda: None
        self.bot.get_recent_price_data = lambda n: list(self.bot.xrp_price_history)[-n:]
        self.current_index = 0
        self.trades = []
        self.pnl = 0.0
        self.max_drawdown = 0.0
        self.equity_curve = [10000.0]
    def run(self):
        for i in range(200, len(self.price_history)):
            self.current_index = i
            self.bot.xrp_price_history.extend(self.price_history[i-200:i])
            signal, confidence = self.bot.analyze_xrp_signals()
            if signal in ("BUY", "SELL") and abs(confidence) >= self.bot.confidence_threshold:
                entry = self.price_history[i]
                # Simulate a trade: exit after 10 bars
                exit_index = min(i+10, len(self.price_history)-1)
                exit_price = self.price_history[exit_index]
                size = 100
                if signal == "BUY":
                    trade_pnl = (exit_price - entry) * size
                else:
                    trade_pnl = (entry - exit_price) * size
                self.pnl += trade_pnl
                self.trades.append(trade_pnl)
                self.equity_curve.append(self.equity_curve[-1] + trade_pnl)
                dd = max(self.equity_curve[-1] - max(self.equity_curve), 0)
                self.max_drawdown = min(self.max_drawdown, dd)
        total = len(self.trades)
        wins = sum(1 for t in self.trades if t > 0)
        win_rate = wins / total if total else 0
        print(f"Backtest complete: {total} trades, win rate {win_rate:.2%}, PnL {self.pnl:.2f}, Max Drawdown {self.max_drawdown:.2f}")

def load_price_history(csv_path):
    prices = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row['close']))
    return prices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV file with historical prices (must have a close column)')
    args = parser.parse_args()
    prices = load_price_history(args.csv)
    runner = BacktestRunner(prices)
    runner.run()

if __name__ == '__main__':
    main() 