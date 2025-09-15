import asyncio
from typing import List, Dict, Any


class EventDrivenBacktester:
	"""Minimal event-driven backtester.
	Feeds OHLCV bars to a provided bot instance in real-time order.
	Safe-by-default: only updates price history and invokes a lightweight hook
	so it won't disrupt existing live logic.
	"""

	def __init__(self, bot, ohlcv: List[Dict[str, Any]]):
		self.bot = bot
		self.ohlcv = ohlcv or []
		self.queue: asyncio.Queue = asyncio.Queue()

	async def _feed(self):
		for bar in self.ohlcv:
			await self.queue.put({"type": "bar", "bar": bar})

	async def run(self, process_hook: str = "update_only"):
		"""Run the backtest.
		process_hook:
		- "update_only": update price history and skip heavy logic
		- "analyze": update and call a cheap analysis method if present
		"""
		await self._feed()
		while not self.queue.empty():
			ev = await self.queue.get()
			if ev.get("type") != "bar":
				continue
			bar = ev["bar"]
			close_px = float(bar.get("close") or bar.get("c") or 0.0)
			if close_px <= 0:
				continue
			# Update bot price history safely
			try:
				self.bot.get_current_price = lambda symbol="XRP": close_px
				if hasattr(self.bot, "_price_history_lock") and hasattr(self.bot, "price_history"):
					with self.bot._price_history_lock:
						self.bot.price_history.append(close_px)
						if len(self.bot.price_history) > 1000:
							self.bot.price_history.popleft()
			except Exception:
				pass
			# Optional cheap hooks
			if process_hook == "analyze":
				try:
					if hasattr(self.bot, "_check_momentum_filter"):
						self.bot._check_momentum_filter(list(self.bot.price_history), "BUY")
				except Exception:
					pass
			await asyncio.sleep(0)
		return True


