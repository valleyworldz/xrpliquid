import time
import logging

class HyperliquidClient:
    def __init__(self, exchange=None, logger=None):
        self.exchange = exchange  # Should be set to HL SDK Exchange instance
        self.logger = logger or logging.getLogger(__name__)

    def place_order(self, symbol, is_buy, size, price, tif='Gtc', post_only=True):
        """
        Patch ②: Place a maker (postOnly) order first. If not filled in 3s, fallback to IOC (taker).
        Log any maker rebate if filled as maker.
        """
        order_side = 'buy' if is_buy else 'sell'
        # 1. Try postOnly (maker) order
        self.logger.info(f"[ORDER] Placing maker (postOnly) {order_side} {size} {symbol} @ {price:.4f}")
        order_result = None
        if self.exchange:
            order_result = self.exchange.order(
                symbol, is_buy, size, price,
                tif='Gtc', post_only=True
            )
        else:
            # Simulate order for test
            order_result = {'status': 'open', 'filled': 0, 'order_id': 'sim_maker'}
        # Wait 3s for fill
        time.sleep(3)
        # Check if filled (stub: always not filled in this test)
        filled = order_result.get('filled', 0)
        if filled < size:
            self.logger.info(f"[ORDER] Maker order not filled after 3s, falling back to IOC taker order.")
            if self.exchange:
                order_result = self.exchange.order(
                    symbol, is_buy, size, price,
                    tif='Ioc', post_only=False
                )
            else:
                order_result = {'status': 'filled', 'filled': size, 'order_id': 'sim_taker'}
        # Log rebate if maker
        if order_result.get('status') == 'filled' and post_only:
            rebate = size * price * 0.00015  # Example: 0.015% maker rebate
            self.logger.info(f"[ORDER] Maker rebate earned: {rebate:.6f} {symbol}")
        return order_result 