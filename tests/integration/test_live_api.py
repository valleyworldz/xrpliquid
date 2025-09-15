import os
import pytest

# Try to import the real client, skip if not available
try:
    from src.bot.engine import XRPTradingBot
except ImportError:
    pytest.skip("Modular bot not available", allow_module_level=True)

def has_creds():
    return os.path.exists("config/environments/secure_creds.env") or os.path.exists("config/credentials.json")

@pytest.mark.skipif(not has_creds(), reason="No API credentials available")
def test_live_hyperliquid_api():
    bot = XRPTradingBot()
    price = bot.get_current_price("XRP")
    volume = bot.get_volume_data("XRP")
    assert price is not None and price > 0, "Price should be positive"
    assert volume is not None and volume.get("volume_24h", 0) > 0, "Volume should be positive" 