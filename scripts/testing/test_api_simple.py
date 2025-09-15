#!/usr/bin/env python3
import asyncio
from hyperliquid_sdk.info import Info
from hyperliquid_sdk.exchange import Exchange
from core.utils.credential_handler import SecureCredentialHandler

async def test_api():
    print("Testing API...")
    
    # Load credentials
    handler = SecureCredentialHandler()
    if not handler.initialize():
        print(" Init failed")
        return
    
    creds = handler.load_credentials()
    if not creds:
        print(" Load failed")
        return
    
    # Test API
    try:
        info = Info()
        exchange = Exchange(creds["private_key"])
        
        # Test market info
        meta = await info.get_meta()
        print(f" Meta info: {len(meta['universe'])} coins")
        
        # Test user state
        user = await exchange.get_user_state()
        print(f" User state: {user['user']['address']}")
        
        # Test market data
        l2 = await info.get_l2_snapshot("BTC")
        print(f" Market data: BTC bid {l2['levels']['bids'][0][0]}")
        
        print(" All tests passed!")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api())
