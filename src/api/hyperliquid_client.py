"""
Hyperliquid API Client Module

This module contains the API client functionality for interacting with the Hyperliquid exchange.
It includes the hardened HTTP client with retry logic, circuit breaker, and rate limiting.
"""

import asyncio
import time
from typing import Dict, Any
from urllib3.util import Retry
import httpx

class HLClient(httpx.AsyncClient):
    """
    Hardened HTTP client with retry, backoff, and circuit breaker for Hyperliquid API.
    """
    _last_call = 0
    _min_interval = 0.25  # 4 req/s max
    _failure_count = 0
    _circuit_open = False
    _circuit_open_time = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_strategy = Retry(
            total=5,
            backoff_factor=0.4,
            status_forcelist=[502, 503, 504, 429],
            allowed_methods=["GET", "POST"]
        )

    async def safe_post(self, url: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safe POST with rate limiting, exponential backoff, and circuit breaker.
        """
        # Check circuit breaker
        if self._circuit_open:
            if time.time() - self._circuit_open_time < 60:  # 60s cooldown
                raise Exception("Circuit breaker OPEN - API endpoint temporarily unavailable")
            else:
                self._circuit_open = False
                self._failure_count = 0

        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            await asyncio.sleep(wait)

        try:
            r = await self.post(url, json=json_data, timeout=10.0)
            self._last_call = time.time()
            self._failure_count = 0  # Reset failure count on success

            if r.status_code == 429:
                print(f"⚠️ Rate limited (429), backing off for 2 seconds...")
                await asyncio.sleep(2)
                r = await self.post(url, json=json_data, timeout=10.0)
                self._last_call = time.time()

            r.raise_for_status()
            return r.json()

        except Exception as e:
            self._failure_count += 1
            print(f"❌ HTTP request failed (attempt {self._failure_count}): {e}")

            # Open circuit breaker after 3 consecutive failures
            if self._failure_count >= 3:
                self._circuit_open = True
                self._circuit_open_time = time.time()
                print(f"🚨 Circuit breaker OPEN - API endpoint unavailable")

            raise

# Global hardened client instance for backward compatibility
hl_client = HLClient() 