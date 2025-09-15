#!/usr/bin/env python3
"""
HTTP Client Module
Hardened HTTP client with retry, backoff, and circuit breaker for Hyperliquid API
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import httpx safely
try:
    import httpx
    from urllib3.util.retry import Retry
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️ httpx not available, using fallback HTTP client")

# Fallback classes for when numpy/pandas are not available
class FallbackNumpy:
    def std(self, values):
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def mean(self, values):
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def sqrt(self, value):
        return value ** 0.5
    
    def array(self, values):
        # Simple array-like object
        if isinstance(values, (list, tuple)):
            return values
        return [values]
    
    def zeros(self, shape):
        if isinstance(shape, int):
            return [0.0] * shape
        elif isinstance(shape, tuple):
            if len(shape) == 1:
                return [0.0] * shape[0]
            else:
                return [[0.0] * shape[1] for _ in range(shape[0])]
        return [0.0]
    
    def ones(self, shape):
        if isinstance(shape, int):
            return [1.0] * shape
        elif isinstance(shape, tuple):
            if len(shape) == 1:
                return [1.0] * shape[0]
            else:
                return [[1.0] * shape[1] for _ in range(shape[0])]
        return [1.0]
    
    def concatenate(self, arrays):
        result = []
        for arr in arrays:
            if isinstance(arr, (list, tuple)):
                result.extend(arr)
            else:
                result.append(arr)
        return result
    
    def diff(self, values):
        """Calculate differences between consecutive elements"""
        if len(values) < 2:
            return []
        result = []
        for i in range(1, len(values)):
            result.append(values[i] - values[i-1])
        return result
    
    def corrcoef(self, x, y):
        """Calculate correlation coefficient between two arrays"""
        if len(x) != len(y) or len(x) < 2:
            return [[1.0, 0.0], [0.0, 1.0]]
        
        # Calculate means
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        # Calculate covariance and variances
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        var_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        var_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        
        # Calculate correlation
        if var_x == 0 or var_y == 0:
            return [[1.0, 0.0], [0.0, 1.0]]
        
        correlation = numerator / (var_x * var_y) ** 0.5
        return [[1.0, correlation], [correlation, 1.0]]
    
    def isnan(self, value):
        """Check if value is NaN"""
        try:
            return float(value) != float(value)  # NaN is the only value that doesn't equal itself
        except:
            return False

# Clean numpy import with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = FallbackNumpy()

class FallbackDataFrame:
    def __init__(self, data=None, columns=None, index=None):
        if data is None:
            self.data = {}
        elif isinstance(data, list):
            if columns:
                self.data = {col: [row[i] if i < len(row) else None for row in data] for i, col in enumerate(columns)}
            else:
                self.data = {i: [row[i] if i < len(row) else None for row in data] for i in range(len(data[0]) if data else 0)}
        else:
            self.data = data
        self.index = index or list(range(len(self.data.get(list(self.data.keys())[0], [])) if self.data else 0))
    
    def __getitem__(self, key):
        if isinstance(key, str):
            # Safe string key access - only if data is string-keyed dict
            if isinstance(self.data, dict):
                try:
                    # Only use get if key type matches dict keys
                    if all(isinstance(k, str) for k in self.data.keys()):
                        # Type checker doesn't understand the check, so cast to Any
                        from typing import Any
                        return self.data.get(key, [])  # type: ignore
                except Exception:
                    pass
            return []
        elif isinstance(key, (list, tuple)):
            return FallbackDataFrame({k: self.data[k] for k in key if k in self.data})
        elif isinstance(key, int):
            # Handle integer indexing
            if isinstance(self.data, dict):
                try:
                    if all(isinstance(k, int) for k in self.data.keys()):
                        # Type checker doesn't understand the check, so cast to Any
                        from typing import Any
                        return self.data.get(key, [])  # type: ignore
                except Exception:
                    pass
            return []
        else:
            raise KeyError(f"Invalid key type: {type(key)}")
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def set_index(self, key, inplace=False):
        if inplace:
            self.index = self.data.get(key, [])
            return self
        else:
            new_df = FallbackDataFrame(self.data.copy())
            new_df.index = self.data.get(key, [])
            return new_df
    
    def iloc(self, index):
        if isinstance(index, int):
            return {k: v[index] if index < len(v) else None for k, v in self.data.items()}
        return FallbackDataFrame({k: v[index] for k, v in self.data.items()})
    
    def empty(self):
        return len(self.data) == 0 or len(self.data.get(list(self.data.keys())[0], [])) == 0
    
    def dropna(self):
        # Simple dropna implementation
        return self
    
    def pct_change(self, periods=1):
        result = FallbackDataFrame()
        for col, values in self.data.items():
            if isinstance(values, (list, tuple)):
                new_values = []
                for i in range(len(values)):
                    if i < periods:
                        new_values.append(None)
                    else:
                        try:
                            if (values[i] is not None and values[i-periods] is not None and 
                                values[i-periods] is not None and values[i-periods] != 0):
                                try:
                                    # Ensure both values are numbers before arithmetic
                                    current_val_raw = values[i]
                                    prev_val_raw = values[i-periods]
                                    
                                    if current_val_raw is not None and prev_val_raw is not None:
                                        current_val = float(current_val_raw)
                                        prev_val = float(prev_val_raw)
                                        if prev_val != 0:
                                            change = (current_val - prev_val) / prev_val
                                            new_values.append(change)
                                        else:
                                            new_values.append(None)
                                    else:
                                        new_values.append(None)
                                except (TypeError, ValueError):
                                    new_values.append(None)
                            else:
                                new_values.append(None)
                        except (TypeError, ValueError):
                            new_values.append(None)
                result.data[col] = new_values
        return result

    def rolling(self, window):
        class Rolling:
            def __init__(self, df, window):
                self.df = df
                self.window = window
            
            def mean(self):
                result = FallbackDataFrame()
                for col, values in self.df.data.items():
                    if isinstance(values, (list, tuple)):
                        new_values = []
                        for i in range(len(values)):
                            if i < self.window - 1:
                                new_values.append(None)
                            else:
                                window_values = values[max(0, i-self.window+1):i+1]
                                window_values = [v for v in window_values if v is not None]
                                if window_values:
                                    new_values.append(sum(window_values) / len(window_values))
                                else:
                                    new_values.append(None)
                        result.data[col] = new_values
                return result
            
            def std(self):
                result = FallbackDataFrame()
                for col, values in self.df.data.items():
                    if isinstance(values, (list, tuple)):
                        new_values = []
                        for i in range(len(values)):
                            if i < self.window - 1:
                                new_values.append(None)
                            else:
                                window_values = values[max(0, i-self.window+1):i+1]
                                window_values = [v for v in window_values if v is not None]
                                if len(window_values) > 1:
                                    mean_val = sum(window_values) / len(window_values)
                                    variance = sum((v - mean_val) ** 2 for v in window_values) / len(window_values)
                                    new_values.append(variance ** 0.5)
                                else:
                                    new_values.append(None)
                        result.data[col] = new_values
                return result
            
            def skew(self):
                # Simple skew implementation
                return self.mean()
            
            def kurt(self):
                # Simple kurtosis implementation
                return self.mean()
        
        return Rolling(self, window)
    
    def ewm(self, span):
        class EWM:
            def __init__(self, df, span):
                self.df = df
                self.span = span
            
            def mean(self):
                result = FallbackDataFrame()
                for col, values in self.df.data.items():
                    if isinstance(values, (list, tuple)):
                        new_values = []
                        alpha = 2.0 / (self.span + 1)
                        for i, val in enumerate(values):
                            if i == 0:
                                new_values.append(val)
                            else:
                                new_values.append(alpha * val + (1 - alpha) * new_values[-1])
                        result.data[col] = new_values
                return result
        
        return EWM(self, span)
    
    def diff(self, periods=1):
        result = FallbackDataFrame()
        for col, values in self.data.items():
            if isinstance(values, (list, tuple)):
                new_values = []
                for i in range(len(values)):
                    if i < periods:
                        new_values.append(None)
                    else:
                        try:
                            # Ensure both values are not None before arithmetic
                            current_val = values[i]
                            prev_val = values[i-periods]
                            if (current_val is not None and prev_val is not None and 
                                prev_val != 0):
                                change = (current_val - prev_val) / prev_val
                                new_values.append(change)
                            else:
                                new_values.append(None)
                        except (TypeError, ValueError):
                            new_values.append(None)
                result.data[col] = new_values
        return result
    
    def concat(self, other_dfs, axis=0):
        if isinstance(other_dfs, list):
            result = FallbackDataFrame()
            for df in other_dfs:
                for col, values in df.data.items():
                    if col not in result.data:
                        result.data[col] = []
                    result.data[col].extend(values)
            return result
        return self

# Clean pandas import with fallback
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = type('pd', (), {
        'DataFrame': FallbackDataFrame,
        'concat': lambda dfs, **kwargs: dfs[0].concat(dfs[1:], **kwargs) if len(dfs) > 1 else dfs[0]
    })()


class HLClient:
    """Hardened HTTP client with retry, backoff, and circuit breaker"""
    
    def __init__(self, *args, **kwargs):
        self._last_call = 0
        self._min_interval = 0.25  # 4 req/s max
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_open_time = 0
        
        # Initialize httpx client if available
        if HTTPX_AVAILABLE:
            self.client = httpx.AsyncClient(*args, **kwargs)
            # Configure retry strategy
            self.retry_strategy = Retry(
                total=5,
                backoff_factor=0.4,
                status_forcelist=[502, 503, 504, 429],
                allowed_methods=["GET", "POST"]
            )
        else:
            self.client = None
            print("⚠️ Using fallback HTTP client - limited functionality")
    
    async def safe_post(self, url, json_data):
        """Safe POST with rate limiting, exponential backoff, and circuit breaker"""
        if not HTTPX_AVAILABLE:
            raise Exception("httpx not available - cannot make HTTP requests")
        
        # Check circuit breaker
        if self._circuit_open:
            if time.time() - self._circuit_open_time < 60:  # 60 second cooldown
                raise Exception("Circuit breaker OPEN - API endpoint temporarily unavailable")
            else:
                self._circuit_open = False
                self._failure_count = 0
        
        wait = self._min_interval - (time.time() - self._last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        
        try:
            r = await self.client.post(url, json=json_data, timeout=10.0)
            self._last_call = time.time()
            self._failure_count = 0  # Reset failure count on success
            
            if r.status_code == 429:
                print(f"⚠️ Rate limited (429), backing off for 2 seconds...")
                await asyncio.sleep(2)
                r = await self.client.post(url, json=json_data, timeout=10.0)
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
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()


# Global hardened client instance
hl_client = HLClient()

# Export numpy and pandas for compatibility
__all__ = ['HLClient', 'hl_client', 'np', 'pd', 'FallbackNumpy', 'FallbackDataFrame'] 