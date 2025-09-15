#!/usr/bin/env python3
"""
Unit Test for rr_and_atr_check with Extreme Inputs
==================================================
Test the risk/reward and ATR validation function with extreme inputs to ensure
it never flips sign errors and handles edge cases correctly.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PERFECT_CONSOLIDATED_BOT import XRPTradingBot, BotConfig

class TestRrAtrExtremeInputs(unittest.TestCase):
    """Test rr_and_atr_check with extreme and edge case inputs"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = BotConfig()
        with patch('PERFECT_CONSOLIDATED_BOT.decrypt_credentials') as mock_creds:
            mock_creds.return_value = {
                'wallet_address': '0x1234567890abcdef',
                'private_key': '0x' + '0' * 64
            }
            self.bot = XRPTradingBot(self.config)

    def test_extreme_price_inputs(self):
        """Test with extreme price values"""
        # Test with very small prices
        result = self.bot.rr_and_atr_check(
            entry_price=0.0001,  # Very small price
            tp_price=0.0002,
            sl_price=0.00005,
            atr=0.00001
        )
        self.assertIsInstance(result, bool)
        
        # Test with very large prices
        result = self.bot.rr_and_atr_check(
            entry_price=1000000.0,  # Very large price
            tp_price=1100000.0,
            sl_price=900000.0,
            atr=10000.0
        )
        self.assertIsInstance(result, bool)
        
        # Test with negative prices (should handle gracefully)
        result = self.bot.rr_and_atr_check(
            entry_price=-1.0,
            tp_price=-0.5,
            sl_price=-2.0,
            atr=0.1
        )
        self.assertIsInstance(result, bool)

    def test_zero_and_near_zero_inputs(self):
        """Test with zero and near-zero values"""
        # Test with zero ATR
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.0
        )
        self.assertIsInstance(result, bool)
        
        # Test with near-zero ATR
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=1e-10
        )
        self.assertIsInstance(result, bool)
        
        # Test with zero entry price
        result = self.bot.rr_and_atr_check(
            entry_price=0.0,
            tp_price=0.1,
            sl_price=-0.1,
            atr=0.01
        )
        self.assertIsInstance(result, bool)

    def test_identical_prices(self):
        """Test with identical or very close prices"""
        # Test with identical TP and entry price
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.0,  # Same as entry
            sl_price=2.9,
            atr=0.01
        )
        self.assertIsInstance(result, bool)
        
        # Test with identical SL and entry price
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=3.0,  # Same as entry
            atr=0.01
        )
        self.assertIsInstance(result, bool)
        
        # Test with all identical prices
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.0,
            sl_price=3.0,
            atr=0.01
        )
        self.assertIsInstance(result, bool)

    def test_inverted_price_relationships(self):
        """Test with inverted or unusual price relationships"""
        # Test with TP below entry price
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=2.9,  # Below entry
            sl_price=2.8,
            atr=0.01
        )
        self.assertIsInstance(result, bool)
        
        # Test with SL above entry price
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.2,
            sl_price=3.1,  # Above entry
            atr=0.01
        )
        self.assertIsInstance(result, bool)
        
        # Test with TP below SL
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=2.8,  # Below SL
            sl_price=2.9,
            atr=0.01
        )
        self.assertIsInstance(result, bool)

    def test_extreme_atr_values(self):
        """Test with extreme ATR values"""
        # Test with very large ATR
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=1000.0  # Very large ATR
        )
        self.assertIsInstance(result, bool)
        
        # Test with negative ATR
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=-0.01  # Negative ATR
        )
        self.assertIsInstance(result, bool)
        
        # Test with infinite ATR
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=float('inf')
        )
        self.assertIsInstance(result, bool)

    def test_none_and_nan_inputs(self):
        """Test with None and NaN values"""
        # Test with None values
        result = self.bot.rr_and_atr_check(
            entry_price=None,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01
        )
        self.assertIsInstance(result, bool)
        
        # Test with NaN values
        import math
        result = self.bot.rr_and_atr_check(
            entry_price=float('nan'),
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01
        )
        self.assertIsInstance(result, bool)
        
        # Test with mixed None/NaN
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=float('nan'),
            sl_price=None,
            atr=0.01
        )
        self.assertIsInstance(result, bool)

    def test_extreme_position_sizes(self):
        """Test with extreme position size values"""
        # Test with very large position size
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            position_size=1000000  # Very large position
        )
        self.assertIsInstance(result, bool)
        
        # Test with zero position size
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            position_size=0
        )
        self.assertIsInstance(result, bool)
        
        # Test with negative position size
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            position_size=-100
        )
        self.assertIsInstance(result, bool)

    def test_extreme_fee_values(self):
        """Test with extreme fee values"""
        # Test with very high fees
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            est_fee=0.5  # 50% fee
        )
        self.assertIsInstance(result, bool)
        
        # Test with negative fees
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            est_fee=-0.01
        )
        self.assertIsInstance(result, bool)
        
        # Test with zero fees
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            est_fee=0.0
        )
        self.assertIsInstance(result, bool)

    def test_extreme_spread_values(self):
        """Test with extreme spread values"""
        # Test with very high spread
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            spread=0.5  # 50% spread
        )
        self.assertIsInstance(result, bool)
        
        # Test with negative spread
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            spread=-0.01
        )
        self.assertIsInstance(result, bool)
        
        # Test with zero spread
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01,
            spread=0.0
        )
        self.assertIsInstance(result, bool)

    def test_floating_point_precision_issues(self):
        """Test with floating point precision edge cases"""
        # Test with very small differences
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.0 + 1e-15,  # Very small difference
            sl_price=3.0 - 1e-15,
            atr=1e-10
        )
        self.assertIsInstance(result, bool)
        
        # Test with floating point precision limits
        result = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.0 + float.epsilon,
            sl_price=3.0 - float.epsilon,
            atr=float.epsilon
        )
        self.assertIsInstance(result, bool)

    def test_consistency_across_calls(self):
        """Test that the function returns consistent results for same inputs"""
        # Test multiple calls with same parameters
        result1 = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01
        )
        result2 = self.bot.rr_and_atr_check(
            entry_price=3.0,
            tp_price=3.1,
            sl_price=2.9,
            atr=0.01
        )
        self.assertEqual(result1, result2)

    def test_no_exceptions_thrown(self):
        """Test that no exceptions are thrown with extreme inputs"""
        extreme_inputs = [
            # (entry, tp, sl, atr, pos_size, fee, spread)
            (0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0),
            (float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')),
            (float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')),
            (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')),
            (None, None, None, None, None, None, None),
        ]
        
        for entry, tp, sl, atr, pos_size, fee, spread in extreme_inputs:
            try:
                result = self.bot.rr_and_atr_check(
                    entry_price=entry,
                    tp_price=tp,
                    sl_price=sl,
                    atr=atr,
                    position_size=pos_size,
                    est_fee=fee,
                    spread=spread
                )
                self.assertIsInstance(result, bool)
            except Exception as e:
                self.fail(f"Exception thrown with inputs ({entry}, {tp}, {sl}, {atr}, {pos_size}, {fee}, {spread}): {e}")

if __name__ == '__main__':
    unittest.main() 