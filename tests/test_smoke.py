#!/usr/bin/env python3
"""
Smoke Test for XRP Trading Bot
==============================
Tests that the bot can be instantiated and run a single cycle in dry-run mode.
This catches immediate issues without requiring real credentials or network access.
"""

import unittest
import sys
import os
import asyncio
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PERFECT_CONSOLIDATED_BOT import XRPTradingBot, BotConfig

class TestSmokeTest(unittest.TestCase):
    """Smoke test for bot instantiation and basic functionality"""

    def setUp(self):
        """Set up test environment"""
        self.config = BotConfig()
        
        # Mock all external dependencies
        self.patches = [
            patch('PERFECT_CONSOLIDATED_BOT.decrypt_credentials'),
            patch('PERFECT_CONSOLIDATED_BOT.Info'),
            patch('PERFECT_CONSOLIDATED_BOT.Exchange'),
            patch('requests.post'),
            patch('requests.get'),
        ]
        
        # Start all patches
        for p in self.patches:
            p.start()
        
        # Mock credential decryption
        self.patches[0].return_value = {
            'wallet_address': '0x1234567890abcdef',
            'private_key': '0x' + '0' * 64
        }
        
        # Mock Hyperliquid clients
        mock_info = MagicMock()
        mock_exchange = MagicMock()
        self.patches[1].return_value = mock_info
        self.patches[2].return_value = mock_exchange
        
        # Mock API responses
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "universe": [{
                "name": "XRP",
                "szDecimals": 0,
                "minSz": 1.0,
                "maxLeverage": 20,
                "pxDecimals": 4
            }],
            "fundingRates": [{
                "coin": "XRP",
                "rate": "0.0001"
            }]
        }
        self.patches[3].return_value = mock_response
        self.patches[4].return_value = mock_response

    def tearDown(self):
        """Clean up patches"""
        for p in self.patches:
            p.stop()

    def test_bot_instantiation(self):
        """Test that the bot can be instantiated"""
        try:
            bot = XRPTradingBot(self.config)
            self.assertIsNotNone(bot)
            self.assertTrue(hasattr(bot, 'config'))
            self.assertTrue(hasattr(bot, 'logger'))
        except Exception as e:
            self.fail(f"Bot instantiation failed: {e}")

    def test_bot_config_validation(self):
        """Test that bot config parameters are valid"""
        bot = XRPTradingBot(self.config)
        
        # Test critical config parameters
        self.assertGreaterEqual(bot.config.confidence_threshold, 0)
        self.assertLessEqual(bot.config.confidence_threshold, 1)
        self.assertGreater(bot.config.min_xrp, 0)
        self.assertGreater(bot.config.risk_per_trade, 0)
        self.assertLess(bot.config.risk_per_trade, 1)

    def test_bot_methods_exist(self):
        """Test that critical bot methods exist"""
        bot = XRPTradingBot(self.config)
        
        # Test that critical methods exist
        required_methods = [
            'get_current_price',
            'calculate_position_size',
            'place_order',
            'analyze_xrp_signals',
            'check_risk_limits',
            '_align_price_to_tick',
            'align_up',
            'align_down',
            'short_addr'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(bot, method), f"Missing method: {method}")

    @pytest.mark.slow
    def test_single_trading_cycle_dry_run(self):
        """Test a single trading cycle in dry-run mode"""
        bot = XRPTradingBot(self.config)
        bot.dry_run_mode = True
        
        # Mock price data
        with patch.object(bot, 'get_current_price', return_value=3.0):
            with patch.object(bot, 'get_account_status', return_value={'free_collateral': 1000.0}):
                with patch.object(bot, 'analyze_xrp_signals', return_value={
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'test'
                }):
                    # Run a single cycle
                    try:
                        # This should not raise any exceptions
                        bot.check_risk_limits()
                        bot.get_current_price()
                        bot.analyze_xrp_signals()
                    except Exception as e:
                        self.fail(f"Single trading cycle failed: {e}")

    def test_price_alignment_wrappers(self):
        """Test the price alignment wrapper functions"""
        bot = XRPTradingBot(self.config)
        
        price = 3.12345
        tick_size = 0.0001
        
        # Test align_up
        up_result = bot.align_up(price, tick_size)
        self.assertIsInstance(up_result, type(3.14))  # Should be Decimal or float
        self.assertGreaterEqual(float(up_result), price)
        
        # Test align_down
        down_result = bot.align_down(price, tick_size)
        self.assertIsInstance(down_result, type(3.14))  # Should be Decimal or float
        self.assertLessEqual(float(down_result), price)

    def test_short_addr_helper(self):
        """Test the short_addr helper function"""
        bot = XRPTradingBot(self.config)
        
        addr = "0x1234567890abcdef1234567890abcdef12345678"
        result = bot.short_addr(addr)
        
        # Should truncate to 6 chars
        self.assertEqual(result, "0x1234")
        
        # Should handle edge cases
        self.assertEqual(bot.short_addr(None), '')
        self.assertEqual(bot.short_addr(''), '')

    def test_logging_setup(self):
        """Test that logging is properly configured"""
        bot = XRPTradingBot(self.config)
        
        # Test that logger exists and has proper level
        self.assertIsNotNone(bot.logger)
        self.assertTrue(hasattr(bot.logger, 'info'))
        self.assertTrue(hasattr(bot.logger, 'warning'))
        self.assertTrue(hasattr(bot.logger, 'error'))

if __name__ == '__main__':
    unittest.main() 