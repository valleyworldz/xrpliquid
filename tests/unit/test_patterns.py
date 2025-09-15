import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def test_patterns_import():
    import src.bot.patterns
    assert True

def test_ema_rsi_filter():
    from src.bot.patterns import AdvancedPatternAnalyzer
    analyzer = AdvancedPatternAnalyzer()
    # Generate price history for bullish cross (EMA50>EMA200) and RSI>55
    price_history = [1]*150 + [2]*50  # Uptrend
    result = analyzer.analyze_xrp_patterns(price_history)
    print('Uptrend:', result)
    assert result['signal'] == 'BUY'
    # Generate price history for bearish cross (EMA50<EMA200) and RSI<45
    price_history = list(reversed([1 + i*0.01 for i in range(200)]))  # Gradual downtrend
    result = analyzer.analyze_xrp_patterns(price_history)
    print('Downtrend:', result)
    assert result['signal'] == 'SELL'
    # Generate price history for no clear signal
    price_history = [1]*200
    result = analyzer.analyze_xrp_patterns(price_history)
    print('Flat:', result)
    assert result['signal'] == 'HOLD'

def test_rsi_threshold_patch8():
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.bot.patterns import AdvancedPatternAnalyzer
    analyzer = AdvancedPatternAnalyzer()
    # Uptrend, RSI just above 52
    price_history = [1 + i*0.01 for i in range(200)]
    # Artificially boost RSI to just above 52
    for i in range(50):
        price_history[-(i+1)] += 0.02
    result = analyzer.analyze_xrp_patterns(price_history)
    assert result['signal'] == 'BUY'
    # Downtrend, RSI just below 48
    price_history = list(reversed([1 + i*0.01 for i in range(200)]))
    for i in range(50):
        price_history[-(i+1)] -= 0.02
    result = analyzer.analyze_xrp_patterns(price_history)
    assert result['signal'] == 'SELL'
    # Flat, RSI between 48 and 52
    price_history = [1.0]*200
    result = analyzer.analyze_xrp_patterns(price_history)
    assert result['signal'] == 'HOLD' 