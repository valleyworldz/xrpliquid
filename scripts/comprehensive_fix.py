#!/usr/bin/env python3
"""
COMPREHENSIVE FIX
Fixes all f-string syntax errors from the symbol_cfg patches
"""

import re

def fix_all_symbol_cfg_issues():
    with open("newbotcode.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix all instances of the problematic pattern
    # Replace: getattr(self.symbol_cfg, "base", "XRP") if hasattr(self.symbol_cfg, "base") else "XRP"
    # With: getattr(self.symbol_cfg, 'base', 'XRP') if hasattr(self.symbol_cfg, 'base') else 'XRP'
    
    content = re.sub(
        r'getattr\(self\.symbol_cfg, "base", "XRP"\) if hasattr\(self\.symbol_cfg, "base"\) else "XRP"',
        "getattr(self.symbol_cfg, 'base', 'XRP') if hasattr(self.symbol_cfg, 'base') else 'XRP'",
        content
    )
    
    content = re.sub(
        r'getattr\(self\.symbol_cfg, "market", "CRYPTO"\) if hasattr\(self\.symbol_cfg, "market"\) else "CRYPTO"',
        "getattr(self.symbol_cfg, 'market', 'CRYPTO') if hasattr(self.symbol_cfg, 'market') else 'CRYPTO'",
        content
    )
    
    content = re.sub(
        r'getattr\(self\.symbol_cfg, "quote", "USDT"\) if hasattr\(self\.symbol_cfg, "quote"\) else "USDT"',
        "getattr(self.symbol_cfg, 'quote', 'USDT') if hasattr(self.symbol_cfg, 'quote') else 'USDT'",
        content
    )
    
    content = re.sub(
        r'getattr\(self\.symbol_cfg, "hl_name", "XRP/USDT"\) if hasattr\(self\.symbol_cfg, "hl_name"\) else "XRP/USDT"',
        "getattr(self.symbol_cfg, 'hl_name', 'XRP/USDT') if hasattr(self.symbol_cfg, 'hl_name') else 'XRP/USDT'",
        content
    )
    
    content = re.sub(
        r'getattr\(self\.symbol_cfg, "binance_pair", "XRPUSDT"\) if hasattr\(self\.symbol_cfg, "binance_pair"\) else "XRPUSDT"',
        "getattr(self.symbol_cfg, 'binance_pair', 'XRPUSDT') if hasattr(self.symbol_cfg, 'binance_pair') else 'XRPUSDT'",
        content
    )
    
    content = re.sub(
        r'getattr\(self\.symbol_cfg, "yahoo_ticker", "XRP-USD"\) if hasattr\(self\.symbol_cfg, "yahoo_ticker"\) else "XRP-USD"',
        "getattr(self.symbol_cfg, 'yahoo_ticker', 'XRP-USD') if hasattr(self.symbol_cfg, 'yahoo_ticker') else 'XRP-USD'",
        content
    )
    
    # Write the fixed file
    with open("newbotcode.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ All symbol_cfg syntax issues fixed!")

if __name__ == "__main__":
    fix_all_symbol_cfg_issues()
