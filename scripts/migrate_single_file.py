#!/usr/bin/env python3
"""
Migration script: Splits PERFECT_CONSOLIDATED_BOT.py into modular files under src/bot/ and src/config.py.
- Extracts main trading loop and helpers to engine.py
- AdvancedPatternAnalyzer to patterns.py
- Risk helpers to risk.py
- HL SDK wrapper to hyperclient.py
- Constants/config to config.py
- Creates __init__.py as needed
- Prints progress for each step
"""
import os
import re
import ast
from pathlib import Path

SRC = Path(__file__).parent.parent / 'src'
BOT = SRC / 'bot'
CONFIG = SRC / 'config.py'
MONOLITH = Path(__file__).parent.parent / 'PERFECT_CONSOLIDATED_BOT.py'

os.makedirs(BOT, exist_ok=True)

with open(MONOLITH, 'r', encoding='utf-8') as f:
    code = f.read()

# --- Extract config/constants ---
print('Extracting config/constants...')
config_lines = []
for line in code.splitlines():
    if re.match(r'^[A-Z_]{3,}\s*=.*', line) and 'import' not in line:
        config_lines.append(line)
with open(CONFIG, 'w', encoding='utf-8') as f:
    f.write('# Auto-generated config/constants\n')
    f.write('\n'.join(config_lines) + '\n')

# --- Extract AdvancedPatternAnalyzer ---
print('Extracting AdvancedPatternAnalyzer...')
pattern_code = re.findall(r'(class AdvancedPatternAnalyzer[\s\S]+?)(^class |^def |^if __name__|^#|\Z)', code, re.MULTILINE)
if pattern_code:
    with open(BOT / 'patterns.py', 'w', encoding='utf-8') as f:
        f.write(pattern_code[0][0])
        f.write('\n')

# --- Extract XRPTradingBot and helpers ---
print('Extracting XRPTradingBot and trading engine...')
engine_code = re.findall(r'(class XRPTradingBot[\s\S]+?)(^class |^def |^if __name__|^#|\Z)', code, re.MULTILINE)
if engine_code:
    with open(BOT / 'engine.py', 'w', encoding='utf-8') as f:
        f.write(engine_code[0][0])
        f.write('\n')

# --- Extract risk helpers ---
print('Extracting risk helpers...')
risk_code = re.findall(r'(def [\w_]*risk[\w_]*\(.*\):[\s\S]+?)(^def |^class |^if __name__|^#|\Z)', code, re.MULTILINE)
if risk_code:
    with open(BOT / 'risk.py', 'w', encoding='utf-8') as f:
        for block in risk_code:
            f.write(block[0])
            f.write('\n')

# --- Extract HL SDK wrapper (hyperclient) ---
print('Extracting HL SDK wrapper...')
hyper_code = re.findall(r'(class HyperliquidClient[\s\S]+?)(^class |^def |^if __name__|^#|\Z)', code, re.MULTILINE)
if hyper_code:
    with open(BOT / 'hyperclient.py', 'w', encoding='utf-8') as f:
        f.write(hyper_code[0][0])
        f.write('\n')

# --- Create __init__.py files ---
print('Creating __init__.py files...')
for d in [BOT, SRC]:
    with open(d / '__init__.py', 'w', encoding='utf-8') as f:
        f.write('# Auto-generated\n')

print('Migration complete! Review src/bot/ and src/config.py.') 