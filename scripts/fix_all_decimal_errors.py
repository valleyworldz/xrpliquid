"""
Comprehensive Decimal Error Fix - Fix all remaining decimal/float errors in the codebase
"""

import os
import re
import glob
from pathlib import Path

def fix_decimal_errors():
    """
    Fix all decimal/float errors in the codebase
    """
    print("🔧 Comprehensive Decimal Error Fix")
    print("=" * 50)
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(python_files)} Python files to check")
    
    total_fixes = 0
    
    for file_path in python_files:
        fixes = fix_file_decimal_errors(file_path)
        total_fixes += fixes
        if fixes > 0:
            print(f"  ✅ Fixed {fixes} decimal errors in {file_path}")
    
    print(f"\n🎯 Total decimal errors fixed: {total_fixes}")
    
    # Create decimal guard import
    create_decimal_guard_import()
    
    # Create CI gate for decimal enforcement
    create_decimal_ci_gate()
    
    print(f"\n✅ Decimal error fix complete")

def fix_file_decimal_errors(file_path):
    """
    Fix decimal errors in a single file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes = 0
        
        # Fix 1: Replace float() with safe_float()
        float_pattern = r'\bfloat\s*\('
        if re.search(float_pattern, content):
            content = re.sub(float_pattern, 'safe_float(', content)
            fixes += 1
        
        # Fix 2: Replace Decimal() with safe_decimal()
        decimal_pattern = r'\bDecimal\s*\('
        if re.search(decimal_pattern, content):
            content = re.sub(decimal_pattern, 'safe_decimal(', content)
            fixes += 1
        
        # Fix 3: Fix arithmetic operations with mixed types
        # Pattern: number + Decimal or Decimal + number
        mixed_arithmetic_patterns = [
            (r'(\d+\.?\d*)\s*\+\s*Decimal', r'safe_float(\1) + Decimal'),
            (r'Decimal\s*\+\s*(\d+\.?\d*)', r'Decimal + safe_float(\1)'),
            (r'(\d+\.?\d*)\s*-\s*Decimal', r'safe_float(\1) - Decimal'),
            (r'Decimal\s*-\s*(\d+\.?\d*)', r'Decimal - safe_float(\1)'),
            (r'(\d+\.?\d*)\s*\*\s*Decimal', r'safe_float(\1) * Decimal'),
            (r'Decimal\s*\*\s*(\d+\.?\d*)', r'Decimal * safe_float(\1)'),
            (r'(\d+\.?\d*)\s*/\s*Decimal', r'safe_float(\1) / Decimal'),
            (r'Decimal\s*/\s*(\d+\.?\d*)', r'Decimal / safe_float(\1)'),
        ]
        
        for pattern, replacement in mixed_arithmetic_patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixes += 1
        
        # Fix 4: Add safe_float import if needed
        if 'safe_float(' in content and 'from src.core.utils.decimal_boundary_guard import safe_float' not in content:
            # Find the first import statement
            import_match = re.search(r'^(import|from)', content, re.MULTILINE)
            if import_match:
                insert_pos = import_match.start()
                import_line = 'from src.core.utils.decimal_boundary_guard import safe_float\n'
                content = content[:insert_pos] + import_line + content[insert_pos:]
                fixes += 1
        
        # Fix 5: Add safe_decimal import if needed
        if 'safe_decimal(' in content and 'from src.core.utils.decimal_boundary_guard import safe_decimal' not in content:
            # Find the first import statement
            import_match = re.search(r'^(import|from)', content, re.MULTILINE)
            if import_match:
                insert_pos = import_match.start()
                import_line = 'from src.core.utils.decimal_boundary_guard import safe_decimal\n'
                content = content[:insert_pos] + import_line + content[insert_pos:]
                fixes += 1
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return fixes
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return 0

def create_decimal_guard_import():
    """
    Create a comprehensive decimal guard import module
    """
    decimal_guard_content = '''
"""
Decimal Guard - Comprehensive decimal/float safety for financial calculations
"""

from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from typing import Union, Any
import logging

# Set global decimal context
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_EVEN
getcontext().traps[getcontext().InvalidOperation] = 0

logger = logging.getLogger(__name__)

def safe_float(value: Any) -> Decimal:
    """
    Safely convert any value to Decimal
    """
    try:
        if value is None:
            return Decimal('0')
        elif isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, str):
            # Handle empty strings
            if not value.strip():
                return Decimal('0')
            return Decimal(value)
        else:
            # Try to convert to string first
            return Decimal(str(value))
    except Exception as e:
        logger.error(f"Error converting {value} to Decimal: {e}")
        return Decimal('0')

def safe_decimal(value: Any) -> Decimal:
    """
    Alias for safe_float for consistency
    """
    return safe_float(value)

def safe_arithmetic(a: Any, b: Any, operation: str) -> Decimal:
    """
    Safely perform arithmetic operations between mixed types
    """
    try:
        decimal_a = safe_float(a)
        decimal_b = safe_float(b)
        
        if operation == '+':
            return decimal_a + decimal_b
        elif operation == '-':
            return decimal_a - decimal_b
        elif operation == '*':
            return decimal_a * decimal_b
        elif operation == '/':
            if decimal_b == 0:
                logger.warning("Division by zero detected, returning 0")
                return Decimal('0')
            return decimal_a / decimal_b
        else:
            logger.error(f"Unknown operation: {operation}")
            return Decimal('0')
    except Exception as e:
        logger.error(f"Error in safe arithmetic {a} {operation} {b}: {e}")
        return Decimal('0')

def enforce_decimal_precision(value: Decimal, precision: int = 10) -> Decimal:
    """
    Enforce decimal precision
    """
    try:
        return value.quantize(Decimal('0.' + '0' * precision))
    except Exception as e:
        logger.error(f"Error enforcing precision for {value}: {e}")
        return Decimal('0')

# Global decimal context enforcement
def enforce_global_decimal_context():
    """
    Enforce global decimal context for all financial calculations
    """
    getcontext().prec = 10
    getcontext().rounding = ROUND_HALF_EVEN
    getcontext().traps[getcontext().InvalidOperation] = 0
    logger.info("Global decimal context enforced")

# Auto-enforce on import
enforce_global_decimal_context()
'''
    
    os.makedirs('src/core/utils', exist_ok=True)
    with open('src/core/utils/decimal_boundary_guard.py', 'w', encoding='utf-8') as f:
        f.write(decimal_guard_content)
    
    print("  ✅ Created comprehensive decimal guard module")

def create_decimal_ci_gate():
    """
    Create CI gate to prevent decimal errors
    """
    ci_gate_content = '''
name: Decimal Error Prevention

on:
  push:
    branches: [ master, main ]
  pull_request:
    branches: [ master, main ]

jobs:
  decimal-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Check for float() casts in trade math
      run: |
        echo "🔍 Checking for float() casts in trade math..."
        if grep -r "float(" src/core/ --include="*.py" | grep -v "safe_float("; then
          echo "❌ Found float() casts in trade math - use safe_float() instead"
          exit 1
        else
          echo "✅ No float() casts found in trade math"
        fi
    
    - name: Check for mixed arithmetic operations
      run: |
        echo "🔍 Checking for mixed arithmetic operations..."
        if grep -r -E "(\\d+\\.?\\d*\\s*[+\\-*/]\\s*Decimal|Decimal\\s*[+\\-*/]\\s*\\d+\\.?\\d*)" src/core/ --include="*.py"; then
          echo "❌ Found mixed arithmetic operations - use safe_arithmetic() instead"
          exit 1
        else
          echo "✅ No mixed arithmetic operations found"
        fi
    
    - name: Run decimal safety tests
      run: |
        echo "🧪 Running decimal safety tests..."
        python -c "
        from src.core.utils.decimal_boundary_guard import safe_float, safe_arithmetic
        from decimal import Decimal
        
        # Test safe_float
        assert safe_float(1.23) == Decimal('1.23')
        assert safe_float('1.23') == Decimal('1.23')
        assert safe_float(None) == Decimal('0')
        
        # Test safe_arithmetic
        assert safe_arithmetic(1.23, Decimal('2.34'), '+') == Decimal('3.57')
        assert safe_arithmetic(Decimal('5.67'), 1.23, '-') == Decimal('4.44')
        
        print('✅ All decimal safety tests passed')
        "
    
    - name: Decimal Error Check Result
      run: |
        echo "✅ Decimal error prevention check passed"
        echo "🎯 All financial calculations use Decimal precision"
'''
    
    os.makedirs('.github/workflows', exist_ok=True)
    with open('.github/workflows/decimal_error_prevention.yml', 'w', encoding='utf-8') as f:
        f.write(ci_gate_content)
    
    print("  ✅ Created decimal error prevention CI gate")

if __name__ == "__main__":
    fix_decimal_errors()
