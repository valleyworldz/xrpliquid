"""
Apply Decimal Fix - Systematic replacement of float() with Decimal in main_bot.py
"""

import re
import os
import shutil
from pathlib import Path

def apply_decimal_fix_to_file(file_path: str, backup: bool = True):
    """
    Apply decimal fix to a Python file by replacing float() with safe_float()
    """
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Create backup
    if backup:
        backup_path = f"{file_path}.backup"
        shutil.copy2(file_path, backup_path)
        print(f"📄 Created backup: {backup_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    replacements = 0
    
    # Add import at the top if not present
    if "from src.core.utils.decimal_boundary_guard import safe_float" not in content:
        # Find the first import line
        lines = content.split('\n')
        import_line = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_line = i
                break
        
        if import_line >= 0:
            lines.insert(import_line, "from src.core.utils.decimal_boundary_guard import safe_float")
            content = '\n'.join(lines)
            replacements += 1
            print(f"✅ Added import for safe_float")
    
    # Pattern 1: float(value) -> safe_float(value) - but only for simple cases
    # We'll be more selective to avoid breaking complex expressions
    
    # Simple float() calls with single variables
    pattern1 = r'float\(([a-zA-Z_][a-zA-Z0-9_]*)\)'
    matches1 = re.findall(pattern1, content)
    
    for match in matches1:
        # Skip if it's already a safe_float call
        if 'safe_float(' in match:
            continue
        
        # Replace float(match) with safe_float(match)
        old_pattern = f'float({match})'
        new_pattern = f'safe_float({match})'
        content = content.replace(old_pattern, new_pattern)
        replacements += 1
        print(f"✅ Replaced: {old_pattern} -> {new_pattern}")
    
    # Pattern 2: float(price_data['field']) -> safe_float(price_data['field'])
    pattern2 = r'float\(([^)]+\[\'[^\']+\'\][^)]*)\)'
    matches2 = re.findall(pattern2, content)
    
    for match in matches2:
        old_pattern = f'float({match})'
        new_pattern = f'safe_float({match})'
        content = content.replace(old_pattern, new_pattern)
        replacements += 1
        print(f"✅ Replaced: {old_pattern} -> {new_pattern}")
    
    # Pattern 3: float(position.get('field', 0)) -> safe_float(position.get('field', 0))
    pattern3 = r'float\(([^)]+\.get\([^)]+\)[^)]*)\)'
    matches3 = re.findall(pattern3, content)
    
    for match in matches3:
        old_pattern = f'float({match})'
        new_pattern = f'safe_float({match})'
        content = content.replace(old_pattern, new_pattern)
        replacements += 1
        print(f"✅ Replaced: {old_pattern} -> {new_pattern}")
    
    # Pattern 4: float(account_status.get('field', 0)) -> safe_float(account_status.get('field', 0))
    pattern4 = r'float\(([^)]+\.get\([^)]+\)[^)]*)\)'
    matches4 = re.findall(pattern4, content)
    
    for match in matches4:
        old_pattern = f'float({match})'
        new_pattern = f'safe_float({match})'
        content = content.replace(old_pattern, new_pattern)
        replacements += 1
        print(f"✅ Replaced: {old_pattern} -> {new_pattern}")
    
    # Write the modified content back
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ File updated: {file_path}")
        print(f"📊 Total replacements: {replacements}")
        return True
    else:
        print(f"ℹ️ No changes needed: {file_path}")
        return False

def main():
    """Main function"""
    print("🔢 Apply Decimal Fix")
    print("=" * 50)
    
    # Apply fix to main_bot.py
    main_bot_path = "src/core/main_bot.py"
    if os.path.exists(main_bot_path):
        print(f"🎯 Applying decimal fix to main_bot.py...")
        apply_decimal_fix_to_file(main_bot_path)
    else:
        print(f"❌ main_bot.py not found at {main_bot_path}")
    
    print(f"\n✅ Decimal fix application complete!")

if __name__ == "__main__":
    main()
