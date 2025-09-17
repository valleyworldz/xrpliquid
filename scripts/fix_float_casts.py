"""
Fix Float Casts - Systematic replacement of float() with Decimal() in main_bot.py
"""

import re
import os
import shutil
from pathlib import Path

def fix_float_casts_in_file(file_path: str, backup: bool = True):
    """
    Fix float() casts in a Python file by replacing with Decimal conversions
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
    if "from src.core.utils.float_to_decimal_converter import safe_float" not in content:
        # Find the first import line
        lines = content.split('\n')
        import_line = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_line = i
                break
        
        if import_line >= 0:
            lines.insert(import_line, "from src.core.utils.float_to_decimal_converter import safe_float")
            content = '\n'.join(lines)
            replacements += 1
            print(f"✅ Added import for safe_float")
    
    # Pattern 1: float(value) -> safe_float(value)
    pattern1 = r'float\(([^)]+)\)'
    matches1 = re.findall(pattern1, content)
    
    for match in matches1:
        # Skip if it's already a safe_float call
        if 'safe_float(' in match:
            continue
        
        # Skip if it's a complex expression that might break
        if any(op in match for op in ['+', '-', '*', '/', '//', '%', '**']):
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

def fix_float_casts_in_directory(directory: str, pattern: str = "*.py"):
    """
    Fix float() casts in all Python files in a directory
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    python_files = list(directory_path.rglob(pattern))
    total_files = len(python_files)
    fixed_files = 0
    
    print(f"🔍 Found {total_files} Python files to process")
    
    for file_path in python_files:
        print(f"\n📄 Processing: {file_path}")
        if fix_float_casts_in_file(str(file_path)):
            fixed_files += 1
    
    print(f"\n📊 Summary:")
    print(f"  Total files: {total_files}")
    print(f"  Fixed files: {fixed_files}")
    print(f"  Unchanged files: {total_files - fixed_files}")

def main():
    """Main function"""
    print("🔢 Float Casts Fixer")
    print("=" * 50)
    
    # Fix main_bot.py specifically
    main_bot_path = "src/core/main_bot.py"
    if os.path.exists(main_bot_path):
        print(f"🎯 Fixing main_bot.py...")
        fix_float_casts_in_file(main_bot_path)
    else:
        print(f"❌ main_bot.py not found at {main_bot_path}")
    
    # Fix other critical files
    critical_files = [
        "src/core/api/hyperliquid_api.py",
        "src/core/execution/",
        "src/core/risk/",
        "src/core/pnl/"
    ]
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                print(f"\n🎯 Fixing {file_path}...")
                fix_float_casts_in_file(file_path)
            elif os.path.isdir(file_path):
                print(f"\n🎯 Fixing directory {file_path}...")
                fix_float_casts_in_directory(file_path)
        else:
            print(f"⚠️ Path not found: {file_path}")
    
    print(f"\n✅ Float casts fixing complete!")

if __name__ == "__main__":
    main()
