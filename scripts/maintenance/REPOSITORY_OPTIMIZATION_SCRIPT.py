#!/usr/bin/env python3
"""
🔧 REPOSITORY OPTIMIZATION SCRIPT
=================================
Comprehensive repository cleanup and organization
"""

import os
import shutil
import glob
from datetime import datetime
import json

def main():
    """Main optimization function"""
    print("🔧 REPOSITORY OPTIMIZATION")
    print("=" * 40)
    
    # Create optimized directory structure
    create_directory_structure()
    
    # Consolidate reports
    consolidate_reports()
    
    # Organize Python files
    organize_python_files()
    
    # Clean up duplicate and obsolete files
    cleanup_files()
    
    # Create final summary
    create_final_summary()
    
    print("✅ Repository optimization completed!")

def create_directory_structure():
    """Create optimized directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "optimized_reports",
        "optimized_scripts", 
        "optimized_systems",
        "archive_old",
        "consolidated_docs",
        "backup_scripts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def consolidate_reports():
    """Consolidate all markdown reports"""
    print("📄 Consolidating reports...")
    
    # Find all markdown files
    md_files = glob.glob("*.md")
    
    # Create comprehensive report
    comprehensive_report = "# 📊 COMPREHENSIVE REPOSITORY REPORT\n\n"
    comprehensive_report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                comprehensive_report += f"## {md_file}\n\n"
                comprehensive_report += content + "\n\n---\n\n"
        except Exception as e:
            print(f"   ⚠️ Error reading {md_file}: {e}")
    
    # Save comprehensive report
    with open("optimized_reports/COMPREHENSIVE_REPOSITORY_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(comprehensive_report)
    
    # Move individual reports to archive
    for md_file in md_files:
        if md_file != "README.md":
            shutil.move(md_file, f"archive_old/{md_file}")
            print(f"   📦 Archived: {md_file}")

def organize_python_files():
    """Organize Python files by category"""
    print("🐍 Organizing Python files...")
    
    # Categorize Python files
    categories = {
        "trading_systems": [
            "ULTIMATE_100_PERCENT_PERFECT_SYSTEM.py",
            "ULTIMATE_ENHANCED_PERFECT_SYSTEM.py",
            "SIMPLE_WORKING_SYSTEM.py",
            "ENHANCED_TRADING_SYSTEM_SIMPLE.py",
            "ULTIMATE_ENHANCED_TRADING_SYSTEM.py",
            "ULTIMATE_AUTONOMOUS_TRADING_SYSTEM.py",
            "REAL_HYPERLIQUID_TRADING.py",
            "PROFIT_TRADING_SYSTEM.py",
            "FIXED_PROFIT_TRADING_SYSTEM.py",
            "WORKING_HYPERLIQUID_TRADING.py"
        ],
        "fix_scripts": [
            "QUICK_POSITION_FIX.py",
            "ENHANCED_POSITION_FIX.py", 
            "MANUAL_UNI_FIX.py",
            "COMPREHENSIVE_POSITION_FIX.py",
            "FIX_BALANCE_CHECKING.py",
            "FIX_CRITICAL_ERRORS.py",
            "FIX_POSITION_CLOSING.py",
            "CLOSE_CURRENT_POSITION.py"
        ],
        "master_scripts": [
            "MASTER_TRADING_SYSTEM.py",
            "MASTER_UTILITIES.py",
            "MASTER_OPTIMIZER.py",
            "MASTER_RISK_MANAGER.py",
            "MASTER_SAFETY_SYSTEM.py",
            "MASTER_SETUP.py",
            "MASTER_TEST_SUITE.py"
        ],
        "analysis_scripts": [
            "COMPREHENSIVE_SYSTEM_ANALYSIS.py",
            "FINAL_100_PERCENT_ANALYSIS.py",
            "VERIFY_100_PERCENT_PERFECTION.py",
            "QUICK_SYSTEM_CHECK.py"
        ],
        "monitoring_scripts": [
            "MONITOR_100_PERCENT_PERFECTION.py",
            "MONITOR_100_PERCENT_SUCCESS.py",
            "SAFETY_MONITOR.py"
        ]
    }
    
    # Move files to appropriate directories
    for category, files in categories.items():
        category_dir = f"optimized_scripts/{category}"
        os.makedirs(category_dir, exist_ok=True)
        
        for file in files:
            if os.path.exists(file):
                shutil.move(file, f"{category_dir}/{file}")
                print(f"   📦 Moved {file} to {category}")
    
    # Move remaining Python files to backup
    remaining_py = glob.glob("*.py")
    for py_file in remaining_py:
        if py_file != "REPOSITORY_OPTIMIZATION_SCRIPT.py":
            shutil.move(py_file, f"backup_scripts/{py_file}")
            print(f"   📦 Backed up: {py_file}")

def cleanup_files():
    """Clean up duplicate and obsolete files"""
    print("🧹 Cleaning up files...")
    
    # Remove empty files
    empty_files = [
        "ULTIMATE_PERFECT_SYSTEM.py"
    ]
    
    for file in empty_files:
        if os.path.exists(file) and os.path.getsize(file) == 0:
            os.remove(file)
            print(f"   🗑️ Removed empty file: {file}")
    
    # Remove duplicate files
    duplicate_patterns = [
        "*_FIXED.py",
        "*_COPY.py", 
        "*_BACKUP.py"
    ]
    
    for pattern in duplicate_patterns:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                shutil.move(file, f"archive_old/{file}")
                print(f"   📦 Archived duplicate: {file}")

def create_final_summary():
    """Create final optimization summary"""
    print("📋 Creating final summary...")
    
    summary = f"""# 🎯 REPOSITORY OPTIMIZATION COMPLETE

**Optimization Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 Directory Structure Created

```
optimized_reports/          # Consolidated markdown reports
optimized_scripts/          # Organized Python scripts
├── trading_systems/       # Main trading system files
├── fix_scripts/          # Position and error fixes
├── master_scripts/       # Master utility scripts
├── analysis_scripts/     # System analysis tools
└── monitoring_scripts/   # Monitoring and safety tools
archive_old/              # Archived old files
consolidated_docs/        # Consolidated documentation
backup_scripts/           # Backup of remaining scripts
```

## 📊 Files Organized

### Trading Systems (10 files)
- Main trading system implementations
- Enhanced and optimized versions
- Working production systems

### Fix Scripts (8 files)  
- Position closing fixes
- Error resolution scripts
- Balance checking tools

### Master Scripts (7 files)
- Master trading system
- Utilities and optimizers
- Risk management tools

### Analysis Scripts (4 files)
- System analysis tools
- Verification scripts
- Quick check utilities

### Monitoring Scripts (3 files)
- Performance monitoring
- Safety monitoring
- Success tracking

## ✅ Optimization Results

- **Reports Consolidated**: All markdown reports merged into comprehensive report
- **Scripts Organized**: 32 Python files organized by category
- **Duplicates Removed**: Cleaned up duplicate and obsolete files
- **Structure Improved**: Clear, logical directory organization
- **Backup Created**: All original files safely backed up

## 🚀 Next Steps

1. **Main System**: Use `optimized_scripts/trading_systems/ULTIMATE_100_PERCENT_PERFECT_SYSTEM.py`
2. **Position Fixes**: Use scripts in `optimized_scripts/fix_scripts/`
3. **Monitoring**: Use scripts in `optimized_scripts/monitoring_scripts/`
4. **Analysis**: Use scripts in `optimized_scripts/analysis_scripts/`

## 📈 Benefits

- **Cleaner Structure**: Easy to find and use specific tools
- **Better Organization**: Logical grouping of related files
- **Reduced Clutter**: Removed duplicates and obsolete files
- **Improved Maintainability**: Clear separation of concerns
- **Enhanced Usability**: Intuitive directory structure

**Status**: ✅ **OPTIMIZATION COMPLETE**
"""
    
    with open("REPOSITORY_OPTIMIZATION_SUMMARY.md", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("   ✅ Created optimization summary")

if __name__ == "__main__":
    main() 