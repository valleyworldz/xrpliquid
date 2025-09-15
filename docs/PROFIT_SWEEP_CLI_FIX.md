# CLI Argument Conflict Resolution ✅

## Issue Identified
The original implementation used `--sweep` as the CLI flag, but this conflicted with an existing argument in the bot:
- **Existing**: `--sweep` for "Run parameter sweep for ATR and ensemble thresholds in backtest"
- **New**: `--sweep` for "Enable perp→spot profit sweeping"

This caused an `ArgumentError: argument --sweep: conflicting option string: --sweep` error.

## Solution Applied
Changed the profit sweeping CLI flag from `--sweep` to `--profit-sweep` to avoid conflicts.

### Changes Made:

1. **newbotcode.py**: Updated argument definition
   ```python
   # Before
   parser.add_argument("--sweep", action="store_true", help="Enable perp→spot profit sweeping...")
   
   # After  
   parser.add_argument("--profit-sweep", action="store_true", help="Enable perp->spot profit sweeping...")
   ```

2. **newbotcode.py**: Updated initialization code
   ```python
   # Before
   if bool(getattr(args, 'sweep', False)):
   
   # After
   if bool(getattr(args, 'profit_sweep', False)):
   ```

3. **sweep/README.md**: Updated all documentation references
   ```bash
   # Before
   python newbotcode.py --sweep
   
   # After
   python newbotcode.py --profit-sweep
   ```

4. **Unicode Character Fix**: Also replaced `→` with `->` to prevent encoding issues on Windows.

## Verification
✅ Argument parsing now works without conflicts  
✅ Bot imports successfully with `--profit-sweep` flag  
✅ No Unicode encoding errors  
✅ Existing `--sweep` functionality for backtesting preserved  

## Updated Usage

**Enable profit sweeping:**
```bash
python newbotcode.py --profit-sweep
```

**With other flags:**
```bash
python newbotcode.py --profit-sweep --prometheus --verbose
```

**Environment variable override still works:**
```bash
export SWEEP_ENABLED=true
python newbotcode.py  # No flag needed if env var set
```

## Status: RESOLVED ✅

The profit sweep engine is now fully functional with the `--profit-sweep` CLI flag.
