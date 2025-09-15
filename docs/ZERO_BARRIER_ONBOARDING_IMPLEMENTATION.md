# Zero-Barrier Onboarding Implementation

## Overview
This document describes the implementation of **zero-barrier onboarding** that allows **any wallet size (even $5)** to run the triangle bot safely by working around Hyperliquid's $10 minimum notional rule.

## 🎯 Problem Statement

Hyperliquid's node **rejects every TP/SL trigger whose size × price < $10** ("`MinTradeNtl`" error). We can't literally remove the minimum — we must *work around* it in three ways:

## ✅ Solution: Three-Tier System

### **Tier A: $10+ Notional** 
- **Wallet equity**: `size·px ≥ $10`
- **Bot behavior**: Posts native reduce-only TP/SL just like now
- **Rationale**: Meets chain rule ⇒ safest

### **Tier B: $2 – $10 Notional**
- **Wallet equity**: `size·px < $10` **but can reach $10** by upping leverage (≤ 10×)
- **Bot behavior**: 
  1. Compute size the usual way
  2. If `notional < $10` raise leverage until `entryNotional ≥ $10`
  3. Post native TP/SL
- **Rationale**: Keeps on-chain safety, marries tiny equity with small controlled leverage

### **Tier C: < $2 Notional**
- **Wallet equity**: Even @10× leverage `entryNotional < $10`
- **Bot behavior**:
  1. Enter with the *smallest* tick-sized order (often 1 contract)
  2. **Simulate** TP/SL off-chain: background coroutine polls mark-price every 2s
  3. If `tp` or `sl` level is crossed, fire market close
- **Rationale**: Nothing smaller is allowed on-chain; synthetic leg is the only escape hatch

## 🔧 Implementation Details

### 1. Safe Position Size Helper

```python
def safe_position_size(self, equity_usd: float, entry_px: float, sl_px: float, risk_frac: float, max_leverage: int = 10) -> tuple[float, int]:
    """
    ZERO-BARRIER ONBOARDING: Returns (size_decimal, leverage_int) that
    – risks <= risk_frac * equity
    – respects Hyperliquid's $10 trigger rule if possible
    – otherwise falls back to "synthetic mode"
    
    Tiers:
    A · $10+ : size·px ≥ $10 → native TP/SL
    B · $2-$10 : size·px < $10 but can reach $10 with leverage (≤ 10×)
    C · < $2 : Even @10× leverage entryNotional < $10 → synthetic TP/SL
    """
```

### 2. Synthetic Guardian for Tier C

```python
async def activate_offchain_guardian(self, tp_px: float, sl_px: float, position_size: float, is_long: bool):
    """
    ZERO-BARRIER TIER C: Synthetic TP/SL guardian for tiny wallets
    Polls mark-price every 2s and fires market close when TP/SL levels are crossed
    """
```

### 3. Trading Loop Integration

```python
# ZERO-BARRIER ONBOARDING: Calculate position size with tiered approach
size, leverage = self.safe_position_size(
    equity_usd=free_collateral,
    entry_px=entry_price,
    sl_px=sl_price,
    risk_frac=risk_settings['risk_per_trade'],
    max_leverage=10  # Safe ceiling for tiny wallets
)

# Determine tier and execution method
notional = size * entry_price
if leverage == 0:
    # Tier C: Synthetic TP/SL required
    self.logger.info(f"🛡️ Tier C: Synthetic TP/SL (notional=${notional:.2f} < $10)")
    position_size = size
    use_synthetic = True
elif leverage == 1:
    # Tier A: Native TP/SL
    self.logger.info(f"✅ Tier A: Native TP/SL (notional=${notional:.2f} ≥ $10)")
    position_size = size
    use_synthetic = False
else:
    # Tier B: Leveraged native TP/SL
    self.logger.info(f"⚡ Tier B: Leveraged TP/SL (notional=${notional * leverage:.2f} ≥ $10, leverage={leverage}×)")
    position_size = size
    use_synthetic = False
```

## 📊 Configuration for $100 Starter

```python
# BotConfig class additions
risk_per_trade = 0.03          # 3% of equity
max_leverage = 10              # Safe ceiling on tiny wallets
synthetic_guardian_ms = 2000   # Poll mark-price every 2s
min_native_notional = 10       # Keep for clarity
```

## 🎯 Why 3% Risk?

On $100 that's $3 risk; with ATR-based SL ≈ 1.5¢ the bot sizes to **200 XRP**
→ native notional ≈ $120 so Tier A logic is used straight away.

If the user instead begins with **$5**:
- risk$ = $0.15 → raw size ≈ 10 XRP → notional $6 → even at 10× lev ⇒ $60 (> $10) → Tier B (native, 10× lev)
- If the coin price soars and even 1 contract is < $10 notional (rare), Tier C kicks in and synthetic exits keep the user protected.

## 🛡️ Safety Features

### 1. **Dead-Man Switch**
- Orders are automatically cancelled after 60 seconds
- Prevents orphaned orders from tiny wallets

### 2. **Funding Rate Skip**
- Skips openings when minutes_to_next_funding() < 8
- Protects tiny positions from funding rate volatility

### 3. **Tick Alignment**
- Every price properly aligned to tick size
- Ensures orders are accepted by Hyperliquid

### 4. **Synthetic Guardian**
- Background coroutine polls mark-price every 2s
- Fires market close when TP/SL levels are crossed
- Always clears—even with $2 notional

## 📈 Expected Performance by Wallet Size

### **$100 Wallet (Tier A)**
- Risk per trade: $3 (3% of $100)
- Position size: ~200 XRP
- Notional value: ~$120 (meets $10 minimum)
- Execution: Native TP/SL triggers

### **$50 Wallet (Tier B)**
- Risk per trade: $1.50 (3% of $50)
- Position size: ~100 XRP
- Notional value: ~$60 (requires 2× leverage)
- Execution: Leveraged native TP/SL

### **$10 Wallet (Tier C)**
- Risk per trade: $0.30 (3% of $10)
- Position size: ~20 XRP
- Notional value: ~$12 (requires 5× leverage)
- Execution: Leveraged native TP/SL

### **$5 Wallet (Tier C)**
- Risk per trade: $0.15 (3% of $5)
- Position size: ~10 XRP
- Notional value: ~$6 (requires 10× leverage)
- Execution: Leveraged native TP/SL

### **$2 Wallet (Tier C)**
- Risk per trade: $0.06 (3% of $2)
- Position size: ~4 XRP
- Notional value: ~$2.40 (even 10× leverage < $10)
- Execution: Synthetic TP/SL guardian

## 🚀 Usage Examples

### **Example 1: $100 Wallet**
```python
# Input
equity = 100.0
entry_px = 0.60
sl_px = 0.585
risk_frac = 0.03

# Output
size, leverage = safe_position_size(equity, entry_px, sl_px, risk_frac)
# Returns: (200.0, 1)  # Tier A: Native TP/SL
```

### **Example 2: $5 Wallet**
```python
# Input
equity = 5.0
entry_px = 0.60
sl_px = 0.585
risk_frac = 0.03

# Output
size, leverage = safe_position_size(equity, entry_px, sl_px, risk_frac)
# Returns: (10.0, 10)  # Tier B: Leveraged TP/SL
```

### **Example 3: $1 Wallet**
```python
# Input
equity = 1.0
entry_px = 0.60
sl_px = 0.585
risk_frac = 0.03

# Output
size, leverage = safe_position_size(equity, entry_px, sl_px, risk_frac)
# Returns: (2.0, 0)  # Tier C: Synthetic TP/SL
```

## ⚠️ Important Notes

1. **No Chain Rule Violation**: We do **not** touch Hyperliquid's on-chain rule
2. **Leverage Safety**: Leverage is capped at 10× for tiny wallets
3. **Synthetic Protection**: Off-chain watchdog ensures any wallet can run safely
4. **Market Orders**: Synthetic exits use market orders that always clear
5. **Real-Time Monitoring**: Guardian polls every 2 seconds for immediate response

## 🎯 Bottom Line

You do **not** touch Hyperliquid's on-chain rule; you simply:

1. **Boost leverage** (still capped) when a tiny wallet can *just* meet $10
2. **Fall back to off-chain watchdog** when leverage can't save the day
3. **Flash out of trades** at intended TP/SL—so any wallet, even $1, can run the bot without invisible risk

---

**Status**: ✅ **FULLY IMPLEMENTED** - Zero-barrier onboarding system ready for any wallet size! 