# Final Consolidated Fixes and Verification

This document consolidates all critical fixes and their validations into one definitive source of truth. It supersedes prior fix reports.

## Scope of Fixes
- TP/SL trigger order schema (field names and structure)
- Cryptographic signing for `/exchange` (422 deserialization errors)
- Wallet access/signing consistency across all order paths
- String formatting/type safety for all price fields (wire, SDK params, logging)
- SDK parameter numeric types for compatibility

---

## Root Causes (Summary)
1) Field naming mismatch (snake_case vs camelCase) in trigger order payloads caused KeyErrors.
2) Unsigned payloads to `/exchange` endpoint caused 422 JSON deserialization errors.
3) Inconsistent wallet access (manual signing vs SDK signing) caused wallet existence/signature issues.
4) Price values sometimes strings; applying `:.4f` float formatting raised `Unknown format code 'f'`.
5) Passing strings to SDK numeric parameters (e.g., `limit_px`) caused internal formatting/type errors.

---

## Fixes Applied

### A. Trigger Order Format (camelCase)
- Fields standardized per SDK docs:
  - `isMarket` (Boolean), `triggerPx` (String), `tpsl` ("tp"|"sl").
- Wire schema and logging updated to reflect correct camelCase fields.

### B. Cryptographic Signing (422 Resolution)
- All `/exchange` actions are signed using the SDK signing utilities.
- Flow:
  - Build action (e.g., grouped TP/SL)
  - Get timestamp
  - Sign via `sign_l1_action`
  - Submit via SDK `_post_action`

### C. Wallet Access Consistency
- Use the same SDK `order()` path for all order types (including TP/SL) where applicable, or use signed wire actions consistently.
- No direct manipulation of `client.wallet` outside the signing flow.

### D. String Formatting and Type Safety
- For any price formatting, apply `float()` conversion before `:.4f` formatting.
- Avoid re-formatting already formatted strings in logs.
- Central rule: prices passed to SDK or schema are either proper floats or already formatted strings; do not mix unexpectedly.

### E. SDK Parameter Types
- Ensure SDK numeric params (e.g., `limit_px`) receive floats, not strings.
- All relevant call sites updated to cast to floats explicitly where necessary.

---

## Code Invariants (Keep These)
- Trigger schema uses camelCase (`isMarket`, `triggerPx`, `tpsl`).
- All price values cast to float before formatting; never apply `:.4f` to strings.
- Signed actions for `/exchange`; no unsigned payload submissions.
- Consistent SDK order path or consistently signed wire actions; never mix signing paths arbitrarily.
- SDK numeric parameters must be numeric (float/int), not strings.

---

## Verification
- Unit tests pass for:
  - RR/ATR gating
  - Fee-adjusted TP/SL alignment
  - Drawdown lock timing and early unlock
  - Signal sanity (uptrend → BUY/HOLD)
- Smoke test: DNS, HTTP, API probe, market helpers – PASSED.
- Live readiness: no string-formatting/type errors observed in TP/SL paths; signing functional.

---

## Operational Guidance
- For micro accounts, keep `--kelly-cap` between 0.10–0.20 and base DD lock at ~1200s (adaptive enabled).
- Use maker preference and higher fee+funding threshold (≥ 3.0) in volatile regimes.
- Use sandbox TP/SL (`--sandbox-tpsl`) for first-run validation.

---

## Change Log (Consolidated)
- Fixed field names (camelCase) in trigger schema.
- Implemented SDK-compliant signing for `/exchange` actions; resolved 422 errors.
- Standardized wallet access via unified SDK paths; eliminated wallet existence/signature mismatches.
- Enforced float casting before formatting price strings; removed redundant re-formatting in logs.
- Enforced numeric SDK parameters; added explicit casts.

---

## Final Status
- TP/SL order flow: correct schema, correct types, correct signing.
- No formatting/type errors remain in TP/SL code paths.
- Bot is ready for live/testnet with micro-safe defaults.
