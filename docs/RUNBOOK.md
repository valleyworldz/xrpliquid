# RUNBOOK.md

## Human-in-the-Loop Safeguards

- **Fat-finger limits:** Orders > $50k notional blocked.
- **Kill switch:** Run `python run_bot.py --emergency-mode` to halt all trading.
- **Decision trees:**  

  - If WS/API desync → fail closed, reconnect.  
  - If VaR > 2% intraday → auto-scale down positions.  
  - If funding API mismatch → trigger fallback source.  

## Emergency Procedures
- Contact ops lead immediately.
- Run `scripts/failover.py` to swap to backup venue.
- Validate ledger reconciliation via `reports/exchange_vs_ledger.json`.