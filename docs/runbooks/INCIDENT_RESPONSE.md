# Incident Response Runbook — XRP Modular Trading Bot

## 1. Detect & Triage
- **Monitor:**
  - Prometheus metrics (e.g., trade count, error count, PnL)
  - Structured logs (JSON, look for ERROR/WARNING)
  - Alerts from CI/CD or monitoring tools
- **Triage:**
  - Is trading halted? Are losses abnormal? Is the API down?
  - Check `xrp_bot.log`, Prometheus `/metrics`, and CI status

## 2. Immediate Actions
- **Stop the bot:**
  - Native: `Ctrl+C` or `kill <pid>`
  - Docker: `docker-compose down`
- **Preserve logs:**
  - Copy `logs/`, `xrp_bot.log`, and any relevant system logs
- **Notify:**
  - SRE/lead dev, and (if needed) exchange support

## 3. Common Scenarios & Playbooks
- **API outage:**
  - Confirm with status page or other users
  - Retry after 5–15 minutes
  - If persistent, disable auto-trading and escalate
- **Exchange error (order rejected, balance mismatch):**
  - Check logs for error details
  - Validate API keys and account status
  - If funds at risk, withdraw to cold storage
- **Abnormal losses or runaway trading:**
  - Stop the bot immediately
  - Review last 50 trades and logs
  - Check for config or code changes
- **Config/secret leak:**
  - Rotate all API keys and secrets
  - Audit access logs and repository
  - Notify stakeholders

## 4. Recovery Steps
- Fix root cause (code, config, or infra)
- Restore from last known good config/backup
- Test in backtest or paper mode before resuming live trading
- Document the incident and actions taken

## 5. Postmortem & Improvement
- Hold a postmortem with SRE/lead dev
- Update runbooks and CI/CD as needed
- Add new alerts or tests if gaps were found
- Share lessons learned with the team

---
**Always prioritize capital preservation and clear communication.** 