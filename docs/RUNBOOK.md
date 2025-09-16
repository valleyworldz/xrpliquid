# 📋 XRPLiquid Operational Runbook

## Daily Operations

### 1. System Startup
```bash
# Check system status
python -c "from src.core.engines.ultra_efficient_xrp_system import UltraEfficientXRPSystem; print('System ready')"

# Start data capture
python src/data_capture/enhanced_tick_capture.py &

# Start trading system
python run_bot.py
```

### 2. Daily Reconciliation
```bash
# Run daily reconciliation
python src/core/accounting/daily_reconciliation.py

# Check reconciliation status
cat reports/reconciliation/exchange_vs_ledger_$(date +%Y-%m-%d).json
```

### 3. Risk Monitoring
```bash
# Check VaR/ES
cat reports/risk/var_es.json

# Check kill-switch status
cat reports/risk/hysteresis_state.json
```

## Emergency Procedures

### 1. Kill-Switch Activation
If kill-switch is triggered:
1. Check `reports/risk_events/risk_events.jsonl` for details
2. Review `reports/risk/hysteresis_state.json` for status
3. Wait for cooldown period before manual intervention
4. Monitor recovery stages in hysteresis manager

### 2. System Recovery
```bash
# Check system health
python scripts/check_system_health.py

# Restart components if needed
pkill -f "enhanced_tick_capture"
python src/data_capture/enhanced_tick_capture.py &
```

### 3. Data Recovery
```bash
# Check data integrity
python scripts/verify_data_integrity.py

# Restore from backup if needed
python scripts/restore_from_backup.py
```

## Monitoring

### 1. Key Metrics
- P95 latency: < 100ms
- Sharpe ratio: > 1.0
- Max drawdown: < 5%
- Maker ratio: > 60%

### 2. Alerts
- Kill-switch activation
- High latency (> 200ms)
- Large drawdown (> 3%)
- Data capture failures

### 3. Health Checks
```bash
# System health
curl http://localhost:8080/healthz

# Data capture health
curl http://localhost:8080/data_health

# Trading health
curl http://localhost:8080/trading_health
```

## Maintenance

### 1. Daily Tasks
- [ ] Check system status
- [ ] Review reconciliation reports
- [ ] Monitor risk metrics
- [ ] Check data capture

### 2. Weekly Tasks
- [ ] Review performance reports
- [ ] Update risk parameters
- [ ] Clean up old data
- [ ] Security scan

### 3. Monthly Tasks
- [ ] Full system backup
- [ ] Performance review
- [ ] Security audit
- [ ] Documentation update

## Troubleshooting

### 1. Common Issues

#### High Latency
- Check network connectivity
- Review system resources
- Check for memory leaks
- Review order queue

#### Data Capture Issues
- Check WebSocket connection
- Review data storage
- Check disk space
- Verify data format

#### Trading Issues
- Check exchange connectivity
- Review order status
- Check risk limits
- Verify position sizes

### 2. Log Analysis
```bash
# Check system logs
tail -f logs/system.log

# Check trading logs
tail -f logs/trading.log

# Check error logs
tail -f logs/error.log
```

## Contact Information

### 1. Escalation
- Level 1: System monitoring
- Level 2: Technical support
- Level 3: Development team

### 2. Emergency Contacts
- On-call engineer: [Contact Info]
- System administrator: [Contact Info]
- Risk manager: [Contact Info]

## Appendix

### 1. Configuration Files
- `config/sizing_by_regime.json`: Position sizing
- `config/risk_parameters.json`: Risk limits
- `config/trading_parameters.json`: Trading parameters

### 2. Key Scripts
- `scripts/check_system_health.py`: Health checks
- `scripts/verify_data_integrity.py`: Data verification
- `scripts/backup_system.py`: System backup

### 3. Important URLs
- System dashboard: `reports/executive_dashboard.html`
- Performance reports: `reports/tearsheets/`
- Risk reports: `reports/risk/`