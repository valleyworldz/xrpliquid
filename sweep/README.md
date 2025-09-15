# Perp→Spot Profit Sweeping Engine v1.0

Automatically moves realized USDC profits from Perps to Spot on Hyperliquid with safety guards, cooldowns, and comprehensive observability.

## 🎯 Executive Summary

**Why**: Reduce liquidation risk, de-risk realized gains, smooth PnL distribution.

**What**: A sweep engine that, per loop, checks gates and when safe calls `usdClassTransfer(toPerp=false)` to move USDC Perp → Spot.

**How**: Equity-adaptive thresholds, staleness + cooldown + blackout gates, in-position post-buffer projection, vol-aware accumulator, funding-impact guard, Prometheus metrics.

**Outcome**: A reversible, observable feature that preserves trading continuity while locking profits.

## 🚀 Quick Start

### Enable Sweep Engine

Add the `--profit-sweep` flag when starting the bot:

```bash
python newbotcode.py --profit-sweep
```

### Environment Configuration

Configure via environment variables:

```bash
# Enable/disable (overridden by --profit-sweep flag)
export SWEEP_ENABLED=true

# Thresholds
export SWEEP_MIN_SWEEP_USDC=20
export SWEEP_EQUITY_TRIGGER_PCT=0.05

# Safety guards  
export SWEEP_INPOS_MIN_BUFFER_BPS=3000
export SWEEP_INPOS_POST_FLOOR_BPS=2000

# Timing
export SWEEP_COOLDOWN_S=1800  # 30 minutes
export SWEEP_JITTER_S=120     # ±2 minutes
```

### Test the CLI

```bash
# Test with mock data
python -m sweep.cli --dry-run --verbose

# Force a sweep (ignoring cooldowns)
python -m sweep.cli --force-sweep --dry-run
```

## 📦 Package Structure

```
sweep/
├── __init__.py          # Main exports
├── config.py            # Environment-driven configuration
├── state.py             # Persistent state management
├── metrics.py           # Prometheus metrics
├── engine.py            # Core sweep logic
├── transfer.py          # Hyperliquid API integration
├── funding.py           # Funding rate utilities
├── volatility.py        # Volatility analysis
├── cli.py               # Standalone CLI tool
└── README.md            # This file
```

## 🛡️ Safety Guardrails

### Adaptive Triggers
- **Trigger**: `max($20, 5% of equity)`
- **Reserve**: Always keep $150+ on perps
- **Cooldown**: 30 minutes ± 2 minute jitter

### Position Mode Guards
When in position, additional safety checks:
- **Pre-buffer**: ≥ 3000 bps from liquidation
- **Post-buffer**: ≥ 2000 bps after sweep
- **Max sweep**: ≤ 33% of withdrawable

### Funding Blackouts
- Skip sweeps 10 minutes before funding
- Extended to 15 minutes if high funding impact (≥20 bps)

### Accumulator System
- Batches small profits until trigger threshold
- Vol-aware cap: `min($200, 5% equity) × vol_multiplier`
- Flushes when flat with headroom

## 📊 Prometheus Metrics

### Counters
- `xrpbot_sweep_success_total` - Successful sweeps
- `xrpbot_sweep_fail_total` - Failed sweeps  
- `xrpbot_sweep_skipped_total{reason}` - Skipped sweeps by reason

### Gauges
- `xrpbot_sweep_last_amount` - Last sweep amount (USDC)
- `xrpbot_sweep_equity_usdc` - Current equity
- `xrpbot_sweep_withdrawable_usdc` - Available to withdraw
- `xrpbot_sweep_pending_usdc` - Accumulator pending
- `xrpbot_sweep_cooldown_remaining_s` - Cooldown remaining
- `xrpbot_sweep_post_buffer_bps` - Post-sweep liquidation buffer

### Histograms  
- `xrpbot_sweep_amount_histogram` - Distribution of sweep amounts
- `xrpbot_post_buffer_bps_histogram` - Distribution of post-buffers

## 🔧 Configuration Reference

### Core Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `SWEEP_ENABLED` | `false` | Enable/disable sweep engine |
| `SWEEP_MIN_SWEEP_USDC` | `20` | Minimum sweep amount |
| `SWEEP_EQUITY_TRIGGER_PCT` | `0.05` | 5% of equity trigger |
| `SWEEP_MIN_RESERVE_USDC` | `150` | Reserve kept on perps |

### Position Safety
| Variable | Default | Description |
|----------|---------|-------------|
| `SWEEP_INPOS_MIN_BUFFER_BPS` | `3000` | Min pre-sweep buffer |
| `SWEEP_INPOS_POST_FLOOR_BPS` | `2000` | Min post-sweep buffer |
| `SWEEP_INPOS_MAX_SWEEP_PCT` | `0.33` | Max 33% of withdrawable |

### Timing & Cooldowns
| Variable | Default | Description |
|----------|---------|-------------|
| `SWEEP_COOLDOWN_S` | `1800` | Base cooldown (30 min) |
| `SWEEP_JITTER_S` | `120` | Jitter range (±2 min) |
| `SWEEP_MAX_STALENESS_S` | `60` | Max price staleness |

### Funding Protection
| Variable | Default | Description |
|----------|---------|-------------|
| `SWEEP_FUNDING_AWARE` | `true` | Enable funding blackouts |
| `SWEEP_FUNDING_BLACKOUT_MIN` | `10` | Base blackout minutes |
| `SWEEP_FUNDING_IMPACT_GUARD_BPS` | `20` | Impact threshold for extended blackout |
| `SWEEP_FUNDING_BLACKOUT_HI_MIN` | `15` | Extended blackout minutes |

### Accumulator
| Variable | Default | Description |
|----------|---------|-------------|
| `SWEEP_ACCUM_ENABLED` | `true` | Enable accumulator |
| `SWEEP_MAX_PENDING_CAP_USD` | `200` | Max accumulator cap |
| `SWEEP_MAX_PENDING_PCT_EQUITY` | `0.05` | Cap as % of equity |
| `SWEEP_VOL_HIGH` | `2.0` | High volatility threshold |
| `SWEEP_VOL_MULTIPLIER_HIGH` | `1.5` | Vol multiplier for caps |

## 🔍 Monitoring & Observability

### Skip Reasons
- `disabled` - Sweep engine disabled
- `cooldown` - Within cooldown period
- `funding_blackout` - Near funding time
- `funding_impact_guard` - High funding impact
- `too_small` - Below minimum threshold
- `no_headroom_flat` - Insufficient funds (flat)
- `no_headroom_inpos` - Insufficient funds (in position)
- `low_pre_buffer` - Pre-sweep buffer too low
- `post_floor_violation` - Post-sweep buffer too low
- `dedupe` - Duplicate request

### Grafana Dashboards

**Sweep Activity Panel**:
```promql
increase(xrpbot_sweep_success_total[1h])
increase(xrpbot_sweep_fail_total[1h])  
increase(xrpbot_sweep_skipped_total[1h]) by (reason)
```

**Safety Monitoring**:
```promql
xrpbot_sweep_post_buffer_bps  # Should stay > 2000
xrpbot_sweep_equity_usdc
xrpbot_sweep_withdrawable_usdc
```

### Alerts
- **Critical**: `xrpbot_sweep_post_buffer_bps < 2200` 
- **Warning**: `increase(xrpbot_sweep_fail_total[1h]) > 2`
- **Info**: No sweeps in 24h with equity swings > 10%

## 🧪 Testing & Validation

### Unit Tests
```bash
python test_sweep_integration.py
```

### Manual Testing
```bash
# Test configuration
python -c "from sweep import SweepCfg; print(SweepCfg())"

# Test state persistence  
python -c "from sweep import SweepState; s = SweepState('test.json'); s.save(); print('OK')"

# Test CLI
python -m sweep.cli --dry-run --verbose
```

### Integration Testing
```bash
# Enable sweep with bot
python newbotcode.py --profit-sweep --prometheus

# Monitor metrics
curl http://localhost:8000/metrics | grep sweep
```

## 🚨 Operational Runbook

### Common Operations

**Speed up sweeps** (15-minute cooldown):
```bash
export SWEEP_COOLDOWN_S=900
```

**Tighter safety margins**:
```bash
export SWEEP_INPOS_MIN_BUFFER_BPS=3500
export SWEEP_INPOS_POST_FLOOR_BPS=2500
```

**Lower trigger for small accounts**:
```bash
export SWEEP_EQUITY_TRIGGER_PCT=0.03
```

**Disable during funding stress**:
```bash
export SWEEP_FUNDING_BLACKOUT_MIN=15
export SWEEP_FUNDING_IMPACT_GUARD_BPS=15
```

**Emergency disable**:
```bash
export SWEEP_ENABLED=false
# Or restart bot without --profit-sweep flag
```

### Troubleshooting

**No sweeps happening**:
1. Check `xrpbot_sweep_skipped_total` metrics for reasons
2. Verify equity > trigger threshold  
3. Check cooldown remaining
4. Verify not in funding blackout

**Sweeps too frequent**:
1. Increase `SWEEP_COOLDOWN_S`
2. Increase `SWEEP_EQUITY_TRIGGER_PCT`
3. Reduce `SWEEP_FLAT_SWEEP_PCT`

**Post-buffer violations**:
1. Check position size vs equity ratio
2. Increase `SWEEP_INPOS_POST_FLOOR_BPS`
3. Reduce `SWEEP_INPOS_MAX_SWEEP_PCT`

## 🔗 API Integration

### Hyperliquid Integration
The engine uses `exchange.usd_class_transfer(amount, to_perp=False)` for transfers.

**Response handling**:
- Success: `{"status": "ok"}` or no error field
- Failure: Exception or `{"error": "..."}` 

### Bot Integration
The sweep engine integrates into the main trading loop after account status updates:

```python
# In main trading loop
if self.sweep_enabled:
    sweep_result = maybe_sweep_to_spot(
        exchange=self.resilient_exchange,
        state=self.sweep_state,
        cfg=self.sweep_cfg,
        user_state=user_state,
        pos=position_data,
        mark_px=current_price,
        vol_ratio=volatility_score,
        next_hour_funding_rate=funding_rate,
        position_notional=notional_value
    )
```

## 🎛️ Advanced Configuration

### Volatility-Aware Behavior
When volatility ratio ≥ 2.0:
- Accumulator cap increases by 1.5x
- Extra jitter added to cooldowns
- More conservative in-position sweeps

### Chain Support
Currently supports:
- `Mainnet` (default)
- `Testnet` 

Set via `HL_CHAIN` environment variable.

### State Persistence
State is persisted to `sweep.state.json` (configurable via `SWEEP_ACCUM_FILE`):
```json
{
  "last_sweep_ts": 1234567890.0,
  "pending_accum": 125.50,
  "last_withdrawable": 1500.0,
  "last_nonce": 1692847200000
}
```

## 🚀 Production Deployment

### Pre-deployment Checklist
- [ ] Set `SWEEP_ENABLED=true` in environment
- [ ] Configure appropriate thresholds for account size
- [ ] Verify Prometheus metrics endpoint
- [ ] Test with `--dry-run` first
- [ ] Monitor post-buffer metrics closely
- [ ] Set up Grafana alerts

### Rollout Strategy
1. **Canary**: Deploy to 1 instance, monitor 24h
2. **Validation**: Verify no post-buffer violations
3. **Gradual**: 25% → 50% → 100% of instances
4. **Monitor**: Success/fail rates, skip reason distribution

### Success Criteria
- Zero post-buffer floor violations
- Sweep success rate > 95%
- Reasonable skip reason distribution
- No liquidations related to sweeps

---

## 📞 Support

For issues or questions:
1. Check Prometheus metrics for skip reasons
2. Review logs for error details
3. Test with CLI dry-run mode
4. Consult operational runbook above

**Version**: 1.0.0  
**Last Updated**: August 2025
