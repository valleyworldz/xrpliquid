# 📊 Strategy Specification

## Overview
This document specifies the mathematical foundations, parameters, and implementation details of the Hat Manifesto Ultimate Trading System.

## Mathematical Foundations

### 1. Funding Rate Arbitrage

#### Net Edge Calculation
```
net_edge = expected_funding_rate - taker_fee - expected_slippage
```

Where:
- `expected_funding_rate`: Predicted funding rate for next cycle
- `taker_fee`: 0.05% (5 bps) for perpetuals
- `expected_slippage`: Market impact estimation

#### Position Sizing
```
position_size = min(
    risk_budget / (atr * atr_multiplier),
    vol_target / current_volatility * equity * max_position_percent,
    equity_at_risk_percent * account_equity
)
```

### 2. Risk Management

#### ATR-Based Position Sizing
```
atr_position_size = risk_per_trade / (atr * atr_multiplier)
```

#### Volatility Targeting
```
vol_position_size = (vol_target / current_vol) * scaling_factor * equity * max_position_percent
```

#### Equity-at-Risk
```
ear_position_size = equity_at_risk_percent * account_equity
```

### 3. Fee Optimization

#### Maker vs Taker Decision
```
use_maker = (urgency_score < 0.7) AND 
            (fill_probability > 0.8) AND 
            (expected_fill_time < 60s)
```

#### Net Fee Calculation
```
net_fee = base_fee * (1 - volume_discount) * (1 - hype_discount) - maker_rebate
```

## Strategy Parameters

### Core Parameters
- **Initial Capital**: $10,000
- **Max Position Size**: 10% of equity
- **Equity-at-Risk**: 5% per trade
- **ATR Multiplier**: 2.0
- **Volatility Target**: 15% annual
- **Max Leverage**: 10x

### Risk Parameters
- **Daily Drawdown Limit**: 2%
- **Rolling Drawdown Limit**: 5%
- **Kill Switch Threshold**: 8%
- **Cooldown Period**: 24 hours

### Fee Parameters
- **Perpetual Maker Fee**: 0.01%
- **Perpetual Taker Fee**: 0.05%
- **Maker Rebate**: 0.005%
- **HYPE Staking Discount**: 50%

### Funding Parameters
- **Funding Cycle**: 1 hour
- **Min Net Edge**: 5 bps
- **Target Net Edge**: 10 bps
- **Max Net Edge**: 50 bps

## Implementation Details

### State Machine
```
IDLE → SIGNAL → PLACE → ACK → LIVE → FILL/REJECT → RECONCILE → ATTRIB
```

### Order Routing
1. **Default**: Post-only (maker) orders
2. **Promotion**: Taker orders on urgency
3. **Validation**: Pre-check all parameters
4. **Execution**: Via Hyperliquid API

### Monitoring
- **Real-time**: P&L, positions, risk metrics
- **Structured Logs**: JSON format with full observability
- **Prometheus**: Performance counters and metrics
- **Alerts**: Risk violations, system errors

## Performance Targets

### Risk-Adjusted Returns
- **Target Sharpe Ratio**: > 2.0
- **Max Drawdown**: < 5%
- **Win Rate**: > 60%
- **Profit Factor**: > 1.5

### Execution Quality
- **Maker Ratio**: > 80%
- **Avg Slippage**: < 2 bps
- **Fill Rate**: > 95%
- **Latency**: < 100ms p95

### System Reliability
- **Uptime**: > 99.9%
- **Error Rate**: < 0.1%
- **Recovery Time**: < 30s
- **Data Consistency**: 100%

## Validation Rules

### Order Validation
- Tick size compliance
- Minimum notional requirements
- Margin sufficiency
- Leverage limits
- Position size limits

### Risk Validation
- Equity-at-risk limits
- Drawdown thresholds
- Correlation limits
- Liquidity requirements

### System Validation
- API connectivity
- Data freshness
- State consistency
- Error handling

## Backtesting Framework

### Walk-Forward Analysis
- **Training Period**: 30 days
- **Testing Period**: 7 days
- **Step Size**: 1 day
- **Min Training Periods**: 100

### Regime Detection
- **Bull Market**: > 2% rolling return
- **Bear Market**: < -2% rolling return
- **Chop Market**: -2% to 2% rolling return
- **Volatility Regimes**: Low/Medium/High terciles

### Performance Metrics
- Total return, Sharpe ratio, max drawdown
- Component attribution (directional/fees/funding/slippage)
- Regime-specific performance
- Risk-adjusted metrics

## Monitoring and Alerting

### Key Metrics
- Real-time P&L and positions
- Risk metrics (VaR, drawdown, exposure)
- Execution quality (slippage, fees, fill rates)
- System health (latency, errors, connectivity)

### Alert Conditions
- Drawdown threshold breaches
- Risk limit violations
- System errors or timeouts
- Unusual market conditions
- API connectivity issues

### Response Procedures
- Automatic risk reduction
- Emergency position closure
- System restart procedures
- Manual intervention protocols

## Compliance and Security

### Security Measures
- API key rotation
- Secret detection
- Log redaction
- Access controls
- Audit trails

### Compliance Requirements
- Trade reporting
- Risk monitoring
- Error logging
- Performance attribution
- Regulatory compliance

## Maintenance and Updates

### Regular Maintenance
- Daily system health checks
- Weekly performance reviews
- Monthly parameter optimization
- Quarterly security audits

### Update Procedures
- Version control
- Testing protocols
- Deployment procedures
- Rollback capabilities
- Change documentation

## Appendix

### A. Hyperliquid-Specific Parameters
- Tick sizes, minimum notionals
- Fee structures, rebates
- Order types, time-in-force
- Margin requirements
- Funding mechanisms

### B. Risk Models
- VaR calculations
- Stress testing
- Scenario analysis
- Monte Carlo simulations
- Historical backtesting

### C. Performance Attribution
- Strategy decomposition
- Market regime analysis
- Risk factor analysis
- Fee impact analysis
- Slippage analysis
