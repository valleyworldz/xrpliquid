# 🎯 Hyperliquid Invariants

## Overview
This document defines the critical invariants and constraints specific to the Hyperliquid exchange that must be maintained at all times.

## Exchange-Specific Invariants

### 1. Tick Size Compliance
**Invariant**: All order prices must be aligned to the exchange-defined tick size.

#### Tick Sizes
- **XRP**: 0.0001 (4 decimal places)
- **BTC**: 0.01 (2 decimal places)
- **ETH**: 0.01 (2 decimal places)
- **SOL**: 0.001 (3 decimal places)
- **ARB**: 0.0001 (4 decimal places)

#### Validation
```python
def validate_tick_size(price: float, symbol: str) -> bool:
    tick_size = TICK_SIZES[symbol]
    return price % tick_size == 0
```

### 2. Minimum Notional Requirements
**Invariant**: All orders must meet minimum notional value requirements.

#### Minimum Notionals
- **XRP**: $1.00
- **BTC**: $10.00
- **ETH**: $10.00
- **SOL**: $5.00
- **ARB**: $1.00

#### Validation
```python
def validate_min_notional(quantity: float, price: float, symbol: str) -> bool:
    notional = quantity * price
    min_notional = MIN_NOTIONALS[symbol]
    return notional >= min_notional
```

### 3. Funding Rate Cycles
**Invariant**: Funding rate arbitrage must align with 1-hour funding cycles.

#### Funding Schedule
- **Cycle Duration**: 1 hour (3600 seconds)
- **Funding Times**: Every hour on the hour (00:00, 01:00, 02:00, etc.)
- **Payment Frequency**: 1 hour
- **Rate Calculation**: 8-hour TWAP of premium index

#### Implementation
```python
def get_next_funding_time() -> float:
    current_time = time.time()
    current_hour = int(current_time // 3600) * 3600
    return current_hour + 3600
```

### 4. Order Type Constraints
**Invariant**: Only supported order types and time-in-force combinations are allowed.

#### Supported Order Types
- **Limit Orders**: With GTC, IOC, ALO time-in-force
- **Market Orders**: With IOC time-in-force
- **Stop-Limit Orders**: With GTC time-in-force
- **Scale Orders**: With GTC time-in-force
- **TWAP Orders**: With GTC time-in-force

#### Validation
```python
def validate_order_type(order_type: str, time_in_force: str) -> bool:
    valid_combinations = {
        'limit': ['GTC', 'IOC', 'ALO'],
        'market': ['IOC'],
        'stop_limit': ['GTC'],
        'scale': ['GTC'],
        'twap': ['GTC']
    }
    return time_in_force in valid_combinations.get(order_type, [])
```

### 5. Margin Requirements
**Invariant**: All positions must maintain sufficient margin requirements.

#### Margin Requirements
- **Initial Margin**: 10% of position value
- **Maintenance Margin**: 5% of position value
- **Leverage Limits**: 1x to 50x
- **Isolated Margin**: Per-position margin isolation

#### Validation
```python
def validate_margin(position_value: float, leverage: float, available_margin: float) -> bool:
    required_margin = position_value / leverage * 0.1  # 10% initial margin
    return required_margin <= available_margin
```

### 6. Position Size Limits
**Invariant**: All positions must be within exchange-defined size limits.

#### Position Limits
- **Maximum Position Size**: $1,000,000 USD
- **Minimum Position Size**: $10 USD
- **Maximum Leverage**: 50x
- **Minimum Leverage**: 1x

#### Validation
```python
def validate_position_size(quantity: float, price: float) -> bool:
    position_value = quantity * price
    return 10.0 <= position_value <= 1000000.0
```

### 7. Fee Structure Compliance
**Invariant**: All fee calculations must use the correct Hyperliquid fee structure.

#### Fee Structure
- **Perpetual Maker Fee**: 0.01% (1 bp)
- **Perpetual Taker Fee**: 0.05% (5 bps)
- **Spot Maker Fee**: 0.02% (2 bps)
- **Spot Taker Fee**: 0.06% (6 bps)
- **Maker Rebate**: 0.005% (0.5 bp) for perpetuals, 0.01% (1 bp) for spot

#### Volume Tiers
- **Tier 1**: $0-1M volume, 0% discount
- **Tier 2**: $1M-5M volume, 10% maker discount, 5% taker discount
- **Tier 3**: $5M-20M volume, 20% maker discount, 10% taker discount
- **Tier 4**: $20M+ volume, 30% maker discount, 15% taker discount

#### HYPE Staking Discount
- **Discount**: 50% off all fees
- **Requirement**: Stake HYPE tokens
- **Application**: Applied after volume tier discounts

### 8. Reduce-Only Order Constraints
**Invariant**: Reduce-only orders must not increase position size.

#### Validation
```python
def validate_reduce_only(side: str, quantity: float, current_position: float) -> bool:
    if side == 'buy':
        return current_position < 0  # Can only buy to reduce short position
    else:  # sell
        return current_position > 0  # Can only sell to reduce long position
```

### 9. Post-Only Order Constraints
**Invariant**: Post-only orders must not cross the spread.

#### Validation
```python
def validate_post_only(side: str, price: float, bid: float, ask: float) -> bool:
    if side == 'buy':
        return price < ask  # Buy order below ask
    else:  # sell
        return price > bid  # Sell order above bid
```

### 10. Market Data Freshness
**Invariant**: All trading decisions must use fresh market data.

#### Data Requirements
- **Price Data**: < 1 second old
- **Order Book**: < 1 second old
- **Volume Data**: < 5 seconds old
- **Funding Rates**: < 1 minute old

#### Validation
```python
def validate_data_freshness(timestamp: float, max_age: float) -> bool:
    return time.time() - timestamp < max_age
```

## System-Level Invariants

### 1. State Consistency
**Invariant**: System state must be consistent across all components.

#### Consistency Checks
- Position sizes match across all systems
- Account balances are synchronized
- Order states are consistent
- Risk metrics are up-to-date

### 2. Error Handling
**Invariant**: All errors must be handled gracefully without system failure.

#### Error Categories
- **API Errors**: Rate limits, timeouts, rejections
- **Network Errors**: Connectivity issues, timeouts
- **Validation Errors**: Parameter validation failures
- **Risk Errors**: Risk limit violations

### 3. Idempotence
**Invariant**: All operations must be idempotent.

#### Idempotence Requirements
- Order placement with same CLID
- Position updates
- Risk calculations
- State transitions

### 4. Atomicity
**Invariant**: All multi-step operations must be atomic.

#### Atomic Operations
- Order placement and risk updates
- Position changes and P&L updates
- State transitions
- Error recovery

## Monitoring and Alerting

### Invariant Violations
- **Severity**: Critical
- **Response**: Immediate system halt
- **Recovery**: Manual intervention required
- **Prevention**: Automated validation

### Monitoring Metrics
- Tick size compliance rate
- Minimum notional compliance rate
- Funding cycle alignment
- Order type validation rate
- Margin requirement compliance
- Position size compliance
- Fee calculation accuracy
- Reduce-only validation rate
- Post-only validation rate
- Data freshness compliance

### Alert Conditions
- Any invariant violation
- High error rates
- Data staleness
- System inconsistencies
- Risk limit breaches

## Testing and Validation

### Unit Tests
- Individual invariant validation
- Edge case testing
- Error condition testing
- Performance testing

### Integration Tests
- End-to-end invariant testing
- System consistency testing
- Error handling testing
- Recovery testing

### Property Tests
- Invariant preservation
- State consistency
- Error handling
- Performance characteristics

## Compliance and Auditing

### Audit Trail
- All invariant checks logged
- Violations tracked and reported
- System state changes recorded
- Error conditions documented

### Compliance Reporting
- Daily invariant compliance reports
- Weekly system health reports
- Monthly performance reviews
- Quarterly security audits

## Maintenance and Updates

### Regular Maintenance
- Daily invariant validation
- Weekly system health checks
- Monthly parameter updates
- Quarterly security reviews

### Update Procedures
- Invariant changes require approval
- Testing protocols for new invariants
- Documentation updates
- Training for new requirements

## Appendix

### A. Hyperliquid API Endpoints
- Order placement and management
- Position and account information
- Market data and funding rates
- Risk and margin information

### B. Error Codes and Handling
- API error codes and meanings
- Error handling procedures
- Recovery protocols
- Escalation procedures

### C. Performance Benchmarks
- Invariant validation performance
- System response times
- Error rates and recovery times
- Compliance metrics
