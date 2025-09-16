# 📊 Observability Runbook

## Overview
This runbook provides comprehensive guidance for monitoring, debugging, and maintaining the Hat Manifesto Ultimate Trading System.

## System Architecture

### Core Components
- **Trading State Machine**: Order lifecycle management
- **Risk Manager**: Position sizing and risk monitoring
- **API Validator**: Pre-check validation
- **Maker Router**: Order routing and execution
- **Funding Scheduler**: Funding arbitrage management
- **Trade Ledger**: Comprehensive trade recording
- **Backtest Harness**: Strategy validation

### Data Flow
```
Market Data → Risk Manager → State Machine → API Validator → Maker Router → Exchange
     ↓              ↓              ↓              ↓              ↓
Trade Ledger ← Performance Metrics ← Observability ← Logs ← Execution Results
```

## Monitoring Dashboard

### Key Metrics

#### Performance Metrics
- **P95 Loop Latency**: < 100ms
- **P99 Loop Latency**: < 200ms
- **Average Loop Latency**: < 50ms
- **Orders Placed**: Real-time count
- **Orders Filled**: Real-time count
- **Orders Cancelled**: Real-time count
- **Orders Rejected**: Real-time count

#### Execution Quality
- **Maker Ratio**: > 80%
- **Average Slippage**: < 2 bps
- **Average Fee**: < 3 bps
- **Fill Rate**: > 95%
- **Rebate P&L**: Real-time tracking

#### Risk Metrics
- **Current Drawdown**: Real-time
- **Max Drawdown**: Historical
- **VaR (95%)**: Real-time
- **Position Count**: Real-time
- **Total Exposure**: Real-time
- **Margin Ratio**: Real-time

#### P&L Metrics
- **Realized P&L**: Real-time
- **Unrealized P&L**: Real-time
- **Funding P&L**: Real-time
- **Fee P&L**: Real-time
- **Slippage P&L**: Real-time
- **Net P&L**: Real-time

### Alerting Rules

#### Critical Alerts
- **Kill Switch Activated**: Immediate notification
- **Drawdown > 5%**: Immediate notification
- **System Error Rate > 1%**: Immediate notification
- **API Connectivity Lost**: Immediate notification
- **Risk Limit Violation**: Immediate notification

#### Warning Alerts
- **Drawdown > 2%**: Warning notification
- **Maker Ratio < 60%**: Warning notification
- **Slippage > 5 bps**: Warning notification
- **Fill Rate < 90%**: Warning notification
- **Latency > 200ms**: Warning notification

#### Info Alerts
- **Daily P&L Report**: Daily summary
- **Weekly Performance Report**: Weekly summary
- **Monthly Risk Report**: Monthly summary
- **System Health Check**: Hourly summary

## Logging and Observability

### Structured Logging

#### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information and state changes
- **WARNING**: Warning conditions and recoverable errors
- **ERROR**: Error conditions that don't stop the system
- **CRITICAL**: Critical errors that may stop the system

#### Log Format
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "component": "state_machine",
  "event_type": "state_transition",
  "data": {
    "from_state": "idle",
    "to_state": "signal",
    "signal_id": "signal_123",
    "duration_ms": 15.5
  }
}
```

#### Key Log Events
- **State Transitions**: All state machine transitions
- **Order Events**: Order placement, fills, cancellations
- **Risk Events**: Risk limit checks, violations
- **Performance Events**: Latency, throughput metrics
- **Error Events**: All errors and exceptions
- **System Events**: Startup, shutdown, health checks

### Prometheus Metrics

#### Counter Metrics
- `orders_placed_total`: Total orders placed
- `orders_filled_total`: Total orders filled
- `orders_cancelled_total`: Total orders cancelled
- `orders_rejected_total`: Total orders rejected
- `errors_total`: Total errors by type
- `risk_violations_total`: Total risk violations

#### Gauge Metrics
- `current_drawdown`: Current drawdown percentage
- `position_count`: Current position count
- `total_exposure`: Total exposure in USD
- `margin_ratio`: Current margin ratio
- `maker_ratio`: Current maker ratio
- `system_health`: System health score (0-1)

#### Histogram Metrics
- `order_latency_seconds`: Order execution latency
- `loop_duration_seconds`: Main loop duration
- `validation_time_seconds`: Validation time
- `risk_calculation_time_seconds`: Risk calculation time

### Distributed Tracing

#### Trace Spans
- **Order Lifecycle**: Complete order journey
- **Risk Calculation**: Risk assessment process
- **API Validation**: Validation process
- **State Transitions**: State machine transitions
- **Market Data**: Data retrieval and processing

#### Trace Context
- **Trace ID**: Unique identifier for each request
- **Span ID**: Unique identifier for each operation
- **Parent Span ID**: Parent operation identifier
- **Tags**: Key-value pairs for filtering
- **Logs**: Structured log entries

## Debugging Procedures

### Common Issues

#### High Latency
1. **Check System Resources**: CPU, memory, disk I/O
2. **Review Logs**: Look for bottlenecks in processing
3. **Check Network**: API connectivity and response times
4. **Analyze Metrics**: Identify slow components
5. **Review Code**: Look for inefficient algorithms

#### Order Rejections
1. **Check Validation**: Review pre-check validation results
2. **Verify Parameters**: Ensure tick size, notional compliance
3. **Check Risk Limits**: Verify risk management constraints
4. **Review Market Data**: Ensure fresh and accurate data
5. **Check API Status**: Verify exchange connectivity

#### Risk Violations
1. **Check Position Sizes**: Verify position sizing calculations
2. **Review Risk Metrics**: Check VaR, drawdown calculations
3. **Verify Limits**: Ensure risk limits are properly set
4. **Check Market Conditions**: Review volatility and market state
5. **Review Risk Model**: Verify risk model accuracy

#### System Errors
1. **Check Logs**: Review error logs for root cause
2. **Verify Dependencies**: Check external service status
3. **Review Configuration**: Ensure proper configuration
4. **Check Resources**: Verify system resources
5. **Test Recovery**: Verify error recovery procedures

### Debugging Tools

#### Log Analysis
```bash
# Search for specific events
grep "state_transition" logs/trading.log | jq '.data'

# Filter by time range
grep "2024-01-01T12:" logs/trading.log | jq '.data'

# Search for errors
grep "ERROR" logs/trading.log | jq '.data'

# Performance analysis
grep "latency" logs/trading.log | jq '.data.duration_ms'
```

#### Metric Analysis
```bash
# Query Prometheus metrics
curl "http://localhost:9090/api/v1/query?query=orders_placed_total"

# Get metric ranges
curl "http://localhost:9090/api/v1/query_range?query=current_drawdown&start=2024-01-01T00:00:00Z&end=2024-01-01T23:59:59Z&step=1m"

# Alert status
curl "http://localhost:9090/api/v1/alerts"
```

#### System Health Checks
```bash
# Check system status
python scripts/monitoring/check_system_health.py

# Verify API connectivity
python scripts/monitoring/check_api_connectivity.py

# Test risk calculations
python scripts/monitoring/test_risk_calculations.py

# Validate system state
python scripts/monitoring/validate_system_state.py
```

## Performance Optimization

### Latency Optimization
1. **Reduce API Calls**: Batch operations where possible
2. **Optimize Algorithms**: Use efficient data structures
3. **Cache Results**: Cache frequently accessed data
4. **Parallel Processing**: Use async/await for I/O operations
5. **Profile Code**: Identify and optimize bottlenecks

### Throughput Optimization
1. **Batch Operations**: Group related operations
2. **Connection Pooling**: Reuse connections
3. **Async Processing**: Use asynchronous operations
4. **Load Balancing**: Distribute load across instances
5. **Resource Scaling**: Scale resources based on load

### Memory Optimization
1. **Garbage Collection**: Optimize garbage collection
2. **Memory Pools**: Use memory pools for frequent allocations
3. **Data Structures**: Use efficient data structures
4. **Cache Management**: Implement proper cache eviction
5. **Memory Monitoring**: Monitor memory usage patterns

## Incident Response

### Incident Classification

#### Severity Levels
- **P1 - Critical**: System down, data loss, security breach
- **P2 - High**: Major functionality impacted, performance degraded
- **P3 - Medium**: Minor functionality impacted, workaround available
- **P4 - Low**: Cosmetic issues, minor bugs

#### Response Times
- **P1**: 15 minutes
- **P2**: 1 hour
- **P3**: 4 hours
- **P4**: 24 hours

### Incident Response Process

#### 1. Detection and Alerting
- Automated monitoring detects issues
- Alerts sent to on-call team
- Initial assessment of severity

#### 2. Investigation and Diagnosis
- Gather relevant logs and metrics
- Identify root cause
- Assess impact and scope
- Determine resolution approach

#### 3. Resolution and Recovery
- Implement fix or workaround
- Verify resolution
- Monitor system stability
- Document incident

#### 4. Post-Incident Review
- Conduct post-mortem
- Identify improvements
- Update procedures
- Share learnings

### Emergency Procedures

#### Kill Switch Activation
1. **Immediate Action**: Stop all trading
2. **Position Closure**: Close all open positions
3. **Risk Assessment**: Evaluate current risk exposure
4. **System Shutdown**: Graceful system shutdown
5. **Investigation**: Determine cause and impact

#### System Recovery
1. **Health Check**: Verify system components
2. **Data Validation**: Ensure data consistency
3. **Risk Validation**: Verify risk calculations
4. **Gradual Restart**: Restart components gradually
5. **Monitoring**: Enhanced monitoring during recovery

## Maintenance Procedures

### Daily Maintenance
- **System Health Check**: Verify all components
- **Performance Review**: Check key metrics
- **Error Review**: Review and address errors
- **Backup Verification**: Verify backup integrity
- **Security Check**: Review security logs

### Weekly Maintenance
- **Performance Analysis**: Analyze performance trends
- **Risk Review**: Review risk metrics and limits
- **Configuration Review**: Review and update configuration
- **Dependency Update**: Update dependencies
- **Documentation Update**: Update documentation

### Monthly Maintenance
- **Security Audit**: Comprehensive security review
- **Performance Optimization**: Optimize performance
- **Capacity Planning**: Plan for capacity needs
- **Disaster Recovery**: Test disaster recovery procedures
- **Compliance Review**: Review compliance requirements

### Quarterly Maintenance
- **System Upgrade**: Plan and execute upgrades
- **Architecture Review**: Review system architecture
- **Process Improvement**: Improve operational processes
- **Training Update**: Update team training
- **Vendor Review**: Review vendor relationships

## Troubleshooting Guide

### Common Error Messages

#### "Order rejected: Invalid tick size"
- **Cause**: Price not aligned to tick size
- **Solution**: Adjust price to nearest tick
- **Prevention**: Use pre-check validation

#### "Order rejected: Insufficient margin"
- **Cause**: Not enough margin for position
- **Solution**: Reduce position size or add margin
- **Prevention**: Check margin requirements

#### "Order rejected: Reduce-only violation"
- **Cause**: Reduce-only order would increase position
- **Solution**: Check current position and order side
- **Prevention**: Validate reduce-only logic

#### "System error: State machine timeout"
- **Cause**: State transition taking too long
- **Solution**: Check system performance and network
- **Prevention**: Optimize state machine performance

### Performance Issues

#### High Latency
- **Check**: System resources, network, API response times
- **Solution**: Optimize code, scale resources, check network
- **Prevention**: Monitor performance metrics

#### Low Throughput
- **Check**: System bottlenecks, resource constraints
- **Solution**: Optimize algorithms, scale resources
- **Prevention**: Load testing, capacity planning

#### Memory Issues
- **Check**: Memory usage, garbage collection, memory leaks
- **Solution**: Optimize memory usage, fix leaks
- **Prevention**: Memory monitoring, profiling

### Data Issues

#### Stale Data
- **Check**: Data freshness, update frequency
- **Solution**: Increase update frequency, check data sources
- **Prevention**: Data freshness monitoring

#### Inconsistent Data
- **Check**: Data synchronization, state consistency
- **Solution**: Fix synchronization, validate state
- **Prevention**: Data validation, consistency checks

#### Missing Data
- **Check**: Data sources, network connectivity
- **Solution**: Restore data sources, check connectivity
- **Prevention**: Data backup, redundancy

## Appendix

### A. Monitoring Tools
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Dashboards and visualization
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation and analysis
- **PagerDuty**: Incident management

### B. Alerting Channels
- **Slack**: Team notifications
- **Email**: Detailed notifications
- **SMS**: Critical alerts
- **PagerDuty**: Incident escalation
- **Webhook**: Custom integrations

### C. Runbook Templates
- **Incident Response**: Standard incident response
- **Emergency Procedures**: Emergency response
- **Maintenance Procedures**: Standard maintenance
- **Troubleshooting**: Common issues and solutions
- **Performance Optimization**: Performance improvement

### D. Contact Information
- **On-Call Team**: Primary response team
- **Escalation Contacts**: Management escalation
- **Vendor Contacts**: External service providers
- **Emergency Contacts**: Critical situation contacts
- **Documentation Contacts**: Documentation updates
