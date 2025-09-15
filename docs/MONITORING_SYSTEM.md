# 📊 Monitoring System Documentation

## Overview

The XRP Trading System includes a comprehensive monitoring stack built on Prometheus and Grafana, providing real-time visibility into trading performance, system health, and key metrics.

## Architecture

### Components

1. **Prometheus Metrics Collector** - Collects and exposes trading metrics
2. **Prometheus Server** - Stores and queries metrics data
3. **Grafana Dashboard** - Visualizes metrics with interactive dashboards
4. **Alert Manager** - Handles alerts and notifications
5. **Node Exporter** - System-level metrics (optional)

### Ports

- **Prometheus**: 9090
- **Grafana**: 3000
- **Metrics Collector**: 8000 (live trading), 8001 (paper trading)
- **Alert Manager**: 9093
- **Node Exporter**: 9100

## Metrics Collected

### Trading Metrics

#### Trade Execution
- `trading_trades_total` - Total number of trades executed
- `trading_trades_successful_total` - Successful trades
- `trading_trades_failed_total` - Failed trades
- `trading_order_latency_seconds` - Order execution latency

#### PnL and Performance
- `trading_pnl_total_usd` - Total profit and loss
- `trading_pnl_realized_usd` - Realized PnL
- `trading_pnl_unrealized_usd` - Unrealized PnL
- `trading_pnl_percentage` - PnL as percentage of capital
- `trading_win_rate_percentage` - Win rate percentage
- `trading_sharpe_ratio` - Sharpe ratio
- `trading_profit_factor` - Profit factor

#### Position Management
- `trading_position_size` - Current position size
- `trading_position_value_usd` - Position value in USD
- `trading_avg_entry_price` - Average entry price

### Execution Quality

#### Latency Metrics
- `trading_order_latency_seconds` - Order execution latency
- `trading_api_latency_seconds` - API call latency

#### Slippage Analysis
- `trading_slippage_basis_points` - Slippage in basis points
- `trading_slippage_cost_usd_total` - Total slippage cost
- `trading_market_impact_percentage` - Market impact percentage

### Fees and Costs
- `trading_fees_paid_usd_total` - Total fees paid
- `trading_funding_payments_usd_total` - Funding payments
- `trading_funding_rate` - Current funding rate

### Market Data
- `trading_price_current` - Current market price
- `trading_price_change_24h_percentage` - 24h price change
- `trading_volume_24h_usd` - 24h trading volume
- `trading_spread_basis_points` - Bid-ask spread

### System Performance
- `trading_system_uptime_seconds` - System uptime
- `trading_cycle_count` - Trading cycles completed
- `trading_emergency_mode` - Emergency mode status
- `trading_margin_usage_percentage` - Margin usage

### Risk Metrics
- `trading_drawdown_percentage` - Current drawdown
- `trading_max_drawdown_percentage` - Maximum drawdown
- `trading_volatility` - Current volatility

## Grafana Dashboard

### Dashboard Panels

#### 1. PnL Overview
- **Total PnL (USD)** - Real-time profit/loss tracking
- **PnL Percentage** - Returns as percentage of capital

#### 2. Execution Quality
- **Order Execution Latency** - 50th, 95th, 99th percentiles
- **Slippage (Basis Points)** - Slippage distribution
- **Trade Rate** - Trades per second

#### 3. Costs and Fees
- **Fees Paid (USD/sec)** - Fee rate over time
- **Funding Rate** - Current funding rates

#### 4. Market Data
- **Current Price** - Real-time price tracking
- **Bid-Ask Spread** - Spread in basis points

#### 5. Performance Metrics
- **Win Rate (%)** - Success rate over time
- **Sharpe Ratio** - Risk-adjusted returns
- **Drawdown (%)** - Risk metrics

### Dashboard Features

- **Real-time Updates** - 5-second refresh rate
- **Interactive Panels** - Zoom, pan, and drill-down
- **Time Range Selection** - Flexible time windows
- **Alert Integration** - Visual alert indicators
- **Export Capabilities** - PNG, PDF, CSV export

## Alerting Rules

### Critical Alerts

#### High Latency
- **Warning**: 95th percentile > 100ms for 1 minute
- **Critical**: 95th percentile > 500ms for 30 seconds

#### High Slippage
- **Warning**: 95th percentile > 50 bps for 2 minutes
- **Critical**: 95th percentile > 100 bps for 1 minute

#### Large Losses
- **Warning**: Total PnL < -$1,000 for 1 minute
- **Critical**: Total PnL < -$5,000 for 30 seconds

#### High Drawdown
- **Warning**: Drawdown < -10% for 2 minutes
- **Critical**: Drawdown < -20% for 1 minute

#### System Issues
- **Critical**: Emergency mode active
- **Critical**: System down for 30 seconds
- **Warning**: High margin usage > 80%
- **Critical**: Margin usage > 95%

### Performance Alerts

#### Low Performance
- **Warning**: Win rate < 40% for 5 minutes
- **Warning**: Sharpe ratio < 1.0 for 10 minutes
- **Critical**: Negative Sharpe ratio for 5 minutes

#### High Risk
- **Warning**: High volatility > 5% for 5 minutes
- **Warning**: Low profit factor < 1.2 for 10 minutes
- **Critical**: Negative profit factor for 5 minutes

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ with required packages
- 4GB+ RAM for monitoring stack

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Monitoring Stack**
   ```bash
   python scripts/start_monitoring.py start
   ```

3. **Access Dashboards**
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - Metrics: http://localhost:8000/metrics

### Docker Compose Setup

1. **Start Services**
   ```bash
   cd monitoring
   docker-compose up -d
   ```

2. **Verify Services**
   ```bash
   docker-compose ps
   ```

3. **View Logs**
   ```bash
   docker-compose logs -f
   ```

### Manual Setup

1. **Start Prometheus**
   ```bash
   prometheus --config.file=monitoring/prometheus/prometheus.yml
   ```

2. **Start Grafana**
   ```bash
   grafana-server --config=monitoring/grafana/grafana.ini
   ```

3. **Start Metrics Collector**
   ```python
   from src.core.monitoring.prometheus_metrics import get_metrics_collector
   metrics = get_metrics_collector(port=8000)
   ```

## Usage Examples

### Basic Metrics Collection

```python
from src.core.monitoring.prometheus_metrics import get_metrics_collector

# Initialize metrics collector
metrics = get_metrics_collector(port=8000)

# Record a trade
metrics.record_trade(
    strategy="Ultra-Efficient XRP System",
    side="BUY",
    symbol="XRP",
    order_type="MARKET",
    is_live=True,
    success=True
)

# Record slippage
metrics.record_slippage(
    strategy="Ultra-Efficient XRP System",
    side="BUY",
    symbol="XRP",
    order_type="MARKET",
    slippage_bps=2.5,
    slippage_cost=0.25
)

# Update PnL
metrics.update_pnl(
    strategy="Ultra-Efficient XRP System",
    symbol="XRP",
    total_pnl=150.75,
    realized_pnl=100.25,
    unrealized_pnl=50.50,
    pnl_percentage=1.5075
)
```

### Integration with Trading Systems

```python
# In your trading system
from src.core.monitoring.prometheus_metrics import record_trade_metrics

# After executing a trade
trade_data = {
    'strategy': 'Ultra-Efficient XRP System',
    'side': 'BUY',
    'symbol': 'XRP',
    'order_type': 'MARKET',
    'is_live_trade': True,
    'success': True,
    'slippage_bps': 2.5,
    'slippage_cost': 0.25,
    'fees_paid': 0.1,
    'latency_ms': 15.5
}

# Record metrics automatically
record_trade_metrics(trade_data, metrics_collector)
```

### Custom Metrics

```python
# Record custom metrics
metrics.record_fees(
    strategy="Ultra-Efficient XRP System",
    symbol="XRP",
    fee_type="trading",
    fee_amount=0.1
)

metrics.update_funding_rate("XRP", 0.0001)

metrics.update_market_data(
    symbol="XRP",
    price=0.5234,
    price_change_24h=2.5,
    volume_24h=1000000.0,
    spread_bps=3.2
)
```

## Monitoring Best Practices

### 1. Metric Naming
- Use consistent naming conventions
- Include relevant labels for filtering
- Avoid high-cardinality labels

### 2. Alert Thresholds
- Set realistic thresholds based on historical data
- Use multiple severity levels
- Test alerts regularly

### 3. Dashboard Design
- Group related metrics together
- Use appropriate visualization types
- Include time range selectors

### 4. Performance Optimization
- Limit metric collection frequency
- Use efficient data types
- Monitor Prometheus resource usage

### 5. Data Retention
- Configure appropriate retention periods
- Use downsampling for long-term storage
- Archive old data when needed

## Troubleshooting

### Common Issues

#### Metrics Not Appearing
- Check if metrics collector is running
- Verify port configuration
- Check Prometheus scrape configuration

#### High Memory Usage
- Reduce metric collection frequency
- Limit label cardinality
- Increase Prometheus memory limits

#### Dashboard Not Loading
- Verify Grafana datasource configuration
- Check Prometheus connectivity
- Restart Grafana service

#### Alerts Not Firing
- Verify alert rule syntax
- Check AlertManager configuration
- Test alert expressions

### Debug Commands

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Test Prometheus query
curl "http://localhost:9090/api/v1/query?query=trading_trades_total"

# Check Grafana health
curl http://localhost:3000/api/health

# View Prometheus targets
curl http://localhost:9090/api/v1/targets
```

## Advanced Configuration

### Custom Dashboards

1. Create dashboard JSON file
2. Place in `monitoring/grafana/dashboards/`
3. Restart Grafana or use provisioning

### Custom Alerts

1. Add alert rules to `monitoring/prometheus/rules/`
2. Update Prometheus configuration
3. Reload Prometheus configuration

### External Storage

```yaml
# prometheus.yml
remote_write:
  - url: "http://influxdb:8086/api/v1/prom/write"
    basic_auth:
      username: "admin"
      password: "password"
```

## Security Considerations

### Access Control
- Use authentication for Grafana
- Restrict Prometheus access
- Secure metrics endpoints

### Data Privacy
- Avoid logging sensitive data
- Use secure communication
- Regular security updates

### Network Security
- Use firewalls to restrict access
- Enable TLS for external access
- Monitor access logs

## Performance Tuning

### Prometheus Optimization
- Increase scrape intervals for less critical metrics
- Use recording rules for complex queries
- Optimize storage configuration

### Grafana Optimization
- Limit dashboard refresh rates
- Use query caching
- Optimize panel queries

### System Resources
- Monitor CPU and memory usage
- Scale resources as needed
- Use SSD storage for better performance

## Conclusion

The monitoring system provides comprehensive visibility into the XRP trading system's performance, enabling proactive management and optimization. With real-time metrics, interactive dashboards, and intelligent alerting, traders can make informed decisions and maintain optimal system performance.
