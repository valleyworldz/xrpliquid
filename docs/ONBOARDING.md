# 🚀 XRPLiquid Onboarding Guide

## Welcome to XRPLiquid

XRPLiquid is an institutional-grade XRP trading system built on the Hat Manifesto framework. This guide will help you get started with the system.

## Prerequisites

### 1. System Requirements
- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 8GB RAM minimum
- **Storage**: 100GB free space
- **Network**: Stable internet connection

### 2. Required Software
- **Python**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **IDE**: VS Code, PyCharm, or similar
- **Terminal**: Command line interface

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/valleyworldz/xrpliquid.git
cd xrpliquid
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
```bash
# Copy configuration template
cp config/template.json config/config.json

# Edit configuration
nano config/config.json
```

## Quick Start

### 1. Run System Check
```bash
python -c "from src.core.engines.ultra_efficient_xrp_system import UltraEfficientXRPSystem; print('System ready')"
```

### 2. Start Data Capture
```bash
python src/data_capture/enhanced_tick_capture.py
```

### 3. Run Backtest
```bash
python run_hat_manifesto_backtest.py
```

### 4. View Results
```bash
# Open dashboard
open reports/executive_dashboard.html

# View tearsheet
open reports/tearsheets/comprehensive_tearsheet.html
```

## System Overview

### 1. Hat Manifesto Framework
The system implements 9 specialized roles:
- **Hyperliquid Exchange Architect**: Exchange integration
- **Chief Quantitative Strategist**: Strategy development
- **Market Microstructure Analyst**: Order routing
- **Low-Latency Engineer**: Performance optimization
- **Automated Execution Manager**: Order management
- **Risk Oversight Officer**: Risk management
- **Cryptographic Security Architect**: Security
- **Performance Quant Analyst**: Analytics
- **Machine Learning Research Scientist**: ML models

### 2. Key Components
- **Trading Engine**: Core trading logic
- **Data Capture**: Real-time data collection
- **Risk Management**: Position sizing and limits
- **Analytics**: Performance measurement
- **Reporting**: Comprehensive reports

## Configuration

### 1. Trading Parameters
Edit `config/trading_parameters.json`:
```json
{
  "position_size": 1000,
  "stop_loss": 0.02,
  "take_profit": 0.04,
  "max_drawdown": 0.05
}
```

### 2. Risk Parameters
Edit `config/risk_parameters.json`:
```json
{
  "max_daily_drawdown": 0.05,
  "var_confidence_level": 0.95,
  "position_risk_limit": 0.02
}
```

### 3. Exchange Configuration
Edit `config/exchange_config.json`:
```json
{
  "exchange": "hyperliquid",
  "api_key": "your_api_key",
  "secret_key": "your_secret_key",
  "testnet": true
}
```

## Running the System

### 1. Development Mode
```bash
# Run with debug logging
python run_bot.py --debug

# Run specific component
python src/core/engines/ultra_efficient_xrp_system.py
```

### 2. Production Mode
```bash
# Run with production settings
python run_bot.py --production

# Run with specific configuration
python run_bot.py --config config/production.json
```

### 3. Backtesting
```bash
# Run comprehensive backtest
python run_hat_manifesto_backtest.py

# Run specific strategy
python src/core/strategies/funding_arbitrage.py
```

## Monitoring

### 1. System Health
```bash
# Check system status
python scripts/check_system_health.py

# View logs
tail -f logs/system.log
```

### 2. Performance Metrics
```bash
# View performance dashboard
open reports/executive_dashboard.html

# Check latency metrics
cat reports/latency/latency_analysis.json
```

### 3. Risk Monitoring
```bash
# Check risk metrics
cat reports/risk/var_es.json

# View kill-switch status
cat reports/risk/hysteresis_state.json
```

## Development

### 1. Code Structure
```
src/
├── core/           # Core system components
├── strategies/     # Trading strategies
├── data_capture/   # Data collection
├── analytics/      # Performance analysis
└── utils/          # Utility functions
```

### 2. Adding New Strategies
```python
# Create new strategy file
# src/strategies/my_strategy.py

class MyStrategy:
    def __init__(self, config):
        self.config = config
    
    def generate_signal(self, data):
        # Strategy logic here
        return signal
```

### 3. Testing
```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_strategy.py
```

## Troubleshooting

### 1. Common Issues

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Configuration Issues
```bash
# Validate configuration
python scripts/validate_config.py

# Check configuration format
python -c "import json; json.load(open('config/config.json'))"
```

#### Data Issues
```bash
# Check data directory
ls -la data/

# Verify data format
python scripts/verify_data_format.py
```

### 2. Getting Help
- **Documentation**: Check `docs/` directory
- **Issues**: Create GitHub issue
- **Discussions**: Use GitHub discussions
- **Email**: Contact development team

## Best Practices

### 1. Development
- **Code Style**: Follow PEP 8
- **Testing**: Write unit tests
- **Documentation**: Document all functions
- **Version Control**: Use Git properly

### 2. Trading
- **Risk Management**: Always use stop-losses
- **Position Sizing**: Follow risk limits
- **Monitoring**: Watch system health
- **Backtesting**: Test before live trading

### 3. Security
- **API Keys**: Keep secure
- **Access Control**: Use least privilege
- **Monitoring**: Watch for anomalies
- **Updates**: Keep system updated

## Next Steps

### 1. Learn More
- Read `docs/ARCHITECTURE.md`
- Study `docs/RUNBOOK.md`
- Review `docs/SECURITY.md`

### 2. Practice
- Run backtests
- Experiment with strategies
- Monitor performance
- Analyze results

### 3. Contribute
- Report bugs
- Suggest improvements
- Submit pull requests
- Share knowledge

## Support

### 1. Documentation
- **Architecture**: `docs/ARCHITECTURE.md`
- **Operations**: `docs/RUNBOOK.md`
- **Security**: `docs/SECURITY.md`
- **SLOs**: `docs/SLOs.md`

### 2. Community
- **GitHub**: Repository and issues
- **Discussions**: Community forum
- **Wiki**: Additional documentation
- **Examples**: Code examples

### 3. Professional Support
- **Consulting**: Custom development
- **Training**: Team training
- **Support**: Technical support
- **Maintenance**: System maintenance

Welcome to XRPLiquid! 🚀