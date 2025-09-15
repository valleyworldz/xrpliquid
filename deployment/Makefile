# Makefile for Perfect Consolidated Bot

.PHONY: test pytest run bot help

# Default target
help:
	@echo "Available targets:"
	@echo "  test     - Run all tests"
	@echo "  pytest   - Run pytest suite"
	@echo "  run      - Run the trading bot"
	@echo "  install  - Install dependencies"
	@echo "  clean    - Clean up generated files"

# Run all tests
test:
	@echo "🧪 Running all tests..."
	python tests/test_risk.py

# Run pytest suite (if pytest is installed)
pytest:
	@echo "🧪 Running pytest suite..."
	pytest -q tests/

# Run the trading bot
run:
	@echo "🚀 Starting Perfect Consolidated Bot..."
	python PERFECT_CONSOLIDATED_BOT.py

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install numpy httpx cryptography eth-account

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.log" -delete
	find . -name "*.tmp" -delete

# Development helpers
dev-setup: install
	@echo "🔧 Development setup complete"

# Quick test run
quick-test:
	@echo "⚡ Quick test run..."
	python -c "from PERFECT_CONSOLIDATED_BOT import AdvancedRiskManager; import logging; rm = AdvancedRiskManager(logging.getLogger('test')); print('✅ Risk manager initialized successfully')" 