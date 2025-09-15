#!/bin/bash
# UNIVERSAL TRADING BOT - EASY TOKEN SWITCHING
# Simple script to run the trading bot with different tokens

echo "🚀 UNIVERSAL TRADING BOT - EASY TOKEN SWITCHING"
echo "================================================"

# Function to show usage
show_usage() {
    echo "Usage:"
    echo "  ./trade.sh MAV          # Trade MAV token"
    echo "  ./trade.sh TRUMP        # Trade TRUMP token"
    echo "  ./trade.sh HYPE         # Trade HYPE token"
    echo "  ./trade.sh --interactive # Interactive token selection"
    echo "  ./trade.sh --list       # List available tokens"
    echo ""
    echo "Available tokens: MAV, TRUMP, HYPE, BTC, ETH, SOL, DOGE, PEPE, WIF, BONK"
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    show_usage
    echo ""
    echo "🎯 Starting with default token (MAV)..."
    cd /home/ubuntu/hypeliquidOG
    python3 universal_trader.py
    exit 0
fi

# Handle arguments
case "$1" in
    --help|-h)
        show_usage
        exit 0
        ;;
    --list|-l)
        cd /home/ubuntu/hypeliquidOG
        python3 universal_trader.py --list-tokens
        exit 0
        ;;
    --interactive|-i)
        cd /home/ubuntu/hypeliquidOG
        python3 universal_trader.py --interactive
        exit 0
        ;;
    *)
        TOKEN="$1"
        echo "🎯 Starting trading bot with token: $TOKEN"
        cd /home/ubuntu/hypeliquidOG
        python3 universal_trader.py --token "$TOKEN"
        exit 0
        ;;
esac

