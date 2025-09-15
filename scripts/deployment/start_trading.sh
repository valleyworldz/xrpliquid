#!/bin/bash
# 🚀 HypeLiquidOG Startup Script
# =============================
# 
# Automated startup script for the HypeLiquidOG trading system
# Handles environment setup, dependency checks, and system launch
#
# Usage:
#   ./start_trading.sh              # Start with default settings
#   ./start_trading.sh --gui        # Start with GUI interface
#   ./start_trading.sh --monitor    # Start with performance monitor
#   ./start_trading.sh --setup      # Run setup wizard
#
# Author: HypeLiquidOG Team
# Version: 2.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
print_banner() {
    echo -e "${CYAN}"
    echo "🚀 ============================================ 🚀"
    echo "   HypeLiquidOG - Advanced Trading System"
    echo "🚀 ============================================ 🚀"
    echo -e "${NC}"
}

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    print_message $BLUE "🔍 Checking Python version..."
    
    if ! command_exists python3; then
        print_message $RED "❌ Python 3 is not installed!"
        print_message $YELLOW "   Please install Python 3.11+ and try again."
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    required_version="3.11"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_message $RED "❌ Python $required_version+ is required!"
        print_message $YELLOW "   Current version: $python_version"
        print_message $YELLOW "   Please upgrade Python and try again."
        exit 1
    fi
    
    print_message $GREEN "✅ Python version: $python_version"
}

# Setup virtual environment
setup_venv() {
    print_message $BLUE "🔧 Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        print_message $YELLOW "📦 Creating virtual environment..."
        python3 -m venv venv
        print_message $GREEN "✅ Virtual environment created"
    else
        print_message $GREEN "✅ Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_message $BLUE "🔄 Activating virtual environment..."
    source venv/bin/activate
    print_message $GREEN "✅ Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_message $BLUE "📦 Installing/updating dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_message $RED "❌ requirements.txt not found!"
        print_message $YELLOW "   Running setup to create requirements..."
        python3 setup.py
    fi
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    print_message $GREEN "✅ Dependencies installed successfully"
}

# Check credentials
check_credentials() {
    print_message $BLUE "🔐 Checking credentials..."
    
    if [ ! -f "credentials/encrypted_credentials.dat" ]; then
        print_message $YELLOW "⚠️ No credentials found!"
        print_message $BLUE "🔧 Running credential setup..."
        python3 setup_credentials.py
    else
        print_message $GREEN "✅ Credentials found"
    fi
}

# Check system health
check_system_health() {
    print_message $BLUE "🏥 Checking system health..."
    
    # Check disk space
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 1048576 ]; then  # Less than 1GB
        print_message $YELLOW "⚠️ Low disk space: $(df -h . | tail -1 | awk '{print $4}') available"
    fi
    
    # Check internet connectivity
    if ! ping -c 1 google.com >/dev/null 2>&1; then
        print_message $RED "❌ No internet connection!"
        print_message $YELLOW "   Internet connection is required for trading."
        exit 1
    fi
    
    # Check if HyperLiquid API is accessible
    if command_exists curl; then
        if ! curl -s --connect-timeout 5 https://api.hyperliquid.xyz/info >/dev/null; then
            print_message $YELLOW "⚠️ HyperLiquid API may be unreachable"
            print_message $YELLOW "   This could affect trading performance."
        fi
    fi
    
    print_message $GREEN "✅ System health check passed"
}

# Start trading system
start_trading() {
    print_message $GREEN "🚀 Starting HypeLiquidOG Trading System..."
    print_message $BLUE "📊 Initializing autonomous trading engine..."
    
    # Set environment variables for optimal performance
    export PYTHONUNBUFFERED=1
    export HYPELIQUID_OPTIMIZED=1
    
    # Start the main trading system
    python3 progressive_auto_scaling_system_v21.py
}

# Start GUI interface
start_gui() {
    print_message $GREEN "🎮 Starting HypeLiquidOG GUI Interface..."
    print_message $BLUE "🖥️ Launching professional trading dashboard..."
    
    # Check if GUI dependencies are available
    python3 -c "import tkinter" 2>/dev/null || {
        print_message $RED "❌ GUI dependencies not available!"
        print_message $YELLOW "   Install tkinter: sudo apt-get install python3-tk (Ubuntu/Debian)"
        exit 1
    }
    
    # Start the GUI
    python3 position_control_gui.py
}

# Start performance monitor
start_monitor() {
    print_message $GREEN "📊 Starting Performance Monitor..."
    print_message $BLUE "📈 Launching real-time analytics dashboard..."
    
    # Start the performance monitor
    python3 performance_monitor.py
}

# Run setup wizard
run_setup() {
    print_message $GREEN "🔧 Running Setup Wizard..."
    print_message $BLUE "⚙️ Configuring HypeLiquidOG system..."
    
    # Run the setup script
    python3 setup.py
}

# Show usage information
show_usage() {
    echo "🚀 HypeLiquidOG Startup Script"
    echo ""
    echo "Usage:"
    echo "  ./start_trading.sh              Start autonomous trading system"
    echo "  ./start_trading.sh --gui        Start with GUI interface"
    echo "  ./start_trading.sh --monitor    Start performance monitor"
    echo "  ./start_trading.sh --setup      Run setup wizard"
    echo "  ./start_trading.sh --help       Show this help message"
    echo ""
    echo "Options:"
    echo "  --gui        Launch the professional GUI interface"
    echo "  --monitor    Launch the performance monitoring dashboard"
    echo "  --setup      Run the initial setup and configuration"
    echo "  --help       Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./start_trading.sh              # Start trading with default settings"
    echo "  ./start_trading.sh --gui        # Start with visual interface"
    echo "  ./start_trading.sh --monitor    # Monitor existing trading session"
}

# Cleanup function
cleanup() {
    print_message $YELLOW "🔄 Cleaning up..."
    # Add any cleanup tasks here
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_banner
    
    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --setup)
            check_python
            setup_venv
            install_dependencies
            run_setup
            exit 0
            ;;
        --gui)
            MODE="gui"
            ;;
        --monitor)
            MODE="monitor"
            ;;
        "")
            MODE="trading"
            ;;
        *)
            print_message $RED "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
    
    # System checks
    check_python
    setup_venv
    install_dependencies
    check_credentials
    check_system_health
    
    # Start the appropriate mode
    case "$MODE" in
        gui)
            start_gui
            ;;
        monitor)
            start_monitor
            ;;
        trading)
            start_trading
            ;;
    esac
}

# Run main function with all arguments
main "$@"

