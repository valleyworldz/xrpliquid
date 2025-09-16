#!/bin/bash

# Market Data Capture Scheduler
# Launches both tick listener and funding logger as sidecar processes

set -e

# Configuration
DATA_DIR="data"
LOG_DIR="logs"
PID_DIR="pids"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# Function to start a process
start_process() {
    local name=$1
    local command=$2
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$PID_DIR/${name}.pid"
    
    echo "Starting $name..."
    
    # Check if already running
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "$name is already running (PID: $(cat "$pid_file"))"
        return 0
    fi
    
    # Start process in background
    nohup $command > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    echo "$name started with PID: $pid"
    echo "Logs: $log_file"
}

# Function to stop a process
stop_process() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (PID: $pid)..."
            kill "$pid"
            rm -f "$pid_file"
            echo "$name stopped"
        else
            echo "$name is not running"
            rm -f "$pid_file"
        fi
    else
        echo "$name is not running"
    fi
}

# Function to check process status
check_status() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$name is running (PID: $pid)"
            return 0
        else
            echo "$name is not running (stale PID file)"
            rm -f "$pid_file"
            return 1
        fi
    else
        echo "$name is not running"
        return 1
    fi
}

# Function to show logs
show_logs() {
    local name=$1
    local log_file="$LOG_DIR/${name}.log"
    
    if [ -f "$log_file" ]; then
        echo "=== $name logs (last 50 lines) ==="
        tail -n 50 "$log_file"
    else
        echo "No logs found for $name"
    fi
}

# Main script logic
case "$1" in
    start)
        echo "Starting Market Data Capture System..."
        
        # Start tick listener
        start_process "tick_listener" "python src/data_capture/tick_listener.py"
        
        # Start funding logger
        start_process "funding_logger" "python src/data_capture/funding_logger.py"
        
        echo "Market Data Capture System started"
        echo "Use '$0 status' to check status"
        echo "Use '$0 logs' to view logs"
        ;;
    
    stop)
        echo "Stopping Market Data Capture System..."
        
        # Stop tick listener
        stop_process "tick_listener"
        
        # Stop funding logger
        stop_process "funding_logger"
        
        echo "Market Data Capture System stopped"
        ;;
    
    restart)
        echo "Restarting Market Data Capture System..."
        $0 stop
        sleep 2
        $0 start
        ;;
    
    status)
        echo "Market Data Capture System Status:"
        echo "=================================="
        check_status "tick_listener"
        check_status "funding_logger"
        ;;
    
    logs)
        echo "Market Data Capture System Logs:"
        echo "================================"
        show_logs "tick_listener"
        echo ""
        show_logs "funding_logger"
        ;;
    
    convert)
        echo "Converting tick data to Parquet..."
        date=${2:-$(date +%Y-%m-%d)}
        python -c "
from src.data_capture.tick_listener import TickListener
listener = TickListener()
listener.convert_to_parquet('$date')
print('Conversion completed')
"
        ;;
    
    summary)
        echo "Market Data Capture Summary:"
        echo "============================"
        
        # Show data directory contents
        echo "Data Directory Contents:"
        ls -la "$DATA_DIR" 2>/dev/null || echo "No data directory found"
        
        echo ""
        echo "Tick Data Files:"
        ls -la "$DATA_DIR/ticks" 2>/dev/null || echo "No tick data found"
        
        echo ""
        echo "Funding Data Files:"
        ls -la "$DATA_DIR/funding" 2>/dev/null || echo "No funding data found"
        
        echo ""
        echo "Warehouse Data:"
        ls -la "$DATA_DIR/warehouse" 2>/dev/null || echo "No warehouse data found"
        ;;
    
    clean)
        echo "Cleaning old data files..."
        days=${2:-7}
        
        # Clean old tick files (keep last 7 days by default)
        find "$DATA_DIR/ticks" -name "*.jsonl" -mtime +$days -delete 2>/dev/null || true
        find "$DATA_DIR/ticks" -name "*.parquet" -mtime +$days -delete 2>/dev/null || true
        
        # Clean old funding files
        find "$DATA_DIR/funding" -name "*.json" -mtime +$days -delete 2>/dev/null || true
        
        # Clean old log files
        find "$LOG_DIR" -name "*.log" -mtime +$days -delete 2>/dev/null || true
        
        echo "Cleaned files older than $days days"
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|convert|summary|clean}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the market data capture system"
        echo "  stop      - Stop the market data capture system"
        echo "  restart   - Restart the market data capture system"
        echo "  status    - Check the status of running processes"
        echo "  logs      - Show recent logs from all processes"
        echo "  convert   - Convert tick data to Parquet format"
        echo "  summary   - Show data directory summary"
        echo "  clean     - Clean old data files (default: 7 days)"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 status"
        echo "  $0 logs"
        echo "  $0 convert 2025-09-15"
        echo "  $0 clean 14"
        exit 1
        ;;
esac
