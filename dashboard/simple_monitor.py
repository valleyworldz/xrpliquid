#!/usr/bin/env python3
"""
Simple Performance Monitoring Dashboard for XRP Trading Bot
"""

from flask import Flask, render_template, jsonify
import json
import os
import time
from datetime import datetime, timedelta
import csv

app = Flask(__name__)

class BotMonitor:
    """Monitor class for bot performance tracking"""
    
    def __init__(self):
        self.trades_file = "trades_log.csv"
        self.config_file = "config/trading_config.json"
    
    def get_recent_trades(self, limit=20):
        """Get recent trades from CSV file"""
        trades = []
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trades.append(row)
                # Return most recent trades first
                return trades[-limit:][::-1]
        except Exception as e:
            print(f"Error reading trades: {e}")
        return trades
    
    def get_performance_stats(self):
        """Calculate performance statistics"""
        trades = self.get_recent_trades(1000)  # Get all trades
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if float(trade.get('pnl', 0)) > 0)
        total_pnl = sum(float(trade.get('pnl', 0)) for trade in trades)
        
        pnl_values = [float(trade.get('pnl', 0)) for trade in trades]
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'best_trade': max(pnl_values) if pnl_values else 0,
            'worst_trade': min(pnl_values) if pnl_values else 0
        }
        
        return stats
    
    def get_config(self):
        """Get bot configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading config: {e}")
        return {}

# Initialize monitor
monitor = BotMonitor()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    stats = monitor.get_performance_stats()
    config = monitor.get_config()
    recent_trades = monitor.get_recent_trades(10)
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         config=config, 
                         trades=recent_trades)

@app.route('/api/stats')
def api_stats():
    """API endpoint for performance statistics"""
    return jsonify(monitor.get_performance_stats())

@app.route('/api/trades')
def api_trades():
    """API endpoint for recent trades"""
    trades = monitor.get_recent_trades(50)
    return jsonify(trades)

@app.route('/api/config')
def api_config():
    """API endpoint for configuration"""
    return jsonify(monitor.get_config())

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('dashboard/templates', exist_ok=True)
    
    # Create simple HTML template
    template_html = """
<!DOCTYPE html>
<html>
<head>
    <title>XRP Trading Bot Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #f5f5f5; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #333; }
        .stat-label { color: #666; margin-top: 5px; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .trades-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .trades-table th, .trades-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .trades-table th { background-color: #f2f2f2; }
        .refresh-btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🚀 XRP Trading Bot Dashboard</h1>
    
    <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
    
    <h2>📊 Performance Statistics</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{{ stats.total_trades }}</div>
            <div class="stat-label">Total Trades</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ "%.1f"|format(stats.win_rate) }}%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {{ 'positive' if stats.total_pnl > 0 else 'negative' }}">${{ "%.2f"|format(stats.total_pnl) }}</div>
            <div class="stat-label">Total PnL</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${{ "%.2f"|format(stats.avg_pnl) }}</div>
            <div class="stat-label">Average PnL</div>
        </div>
        <div class="stat-card">
            <div class="stat-value positive">${{ "%.2f"|format(stats.best_trade) }}</div>
            <div class="stat-label">Best Trade</div>
        </div>
        <div class="stat-card">
            <div class="stat-value negative">${{ "%.2f"|format(stats.worst_trade) }}</div>
            <div class="stat-label">Worst Trade</div>
        </div>
    </div>
    
    <h2>📈 Recent Trades</h2>
    <table class="trades-table">
        <thead>
            <tr>
                <th>Time</th>
                <th>Side</th>
                <th>Size</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>PnL</th>
                <th>R:R</th>
            </tr>
        </thead>
        <tbody>
            {% for trade in trades %}
            <tr>
                <td>{{ trade.timestamp }}</td>
                <td>{{ trade.side }}</td>
                <td>{{ trade.size }}</td>
                <td>${{ "%.4f"|format(trade.entry|float) }}</td>
                <td>${{ "%.4f"|format(trade.exit|float) }}</td>
                <td class="{{ 'positive' if trade.pnl|float > 0 else 'negative' }}">${{ "%.2f"|format(trade.pnl|float) }}</td>
                <td>{{ "%.2f"|format(trade.rr|float) }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
    """
    
    with open('dashboard/templates/dashboard.html', 'w') as f:
        f.write(template_html)
    
    print("🚀 Starting XRP Trading Bot Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔄 Auto-refresh every 30 seconds")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 