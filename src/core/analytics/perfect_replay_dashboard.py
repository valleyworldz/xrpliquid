"""
Perfect Replay Dashboard
Computes day-selectable realized PnL from captured tick tape.
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerfectReplayDashboard:
    """Dashboard that computes PnL from captured tick tape."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.data_dir = self.repo_root / "data"
        self.reports_dir = self.repo_root / "reports"
        
        # Trading parameters
        self.initial_capital = 100.0  # $100 starting capital
        self.position_size = 1000  # XRP position size
        self.fee_rate = 0.0005  # 0.05% fee rate
    
    def load_tick_data(self, date: str) -> pd.DataFrame:
        """Load tick data for a specific date."""
        tick_file = self.data_dir / "ticks" / f"xrp_{date}.jsonl"
        
        if not tick_file.exists():
            logger.warning(f"Tick data not found for {date}")
            return pd.DataFrame()
        
        # Load tick data
        ticks = []
        with open(tick_file, 'r') as f:
            for line in f:
                try:
                    tick = json.loads(line.strip())
                    ticks.append(tick)
                except json.JSONDecodeError:
                    continue
        
        if not ticks:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(ticks)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"📊 Loaded {len(df)} ticks for {date}")
        return df
    
    def load_funding_data(self, date: str) -> pd.DataFrame:
        """Load funding data for a specific date."""
        funding_file = self.data_dir / "funding" / f"xrp_{date}.json"
        
        if not funding_file.exists():
            logger.warning(f"Funding data not found for {date}")
            return pd.DataFrame()
        
        with open(funding_file, 'r') as f:
            funding_data = json.load(f)
        
        if not funding_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(funding_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
        df['mark_price'] = pd.to_numeric(df['mark_price'], errors='coerce')
        
        logger.info(f"💰 Loaded {len(df)} funding events for {date}")
        return df
    
    def simulate_trading_strategy(self, tick_data: pd.DataFrame, funding_data: pd.DataFrame) -> Dict:
        """Simulate trading strategy on tick data."""
        if tick_data.empty:
            return self.get_empty_results()
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # XRP position
        trades = []
        funding_payments = []
        
        # Simple mean reversion strategy
        price_window = 100  # Look at last 100 ticks
        entry_threshold = 0.001  # 0.1% threshold
        
        for i, (_, tick) in enumerate(tick_data.iterrows()):
            current_price = tick['price']
            current_time = tick['timestamp']
            
            # Calculate moving average
            if i >= price_window:
                recent_prices = tick_data.iloc[i-price_window:i]['price']
                ma_price = recent_prices.mean()
                
                # Entry signal: price below MA
                if current_price < ma_price * (1 - entry_threshold) and position == 0:
                    # Buy XRP
                    position = self.position_size
                    trade_cost = position * current_price
                    fee = trade_cost * self.fee_rate
                    capital -= (trade_cost + fee)
                    
                    trades.append({
                        'timestamp': current_time,
                        'side': 'buy',
                        'size': position,
                        'price': current_price,
                        'fee': fee,
                        'capital': capital
                    })
                
                # Exit signal: price above MA
                elif current_price > ma_price * (1 + entry_threshold) and position > 0:
                    # Sell XRP
                    trade_proceeds = position * current_price
                    fee = trade_proceeds * self.fee_rate
                    capital += (trade_proceeds - fee)
                    
                    trades.append({
                        'timestamp': current_time,
                        'side': 'sell',
                        'size': position,
                        'price': current_price,
                        'fee': fee,
                        'capital': capital
                    })
                    
                    position = 0
            
            # Calculate funding payments
            if not funding_data.empty:
                # Find funding rate for current time
                funding_match = funding_data[
                    funding_data['timestamp'] <= current_time
                ].iloc[-1] if len(funding_data[funding_data['timestamp'] <= current_time]) > 0 else None
                
                if funding_match is not None and position != 0:
                    funding_payment = position * funding_match['mark_price'] * funding_match['funding_rate']
                    capital += funding_payment
                    
                    funding_payments.append({
                        'timestamp': current_time,
                        'funding_rate': funding_match['funding_rate'],
                        'payment': funding_payment,
                        'capital': capital
                    })
        
        # Calculate final results
        final_capital = capital + (position * tick_data.iloc[-1]['price'] if position > 0 else 0)
        total_return = final_capital - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'return_percentage': return_pct,
            'total_trades': len(trades),
            'total_funding_payments': len(funding_payments),
            'trades': trades,
            'funding_payments': funding_payments,
            'final_position': position
        }
    
    def get_empty_results(self) -> Dict:
        """Return empty results when no data is available."""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'total_return': 0.0,
            'return_percentage': 0.0,
            'total_trades': 0,
            'total_funding_payments': 0,
            'trades': [],
            'funding_payments': [],
            'final_position': 0
        }
    
    def generate_replay_report(self, date: str) -> Dict:
        """Generate replay report for a specific date."""
        logger.info(f"🎬 Generating replay report for {date}...")
        
        # Load data
        tick_data = self.load_tick_data(date)
        funding_data = self.load_funding_data(date)
        
        # Simulate strategy
        results = self.simulate_trading_strategy(tick_data, funding_data)
        
        # Generate report
        report = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'data_availability': {
                'ticks_available': len(tick_data),
                'funding_events_available': len(funding_data),
                'data_quality': 'complete' if len(tick_data) > 0 else 'no_data'
            },
            'trading_results': results,
            'performance_metrics': {
                'sharpe_ratio': self.calculate_sharpe_ratio(results['trades']),
                'max_drawdown': self.calculate_max_drawdown(results['trades']),
                'win_rate': self.calculate_win_rate(results['trades']),
                'avg_trade_duration': self.calculate_avg_trade_duration(results['trades'])
            }
        }
        
        return report
    
    def calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio from trades."""
        if len(trades) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(trades)):
            if trades[i]['side'] == 'sell':
                prev_trade = trades[i-1]
                if prev_trade['side'] == 'buy':
                    trade_return = (trades[i]['price'] - prev_trade['price']) / prev_trade['price']
                    returns.append(trade_return)
        
        if not returns:
            return 0.0
        
        import numpy as np
        return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
    
    def calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0.0
        
        capitals = [trade['capital'] for trade in trades]
        if not capitals:
            return 0.0
        
        peak = capitals[0]
        max_dd = 0.0
        
        for capital in capitals:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if len(trades) < 2:
            return 0.0
        
        winning_trades = 0
        total_trades = 0
        
        for i in range(1, len(trades)):
            if trades[i]['side'] == 'sell':
                prev_trade = trades[i-1]
                if prev_trade['side'] == 'buy':
                    total_trades += 1
                    if trades[i]['price'] > prev_trade['price']:
                        winning_trades += 1
        
        return (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
    
    def calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in minutes."""
        if len(trades) < 2:
            return 0.0
        
        durations = []
        for i in range(1, len(trades)):
            if trades[i]['side'] == 'sell':
                prev_trade = trades[i-1]
                if prev_trade['side'] == 'buy':
                    duration = (trades[i]['timestamp'] - prev_trade['timestamp']).total_seconds() / 60
                    durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0.0
    
    def save_replay_report(self, report: Dict) -> Path:
        """Save replay report to file."""
        replay_dir = self.reports_dir / "replay"
        replay_dir.mkdir(exist_ok=True)
        
        report_file = replay_dir / f"replay_{report['date']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Replay report saved: {report_file}")
        return report_file
    
    def generate_replay_dashboard(self, date: str) -> str:
        """Generate HTML dashboard for replay results."""
        report = self.generate_replay_report(date)
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🎬 Perfect Replay Dashboard - {date}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                       gap: 15px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; 
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E8B57; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .negative {{ color: #DC143C; }}
        .positive {{ color: #2E8B57; }}
        .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; 
                   box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Perfect Replay Dashboard</h1>
        <p>Date: {date}</p>
        <p>Initial Capital: ${report['trading_results']['initial_capital']:.2f}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value {'positive' if report['trading_results']['total_return'] >= 0 else 'negative'}">
                ${report['trading_results']['final_capital']:.2f}
            </div>
            <div class="metric-label">Final Capital</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value {'positive' if report['trading_results']['total_return'] >= 0 else 'negative'}">
                ${report['trading_results']['total_return']:.2f}
            </div>
            <div class="metric-label">Total Return</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value {'positive' if report['trading_results']['return_percentage'] >= 0 else 'negative'}">
                {report['trading_results']['return_percentage']:.2f}%
            </div>
            <div class="metric-label">Return %</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value positive">
                {report['trading_results']['total_trades']}
            </div>
            <div class="metric-label">Total Trades</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value positive">
                {report['performance_metrics']['sharpe_ratio']:.2f}
            </div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value negative">
                {report['performance_metrics']['max_drawdown']:.2%}
            </div>
            <div class="metric-label">Max Drawdown</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Data Availability</h2>
        <p><strong>Ticks Available:</strong> {report['data_availability']['ticks_available']:,}</p>
        <p><strong>Funding Events:</strong> {report['data_availability']['funding_events_available']}</p>
        <p><strong>Data Quality:</strong> {report['data_availability']['data_quality']}</p>
    </div>
    
    <div class="section">
        <h2>🎯 Performance Summary</h2>
        <p><strong>Win Rate:</strong> {report['performance_metrics']['win_rate']:.1f}%</p>
        <p><strong>Avg Trade Duration:</strong> {report['performance_metrics']['avg_trade_duration']:.1f} minutes</p>
        <p><strong>Total Funding Payments:</strong> {report['trading_results']['total_funding_payments']}</p>
    </div>
</body>
</html>"""
        
        return html
    
    def save_replay_dashboard(self, date: str) -> Path:
        """Save replay dashboard HTML."""
        html = self.generate_replay_dashboard(date)
        
        dashboard_dir = self.reports_dir / "replay"
        dashboard_dir.mkdir(exist_ok=True)
        
        dashboard_file = dashboard_dir / f"replay_dashboard_{date}.html"
        with open(dashboard_file, 'w') as f:
            f.write(html)
        
        logger.info(f"💾 Replay dashboard saved: {dashboard_file}")
        return dashboard_file


def main():
    """Main function to demonstrate perfect replay."""
    dashboard = PerfectReplayDashboard()
    
    # Generate replay for today
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Generate report
    report = dashboard.generate_replay_report(today)
    dashboard.save_replay_report(report)
    
    # Generate dashboard
    dashboard.save_replay_dashboard(today)
    
    print(f"Replay Results for {today}:")
    print(f"Initial Capital: ${report['trading_results']['initial_capital']:.2f}")
    print(f"Final Capital: ${report['trading_results']['final_capital']:.2f}")
    print(f"Total Return: ${report['trading_results']['total_return']:.2f}")
    print(f"Return %: {report['trading_results']['return_percentage']:.2f}%")
    
    print("✅ Perfect replay demonstration completed")


if __name__ == "__main__":
    main()
