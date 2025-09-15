
# Comprehensive Performance Tracking
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class PerformanceTracker:
    def __init__(self):
        self.performance_data = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'daily_returns': [],
            'trade_history': []
        }
        
    def update_performance(self, trade_result):
        """Update performance metrics with new trade"""
        try:
            self.performance_data['total_trades'] += 1
            
            if trade_result['pnl'] > 0:
                self.performance_data['winning_trades'] += 1
                self.performance_data['total_profit'] += trade_result['pnl']
            else:
                self.performance_data['losing_trades'] += 1
                self.performance_data['total_loss'] += abs(trade_result['pnl'])
            
            self.calculate_metrics()
            self.save_performance_data()
            
        except Exception as e:
            print(f"Performance update error: {e}")
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        try:
            total_trades = self.performance_data['total_trades']
            if total_trades > 0:
                self.performance_data['win_rate'] = self.performance_data['winning_trades'] / total_trades
            
            total_loss = self.performance_data['total_loss']
            if total_loss > 0:
                self.performance_data['profit_factor'] = self.performance_data['total_profit'] / total_loss
            
            if len(self.performance_data['daily_returns']) > 1:
                returns = self.performance_data['daily_returns']
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                if std_return > 0:
                    self.performance_data['sharpe_ratio'] = avg_return / std_return
            
        except Exception as e:
            print(f"Metrics calculation error: {e}")
    
    def save_performance_data(self):
        """Save performance data to file"""
        try:
            os.makedirs('reports/performance', exist_ok=True)
            
            filename = f'reports/performance/performance_{datetime.now().strftime("%Y%m%d")}.json'
            
            with open(filename, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
                
        except Exception as e:
            print(f"Performance save error: {e}")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_trades': self.performance_data['total_trades'],
                    'win_rate': f"{self.performance_data['win_rate']:.2%}",
                    'profit_factor': f"{self.performance_data['profit_factor']:.2f}",
                    'total_profit': f"${self.performance_data['total_profit']:.2f}",
                    'sharpe_ratio': f"{self.performance_data['sharpe_ratio']:.2f}",
                    'max_drawdown': f"{self.performance_data['max_drawdown']:.2%}"
                },
                'detailed_metrics': self.performance_data
            }
            
            os.makedirs('reports', exist_ok=True)
            filename = f'reports/performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            print("Performance report generated")
            return report
            
        except Exception as e:
            print(f"Report generation error: {e}")
            return None
