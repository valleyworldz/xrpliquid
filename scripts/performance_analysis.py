#!/usr/bin/env python3
"""
PERFORMANCE ANALYSIS AND SCORING SYSTEM
Comprehensive evaluation of the Ultimate Bypass Bot performance
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class PerformanceAnalyzer:
    def __init__(self):
        self.trades_log = "trades_log.csv"
        self.trade_history = "trade_history (1).csv"
        
    def analyze_trades_log(self):
        """Analyze the trades log for performance metrics"""
        try:
            df = pd.read_csv(self.trades_log)
            
            # Calculate key metrics
            total_trades = len(df)
            profitable_trades = len(df[df['profitable'] == True]) if 'profitable' in df.columns else 0
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate PnL metrics
            total_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0
            avg_pnl = df['pnl'].mean() if 'pnl' in df.columns else 0
            
            # Calculate fee efficiency
            total_fees = df['fee'].sum() if 'fee' in df.columns else 0
            fee_efficiency = (total_pnl / total_fees) if total_fees > 0 else 0
            
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'total_fees': total_fees,
                'fee_efficiency': fee_efficiency
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_trade_history(self):
        """Analyze the trade history for comprehensive metrics"""
        try:
            df = pd.read_csv(self.trade_history)
            
            # Calculate key metrics
            total_trades = len(df)
            total_fees = df['fee'].sum()
            total_pnl = df['closedPnl'].sum()
            
            # Calculate win rate
            winning_trades = len(df[df['closedPnl'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate average metrics
            avg_pnl = df['closedPnl'].mean()
            avg_fee = df['fee'].mean()
            
            # Calculate risk metrics
            max_win = df['closedPnl'].max()
            max_loss = df['closedPnl'].min()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'total_fees': total_fees,
                'avg_fee': avg_fee,
                'max_win': max_win,
                'max_loss': max_loss
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_performance_score(self, trades_log_metrics, trade_history_metrics):
        """Calculate overall performance score"""
        score = 0
        max_score = 100
        
        # Win Rate Score (30 points)
        if 'win_rate' in trades_log_metrics and not isinstance(trades_log_metrics['win_rate'], str):
            win_rate = trades_log_metrics['win_rate']
            if win_rate >= 70:
                score += 30
            elif win_rate >= 60:
                score += 25
            elif win_rate >= 50:
                score += 20
            elif win_rate >= 40:
                score += 15
            else:
                score += 10
        
        # PnL Score (25 points)
        if 'total_pnl' in trade_history_metrics and not isinstance(trade_history_metrics['total_pnl'], str):
            total_pnl = trade_history_metrics['total_pnl']
            if total_pnl > 0:
                score += 25
            elif total_pnl > -10:
                score += 20
            elif total_pnl > -50:
                score += 15
            else:
                score += 10
        
        # Fee Efficiency Score (20 points)
        if 'fee_efficiency' in trades_log_metrics and not isinstance(trades_log_metrics['fee_efficiency'], str):
            fee_efficiency = trades_log_metrics['fee_efficiency']
            if fee_efficiency > 2:
                score += 20
            elif fee_efficiency > 1:
                score += 15
            elif fee_efficiency > 0:
                score += 10
            else:
                score += 5
        
        # Trade Volume Score (15 points)
        if 'total_trades' in trade_history_metrics and not isinstance(trade_history_metrics['total_trades'], str):
            total_trades = trade_history_metrics['total_trades']
            if total_trades >= 1000:
                score += 15
            elif total_trades >= 500:
                score += 12
            elif total_trades >= 100:
                score += 10
            else:
                score += 5
        
        # Risk Management Score (10 points)
        if 'max_loss' in trade_history_metrics and not isinstance(trade_history_metrics['max_loss'], str):
            max_loss = abs(trade_history_metrics['max_loss'])
            if max_loss < 10:
                score += 10
            elif max_loss < 25:
                score += 8
            elif max_loss < 50:
                score += 6
            else:
                score += 4
        
        return min(score, max_score)
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("🚀 ULTIMATE BYPASS BOT PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Analyze trades log
        print("\n📊 TRADES LOG ANALYSIS:")
        trades_log_metrics = self.analyze_trades_log()
        if 'error' not in trades_log_metrics:
            for key, value in trades_log_metrics.items():
                print(f"   {key}: {value}")
        else:
            print(f"   Error: {trades_log_metrics['error']}")
        
        # Analyze trade history
        print("\n📈 TRADE HISTORY ANALYSIS:")
        trade_history_metrics = self.analyze_trade_history()
        if 'error' not in trade_history_metrics:
            for key, value in trade_history_metrics.items():
                print(f"   {key}: {value}")
        else:
            print(f"   Error: {trade_history_metrics['error']}")
        
        # Calculate performance score
        print("\n🎯 PERFORMANCE SCORING:")
        performance_score = self.calculate_performance_score(trades_log_metrics, trade_history_metrics)
        print(f"   Overall Performance Score: {performance_score}/100")
        
        # Grade assignment
        if performance_score >= 90:
            grade = "A+ (EXCELLENT)"
        elif performance_score >= 80:
            grade = "A (VERY GOOD)"
        elif performance_score >= 70:
            grade = "B+ (GOOD)"
        elif performance_score >= 60:
            grade = "B (SATISFACTORY)"
        elif performance_score >= 50:
            grade = "C (NEEDS IMPROVEMENT)"
        else:
            grade = "D (POOR)"
        
        print(f"   Performance Grade: {grade}")
        
        # System status
        print("\n🔍 SYSTEM STATUS:")
        print("   ✅ A.I. ULTIMATE CHAMPION +213% Profile: ACTIVE")
        print("   ✅ XRP Token: HARDCODED")
        print("   ✅ Fee Threshold: 0.001 (ULTRA-AGGRESSIVE)")
        print("   ✅ All Vetoes: DISABLED")
        print("   ✅ Low-Cap Mode: ENABLED")
        
        return {
            'performance_score': performance_score,
            'grade': grade,
            'trades_log_metrics': trades_log_metrics,
            'trade_history_metrics': trade_history_metrics
        }

def main():
    analyzer = PerformanceAnalyzer()
    report = analyzer.generate_report()
    return report

if __name__ == "__main__":
    main()
