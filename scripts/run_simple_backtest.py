#!/usr/bin/env python3
"""
📊 SIMPLE BACKTEST RUNNER
=========================
Generates comprehensive backtest results with trade ledger and tearsheets.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

def generate_synthetic_trades(start_date, end_date, initial_capital=10000):
    """Generate synthetic trade data for demonstration"""
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate synthetic price data
    np.random.seed(42)
    base_price = 0.5
    returns = np.random.normal(0, 0.02, len(date_range))  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate trades
    trades = []
    balance = initial_capital
    position = 0.0
    
    for i, (timestamp, price) in enumerate(zip(date_range, prices)):
        # Generate trade signals
        if i % 24 == 0:  # Trade once per day
            # Random strategy selection
            strategy = np.random.choice(['BUY', 'SCALP', 'FUNDING_ARBITRAGE'])
            
            # Random side
            side = np.random.choice(['buy', 'sell'])
            
            # Position sizing
            if side == 'buy' and balance > 100:
                quantity = min(balance * 0.1 / price, 1000)  # 10% of balance
                cost = quantity * price
                fee = cost * 0.0005  # 0.05% fee
                total_cost = cost + fee
                
                if total_cost <= balance:
                    balance -= total_cost
                    position += quantity
                    
                    # Calculate slippage
                    slippage_bps = np.random.uniform(1, 5)  # 1-5 bps slippage
                    fill_price = price * (1 + slippage_bps / 10000)
                    
                    # Calculate funding (if applicable)
                    funding = 0.0
                    if strategy == 'FUNDING_ARBITRAGE':
                        funding = quantity * price * 0.0001  # 0.01% funding
                    
                    trade = {
                        'ts': timestamp.timestamp(),
                        'strategy_name': strategy,
                        'side': side,
                        'qty': quantity,
                        'price': price,
                        'fee': fee,
                        'fee_bps': (fee / cost) * 10000 if cost > 0 else 0,
                        'funding': funding,
                        'slippage_bps': slippage_bps,
                        'pnl_realized': 0.0,  # Will calculate later
                        'pnl_unrealized': 0.0,
                        'reason_code': 'strategy_signal',
                        'maker_flag': np.random.random() > 0.3,  # 70% maker
                        'order_state': 'filled',
                        'regime_label': 'normal',
                        'symbol': 'XRP',
                        'leverage': 1.0,
                        'margin_used': 0.0,
                        'position_size': position,
                        'account_balance': balance,
                        'latency_ms': np.random.uniform(10, 100),
                        'retry_count': 0,
                        'error_code': None,
                        'var_95': 0.0,
                        'max_drawdown': 0.0,
                        'sharpe_ratio': 0.0,
                        'volatility': 0.02,
                        'volume_24h': 1000000,
                        'spread_bps': 2.0,
                    }
                    trades.append(trade)
            
            elif side == 'sell' and position > 0:
                quantity = min(position * 0.5, 1000)  # Sell 50% of position
                proceeds = quantity * price
                fee = proceeds * 0.0005  # 0.05% fee
                net_proceeds = proceeds - fee
                
                balance += net_proceeds
                position -= quantity
                
                # Calculate slippage
                slippage_bps = np.random.uniform(1, 5)  # 1-5 bps slippage
                fill_price = price * (1 - slippage_bps / 10000)
                
                # Calculate P&L
                pnl = (fill_price - base_price) * quantity - fee
                
                trade = {
                    'ts': timestamp.timestamp(),
                    'strategy_name': strategy,
                    'side': side,
                    'qty': quantity,
                    'price': price,
                    'fee': fee,
                    'fee_bps': (fee / proceeds) * 10000 if proceeds > 0 else 0,
                    'funding': 0.0,
                    'slippage_bps': slippage_bps,
                    'pnl_realized': pnl,
                    'pnl_unrealized': 0.0,
                    'reason_code': 'strategy_signal',
                    'maker_flag': np.random.random() > 0.3,  # 70% maker
                    'order_state': 'filled',
                    'regime_label': 'normal',
                    'symbol': 'XRP',
                    'leverage': 1.0,
                    'margin_used': 0.0,
                    'position_size': position,
                    'account_balance': balance,
                    'latency_ms': np.random.uniform(10, 100),
                    'retry_count': 0,
                    'error_code': None,
                    'var_95': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'volatility': 0.02,
                    'volume_24h': 1000000,
                    'spread_bps': 2.0,
                }
                trades.append(trade)
    
    return trades

def generate_tearsheet(trades, initial_capital):
    """Generate HTML tearsheet"""
    
    if not trades:
        return "<html><body><h1>No trades found</h1></body></html>"
    
    df = pd.DataFrame(trades)
    
    # Calculate performance metrics
    total_return = (df['account_balance'].iloc[-1] - initial_capital) / initial_capital
    total_trades = len(df)
    winning_trades = len(df[df['pnl_realized'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate Sharpe ratio (simplified)
    returns = df['pnl_realized'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Strategy performance
    strategy_perf = df.groupby('strategy_name').agg({
        'pnl_realized': 'sum',
        'fee': 'sum',
        'qty': 'count'
    }).to_dict()
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>📊 Comprehensive Backtest Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
            .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; min-width: 150px; text-align: center; }}
            .performance {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }}
            .risk {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Comprehensive Backtest Results</h1>
            <p>Hat Manifesto Ultimate Trading System</p>
            <p>📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>📊 Period: {datetime.fromtimestamp(df['ts'].min()).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(df['ts'].max()).strftime('%Y-%m-%d')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Performance Summary</h2>
            <div class="metric performance">
                <h3>Total Return</h3>
                <h2>{total_return:.2%}</h2>
            </div>
            <div class="metric performance">
                <h3>Sharpe Ratio</h3>
                <h2>{sharpe_ratio:.2f}</h2>
            </div>
            <div class="metric risk">
                <h3>Max Drawdown</h3>
                <h2>{max_drawdown:.2%}</h2>
            </div>
            <div class="metric performance">
                <h3>Win Rate</h3>
                <h2>{win_rate:.2%}</h2>
            </div>
            <div class="metric performance">
                <h3>Total Trades</h3>
                <h2>{total_trades}</h2>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Strategy Performance</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f8f9fa;">
                    <th style="padding: 10px; border: 1px solid #ddd;">Strategy</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Trades</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Total P&L</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Total Fees</th>
                </tr>
    """
    
    for strategy, data in strategy_perf['qty'].items():
        html += f"""
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">{strategy}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{data}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${strategy_perf['pnl_realized'][strategy]:.2f}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${strategy_perf['fee'][strategy]:.2f}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Trade Analysis</h2>
            <p>Total Volume: ${:,.2f}</p>
            <p>Total Fees: ${:,.2f}</p>
            <p>Average Slippage: {:.2f} bps</p>
            <p>Maker Ratio: {:.1%}</p>
        </div>
    </body>
    </html>
    """.format(
        (df['qty'] * df['price']).sum(),
        df['fee'].sum(),
        df['slippage_bps'].mean(),
        len(df[df['maker_flag'] == True]) / len(df) if len(df) > 0 else 0
    )
    
    return html

def main():
    """Main function"""
    print("📊 [SIMPLE_BACKTEST] Starting comprehensive backtest...")
    
    # Configuration
    start_date = "2022-01-01"
    end_date = "2025-09-15"
    initial_capital = 10000.0
    
    # Generate trades
    print("📊 [SIMPLE_BACKTEST] Generating synthetic trades...")
    trades = generate_synthetic_trades(start_date, end_date, initial_capital)
    
    if not trades:
        print("❌ [SIMPLE_BACKTEST] No trades generated")
        return
    
    print(f"📊 [SIMPLE_BACKTEST] Generated {len(trades)} trades")
    
    # Create reports directory
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Save trade ledger
    df = pd.DataFrame(trades)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV
    csv_path = reports_dir / f'comprehensive_backtest_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"📊 [SIMPLE_BACKTEST] Saved trade ledger: {csv_path}")
    
    # Save Parquet
    parquet_path = reports_dir / f'comprehensive_backtest_{timestamp}.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"📊 [SIMPLE_BACKTEST] Saved trade ledger: {parquet_path}")
    
    # Generate tearsheet
    print("📊 [SIMPLE_BACKTEST] Generating tearsheet...")
    tearsheet = generate_tearsheet(trades, initial_capital)
    
    # Save tearsheet
    tearsheet_path = reports_dir / f'comprehensive_backtest_{timestamp}.html'
    with open(tearsheet_path, 'w', encoding='utf-8') as f:
        f.write(tearsheet)
    print(f"📊 [SIMPLE_BACKTEST] Saved tearsheet: {tearsheet_path}")
    
    # Generate JSON report
    report = {
        'backtest_info': {
            'timestamp': timestamp,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
        },
        'performance_metrics': {
            'total_return': (df['account_balance'].iloc[-1] - initial_capital) / initial_capital,
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl_realized'] > 0]),
            'total_fees': df['fee'].sum(),
            'total_pnl': df['pnl_realized'].sum(),
        },
        'strategy_performance': df.groupby('strategy_name').agg({
            'pnl_realized': 'sum',
            'fee': 'sum',
            'qty': 'count'
        }).to_dict(),
    }
    
    json_path = reports_dir / f'comprehensive_backtest_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📊 [SIMPLE_BACKTEST] Saved JSON report: {json_path}")
    
    print("📊 [SIMPLE_BACKTEST] Comprehensive backtest completed successfully!")
    print(f"📊 [SIMPLE_BACKTEST] Total Return: {report['performance_metrics']['total_return']:.2%}")
    print(f"📊 [SIMPLE_BACKTEST] Total Trades: {report['performance_metrics']['total_trades']}")
    print(f"📊 [SIMPLE_BACKTEST] Win Rate: {report['performance_metrics']['winning_trades'] / report['performance_metrics']['total_trades']:.2%}")

if __name__ == "__main__":
    main()
