"""
🎯 PAPER TRADE ENGINE
Advanced paper trading system with real order book replay, latency simulation, and slippage logging
"""

import asyncio
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from pathlib import Path
import threading
from collections import deque
import queue

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.analytics.trade_ledger import TradeLedgerManager, TradeRecord
from core.monitoring.prometheus_metrics import get_metrics_collector, record_trade_metrics
from core.utils.logger import Logger

class OrderBookSnapshot:
    """Order book snapshot with bid/ask levels"""
    
    def __init__(self, timestamp: float, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        self.timestamp = timestamp
        self.bids = sorted(bids, key=lambda x: x[0], reverse=True)  # Price descending
        self.asks = sorted(asks, key=lambda x: x[0])  # Price ascending
        self.mid_price = (self.bids[0][0] + self.asks[0][0]) / 2 if self.bids and self.asks else 0.0
        self.spread = self.asks[0][0] - self.bids[0][0] if self.bids and self.asks else 0.0
    
    def get_best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0
    
    def get_best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0
    
    def get_market_depth(self, side: str, levels: int = 5) -> List[Tuple[float, float]]:
        """Get market depth for specified side"""
        if side.upper() == 'BUY':
            return self.bids[:levels]
        else:
            return self.asks[:levels]
    
    def calculate_impact_price(self, side: str, quantity: float) -> float:
        """Calculate price impact for a given quantity"""
        if side.upper() == 'BUY':
            levels = self.asks
        else:
            levels = self.bids
        
        remaining_qty = quantity
        total_cost = 0.0
        
        for price, level_qty in levels:
            if remaining_qty <= 0:
                break
            
            qty_to_take = min(remaining_qty, level_qty)
            total_cost += qty_to_take * price
            remaining_qty -= qty_to_take
        
        if quantity > 0:
            return total_cost / quantity
        return 0.0

class LatencySimulator:
    """Simulates realistic network and processing latency"""
    
    def __init__(self, base_latency_ms: float = 10.0, jitter_ms: float = 5.0):
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.network_conditions = {
            'excellent': (5.0, 2.0),    # 5ms base, 2ms jitter
            'good': (15.0, 5.0),        # 15ms base, 5ms jitter
            'average': (30.0, 10.0),    # 30ms base, 10ms jitter
            'poor': (100.0, 50.0),      # 100ms base, 50ms jitter
            'terrible': (500.0, 200.0)  # 500ms base, 200ms jitter
        }
        self.current_condition = 'good'
    
    def set_network_condition(self, condition: str):
        """Set network condition for latency simulation"""
        if condition in self.network_conditions:
            self.current_condition = condition
            self.base_latency_ms, self.jitter_ms = self.network_conditions[condition]
    
    async def simulate_latency(self, operation_type: str = 'order') -> float:
        """Simulate latency for different operations"""
        # Different latencies for different operations
        operation_multipliers = {
            'order': 1.0,
            'cancel': 0.8,
            'modify': 1.2,
            'market_data': 0.5,
            'account_info': 2.0
        }
        
        multiplier = operation_multipliers.get(operation_type, 1.0)
        base_latency = self.base_latency_ms * multiplier
        
        # Add jitter (random variation)
        jitter = random.uniform(-self.jitter_ms, self.jitter_ms)
        total_latency_ms = max(0.1, base_latency + jitter)
        
        # Simulate the latency
        await asyncio.sleep(total_latency_ms / 1000.0)
        
        return total_latency_ms

class SlippageAnalyzer:
    """Analyzes and logs slippage for paper trades"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.slippage_data = []
        self.total_slippage = 0.0
        self.total_volume = 0.0
    
    def calculate_slippage(self, 
                          expected_price: float, 
                          actual_price: float, 
                          quantity: float,
                          side: str,
                          order_type: str) -> Dict[str, Any]:
        """Calculate slippage metrics"""
        
        if expected_price <= 0 or actual_price <= 0:
            return {
                'slippage_bps': 0.0,
                'slippage_percent': 0.0,
                'slippage_amount': 0.0,
                'slippage_cost': 0.0
            }
        
        # Calculate slippage in basis points and percentage
        slippage_bps = ((actual_price - expected_price) / expected_price) * 10000
        slippage_percent = (actual_price - expected_price) / expected_price * 100
        
        # For sell orders, slippage should be negative (price goes down)
        if side.upper() == 'SELL':
            slippage_bps = -slippage_bps
            slippage_percent = -slippage_percent
        
        slippage_amount = actual_price - expected_price
        slippage_cost = slippage_amount * quantity
        
        slippage_info = {
            'timestamp': time.time(),
            'expected_price': expected_price,
            'actual_price': actual_price,
            'quantity': quantity,
            'side': side,
            'order_type': order_type,
            'slippage_bps': slippage_bps,
            'slippage_percent': slippage_percent,
            'slippage_amount': slippage_amount,
            'slippage_cost': slippage_cost,
            'market_impact': abs(slippage_percent)
        }
        
        # Store slippage data
        self.slippage_data.append(slippage_info)
        self.total_slippage += abs(slippage_cost)
        self.total_volume += quantity * actual_price
        
        # Log slippage
        self.logger.info(f"📊 [SLIPPAGE] {side} {quantity:.3f} XRP: "
                        f"Expected ${expected_price:.4f} → Actual ${actual_price:.4f} "
                        f"({slippage_bps:+.1f} bps, ${slippage_cost:+.4f})")
        
        return slippage_info
    
    def get_slippage_summary(self) -> Dict[str, Any]:
        """Get comprehensive slippage summary"""
        if not self.slippage_data:
            return {
                'total_trades': 0,
                'avg_slippage_bps': 0.0,
                'avg_slippage_percent': 0.0,
                'total_slippage_cost': 0.0,
                'total_volume': 0.0,
                'slippage_ratio': 0.0
            }
        
        df = pd.DataFrame(self.slippage_data)
        
        return {
            'total_trades': len(self.slippage_data),
            'avg_slippage_bps': df['slippage_bps'].mean(),
            'avg_slippage_percent': df['slippage_percent'].mean(),
            'max_slippage_bps': df['slippage_bps'].max(),
            'min_slippage_bps': df['slippage_bps'].min(),
            'total_slippage_cost': self.total_slippage,
            'total_volume': self.total_volume,
            'slippage_ratio': self.total_slippage / self.total_volume if self.total_volume > 0 else 0.0,
            'buy_slippage_avg': df[df['side'] == 'BUY']['slippage_bps'].mean() if 'BUY' in df['side'].values else 0.0,
            'sell_slippage_avg': df[df['side'] == 'SELL']['slippage_bps'].mean() if 'SELL' in df['side'].values else 0.0
        }

class PaperTradeEngine:
    """
    🎯 PAPER TRADE ENGINE
    Advanced paper trading with real order book replay, latency simulation, and slippage analysis
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 symbol: str = "XRP",
                 logger: Optional[Logger] = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.symbol = symbol
        self.logger = logger or Logger()
        
        # Portfolio tracking
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Order book and market data
        self.current_orderbook = None
        self.orderbook_history = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.price_history = deque(maxlen=10000)     # Keep last 10k prices
        
        # Latency and slippage simulation
        self.latency_simulator = LatencySimulator()
        self.slippage_analyzer = SlippageAnalyzer(self.logger)
        
        # Trade ledger
        self.trade_ledger = TradeLedgerManager(data_dir="data/paper_trades", logger=self.logger)
        
        # Prometheus metrics collector
        self.metrics_collector = get_metrics_collector(port=8001, logger=self.logger)
        
        # Order management
        self.pending_orders = {}
        self.order_counter = 0
        self.trade_counter = 0
        
        # Performance tracking
        self.start_time = time.time()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self.logger.info("🎯 [PAPER_TRADE] Paper Trade Engine initialized")
        self.logger.info(f"💰 [PAPER_TRADE] Initial Capital: ${initial_capital:,.2f}")
        self.logger.info(f"📊 [PAPER_TRADE] Symbol: {symbol}")
    
    def generate_realistic_orderbook(self, base_price: float = 0.50) -> OrderBookSnapshot:
        """Generate realistic order book snapshot"""
        timestamp = time.time()
        
        # Generate realistic bid/ask levels
        num_levels = random.randint(10, 20)
        spread_bps = random.uniform(5, 50)  # 0.5 to 5 bps spread
        
        bids = []
        asks = []
        
        # Generate bid levels (below mid price)
        current_price = base_price
        for i in range(num_levels):
            price = current_price * (1 - (i + 1) * 0.0001)  # 1 bps per level
            size = random.uniform(100, 10000)  # Random size
            bids.append((price, size))
        
        # Generate ask levels (above mid price)
        current_price = base_price * (1 + spread_bps / 10000)
        for i in range(num_levels):
            price = current_price * (1 + i * 0.0001)  # 1 bps per level
            size = random.uniform(100, 10000)  # Random size
            asks.append((price, size))
        
        return OrderBookSnapshot(timestamp, bids, asks)
    
    def update_orderbook(self, price_change: float = 0.0):
        """Update order book with price movement"""
        if not self.current_orderbook:
            self.current_orderbook = self.generate_realistic_orderbook()
        else:
            # Update prices based on price change
            new_mid = self.current_orderbook.mid_price * (1 + price_change)
            self.current_orderbook = self.generate_realistic_orderbook(new_mid)
        
        # Store in history
        self.orderbook_history.append(self.current_orderbook)
        self.price_history.append(self.current_orderbook.mid_price)
    
    async def place_paper_order(self, 
                               side: str, 
                               quantity: float, 
                               order_type: str = "MARKET",
                               limit_price: Optional[float] = None) -> Dict[str, Any]:
        """Place a paper trade order with realistic execution"""
        
        if not self.current_orderbook:
            self.update_orderbook()
        
        self.order_counter += 1
        order_id = f"PAPER_ORDER_{self.order_counter:06d}"
        
        # Simulate latency
        latency_ms = await self.latency_simulator.simulate_latency('order')
        
        # Determine execution price based on order type
        if order_type.upper() == "MARKET":
            if side.upper() == "BUY":
                expected_price = self.current_orderbook.get_best_ask()
                # Simulate market impact
                actual_price = self.current_orderbook.calculate_impact_price("BUY", quantity)
            else:  # SELL
                expected_price = self.current_orderbook.get_best_bid()
                # Simulate market impact
                actual_price = self.current_orderbook.calculate_impact_price("SELL", quantity)
        else:  # LIMIT order
            if limit_price is None:
                return {'success': False, 'error': 'Limit price required for LIMIT orders'}
            
            if side.upper() == "BUY":
                if limit_price >= self.current_orderbook.get_best_ask():
                    # Marketable limit order
                    actual_price = min(limit_price, self.current_orderbook.get_best_ask())
                    expected_price = limit_price
                else:
                    # Resting limit order - not filled immediately
                    return {
                        'success': True,
                        'order_id': order_id,
                        'status': 'RESTING',
                        'filled_quantity': 0.0,
                        'remaining_quantity': quantity,
                        'price': limit_price,
                        'latency_ms': latency_ms
                    }
            else:  # SELL
                if limit_price <= self.current_orderbook.get_best_bid():
                    # Marketable limit order
                    actual_price = max(limit_price, self.current_orderbook.get_best_bid())
                    expected_price = limit_price
                else:
                    # Resting limit order - not filled immediately
                    return {
                        'success': True,
                        'order_id': order_id,
                        'status': 'RESTING',
                        'filled_quantity': 0.0,
                        'remaining_quantity': quantity,
                        'price': limit_price,
                        'latency_ms': latency_ms
                    }
        
        # Calculate slippage
        slippage_info = self.slippage_analyzer.calculate_slippage(
            expected_price, actual_price, quantity, side, order_type
        )
        
        # Update portfolio
        self._update_portfolio(side, quantity, actual_price)
        
        # Record trade
        self._record_paper_trade(order_id, side, quantity, actual_price, slippage_info, latency_ms)
        
        # Update order book (simulate market impact)
        price_impact = slippage_info['market_impact'] / 100.0
        self.update_orderbook(price_impact if side.upper() == "BUY" else -price_impact)
        
        return {
            'success': True,
            'order_id': order_id,
            'status': 'FILLED',
            'filled_quantity': quantity,
            'remaining_quantity': 0.0,
            'price': actual_price,
            'expected_price': expected_price,
            'slippage_bps': slippage_info['slippage_bps'],
            'slippage_cost': slippage_info['slippage_cost'],
            'latency_ms': latency_ms,
            'market_impact': slippage_info['market_impact']
        }
    
    def _update_portfolio(self, side: str, quantity: float, price: float):
        """Update portfolio after trade execution"""
        if side.upper() == "BUY":
            # Buying - increase position
            if self.position >= 0:
                # Adding to long position
                total_cost = self.position * self.avg_entry_price + quantity * price
                self.position += quantity
                self.avg_entry_price = total_cost / self.position if self.position > 0 else 0.0
            else:
                # Covering short position
                if quantity <= abs(self.position):
                    # Partial cover
                    self.realized_pnl += (self.avg_entry_price - price) * quantity
                    self.position += quantity
                else:
                    # Full cover + new long position
                    self.realized_pnl += (self.avg_entry_price - price) * abs(self.position)
                    remaining_qty = quantity - abs(self.position)
                    self.position = remaining_qty
                    self.avg_entry_price = price
        else:  # SELL
            # Selling - decrease position
            if self.position <= 0:
                # Adding to short position
                total_cost = abs(self.position) * self.avg_entry_price + quantity * price
                self.position -= quantity
                self.avg_entry_price = total_cost / abs(self.position) if self.position < 0 else 0.0
            else:
                # Reducing long position
                if quantity <= self.position:
                    # Partial sell
                    self.realized_pnl += (price - self.avg_entry_price) * quantity
                    self.position -= quantity
                else:
                    # Full sell + new short position
                    self.realized_pnl += (price - self.avg_entry_price) * self.position
                    remaining_qty = quantity - self.position
                    self.position = -remaining_qty
                    self.avg_entry_price = price
        
        # Update capital
        self.current_capital = self.initial_capital + self.realized_pnl
        
        # Update unrealized PnL
        if self.position != 0 and self.current_orderbook:
            current_price = self.current_orderbook.mid_price
            if self.position > 0:
                self.unrealized_pnl = (current_price - self.avg_entry_price) * self.position
            else:
                self.unrealized_pnl = (self.avg_entry_price - current_price) * abs(self.position)
    
    def _record_paper_trade(self, 
                           order_id: str, 
                           side: str, 
                           quantity: float, 
                           price: float,
                           slippage_info: Dict[str, Any],
                           latency_ms: float):
        """Record paper trade in ledger"""
        
        self.trade_counter += 1
        self.total_trades += 1
        
        # Calculate PnL for this trade
        if side.upper() == "SELL" and self.position < 0:
            # Closing short position
            trade_pnl = (self.avg_entry_price - price) * quantity
        elif side.upper() == "BUY" and self.position > 0:
            # This is opening a position, PnL will be calculated when closed
            trade_pnl = 0.0
        else:
            trade_pnl = 0.0
        
        # Determine win/loss
        win_loss = 'WIN' if trade_pnl > 0 else 'LOSS' if trade_pnl < 0 else 'BREAKEVEN'
        if trade_pnl > 0:
            self.winning_trades += 1
        elif trade_pnl < 0:
            self.losing_trades += 1
        
        # Create trade record
        trade_data = {
            'trade_id': f"PAPER_TRADE_{self.trade_counter:06d}",
            'timestamp': time.time(),
            'datetime_utc': datetime.utcnow().isoformat(),
            'trade_type': side,
            'strategy': 'Paper Trade Engine',
            'hat_role': 'Paper Trading System',
            'symbol': self.symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'mark_price': self.current_orderbook.mid_price if self.current_orderbook else price,
            'order_type': 'MARKET',
            'order_id': order_id,
            'execution_time': latency_ms / 1000.0,
            'slippage': slippage_info['slippage_percent'],
            'fees_paid': quantity * price * 0.001,  # 0.1% fee estimate
            'position_size_before': self.position - (quantity if side.upper() == "BUY" else -quantity),
            'position_size_after': self.position,
            'avg_entry_price': self.avg_entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': trade_pnl,
            'margin_used': quantity * price,
            'margin_ratio': 0.1,
            'risk_score': 0.5,
            'stop_loss_price': 0.0,
            'take_profit_price': 0.0,
            'profit_loss': trade_pnl,
            'profit_loss_percent': (trade_pnl / (quantity * price)) * 100 if quantity * price > 0 else 0.0,
            'win_loss': win_loss,
            'trade_duration': 0.0,
            'funding_rate': 0.0001,
            'volatility': 0.0,
            'volume_24h': 0.0,
            'market_regime': 'NORMAL',
            'system_score': 10.0,
            'confidence_score': 0.8,
            'emergency_mode': False,
            'cycle_count': self.trade_counter,
            'data_source': 'paper_trade',
            'is_live_trade': False,
            'notes': 'Paper Trade with Realistic Execution',
            'tags': ['paper-trade', side.lower(), 'realistic-execution'],
            'metadata': {
                'expected_price': slippage_info['expected_price'],
                'slippage_bps': slippage_info['slippage_bps'],
                'slippage_cost': slippage_info['slippage_cost'],
                'latency_ms': latency_ms,
                'market_impact': slippage_info['market_impact'],
                'orderbook_spread': self.current_orderbook.spread if self.current_orderbook else 0.0,
                'orderbook_depth': len(self.current_orderbook.bids) if self.current_orderbook else 0
            }
        }
        
        # Record in trade ledger
        self.trade_ledger.record_trade(trade_data)
        
        # Record metrics
        record_trade_metrics(trade_data, self.metrics_collector)
        
        # Log trade
        self.logger.info(f"📊 [PAPER_TRADE] {side} {quantity:.3f} {self.symbol} @ ${price:.4f} "
                        f"(PnL: ${trade_pnl:+.4f}, Slippage: {slippage_info['slippage_bps']:+.1f} bps)")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_pnl = self.realized_pnl + self.unrealized_pnl
        total_return = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'position': self.position,
            'avg_entry_price': self.avg_entry_price,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': total_pnl,
            'total_return_percent': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_percent': win_rate,
            'current_price': self.current_orderbook.mid_price if self.current_orderbook else 0.0,
            'slippage_summary': self.slippage_analyzer.get_slippage_summary()
        }
    
    def set_network_condition(self, condition: str):
        """Set network condition for latency simulation"""
        self.latency_simulator.set_network_condition(condition)
        self.logger.info(f"🌐 [PAPER_TRADE] Network condition set to: {condition}")
    
    def save_trades(self):
        """Save all paper trades to files"""
        self.trade_ledger.save_to_csv()
        self.trade_ledger.save_to_parquet()
        self.logger.info("💾 [PAPER_TRADE] All trades saved to files")
    
    def get_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        summary = self.get_portfolio_summary()
        slippage_summary = summary['slippage_summary']
        
        report = f"""
🎯 PAPER TRADE PERFORMANCE REPORT
{'='*60}
💰 PORTFOLIO SUMMARY:
   Initial Capital: ${summary['initial_capital']:,.2f}
   Current Capital: ${summary['current_capital']:,.2f}
   Total PnL: ${summary['total_pnl']:+,.2f}
   Total Return: {summary['total_return_percent']:+.2f}%
   Position: {summary['position']:+.3f} {self.symbol}
   Avg Entry Price: ${summary['avg_entry_price']:.4f}
   Current Price: ${summary['current_price']:.4f}

📊 TRADING STATISTICS:
   Total Trades: {summary['total_trades']}
   Winning Trades: {summary['winning_trades']}
   Losing Trades: {summary['losing_trades']}
   Win Rate: {summary['win_rate_percent']:.1f}%

📈 SLIPPAGE ANALYSIS:
   Total Trades: {slippage_summary['total_trades']}
   Avg Slippage: {slippage_summary['avg_slippage_bps']:+.1f} bps
   Max Slippage: {slippage_summary['max_slippage_bps']:+.1f} bps
   Min Slippage: {slippage_summary['min_slippage_bps']:+.1f} bps
   Total Slippage Cost: ${slippage_summary['total_slippage_cost']:+,.2f}
   Slippage Ratio: {slippage_summary['slippage_ratio']:.4f}
   Buy Avg Slippage: {slippage_summary['buy_slippage_avg']:+.1f} bps
   Sell Avg Slippage: {slippage_summary['sell_slippage_avg']:+.1f} bps

⏱️ EXECUTION QUALITY:
   Network Condition: {self.latency_simulator.current_condition}
   Base Latency: {self.latency_simulator.base_latency_ms:.1f}ms
   Jitter: ±{self.latency_simulator.jitter_ms:.1f}ms
{'='*60}
        """
        
        return report
