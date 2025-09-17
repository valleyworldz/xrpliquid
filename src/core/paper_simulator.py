#!/usr/bin/env python3
"""
📊 PAPER TRADING SIMULATOR
==========================

Backtesting engine for strategy parameter validation and optimization.
Simulates trading strategies on historical data without real money.

Features:
- Historical data simulation
- Strategy performance metrics
- Risk-adjusted returns calculation
- Parameter validation for HPO
"""

from src.core.utils.decimal_boundary_guard import safe_float
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta
from core.utils.logger import Logger
from core.api.hyperliquid_api import HyperliquidAPI
from core.strategies.scalping import Scalping
from core.strategies.grid_trading import GridTrading
from core.strategies.mean_reversion import MeanReversion

class PaperSimulator:
    """
    Paper Trading Simulator for Strategy Backtesting
    """
    
    def __init__(self, config=None):
        self.logger = Logger()
        self.config = config
        self.api = HyperliquidAPI()
        self.initial_balance = 1000.0  # Starting balance for simulation
        self.commission_rate = 0.0005  # 0.05% commission per trade
        self.trades = []  # Initialize trades list
        
    def get_historical_data(self, token: str, lookback_minutes: int) -> pd.DataFrame:
        """Fetch historical price data for simulation"""
        try:
            # Get recent price data from Hyperliquid API
            # This is a simplified version - in production, you'd fetch actual historical data
            current_price = self.api.get_market_data(token)
            
            if not current_price:
                self.logger.warning(f"[SIM] No market data for {token}")
                return pd.DataFrame()
            
            # Generate synthetic historical data for simulation
            # In production, replace with actual historical data from API
            base_price = current_price["price"]
            timestamps = []
            prices = []
            volumes = []
            
            # Generate synthetic data points
            for i in range(lookback_minutes):
                timestamp = datetime.now() - timedelta(minutes=lookback_minutes - i)
                # Add some realistic price movement
                price_change = np.random.normal(0, base_price * 0.001)  # 0.1% volatility
                price = base_price + price_change
                volume = np.random.uniform(0.1, 1.0)  # Random volume
                
                timestamps.append(timestamp)
                prices.append(price)
                volumes.append(volume)
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'price': prices,
                'volume': volumes
            })
            
            self.logger.info(f"[SIM] Generated {len(df)} data points for {token}")
            return df
            
        except Exception as e:
            self.logger.error(f"[SIM] Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def simulate_scalping(self, params: Dict[str, Any], data: pd.DataFrame) -> Tuple[float, float, float]:
        """Simulate scalping strategy"""
        try:
            balance = self.initial_balance
            trades = []
            positions = []
            
            spread_threshold = params.get("spread_threshold", 0.001)
            tp_pips = params.get("tp_pips", 5)
            sl_pips = params.get("sl_pips", 3)
            min_volume = params.get("min_volume", 0.005)
            max_orders = params.get("max_orders", 5)
            
            for i in range(1, len(data)):
                current_price = data.iloc[i]['price']
                prev_price = data.iloc[i-1]['price']
                
                # Calculate spread
                spread = abs(current_price - prev_price) / prev_price
                
                # Scalping logic
                if spread > spread_threshold and len(positions) < max_orders:
                    # Calculate position size
                    position_size = min(balance * 0.1 / current_price, min_volume)
                    
                    if position_size > 0:
                        # Open position
                        entry_price = current_price
                        position = {
                            'entry_price': entry_price,
                            'size': position_size,
                            'entry_time': i,
                            'tp_price': entry_price * (1 + tp_pips * 0.0001),
                            'sl_price': entry_price * (1 - sl_pips * 0.0001)
                        }
                        positions.append(position)
                        
                        # Deduct commission
                        balance -= position_size * current_price * self.commission_rate
                        
                        self.logger.debug(f"[SIM] Opened scalping position: {position_size:.4f} @ {entry_price:.4f}")
                
                # Check existing positions
                for pos in positions[:]:  # Copy list to avoid modification during iteration
                    if current_price >= pos['tp_price']:
                        # Take profit
                        pnl = (pos['tp_price'] - pos['entry_price']) * pos['size']
                        balance += pnl - (pos['size'] * pos['tp_price'] * self.commission_rate)
                        trades.append(pnl)
                        positions.remove(pos)
                        
                        self.logger.debug(f"[SIM] TP hit: +{pnl:.4f}")
                        
                    elif current_price <= pos['sl_price']:
                        # Stop loss
                        pnl = (pos['sl_price'] - pos['entry_price']) * pos['size']
                        balance += pnl - (pos['size'] * pos['sl_price'] * self.commission_rate)
                        trades.append(pnl)
                        positions.remove(pos)
                        
                        self.logger.debug(f"[SIM] SL hit: {pnl:.4f}")
            
            # Close any remaining positions at final price
            final_price = data.iloc[-1]['price']
            for pos in positions:
                pnl = (final_price - pos['entry_price']) * pos['size']
                balance += pnl - (pos['size'] * final_price * self.commission_rate)
                trades.append(pnl)
            
            # Calculate metrics
            total_pnl = safe_float(balance - self.initial_balance)
            volatility = safe_float(np.std(trades) if trades else 0.0)
            sharpe = safe_float(total_pnl / volatility if volatility > 0 else 0.0)
            
            return total_pnl, volatility, sharpe
            
        except Exception as e:
            self.logger.error(f"[SIM] Error in scalping simulation: {e}")
            return 0.0, 0.0, 0.0
    
    def simulate_grid_trading(self, params: Dict[str, Any], data: pd.DataFrame) -> Tuple[float, float, float]:
        """Simulate grid trading strategy"""
        try:
            balance = self.initial_balance
            trades = []
            grid_orders = []
            
            grid_size = params.get("grid_size", 0.001)
            num_grids = params.get("num_grids", 10)
            order_quantity = params.get("order_quantity", 0.01)
            grid_spread = params.get("grid_spread", 0.005)
            
            # Initialize grid around current price
            current_price = data.iloc[0]['price']
            grid_center = current_price
            
            # Create grid orders
            for i in range(num_grids):
                buy_price = grid_center * (1 - grid_spread/2 + i * grid_size)
                sell_price = grid_center * (1 + grid_spread/2 - i * grid_size)
                
                grid_orders.append({
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'quantity': order_quantity,
                    'filled': False
                })
            
            # Simulate grid trading
            for i in range(len(data)):
                current_price = data.iloc[i]['price']
                
                for order in grid_orders:
                    if not order['filled']:
                        if current_price <= order['buy_price']:
                            # Execute buy order
                            cost = order['quantity'] * order['buy_price']
                            if balance >= cost:
                                balance -= cost + (cost * self.commission_rate)
                                order['filled'] = True
                                order['entry_price'] = order['buy_price']
                                
                        elif current_price >= order['sell_price']:
                            # Execute sell order
                            revenue = order['quantity'] * order['sell_price']
                            balance += revenue - (revenue * self.commission_rate)
                            order['filled'] = True
                            
                            # Calculate PnL if we had a buy order
                            if 'entry_price' in order:
                                pnl = (order['sell_price'] - order['entry_price']) * order['quantity']
                                trades.append(pnl)
            
            # Calculate metrics
            total_pnl = safe_float(balance - self.initial_balance)
            volatility = safe_float(np.std(trades) if trades else 0.0)
            sharpe = safe_float(total_pnl / volatility if volatility > 0 else 0.0)
            
            return total_pnl, volatility, sharpe
            
        except Exception as e:
            self.logger.error(f"[SIM] Error in grid trading simulation: {e}")
            return 0.0, 0.0, 0.0
    
    def simulate_mean_reversion(self, params: Dict[str, Any], data: pd.DataFrame) -> Tuple[float, float, float]:
        """Simulate mean reversion strategy"""
        try:
            balance = self.initial_balance
            trades = []
            position = None
            
            entry_deviation = params.get("entry_deviation", 0.001)
            exit_deviation = params.get("exit_deviation", 0.0005)
            max_position_size = params.get("max_position_size", 0.01)
            standard_deviations = params.get("standard_deviations", 2.0)
            lookback_period = params.get("lookback_period", 50)
            
            for i in range(lookback_period, len(data)):
                # Calculate moving average and standard deviation
                recent_prices = data.iloc[i-lookback_period:i]['price']
                ma = recent_prices.mean()
                std = recent_prices.std()
                
                current_price = data.iloc[i]['price']
                
                # Mean reversion logic
                if position is None:
                    # Check for entry signal
                    deviation = abs(current_price - ma) / ma
                    
                    if deviation > entry_deviation:
                        # Open position
                        position_size = min(balance * 0.1 / current_price, max_position_size)
                        
                        if current_price > ma:
                            # Price above MA, expect reversion down
                            position = {
                                'type': 'short',
                                'entry_price': current_price,
                                'size': position_size,
                                'entry_time': i
                            }
                        else:
                            # Price below MA, expect reversion up
                            position = {
                                'type': 'long',
                                'entry_price': current_price,
                                'size': position_size,
                                'entry_time': i
                            }
                        
                        # Deduct commission
                        balance -= position_size * current_price * self.commission_rate
                
                elif position is not None:
                    # Check for exit signal
                    deviation = abs(current_price - ma) / ma
                    
                    if deviation < exit_deviation:
                        # Close position
                        if position['type'] == 'long':
                            pnl = (current_price - position['entry_price']) * position['size']
                        else:
                            pnl = (position['entry_price'] - current_price) * position['size']
                        
                        balance += pnl - (position['size'] * current_price * self.commission_rate)
                        trades.append(pnl)
                        position = None
            
            # Close any remaining position
            if position is not None:
                final_price = data.iloc[-1]['price']
                if position['type'] == 'long':
                    pnl = (final_price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - final_price) * position['size']
                
                balance += pnl - (position['size'] * final_price * self.commission_rate)
                trades.append(pnl)
            
            # Calculate metrics
            total_pnl = safe_float(balance - self.initial_balance)
            volatility = safe_float(np.std(trades) if trades else 0.0)
            sharpe = safe_float(total_pnl / volatility if volatility > 0 else 0.0)
            
            return total_pnl, volatility, sharpe
            
        except Exception as e:
            self.logger.error(f"[SIM] Error in mean reversion simulation: {e}")
            return 0.0, 0.0, 0.0
    
    def simulate(self, strategy: str, params: Dict[str, Any], lookback_minutes: int = 240, token: str = "DOGE") -> Tuple[float, float, float]:
        """Main simulation method"""
        try:
            self.logger.info(f"[SIM] Starting {strategy} simulation for {token}")
            
            # Get historical data
            data = self.get_historical_data(token, lookback_minutes)
            
            if data.empty:
                self.logger.warning(f"[SIM] No data available for {token}")
                return 0.0, 0.0, 0.0
            
            # Run strategy simulation
            if strategy == "scalping":
                pnl, volatility, sharpe = self.simulate_scalping(params, data)
            elif strategy == "grid_trading":
                pnl, volatility, sharpe = self.simulate_grid_trading(params, data)
            elif strategy == "mean_reversion":
                pnl, volatility, sharpe = self.simulate_mean_reversion(params, data)
            else:
                self.logger.error(f"[SIM] Unknown strategy: {strategy}")
                return 0.0, 0.0, 0.0
            
            self.logger.info(f"[SIM] {strategy} simulation complete - PnL: {pnl:.4f}, Vol: {volatility:.4f}, Sharpe: {sharpe:.4f}")
            
            return pnl, volatility, sharpe
            
        except Exception as e:
            self.logger.error(f"[SIM] Error in simulation: {e}")
            return 0.0, 0.0, 0.0
    
    def get_simulation_summary(self, strategy: str, params: Dict[str, Any], lookback_minutes: int = 240, token: str = "DOGE") -> Dict[str, Any]:
        """Get comprehensive simulation summary"""
        try:
            pnl, volatility, sharpe = self.simulate(strategy, params, lookback_minutes, token)
            
            return {
                'strategy': strategy,
                'token': token,
                'pnl': pnl,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': self._calculate_max_drawdown(),
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor(),
                'total_trades': len(self.trades),
                'simulation_duration': f"{lookback_minutes} minutes"
            }
            
        except Exception as e:
            self.logger.error(f"[SIM] Error in simulation summary: {e}")
            return {}
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown (placeholder)"""
        return 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate (placeholder)"""
        return 0.5
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (placeholder)"""
        return 1.0
    
    def run_simulation(self, tokens: list, strategies: list, duration_days: int = 30) -> Dict[str, Any]:
        """
        Run paper trading simulation for multiple tokens and strategies
        
        Args:
            tokens: List of tokens to simulate
            strategies: List of strategies to test
            duration_days: Duration of simulation in days
            
        Returns:
            Dictionary with simulation results
        """
        try:
            self.logger.info(f"[SIM] Starting paper trading simulation for {tokens} with {strategies}")
            
            results = {
                'tokens': tokens,
                'strategies': strategies,
                'duration_days': duration_days,
                'simulations': {},
                'summary': {}
            }
            
            # Default parameters for each strategy
            strategy_params = {
                'scalping': {
                    'spread_threshold': 0.001,
                    'tp_pips': 5,
                    'sl_pips': 3,
                    'min_volume': 0.005,
                    'max_orders': 5
                },
                'grid_trading': {
                    'grid_size': 0.001,
                    'num_grids': 10,
                    'order_quantity': 0.01,
                    'grid_spread': 0.005
                },
                'mean_reversion': {
                    'lookback_period': 20,
                    'entry_threshold': 2.0,
                    'exit_threshold': 0.5,
                    'position_size': 0.1
                }
            }
            
            # Run simulations for each token and strategy combination
            for token in tokens:
                results['simulations'][token] = {}
                
                for strategy in strategies:
                    self.logger.info(f"[SIM] Running {strategy} simulation for {token}")
                    
                    # Get historical data
                    lookback_minutes = duration_days * 24 * 60  # Convert days to minutes
                    data = self.get_historical_data(token, lookback_minutes)
                    
                    if data.empty:
                        self.logger.warning(f"[SIM] No data available for {token}, skipping")
                        continue
                    
                    # Get parameters for this strategy
                    params = strategy_params.get(strategy, {})
                    
                    # Run simulation
                    simulation_result = self.get_simulation_summary(strategy, params, lookback_minutes, token)
                    results['simulations'][token][strategy] = simulation_result
            
            # Calculate overall summary
            total_pnl = 0.0
            total_trades = 0
            successful_strategies = 0
            
            for token in results['simulations']:
                for strategy in results['simulations'][token]:
                    sim_result = results['simulations'][token][strategy]
                    if sim_result:
                        total_pnl += sim_result.get('pnl', 0.0)
                        total_trades += sim_result.get('total_trades', 0)
                        successful_strategies += 1
            
            results['summary'] = {
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'successful_strategies': successful_strategies,
                'average_pnl_per_strategy': total_pnl / successful_strategies if successful_strategies > 0 else 0.0
            }
            
            self.logger.info(f"[SIM] Paper trading simulation completed. Total PnL: ${total_pnl:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"[SIM] Error in run_simulation: {e}")
            return {'error': str(e)}
