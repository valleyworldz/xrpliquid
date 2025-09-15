#!/usr/bin/env python3
"""
🚀 ADVANCED GRID TRADING ENGINE
===============================
Intelligent grid trading with ML optimization, volatility-based spacing,
and multi-pair coordination using live and historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
import json
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests
import time

@dataclass
class GridLevel:
    """Represents a single grid level"""
    price: float
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    filled: bool = False
    profit_target: float = 0.0
    volume: float = 0.0

@dataclass
class GridConfig:
    """Grid configuration parameters"""
    pair: str
    base_price: float
    grid_count: int = 20
    total_investment: float = 1000.0
    min_spacing_pct: float = 0.001  # 0.1%
    max_spacing_pct: float = 0.05   # 5%
    volatility_multiplier: float = 2.0
    rebalance_threshold: float = 0.1  # 10%

class AdvancedGridEngine:
    """Advanced grid trading engine with ML optimization"""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.grid_levels = []
        self.active_orders = {}
        self.performance_stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0.0,
            'grid_efficiency': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.last_rebalance = datetime.now()
        
        # Initialize ML components
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.logger.info("🚀 Advanced Grid Engine initialized!")
    
    def get_free_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get free historical data from CoinGecko"""
        try:
            # Convert symbol to CoinGecko format
            symbol_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum', 
                'SOL': 'solana',
                'AVAX': 'avalanche-2'
            }
            
            coin_id = symbol_map.get(symbol.upper().replace('/USDT', ''), symbol.lower())
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                prices = data['prices']
                volumes = data['total_volumes']
                
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['volume'] = [v[1] for v in volumes]
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                self.logger.info(f"✅ Retrieved {len(df)} historical data points")
                return df
                
        except Exception as e:
            self.logger.warning(f"Failed to get historical data: {e}")
            
        # Return empty DataFrame if failed
        return pd.DataFrame()
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive volatility metrics"""
        if len(df) < 2:
            return {'volatility': 0.02, 'trend': 0.0, 'momentum': 0.0}
            
        # Price returns
        df['returns'] = df['price'].pct_change().fillna(0)
        
        # Historical volatility (annualized)
        volatility = df['returns'].std() * np.sqrt(365 * 24)  # Hourly to annual
        
        # Trend analysis
        prices = df['price'].values
        x = np.arange(len(prices))
        trend = np.polyfit(x, prices, 1)[0] / np.mean(prices) if len(prices) > 1 else 0
        
        # Momentum (rate of change)
        momentum = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0
        
        return {
            'volatility': max(volatility, 0.001),  # Minimum 0.1%
            'trend': trend,
            'momentum': momentum,
            'price_range': (df['price'].max() - df['price'].min()) / df['price'].mean()
        }
    
    def train_ml_model(self, historical_data: pd.DataFrame):
        """Train ML model for optimal grid spacing prediction"""
        if len(historical_data) < 100:
            self.logger.warning("Insufficient data for ML training")
            return
            
        try:
            # Prepare features and targets
            features = []
            targets = []
            
            window_size = 24  # 24 hours
            
            for i in range(window_size, len(historical_data) - 24):
                window_data = historical_data.iloc[i-window_size:i]
                future_data = historical_data.iloc[i:i+24]
                
                # Features (market conditions)
                volatility = window_data['price'].pct_change().std()
                trend = (window_data['price'].iloc[-1] - window_data['price'].iloc[0]) / window_data['price'].iloc[0]
                volume_trend = (window_data['volume'].iloc[-1] - window_data['volume'].iloc[0]) / window_data['volume'].iloc[0]
                price_position = (window_data['price'].iloc[-1] - window_data['price'].min()) / (window_data['price'].max() - window_data['price'].min())
                
                feature_vector = [volatility, trend, volume_trend, price_position]
                
                # Target (optimal grid spacing for this period)
                future_volatility = future_data['price'].pct_change().std()
                optimal_spacing = min(max(future_volatility * 2, 0.001), 0.05)
                
                features.append(feature_vector)
                targets.append(optimal_spacing)
            
            X = np.array(features)
            y = np.array(targets)
            
            # Train model
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            score = self.model.score(X_scaled, y)
            self.logger.info(f"✅ ML model trained with R² score: {score:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ ML training failed: {e}")
    
    def predict_optimal_spacing(self, current_data: pd.DataFrame) -> float:
        """Predict optimal grid spacing using ML model"""
        if not self.is_trained or len(current_data) < 24:
            return 0.01  # Default 1% spacing
            
        try:
            # Prepare current features
            volatility = current_data['price'].pct_change().std()
            trend = (current_data['price'].iloc[-1] - current_data['price'].iloc[0]) / current_data['price'].iloc[0]
            volume_trend = (current_data['volume'].iloc[-1] - current_data['volume'].iloc[0]) / current_data['volume'].iloc[0]
            price_position = (current_data['price'].iloc[-1] - current_data['price'].min()) / (current_data['price'].max() - current_data['price'].min())
            
            features = np.array([[volatility, trend, volume_trend, price_position]])
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            optimal_spacing = max(min(prediction, self.config.max_spacing_pct), self.config.min_spacing_pct)
            
            self.logger.info(f"📊 ML predicted spacing: {optimal_spacing:.4f}")
            return optimal_spacing
            
        except Exception as e:
            self.logger.error(f"❌ ML prediction failed: {e}")
            return 0.01
    
    def create_grid_levels(self, center_price: float, spacing_pct: float) -> List[GridLevel]:
        """Create optimized grid levels"""
        levels = []
        investment_per_level = self.config.total_investment / self.config.grid_count
        
        # Create levels above and below center price
        for i in range(self.config.grid_count // 2):
            # Levels below center (buy levels)
            buy_price = center_price * (1 - spacing_pct * (i + 1))
            levels.append(GridLevel(
                price=buy_price,
                profit_target=buy_price * (1 + spacing_pct),
                volume=investment_per_level / buy_price
            ))
            
            # Levels above center (sell levels)
            sell_price = center_price * (1 + spacing_pct * (i + 1))
            levels.append(GridLevel(
                price=sell_price,
                profit_target=sell_price * (1 - spacing_pct),
                volume=investment_per_level / sell_price
            ))
        
        # Sort by price
        levels.sort(key=lambda x: x.price)
        
        self.logger.info(f"🎯 Created {len(levels)} grid levels from "
                        f"${levels[0].price:.4f} to ${levels[-1].price:.4f}")
        
        return levels
    
    def should_rebalance_grid(self, current_price: float) -> bool:
        """Determine if grid needs rebalancing"""
        if not self.grid_levels:
            return True
            
        # Check if price moved outside grid range
        min_price = min(level.price for level in self.grid_levels)
        max_price = max(level.price for level in self.grid_levels)
        
        if current_price < min_price * (1 - self.config.rebalance_threshold):
            self.logger.info("📈 Price moved below grid range - rebalancing needed")
            return True
            
        if current_price > max_price * (1 + self.config.rebalance_threshold):
            self.logger.info("📉 Price moved above grid range - rebalancing needed")
            return True
            
        # Check time-based rebalancing (every 24 hours)
        if datetime.now() - self.last_rebalance > timedelta(hours=24):
            self.logger.info("⏰ Time-based rebalancing triggered")
            return True
            
        return False
    
    def execute_grid_strategy(self, current_price: float) -> Dict[str, any]:
        """Execute the advanced grid trading strategy"""
        try:
            # Get fresh data for analysis
            historical_data = self.get_free_historical_data(self.config.pair.split('/')[0], days=7)
            
            # Train/update ML model if we have data
            if len(historical_data) > 0 and not self.is_trained:
                self.train_ml_model(historical_data)
            
            # Check if rebalancing is needed
            if self.should_rebalance_grid(current_price):
                if len(historical_data) > 0:
                    spacing = self.predict_optimal_spacing(historical_data.tail(24))
                else:
                    spacing = 0.01  # Default spacing
                    
                self.grid_levels = self.create_grid_levels(current_price, spacing)
                self.last_rebalance = datetime.now()
                
                self.logger.info(f"🔄 Grid rebalanced with {spacing:.4f} spacing")
            
            # Execute trading logic
            executed_trades = []
            
            for level in self.grid_levels:
                if not level.filled:
                    # Check for buy opportunities (price at or below level)
                    if current_price <= level.price and not level.buy_order_id:
                        trade = {
                            'type': 'buy',
                            'price': level.price,
                            'volume': level.volume,
                            'target': level.profit_target
                        }
                        executed_trades.append(trade)
                        level.buy_order_id = f"buy_{int(time.time())}"
                        
                        self.logger.info(f"🟢 Grid BUY at ${level.price:.4f} "
                                       f"(Target: ${level.profit_target:.4f})")
                    
                    # Check for sell opportunities (price at or above profit target)
                    elif current_price >= level.profit_target and level.buy_order_id:
                        trade = {
                            'type': 'sell',
                            'price': current_price,
                            'volume': level.volume,
                            'profit': (current_price - level.price) * level.volume
                        }
                        executed_trades.append(trade)
                        level.filled = True
                        
                        # Update performance stats
                        self.performance_stats['total_trades'] += 1
                        self.performance_stats['profitable_trades'] += 1
                        self.performance_stats['total_profit'] += trade['profit']
                        
                        self.logger.info(f"🔴 Grid SELL at ${current_price:.4f} "
                                       f"(Profit: ${trade['profit']:.4f})")
            
            # Calculate grid efficiency
            if self.performance_stats['total_trades'] > 0:
                self.performance_stats['grid_efficiency'] = (
                    self.performance_stats['profitable_trades'] / 
                    self.performance_stats['total_trades']
                )
            
            return {
                'executed_trades': executed_trades,
                'grid_levels': len(self.grid_levels),
                'performance': self.performance_stats,
                'ml_trained': self.is_trained
            }
            
        except Exception as e:
            self.logger.error(f"❌ Grid execution failed: {e}")
            return {'executed_trades': [], 'error': str(e)}
    
    def get_grid_status(self) -> Dict[str, any]:
        """Get current grid status and performance"""
        if not self.grid_levels:
            return {'status': 'not_initialized'}
            
        active_levels = sum(1 for level in self.grid_levels if not level.filled)
        filled_levels = len(self.grid_levels) - active_levels
        
        return {
            'total_levels': len(self.grid_levels),
            'active_levels': active_levels,
            'filled_levels': filled_levels,
            'performance': self.performance_stats,
            'last_rebalance': self.last_rebalance.isoformat(),
            'ml_trained': self.is_trained,
            'price_range': {
                'min': min(level.price for level in self.grid_levels),
                'max': max(level.price for level in self.grid_levels)
            } if self.grid_levels else None
        }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test the advanced grid engine
    config = GridConfig(
        pair='BTC/USDT',
        base_price=45000.0,
        grid_count=20,
        total_investment=1000.0,
        volatility_multiplier=1.5
    )
    
    # Initialize engine
    engine = AdvancedGridEngine(config)
    
    # Simulate some price movements
    test_prices = [45000, 44500, 45500, 44000, 46000, 45200]
    
    for price in test_prices:
        print(f"\n💰 Testing price: ${price}")
        result = engine.execute_grid_strategy(price)
        print(f"📊 Result: {result}")
        
        status = engine.get_grid_status()
        print(f"🌐 Status: {status}") 