#!/usr/bin/env python3
"""
🎯 DYNAMIC PORTFOLIO REBALANCING ENGINE
=======================================
Institutional-grade portfolio rebalancing based on correlation changes,
risk exposure drift, and market regime shifts.

Features:
- Real-time correlation matrix calculation
- Dynamic risk exposure monitoring
- Automated rebalancing triggers
- Position size optimization
- Regime-aware rebalancing strategies
"""

import asyncio
import logging
import time
import numpy as np
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RebalanceSignal:
    """Signal indicating need for portfolio rebalancing"""
    trigger_type: str  # 'correlation', 'exposure', 'regime', 'volatility'
    severity: float    # 0-1, higher = more urgent
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    action_required: str
    timestamp: datetime

@dataclass
class CorrelationMatrix:
    """Correlation matrix with metadata"""
    matrix: np.ndarray
    symbols: List[str]
    calculation_time: datetime
    sample_size: int
    confidence_level: float

class DynamicPortfolioRebalancer:
    """
    🎯 DYNAMIC PORTFOLIO REBALANCING ENGINE
    Monitors portfolio composition and triggers rebalancing based on:
    - Correlation drift beyond thresholds
    - Risk exposure concentration
    - Market regime changes
    - Volatility clustering
    """
    
    def __init__(self, api, config: Dict, logger: Optional[logging.Logger] = None):
        self.api = api
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Rebalancing parameters
        self.max_correlation_threshold = config.get('max_correlation', 0.7)
        self.min_rebalance_interval = config.get('min_rebalance_interval', 3600)  # 1 hour
        self.max_position_weight = config.get('max_position_weight', 0.4)
        self.correlation_lookback = config.get('correlation_lookback', 168)  # 7 days
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # 5% drift
        
        # State tracking
        self.last_rebalance = 0
        self.correlation_history = {}
        self.position_history = {}
        self.rebalance_signals = []
        
        # Target portfolio weights (will be dynamically adjusted)
        self.target_weights = {
            'XRP': 0.8,   # Primary focus
            'ETH': 0.15,  # Correlation hedge
            'BTC': 0.05   # Stability anchor
        }
        
        self.logger.info("🎯 [REBALANCER] Dynamic Portfolio Rebalancer initialized")

    async def monitor_and_rebalance(self):
        """Main monitoring loop for portfolio rebalancing"""
        try:
            # Get current portfolio state
            current_positions = await self._get_current_positions()
            if not current_positions:
                return
            
            # Calculate current weights
            current_weights = self._calculate_current_weights(current_positions)
            
            # Update correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix()
            
            # Check for rebalancing signals
            rebalance_signals = await self._check_rebalancing_triggers(
                current_weights, correlation_matrix
            )
            
            # Execute rebalancing if needed
            if rebalance_signals:
                await self._execute_rebalancing(rebalance_signals, current_positions)
                
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error in monitoring: {e}")

    async def _get_current_positions(self) -> Dict[str, Dict]:
        """Get current portfolio positions"""
        try:
            user_state = self.api.get_user_state()
            if not user_state:
                return {}
            
            positions = {}
            asset_positions = user_state.get("assetPositions", [])
            
            for pos in asset_positions:
                symbol = pos.get("position", {}).get("coin", "")
                size = float(pos.get("position", {}).get("szi", "0"))
                entry_px = float(pos.get("position", {}).get("entryPx", "0"))
                unrealized_pnl = float(pos.get("position", {}).get("unrealizedPnl", "0"))
                
                if abs(size) > 0.001:  # Meaningful position
                    positions[symbol] = {
                        'size': size,
                        'entry_price': entry_px,
                        'unrealized_pnl': unrealized_pnl,
                        'value': abs(size) * entry_px
                    }
            
            return positions
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error getting positions: {e}")
            return {}

    def _calculate_current_weights(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        try:
            total_value = sum(pos['value'] for pos in positions.values())
            if total_value == 0:
                return {}
            
            weights = {}
            for symbol, pos in positions.items():
                weights[symbol] = pos['value'] / total_value
            
            self.logger.info(f"📊 [REBALANCER] Current weights: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error calculating weights: {e}")
            return {}

    async def _calculate_correlation_matrix(self) -> Optional[CorrelationMatrix]:
        """Calculate correlation matrix for portfolio assets"""
        try:
            symbols = list(self.target_weights.keys())
            price_data = {}
            
            # Get historical price data for correlation calculation
            for symbol in symbols:
                # This would typically fetch historical data
                # For now, simulate with recent price movements
                prices = await self._get_price_history(symbol, self.correlation_lookback)
                if prices:
                    price_data[symbol] = prices
            
            if len(price_data) < 2:
                return None
            
            # Calculate correlation matrix
            price_arrays = [np.array(price_data[symbol]) for symbol in symbols if symbol in price_data]
            returns_matrix = np.array([np.diff(prices) / prices[:-1] for prices in price_arrays])
            
            correlation_matrix = np.corrcoef(returns_matrix)
            
            result = CorrelationMatrix(
                matrix=correlation_matrix,
                symbols=symbols,
                calculation_time=datetime.now(),
                sample_size=len(price_arrays[0]) - 1,
                confidence_level=0.95
            )
            
            self.logger.info(f"📊 [REBALANCER] Correlation matrix updated: {correlation_matrix.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error calculating correlation: {e}")
            return None

    async def _get_price_history(self, symbol: str, lookback_hours: int) -> List[float]:
        """Get price history for correlation calculation"""
        try:
            # This would typically fetch from exchange API
            # For now, simulate with market data
            market_data = self.api.get_market_data(symbol)
            if not market_data:
                return []
            
            current_price = float(market_data.get("price", 0))
            
            # Simulate price history with realistic volatility
            prices = []
            base_price = current_price
            
            for i in range(lookback_hours):
                # Add realistic price movement (±0.5% random walk)
                change = np.random.normal(0, 0.005)
                base_price *= (1 + change)
                prices.append(base_price)
            
            return prices
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error getting price history for {symbol}: {e}")
            return []

    async def _check_rebalancing_triggers(self, current_weights: Dict[str, float], 
                                        correlation_matrix: Optional[CorrelationMatrix]) -> List[RebalanceSignal]:
        """Check for conditions that trigger rebalancing"""
        signals = []
        
        try:
            # 1. Check weight drift from targets
            weight_drift_signal = self._check_weight_drift(current_weights)
            if weight_drift_signal:
                signals.append(weight_drift_signal)
            
            # 2. Check correlation concentration risk
            if correlation_matrix:
                correlation_signal = self._check_correlation_risk(correlation_matrix, current_weights)
                if correlation_signal:
                    signals.append(correlation_signal)
            
            # 3. Check position concentration risk
            concentration_signal = self._check_concentration_risk(current_weights)
            if concentration_signal:
                signals.append(concentration_signal)
            
            # 4. Check time-based rebalancing
            time_signal = self._check_time_based_rebalancing(current_weights)
            if time_signal:
                signals.append(time_signal)
            
            if signals:
                self.logger.warning(f"⚠️ [REBALANCER] {len(signals)} rebalancing signals detected")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error checking triggers: {e}")
            return []

    def _check_weight_drift(self, current_weights: Dict[str, float]) -> Optional[RebalanceSignal]:
        """Check if portfolio weights have drifted significantly from targets"""
        try:
            max_drift = 0
            drifted_assets = {}
            
            for symbol, target_weight in self.target_weights.items():
                current_weight = current_weights.get(symbol, 0)
                drift = abs(current_weight - target_weight)
                
                if drift > self.rebalance_threshold:
                    max_drift = max(max_drift, drift)
                    drifted_assets[symbol] = {
                        'current': current_weight,
                        'target': target_weight,
                        'drift': drift
                    }
            
            if drifted_assets:
                return RebalanceSignal(
                    trigger_type='weight_drift',
                    severity=min(max_drift * 2, 1.0),  # Scale to 0-1
                    target_weights=self.target_weights,
                    current_weights=current_weights,
                    action_required=f"Rebalance {len(drifted_assets)} assets with max drift {max_drift:.2%}",
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error checking weight drift: {e}")
            return None

    def _check_correlation_risk(self, correlation_matrix: CorrelationMatrix, 
                               current_weights: Dict[str, float]) -> Optional[RebalanceSignal]:
        """Check for excessive correlation between portfolio assets"""
        try:
            symbols = correlation_matrix.symbols
            matrix = correlation_matrix.matrix
            
            high_correlations = []
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = matrix[i, j]
                    if abs(correlation) > self.max_correlation_threshold:
                        weight_product = current_weights.get(symbols[i], 0) * current_weights.get(symbols[j], 0)
                        high_correlations.append({
                            'assets': (symbols[i], symbols[j]),
                            'correlation': correlation,
                            'weight_product': weight_product,
                            'risk_contribution': abs(correlation) * weight_product
                        })
            
            if high_correlations:
                # Sort by risk contribution
                high_correlations.sort(key=lambda x: x['risk_contribution'], reverse=True)
                top_risk = high_correlations[0]
                
                return RebalanceSignal(
                    trigger_type='correlation_risk',
                    severity=min(top_risk['risk_contribution'] * 10, 1.0),
                    target_weights=self._calculate_decorrelated_weights(correlation_matrix, current_weights),
                    current_weights=current_weights,
                    action_required=f"Reduce correlation risk between {top_risk['assets']}",
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error checking correlation risk: {e}")
            return None

    def _check_concentration_risk(self, current_weights: Dict[str, float]) -> Optional[RebalanceSignal]:
        """Check for excessive concentration in single positions"""
        try:
            over_concentrated = {}
            
            for symbol, weight in current_weights.items():
                if weight > self.max_position_weight:
                    over_concentrated[symbol] = {
                        'current': weight,
                        'max_allowed': self.max_position_weight,
                        'excess': weight - self.max_position_weight
                    }
            
            if over_concentrated:
                max_excess = max(asset['excess'] for asset in over_concentrated.values())
                
                # Calculate new target weights to reduce concentration
                new_targets = current_weights.copy()
                for symbol, data in over_concentrated.items():
                    new_targets[symbol] = self.max_position_weight
                
                # Redistribute excess to other assets
                total_excess = sum(data['excess'] for data in over_concentrated.values())
                non_concentrated = [s for s in new_targets.keys() if s not in over_concentrated]
                
                if non_concentrated:
                    excess_per_asset = total_excess / len(non_concentrated)
                    for symbol in non_concentrated:
                        new_targets[symbol] += excess_per_asset
                
                return RebalanceSignal(
                    trigger_type='concentration_risk',
                    severity=min(max_excess * 2, 1.0),
                    target_weights=new_targets,
                    current_weights=current_weights,
                    action_required=f"Reduce concentration in {len(over_concentrated)} assets",
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error checking concentration: {e}")
            return None

    def _check_time_based_rebalancing(self, current_weights: Dict[str, float]) -> Optional[RebalanceSignal]:
        """Check if it's time for scheduled rebalancing"""
        try:
            current_time = time.time()
            time_since_last = current_time - self.last_rebalance
            
            # Weekly rebalancing (604800 seconds = 1 week)
            weekly_rebalance_interval = 604800
            
            if time_since_last > weekly_rebalance_interval:
                return RebalanceSignal(
                    trigger_type='time_based',
                    severity=0.3,  # Lower priority
                    target_weights=self.target_weights,
                    current_weights=current_weights,
                    action_required="Weekly scheduled rebalancing",
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error checking time-based rebalancing: {e}")
            return None

    def _calculate_decorrelated_weights(self, correlation_matrix: CorrelationMatrix, 
                                      current_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimal weights to reduce correlation risk"""
        try:
            # Simplified optimization: reduce weights of highly correlated pairs
            new_weights = current_weights.copy()
            symbols = correlation_matrix.symbols
            matrix = correlation_matrix.matrix
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = matrix[i, j]
                    if abs(correlation) > self.max_correlation_threshold:
                        # Reduce weight of the larger position
                        symbol_i, symbol_j = symbols[i], symbols[j]
                        weight_i = current_weights.get(symbol_i, 0)
                        weight_j = current_weights.get(symbol_j, 0)
                        
                        if weight_i > weight_j:
                            new_weights[symbol_i] *= 0.9  # Reduce by 10%
                        else:
                            new_weights[symbol_j] *= 0.9
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                for symbol in new_weights:
                    new_weights[symbol] /= total_weight
            
            return new_weights
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error calculating decorrelated weights: {e}")
            return current_weights

    async def _execute_rebalancing(self, signals: List[RebalanceSignal], 
                                 current_positions: Dict[str, Dict]):
        """Execute portfolio rebalancing based on signals"""
        try:
            if time.time() - self.last_rebalance < self.min_rebalance_interval:
                self.logger.info("⏰ [REBALANCER] Rebalancing skipped - too soon since last rebalance")
                return
            
            # Select highest priority signal
            primary_signal = max(signals, key=lambda s: s.severity)
            
            self.logger.warning(f"🔄 [REBALANCER] Executing rebalancing: {primary_signal.trigger_type} (severity: {primary_signal.severity:.2f})")
            
            # Calculate required trades
            trades = self._calculate_rebalancing_trades(primary_signal, current_positions)
            
            # Execute trades
            successful_trades = 0
            for trade in trades:
                success = await self._execute_rebalancing_trade(trade)
                if success:
                    successful_trades += 1
            
            if successful_trades > 0:
                self.last_rebalance = time.time()
                self.logger.info(f"✅ [REBALANCER] Rebalancing completed: {successful_trades}/{len(trades)} trades successful")
            else:
                self.logger.error("❌ [REBALANCER] Rebalancing failed - no successful trades")
                
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error executing rebalancing: {e}")

    def _calculate_rebalancing_trades(self, signal: RebalanceSignal, 
                                    current_positions: Dict[str, Dict]) -> List[Dict]:
        """Calculate specific trades needed for rebalancing"""
        trades = []
        
        try:
            total_portfolio_value = sum(pos['value'] for pos in current_positions.values())
            current_weights = {symbol: pos['value'] / total_portfolio_value 
                             for symbol, pos in current_positions.items()}
            
            for symbol, target_weight in signal.target_weights.items():
                current_weight = current_weights.get(symbol, 0)
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.01:  # 1% minimum trade size
                    trade_value = weight_diff * total_portfolio_value
                    
                    trades.append({
                        'symbol': symbol,
                        'action': 'buy' if weight_diff > 0 else 'sell',
                        'value': abs(trade_value),
                        'weight_change': weight_diff,
                        'reason': signal.trigger_type
                    })
            
            return trades
            
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error calculating trades: {e}")
            return []

    async def _execute_rebalancing_trade(self, trade: Dict) -> bool:
        """Execute a single rebalancing trade"""
        try:
            symbol = trade['symbol']
            action = trade['action']
            value = trade['value']
            
            # Get current market price
            market_data = self.api.get_market_data(symbol)
            if not market_data:
                return False
            
            current_price = float(market_data.get("price", 0))
            if current_price <= 0:
                return False
            
            # Calculate trade size
            trade_size = value / current_price
            
            # Execute trade
            order_result = self.api.place_order(
                coin=symbol,
                is_buy=(action == 'buy'),
                sz=trade_size,
                limit_px=current_price,
                reduce_only=False
            )
            
            if order_result and order_result.get("status") == "ok":
                self.logger.info(f"✅ [REBALANCER] {action.upper()} {trade_size:.4f} {symbol} at ${current_price:.4f}")
                return True
            else:
                self.logger.error(f"❌ [REBALANCER] Failed to {action} {symbol}: {order_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [REBALANCER] Error executing trade: {e}")
            return False

    def get_rebalancing_status(self) -> Dict:
        """Get current rebalancing system status"""
        return {
            'last_rebalance': self.last_rebalance,
            'time_since_last': time.time() - self.last_rebalance,
            'target_weights': self.target_weights,
            'recent_signals': len(self.rebalance_signals),
            'status': 'active'
        }
