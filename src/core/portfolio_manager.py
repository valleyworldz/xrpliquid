#!/usr/bin/env python3
"""
📈 PORTFOLIO MANAGER
===================

Multi-token ensemble trading with dynamic portfolio allocation.
Manages risk across multiple tokens and strategies.

Features:
- Multi-token portfolio allocation
- Strategy ensemble scoring
- Dynamic rebalancing
- Risk-adjusted position sizing

Portfolio Manager with Profit Rotation Framework
- Automatically closes profitable positions
- Immediately reinvests freed capital
- Soft concurrency limits for safety
- Per-symbol cooldowns to avoid FOMO
"""

from src.core.utils.decimal_boundary_guard import safe_float
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
from core.utils.config_manager import ConfigManager
from core.utils.logger import Logger
from core.api.hyperliquid_api import HyperliquidAPI
from core.strategies import StrategyManager
from core.utils.meta_manager import MetaManager
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from core.engines.risk_management import RiskManagement
import json

@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    asset_id: int
    quantity: float
    entry_price: float
    current_price: float
    side: str  # 'long' or 'short'
    entry_time: datetime
    unrealized_pnl: float
    unrealized_pnl_pct: float

@dataclass
class TradeSignal:
    """Represents a trading signal"""
    symbol: str
    confidence: float
    sharpe_ratio: float
    signal_strength: float
    timestamp: datetime

class PortfolioManager:
    """
    Portfolio Manager for Multi-Token Ensemble Trading
    """
    
    def __init__(self, strategy_manager: StrategyManager, api_client: HyperliquidAPI, config: ConfigManager, meta_manager: MetaManager):
        self.strategy_manager = strategy_manager
        self.api = api_client
        self.config = config
        self.meta_manager = meta_manager
        self.logger = Logger()
        self.risk_manager = RiskManagement(config)
        self.token_universe = config.get("portfolio.tokens", ["DOGE", "ETH", "SOL", "BTC", "TRUMP"])
        self.current_allocations = {}
        self.portfolio_history = []
        
        # Detect trading mode (spot vs perp) from API or configuration
        self.trading_mode = self._detect_trading_mode()
        self.logger.info(f"[PORTFOLIO] Detected trading mode: {self.trading_mode.upper()}")
        
        # Load mode-specific configuration
        mode_config_key = f"{self.trading_mode}_trading"
        mode_config = config.get(mode_config_key, {})
        
        # Profit rotation configuration with mode-specific defaults
        self.profit_rotation_config = config.get("trading.profit_rotation", {})
        self.profit_target_pct = mode_config.get("profit_target_pct", self.profit_rotation_config.get("profit_target_pct", 0.005))
        self.max_open_positions = mode_config.get("max_positions", self.profit_rotation_config.get("max_open_positions", 3))
        self.cooldown_minutes = self.profit_rotation_config.get("cooldown_minutes", 5)
        self.force_close_profitable = self.profit_rotation_config.get("force_close_profitable", True)
        self.auto_reinvest = self.profit_rotation_config.get("auto_reinvest", True)
        
        # Mode-specific position sizing
        if self.trading_mode == "spot":
            self.max_positions = self.config.get("spot_trading.max_positions", 3)
            self.position_size_multiplier = self.config.get("spot_trading.position_size_multiplier", 0.35)  # Increased from 0.25
            self.max_position_pct = self.config.get("spot_trading.max_position_pct", 0.4)  # Increased from 0.3
            self.min_order_value = self.config.get("spot_trading.min_order_value", 25.0)  # Increased from 12.0
        else:  # perpetual mode
            self.max_positions = self.config.get("perpetual_trading.max_positions", 5)
            self.position_size_multiplier = self.config.get("perpetual_trading.position_size_multiplier", 0.3)  # Increased from 0.2
            self.max_position_pct = self.config.get("perpetual_trading.max_position_pct", 0.6)  # Increased from 0.5
            self.min_order_value = self.config.get("perpetual_trading.min_order_value", 25.0)  # Increased from 15.0
            self.default_leverage = self.config.get("perpetual_trading.default_leverage", 2.0)
        
        # Mode-specific features
        if self.trading_mode == "perp":
            perp_config = self.config.get("perpetual_trading", {})
            self.max_leverage = perp_config.get("max_leverage", 5.0)
            self.margin_buffer = perp_config.get("margin_buffer", 0.2)
            self.logger.info(f"[PORTFOLIO] PERP MODE - Leverage: {self.default_leverage}x, Max positions: {self.max_open_positions}")
        else:
            spot_config = self.config.get("spot_trading", {})
            self.balance_threshold = spot_config.get("balance_threshold", 10.0)
            self.logger.info(f"[PORTFOLIO] SPOT MODE - Balance threshold: ${self.balance_threshold}, Max positions: {self.max_open_positions}")
        
        # State tracking
        self.open_positions: Dict[str, Position] = {}
        self.position_cooldowns: Dict[str, datetime] = {}
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Immediately seed with existing positions from API
        self._seed_existing_positions()
        
        self.logger.info(f"[PORTFOLIO] Mode-specific initialization complete:")
        self.logger.info(f"  - Trading mode: {self.trading_mode.upper()}")
        self.logger.info(f"  - Profit target: {self.profit_target_pct:.3f}%")
        self.logger.info(f"  - Min order value: ${self.min_order_value}")
        self.logger.info(f"  - Position size multiplier: {self.position_size_multiplier}")
        self.logger.info(f"  - Max positions: {self.max_open_positions}")
        self.logger.info(f"[PORTFOLIO] Seeded with {len(self.open_positions)} existing positions")
        
    def _detect_trading_mode(self) -> str:
        """Detect whether we're trading spot or perpetual assets"""
        try:
            # Check if there's an explicit mode setting
            explicit_mode = self.config.get("trading.asset_type")
            if explicit_mode in ["spot", "perp"]:
                return explicit_mode
            
            # Try to detect from available tokens
            available_tokens = self.config.get("available_tokens", [])
            if available_tokens and self.meta_manager:
                # Check the first few tokens to see if they're spot or perp
                for token in available_tokens[:3]:
                    asset_id = self.meta_manager.get_asset_id(token)
                    if asset_id is not None:
                        # In Hyperliquid, spot assets start at 10000
                        if asset_id >= 10000:
                            return "spot"
                        else:
                            return "perp"
            
            # Default to perp if detection fails
            self.logger.warning("[PORTFOLIO] Could not detect trading mode, defaulting to PERP")
            return "perp"
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error detecting trading mode: {e}")
            return "perp"
    
    def _seed_existing_positions(self):
        """Seed portfolio manager with existing positions from API"""
        try:
            # Get user state to see existing positions
            user_state = self.api.get_user_state()
            if not user_state or "assetPositions" not in user_state:
                self.logger.warning("[PORTFOLIO] Could not retrieve existing positions from user state")
                return
            
            asset_positions = user_state["assetPositions"]
            self.open_positions = {}
            
            for position_data in asset_positions:
                if "position" not in position_data:
                    continue
                
                position = position_data["position"]
                symbol = position.get("coin")
                size = safe_float(position.get("szi", "0"))
                
                if symbol and size != 0:
                    # Create Position object
                    entry_price = safe_float(position.get("entryPx", "0"))
                    unrealized_pnl = safe_float(position.get("unrealizedPnl", "0"))
                    position_value = safe_float(position.get("positionValue", "0"))
                    unrealized_pnl_pct = (unrealized_pnl / position_value * 100) if position_value > 0 else 0
                    
                    # Get asset ID from meta manager
                    asset_id = 0
                    if self.meta_manager:
                        raw_asset_id = self.meta_manager.get_asset_id(symbol)
                        if raw_asset_id is not None:
                            asset_id = int(raw_asset_id)
                    
                    pos = Position(
                        symbol=symbol,
                        asset_id=asset_id,
                        quantity=abs(size),
                        entry_price=entry_price,
                        current_price=entry_price,  # Will be updated with current price
                        side="long" if size > 0 else "short",
                        entry_time=datetime.now(),  # Approximate
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct
                    )
                    self.open_positions[symbol] = pos
                    self.logger.info(f"[PORTFOLIO] Seeded position: {symbol} {size} @ ${entry_price:.2f} (PnL: {unrealized_pnl_pct:.3f}%)")
            
            self.logger.info(f"[PORTFOLIO] Successfully seeded {len(self.open_positions)} existing positions")
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error seeding existing positions: {e}")
            self.open_positions = {}
    
    def compute_strategy_scores(self, token: str) -> Dict[str, float]:
        """Compute confidence scores for each strategy on a token"""
        try:
            scores = {}
            
            # Get available strategies
            strategies = ["scalping", "grid_trading", "mean_reversion", "rl_ai"]
            
            for strategy in strategies:
                if not self.config.get(f"strategies.{strategy}.enabled", False):
                    continue
                
                # Get historical Sharpe ratio for this strategy
                historical_sharpe = self.config.get(f"strategies.{strategy}.optimized_sharpe", 1.0)
                
                # Get current market conditions
                market_data = self.api.get_market_data(token)
                if not market_data:
                    continue
                
                # Calculate strategy-specific confidence
                confidence = self._calculate_strategy_confidence(strategy, token, market_data)
                
                # Combine with historical performance
                scores[strategy] = confidence * historical_sharpe
                
            return scores
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error computing strategy scores for {token}: {e}")
            return {}
    
    def _calculate_strategy_confidence(self, strategy: str, token: str, market_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a specific strategy"""
        try:
            price = market_data.get("price", 0)
            if price <= 0:
                return 0.0
            
            # Strategy-specific confidence calculations
            if strategy == "scalping":
                # Scalping works best with high volatility
                volatility = self._estimate_volatility(token)
                return min(volatility * 10, 1.0)  # Scale volatility to 0-1
                
            elif strategy == "grid_trading":
                # Grid trading works best in sideways markets
                trend_strength = self._estimate_trend_strength(token)
                return 1.0 - abs(trend_strength)  # Lower trend = higher confidence
                
            elif strategy == "mean_reversion":
                # Mean reversion works best with clear mean levels
                mean_deviation = self._estimate_mean_deviation(token)
                return min(mean_deviation * 5, 1.0)
                
            elif strategy == "rl_ai":
                # RL AI confidence based on model prediction
                return self._get_rl_confidence(token)
            
            return 0.5  # Default confidence
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error calculating confidence for {strategy}: {e}")
            return 0.5
    
    def _estimate_volatility(self, token: str) -> float:
        """Estimate current volatility for a token"""
        try:
            # Simplified volatility estimation
            # In production, use actual historical price data
            return 0.02  # 2% volatility estimate
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error estimating volatility: {e}")
            return 0.01
    
    def _estimate_trend_strength(self, token: str) -> float:
        """Estimate trend strength for a token"""
        try:
            # Simplified trend estimation
            # In production, use moving averages or other trend indicators
            return 0.1  # Slight uptrend
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error estimating trend strength: {e}")
            return 0.0
    
    def _estimate_mean_deviation(self, token: str) -> float:
        """Estimate deviation from mean for mean reversion"""
        try:
            # Simplified mean deviation estimation
            return 0.005  # 0.5% deviation
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error estimating mean deviation: {e}")
            return 0.0
    
    def _get_rl_confidence(self, token: str) -> float:
        """Get RL AI model confidence"""
        try:
            # Simplified RL confidence
            # In production, use actual model predictions
            return 0.7  # 70% confidence
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error getting RL confidence: {e}")
            return 0.5
    
    def compute_token_scores(self) -> Dict[str, float]:
        """Compute overall scores for each token across all strategies"""
        try:
            scores = {}
            
            for token in self.token_universe:
                strategy_scores = self.compute_strategy_scores(token)
                
                if strategy_scores:
                    # Aggregate strategy scores for this token
                    total_score = sum(strategy_scores.values())
                    scores[token] = total_score
                else:
                    scores[token] = 0.0
            
            return scores
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error computing token scores: {e}")
            return {}
    
    def allocate_portfolio(self, portfolio_value: float) -> Dict[str, float]:
        """Allocate portfolio across tokens based on scores"""
        try:
            scores = self.compute_token_scores()
            
            if not scores:
                self.logger.warning("[PORTFOLIO] No scores available for allocation")
                return {}
            
            # Normalize scores
            total_score = sum(abs(score) for score in scores.values())
            if total_score == 0:
                self.logger.warning("[PORTFOLIO] All scores are zero")
                return {}
            
            # Calculate weights
            weights = {}
            for token, score in scores.items():
                weights[token] = score / total_score
            
            # Apply allocation constraints
            max_token_alloc = self.config.get("portfolio.max_token_alloc", 0.3)
            min_token_alloc = self.config.get("portfolio.min_token_alloc", 0.05)
            
            # Cap maximum allocation per token
            for token in weights:
                weights[token] = min(weights[token], max_token_alloc)
            
            # Ensure minimum allocation for tokens with positive scores
            positive_tokens = [token for token, score in scores.items() if score > 0]
            if positive_tokens:
                for token in positive_tokens:
                    weights[token] = max(weights[token], min_token_alloc)
                
                # Renormalize after applying constraints
                total_weight = sum(weights.values())
                if total_weight > 0:
                    for token in weights:
                        weights[token] /= total_weight
            
            self.logger.info(f"[PORTFOLIO] Allocation computed: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error in portfolio allocation: {e}")
            return {}
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            user_state = self.api.get_user_state()
            if user_state and "marginSummary" in user_state:
                account_value = user_state["marginSummary"].get("accountValue", "0")
                # Remove commas and convert to float
                return safe_float(account_value.replace(",", ""))
            else:
                self.logger.warning("[PORTFOLIO] Could not get portfolio value from user state")
                return 1000.0  # Default value
                
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error getting portfolio value: {e}")
            return 1000.0
    
    def get_token_price(self, token: str) -> float:
        """Get current price for a token"""
        try:
            market_data = self.api.get_market_data(token)
            if market_data:
                return market_data.get("price", 0.0)
            else:
                self.logger.warning(f"[PORTFOLIO] No market data for {token}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error getting price for {token}: {e}")
            return 0.0
    
    def adjust_position(self, token: str, target_size: float) -> bool:
        """Adjust position size for a token"""
        try:
            # Get current position
            current_positions = self.get_open_positions()
            current_position = current_positions.get(token)
            current_size = current_position.quantity if current_position else 0
            
            # Calculate size difference
            size_diff = target_size - current_size
            
            if abs(size_diff) < 0.001:  # No adjustment needed
                return True
            
            # Determine order side and size
            if size_diff > 0:
                side = "buy"
                quantity = size_diff
            else:
                side = "sell"
                quantity = abs(size_diff)
            
            # Get current market price
            current_price = self.get_token_price(token)
            if current_price <= 0:
                self.logger.error(f"[PORTFOLIO] Invalid price for {token}: {current_price}")
                return False
            
            # Place order
            response = self.api.place_order(
                symbol=token,
                side=side,
                quantity=quantity,
                price=current_price,
                order_type="market"
            )
            
            if response and response.get("success") == True:
                self.logger.info(f"[PORTFOLIO] Position adjusted for {token}: {side} {quantity:.4f}")
                return True
            else:
                self.logger.error(f"[PORTFOLIO] Failed to adjust position for {token}: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error adjusting position for {token}: {e}")
            return False
    
    def rebalance_portfolio(self) -> Dict[str, Any]:
        """Rebalance the entire portfolio"""
        try:
            self.logger.info("[PORTFOLIO] Starting portfolio rebalancing")
            
            # Get current portfolio value
            portfolio_value = self.get_portfolio_value()
            
            # Calculate target allocations
            target_allocations = self.allocate_portfolio(portfolio_value)
            
            if not target_allocations:
                self.logger.warning("[PORTFOLIO] No target allocations computed")
                return {}
            
            # Execute rebalancing
            rebalance_results = {}
            
            for token, target_weight in target_allocations.items():
                target_size = portfolio_value * target_weight / self.get_token_price(token)
                
                success = self.adjust_position(token, target_size)
                rebalance_results[token] = {
                    "target_weight": target_weight,
                    "target_size": target_size,
                    "success": success
                }
            
            # Update current allocations
            self.current_allocations = target_allocations
            
            # Log rebalancing summary
            self.logger.info("[PORTFOLIO] Rebalancing complete")
            for token, result in rebalance_results.items():
                if result["success"]:
                    self.logger.info(f"[PORTFOLIO] {token}: {result['target_weight']:.2%} allocation")
            
            # Store in history
            self.portfolio_history.append({
                "timestamp": self._get_current_timestamp(),
                "portfolio_value": portfolio_value,
                "allocations": target_allocations,
                "results": rebalance_results
            })
            
            return rebalance_results
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error in portfolio rebalancing: {e}")
            return {}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary for profit rotation framework"""
        positions = self.get_open_positions()
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        total_unrealized_pnl_pct = sum(pos.unrealized_pnl_pct for pos in positions.values())
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Get total portfolio value from account
        total_value = self.get_portfolio_value()
        
        return {
            'total_value': total_value,
            'open_positions': len(positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_unrealized_pnl_pct': total_unrealized_pnl_pct,
            'total_realized_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'available_capital': self.get_available_capital(),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }
                for pos in positions.values()
            ]
        }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_rebalancing_history(self) -> List[Dict[str, Any]]:
        """Get portfolio rebalancing history"""
        return self.portfolio_history

    def get_open_positions(self) -> Dict[str, Position]:
        """Get current open positions with updated PnL"""
        try:
            user_state = self.api.get_user_state()
            if not user_state or 'assetPositions' not in user_state:
                return {}
                
            positions = {}
            for asset_pos in user_state['assetPositions']:
                if 'position' in asset_pos and asset_pos['position']['szi'] != '0':
                    # Get symbol from asset ID using reverse mapping
                    asset_id = asset_pos['position']['coin']
                    symbol = None
                    
                    # Try to get symbol from meta manager's mapping
                    if self.meta_manager and hasattr(self.meta_manager, '_mapping'):
                        for sym, info in self.meta_manager._mapping.items():
                            if info.get('asset_id') == asset_id:
                                symbol = sym
                                break
                    
                    # Fallback: use asset ID as symbol if mapping not found
                    if not symbol:
                        symbol = str(asset_id)
                    
                    if symbol:
                        quantity = safe_float(asset_pos['position']['szi'])
                        entry_price = safe_float(asset_pos['position']['entryPx'])
                        
                        # Get current price
                        current_price = self.get_current_price(symbol)
                        if current_price <= 0:
                            continue
                            
                        # Calculate PnL
                        side = 'long' if quantity > 0 else 'short'
                        if side == 'long':
                            unrealized_pnl = (current_price - entry_price) * abs(quantity)
                        else:
                            unrealized_pnl = (entry_price - current_price) * abs(quantity)
                            
                        unrealized_pnl_pct = (unrealized_pnl / (entry_price * abs(quantity))) * 100
                        # Ensure asset_id is always int, never None
                        asset_id = 0
                        if self.meta_manager:
                            raw_asset_id = self.meta_manager.get_asset_id(symbol)
                            if raw_asset_id is not None:
                                asset_id = int(raw_asset_id)
                        positions[symbol] = Position(
                            symbol=symbol,
                            asset_id=asset_id,
                            quantity=abs(quantity),
                            entry_price=entry_price,
                            current_price=current_price,
                            side=side,
                            entry_time=datetime.now(),  # We don't have exact entry time
                            unrealized_pnl=unrealized_pnl,
                            unrealized_pnl_pct=unrealized_pnl_pct
                        )
                        
            self.open_positions = positions
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        try:
            market_data = self.api.get_market_data(symbol)
            if market_data and 'price' in market_data:
                return safe_float(market_data['price'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    def should_close_position(self, position: Position) -> bool:
        """Check if position should be closed based on profit target"""
        return position.unrealized_pnl_pct >= self.profit_target_pct
    
    def close_position(self, position: Position) -> bool:
        """Close a position and record the trade"""
        try:
            # Determine order side (opposite of position side)
            order_side = 'sell' if position.side == 'long' else 'buy'
            
            # Place closing order
            order_result = self.api.place_order(
                symbol=position.symbol,
                side=order_side,
                quantity=position.quantity,
                price=position.current_price,
                order_type='market',
                time_in_force='Ioc',
                reduce_only=True
            )
            
            if order_result and order_result.get('success'):
                # Record the trade
                trade_record = {
                    'symbol': position.symbol,
                    'side': position.side,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'exit_price': position.current_price,
                    'pnl': position.unrealized_pnl,
                    'pnl_pct': position.unrealized_pnl_pct,
                    'entry_time': position.entry_time,
                    'exit_time': datetime.now(),
                    'reason': 'profit_target'
                }
                
                self.trade_history.append(trade_record)
                self.total_pnl += position.unrealized_pnl
                self.total_trades += 1
                
                if position.unrealized_pnl > 0:
                    self.winning_trades += 1
                
                # Set cooldown for this symbol
                self.position_cooldowns[position.symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)
                
                self.logger.info(f"✅ Closed profitable position: {position.symbol} "
                               f"({position.side}) - PnL: ${position.unrealized_pnl:.2f} "
                               f"({position.unrealized_pnl_pct:.2f}%)")
                
                return True
            else:
                self.logger.error(f"❌ Failed to close position: {position.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position {position.symbol}: {e}")
            return False
    
    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        if symbol not in self.position_cooldowns:
            return False
            
        cooldown_end = self.position_cooldowns[symbol] + timedelta(minutes=self.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        try:
            user_state = self.api.get_user_state()
            import json
            self.logger.debug(f"FULL USER STATE: {json.dumps(user_state, indent=2)}")
            # Use marginSummary[accountValue] - marginSummary[totalMarginUsed] for free margin
            margin_summary = user_state.get("marginSummary", {})
            account_value = safe_float(margin_summary.get("accountValue", 0))
            margin_used = safe_float(margin_summary.get("totalMarginUsed", 0))
            available_margin = account_value - margin_used
            if available_margin < 10:  # Minimum $10 to open new position
                self.logger.info(f"[PORTFOLIO] Insufficient margin for new position: ${available_margin:.2f}")
                return 0.0
            return available_margin
        except Exception as e:
            self.logger.error(f"Error getting available capital: {e}")
            return 0.0
    
    def rank_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Rank signals by confidence * Sharpe ratio"""
        # Filter out symbols in cooldown
        available_signals = [
            signal for signal in signals 
            if not self.is_symbol_in_cooldown(signal.symbol)
        ]
        
        # Rank by confidence * Sharpe ratio
        ranked_signals = sorted(
            available_signals,
            key=lambda x: x.confidence * x.sharpe_ratio,
            reverse=True
        )
        
        return ranked_signals
    
    def execute_rebalance(self, signals: List[TradeSignal]) -> bool:
        """Execute portfolio rebalancing with profit rotation"""
        try:
            # Step A: Scan and close profitable positions
            positions = self.get_open_positions()
            positions_closed = 0
            
            for symbol, position in positions.items():
                if self.should_close_position(position):
                    if self.close_position(position):
                        positions_closed += 1
            
            # Step B: Check if we can open new positions
            current_positions = len(self.open_positions)
            if self.at_max_positions():
                self.logger.info(f"STATUS At max positions ({current_positions}/{self.max_open_positions}), waiting for profit rotation")
                return True
            
            # Step C: Rank available signals
            ranked_signals = self.rank_signals(signals)
            if not ranked_signals:
                self.logger.info("STATUS No available signals after cooldown filtering")
                return True
            
            # Step D: Open new position with best signal
            best_signal = ranked_signals[0]
            available_capital = self.get_available_capital()
            
            if available_capital < 10:  # Minimum order value
                self.logger.info(f"CAPITAL Insufficient capital: ${available_capital:.2f}")
                return True
            
            # Calculate position size (use 80% of available capital)
            position_value = min(available_capital * 0.8, 20.0)  # Cap at $20
            
            # Get current price and calculate quantity
            current_price = self.get_current_price(best_signal.symbol)
            if current_price <= 0:
                self.logger.error(f"ERROR Invalid price for {best_signal.symbol}")
                return False
            
            quantity = position_value / current_price
            
            # Place buy order
            order_result = self.api.place_order(
                symbol=best_signal.symbol,
                side='buy',
                quantity=quantity,
                price=current_price,
                order_type='limit',
                time_in_force='Gtc',
                reduce_only=False
            )
            
            if order_result and order_result.get('success'):
                self.logger.info(f"Order placed successfully. Order ID: {order_result.get('order_id', 'Unknown')}")
                if order_result.get('filled_immediately'):
                    self.logger.info(f"Order filled immediately. Quantity: {order_result.get('quantity')} Price: ${order_result.get('price')}")
                else:
                    self.logger.info(f"Order status: {order_result.get('status', 'resting')}")
                
                # Update position tracking
                asset_id = 0
                if self.meta_manager:
                    raw_asset_id = self.meta_manager.get_asset_id(best_signal.symbol)
                    if raw_asset_id is not None:
                        asset_id = int(raw_asset_id)
                self.open_positions[best_signal.symbol] = Position(
                    symbol=best_signal.symbol,
                    asset_id=asset_id,
                    quantity=quantity,
                    entry_price=current_price,
                    current_price=current_price,
                    side="long",
                    entry_time=datetime.now(),
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0
                )
                self.position_cooldowns[best_signal.symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)
            else:
                error_msg = order_result.get('error', 'Unknown error') if order_result else 'No response'
                self.logger.error(f"Order failed: {error_msg}")
            
            return True
                
        except Exception as e:
            self.logger.error(f"Error in rebalance: {e}")
            return False
    
    def log_portfolio_status(self):
        """Log current portfolio status"""
        summary = self.get_portfolio_summary()
        
        self.logger.info("STATUS PORTFOLIO STATUS")
        self.logger.info(f"   Open Positions: {summary['open_positions']}")
        self.logger.info(f"   Unrealized PnL: ${summary['total_unrealized_pnl']:.2f} ({summary['total_unrealized_pnl_pct']:.2f}%)")
        self.logger.info(f"   Realized PnL: ${summary['total_realized_pnl']:.2f}")
        self.logger.info(f"   Win Rate: {summary['win_rate']:.1f}% ({summary['winning_trades']}/{summary['total_trades']})")
        self.logger.info(f"   Available Capital: ${summary['available_capital']:.2f}")
        
        if summary['positions']:
            self.logger.info("   Active Positions:")
            for pos in summary['positions']:
                self.logger.info(f"     {pos['symbol']} ({pos['side']}): "
                               f"{pos['quantity']:.4f} @ ${pos['entry_price']:.2f} "
                               f"→ ${pos['current_price']:.2f} "
                               f"({pos['unrealized_pnl_pct']:+.2f}%)")

    def _cleanup_stale_orders(self):
        """Cancel all open orders at the start of each cycle to free up margin."""
        try:
            open_orders = self.api.get_open_orders()
            if not open_orders:
                self.logger.info("[CLEANUP] No open orders to cancel.")
                return
            
            # Initialize tracking for recently cancelled orders
            if not hasattr(self, '_recently_cancelled_orders'):
                self._recently_cancelled_orders = set()
            
            # Filter out orders that are likely already filled or cancelled
            valid_orders = []
            current_time = int(time.time() * 1000)  # Convert to milliseconds
            
            for o in open_orders:
                order_id = o.get("oid") or o.get("orderId") or o.get("id")
                symbol = o.get("coin") or o.get("symbol")
                if not order_id or not symbol:
                    self.logger.warning(f"[CLEANUP] Skipping order with missing info: {o}")
                    continue
                
                # Skip orders that have been processed recently
                if order_id in self._recently_cancelled_orders:
                    continue
                
                # Skip orders that are clearly old (more than 1 minute)
                order_time = o.get("time", 0)
                if order_time and (current_time - order_time) > 60000:  # 1 minute
                    self.logger.info(f"[CLEANUP] Skipping old order {order_id} on {symbol} (age: {(current_time - order_time)/1000:.1f}s)")
                    continue
                
                # Skip specific known problematic orders
                if order_id == "108021255220":
                    self.logger.info(f"[CLEANUP] Skipping known problematic order {order_id} on {symbol}")
                    self._recently_cancelled_orders.add(order_id)
                    continue
                
                valid_orders.append((order_id, symbol))
            
            if not valid_orders:
                self.logger.info("[CLEANUP] No recent orders to cancel.")
                return
            
            # Cancel only recent orders
            for order_id, symbol in valid_orders:
                self.logger.info(f"[CLEANUP] Cancelling order {order_id} on {symbol}")
                try:
                    cancel_resp = self.api.cancel_order(order_id, symbol)
                    self.logger.info(f"[CLEANUP] Cancel response: {cancel_resp}")
                    
                    # Track this order as recently cancelled
                    self._recently_cancelled_orders.add(order_id)
                    
                    # Only verify if cancellation was successful
                    if cancel_resp and cancel_resp.get('status') == 'ok':
                        status = self.api.verify_order_on_exchange(order_id, symbol)
                        self.logger.info(f"[CLEANUP] Post-cancel status: {status}")
                except Exception as e:
                    self.logger.warning(f"[CLEANUP] Failed to cancel order {order_id} on {symbol}: {e}")
                    
            # Clean up old entries from recently cancelled orders (keep only last 5 minutes)
            if len(self._recently_cancelled_orders) > 100:
                # Clear the set if it gets too large
                self._recently_cancelled_orders.clear()
                    
        except Exception as e:
            self.logger.error(f"[CLEANUP] Error during stale order cleanup: {e}")

    def execute_profit_rotation_cycle(self):
        """Execute a complete profit rotation cycle"""
        try:
            self.logger.info("ROTATE Starting profit rotation cycle")

            # Step 0: Cleanup stale open orders
            self._cleanup_stale_orders()
            
            # Step A: Close any positions that meet profit target
            positions_closed = self._close_profitable_positions()
            
            # Step B: Update position tracking
            self._update_position_tracking()
            
            # Step C: Check if we can open new positions
            current_positions = len(self.open_positions)
            if self.at_max_positions():
                self.logger.info(f"STATUS At max positions ({current_positions}/{self.max_open_positions}), waiting for profit rotation")
                return True
            
            # Step D: Generate trading signals
            self.logger.info(f"[STRATEGY] Generating signals for {self.max_open_positions - current_positions} available slots")
            signals = self._generate_trading_signals()
            
            if signals:
                self.logger.info(f"[STRATEGY] Generated {len(signals)} signals, opening up to {self.max_open_positions - current_positions} positions")
                
                # Step E: Open new positions based on signals
                for signal in signals[:self.max_open_positions - current_positions]:
                    self._open_position_from_signal(signal)
            else:
                self.logger.info("[STRATEGY] No trading signals generated")
            
            # Step F: Log cycle summary
            self._log_cycle_summary(positions_closed)
            
            return True
            
        except Exception as e:
            self.logger.error(f"ERROR Error in profit rotation cycle: {e}")
            return False
    
    def _generate_trading_signals(self) -> List[TradeSignal]:
        """Generate trading signals for available tokens"""
        try:
            signals = []
            
            # Get available tokens (not in cooldown and not already open)
            available_tokens = self._get_available_tokens()
            
            if not available_tokens:
                self.logger.info("[STRATEGY] No tokens available (all in cooldown or already open)")
                return signals
            
            # Generate signals for each available token
            for token in available_tokens[:10]:  # Limit to top 10 tokens
                # Get market data
                market_data = self.api.get_market_data(token)
                if not market_data:
                    continue
                
                # Calculate opportunity score
                score = self._calculate_opportunity_score(token, market_data)
                
                if score > 0.3:  # Minimum threshold
                    signal = TradeSignal(
                        symbol=token,
                        confidence=score,
                        sharpe_ratio=score * 2,  # Simplified
                        signal_strength=score,
                        timestamp=datetime.now()
                    )
                    signals.append(signal)
                    self.logger.info(f"[STRATEGY] Generated signal for {token} (confidence: {score:.3f})")
            
            # Sort by confidence
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"[STRATEGY] Error generating trading signals: {e}")
            return []
    
    def _open_position_from_signal(self, signal: TradeSignal):
        """Open a position based on a trading signal"""
        try:
            symbol = signal.symbol
            self.logger.info(f"[EXECUTOR] Placing order for {symbol} (confidence: {signal.confidence:.3f})")
            
            # Get current market price
            market_data = self.api.get_market_data(symbol)
            if not market_data:
                self.logger.error(f"[EXECUTOR] Could not get market data for {symbol}")
                return
            
            current_price = safe_float(market_data.get("price", 0))
            if current_price == 0:
                self.logger.error(f"[EXECUTOR] Invalid price for {symbol}")
                return
            
            # Calculate position size (use mode-specific configuration)
            user_state = self.api.get_user_state()
            if not user_state:
                return
            
            withdrawable = safe_float(user_state.get("withdrawable", "0"))
            if withdrawable < 20:
                self.logger.info(f"[EXECUTOR] Insufficient margin for {symbol}: ${withdrawable:.2f}")
                return
            
            # Use mode-specific position sizing configuration
            # Use configured position size multiplier and minimum order value
            position_value = max(withdrawable * self.position_size_multiplier, self.min_order_value)
            
            # Cap at reasonable maximum to avoid overexposure
            max_position_value = withdrawable * self.max_position_pct
            position_value = min(position_value, max_position_value)
            
            self.logger.info(f"[EXECUTOR] {symbol} position sizing: ${position_value:.2f} (multiplier: {self.position_size_multiplier}, min: ${self.min_order_value})")
            
            raw_quantity = position_value / current_price
            
            # Validate and round order size/price using MetaManager
            asset_id, coin_name = self.api.resolve_symbol_to_asset_id(symbol)
            if asset_id is None:
                self.logger.error(f"Could not resolve symbol {symbol} to asset ID")
                return False
            try:
                quantity, price = self.api.validate_and_round_order(asset_id, raw_quantity, current_price)
            except Exception as e:
                self.logger.error(f"Order validation failed for {symbol}: {e}")
                return False
            order_result = self.api.place_order(
                symbol,
                "buy",
                quantity,
                price,
                order_type='limit',
                time_in_force='Gtc',
                reduce_only=False
            )
            if order_result and order_result.get('success'):
                self.logger.info(f"Order placed successfully. Order ID: {order_result.get('order_id', 'Unknown')}")
                if order_result.get('filled_immediately'):
                    self.logger.info(f"Order filled immediately. Quantity: {order_result.get('quantity')} Price: ${order_result.get('price')}")
                else:
                    self.logger.info(f"Order status: {order_result.get('status', 'resting')}")
                return True
            else:
                error_msg = order_result.get('error', 'Unknown error') if order_result else 'No response'
                self.logger.error(f"Order failed: {error_msg}")
                return False
            
        except Exception as e:
            self.logger.error(f"[EXECUTOR] Error opening position for {signal.symbol}: {e}")
    
    def _close_profitable_positions(self) -> List[Dict]:
        """Close positions that meet profit target"""
        closed_positions = []
        
        try:
            # Get current positions from user state
            user_state = self.api.get_user_state()
            if not user_state or "assetPositions" not in user_state:
                return closed_positions
            
            asset_positions = user_state["assetPositions"]
            
            for position_data in asset_positions:
                if "position" not in position_data:
                    continue
                
                position = position_data["position"]
                symbol = position.get("coin")
                if not symbol:
                    continue
                
                # Calculate unrealized PnL percentage
                unrealized_pnl = safe_float(position.get("unrealizedPnl", "0"))
                position_value = safe_float(position.get("positionValue", "0"))
                
                if position_value > 0:
                    pnl_pct = unrealized_pnl / position_value
                    
                    # Check if position meets profit target
                    if pnl_pct >= self.profit_target_pct:
                        self.logger.info(f"[PORTFOLIO] Closing profitable position: {symbol} at {pnl_pct:.3f}% profit")
                        
                        # Close the position
                        close_result = self._close_position(symbol, position)
                        
                        if close_result:
                            closed_positions.append({
                                "symbol": symbol,
                                "pnl_pct": pnl_pct,
                                "unrealized_pnl": unrealized_pnl,
                                "close_time": datetime.now()
                            })
                            
                            # Update cooldown
                            self.position_cooldowns[symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)
                            
                            # Update profit tracking
                            self.total_pnl += unrealized_pnl
                            self.total_trades += 1
                            
                            self.logger.info(f"[PORTFOLIO] Successfully closed {symbol} position. Total profit: ${self.total_pnl:.2f}")
        
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error closing profitable positions: {e}")
        
        return closed_positions
    
    def _close_position(self, symbol: str, position: Dict) -> bool:
        """Close a specific position"""
        try:
            size = safe_float(position.get("szi", "0"))
            if size == 0:
                return False
            side = "sell" if size > 0 else "buy"
            abs_size = abs(size)
            market_data = self.api.get_market_data(symbol)
            if not market_data or "price" not in market_data:
                self.logger.error(f"[PORTFOLIO] Could not get market price for {symbol}")
                return False
            current_price = safe_float(market_data["price"])
            # Validate and round close order size/price using HyperliquidAPI
            asset_id, coin_name = self.api.resolve_symbol_to_asset_id(symbol)
            if asset_id is None:
                self.logger.error(f"Could not resolve symbol {symbol} to asset ID")
                return False
            try:
                quantity, price = self.api.validate_and_round_order(asset_id, abs_size, current_price)
            except Exception as e:
                self.logger.error(f"Order validation failed for {symbol}: {e}")
                return False
            order_response = self.api.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type="market",
                time_in_force="Gtc",
                reduce_only=False
            )
            self.logger.info(f"[PORTFOLIO] Close order response for {symbol}: {order_response}")
            return order_response.get('success', False)
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error closing position for {symbol}: {e}")
            return False
    
    def _close_position_market_fallback(self, symbol: str, size: float, side: str) -> bool:
        """Fallback method to close position with market order"""
        try:
            self.logger.info(f"[PORTFOLIO] Attempting market order fallback for {symbol}")
            
            # Get current market price
            market_data = self.api.get_market_data(symbol)
            if not market_data or "price" not in market_data:
                self.logger.error(f"[PORTFOLIO] Could not get market price for {symbol} in fallback")
                return False
            
            current_price = safe_float(market_data["price"])
            
            # Place market order with reduce_only
            order_response = self.api.place_order(
                symbol=symbol,
                side=side,
                quantity=size,
                price=current_price,
                order_type="market",
                time_in_force="Gtc",
                reduce_only=True
            )
            
            self.logger.info(f"[PORTFOLIO] Market fallback response for {symbol}: {order_response}")
            
            if order_response and order_response.get('success'):
                order_id = order_response.get('order_id')
                status = order_response.get('status')
                filled_qty = order_response.get('quantity', 0)
                
                # Log detailed order confirmation
                if status == 'filled':
                    self.logger.info(f"[ORDER FILLED] ID: {order_id} Symbol: {symbol} Side: {side.upper()} Qty: {filled_qty} Price: ${current_price} (fallback)")
                elif status == 'resting':
                    self.logger.info(f"[ORDER RESTING] ID: {order_id} Symbol: {symbol} Side: {side.upper()} Qty: {size} Price: ${current_price} (fallback)")
                
                # Verify order on exchange
                if order_id:
                    is_verified = self.api.verify_order_on_exchange(order_id, symbol)
                    if is_verified:
                        self.logger.info(f"[ORDER VERIFIED] {order_id} confirmed on exchange for {symbol} (fallback)")
                    else:
                        self.logger.warning(f"[ORDER NOT VERIFIED] {order_id} not found on exchange for {symbol} (fallback)")
                
                return True
            else:
                error_msg = order_response.get('error', 'Unknown error') if order_response else 'No response'
                self.logger.error(f"[PORTFOLIO] Market fallback failed for {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error in market fallback for {symbol}: {e}")
            return False
    
    def _update_position_tracking(self):
        """Update internal position tracking"""
        try:
            # Get user state to see current positions
            user_state = self.api.get_user_state()
            if not user_state or "assetPositions" not in user_state:
                self.logger.warning("[PORTFOLIO] Could not retrieve user state for position tracking")
                return
            
            asset_positions = user_state["assetPositions"]
            self.open_positions = {}
            
            for position_data in asset_positions:
                if "position" not in position_data:
                    continue
                
                position = position_data["position"]
                symbol = position.get("coin")
                size = safe_float(position.get("szi", "0"))
                
                if symbol and size != 0:
                    # Create a Position object with correct parameters
                    entry_price = safe_float(position.get("entryPx", "0"))
                    current_price = entry_price  # Will be updated with current price
                    side = "long" if size > 0 else "short"
                    unrealized_pnl = safe_float(position.get("unrealizedPnl", "0"))
                    position_value = safe_float(position.get("positionValue", "0"))
                    
                    # Calculate unrealized PnL percentage
                    unrealized_pnl_pct = (unrealized_pnl / position_value * 100) if position_value > 0 else 0
                    
                    # Get asset ID from meta manager
                    asset_id = 0
                    if self.meta_manager:
                        raw_asset_id = self.meta_manager.get_asset_id(symbol)
                        if raw_asset_id is not None:
                            asset_id = int(raw_asset_id)
                    
                    pos = Position(
                        symbol=symbol,
                        asset_id=asset_id,
                        quantity=abs(size),
                        entry_price=entry_price,
                        current_price=current_price,
                        side=side,
                        entry_time=datetime.now(),  # Approximate
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct
                    )
                    self.open_positions[symbol] = pos
            
            self.logger.info(f"[PORTFOLIO] Updated position tracking: {len(self.open_positions)} open positions")
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error updating position tracking: {e}")
    
    def _open_new_positions(self):
        """Open new positions based on available strategies"""
        try:
            # Check available margin
            user_state = self.api.get_user_state()
            if not user_state:
                return
            
            withdrawable = safe_float(user_state.get("withdrawable", "0"))
            if withdrawable < 20:  # Minimum $20 to open new position
                self.logger.info(f"[PORTFOLIO] Insufficient margin for new position: ${withdrawable:.2f}")
                return
            
            # Get available tokens (not in cooldown)
            available_tokens = self._get_available_tokens()
            
            if not available_tokens:
                self.logger.info("[PORTFOLIO] No tokens available (all in cooldown)")
                return
            
            # Select best token and strategy
            best_opportunity = self._select_best_opportunity(available_tokens)
            
            if best_opportunity:
                symbol, strategy_name = best_opportunity
                self._open_position(symbol, strategy_name)
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error opening new positions: {e}")
    
    def _get_available_tokens(self) -> List[str]:
        """Get tokens that are not in cooldown and not already open"""
        try:
            # Get all available tokens from meta manager
            all_tokens = list(self.meta_manager._mapping.keys()) if self.meta_manager else ["BTC", "ETH", "SOL", "DOGE", "AVAX"]
            
            available_tokens = []
            current_time = datetime.now()
            
            for token in all_tokens:
                # Skip if already have position
                if token in self.open_positions:
                    continue
                
                # Skip if in cooldown
                if token in self.position_cooldowns:
                    cooldown_end = self.position_cooldowns[token]
                    if current_time < cooldown_end:
                        continue
                    else:
                        # Cooldown expired, remove from tracking
                        del self.position_cooldowns[token]
                
                available_tokens.append(token)
            
            return available_tokens
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error getting available tokens: {e}")
            return ["BTC", "ETH", "SOL"]  # Fallback
    
    def _select_best_opportunity(self, available_tokens: List[str]) -> Optional[Tuple[str, str]]:
        """Select the best trading opportunity from available tokens"""
        try:
            best_score = 0
            best_opportunity = None
            
            for token in available_tokens[:5]:  # Limit to top 5 tokens
                # Get market data
                market_data = self.api.get_market_data(token)
                if not market_data:
                    continue
                
                # Calculate opportunity score based on volatility and trend
                score = self._calculate_opportunity_score(token, market_data)
                
                if score > best_score:
                    best_score = score
                    best_opportunity = (token, "rl_ai")  # Default to RL AI strategy
            
            if best_opportunity:
                self.logger.info(f"[PORTFOLIO] Selected {best_opportunity[0]} as best opportunity (score: {best_score:.3f})")
            
            return best_opportunity
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error selecting best opportunity: {e}")
            return None
    
    def _calculate_opportunity_score(self, token: str, market_data: Dict) -> float:
        """Calculate opportunity score for a token"""
        try:
            # Simple scoring based on price movement
            # In production, this would use more sophisticated analysis
            
            current_price = safe_float(market_data.get("price", 0))
            if current_price == 0:
                return 0
            
            # Get recent price data (simplified)
            # In production, fetch actual historical data
            price_change = abs(current_price - (current_price * 0.99)) / current_price
            
            # Higher volatility = higher score (for scalping)
            score = price_change * 100
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error calculating opportunity score for {token}: {e}")
            return 0
    
    def _open_position(self, symbol: str, strategy_name: str):
        """Open a new position"""
        try:
            self.logger.info(f"[PORTFOLIO] Opening position for {symbol} using {strategy_name} strategy")
            
            # Get strategy
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if not strategy:
                self.logger.error(f"[PORTFOLIO] Strategy {strategy_name} not found")
                return
            
            # Get market data
            market_data = self.api.get_market_data(symbol)
            if not market_data:
                self.logger.error(f"[PORTFOLIO] Could not get market data for {symbol}")
                return
            
            # Prepare data for strategy
            data = {
                "symbol": symbol,
                "price": safe_float(market_data.get("price", 0)),
                "volume": safe_float(market_data.get("volume", 0)),
                "timestamp": time.time()
            }
            
            # Run strategy
            strategy.run(data, {})
            
            self.logger.info(f"[PORTFOLIO] Successfully initiated position for {symbol}")
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error opening position for {symbol}: {e}")
    
    def _log_cycle_summary(self, closed_positions: List[Dict]):
        """Log summary of the profit rotation cycle"""
        try:
            # Get portfolio summary for status logging
            summary = self.get_portfolio_summary()
            
            self.logger.info(f"[STATUS] Portfolio Value: ${summary.get('total_value', 0):.2f}")
            self.logger.info(f"[STATUS] Open Positions: {len(self.open_positions)}")
            self.logger.info(f"[STATUS] Total PnL: ${summary.get('total_unrealized_pnl', 0):.2f}")
            self.logger.info(f"[STATUS] Total Trades: {self.total_trades}")
            
            self.logger.info(f"[PORTFOLIO] Cycle Summary:")
            self.logger.info(f"  - Open positions: {len(self.open_positions)}")
            self.logger.info(f"  - Positions closed: {len(closed_positions)}")
            self.logger.info(f"  - Total profit: ${self.total_pnl:.2f}")
            self.logger.info(f"  - Total trades: {self.total_trades}")
            
            if closed_positions:
                for pos in closed_positions:
                    self.logger.info(f"  - Closed {pos['symbol']}: {pos['pnl_pct']:.3f}% profit")
            
        except Exception as e:
            self.logger.error(f"[PORTFOLIO] Error logging cycle summary: {e}")

    def at_max_positions(self) -> bool:
        """Check if we're at maximum positions"""
        return len(self.open_positions) >= self.max_open_positions
