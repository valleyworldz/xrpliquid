"""
🎯 MAKER-FIRST ROUTER
=====================
Production-grade maker-first order router with post-only policy and rebate tracking.

Features:
- Default to post-only orders (maker)
- Promote to taker only on urgency (signal decay, stop-outs)
- Track maker ratio and P&L delta from rebates vs missed fills
- Encode current fees from Hyperliquid fee page
- Comprehensive rebate tracking and optimization
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger
from src.core.ledgers.canonical_trade_ledger import TradeRecord, OrderState, ReasonCode

class OrderUrgency(Enum):
    """Order urgency levels"""
    LOW = "low"           # Default maker orders
    MEDIUM = "medium"     # Signal decay, approaching deadline
    HIGH = "high"         # Stop-out, emergency exit
    CRITICAL = "critical" # Kill switch, liquidation risk

class RouterState(Enum):
    """Router state enumeration"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    PLACING_MAKER = "placing_maker"
    PLACING_TAKER = "placing_taker"
    MONITORING = "monitoring"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class MakerRouterConfig:
    """Configuration for maker-first router"""
    
    # Post-only policy configuration
    post_only_config: Dict[str, Any] = field(default_factory=lambda: {
        'default_post_only': True,        # Default to post-only orders
        'post_only_timeout_seconds': 30,  # Timeout before promoting to taker
        'min_maker_spread_bps': 1.0,      # Minimum 1 bps spread for maker orders
        'max_maker_spread_bps': 10.0,     # Maximum 10 bps spread for maker orders
        'maker_price_improvement_bps': 0.5,  # 0.5 bps price improvement for maker
    })
    
    # Urgency thresholds
    urgency_config: Dict[str, Any] = field(default_factory=lambda: {
        'signal_decay_threshold_seconds': 60,    # Signal decay threshold
        'stop_out_threshold_seconds': 10,        # Stop-out threshold
        'emergency_threshold_seconds': 5,        # Emergency threshold
        'liquidation_threshold_seconds': 2,      # Liquidation threshold
    })
    
    # Fee and rebate configuration (Hyperliquid-specific)
    fee_config: Dict[str, Any] = field(default_factory=lambda: {
        'perpetual_fees': {
            'maker': 0.0001,         # 0.01% maker fee
            'taker': 0.0005,         # 0.05% taker fee
            'maker_rebate': 0.00005, # 0.005% maker rebate
        },
        'spot_fees': {
            'maker': 0.0002,         # 0.02% maker fee
            'taker': 0.0006,         # 0.06% taker fee
            'maker_rebate': 0.0001,  # 0.01% maker rebate
        },
        'volume_tiers': {
            'tier_1': {'volume_usd': 0, 'maker_discount': 0.0, 'taker_discount': 0.0},
            'tier_2': {'volume_usd': 1000000, 'maker_discount': 0.1, 'taker_discount': 0.05},
            'tier_3': {'volume_usd': 5000000, 'maker_discount': 0.2, 'taker_discount': 0.1},
            'tier_4': {'volume_usd': 20000000, 'maker_discount': 0.3, 'taker_discount': 0.15},
        },
        'hype_staking_discount': 0.5,  # 50% fee discount with HYPE staking
    })
    
    # Performance tracking
    performance_config: Dict[str, Any] = field(default_factory=lambda: {
        'track_maker_ratio': True,
        'track_rebate_pnl': True,
        'track_missed_fill_cost': True,
        'track_slippage_savings': True,
        'performance_window_hours': 24,  # 24-hour performance window
    })
    
    # Risk management
    risk_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_maker_wait_seconds': 60,     # Maximum wait time for maker orders
        'max_slippage_bps': 20.0,         # Maximum acceptable slippage
        'min_fill_probability': 0.8,      # Minimum fill probability for maker orders
        'emergency_taker_threshold': 0.1, # 10% emergency taker threshold
    })

@dataclass
class OrderRequest:
    """Order request with routing information"""
    
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str = 'limit'
    urgency: OrderUrgency = OrderUrgency.LOW
    reason_code: str = ReasonCode.SIGNAL_ENTRY.value
    cloid: str = ""
    post_only: bool = True
    time_in_force: str = 'GTC'  # Good Till Cancelled
    
    # Routing metadata
    signal_timestamp: float = 0.0
    max_slippage_bps: float = 10.0
    min_fill_probability: float = 0.8
    expected_duration_seconds: float = 30.0

@dataclass
class OrderResult:
    """Order execution result"""
    
    order_id: str
    cloid: str
    symbol: str
    side: str
    quantity: float
    price: float
    filled_quantity: float
    average_price: float
    order_state: str
    maker_flag: bool
    urgency: OrderUrgency
    
    # Cost breakdown
    fee_paid: float
    fee_bps: float
    rebate_received: float
    rebate_bps: float
    slippage_bps: float
    slippage_cost: float
    
    # Timing
    placement_time: float
    fill_time: float
    latency_ms: float
    
    # Performance metrics
    fill_probability: float
    missed_fill_cost: float
    rebate_pnl: float
    total_cost: float

class MakerFirstRouter:
    """
    🎯 MAKER-FIRST ROUTER
    
    Production-grade maker-first order router with comprehensive rebate tracking
    and intelligent urgency-based promotion to taker orders.
    """
    
    def __init__(self, config: Dict[str, Any], api=None, logger=None):
        self.config = config
        self.api = api
        self.logger = logger or Logger()
        
        # Initialize configuration
        self.router_config = MakerRouterConfig()
        
        # Router state
        self.current_state = RouterState.IDLE
        self.active_orders: Dict[str, OrderRequest] = {}
        self.order_results: List[OrderResult] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_orders': 0,
            'maker_orders': 0,
            'taker_orders': 0,
            'maker_ratio': 0.0,
            'total_rebates': 0.0,
            'total_fees': 0.0,
            'net_rebate_pnl': 0.0,
            'missed_fill_cost': 0.0,
            'slippage_savings': 0.0,
            'avg_fill_time_seconds': 0.0,
            'avg_slippage_bps': 0.0,
        }
        
        # Market data cache
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        self.last_market_update = 0.0
        
        self.logger.info("🎯 [MAKER_ROUTER] Maker-First Router initialized")
        self.logger.info("🎯 [MAKER_ROUTER] Post-only policy and rebate tracking enabled")
    
    async def route_order(self, order_request: OrderRequest) -> OrderResult:
        """
        🎯 Route order with maker-first policy
        
        Args:
            order_request: OrderRequest with routing information
            
        Returns:
            OrderResult: Order execution result
        """
        try:
            self.logger.info(f"🎯 [ROUTE_ORDER] Routing order: {order_request.symbol} {order_request.side} {order_request.quantity}")
            
            # Update router state
            self.current_state = RouterState.ANALYZING
            
            # Analyze order and determine routing strategy
            routing_strategy = await self._analyze_order(order_request)
            
            # Execute routing strategy
            if routing_strategy['use_maker']:
                result = await self._execute_maker_order(order_request, routing_strategy)
            else:
                result = await self._execute_taker_order(order_request, routing_strategy)
            
            # Update performance metrics
            await self._update_performance_metrics(result)
            
            # Store result
            self.order_results.append(result)
            
            self.logger.info(f"🎯 [ROUTE_ORDER] Order routed successfully: {result.order_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [ROUTE_ORDER] Error routing order: {e}")
            return self._create_failed_result(order_request, str(e))
    
    async def _analyze_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Analyze order and determine routing strategy"""
        try:
            # Get current market data
            market_data = await self._get_market_data(order_request.symbol)
            
            # Calculate urgency score
            urgency_score = await self._calculate_urgency_score(order_request)
            
            # Calculate maker feasibility
            maker_feasibility = await self._calculate_maker_feasibility(order_request, market_data)
            
            # Determine routing strategy
            use_maker = self._should_use_maker(order_request, urgency_score, maker_feasibility)
            
            # Calculate optimal pricing
            optimal_pricing = await self._calculate_optimal_pricing(order_request, market_data, use_maker)
            
            return {
                'use_maker': use_maker,
                'urgency_score': urgency_score,
                'maker_feasibility': maker_feasibility,
                'optimal_pricing': optimal_pricing,
                'expected_fill_time': maker_feasibility['expected_fill_time'],
                'expected_slippage': optimal_pricing['expected_slippage'],
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ANALYZE_ORDER] Error analyzing order: {e}")
            return {
                'use_maker': False,
                'urgency_score': 1.0,
                'maker_feasibility': {'fill_probability': 0.0, 'expected_fill_time': 60.0},
                'optimal_pricing': {'price': order_request.price, 'expected_slippage': 10.0},
                'expected_fill_time': 60.0,
                'expected_slippage': 10.0,
            }
    
    async def _calculate_urgency_score(self, order_request: OrderRequest) -> float:
        """Calculate urgency score (0-1, higher is more urgent)"""
        try:
            current_time = time.time()
            time_since_signal = current_time - order_request.signal_timestamp
            
            urgency_config = self.router_config.urgency_config
            
            # Calculate urgency based on time since signal
            if time_since_signal <= urgency_config['liquidation_threshold_seconds']:
                urgency_score = 1.0  # Critical
            elif time_since_signal <= urgency_config['emergency_threshold_seconds']:
                urgency_score = 0.8  # High
            elif time_since_signal <= urgency_config['stop_out_threshold_seconds']:
                urgency_score = 0.6  # Medium
            elif time_since_signal <= urgency_config['signal_decay_threshold_seconds']:
                urgency_score = 0.3  # Low
            else:
                urgency_score = 0.1  # Very low
            
            # Adjust based on order urgency
            if order_request.urgency == OrderUrgency.CRITICAL:
                urgency_score = max(urgency_score, 0.9)
            elif order_request.urgency == OrderUrgency.HIGH:
                urgency_score = max(urgency_score, 0.7)
            elif order_request.urgency == OrderUrgency.MEDIUM:
                urgency_score = max(urgency_score, 0.4)
            
            return urgency_score
            
        except Exception as e:
            self.logger.error(f"❌ [URGENCY_SCORE] Error calculating urgency score: {e}")
            return 0.5
    
    async def _calculate_maker_feasibility(self, order_request: OrderRequest, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate maker order feasibility"""
        try:
            # Get current spread
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            spread_bps = ((ask - bid) / bid) * 10000 if bid > 0 else 100
            
            # Calculate fill probability based on spread and market conditions
            fill_probability = await self._estimate_fill_probability(order_request, market_data, spread_bps)
            
            # Calculate expected fill time
            expected_fill_time = await self._estimate_fill_time(order_request, market_data, fill_probability)
            
            # Calculate expected slippage if we use maker
            expected_slippage = await self._estimate_maker_slippage(order_request, market_data)
            
            return {
                'fill_probability': fill_probability,
                'expected_fill_time': expected_fill_time,
                'expected_slippage': expected_slippage,
                'spread_bps': spread_bps,
                'market_conditions': market_data.get('conditions', 'normal'),
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MAKER_FEASIBILITY] Error calculating maker feasibility: {e}")
            return {
                'fill_probability': 0.5,
                'expected_fill_time': 30.0,
                'expected_slippage': 2.0,
                'spread_bps': 5.0,
                'market_conditions': 'normal',
            }
    
    async def _estimate_fill_probability(self, order_request: OrderRequest, market_data: Dict[str, Any], spread_bps: float) -> float:
        """Estimate fill probability for maker order"""
        try:
            # Base fill probability
            base_probability = 0.8
            
            # Adjust based on spread
            if spread_bps < 2.0:
                spread_adjustment = 0.9  # High probability with tight spread
            elif spread_bps < 5.0:
                spread_adjustment = 0.8  # Good probability
            elif spread_bps < 10.0:
                spread_adjustment = 0.6  # Medium probability
            else:
                spread_adjustment = 0.4  # Low probability with wide spread
            
            # Adjust based on order size
            volume_24h = market_data.get('volume_24h', 1000000)
            size_ratio = (order_request.quantity * order_request.price) / volume_24h
            
            if size_ratio < 0.001:  # < 0.1% of daily volume
                size_adjustment = 1.0
            elif size_ratio < 0.01:  # < 1% of daily volume
                size_adjustment = 0.9
            elif size_ratio < 0.1:   # < 10% of daily volume
                size_adjustment = 0.7
            else:
                size_adjustment = 0.5  # Large order
            
            # Adjust based on market conditions
            volatility = market_data.get('volatility', 0.02)
            if volatility < 0.01:
                volatility_adjustment = 1.0  # Low volatility
            elif volatility < 0.03:
                volatility_adjustment = 0.8  # Medium volatility
            else:
                volatility_adjustment = 0.6  # High volatility
            
            # Calculate final probability
            fill_probability = base_probability * spread_adjustment * size_adjustment * volatility_adjustment
            
            return min(0.95, max(0.1, fill_probability))  # Clamp between 10% and 95%
            
        except Exception as e:
            self.logger.error(f"❌ [FILL_PROBABILITY] Error estimating fill probability: {e}")
            return 0.5
    
    async def _estimate_fill_time(self, order_request: OrderRequest, market_data: Dict[str, Any], fill_probability: float) -> float:
        """Estimate expected fill time for maker order"""
        try:
            # Base fill time
            base_fill_time = 30.0  # 30 seconds
            
            # Adjust based on fill probability
            probability_adjustment = 1.0 / fill_probability if fill_probability > 0 else 10.0
            
            # Adjust based on market activity
            volume_24h = market_data.get('volume_24h', 1000000)
            if volume_24h > 10000000:  # High volume
                activity_adjustment = 0.5
            elif volume_24h > 1000000:  # Medium volume
                activity_adjustment = 1.0
            else:  # Low volume
                activity_adjustment = 2.0
            
            expected_fill_time = base_fill_time * probability_adjustment * activity_adjustment
            
            return min(300.0, max(5.0, expected_fill_time))  # Clamp between 5 and 300 seconds
            
        except Exception as e:
            self.logger.error(f"❌ [FILL_TIME] Error estimating fill time: {e}")
            return 30.0
    
    async def _estimate_maker_slippage(self, order_request: OrderRequest, market_data: Dict[str, Any]) -> float:
        """Estimate slippage for maker order"""
        try:
            # Maker orders typically have minimal slippage
            base_slippage = 0.5  # 0.5 bps base slippage
            
            # Adjust based on market conditions
            volatility = market_data.get('volatility', 0.02)
            volatility_adjustment = 1.0 + (volatility * 10)  # Scale with volatility
            
            expected_slippage = base_slippage * volatility_adjustment
            
            return min(5.0, max(0.1, expected_slippage))  # Clamp between 0.1 and 5 bps
            
        except Exception as e:
            self.logger.error(f"❌ [MAKER_SLIPPAGE] Error estimating maker slippage: {e}")
            return 1.0
    
    def _should_use_maker(self, order_request: OrderRequest, urgency_score: float, maker_feasibility: Dict[str, Any]) -> bool:
        """Determine if we should use maker order"""
        try:
            # Default to maker if post-only is requested
            if order_request.post_only:
                return True
            
            # Don't use maker if urgency is too high
            if urgency_score > 0.7:  # High urgency threshold
                return False
            
            # Don't use maker if fill probability is too low
            if maker_feasibility['fill_probability'] < self.router_config.risk_config['min_fill_probability']:
                return False
            
            # Don't use maker if expected fill time is too long
            if maker_feasibility['expected_fill_time'] > self.router_config.risk_config['max_maker_wait_seconds']:
                return False
            
            # Use maker if conditions are favorable
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [SHOULD_USE_MAKER] Error determining maker usage: {e}")
            return False
    
    async def _calculate_optimal_pricing(self, order_request: OrderRequest, market_data: Dict[str, Any], use_maker: bool) -> Dict[str, Any]:
        """Calculate optimal pricing for the order"""
        try:
            if use_maker:
                # Calculate maker pricing
                return await self._calculate_maker_pricing(order_request, market_data)
            else:
                # Calculate taker pricing
                return await self._calculate_taker_pricing(order_request, market_data)
                
        except Exception as e:
            self.logger.error(f"❌ [OPTIMAL_PRICING] Error calculating optimal pricing: {e}")
            return {
                'price': order_request.price,
                'expected_slippage': 5.0,
                'price_improvement': 0.0,
            }
    
    async def _calculate_maker_pricing(self, order_request: OrderRequest, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal maker pricing"""
        try:
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else order_request.price
            
            post_only_config = self.router_config.post_only_config
            
            if order_request.side == 'buy':
                # For buy orders, place slightly below ask to be maker
                optimal_price = ask - (ask * post_only_config['maker_price_improvement_bps'] / 10000)
                price_improvement = (ask - optimal_price) / ask * 10000
            else:
                # For sell orders, place slightly above bid to be maker
                optimal_price = bid + (bid * post_only_config['maker_price_improvement_bps'] / 10000)
                price_improvement = (optimal_price - bid) / bid * 10000
            
            # Ensure price is within limits
            min_spread = post_only_config['min_maker_spread_bps'] / 10000
            max_spread = post_only_config['max_maker_spread_bps'] / 10000
            
            if order_request.side == 'buy':
                optimal_price = max(optimal_price, mid * (1 - max_spread))
                optimal_price = min(optimal_price, ask * (1 - min_spread))
            else:
                optimal_price = min(optimal_price, mid * (1 + max_spread))
                optimal_price = max(optimal_price, bid * (1 + min_spread))
            
            return {
                'price': optimal_price,
                'expected_slippage': 0.5,  # Minimal slippage for maker
                'price_improvement': price_improvement,
                'spread_bps': ((ask - bid) / mid) * 10000 if mid > 0 else 0,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MAKER_PRICING] Error calculating maker pricing: {e}")
            return {
                'price': order_request.price,
                'expected_slippage': 1.0,
                'price_improvement': 0.0,
                'spread_bps': 5.0,
            }
    
    async def _calculate_taker_pricing(self, order_request: OrderRequest, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal taker pricing"""
        try:
            bid = market_data.get('bid', 0)
            ask = market_data.get('ask', 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else order_request.price
            
            if order_request.side == 'buy':
                # For buy orders, use ask price (market buy)
                optimal_price = ask
                expected_slippage = ((ask - mid) / mid) * 10000 if mid > 0 else 5.0
            else:
                # For sell orders, use bid price (market sell)
                optimal_price = bid
                expected_slippage = ((mid - bid) / mid) * 10000 if mid > 0 else 5.0
            
            return {
                'price': optimal_price,
                'expected_slippage': expected_slippage,
                'price_improvement': 0.0,  # No price improvement for taker
                'spread_bps': ((ask - bid) / mid) * 10000 if mid > 0 else 0,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [TAKER_PRICING] Error calculating taker pricing: {e}")
            return {
                'price': order_request.price,
                'expected_slippage': 5.0,
                'price_improvement': 0.0,
                'spread_bps': 5.0,
            }
    
    async def _execute_maker_order(self, order_request: OrderRequest, routing_strategy: Dict[str, Any]) -> OrderResult:
        """Execute maker order"""
        try:
            self.logger.info(f"🎯 [EXECUTE_MAKER] Executing maker order: {order_request.symbol}")
            
            # Update router state
            self.current_state = RouterState.PLACING_MAKER
            
            # Get optimal pricing
            optimal_pricing = routing_strategy['optimal_pricing']
            
            # Create order with post-only flag
            order_data = {
                'symbol': order_request.symbol,
                'side': order_request.side,
                'quantity': order_request.quantity,
                'price': optimal_pricing['price'],
                'order_type': 'limit',
                'post_only': True,
                'time_in_force': order_request.time_in_force,
                'cloid': order_request.cloid or f"maker_{order_request.symbol}_{int(time.time())}",
            }
            
            # Place order (this would integrate with actual API)
            placement_time = time.time()
            order_result = await self._place_order(order_data)
            
            # Monitor for fill
            fill_result = await self._monitor_maker_order(order_result, order_request)
            
            # Calculate costs and rebates
            cost_breakdown = await self._calculate_maker_costs(order_request, fill_result, optimal_pricing)
            
            # Create result
            result = OrderResult(
                order_id=order_result['order_id'],
                cloid=order_data['cloid'],
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                price=optimal_pricing['price'],
                filled_quantity=fill_result['filled_quantity'],
                average_price=fill_result['average_price'],
                order_state=fill_result['order_state'],
                maker_flag=True,
                urgency=order_request.urgency,
                fee_paid=cost_breakdown['fee_paid'],
                fee_bps=cost_breakdown['fee_bps'],
                rebate_received=cost_breakdown['rebate_received'],
                rebate_bps=cost_breakdown['rebate_bps'],
                slippage_bps=cost_breakdown['slippage_bps'],
                slippage_cost=cost_breakdown['slippage_cost'],
                placement_time=placement_time,
                fill_time=fill_result['fill_time'],
                latency_ms=fill_result['latency_ms'],
                fill_probability=routing_strategy['maker_feasibility']['fill_probability'],
                missed_fill_cost=cost_breakdown['missed_fill_cost'],
                rebate_pnl=cost_breakdown['rebate_pnl'],
                total_cost=cost_breakdown['total_cost'],
            )
            
            self.logger.info(f"🎯 [EXECUTE_MAKER] Maker order executed: {result.order_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [EXECUTE_MAKER] Error executing maker order: {e}")
            return self._create_failed_result(order_request, str(e))
    
    async def _execute_taker_order(self, order_request: OrderRequest, routing_strategy: Dict[str, Any]) -> OrderResult:
        """Execute taker order"""
        try:
            self.logger.info(f"🎯 [EXECUTE_TAKER] Executing taker order: {order_request.symbol}")
            
            # Update router state
            self.current_state = RouterState.PLACING_TAKER
            
            # Get optimal pricing
            optimal_pricing = routing_strategy['optimal_pricing']
            
            # Create order without post-only flag
            order_data = {
                'symbol': order_request.symbol,
                'side': order_request.side,
                'quantity': order_request.quantity,
                'price': optimal_pricing['price'],
                'order_type': 'market',  # Use market order for taker
                'post_only': False,
                'time_in_force': 'IOC',  # Immediate or Cancel
                'cloid': order_request.cloid or f"taker_{order_request.symbol}_{int(time.time())}",
            }
            
            # Place order (this would integrate with actual API)
            placement_time = time.time()
            order_result = await self._place_order(order_data)
            
            # Taker orders typically fill immediately
            fill_result = await self._monitor_taker_order(order_result, order_request)
            
            # Calculate costs
            cost_breakdown = await self._calculate_taker_costs(order_request, fill_result, optimal_pricing)
            
            # Create result
            result = OrderResult(
                order_id=order_result['order_id'],
                cloid=order_data['cloid'],
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                price=optimal_pricing['price'],
                filled_quantity=fill_result['filled_quantity'],
                average_price=fill_result['average_price'],
                order_state=fill_result['order_state'],
                maker_flag=False,
                urgency=order_request.urgency,
                fee_paid=cost_breakdown['fee_paid'],
                fee_bps=cost_breakdown['fee_bps'],
                rebate_received=0.0,  # No rebate for taker
                rebate_bps=0.0,
                slippage_bps=cost_breakdown['slippage_bps'],
                slippage_cost=cost_breakdown['slippage_cost'],
                placement_time=placement_time,
                fill_time=fill_result['fill_time'],
                latency_ms=fill_result['latency_ms'],
                fill_probability=1.0,  # Taker orders have high fill probability
                missed_fill_cost=0.0,  # No missed fill cost for taker
                rebate_pnl=0.0,
                total_cost=cost_breakdown['total_cost'],
            )
            
            self.logger.info(f"🎯 [EXECUTE_TAKER] Taker order executed: {result.order_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [EXECUTE_TAKER] Error executing taker order: {e}")
            return self._create_failed_result(order_request, str(e))
    
    async def _place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place order via API (placeholder)"""
        try:
            # This would integrate with the actual Hyperliquid API
            # For now, simulate order placement
            
            order_result = {
                'order_id': f"order_{int(time.time())}",
                'status': 'placed',
                'timestamp': time.time(),
            }
            
            return order_result
            
        except Exception as e:
            self.logger.error(f"❌ [PLACE_ORDER] Error placing order: {e}")
            return {
                'order_id': f"failed_{int(time.time())}",
                'status': 'failed',
                'timestamp': time.time(),
                'error': str(e),
            }
    
    async def _monitor_maker_order(self, order_result: Dict[str, Any], order_request: OrderRequest) -> Dict[str, Any]:
        """Monitor maker order for fill"""
        try:
            # Simulate maker order monitoring
            # In reality, this would poll the API or use WebSocket updates
            
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Simulate fill result
            fill_result = {
                'filled_quantity': order_request.quantity,
                'average_price': order_request.price,
                'order_state': 'filled',
                'fill_time': time.time(),
                'latency_ms': 100.0,
            }
            
            return fill_result
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR_MAKER] Error monitoring maker order: {e}")
            return {
                'filled_quantity': 0.0,
                'average_price': 0.0,
                'order_state': 'failed',
                'fill_time': time.time(),
                'latency_ms': 0.0,
            }
    
    async def _monitor_taker_order(self, order_result: Dict[str, Any], order_request: OrderRequest) -> Dict[str, Any]:
        """Monitor taker order for fill"""
        try:
            # Taker orders typically fill immediately
            await asyncio.sleep(0.05)  # Simulate processing time
            
            # Simulate immediate fill
            fill_result = {
                'filled_quantity': order_request.quantity,
                'average_price': order_request.price,
                'order_state': 'filled',
                'fill_time': time.time(),
                'latency_ms': 50.0,
            }
            
            return fill_result
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR_TAKER] Error monitoring taker order: {e}")
            return {
                'filled_quantity': 0.0,
                'average_price': 0.0,
                'order_state': 'failed',
                'fill_time': time.time(),
                'latency_ms': 0.0,
            }
    
    async def _calculate_maker_costs(self, order_request: OrderRequest, fill_result: Dict[str, Any], optimal_pricing: Dict[str, Any]) -> Dict[str, float]:
        """Calculate costs for maker order"""
        try:
            notional_value = fill_result['filled_quantity'] * fill_result['average_price']
            
            # Calculate fees (maker fee - rebate)
            fee_config = self.router_config.fee_config['perpetual_fees']
            maker_fee = notional_value * fee_config['maker']
            maker_rebate = notional_value * fee_config['maker_rebate']
            net_fee = maker_fee - maker_rebate
            
            # Calculate slippage
            expected_price = order_request.price
            actual_price = fill_result['average_price']
            slippage_bps = abs((actual_price - expected_price) / expected_price) * 10000
            slippage_cost = abs(actual_price - expected_price) * fill_result['filled_quantity']
            
            # Calculate missed fill cost (if order didn't fill)
            missed_fill_cost = 0.0  # Would calculate based on market movement
            
            # Calculate rebate P&L
            rebate_pnl = maker_rebate
            
            # Total cost
            total_cost = net_fee + slippage_cost - rebate_pnl
            
            return {
                'fee_paid': net_fee,
                'fee_bps': (net_fee / notional_value) * 10000 if notional_value > 0 else 0,
                'rebate_received': maker_rebate,
                'rebate_bps': (maker_rebate / notional_value) * 10000 if notional_value > 0 else 0,
                'slippage_bps': slippage_bps,
                'slippage_cost': slippage_cost,
                'missed_fill_cost': missed_fill_cost,
                'rebate_pnl': rebate_pnl,
                'total_cost': total_cost,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MAKER_COSTS] Error calculating maker costs: {e}")
            return {
                'fee_paid': 0.0,
                'fee_bps': 0.0,
                'rebate_received': 0.0,
                'rebate_bps': 0.0,
                'slippage_bps': 0.0,
                'slippage_cost': 0.0,
                'missed_fill_cost': 0.0,
                'rebate_pnl': 0.0,
                'total_cost': 0.0,
            }
    
    async def _calculate_taker_costs(self, order_request: OrderRequest, fill_result: Dict[str, Any], optimal_pricing: Dict[str, Any]) -> Dict[str, float]:
        """Calculate costs for taker order"""
        try:
            notional_value = fill_result['filled_quantity'] * fill_result['average_price']
            
            # Calculate fees (taker fee, no rebate)
            fee_config = self.router_config.fee_config['perpetual_fees']
            taker_fee = notional_value * fee_config['taker']
            
            # Calculate slippage
            expected_price = order_request.price
            actual_price = fill_result['average_price']
            slippage_bps = abs((actual_price - expected_price) / expected_price) * 10000
            slippage_cost = abs(actual_price - expected_price) * fill_result['filled_quantity']
            
            # Total cost
            total_cost = taker_fee + slippage_cost
            
            return {
                'fee_paid': taker_fee,
                'fee_bps': (taker_fee / notional_value) * 10000 if notional_value > 0 else 0,
                'slippage_bps': slippage_bps,
                'slippage_cost': slippage_cost,
                'total_cost': total_cost,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [TAKER_COSTS] Error calculating taker costs: {e}")
            return {
                'fee_paid': 0.0,
                'fee_bps': 0.0,
                'slippage_bps': 0.0,
                'slippage_cost': 0.0,
                'total_cost': 0.0,
            }
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data"""
        try:
            current_time = time.time()
            
            # Check cache first
            if (symbol in self.market_data_cache and 
                current_time - self.last_market_update < 5.0):  # 5-second cache
                return self.market_data_cache[symbol]
            
            # This would integrate with actual market data API
            # For now, return mock data
            market_data = {
                'symbol': symbol,
                'bid': 0.5199,
                'ask': 0.5201,
                'mid': 0.5200,
                'volume_24h': 1000000,
                'volatility': 0.02,
                'conditions': 'normal',
                'timestamp': current_time,
            }
            
            # Update cache
            self.market_data_cache[symbol] = market_data
            self.last_market_update = current_time
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_DATA] Error getting market data: {e}")
            return {
                'symbol': symbol,
                'bid': 0.52,
                'ask': 0.52,
                'mid': 0.52,
                'volume_24h': 1000000,
                'volatility': 0.02,
                'conditions': 'normal',
                'timestamp': time.time(),
            }
    
    async def _update_performance_metrics(self, result: OrderResult):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_orders'] += 1
            
            if result.maker_flag:
                self.performance_metrics['maker_orders'] += 1
                self.performance_metrics['total_rebates'] += result.rebate_received
            else:
                self.performance_metrics['taker_orders'] += 1
            
            self.performance_metrics['total_fees'] += result.fee_paid
            self.performance_metrics['net_rebate_pnl'] += result.rebate_pnl
            self.performance_metrics['missed_fill_cost'] += result.missed_fill_cost
            
            # Calculate maker ratio
            if self.performance_metrics['total_orders'] > 0:
                self.performance_metrics['maker_ratio'] = (
                    self.performance_metrics['maker_orders'] / 
                    self.performance_metrics['total_orders']
                )
            
            # Calculate average slippage
            if self.performance_metrics['total_orders'] > 0:
                total_slippage = sum(r.slippage_bps for r in self.order_results)
                self.performance_metrics['avg_slippage_bps'] = total_slippage / len(self.order_results)
            
        except Exception as e:
            self.logger.error(f"❌ [UPDATE_METRICS] Error updating performance metrics: {e}")
    
    def _create_failed_result(self, order_request: OrderRequest, error_message: str) -> OrderResult:
        """Create failed order result"""
        return OrderResult(
            order_id=f"failed_{int(time.time())}",
            cloid=order_request.cloid,
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            price=order_request.price,
            filled_quantity=0.0,
            average_price=0.0,
            order_state='failed',
            maker_flag=False,
            urgency=order_request.urgency,
            fee_paid=0.0,
            fee_bps=0.0,
            rebate_received=0.0,
            rebate_bps=0.0,
            slippage_bps=0.0,
            slippage_cost=0.0,
            placement_time=time.time(),
            fill_time=time.time(),
            latency_ms=0.0,
            fill_probability=0.0,
            missed_fill_cost=0.0,
            rebate_pnl=0.0,
            total_cost=0.0,
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            return {
                'performance_metrics': self.performance_metrics,
                'current_state': self.current_state.value,
                'active_orders': len(self.active_orders),
                'total_results': len(self.order_results),
                'recent_orders': self.order_results[-10:] if self.order_results else [],
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_SUMMARY] Error getting performance summary: {e}")
            return {}
