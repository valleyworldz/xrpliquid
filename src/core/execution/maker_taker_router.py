"""
🎯 MAKER/TAKER ROUTER WITH SLIPPAGE MODELING
============================================
Production-grade order routing with maker/taker optimization and slippage modeling.

Features:
- Post-only maker routing by default
- Taker promotion on urgency
- Slippage modeling and tracking
- Maker ratio optimization
- Rebate vs opportunity cost analysis
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.utils.logger import Logger

@dataclass
class OrderRoutingConfig:
    """Configuration for maker/taker routing"""
    
    # Routing preferences
    default_to_maker: bool = True          # Default to maker orders
    maker_timeout_seconds: int = 30        # Timeout before promoting to taker
    urgency_threshold: float = 0.01        # 1% price movement for urgency
    
    # Fee structure
    maker_fee: float = 0.0001              # 0.01% maker fee
    taker_fee: float = 0.0005              # 0.05% taker fee
    maker_rebate: float = 0.00005          # 0.005% maker rebate
    
    # Slippage modeling
    base_slippage: float = 0.0002          # 0.02% base slippage
    volatility_multiplier: float = 2.0     # Volatility impact on slippage
    volume_impact_factor: float = 0.5      # Volume impact on slippage
    
    # Performance tracking
    target_maker_ratio: float = 0.8        # Target 80% maker orders
    min_maker_ratio: float = 0.6           # Minimum 60% maker orders

@dataclass
class OrderRequest:
    """Order request data structure"""
    
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'limit', 'market', 'post_only'
    strategy: str
    urgency_level: float = 0.0  # 0.0 = normal, 1.0 = urgent
    max_slippage_bps: float = 5.0  # Maximum slippage in basis points
    timeout_seconds: int = 30

@dataclass
class OrderResult:
    """Order execution result"""
    
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    executed_price: float
    executed_quantity: float
    order_type: str
    is_maker: bool
    fee: float
    rebate: float
    net_fee: float
    slippage_bps: float
    latency_ms: float
    retry_count: int
    success: bool
    error_message: Optional[str] = None

class MakerTakerRouter:
    """
    🎯 MAKER/TAKER ROUTER WITH SLIPPAGE MODELING
    
    Production-grade order routing with maker/taker optimization,
    slippage modeling, and performance tracking.
    """
    
    def __init__(self, config: OrderRoutingConfig, logger=None):
        self.config = config
        self.logger = logger or Logger()
        
        # Routing state
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.order_history: List[OrderResult] = []
        
        # Performance tracking
        self.maker_orders = 0
        self.taker_orders = 0
        self.total_fees = 0.0
        self.total_rebates = 0.0
        self.total_slippage = 0.0
        self.missed_fills = 0
        
        # Market conditions
        self.current_volatility = 0.02
        self.current_volume = 1000000.0
        self.current_spread = 0.0001
        
        self.logger.info("🎯 [MAKER_TAKER_ROUTER] Maker/Taker Router initialized")
        self.logger.info(f"🎯 [MAKER_TAKER_ROUTER] Default to maker: {self.config.default_to_maker}")
        self.logger.info(f"🎯 [MAKER_TAKER_ROUTER] Target maker ratio: {self.config.target_maker_ratio:.1%}")
    
    async def route_order(self, order_request: OrderRequest) -> OrderResult:
        """
        Route order with maker/taker optimization
        
        Args:
            order_request: OrderRequest with order details
            
        Returns:
            OrderResult: Order execution result
        """
        try:
            self.logger.info(f"🎯 [ROUTE_ORDER] Routing order: {order_request.symbol} {order_request.side} {order_request.quantity} @ {order_request.price}")
            
            # Determine routing strategy
            routing_strategy = await self._determine_routing_strategy(order_request)
            
            # Execute order based on strategy
            if routing_strategy == 'maker_first':
                result = await self._execute_maker_first(order_request)
            elif routing_strategy == 'taker_only':
                result = await self._execute_taker_only(order_request)
            else:
                result = await self._execute_hybrid(order_request)
            
            # Update performance metrics
            await self._update_performance_metrics(result)
            
            # Log result
            self.logger.info(f"🎯 [ROUTE_ORDER] Order executed: {result.order_id} - {result.order_type} - Maker: {result.is_maker}")
            self.logger.info(f"🎯 [ROUTE_ORDER] Fee: ${result.fee:.4f}, Rebate: ${result.rebate:.4f}, Slippage: {result.slippage_bps:.2f} bps")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [ROUTE_ORDER] Error routing order: {e}")
            return OrderResult(
                order_id="",
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                price=order_request.price,
                executed_price=0.0,
                executed_quantity=0.0,
                order_type=order_request.order_type,
                is_maker=False,
                fee=0.0,
                rebate=0.0,
                net_fee=0.0,
                slippage_bps=0.0,
                latency_ms=0.0,
                retry_count=0,
                success=False,
                error_message=str(e)
            )
    
    async def _determine_routing_strategy(self, order_request: OrderRequest) -> str:
        """Determine optimal routing strategy"""
        try:
            # Check urgency level
            if order_request.urgency_level > 0.7:
                return 'taker_only'
            
            # Check current maker ratio
            current_maker_ratio = self.maker_orders / (self.maker_orders + self.taker_orders) if (self.maker_orders + self.taker_orders) > 0 else 0.5
            
            if current_maker_ratio < self.config.min_maker_ratio:
                return 'maker_first'
            
            # Check market conditions
            if self.current_volatility > 0.05:  # High volatility
                return 'taker_only'
            
            # Default to maker first
            return 'maker_first'
            
        except Exception as e:
            self.logger.error(f"❌ [ROUTING_STRATEGY] Error determining routing strategy: {e}")
            return 'maker_first'
    
    async def _execute_maker_first(self, order_request: OrderRequest) -> OrderResult:
        """Execute maker-first routing strategy"""
        try:
            start_time = time.time()
            
            # Try maker order first
            maker_result = await self._execute_maker_order(order_request)
            
            if maker_result.success:
                return maker_result
            
            # If maker order fails, promote to taker
            self.logger.info(f"🎯 [MAKER_FIRST] Maker order failed, promoting to taker")
            taker_result = await self._execute_taker_order(order_request)
            
            # Update retry count
            taker_result.retry_count = 1
            
            return taker_result
            
        except Exception as e:
            self.logger.error(f"❌ [MAKER_FIRST] Error in maker-first execution: {e}")
            return await self._create_failed_result(order_request, str(e))
    
    async def _execute_taker_only(self, order_request: OrderRequest) -> OrderResult:
        """Execute taker-only routing strategy"""
        try:
            return await self._execute_taker_order(order_request)
            
        except Exception as e:
            self.logger.error(f"❌ [TAKER_ONLY] Error in taker-only execution: {e}")
            return await self._create_failed_result(order_request, str(e))
    
    async def _execute_hybrid(self, order_request: OrderRequest) -> OrderResult:
        """Execute hybrid routing strategy"""
        try:
            # Start with maker order
            maker_result = await self._execute_maker_order(order_request)
            
            if maker_result.success:
                return maker_result
            
            # If maker order fails, try taker
            taker_result = await self._execute_taker_order(order_request)
            taker_result.retry_count = 1
            
            return taker_result
            
        except Exception as e:
            self.logger.error(f"❌ [HYBRID] Error in hybrid execution: {e}")
            return await self._create_failed_result(order_request, str(e))
    
    async def _execute_maker_order(self, order_request: OrderRequest) -> OrderResult:
        """Execute maker order"""
        try:
            start_time = time.time()
            
            # Simulate order execution
            # In production, this would interact with the exchange API
            
            # Calculate slippage (maker orders have minimal slippage)
            slippage_bps = self._calculate_slippage(order_request, is_maker=True)
            
            # Calculate fees and rebates
            notional_value = order_request.quantity * order_request.price
            fee = notional_value * self.config.maker_fee
            rebate = notional_value * self.config.maker_rebate
            net_fee = fee - rebate
            
            # Simulate execution
            executed_price = order_request.price * (1 + slippage_bps / 10000)
            executed_quantity = order_request.quantity
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = OrderResult(
                order_id=f"maker_{int(time.time() * 1000)}",
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                price=order_request.price,
                executed_price=executed_price,
                executed_quantity=executed_quantity,
                order_type='post_only',
                is_maker=True,
                fee=fee,
                rebate=rebate,
                net_fee=net_fee,
                slippage_bps=slippage_bps,
                latency_ms=latency_ms,
                retry_count=0,
                success=True,
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [MAKER_ORDER] Error executing maker order: {e}")
            return await self._create_failed_result(order_request, str(e))
    
    async def _execute_taker_order(self, order_request: OrderRequest) -> OrderResult:
        """Execute taker order"""
        try:
            start_time = time.time()
            
            # Simulate order execution
            # In production, this would interact with the exchange API
            
            # Calculate slippage (taker orders have higher slippage)
            slippage_bps = self._calculate_slippage(order_request, is_maker=False)
            
            # Calculate fees (no rebate for taker orders)
            notional_value = order_request.quantity * order_request.price
            fee = notional_value * self.config.taker_fee
            rebate = 0.0
            net_fee = fee
            
            # Simulate execution
            executed_price = order_request.price * (1 + slippage_bps / 10000)
            executed_quantity = order_request.quantity
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = OrderResult(
                order_id=f"taker_{int(time.time() * 1000)}",
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                price=order_request.price,
                executed_price=executed_price,
                executed_quantity=executed_quantity,
                order_type='market',
                is_maker=False,
                fee=fee,
                rebate=rebate,
                net_fee=net_fee,
                slippage_bps=slippage_bps,
                latency_ms=latency_ms,
                retry_count=0,
                success=True,
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ [TAKER_ORDER] Error executing taker order: {e}")
            return await self._create_failed_result(order_request, str(e))
    
    def _calculate_slippage(self, order_request: OrderRequest, is_maker: bool) -> float:
        """Calculate expected slippage"""
        try:
            # Base slippage
            base_slippage = self.config.base_slippage
            
            # Volatility impact
            volatility_impact = self.current_volatility * self.config.volatility_multiplier
            
            # Volume impact
            volume_impact = (order_request.quantity / self.current_volume) * self.config.volume_impact_factor
            
            # Maker vs taker adjustment
            if is_maker:
                slippage_multiplier = 0.1  # Maker orders have minimal slippage
            else:
                slippage_multiplier = 1.0  # Taker orders have full slippage
            
            # Calculate total slippage
            total_slippage = (base_slippage + volatility_impact + volume_impact) * slippage_multiplier
            
            # Convert to basis points
            slippage_bps = total_slippage * 10000
            
            # Cap at maximum slippage
            slippage_bps = min(slippage_bps, order_request.max_slippage_bps)
            
            return slippage_bps
            
        except Exception as e:
            self.logger.error(f"❌ [CALCULATE_SLIPPAGE] Error calculating slippage: {e}")
            return 0.0
    
    async def _create_failed_result(self, order_request: OrderRequest, error_message: str) -> OrderResult:
        """Create failed order result"""
        return OrderResult(
            order_id="",
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            price=order_request.price,
            executed_price=0.0,
            executed_quantity=0.0,
            order_type=order_request.order_type,
            is_maker=False,
            fee=0.0,
            rebate=0.0,
            net_fee=0.0,
            slippage_bps=0.0,
            latency_ms=0.0,
            retry_count=0,
            success=False,
            error_message=error_message
        )
    
    async def _update_performance_metrics(self, result: OrderResult):
        """Update performance metrics"""
        try:
            # Add to order history
            self.order_history.append(result)
            
            # Update counters
            if result.is_maker:
                self.maker_orders += 1
            else:
                self.taker_orders += 1
            
            # Update financial metrics
            self.total_fees += result.fee
            self.total_rebates += result.rebate
            self.total_slippage += result.slippage_bps / 10000 * result.quantity * result.executed_price
            
            # Check for missed fills
            if not result.success:
                self.missed_fills += 1
            
        except Exception as e:
            self.logger.error(f"❌ [UPDATE_METRICS] Error updating performance metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            total_orders = self.maker_orders + self.taker_orders
            maker_ratio = self.maker_orders / total_orders if total_orders > 0 else 0.0
            
            return {
                'total_orders': total_orders,
                'maker_orders': self.maker_orders,
                'taker_orders': self.taker_orders,
                'maker_ratio': maker_ratio,
                'target_maker_ratio': self.config.target_maker_ratio,
                'maker_ratio_vs_target': maker_ratio - self.config.target_maker_ratio,
                'total_fees': self.total_fees,
                'total_rebates': self.total_rebates,
                'net_fees': self.total_fees - self.total_rebates,
                'total_slippage': self.total_slippage,
                'missed_fills': self.missed_fills,
                'fill_rate': (total_orders - self.missed_fills) / total_orders if total_orders > 0 else 0.0,
                'avg_slippage_bps': np.mean([order.slippage_bps for order in self.order_history]) if self.order_history else 0.0,
                'avg_latency_ms': np.mean([order.latency_ms for order in self.order_history]) if self.order_history else 0.0,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_SUMMARY] Error getting performance summary: {e}")
            return {}
    
    def update_market_conditions(self, volatility: float, volume: float, spread: float):
        """Update market conditions for routing decisions"""
        try:
            self.current_volatility = volatility
            self.current_volume = volume
            self.current_spread = spread
            
            self.logger.info(f"🎯 [MARKET_CONDITIONS] Updated: Vol={volatility:.3f}, Vol={volume:,.0f}, Spread={spread:.5f}")
            
        except Exception as e:
            self.logger.error(f"❌ [MARKET_CONDITIONS] Error updating market conditions: {e}")
