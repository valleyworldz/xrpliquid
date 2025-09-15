"""
🏗️ THE HYPERLIQUID EXCHANGE ARCHITECT
"I built this place. I know its every secret passage, weak point, and pressure point."

This module implements advanced Hyperliquid-specific optimizations:
- vAMM mechanics exploitation
- Funding rate arbitrage (hourly vs 8h on other platforms)
- Liquidation flow capture
- Maker-taker optimization for continuous rebates
- HLP vault optimization
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import logging
from collections import deque

@dataclass
class FundingRateOpportunity:
    """Represents a funding rate arbitrage opportunity"""
    symbol: str
    current_rate: float
    predicted_rate: float
    time_to_funding: int  # seconds
    expected_profit: float
    risk_score: float
    confidence: float

@dataclass
class LiquidationFlow:
    """Represents liquidation flow data"""
    symbol: str
    liquidation_size: float
    liquidation_price: float
    time_detected: float
    backstop_ratio: float
    estimated_recovery: float

@dataclass
class vAMMOpportunity:
    """Represents vAMM-specific trading opportunities"""
    symbol: str
    impact_price: float
    oracle_price: float
    premium: float
    funding_rate: float
    optimal_size: float
    expected_slippage: float

class HyperliquidArchitect:
    """
    The Hyperliquid Exchange Architect - Master of the vAMM
    
    This class exploits Hyperliquid's unique mechanics:
    1. Hourly funding rates (vs 8h on other platforms)
    2. vAMM with no clearance fees
    3. Liquidation flow capture
    4. Maker rebates and taker optimization
    5. HLP vault integration
    """
    
    def __init__(self, api_client, logger=None):
        self.api = api_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Funding rate arbitrage parameters
        self.funding_opportunities: Dict[str, FundingRateOpportunity] = {}
        self.funding_history: Dict[str, deque] = {}
        self.funding_predictions: Dict[str, float] = {}
        
        # Liquidation flow tracking
        self.liquidation_flows: List[LiquidationFlow] = []
        self.liquidation_cooldowns: Dict[str, float] = {}
        
        # vAMM optimization
        self.vamm_opportunities: Dict[str, vAMMOpportunity] = {}
        self.oracle_prices: Dict[str, float] = {}
        self.impact_prices: Dict[str, float] = {}
        
        # Performance tracking
        self.architect_metrics = {
            'funding_profits': 0.0,
            'liquidation_captures': 0,
            'vamm_optimizations': 0,
            'maker_rebates_earned': 0.0,
            'total_opportunities': 0
        }
        
        # Initialize funding rate monitoring
        self._initialize_funding_monitoring()
        
    def _initialize_funding_monitoring(self):
        """Initialize funding rate monitoring for all active symbols"""
        try:
            # Get all active symbols from meta
            meta = self.api.meta_manager._mapping
            for symbol, asset_info in meta.items():
                if asset_info.get('type') == 'perp':
                    self.funding_history[symbol] = deque(maxlen=24)  # 24 hours of data
                    self.logger.info(f"Initialized funding monitoring for {symbol}")
        except Exception as e:
            self.logger.error(f"Failed to initialize funding monitoring: {e}")
    
    async def analyze_funding_rate_arbitrage(self, symbol: str) -> Optional[FundingRateOpportunity]:
        """
        Analyze funding rate arbitrage opportunities
        
        Hyperliquid's hourly funding vs 8h on other platforms creates unique opportunities
        """
        try:
            # Get current funding rate
            current_rate = await self._get_current_funding_rate(symbol)
            if current_rate is None:
                return None
            
            # Get time to next funding
            time_to_funding = await self._get_time_to_next_funding()
            
            # Predict next funding rate using ML model
            predicted_rate = await self._predict_funding_rate(symbol, current_rate)
            
            # Calculate expected profit
            expected_profit = self._calculate_funding_profit(
                current_rate, predicted_rate, time_to_funding
            )
            
            # Calculate risk score
            risk_score = self._calculate_funding_risk(symbol, current_rate, predicted_rate)
            
            # Calculate confidence
            confidence = self._calculate_funding_confidence(symbol, current_rate)
            
            if expected_profit > 0.001 and confidence > 0.7:  # 0.1% minimum profit
                opportunity = FundingRateOpportunity(
                    symbol=symbol,
                    current_rate=current_rate,
                    predicted_rate=predicted_rate,
                    time_to_funding=time_to_funding,
                    expected_profit=expected_profit,
                    risk_score=risk_score,
                    confidence=confidence
                )
                
                self.funding_opportunities[symbol] = opportunity
                self.architect_metrics['total_opportunities'] += 1
                
                self.logger.info(f"Funding opportunity detected for {symbol}: "
                               f"Rate={current_rate:.6f}, Profit={expected_profit:.4f}, "
                               f"Confidence={confidence:.2f}")
                
                return opportunity
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing funding arbitrage for {symbol}: {e}")
            return None
    
    async def detect_liquidation_flow(self, symbol: str) -> Optional[LiquidationFlow]:
        """
        Detect and analyze liquidation flow opportunities
        
        Hyperliquid's liquidation system creates predictable flow patterns
        """
        try:
            # Get current position and liquidation data
            user_state = await self._get_user_state()
            positions = user_state.get('assetPositions', [])
            
            # Find position for symbol
            position = None
            for pos in positions:
                if pos.get('position', {}).get('coin') == symbol:
                    position = pos
                    break
            
            if not position:
                return None
            
            # Calculate backstop ratio
            backstop_ratio = self._calculate_backstop_ratio(position)
            
            # Check if liquidation is imminent
            if backstop_ratio < 0.67:  # Hyperliquid liquidation threshold
                liquidation_size = self._estimate_liquidation_size(position)
                liquidation_price = self._estimate_liquidation_price(symbol, position)
                estimated_recovery = self._estimate_liquidation_recovery(symbol, liquidation_size)
                
                # Check cooldown
                if symbol in self.liquidation_cooldowns:
                    if time.time() - self.liquidation_cooldowns[symbol] < 30:  # 30s cooldown
                        return None
                
                flow = LiquidationFlow(
                    symbol=symbol,
                    liquidation_size=liquidation_size,
                    liquidation_price=liquidation_price,
                    time_detected=time.time(),
                    backstop_ratio=backstop_ratio,
                    estimated_recovery=estimated_recovery
                )
                
                self.liquidation_flows.append(flow)
                self.liquidation_cooldowns[symbol] = time.time()
                self.architect_metrics['liquidation_captures'] += 1
                
                self.logger.warning(f"Liquidation flow detected for {symbol}: "
                                  f"Size={liquidation_size:.2f}, Price={liquidation_price:.4f}, "
                                  f"Recovery={estimated_recovery:.4f}")
                
                return flow
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidation flow for {symbol}: {e}")
            return None
    
    async def optimize_vamm_execution(self, symbol: str, side: str, size: float) -> Dict[str, Any]:
        """
        Optimize execution using vAMM mechanics
        
        Exploits Hyperliquid's vAMM for optimal execution
        """
        try:
            # Get current oracle and impact prices
            oracle_price = await self._get_oracle_price(symbol)
            impact_price = await self._calculate_impact_price(symbol, side, size)
            
            # Calculate premium
            premium = (impact_price - oracle_price) / oracle_price
            
            # Get current funding rate
            funding_rate = await self._get_current_funding_rate(symbol)
            
            # Calculate optimal execution parameters
            optimal_size = self._calculate_optimal_vamm_size(symbol, side, size, premium)
            expected_slippage = self._calculate_expected_slippage(symbol, side, optimal_size)
            
            # Create vAMM opportunity
            opportunity = vAMMOpportunity(
                symbol=symbol,
                impact_price=impact_price,
                oracle_price=oracle_price,
                premium=premium,
                funding_rate=funding_rate or 0.0,
                optimal_size=optimal_size,
                expected_slippage=expected_slippage
            )
            
            self.vamm_opportunities[symbol] = opportunity
            self.architect_metrics['vamm_optimizations'] += 1
            
            # Return optimized execution parameters
            return {
                'optimal_size': optimal_size,
                'expected_slippage': expected_slippage,
                'premium': premium,
                'funding_rate': funding_rate,
                'oracle_price': oracle_price,
                'impact_price': impact_price,
                'execution_strategy': self._determine_execution_strategy(opportunity)
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing vAMM execution for {symbol}: {e}")
            return {}
    
    async def optimize_maker_taker_strategy(self, symbol: str, side: str, size: float) -> Dict[str, Any]:
        """
        Optimize maker-taker strategy for maximum rebates
        
        Hyperliquid offers maker rebates and optimized taker fees
        """
        try:
            # Get current VIP tier and staking status
            vip_tier = await self._get_vip_tier()
            staking_tier = await self._get_staking_tier()
            
            # Calculate maker rebate
            maker_rebate = self._calculate_maker_rebate(vip_tier, staking_tier)
            
            # Calculate taker fee
            taker_fee = self._calculate_taker_fee(vip_tier, staking_tier)
            
            # Determine optimal order type
            order_type = self._determine_optimal_order_type(
                symbol, side, size, maker_rebate, taker_fee
            )
            
            # Calculate spread for maker orders
            optimal_spread = self._calculate_optimal_spread(symbol, maker_rebate)
            
            return {
                'order_type': order_type,
                'maker_rebate': maker_rebate,
                'taker_fee': taker_fee,
                'optimal_spread': optimal_spread,
                'expected_fee': maker_rebate if order_type == 'limit' else taker_fee,
                'strategy': 'maker_preferred' if maker_rebate > 0 else 'taker_optimized'
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing maker-taker strategy for {symbol}: {e}")
            return {}
    
    async def analyze_hlp_vault_opportunity(self) -> Dict[str, Any]:
        """
        Analyze HLP vault opportunities for passive income
        
        Hyperliquid's HLP vault offers competitive yields
        """
        try:
            # Get current HLP vault data
            vault_data = await self._get_hlp_vault_data()
            
            # Calculate current APY
            current_apy = vault_data.get('apy', 0.0)
            
            # Calculate optimal allocation
            optimal_allocation = self._calculate_optimal_hlp_allocation(current_apy)
            
            # Calculate expected returns
            expected_returns = self._calculate_hlp_returns(optimal_allocation, current_apy)
            
            return {
                'current_apy': current_apy,
                'optimal_allocation': optimal_allocation,
                'expected_returns': expected_returns,
                'recommendation': 'deposit' if current_apy > 0.15 else 'hold',
                'auto_compound': True,
                'rebalance_threshold': 0.05
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing HLP vault opportunity: {e}")
            return {}
    
    # Helper methods for funding rate analysis
    async def _get_current_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for symbol"""
        try:
            # This would integrate with Hyperliquid's funding rate API
            # For now, return a mock value
            return 0.0001  # 0.01% hourly
        except Exception as e:
            self.logger.error(f"Error getting funding rate for {symbol}: {e}")
            return None
    
    async def _get_time_to_next_funding(self) -> int:
        """Get seconds until next funding"""
        try:
            # Calculate time to next hour
            now = time.time()
            next_hour = ((int(now) // 3600) + 1) * 3600
            return int(next_hour - now)
        except Exception as e:
            self.logger.error(f"Error calculating time to next funding: {e}")
            return 3600
    
    async def _predict_funding_rate(self, symbol: str, current_rate: float) -> float:
        """Predict next funding rate using ML model"""
        try:
            # Simple prediction model - in production, use ML
            history = self.funding_history.get(symbol, deque())
            if len(history) > 5:
                recent_avg = np.mean(list(history)[-5:])
                # Predict based on trend
                trend = (current_rate - recent_avg) * 0.5
                return current_rate + trend
            return current_rate
        except Exception as e:
            self.logger.error(f"Error predicting funding rate for {symbol}: {e}")
            return current_rate
    
    def _calculate_funding_profit(self, current_rate: float, predicted_rate: float, 
                                time_to_funding: int) -> float:
        """Calculate expected profit from funding arbitrage"""
        try:
            # Calculate profit based on rate difference and time
            rate_diff = abs(predicted_rate - current_rate)
            time_factor = min(time_to_funding / 3600, 1.0)  # Normalize to 1 hour
            return rate_diff * time_factor
        except Exception as e:
            self.logger.error(f"Error calculating funding profit: {e}")
            return 0.0
    
    def _calculate_funding_risk(self, symbol: str, current_rate: float, 
                              predicted_rate: float) -> float:
        """Calculate risk score for funding arbitrage"""
        try:
            # Risk based on rate volatility and symbol
            rate_volatility = abs(predicted_rate - current_rate)
            base_risk = 0.1  # Base risk
            
            # Higher risk for volatile rates
            if rate_volatility > 0.001:  # 0.1%
                base_risk += 0.2
            
            return min(base_risk, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating funding risk: {e}")
            return 0.5
    
    def _calculate_funding_confidence(self, symbol: str, current_rate: float) -> float:
        """Calculate confidence in funding prediction"""
        try:
            # Confidence based on historical accuracy
            history = self.funding_history.get(symbol, deque())
            if len(history) < 5:
                return 0.5  # Low confidence with little data
            
            # Calculate prediction accuracy
            recent_rates = list(history)[-5:]
            rate_std = np.std(recent_rates)
            
            # Higher confidence with lower volatility
            confidence = max(0.1, 1.0 - (rate_std * 1000))
            return min(confidence, 0.95)
        except Exception as e:
            self.logger.error(f"Error calculating funding confidence: {e}")
            return 0.5
    
    # Helper methods for liquidation flow analysis
    async def _get_user_state(self) -> Dict[str, Any]:
        """Get current user state"""
        try:
            return await self.api.get_user_state()
        except Exception as e:
            self.logger.error(f"Error getting user state: {e}")
            return {}
    
    def _calculate_backstop_ratio(self, position: Dict[str, Any]) -> float:
        """Calculate backstop ratio for position"""
        try:
            # This would calculate the actual backstop ratio
            # For now, return a mock value
            return 0.8  # 80% backstop ratio
        except Exception as e:
            self.logger.error(f"Error calculating backstop ratio: {e}")
            return 1.0
    
    def _estimate_liquidation_size(self, position: Dict[str, Any]) -> float:
        """Estimate liquidation size"""
        try:
            # This would calculate actual liquidation size
            # For now, return a mock value
            return 1000.0  # $1000 liquidation
        except Exception as e:
            self.logger.error(f"Error estimating liquidation size: {e}")
            return 0.0
    
    def _estimate_liquidation_price(self, symbol: str, position: Dict[str, Any]) -> float:
        """Estimate liquidation price"""
        try:
            # This would calculate actual liquidation price
            # For now, return a mock value
            return 0.5  # Mock price
        except Exception as e:
            self.logger.error(f"Error estimating liquidation price: {e}")
            return 0.0
    
    def _estimate_liquidation_recovery(self, symbol: str, size: float) -> float:
        """Estimate liquidation recovery potential"""
        try:
            # This would calculate actual recovery potential
            # For now, return a mock value
            return 0.02  # 2% recovery
        except Exception as e:
            self.logger.error(f"Error estimating liquidation recovery: {e}")
            return 0.0
    
    # Helper methods for vAMM optimization
    async def _get_oracle_price(self, symbol: str) -> float:
        """Get oracle price for symbol"""
        try:
            # This would get actual oracle price
            # For now, return a mock value
            return 0.5  # Mock oracle price
        except Exception as e:
            self.logger.error(f"Error getting oracle price for {symbol}: {e}")
            return 0.0
    
    async def _calculate_impact_price(self, symbol: str, side: str, size: float) -> float:
        """Calculate impact price for trade"""
        try:
            # This would calculate actual impact price
            # For now, return a mock value
            return 0.5  # Mock impact price
        except Exception as e:
            self.logger.error(f"Error calculating impact price for {symbol}: {e}")
            return 0.0
    
    def _calculate_optimal_vamm_size(self, symbol: str, side: str, 
                                   size: float, premium: float) -> float:
        """Calculate optimal vAMM size"""
        try:
            # Optimize size based on premium and funding rate
            if premium > 0.001:  # 0.1% premium
                return size * 0.8  # Reduce size for high premium
            return size
        except Exception as e:
            self.logger.error(f"Error calculating optimal vAMM size: {e}")
            return size
    
    def _calculate_expected_slippage(self, symbol: str, side: str, size: float) -> float:
        """Calculate expected slippage"""
        try:
            # This would calculate actual slippage
            # For now, return a mock value
            return 0.001  # 0.1% slippage
        except Exception as e:
            self.logger.error(f"Error calculating expected slippage: {e}")
            return 0.0
    
    def _determine_execution_strategy(self, opportunity: vAMMOpportunity) -> str:
        """Determine optimal execution strategy"""
        try:
            if opportunity.premium > 0.002:  # 0.2% premium
                return "aggressive_maker"
            elif opportunity.funding_rate > 0.0005:  # 0.05% funding
                return "funding_arbitrage"
            else:
                return "standard_execution"
        except Exception as e:
            self.logger.error(f"Error determining execution strategy: {e}")
            return "standard_execution"
    
    # Helper methods for maker-taker optimization
    async def _get_vip_tier(self) -> int:
        """Get current VIP tier"""
        try:
            # This would get actual VIP tier
            # For now, return a mock value
            return 0  # Tier 0
        except Exception as e:
            self.logger.error(f"Error getting VIP tier: {e}")
            return 0
    
    async def _get_staking_tier(self) -> str:
        """Get current staking tier"""
        try:
            # This would get actual staking tier
            # For now, return a mock value
            return "wood"  # Wood tier
        except Exception as e:
            self.logger.error(f"Error getting staking tier: {e}")
            return "wood"
    
    def _calculate_maker_rebate(self, vip_tier: int, staking_tier: str) -> float:
        """Calculate maker rebate"""
        try:
            # Base rebate by tier
            tier_rebates = {0: 0.015, 1: 0.012, 2: 0.008, 3: 0.004, 4: 0.000, 5: 0.000, 6: 0.000}
            base_rebate = tier_rebates.get(vip_tier, 0.015)
            
            # Staking discount
            staking_discounts = {"wood": 0.05, "bronze": 0.10, "silver": 0.15, 
                               "gold": 0.20, "platinum": 0.30, "diamond": 0.40}
            discount = staking_discounts.get(staking_tier, 0.0)
            
            return base_rebate * (1 - discount)
        except Exception as e:
            self.logger.error(f"Error calculating maker rebate: {e}")
            return 0.015
    
    def _calculate_taker_fee(self, vip_tier: int, staking_tier: str) -> float:
        """Calculate taker fee"""
        try:
            # Base fee by tier
            tier_fees = {0: 0.045, 1: 0.040, 2: 0.035, 3: 0.030, 4: 0.028, 5: 0.026, 6: 0.024}
            base_fee = tier_fees.get(vip_tier, 0.045)
            
            # Staking discount
            staking_discounts = {"wood": 0.05, "bronze": 0.10, "silver": 0.15, 
                               "gold": 0.20, "platinum": 0.30, "diamond": 0.40}
            discount = staking_discounts.get(staking_tier, 0.0)
            
            return base_fee * (1 - discount)
        except Exception as e:
            self.logger.error(f"Error calculating taker fee: {e}")
            return 0.045
    
    def _determine_optimal_order_type(self, symbol: str, side: str, size: float,
                                    maker_rebate: float, taker_fee: float) -> str:
        """Determine optimal order type"""
        try:
            # Prefer maker if rebate is significant
            if maker_rebate > 0.005:  # 0.5% rebate
                return "limit"
            elif taker_fee < 0.03:  # 3% taker fee
                return "market"
            else:
                return "post_only"
        except Exception as e:
            self.logger.error(f"Error determining optimal order type: {e}")
            return "limit"
    
    def _calculate_optimal_spread(self, symbol: str, maker_rebate: float) -> float:
        """Calculate optimal spread for maker orders"""
        try:
            # Optimal spread based on rebate
            if maker_rebate > 0.01:  # 1% rebate
                return 0.001  # 0.1% spread
            else:
                return 0.002  # 0.2% spread
        except Exception as e:
            self.logger.error(f"Error calculating optimal spread: {e}")
            return 0.002
    
    # Helper methods for HLP vault analysis
    async def _get_hlp_vault_data(self) -> Dict[str, Any]:
        """Get HLP vault data"""
        try:
            # This would get actual HLP vault data
            # For now, return mock data
            return {
                'apy': 0.18,  # 18% APY
                'total_value': 1000000,  # $1M total value
                'user_allocation': 0.0  # User's current allocation
            }
        except Exception as e:
            self.logger.error(f"Error getting HLP vault data: {e}")
            return {}
    
    def _calculate_optimal_hlp_allocation(self, current_apy: float) -> float:
        """Calculate optimal HLP allocation"""
        try:
            if current_apy > 0.20:  # 20% APY
                return 0.30  # 30% allocation
            elif current_apy > 0.15:  # 15% APY
                return 0.20  # 20% allocation
            else:
                return 0.10  # 10% allocation
        except Exception as e:
            self.logger.error(f"Error calculating optimal HLP allocation: {e}")
            return 0.0
    
    def _calculate_hlp_returns(self, allocation: float, apy: float) -> float:
        """Calculate expected HLP returns"""
        try:
            return allocation * apy
        except Exception as e:
            self.logger.error(f"Error calculating HLP returns: {e}")
            return 0.0
    
    def get_architect_metrics(self) -> Dict[str, Any]:
        """Get architect performance metrics"""
        return self.architect_metrics.copy()
    
    def update_funding_history(self, symbol: str, rate: float):
        """Update funding rate history"""
        try:
            if symbol in self.funding_history:
                self.funding_history[symbol].append(rate)
        except Exception as e:
            self.logger.error(f"Error updating funding history: {e}")

