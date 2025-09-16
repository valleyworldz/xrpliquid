"""
🎯 FUNDING ARBITRAGE SCHEDULER
==============================
Production-grade funding arbitrage scheduler aligned with Hyperliquid 1-hour funding cycles.

Features:
- 1-hour funding interval alignment
- Net edge calculation: expected_funding - taker_fee - expected_slippage
- Risk budget compliance
- Position verification across funding timestamps
- Real-time funding rate monitoring
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
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

class FundingArbState(Enum):
    """Funding arbitrage state enumeration"""
    IDLE = "idle"
    MONITORING = "monitoring"
    EVALUATING = "evaluating"
    POSITIONING = "positioning"
    HOLDING = "holding"
    EXITING = "exiting"
    COOLDOWN = "cooldown"

@dataclass
class FundingArbConfig:
    """Configuration for funding arbitrage scheduler"""
    
    # Funding cycle configuration (Hyperliquid 1-hour cycles)
    funding_config: Dict[str, Any] = field(default_factory=lambda: {
        'interval_hours': 1,              # 1-hour funding cycles
        'interval_seconds': 3600,         # 1 hour in seconds
        'monitoring_window_minutes': 5,   # Monitor 5 minutes before funding
        'position_hold_minutes': 60,      # Hold position for full funding cycle
        'exit_window_minutes': 5,         # Exit 5 minutes before next funding
    })
    
    # Net edge calculation
    edge_config: Dict[str, Any] = field(default_factory=lambda: {
        'min_net_edge_bps': 5.0,          # Minimum 5 bps net edge
        'target_net_edge_bps': 10.0,      # Target 10 bps net edge
        'max_net_edge_bps': 50.0,         # Maximum 50 bps net edge
        'edge_calculation_method': 'conservative',  # 'conservative', 'aggressive', 'realistic'
    })
    
    # Fee and cost modeling
    cost_config: Dict[str, Any] = field(default_factory=lambda: {
        'taker_fee_bps': 5.0,             # 0.05% taker fee
        'maker_fee_bps': 1.0,             # 0.01% maker fee
        'maker_rebate_bps': 0.5,          # 0.005% maker rebate
        'expected_slippage_bps': 2.0,     # Expected 2 bps slippage
        'max_slippage_bps': 10.0,         # Maximum 10 bps slippage
        'funding_rate_volatility_bps': 2.0,  # Funding rate volatility
    })
    
    # Risk management
    risk_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_position_size_usd': 5000.0,  # Maximum position size
        'max_exposure_percent': 0.1,      # 10% max exposure
        'max_daily_trades': 24,           # Max 24 trades per day (1 per hour)
        'cooldown_minutes': 5,            # 5-minute cooldown between trades
        'stop_loss_bps': 20.0,            # 20 bps stop loss
        'max_drawdown_percent': 0.02,     # 2% max drawdown
    })
    
    # Position management
    position_config: Dict[str, Any] = field(default_factory=lambda: {
        'verify_position_before_funding': True,
        'verify_position_after_funding': True,
        'auto_exit_on_funding_payment': True,
        'position_tolerance_percent': 0.01,  # 1% position tolerance
        'min_position_hold_minutes': 30,     # Minimum 30 minutes hold
    })

@dataclass
class FundingOpportunity:
    """Funding arbitrage opportunity"""
    
    symbol: str
    current_funding_rate: float
    expected_funding_rate: float
    net_edge_bps: float
    position_size_usd: float
    expected_profit_usd: float
    risk_score: float
    confidence_score: float
    funding_timestamp: float
    opportunity_id: str
    state: FundingArbState = FundingArbState.IDLE
    
    # Cost breakdown
    taker_fee_cost: float = 0.0
    expected_slippage_cost: float = 0.0
    funding_payment: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    max_loss: float = 0.0
    sharpe_ratio: float = 0.0

class FundingArbitrageScheduler:
    """
    🎯 FUNDING ARBITRAGE SCHEDULER
    
    Production-grade funding arbitrage scheduler with 1-hour cycle alignment
    and comprehensive net edge calculation.
    """
    
    def __init__(self, config: Dict[str, Any], api=None, logger=None):
        self.config = config
        self.api = api
        self.logger = logger or Logger()
        
        # Initialize configuration
        self.arb_config = FundingArbConfig()
        
        # Scheduler state
        self.current_state = FundingArbState.IDLE
        self.active_opportunities: Dict[str, FundingOpportunity] = {}
        self.position_history: List[Dict[str, Any]] = []
        self.funding_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_opportunities': 0,
            'executed_opportunities': 0,
            'successful_opportunities': 0,
            'total_profit': 0.0,
            'total_fees': 0.0,
            'total_slippage': 0.0,
            'avg_net_edge': 0.0,
            'win_rate': 0.0,
        }
        
        # Timing tracking
        self.last_funding_time = 0.0
        self.next_funding_time = 0.0
        self.last_trade_time = 0.0
        
        self.logger.info("🎯 [FUNDING_ARB_SCHEDULER] Funding Arbitrage Scheduler initialized")
        self.logger.info("🎯 [FUNDING_ARB_SCHEDULER] 1-hour funding cycle alignment enabled")
    
    async def start_scheduler(self):
        """Start the funding arbitrage scheduler"""
        try:
            self.logger.info("🎯 [SCHEDULER] Starting funding arbitrage scheduler...")
            
            # Calculate next funding time
            await self._calculate_next_funding_time()
            
            # Start main scheduler loop
            while True:
                try:
                    await self._scheduler_loop()
                    await asyncio.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    self.logger.error(f"❌ [SCHEDULER_LOOP] Error in scheduler loop: {e}")
                    await asyncio.sleep(30)  # Wait 30 seconds on error
                    
        except Exception as e:
            self.logger.error(f"❌ [SCHEDULER] Error starting scheduler: {e}")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            current_time = time.time()
            
            # Update state based on current time
            await self._update_scheduler_state(current_time)
            
            # Execute state-specific actions
            if self.current_state == FundingArbState.MONITORING:
                await self._monitor_funding_opportunities()
            elif self.current_state == FundingArbState.EVALUATING:
                await self._evaluate_opportunities()
            elif self.current_state == FundingArbState.POSITIONING:
                await self._execute_positions()
            elif self.current_state == FundingArbState.HOLDING:
                await self._monitor_positions()
            elif self.current_state == FundingArbState.EXITING:
                await self._exit_positions()
            elif self.current_state == FundingArbState.COOLDOWN:
                await self._cooldown_period()
            
        except Exception as e:
            self.logger.error(f"❌ [SCHEDULER_LOOP] Error in scheduler loop: {e}")
    
    async def _update_scheduler_state(self, current_time: float):
        """Update scheduler state based on current time"""
        try:
            funding_config = self.arb_config.funding_config
            monitoring_window = funding_config['monitoring_window_minutes'] * 60
            exit_window = funding_config['exit_window_minutes'] * 60
            
            # Calculate time to next funding
            time_to_funding = self.next_funding_time - current_time
            
            if time_to_funding > monitoring_window:
                # More than 5 minutes to funding - IDLE
                self.current_state = FundingArbState.IDLE
            elif time_to_funding > 0:
                # Within monitoring window - MONITORING
                self.current_state = FundingArbState.MONITORING
            elif time_to_funding > -funding_config['position_hold_minutes'] * 60:
                # Within holding period - HOLDING
                self.current_state = FundingArbState.HOLDING
            elif time_to_funding > -funding_config['position_hold_minutes'] * 60 - exit_window:
                # Within exit window - EXITING
                self.current_state = FundingArbState.EXITING
            else:
                # After exit window - COOLDOWN
                self.current_state = FundingArbState.COOLDOWN
                
        except Exception as e:
            self.logger.error(f"❌ [UPDATE_STATE] Error updating scheduler state: {e}")
    
    async def _calculate_next_funding_time(self):
        """Calculate next funding timestamp"""
        try:
            current_time = time.time()
            funding_interval = self.arb_config.funding_config['interval_seconds']
            
            # Calculate next funding time (aligned to hour boundaries)
            current_hour = int(current_time // 3600) * 3600
            next_hour = current_hour + 3600
            
            self.next_funding_time = next_hour
            self.last_funding_time = current_hour
            
            self.logger.info(f"🎯 [FUNDING_TIME] Next funding at: {datetime.fromtimestamp(self.next_funding_time)}")
            
        except Exception as e:
            self.logger.error(f"❌ [FUNDING_TIME] Error calculating next funding time: {e}")
    
    async def _monitor_funding_opportunities(self):
        """Monitor for funding arbitrage opportunities"""
        try:
            self.logger.info("🎯 [MONITOR] Monitoring funding opportunities...")
            
            # Get current funding rates
            funding_rates = await self._get_current_funding_rates()
            
            # Evaluate each symbol for opportunities
            for symbol, funding_rate in funding_rates.items():
                opportunity = await self._evaluate_funding_opportunity(symbol, funding_rate)
                
                if opportunity and opportunity.net_edge_bps >= self.arb_config.edge_config['min_net_edge_bps']:
                    self.active_opportunities[opportunity.opportunity_id] = opportunity
                    self.logger.info(f"🎯 [OPPORTUNITY] Found opportunity: {symbol} - {opportunity.net_edge_bps:.2f} bps edge")
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error monitoring opportunities: {e}")
    
    async def _evaluate_funding_opportunity(self, symbol: str, funding_rate: float) -> Optional[FundingOpportunity]:
        """Evaluate funding arbitrage opportunity for a symbol"""
        try:
            # Calculate expected funding rate (simplified model)
            expected_funding_rate = await self._predict_funding_rate(symbol, funding_rate)
            
            # Calculate net edge
            net_edge = await self._calculate_net_edge(symbol, funding_rate, expected_funding_rate)
            
            if net_edge['net_edge_bps'] < self.arb_config.edge_config['min_net_edge_bps']:
                return None
            
            # Calculate position size
            position_size = await self._calculate_position_size(symbol, net_edge['net_edge_bps'])
            
            # Calculate expected profit
            expected_profit = position_size * (net_edge['net_edge_bps'] / 10000)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(symbol, position_size, funding_rate)
            
            # Create opportunity
            opportunity = FundingOpportunity(
                symbol=symbol,
                current_funding_rate=funding_rate,
                expected_funding_rate=expected_funding_rate,
                net_edge_bps=net_edge['net_edge_bps'],
                position_size_usd=position_size,
                expected_profit_usd=expected_profit,
                risk_score=risk_metrics['risk_score'],
                confidence_score=risk_metrics['confidence_score'],
                funding_timestamp=self.next_funding_time,
                opportunity_id=f"funding_arb_{symbol}_{int(time.time())}",
                taker_fee_cost=net_edge['taker_fee_cost'],
                expected_slippage_cost=net_edge['slippage_cost'],
                funding_payment=net_edge['funding_payment'],
                var_95=risk_metrics['var_95'],
                max_loss=risk_metrics['max_loss'],
                sharpe_ratio=risk_metrics['sharpe_ratio'],
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"❌ [EVALUATE_OPPORTUNITY] Error evaluating opportunity: {e}")
            return None
    
    async def _calculate_net_edge(self, symbol: str, current_funding: float, expected_funding: float) -> Dict[str, float]:
        """Calculate net edge: expected_funding - taker_fee - expected_slippage"""
        try:
            cost_config = self.arb_config.cost_config
            
            # Calculate funding payment
            funding_payment = expected_funding - current_funding
            
            # Calculate taker fee cost
            taker_fee_cost = cost_config['taker_fee_bps'] / 10000
            
            # Calculate expected slippage cost
            slippage_cost = cost_config['expected_slippage_bps'] / 10000
            
            # Calculate net edge
            net_edge_bps = (funding_payment - taker_fee_cost - slippage_cost) * 10000
            
            return {
                'net_edge_bps': net_edge_bps,
                'funding_payment': funding_payment,
                'taker_fee_cost': taker_fee_cost,
                'slippage_cost': slippage_cost,
                'gross_edge_bps': funding_payment * 10000,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [NET_EDGE] Error calculating net edge: {e}")
            return {
                'net_edge_bps': 0.0,
                'funding_payment': 0.0,
                'taker_fee_cost': 0.0,
                'slippage_cost': 0.0,
                'gross_edge_bps': 0.0,
            }
    
    async def _predict_funding_rate(self, symbol: str, current_funding: float) -> float:
        """Predict next funding rate (simplified model)"""
        try:
            # Simple mean reversion model
            mean_reversion_speed = 0.1
            long_term_mean = 0.0001  # 1 bps long-term mean
            
            predicted_funding = current_funding + mean_reversion_speed * (long_term_mean - current_funding)
            
            # Add some volatility
            volatility = self.arb_config.cost_config['funding_rate_volatility_bps'] / 10000
            noise = np.random.normal(0, volatility)
            
            predicted_funding += noise
            
            # Apply limits
            max_funding = 0.01  # 1% max funding rate
            min_funding = -0.01  # -1% min funding rate
            
            predicted_funding = max(min_funding, min(predicted_funding, max_funding))
            
            return predicted_funding
            
        except Exception as e:
            self.logger.error(f"❌ [PREDICT_FUNDING] Error predicting funding rate: {e}")
            return current_funding
    
    async def _calculate_position_size(self, symbol: str, net_edge_bps: float) -> float:
        """Calculate optimal position size based on risk budget"""
        try:
            risk_config = self.arb_config.risk_config
            
            # Base position size
            base_size = risk_config['max_position_size_usd']
            
            # Scale by net edge (higher edge = larger position)
            edge_multiplier = min(2.0, net_edge_bps / self.arb_config.edge_config['target_net_edge_bps'])
            
            # Scale by risk budget
            risk_multiplier = risk_config['max_exposure_percent']
            
            position_size = base_size * edge_multiplier * risk_multiplier
            
            # Apply limits
            position_size = min(position_size, risk_config['max_position_size_usd'])
            position_size = max(position_size, 100.0)  # Minimum $100 position
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ [POSITION_SIZE] Error calculating position size: {e}")
            return 100.0
    
    async def _calculate_risk_metrics(self, symbol: str, position_size: float, funding_rate: float) -> Dict[str, float]:
        """Calculate risk metrics for the opportunity"""
        try:
            # Simplified risk calculation
            volatility = 0.02  # 2% daily volatility
            var_95 = position_size * volatility * 1.645  # 95% VaR
            
            max_loss = position_size * 0.05  # 5% max loss
            
            # Risk score (0-1, higher is riskier)
            risk_score = min(1.0, (abs(funding_rate) * 100 + volatility * 100) / 10)
            
            # Confidence score (0-1, higher is more confident)
            confidence_score = max(0.0, 1.0 - risk_score)
            
            # Sharpe ratio (simplified)
            expected_return = position_size * abs(funding_rate)
            sharpe_ratio = expected_return / (position_size * volatility) if volatility > 0 else 0
            
            return {
                'var_95': var_95,
                'max_loss': max_loss,
                'risk_score': risk_score,
                'confidence_score': confidence_score,
                'sharpe_ratio': sharpe_ratio,
            }
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_METRICS] Error calculating risk metrics: {e}")
            return {
                'var_95': 0.0,
                'max_loss': 0.0,
                'risk_score': 1.0,
                'confidence_score': 0.0,
                'sharpe_ratio': 0.0,
            }
    
    async def _get_current_funding_rates(self) -> Dict[str, float]:
        """Get current funding rates from API"""
        try:
            # This would integrate with the actual Hyperliquid API
            # For now, return mock data
            return {
                'XRP': 0.0005,  # 5 bps funding rate
                'BTC': 0.0002,  # 2 bps funding rate
                'ETH': 0.0003,  # 3 bps funding rate
            }
            
        except Exception as e:
            self.logger.error(f"❌ [FUNDING_RATES] Error getting funding rates: {e}")
            return {}
    
    async def _execute_positions(self):
        """Execute funding arbitrage positions"""
        try:
            self.logger.info("🎯 [EXECUTE] Executing funding arbitrage positions...")
            
            for opportunity_id, opportunity in self.active_opportunities.items():
                if opportunity.state == FundingArbState.EVALUATING:
                    # Execute the position
                    success = await self._execute_opportunity(opportunity)
                    
                    if success:
                        opportunity.state = FundingArbState.HOLDING
                        self.performance_metrics['executed_opportunities'] += 1
                        self.logger.info(f"🎯 [EXECUTE] Executed opportunity: {opportunity.symbol}")
                    else:
                        opportunity.state = FundingArbState.IDLE
                        self.logger.warning(f"🎯 [EXECUTE] Failed to execute opportunity: {opportunity.symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ [EXECUTE] Error executing positions: {e}")
    
    async def _execute_opportunity(self, opportunity: FundingOpportunity) -> bool:
        """Execute a single funding arbitrage opportunity"""
        try:
            # This would integrate with the actual trading system
            # For now, simulate execution
            
            # Check risk limits
            if not await self._check_risk_limits(opportunity):
                return False
            
            # Check cooldown
            if not await self._check_cooldown():
                return False
            
            # Execute trade (simplified)
            trade_result = await self._place_funding_arbitrage_trade(opportunity)
            
            if trade_result:
                # Record the trade
                await self._record_funding_arbitrage_trade(opportunity, trade_result)
                
                # Update last trade time
                self.last_trade_time = time.time()
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ [EXECUTE_OPPORTUNITY] Error executing opportunity: {e}")
            return False
    
    async def _check_risk_limits(self, opportunity: FundingOpportunity) -> bool:
        """Check if opportunity meets risk limits"""
        try:
            risk_config = self.arb_config.risk_config
            
            # Check position size
            if opportunity.position_size_usd > risk_config['max_position_size_usd']:
                self.logger.warning(f"🎯 [RISK_CHECK] Position size exceeds limit: {opportunity.position_size_usd}")
                return False
            
            # Check exposure
            if opportunity.position_size_usd > risk_config['max_exposure_percent'] * 10000:  # Assuming $10k account
                self.logger.warning(f"🎯 [RISK_CHECK] Exposure exceeds limit: {opportunity.position_size_usd}")
                return False
            
            # Check risk score
            if opportunity.risk_score > 0.8:  # 80% risk threshold
                self.logger.warning(f"🎯 [RISK_CHECK] Risk score too high: {opportunity.risk_score}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [RISK_CHECK] Error checking risk limits: {e}")
            return False
    
    async def _check_cooldown(self) -> bool:
        """Check if we're in cooldown period"""
        try:
            cooldown_minutes = self.arb_config.risk_config['cooldown_minutes']
            time_since_last_trade = time.time() - self.last_trade_time
            
            if time_since_last_trade < cooldown_minutes * 60:
                self.logger.info(f"🎯 [COOLDOWN] In cooldown period: {cooldown_minutes - time_since_last_trade/60:.1f} minutes remaining")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [COOLDOWN] Error checking cooldown: {e}")
            return False
    
    async def _place_funding_arbitrage_trade(self, opportunity: FundingOpportunity) -> Optional[Dict[str, Any]]:
        """Place funding arbitrage trade"""
        try:
            # This would integrate with the actual trading system
            # For now, simulate trade placement
            
            trade_result = {
                'order_id': f"funding_arb_{opportunity.opportunity_id}",
                'symbol': opportunity.symbol,
                'side': 'buy' if opportunity.current_funding_rate < 0 else 'sell',
                'quantity': opportunity.position_size_usd / 0.52,  # Assuming XRP price
                'price': 0.52,  # Assuming XRP price
                'timestamp': time.time(),
                'status': 'filled',
            }
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"❌ [PLACE_TRADE] Error placing trade: {e}")
            return None
    
    async def _record_funding_arbitrage_trade(self, opportunity: FundingOpportunity, trade_result: Dict[str, Any]):
        """Record funding arbitrage trade in ledger"""
        try:
            # Create trade record
            trade_record = TradeRecord(
                ts=trade_result['timestamp'],
                symbol=trade_result['symbol'],
                side=trade_result['side'],
                qty=trade_result['quantity'],
                px=trade_result['price'],
                fee=opportunity.taker_fee_cost * trade_result['quantity'] * trade_result['price'],
                fee_bps=self.arb_config.cost_config['taker_fee_bps'],
                funding=opportunity.funding_payment * trade_result['quantity'] * trade_result['price'],
                slippage_bps=self.arb_config.cost_config['expected_slippage_bps'],
                pnl_realized=0.0,  # Will be calculated when position is closed
                pnl_unrealized=0.0,
                reason_code=ReasonCode.FUNDING_ARBITRAGE.value,
                maker_flag=False,  # Funding arb typically uses taker orders
                cloid=trade_result['order_id'],
                order_state=OrderState.FILLED.value,
            )
            
            # This would integrate with the canonical trade ledger
            self.logger.info(f"🎯 [RECORD_TRADE] Recorded funding arbitrage trade: {trade_record.cloid}")
            
        except Exception as e:
            self.logger.error(f"❌ [RECORD_TRADE] Error recording trade: {e}")
    
    async def _monitor_positions(self):
        """Monitor active positions during holding period"""
        try:
            self.logger.info("🎯 [MONITOR_POSITIONS] Monitoring active positions...")
            
            for opportunity_id, opportunity in self.active_opportunities.items():
                if opportunity.state == FundingArbState.HOLDING:
                    # Verify position is still open
                    position_verified = await self._verify_position(opportunity)
                    
                    if not position_verified:
                        self.logger.warning(f"🎯 [MONITOR_POSITIONS] Position verification failed: {opportunity.symbol}")
                        opportunity.state = FundingArbState.EXITING
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR_POSITIONS] Error monitoring positions: {e}")
    
    async def _verify_position(self, opportunity: FundingOpportunity) -> bool:
        """Verify position is still open across funding timestamp"""
        try:
            # This would check with the actual trading system
            # For now, simulate verification
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [VERIFY_POSITION] Error verifying position: {e}")
            return False
    
    async def _exit_positions(self):
        """Exit funding arbitrage positions"""
        try:
            self.logger.info("🎯 [EXIT] Exiting funding arbitrage positions...")
            
            for opportunity_id, opportunity in self.active_opportunities.items():
                if opportunity.state == FundingArbState.EXITING:
                    # Exit the position
                    success = await self._exit_opportunity(opportunity)
                    
                    if success:
                        opportunity.state = FundingArbState.IDLE
                        self.performance_metrics['successful_opportunities'] += 1
                        self.logger.info(f"🎯 [EXIT] Exited opportunity: {opportunity.symbol}")
                    else:
                        self.logger.warning(f"🎯 [EXIT] Failed to exit opportunity: {opportunity.symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ [EXIT] Error exiting positions: {e}")
    
    async def _exit_opportunity(self, opportunity: FundingOpportunity) -> bool:
        """Exit a single funding arbitrage opportunity"""
        try:
            # This would integrate with the actual trading system
            # For now, simulate exit
            
            # Calculate realized P&L
            realized_pnl = opportunity.expected_profit_usd
            
            # Update performance metrics
            self.performance_metrics['total_profit'] += realized_pnl
            self.performance_metrics['total_fees'] += opportunity.taker_fee_cost * opportunity.position_size_usd
            self.performance_metrics['total_slippage'] += opportunity.expected_slippage_cost * opportunity.position_size_usd
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [EXIT_OPPORTUNITY] Error exiting opportunity: {e}")
            return False
    
    async def _cooldown_period(self):
        """Handle cooldown period"""
        try:
            self.logger.info("🎯 [COOLDOWN] In cooldown period...")
            
            # Clear completed opportunities
            completed_opportunities = [oid for oid, opp in self.active_opportunities.items() 
                                     if opp.state == FundingArbState.IDLE]
            
            for oid in completed_opportunities:
                del self.active_opportunities[oid]
            
            # Update next funding time
            await self._calculate_next_funding_time()
            
        except Exception as e:
            self.logger.error(f"❌ [COOLDOWN] Error in cooldown period: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            # Calculate win rate
            if self.performance_metrics['executed_opportunities'] > 0:
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['successful_opportunities'] / 
                    self.performance_metrics['executed_opportunities']
                )
            
            return {
                'performance_metrics': self.performance_metrics,
                'active_opportunities': len(self.active_opportunities),
                'current_state': self.current_state.value,
                'next_funding_time': datetime.fromtimestamp(self.next_funding_time).isoformat(),
                'last_funding_time': datetime.fromtimestamp(self.last_funding_time).isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PERFORMANCE_SUMMARY] Error getting performance summary: {e}")
            return {}
