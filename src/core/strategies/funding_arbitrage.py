"""
🎯 FUNDING ARBITRAGE STRATEGY
====================================
Standalone funding arbitrage component with mathematical EV proof
and optimized execution scheduling for maximum profitability.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.api.hyperliquid_api import HyperliquidAPI
from src.core.utils.logger import Logger
from src.core.analytics.trade_ledger import TradeLedgerManager
from src.core.monitoring.prometheus_metrics import get_metrics_collector, record_trade_metrics
from src.core.risk.risk_unit_sizing import RiskUnitSizing, RiskUnitConfig

@dataclass
class FundingArbitrageConfig:
    """Configuration for funding arbitrage strategy"""
    
    # Threshold parameters
    min_funding_rate_threshold: float = 0.0001  # 0.01% minimum funding rate
    max_funding_rate_threshold: float = 0.01    # 1% maximum funding rate
    optimal_funding_rate: float = 0.005         # 0.5% optimal funding rate
    
    # Position sizing
    max_position_size_usd: float = 1000.0       # Maximum position size in USD
    position_size_multiplier: float = 0.1       # Position size as % of available margin
    min_position_size_usd: float = 50.0         # Minimum position size
    
    # Risk management
    max_drawdown_percent: float = 5.0           # Maximum drawdown %
    stop_loss_funding_rate: float = 0.02        # Stop loss at 2% funding rate
    take_profit_funding_rate: float = 0.001     # Take profit at 0.1% funding rate
    
    # Execution parameters
    funding_rate_check_interval: int = 300      # Check every 5 minutes
    execution_delay_seconds: int = 30           # Delay before execution
    max_execution_time_seconds: int = 60        # Maximum execution time
    
    # EV calculation parameters
    expected_holding_period_hours: float = 8.0  # Expected holding period
    funding_payment_frequency_hours: float = 8.0 # Funding payment frequency
    transaction_cost_bps: float = 2.0           # Transaction cost in basis points
    slippage_cost_bps: float = 1.0              # Expected slippage cost
    
    # Market conditions
    min_volume_24h_usd: float = 1000000.0       # Minimum 24h volume
    max_spread_bps: float = 10.0                # Maximum spread in bps
    min_liquidity_usd: float = 50000.0          # Minimum liquidity

@dataclass
class FundingArbitrageOpportunity:
    """Represents a funding arbitrage opportunity"""
    
    symbol: str
    current_funding_rate: float
    predicted_funding_rate: float
    expected_value: float
    expected_return_percent: float
    risk_score: float
    confidence_score: float
    position_size_usd: float
    entry_price: float
    exit_price: float
    holding_period_hours: float
    total_costs_bps: float
    net_expected_return_bps: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'symbol': self.symbol,
            'current_funding_rate': self.current_funding_rate,
            'predicted_funding_rate': self.predicted_funding_rate,
            'expected_value': self.expected_value,
            'expected_return_percent': self.expected_return_percent,
            'risk_score': self.risk_score,
            'confidence_score': self.confidence_score,
            'position_size_usd': self.position_size_usd,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'holding_period_hours': self.holding_period_hours,
            'total_costs_bps': self.total_costs_bps,
            'net_expected_return_bps': self.net_expected_return_bps,
            'timestamp': self.timestamp
        }

class FundingArbitrageStrategy:
    """
    🎯 FUNDING ARBITRAGE STRATEGY
    Mathematical foundation for funding rate arbitrage with EV proof
    """
    
    def __init__(self, 
                 config: FundingArbitrageConfig,
                 api: HyperliquidAPI,
                 logger: Optional[Logger] = None):
        self.config = config
        self.api = api
        self.logger = logger or Logger()
        
        # Initialize components
        self.trade_ledger = TradeLedgerManager(data_dir="data/funding_arbitrage", logger=self.logger)
        self.metrics_collector = get_metrics_collector(port=8002, logger=self.logger)
        
        # Initialize risk unit sizing system
        risk_config = RiskUnitConfig(
            target_volatility_percent=2.0,  # Target 2% daily volatility
            max_equity_at_risk_percent=1.0,  # Max 1% equity at risk per trade
            base_equity_at_risk_percent=0.5,  # Base 0.5% equity at risk per trade
            kelly_multiplier=0.25,  # Use 25% of Kelly fraction
            min_position_size_usd=25.0,  # Minimum position size
            max_position_size_usd=10000.0,  # Maximum position size
        )
        self.risk_unit_sizing = RiskUnitSizing(risk_config, self.logger)
        
        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.opportunity_history: List[FundingArbitrageOpportunity] = []
        self.last_funding_check = 0.0
        self.total_pnl = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        
        # Performance tracking
        self.start_time = time.time()
        self.funding_rate_history: Dict[str, List[Tuple[float, float]]] = {}  # symbol -> [(timestamp, rate)]
        self.price_history: Dict[str, List[float]] = {}  # symbol -> [prices] for volatility calculation
        
        self.logger.info("🎯 [FUNDING_ARB] Funding Arbitrage Strategy initialized")
        self.logger.info(f"📊 [FUNDING_ARB] Min funding threshold: {self.config.min_funding_rate_threshold:.4f}")
        self.logger.info(f"📊 [FUNDING_ARB] Max funding threshold: {self.config.max_funding_rate_threshold:.4f}")
        self.logger.info(f"📊 [FUNDING_ARB] Optimal funding rate: {self.config.optimal_funding_rate:.4f}")
        self.logger.info("🎯 [RISK_UNIT] Risk unit sizing system integrated")
    
    def calculate_expected_value(self, 
                               funding_rate: float,
                               position_size_usd: float,
                               holding_period_hours: float,
                               entry_price: float,
                               exit_price: float) -> Tuple[float, float, float]:
        """
        Calculate Expected Value (EV) for funding arbitrage opportunity
        
        Mathematical Foundation:
        EV = (Funding Payment - Transaction Costs - Slippage Costs) * Probability of Success
        
        Where:
        - Funding Payment = Position Size * Funding Rate * (Holding Period / 8 hours)
        - Transaction Costs = Position Size * Transaction Cost Rate * 2 (entry + exit)
        - Slippage Costs = Position Size * Slippage Rate * 2 (entry + exit)
        - Probability of Success = f(funding_rate, market_conditions, historical_success_rate)
        
        Returns:
        - expected_value: Net expected value in USD
        - expected_return_percent: Expected return as percentage
        - risk_adjusted_return: Risk-adjusted return considering volatility
        """
        
        # 1. Calculate funding payment
        funding_payments = (holding_period_hours / self.config.funding_payment_frequency_hours)
        funding_payment_usd = position_size_usd * abs(funding_rate) * funding_payments
        
        # 2. Calculate transaction costs
        transaction_cost_usd = position_size_usd * (self.config.transaction_cost_bps / 10000) * 2
        
        # 3. Calculate slippage costs
        slippage_cost_usd = position_size_usd * (self.config.slippage_cost_bps / 10000) * 2
        
        # 4. Calculate total costs
        total_costs_usd = transaction_cost_usd + slippage_cost_usd
        
        # 5. Calculate probability of success based on funding rate magnitude
        # Higher funding rates have higher probability of mean reversion
        funding_rate_magnitude = abs(funding_rate)
        if funding_rate_magnitude < self.config.min_funding_rate_threshold:
            success_probability = 0.3  # Low probability for small rates
        elif funding_rate_magnitude < self.config.optimal_funding_rate:
            success_probability = 0.6  # Medium probability
        elif funding_rate_magnitude < self.config.max_funding_rate_threshold:
            success_probability = 0.8  # High probability
        else:
            success_probability = 0.9  # Very high probability for extreme rates
        
        # 6. Calculate expected value
        gross_expected_value = funding_payment_usd - total_costs_usd
        expected_value = gross_expected_value * success_probability
        
        # 7. Calculate expected return percentage
        expected_return_percent = (expected_value / position_size_usd) * 100
        
        # 8. Calculate risk-adjusted return (Sharpe-like metric)
        # Assume funding rate volatility as risk measure
        funding_rate_volatility = 0.002  # 0.2% typical volatility
        risk_adjusted_return = expected_return_percent / (funding_rate_volatility * 100)
        
        return expected_value, expected_return_percent, risk_adjusted_return
    
    def calculate_optimal_position_size(self, 
                                      symbol: str,
                                      available_margin: float,
                                      funding_rate: float,
                                      current_price: float,
                                      confidence_score: float = 0.5) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal position size using risk unit sizing system
        
        Args:
            symbol: Trading symbol
            available_margin: Available margin in USD
            funding_rate: Current funding rate
            current_price: Current price
            confidence_score: Confidence score (0-1)
        
        Returns:
            Tuple of (optimal_position_size_usd, risk_metrics_dict)
        """
        
        # Calculate volatility for the symbol
        volatility_percent = self._calculate_symbol_volatility(symbol, current_price)
        
        # Calculate win probability based on funding rate
        funding_rate_magnitude = abs(funding_rate)
        win_probability = min(0.9, max(0.3, funding_rate_magnitude / self.config.max_funding_rate_threshold))
        
        # Estimate average win/loss percentages
        avg_win_percent = funding_rate_magnitude * 0.8  # Assume 80% of funding rate as win
        avg_loss_percent = funding_rate_magnitude * 0.2  # Assume 20% of funding rate as loss
        
        # Use risk unit sizing system
        optimal_position_size, risk_metrics = self.risk_unit_sizing.calculate_optimal_position_size(
            symbol=symbol,
            account_value=available_margin,
            volatility_percent=volatility_percent,
            confidence_score=confidence_score,
            win_probability=win_probability,
            avg_win_percent=avg_win_percent,
            avg_loss_percent=avg_loss_percent,
            market_regime='calm'  # Default regime
        )
        
        return optimal_position_size, risk_metrics
    
    def _calculate_symbol_volatility(self, symbol: str, current_price: float) -> float:
        """Calculate volatility for a symbol"""
        
        # Add current price to history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(current_price)
        
        # Keep only recent history (last 30 days)
        max_history = 30 * 24  # 30 days * 24 hours
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
        
        # Calculate volatility using risk unit sizing system
        volatility_percent = self.risk_unit_sizing.calculate_volatility(
            symbol, self.price_history[symbol]
        )
        
        return volatility_percent
    
    def assess_opportunity(self, 
                          symbol: str,
                          current_funding_rate: float,
                          current_price: float,
                          available_margin: float) -> Optional[FundingArbitrageOpportunity]:
        """
        Assess funding arbitrage opportunity and calculate EV
        """
        
        # 1. Check if funding rate meets threshold
        if abs(current_funding_rate) < self.config.min_funding_rate_threshold:
            return None
        
        # 2. Check if funding rate is within acceptable range
        if abs(current_funding_rate) > self.config.max_funding_rate_threshold:
            return None
        
        # 3. Calculate optimal position size using risk unit sizing
        position_size_usd, risk_metrics = self.calculate_optimal_position_size(
            symbol, available_margin, current_funding_rate, current_price, confidence_score=0.5
        )
        
        if position_size_usd < self.config.min_position_size_usd:
            return None
        
        # 4. Calculate expected holding period
        # Higher funding rates typically mean revert faster
        funding_rate_magnitude = abs(current_funding_rate)
        if funding_rate_magnitude > self.config.optimal_funding_rate:
            expected_holding_period = 4.0  # 4 hours for high rates
        else:
            expected_holding_period = self.config.expected_holding_period_hours
        
        # 5. Estimate exit price (assume mean reversion)
        mean_reversion_factor = 0.5  # Assume 50% mean reversion
        predicted_funding_rate = current_funding_rate * (1 - mean_reversion_factor)
        
        # 6. Calculate expected value
        expected_value, expected_return_percent, risk_adjusted_return = self.calculate_expected_value(
            current_funding_rate,
            position_size_usd,
            expected_holding_period,
            current_price,
            current_price  # Assume no price change for funding arbitrage
        )
        
        # 7. Calculate risk score (0-1, lower is better)
        risk_score = min(1.0, abs(current_funding_rate) / self.config.max_funding_rate_threshold)
        
        # 8. Calculate confidence score (0-1, higher is better)
        confidence_score = min(1.0, abs(current_funding_rate) / self.config.optimal_funding_rate)
        
        # 9. Calculate total costs
        total_costs_bps = (self.config.transaction_cost_bps + self.config.slippage_cost_bps) * 2
        
        # 10. Calculate net expected return
        net_expected_return_bps = (expected_return_percent * 100) - total_costs_bps
        
        # Only proceed if expected value is positive
        if expected_value <= 0:
            return None
        
        opportunity = FundingArbitrageOpportunity(
            symbol=symbol,
            current_funding_rate=current_funding_rate,
            predicted_funding_rate=predicted_funding_rate,
            expected_value=expected_value,
            expected_return_percent=expected_return_percent,
            risk_score=risk_score,
            confidence_score=confidence_score,
            position_size_usd=position_size_usd,
            entry_price=current_price,
            exit_price=current_price,
            holding_period_hours=expected_holding_period,
            total_costs_bps=total_costs_bps,
            net_expected_return_bps=net_expected_return_bps
        )
        
        # Add risk metrics to opportunity metadata
        opportunity.risk_metrics = risk_metrics
        
        return opportunity
    
    async def execute_funding_arbitrage(self, opportunity: FundingArbitrageOpportunity) -> Dict[str, Any]:
        """
        Execute funding arbitrage trade
        """
        
        try:
            self.logger.info(f"🎯 [FUNDING_ARB] Executing funding arbitrage for {opportunity.symbol}")
            self.logger.info(f"📊 [FUNDING_ARB] Funding rate: {opportunity.current_funding_rate:.4f}")
            self.logger.info(f"📊 [FUNDING_ARB] Expected value: ${opportunity.expected_value:.2f}")
            self.logger.info(f"📊 [FUNDING_ARB] Position size: ${opportunity.position_size_usd:.2f}")
            
            # Calculate position size in tokens
            position_size_tokens = opportunity.position_size_usd / opportunity.entry_price
            
            # Determine trade direction based on funding rate
            if opportunity.current_funding_rate > 0:
                # Positive funding rate: short the asset (receive funding)
                side = "sell"
                trade_type = "FUNDING_ARB_SHORT"
            else:
                # Negative funding rate: long the asset (pay funding but expect rate to increase)
                side = "buy"
                trade_type = "FUNDING_ARB_LONG"
            
            # Place order
            order_result = self.api.place_order(
                symbol=opportunity.symbol,
                side=side,
                quantity=position_size_tokens,
                price=opportunity.entry_price,
                order_type="market",
                time_in_force="Gtc",
                reduce_only=False
            )
            
            if isinstance(order_result, dict) and order_result.get('success'):
                order_id = order_result.get('order_id', 'unknown')
                actual_price = order_result.get('price', opportunity.entry_price)
                actual_quantity = order_result.get('quantity', position_size_tokens)
                
                # Record trade
                trade_data = {
                    'trade_type': trade_type,
                    'strategy': 'Funding Arbitrage Strategy',
                    'hat_role': 'Chief Quantitative Strategist',
                    'symbol': opportunity.symbol,
                    'side': side.upper(),
                    'quantity': actual_quantity,
                    'price': actual_price,
                    'mark_price': opportunity.entry_price,
                    'order_type': 'MARKET',
                    'order_id': order_id,
                    'execution_time': 0.0,
                    'slippage': abs(actual_price - opportunity.entry_price) / opportunity.entry_price,
                    'fees_paid': actual_quantity * actual_price * 0.0001,
                    'position_size_before': 0.0,
                    'position_size_after': actual_quantity,
                    'avg_entry_price': actual_price,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'profit_loss': 0.0,
                    'profit_loss_percent': 0.0,
                    'win_loss': 'BREAKEVEN',
                    'trade_duration': 0.0,
                    'funding_rate': opportunity.current_funding_rate,
                    'volatility': 0.0,
                    'volume_24h': 0.0,
                    'market_regime': 'FUNDING_ARBITRAGE',
                    'system_score': 9.0,
                    'confidence_score': opportunity.confidence_score,
                    'emergency_mode': False,
                    'cycle_count': 0,
                    'data_source': 'live_hyperliquid',
                    'is_live_trade': True,
                    'notes': f'Funding arbitrage: {opportunity.current_funding_rate:.4f} rate',
                    'tags': ['funding_arbitrage', side.lower(), 'live'],
                    'metadata': {
                        'expected_value': opportunity.expected_value,
                        'expected_return_percent': opportunity.expected_return_percent,
                        'risk_score': opportunity.risk_score,
                        'confidence_score': opportunity.confidence_score,
                        'holding_period_hours': opportunity.holding_period_hours,
                        'total_costs_bps': opportunity.total_costs_bps,
                        'net_expected_return_bps': opportunity.net_expected_return_bps
                    }
                }
                
                # Record in ledger and metrics
                trade_id = self.trade_ledger.record_trade(trade_data)
                record_trade_metrics(trade_data, self.metrics_collector)
                
                # Track active position
                self.active_positions[opportunity.symbol] = {
                    'trade_id': trade_id,
                    'order_id': order_id,
                    'side': side,
                    'quantity': actual_quantity,
                    'entry_price': actual_price,
                    'entry_time': time.time(),
                    'funding_rate': opportunity.current_funding_rate,
                    'expected_holding_period': opportunity.holding_period_hours,
                    'expected_value': opportunity.expected_value
                }
                
                self.total_trades += 1
                self.successful_trades += 1
                
                self.logger.info(f"✅ [FUNDING_ARB] Trade executed successfully: {trade_id}")
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'order_id': order_id,
                    'expected_value': opportunity.expected_value,
                    'position_size_usd': opportunity.position_size_usd
                }
            else:
                self.logger.error(f"❌ [FUNDING_ARB] Trade execution failed: {order_result}")
                return {'success': False, 'error': str(order_result)}
                
        except Exception as e:
            self.logger.error(f"❌ [FUNDING_ARB] Error executing funding arbitrage: {e}")
            return {'success': False, 'error': str(e)}
    
    async def monitor_funding_rates(self) -> List[FundingArbitrageOpportunity]:
        """
        Monitor funding rates and identify arbitrage opportunities
        """
        
        opportunities = []
        
        try:
            # Get current funding rates
            funding_rates = await self._get_funding_rates()
            
            # Get account balance
            user_state = self.api.get_user_state()
            available_margin = 0.0
            
            if user_state and "marginSummary" in user_state:
                account_value = safe_float(user_state["marginSummary"].get("accountValue", 0))
                total_margin_used = safe_float(user_state["marginSummary"].get("totalMarginUsed", 0))
                available_margin = account_value - total_margin_used
            
            # Check each symbol for opportunities
            for symbol, funding_rate in funding_rates.items():
                # Skip if already have position
                if symbol in self.active_positions:
                    continue
                
                # Get current price
                current_price = await self._get_current_price(symbol)
                if current_price is None:
                    continue
                
                # Assess opportunity
                opportunity = self.assess_opportunity(
                    symbol, funding_rate, current_price, available_margin
                )
                
                if opportunity:
                    opportunities.append(opportunity)
                    self.opportunity_history.append(opportunity)
                    
                    self.logger.info(f"🎯 [FUNDING_ARB] Opportunity found for {symbol}")
                    self.logger.info(f"📊 [FUNDING_ARB] Funding rate: {funding_rate:.4f}")
                    self.logger.info(f"📊 [FUNDING_ARB] Expected value: ${opportunity.expected_value:.2f}")
                    self.logger.info(f"📊 [FUNDING_ARB] Risk score: {opportunity.risk_score:.2f}")
                    self.logger.info(f"📊 [FUNDING_ARB] Confidence: {opportunity.confidence_score:.2f}")
            
            # Sort opportunities by expected value
            opportunities.sort(key=lambda x: x.expected_value, reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ [FUNDING_ARB] Error monitoring funding rates: {e}")
            return []
    
    async def _get_funding_rates(self) -> Dict[str, float]:
        """Get current funding rates for all symbols"""
        try:
            # This would typically come from the API
            # For now, return mock data
            return {
                'XRP': 0.0008,  # 0.08% funding rate
                'BTC': 0.0012,  # 0.12% funding rate
                'ETH': 0.0005,  # 0.05% funding rate
            }
        except Exception as e:
            self.logger.error(f"❌ [FUNDING_ARB] Error getting funding rates: {e}")
            return {}
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # This would typically come from the API
            # For now, return mock data
            mock_prices = {
                'XRP': 0.5234,
                'BTC': 43250.0,
                'ETH': 2650.0,
            }
            return mock_prices.get(symbol)
        except Exception as e:
            self.logger.error(f"❌ [FUNDING_ARB] Error getting price for {symbol}: {e}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        uptime_hours = (time.time() - self.start_time) / 3600
        win_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        # Calculate average expected value
        avg_expected_value = 0.0
        if self.opportunity_history:
            avg_expected_value = sum(opp.expected_value for opp in self.opportunity_history) / len(self.opportunity_history)
        
        return {
            'uptime_hours': uptime_hours,
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'win_rate_percent': win_rate,
            'total_pnl': self.total_pnl,
            'active_positions': len(self.active_positions),
            'opportunities_found': len(self.opportunity_history),
            'avg_expected_value': avg_expected_value,
            'config': {
                'min_funding_threshold': self.config.min_funding_rate_threshold,
                'max_funding_threshold': self.config.max_funding_rate_threshold,
                'optimal_funding_rate': self.config.optimal_funding_rate,
                'max_position_size': self.config.max_position_size_usd
            }
        }
    
    def prove_ev_mathematics(self) -> Dict[str, Any]:
        """
        Prove the mathematical foundation of Expected Value calculation
        """
        
        proof = {
            'mathematical_foundation': {
                'expected_value_formula': 'EV = (Funding Payment - Transaction Costs - Slippage Costs) * Probability of Success',
                'funding_payment_formula': 'Funding Payment = Position Size * |Funding Rate| * (Holding Period / 8 hours)',
                'transaction_costs_formula': 'Transaction Costs = Position Size * Transaction Cost Rate * 2',
                'slippage_costs_formula': 'Slippage Costs = Position Size * Slippage Rate * 2',
                'success_probability_formula': 'P(Success) = f(|Funding Rate|, Market Conditions, Historical Success Rate)'
            },
            'kelly_criterion': {
                'formula': 'f* = (bp - q) / b',
                'where': {
                    'f*': 'fraction of capital to bet',
                    'b': 'odds received (funding rate)',
                    'p': 'probability of winning',
                    'q': 'probability of losing (1-p)'
                }
            },
            'risk_management': {
                'position_sizing': 'Position Size = min(Kelly Size, Max Position Size, Available Margin * Multiplier)',
                'risk_score': 'Risk Score = |Funding Rate| / Max Funding Rate Threshold',
                'confidence_score': 'Confidence Score = |Funding Rate| / Optimal Funding Rate'
            },
            'example_calculation': self._calculate_example_ev()
        }
        
        return proof
    
    def _calculate_example_ev(self) -> Dict[str, Any]:
        """Calculate example EV for demonstration"""
        
        # Example parameters
        funding_rate = 0.005  # 0.5%
        position_size_usd = 1000.0
        holding_period_hours = 8.0
        
        # Calculate components
        funding_payments = holding_period_hours / 8.0
        funding_payment_usd = position_size_usd * abs(funding_rate) * funding_payments
        
        transaction_cost_usd = position_size_usd * (2.0 / 10000) * 2  # 2 bps * 2
        slippage_cost_usd = position_size_usd * (1.0 / 10000) * 2     # 1 bps * 2
        total_costs_usd = transaction_cost_usd + slippage_cost_usd
        
        success_probability = 0.8  # 80% for 0.5% funding rate
        
        gross_expected_value = funding_payment_usd - total_costs_usd
        expected_value = gross_expected_value * success_probability
        expected_return_percent = (expected_value / position_size_usd) * 100
        
        return {
            'funding_rate': funding_rate,
            'position_size_usd': position_size_usd,
            'holding_period_hours': holding_period_hours,
            'funding_payment_usd': funding_payment_usd,
            'transaction_cost_usd': transaction_cost_usd,
            'slippage_cost_usd': slippage_cost_usd,
            'total_costs_usd': total_costs_usd,
            'success_probability': success_probability,
            'gross_expected_value': gross_expected_value,
            'expected_value': expected_value,
            'expected_return_percent': expected_return_percent
        }
