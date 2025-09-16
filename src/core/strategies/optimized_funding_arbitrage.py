"""
🎯 OPTIMIZED FUNDING ARBITRAGE STRATEGY
======================================
Advanced funding arbitrage strategy optimized for profitability
based on comprehensive backtesting analysis.
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
class OptimizedFundingArbitrageConfig:
    """Optimized configuration for funding arbitrage strategy"""
    
    # Funding rate thresholds (optimized based on backtesting)
    min_funding_rate_threshold: float = 0.0008  # 0.08% minimum (increased from 0.01%)
    max_funding_rate_threshold: float = 0.008   # 0.8% maximum (reduced from 1%)
    optimal_funding_rate: float = 0.003         # 0.3% optimal (reduced from 0.5%)
    
    # Position sizing (optimized for profitability)
    max_position_size_usd: float = 2000.0       # Increased from $1000
    position_size_multiplier: float = 0.08      # 8% of capital (increased from 5%)
    min_position_size_usd: float = 200.0        # Increased from $100
    
    # Risk management
    max_drawdown_percent: float = 3.0           # Reduced from 5%
    stop_loss_funding_rate: float = 0.015       # 1.5% stop loss
    take_profit_funding_rate: float = 0.0005    # 0.05% take profit
    
    # Execution parameters (Hyperliquid 1-hour funding cycles)
    funding_rate_check_interval: int = 300      # 5 minutes (aligned with 1-hour cycles)
    execution_delay_seconds: int = 15           # Reduced from 30
    max_execution_time_seconds: int = 45        # Reduced from 60
    funding_cycle_hours: int = 1                # Hyperliquid standard: 1-hour cycles
    
    # Holding period optimization (Hyperliquid 1-hour cycles)
    expected_holding_period_hours: float = 1.0  # Aligned with 1-hour funding cycles
    funding_payment_frequency_hours: float = 1.0  # Hyperliquid standard: 1-hour funding
    
    # Cost optimization
    transaction_cost_bps: float = 0.8           # Reduced from 1.0 bps
    slippage_cost_bps: float = 0.3              # Reduced from 0.5 bps
    
    # Market conditions
    min_volume_24h_usd: float = 2000000.0       # Increased from $1M
    max_spread_bps: float = 8.0                 # Reduced from 10 bps
    min_liquidity_usd: float = 100000.0         # Increased from $50K
    
    # Advanced parameters
    confidence_threshold: float = 0.7           # Higher confidence required
    volatility_adjustment: bool = True          # Enable volatility adjustments
    market_regime_filtering: bool = True        # Enable regime filtering
    dynamic_position_sizing: bool = True        # Enable dynamic sizing

@dataclass
class OptimizedFundingArbitrageOpportunity:
    """Enhanced opportunity structure with advanced metrics"""
    
    # Basic opportunity data
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
    
    # Advanced metrics
    market_regime: str = "neutral"
    volatility_percent: float = 0.0
    liquidity_score: float = 0.0
    execution_score: float = 0.0
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Profitability metrics
    min_profit_threshold_bps: float = 5.0       # Minimum 5 bps profit required
    cost_ratio: float = 0.0                     # Cost to profit ratio
    efficiency_score: float = 0.0               # Overall efficiency score

class OptimizedFundingArbitrageStrategy:
    """
    🎯 OPTIMIZED FUNDING ARBITRAGE STRATEGY
    Advanced funding arbitrage with profitability optimization
    """
    
    def __init__(self, 
                 config: OptimizedFundingArbitrageConfig,
                 api: HyperliquidAPI,
                 logger: Optional[Logger] = None):
        self.config = config
        self.api = api
        self.logger = logger or Logger()
        
        # Initialize components
        self.trade_ledger = TradeLedgerManager(data_dir="data/optimized_funding_arbitrage", logger=self.logger)
        self.metrics_collector = get_metrics_collector(port=8003, logger=self.logger)
        
        # Initialize advanced risk unit sizing
        risk_config = RiskUnitConfig(
            target_volatility_percent=1.5,      # Reduced target volatility
            max_equity_at_risk_percent=0.8,     # Reduced max risk
            base_equity_at_risk_percent=0.3,    # Reduced base risk
            kelly_multiplier=0.15,              # Reduced Kelly multiplier
            min_position_size_usd=config.min_position_size_usd,
            max_position_size_usd=config.max_position_size_usd,
        )
        self.risk_sizing = RiskUnitSizing(risk_config, self.logger)
        
        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.opportunity_history: List[OptimizedFundingArbitrageOpportunity] = []
        self.last_funding_check = 0.0
        self.total_pnl = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        
        # Performance tracking
        self.start_time = time.time()
        self.funding_rate_history: Dict[str, List[Tuple[float, float]]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.volatility_history: Dict[str, List[float]] = {}
        
        # Market regime tracking
        self.current_regime = "neutral"
        self.regime_confidence = 0.0
        self.regime_history: List[Tuple[float, str, float]] = []
        
        # Advanced metrics
        self.efficiency_score = 0.0
        self.cost_efficiency = 0.0
        self.execution_quality = 0.0
        
        self.logger.info("🎯 [OPTIMIZED_FUNDING_ARB] Optimized Funding Arbitrage Strategy initialized")
        self.logger.info(f"📊 [OPTIMIZED_FUNDING_ARB] Min funding threshold: {self.config.min_funding_rate_threshold:.4f}")
        self.logger.info(f"📊 [OPTIMIZED_FUNDING_ARB] Max funding threshold: {self.config.max_funding_rate_threshold:.4f}")
        self.logger.info(f"📊 [OPTIMIZED_FUNDING_ARB] Optimal funding rate: {self.config.optimal_funding_rate:.4f}")
        self.logger.info(f"📊 [OPTIMIZED_FUNDING_ARB] Min position size: ${self.config.min_position_size_usd:,.2f}")
        self.logger.info(f"📊 [OPTIMIZED_FUNDING_ARB] Max position size: ${self.config.max_position_size_usd:,.2f}")
        self.logger.info("🎯 [OPTIMIZED_FUNDING_ARB] Advanced risk management enabled")
    
    def analyze_market_regime(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Analyze current market regime for opportunity filtering"""
        
        try:
            # Get recent price data
            if self.config.symbol not in self.price_history:
                return "neutral", 0.5
            
            prices = self.price_history[self.config.symbol][-100:]  # Last 100 data points
            if len(prices) < 50:
                return "neutral", 0.5
            
            # Calculate regime indicators
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized volatility
            
            # Trend analysis
            sma_short = np.mean(prices[-10:])
            sma_long = np.mean(prices[-30:])
            trend_strength = (sma_short - sma_long) / sma_long
            
            # Volatility analysis
            vol_percentile = np.percentile(self.volatility_history.get(self.config.symbol, [0.1]), 75)
            is_high_vol = volatility > vol_percentile
            
            # Regime classification
            if trend_strength > 0.02 and not is_high_vol:
                regime = "bull"
                confidence = min(0.9, abs(trend_strength) * 10)
            elif trend_strength < -0.02 and not is_high_vol:
                regime = "bear"
                confidence = min(0.9, abs(trend_strength) * 10)
            elif is_high_vol:
                regime = "volatile"
                confidence = min(0.8, volatility * 2)
            else:
                regime = "neutral"
                confidence = 0.5
            
            # Update regime history
            self.regime_history.append((time.time(), regime, confidence))
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZED_FUNDING_ARB] Error analyzing market regime: {e}")
            return "neutral", 0.5
    
    def calculate_advanced_metrics(self, 
                                 symbol: str,
                                 funding_rate: float,
                                 current_price: float,
                                 available_margin: float) -> Dict[str, Any]:
        """Calculate advanced metrics for opportunity assessment"""
        
        try:
            # Volatility calculation
            volatility_percent = self._calculate_symbol_volatility(symbol, current_price)
            
            # Liquidity score (simplified)
            liquidity_score = min(1.0, available_margin / 10000.0)  # Higher margin = better liquidity
            
            # Execution score based on spread and volatility
            spread_penalty = min(1.0, self.config.max_spread_bps / 10.0)
            vol_penalty = min(1.0, volatility_percent / 5.0)
            execution_score = max(0.1, 1.0 - spread_penalty - vol_penalty)
            
            # Cost efficiency
            position_size = min(self.config.max_position_size_usd, available_margin * 0.1)
            total_costs = position_size * (self.config.transaction_cost_bps + self.config.slippage_cost_bps) / 10000
            potential_profit = position_size * abs(funding_rate)
            cost_ratio = total_costs / potential_profit if potential_profit > 0 else 1.0
            
            # Efficiency score
            efficiency_score = (execution_score * liquidity_score * (1 - cost_ratio)) ** 0.5
            
            return {
                'volatility_percent': volatility_percent,
                'liquidity_score': liquidity_score,
                'execution_score': execution_score,
                'cost_ratio': cost_ratio,
                'efficiency_score': efficiency_score,
                'position_size': position_size,
                'total_costs': total_costs,
                'potential_profit': potential_profit
            }
            
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZED_FUNDING_ARB] Error calculating advanced metrics: {e}")
            return {
                'volatility_percent': 0.0,
                'liquidity_score': 0.5,
                'execution_score': 0.5,
                'cost_ratio': 1.0,
                'efficiency_score': 0.0,
                'position_size': 0.0,
                'total_costs': 0.0,
                'potential_profit': 0.0
            }
    
    def assess_optimized_opportunity(self, 
                                   symbol: str,
                                   current_funding_rate: float,
                                   current_price: float,
                                   available_margin: float,
                                   market_data: Optional[Dict[str, Any]] = None) -> Optional[OptimizedFundingArbitrageOpportunity]:
        """Assess funding arbitrage opportunity with advanced filtering"""
        
        try:
            # 1. Basic funding rate check
            funding_rate_magnitude = abs(current_funding_rate)
            if funding_rate_magnitude < self.config.min_funding_rate_threshold:
                return None
            
            if funding_rate_magnitude > self.config.max_funding_rate_threshold:
                return None
            
            # 2. Market regime filtering
            if self.config.market_regime_filtering:
                regime, regime_confidence = self.analyze_market_regime(market_data or {})
                if regime == "volatile" and regime_confidence > 0.7:
                    self.logger.debug(f"🚫 [OPTIMIZED_FUNDING_ARB] Skipping opportunity due to volatile regime")
                    return None
            
            # 3. Calculate advanced metrics
            advanced_metrics = self.calculate_advanced_metrics(
                symbol, current_funding_rate, current_price, available_margin
            )
            
            # 4. Efficiency filtering
            if advanced_metrics['efficiency_score'] < 0.3:
                self.logger.debug(f"🚫 [OPTIMIZED_FUNDING_ARB] Skipping opportunity due to low efficiency: {advanced_metrics['efficiency_score']:.3f}")
                return None
            
            # 5. Cost ratio filtering
            if advanced_metrics['cost_ratio'] > 0.3:  # Costs > 30% of potential profit
                self.logger.debug(f"🚫 [OPTIMIZED_FUNDING_ARB] Skipping opportunity due to high cost ratio: {advanced_metrics['cost_ratio']:.3f}")
                return None
            
            # 6. Calculate optimal position size using risk unit sizing
            position_size_usd, risk_metrics = self.calculate_optimal_position_size(
                symbol, available_margin, current_funding_rate, current_price,
                confidence_score=advanced_metrics['efficiency_score']
            )
            
            if position_size_usd < self.config.min_position_size_usd:
                return None
            
            # 7. Profitability check
            total_costs = position_size_usd * (self.config.transaction_cost_bps + self.config.slippage_cost_bps) / 10000
            potential_profit = position_size_usd * funding_rate_magnitude
            net_profit = potential_profit - total_costs
            
            if net_profit < position_size_usd * self.config.min_profit_threshold_bps / 10000:
                self.logger.debug(f"🚫 [OPTIMIZED_FUNDING_ARB] Skipping opportunity due to insufficient profit: {net_profit:.2f}")
                return None
            
            # 8. Calculate expected value and return
            expected_value = net_profit * 0.8  # Assume 80% success rate
            expected_return_percent = (expected_value / position_size_usd) * 100
            
            # 9. Risk assessment
            risk_score = self._calculate_risk_score(
                funding_rate_magnitude, advanced_metrics['volatility_percent'], 
                advanced_metrics['execution_score']
            )
            
            # 10. Confidence score
            confidence_score = min(0.95, advanced_metrics['efficiency_score'] * 0.7 + 
                                 (1 - risk_score) * 0.3)
            
            if confidence_score < self.config.confidence_threshold:
                return None
            
            # 11. Create optimized opportunity
            opportunity = OptimizedFundingArbitrageOpportunity(
                symbol=symbol,
                current_funding_rate=current_funding_rate,
                predicted_funding_rate=current_funding_rate * 0.8,  # Assume 20% reversion
                expected_value=expected_value,
                expected_return_percent=expected_return_percent,
                risk_score=risk_score,
                confidence_score=confidence_score,
                position_size_usd=position_size_usd,
                entry_price=current_price,
                exit_price=current_price,
                holding_period_hours=self.config.expected_holding_period_hours,
                total_costs_bps=(total_costs / position_size_usd) * 10000,
                net_expected_return_bps=(expected_value / position_size_usd) * 10000,
                market_regime=regime if self.config.market_regime_filtering else "neutral",
                volatility_percent=advanced_metrics['volatility_percent'],
                liquidity_score=advanced_metrics['liquidity_score'],
                execution_score=advanced_metrics['execution_score'],
                risk_metrics=risk_metrics,
                min_profit_threshold_bps=self.config.min_profit_threshold_bps,
                cost_ratio=advanced_metrics['cost_ratio'],
                efficiency_score=advanced_metrics['efficiency_score']
            )
            
            # Record opportunity
            self.opportunity_history.append(opportunity)
            if len(self.opportunity_history) > 1000:
                self.opportunity_history = self.opportunity_history[-1000:]
            
            self.logger.info(f"✅ [OPTIMIZED_FUNDING_ARB] Opportunity identified: {symbol}")
            self.logger.info(f"   Funding rate: {current_funding_rate:.4f}")
            self.logger.info(f"   Position size: ${position_size_usd:,.2f}")
            self.logger.info(f"   Expected return: {expected_return_percent:.2f}%")
            self.logger.info(f"   Efficiency score: {advanced_metrics['efficiency_score']:.3f}")
            self.logger.info(f"   Confidence: {confidence_score:.3f}")
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZED_FUNDING_ARB] Error assessing opportunity: {e}")
            return None
    
    def calculate_optimal_position_size(self, 
                                      symbol: str,
                                      available_margin: float,
                                      funding_rate: float,
                                      current_price: float,
                                      confidence_score: float = 0.5) -> Tuple[float, Dict[str, Any]]:
        """Calculate optimal position size using advanced risk unit sizing"""
        
        try:
            # Calculate volatility for the symbol
            volatility_percent = self._calculate_symbol_volatility(symbol, current_price)
            
            # Calculate win probability based on funding rate and efficiency
            funding_rate_magnitude = abs(funding_rate)
            base_win_prob = min(0.85, max(0.4, funding_rate_magnitude / self.config.max_funding_rate_threshold))
            
            # Adjust win probability based on confidence score
            win_probability = base_win_prob * confidence_score
            
            # Estimate average win/loss percentages
            avg_win_percent = funding_rate_magnitude * 0.9  # Assume 90% of funding rate as win
            avg_loss_percent = funding_rate_magnitude * 0.1  # Assume 10% of funding rate as loss
            
            # Use risk unit sizing system
            optimal_position_size, risk_metrics = self.risk_sizing.calculate_optimal_position_size(
                symbol=symbol,
                account_value=available_margin,
                volatility_percent=volatility_percent,
                confidence_score=confidence_score,
                win_probability=win_probability,
                avg_win_percent=avg_win_percent,
                avg_loss_percent=avg_loss_percent,
                market_regime=self.current_regime
            )
            
            # Apply dynamic position sizing adjustments
            if self.config.dynamic_position_sizing:
                # Adjust based on recent performance
                if len(self.opportunity_history) > 10:
                    recent_opportunities = self.opportunity_history[-10:]
                    avg_efficiency = np.mean([opp.efficiency_score for opp in recent_opportunities])
                    efficiency_multiplier = min(1.2, max(0.8, avg_efficiency))
                    optimal_position_size *= efficiency_multiplier
                
                # Adjust based on market regime
                if self.current_regime == "volatile":
                    optimal_position_size *= 0.7  # Reduce size in volatile markets
                elif self.current_regime in ["bull", "bear"]:
                    optimal_position_size *= 1.1  # Slightly increase in trending markets
            
            # Ensure position size is within bounds
            optimal_position_size = max(self.config.min_position_size_usd, 
                                      min(optimal_position_size, self.config.max_position_size_usd))
            
            return optimal_position_size, risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZED_FUNDING_ARB] Error calculating position size: {e}")
            return self.config.min_position_size_usd, {}
    
    def _calculate_symbol_volatility(self, symbol: str, current_price: float) -> float:
        """Calculate symbol volatility for risk management"""
        
        try:
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(current_price)
            
            # Keep only recent prices (last 720 hours = 30 days)
            if len(self.price_history[symbol]) > 720:
                self.price_history[symbol] = self.price_history[symbol][-720:]
            
            # Calculate volatility if we have enough data
            if len(self.price_history[symbol]) < 24:  # Need at least 24 hours
                return 0.02  # Default 2% volatility
            
            prices = np.array(self.price_history[symbol])
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized volatility
            
            # Update volatility history
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            self.volatility_history[symbol].append(volatility)
            
            if len(self.volatility_history[symbol]) > 1000:
                self.volatility_history[symbol] = self.volatility_history[symbol][-1000:]
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZED_FUNDING_ARB] Error calculating volatility: {e}")
            return 0.02  # Default volatility
    
    def _calculate_risk_score(self, 
                            funding_rate_magnitude: float,
                            volatility_percent: float,
                            execution_score: float) -> float:
        """Calculate comprehensive risk score"""
        
        try:
            # Funding rate risk (higher magnitude = higher risk)
            funding_risk = min(1.0, funding_rate_magnitude / self.config.max_funding_rate_threshold)
            
            # Volatility risk
            volatility_risk = min(1.0, volatility_percent / 5.0)  # 5% volatility = max risk
            
            # Execution risk (lower score = higher risk)
            execution_risk = 1.0 - execution_score
            
            # Combined risk score
            risk_score = (funding_risk * 0.4 + volatility_risk * 0.3 + execution_risk * 0.3)
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZED_FUNDING_ARB] Error calculating risk score: {e}")
            return 0.5  # Default risk score
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        try:
            runtime_hours = (time.time() - self.start_time) / 3600
            
            # Basic metrics
            total_opportunities = len(self.opportunity_history)
            successful_opportunities = len([opp for opp in self.opportunity_history if opp.efficiency_score > 0.5])
            
            # Efficiency metrics
            avg_efficiency = np.mean([opp.efficiency_score for opp in self.opportunity_history]) if self.opportunity_history else 0.0
            avg_cost_ratio = np.mean([opp.cost_ratio for opp in self.opportunity_history]) if self.opportunity_history else 0.0
            
            # Risk metrics
            avg_risk_score = np.mean([opp.risk_score for opp in self.opportunity_history]) if self.opportunity_history else 0.0
            avg_confidence = np.mean([opp.confidence_score for opp in self.opportunity_history]) if self.opportunity_history else 0.0
            
            return {
                'runtime_hours': runtime_hours,
                'total_opportunities': total_opportunities,
                'successful_opportunities': successful_opportunities,
                'success_rate': successful_opportunities / total_opportunities if total_opportunities > 0 else 0.0,
                'avg_efficiency_score': avg_efficiency,
                'avg_cost_ratio': avg_cost_ratio,
                'avg_risk_score': avg_risk_score,
                'avg_confidence': avg_confidence,
                'current_regime': self.current_regime,
                'regime_confidence': self.regime_confidence,
                'total_pnl': self.total_pnl,
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'win_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZED_FUNDING_ARB] Error getting performance metrics: {e}")
            return {}
    
    def update_performance_tracking(self, trade_result: Dict[str, Any]):
        """Update performance tracking with trade results"""
        
        try:
            self.total_trades += 1
            self.total_pnl += trade_result.get('pnl', 0.0)
            
            if trade_result.get('pnl', 0.0) > 0:
                self.successful_trades += 1
            
            # Update efficiency metrics
            if trade_result.get('efficiency_score'):
                self.efficiency_score = (self.efficiency_score * 0.9 + trade_result['efficiency_score'] * 0.1)
            
            if trade_result.get('cost_ratio'):
                self.cost_efficiency = (self.cost_efficiency * 0.9 + (1 - trade_result['cost_ratio']) * 0.1)
            
            if trade_result.get('execution_quality'):
                self.execution_quality = (self.execution_quality * 0.9 + trade_result['execution_quality'] * 0.1)
            
            self.logger.info(f"📊 [OPTIMIZED_FUNDING_ARB] Performance updated: {self.total_trades} trades, {self.successful_trades} successful")
            
        except Exception as e:
            self.logger.error(f"❌ [OPTIMIZED_FUNDING_ARB] Error updating performance: {e}")
