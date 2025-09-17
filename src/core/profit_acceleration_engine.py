#!/usr/bin/env python3
"""
🚀 PROFIT ACCELERATION ENGINE
============================

Advanced profit acceleration system that maximizes returns through:
- Dynamic position sizing based on performance
- Compound growth optimization
- Market momentum capitalization
- Risk-adjusted profit scaling
- Intelligent reinvestment strategies
- Performance-based strategy allocation
"""

from src.core.utils.decimal_boundary_guard import safe_float
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum

from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager
from core.api.hyperliquid_api import HyperliquidAPI

class AccelerationMode(Enum):
    """Profit acceleration modes"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate" 
    AGGRESSIVE = "aggressive"
    TURBO = "turbo"

@dataclass
class AccelerationMetrics:
    """Profit acceleration performance metrics"""
    current_mode: AccelerationMode
    acceleration_factor: float
    compound_rate: float
    profit_velocity: float
    risk_adjusted_return: float
    momentum_score: float
    reinvestment_rate: float
    position_size_multiplier: float
    success_probability: float
    target_achieved: bool

@dataclass
class MarketMomentum:
    """Market momentum analysis"""
    trend_strength: float
    volatility_trend: float
    volume_momentum: float
    price_acceleration: float
    sentiment_momentum: float
    overall_momentum: float
    momentum_confidence: float
    optimal_acceleration: float

class ProfitAccelerationEngine:
    """Advanced profit acceleration system"""
    
    def __init__(self, config: ConfigManager, api: HyperliquidAPI):
        self.config = config
        self.api = api
        self.logger = Logger()
        
        # Load acceleration configuration
        self.accel_config = self.config.get("profit_acceleration", {
            "enabled": True,
            "base_acceleration_factor": 1.2,
            "max_acceleration_factor": 3.0,
            "momentum_threshold": 0.7,
            "profit_threshold": 0.02,
            "compound_frequency": 300,  # 5 minutes
            "risk_scaling_factor": 0.8,
            "performance_lookback": 24,  # hours
            "volatility_adjustment": True,
            "dynamic_sizing": True
        })
        
        # State tracking
        self.current_mode = AccelerationMode.CONSERVATIVE
        self.acceleration_metrics = None
        self.performance_history = []
        self.momentum_history = []
        self.last_compound_time = datetime.now()
        self.session_start_balance = 0.0
        self.peak_balance = 0.0
        self.acceleration_start_time = None
        
        # Performance tracking
        self.profit_snapshots = []
        self.acceleration_decisions = []
        self.compound_events = []
        
        self.logger.info("🚀 [ACCEL] Profit acceleration engine initialized")
    
    def analyze_acceleration_opportunity(self) -> AccelerationMetrics:
        """Analyze current profit acceleration opportunity"""
        try:
            # Get current account state
            user_state = self.api.get_user_state()
            if not user_state:
                return self._get_default_metrics()
            
            current_balance = safe_float(user_state.get("marginSummary", {}).get("accountValue", "0"))
            
            # Initialize session tracking
            if self.session_start_balance == 0:
                self.session_start_balance = current_balance
                self.peak_balance = current_balance
            else:
                self.peak_balance = max(self.peak_balance, current_balance)
            
            # Calculate performance metrics
            session_return = (current_balance - self.session_start_balance) / self.session_start_balance if self.session_start_balance > 0 else 0
            
            # Analyze market momentum
            momentum = self._analyze_market_momentum()
            
            # Calculate profit velocity (rate of profit generation)
            profit_velocity = self._calculate_profit_velocity(current_balance)
            
            # Determine optimal acceleration mode
            optimal_mode = self._determine_acceleration_mode(session_return, momentum, profit_velocity)
            
            # Calculate acceleration factor
            acceleration_factor = self._calculate_acceleration_factor(optimal_mode, momentum, session_return)
            
            # Calculate compound rate
            compound_rate = self._calculate_compound_rate(acceleration_factor, momentum.overall_momentum)
            
            # Calculate position size multiplier
            position_multiplier = self._calculate_position_multiplier(acceleration_factor, momentum)
            
            # Calculate success probability
            success_prob = self._calculate_success_probability(momentum, session_return, profit_velocity)
            
            # Check if target achieved
            target_achieved = session_return >= self.accel_config.get("profit_threshold", 0.02)
            
            # Create acceleration metrics
            self.acceleration_metrics = AccelerationMetrics(
                current_mode=optimal_mode,
                acceleration_factor=acceleration_factor,
                compound_rate=compound_rate,
                profit_velocity=profit_velocity,
                risk_adjusted_return=session_return * (1 - momentum.overall_momentum * 0.1),  # Risk adjustment
                momentum_score=momentum.overall_momentum,
                reinvestment_rate=min(acceleration_factor * 0.8, 1.0),
                position_size_multiplier=position_multiplier,
                success_probability=success_prob,
                target_achieved=target_achieved
            )
            
            # Update mode if changed
            if optimal_mode != self.current_mode:
                self.logger.info(f"🔄 [ACCEL] Mode changed: {self.current_mode.value} → {optimal_mode.value}")
                self.current_mode = optimal_mode
                self.acceleration_start_time = datetime.now()
            
            # Log acceleration analysis
            self._log_acceleration_analysis(momentum)
            
            return self.acceleration_metrics
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error analyzing acceleration opportunity: {e}")
            return self._get_default_metrics()
    
    def _analyze_market_momentum(self) -> MarketMomentum:
        """Analyze market momentum for acceleration opportunities"""
        try:
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            momentum_scores = []
            
            for token in tokens:
                market_data = self.api.get_market_data(token)
                if not market_data:
                    continue
                
                # Price momentum
                if "price_history" in market_data and len(market_data["price_history"]) > 10:
                    prices = np.array(market_data["price_history"][-20:])
                    if len(prices) > 5:
                        # Calculate price acceleration
                        returns = np.diff(prices) / prices[:-1]
                        price_momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
                        
                        # Calculate trend strength
                        trend_slope = np.polyfit(range(len(prices)), prices, 1)[0] / np.mean(prices) if len(prices) > 1 else 0
                        trend_strength = min(abs(trend_slope) * 100, 1.0)
                        
                        # Volume momentum
                        volume = market_data.get("volume", 0)
                        volume_momentum = min(volume / 1000000, 1.0)  # Normalized
                        
                        # Volatility trend (lower volatility = more stable momentum)
                        volatility = np.std(returns) if len(returns) > 1 else 0
                        volatility_score = max(0, 1 - volatility * 20)  # Inverse volatility score
                        
                        # Combined momentum score
                        momentum_score = (
                            abs(price_momentum) * 0.4 + 
                            trend_strength * 0.3 +
                            volume_momentum * 0.2 +
                            volatility_score * 0.1
                        )
                        
                        momentum_scores.append(momentum_score)
            
            # Calculate overall momentum
            if momentum_scores:
                overall_momentum = np.mean(momentum_scores)
                momentum_confidence = 1.0 - np.std(momentum_scores) if len(momentum_scores) > 1 else 0.8
            else:
                overall_momentum = 0.3  # Conservative default
                momentum_confidence = 0.5
            
            # Calculate optimal acceleration
            optimal_acceleration = self._calculate_optimal_acceleration(overall_momentum, momentum_confidence)
            
            momentum = MarketMomentum(
                trend_strength=np.mean([s for s in momentum_scores]) if momentum_scores else 0.3,
                volatility_trend=0.7,  # Simplified
                volume_momentum=0.6,   # Simplified
                price_acceleration=overall_momentum,
                sentiment_momentum=0.5,  # Simplified
                overall_momentum=overall_momentum,
                momentum_confidence=momentum_confidence,
                optimal_acceleration=optimal_acceleration
            )
            
            self.momentum_history.append(momentum)
            
            # Keep momentum history manageable
            if len(self.momentum_history) > 1000:
                self.momentum_history = self.momentum_history[-500:]
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error analyzing market momentum: {e}")
            return MarketMomentum(
                trend_strength=0.3, volatility_trend=0.5, volume_momentum=0.4,
                price_acceleration=0.3, sentiment_momentum=0.4, overall_momentum=0.35,
                momentum_confidence=0.5, optimal_acceleration=1.1
            )
    
    def _calculate_profit_velocity(self, current_balance: float) -> float:
        """Calculate the rate of profit generation"""
        try:
            if len(self.profit_snapshots) < 2:
                self.profit_snapshots.append({
                    'timestamp': datetime.now(),
                    'balance': current_balance
                })
                return 0.0
            
            # Add current snapshot
            self.profit_snapshots.append({
                'timestamp': datetime.now(),
                'balance': current_balance
            })
            
            # Keep only recent snapshots (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.profit_snapshots = [s for s in self.profit_snapshots if s['timestamp'] > cutoff_time]
            
            if len(self.profit_snapshots) < 2:
                return 0.0
            
            # Calculate velocity (profit per minute)
            time_span = (self.profit_snapshots[-1]['timestamp'] - self.profit_snapshots[0]['timestamp']).total_seconds() / 60
            balance_change = self.profit_snapshots[-1]['balance'] - self.profit_snapshots[0]['balance']
            
            if time_span > 0 and self.profit_snapshots[0]['balance'] > 0:
                velocity = (balance_change / self.profit_snapshots[0]['balance']) / time_span  # % per minute
                return max(0, velocity)  # Only positive velocity
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error calculating profit velocity: {e}")
            return 0.0
    
    def _determine_acceleration_mode(self, session_return: float, momentum: MarketMomentum, 
                                   profit_velocity: float) -> AccelerationMode:
        """Determine optimal acceleration mode based on conditions"""
        try:
            # Calculate acceleration score
            score = 0.0
            
            # Session return factor
            if session_return > 0.05:  # > 5%
                score += 3
            elif session_return > 0.02:  # > 2%
                score += 2
            elif session_return > 0.01:  # > 1%
                score += 1
            
            # Momentum factor
            if momentum.overall_momentum > 0.8:
                score += 3
            elif momentum.overall_momentum > 0.6:
                score += 2
            elif momentum.overall_momentum > 0.4:
                score += 1
            
            # Profit velocity factor
            if profit_velocity > 0.001:  # > 0.1% per minute
                score += 2
            elif profit_velocity > 0.0005:  # > 0.05% per minute
                score += 1
            
            # Momentum confidence factor
            if momentum.momentum_confidence > 0.8:
                score += 1
            
            # Determine mode based on score
            if score >= 7:
                return AccelerationMode.TURBO
            elif score >= 5:
                return AccelerationMode.AGGRESSIVE
            elif score >= 3:
                return AccelerationMode.MODERATE
            else:
                return AccelerationMode.CONSERVATIVE
                
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error determining acceleration mode: {e}")
            return AccelerationMode.CONSERVATIVE
    
    def _calculate_acceleration_factor(self, mode: AccelerationMode, momentum: MarketMomentum, 
                                     session_return: float) -> float:
        """Calculate acceleration factor based on mode and conditions"""
        try:
            base_factors = {
                AccelerationMode.CONSERVATIVE: 1.1,
                AccelerationMode.MODERATE: 1.3,
                AccelerationMode.AGGRESSIVE: 1.8,
                AccelerationMode.TURBO: 2.5
            }
            
            base_factor = base_factors.get(mode, 1.1)
            
            # Apply momentum multiplier
            momentum_multiplier = 1.0 + (momentum.overall_momentum - 0.5) * 0.5
            
            # Apply performance multiplier
            performance_multiplier = 1.0 + min(session_return * 2, 0.5)
            
            # Apply confidence multiplier
            confidence_multiplier = 0.5 + momentum.momentum_confidence * 0.5
            
            # Calculate final factor
            acceleration_factor = base_factor * momentum_multiplier * performance_multiplier * confidence_multiplier
            
            # Apply limits
            max_factor = self.accel_config.get("max_acceleration_factor", 3.0)
            min_factor = self.accel_config.get("base_acceleration_factor", 1.2)
            
            return max(min_factor, min(acceleration_factor, max_factor))
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error calculating acceleration factor: {e}")
            return 1.1
    
    def _calculate_compound_rate(self, acceleration_factor: float, momentum: float) -> float:
        """Calculate optimal compound rate"""
        try:
            # Base compound rate
            base_rate = 0.8  # 80% reinvestment
            
            # Adjust based on acceleration factor
            acceleration_adjustment = (acceleration_factor - 1.0) * 0.2
            
            # Adjust based on momentum
            momentum_adjustment = momentum * 0.15
            
            compound_rate = base_rate + acceleration_adjustment + momentum_adjustment
            
            # Apply limits
            return max(0.5, min(compound_rate, 0.95))  # 50% to 95%
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error calculating compound rate: {e}")
            return 0.8
    
    def _calculate_position_multiplier(self, acceleration_factor: float, momentum: MarketMomentum) -> float:
        """Calculate position size multiplier for acceleration"""
        try:
            # Base multiplier from acceleration factor
            base_multiplier = 1.0 + (acceleration_factor - 1.0) * 0.5
            
            # Apply momentum confidence adjustment
            confidence_adjustment = momentum.momentum_confidence * 0.3
            
            # Apply risk scaling
            risk_scaling = self.accel_config.get("risk_scaling_factor", 0.8)
            
            multiplier = (base_multiplier + confidence_adjustment) * risk_scaling
            
            # Apply limits (max 2x position size)
            return max(1.0, min(multiplier, 2.0))
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error calculating position multiplier: {e}")
            return 1.0
    
    def _calculate_success_probability(self, momentum: MarketMomentum, session_return: float, 
                                     profit_velocity: float) -> float:
        """Calculate probability of successful acceleration"""
        try:
            # Base probability
            base_prob = 0.6
            
            # Momentum factor
            momentum_factor = momentum.overall_momentum * momentum.momentum_confidence * 0.3
            
            # Performance factor
            performance_factor = min(session_return * 5, 0.2) if session_return > 0 else 0
            
            # Velocity factor
            velocity_factor = min(profit_velocity * 1000, 0.1)
            
            success_prob = base_prob + momentum_factor + performance_factor + velocity_factor
            
            return max(0.1, min(success_prob, 0.95))
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error calculating success probability: {e}")
            return 0.6
    
    def _calculate_optimal_acceleration(self, momentum: float, confidence: float) -> float:
        """Calculate optimal acceleration based on momentum"""
        try:
            # Base acceleration
            base_accel = 1.1
            
            # Momentum bonus
            momentum_bonus = momentum * 0.8
            
            # Confidence bonus
            confidence_bonus = confidence * 0.4
            
            optimal = base_accel + momentum_bonus + confidence_bonus
            
            return max(1.0, min(optimal, 2.5))
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error calculating optimal acceleration: {e}")
            return 1.1
    
    def execute_profit_acceleration(self) -> Dict[str, Any]:
        """Execute profit acceleration strategy"""
        try:
            if not self.acceleration_metrics:
                self.analyze_acceleration_opportunity()
            
            if not self.acceleration_metrics:
                return {"success": False, "reason": "No acceleration metrics available"}
            
            # Check if acceleration should be executed
            if not self._should_execute_acceleration():
                return {"success": False, "reason": "Acceleration conditions not met"}
            
            # Execute compound reinvestment
            compound_result = self._execute_compound_reinvestment()
            
            # Execute dynamic position sizing
            sizing_result = self._execute_dynamic_sizing()
            
            # Log acceleration execution
            acceleration_event = {
                "timestamp": datetime.now(),
                "mode": self.acceleration_metrics.current_mode.value,
                "acceleration_factor": self.acceleration_metrics.acceleration_factor,
                "compound_rate": self.acceleration_metrics.compound_rate,
                "position_multiplier": self.acceleration_metrics.position_size_multiplier,
                "success_probability": self.acceleration_metrics.success_probability,
                "compound_executed": compound_result.get("success", False),
                "sizing_executed": sizing_result.get("success", False)
            }
            
            self.acceleration_decisions.append(acceleration_event)
            
            self.logger.info(f"🚀 [ACCEL] Acceleration executed - Mode: {self.acceleration_metrics.current_mode.value}, "
                           f"Factor: {self.acceleration_metrics.acceleration_factor:.2f}")
            
            return {
                "success": True,
                "mode": self.acceleration_metrics.current_mode.value,
                "acceleration_factor": self.acceleration_metrics.acceleration_factor,
                "compound_result": compound_result,
                "sizing_result": sizing_result,
                "metrics": self.acceleration_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error executing profit acceleration: {e}")
            return {"success": False, "reason": f"Execution error: {e}"}
    
    def _should_execute_acceleration(self) -> bool:
        """Check if acceleration should be executed"""
        try:
            if not self.acceleration_metrics:
                return False
            
            # Check minimum success probability
            if self.acceleration_metrics.success_probability < 0.6:
                return False
            
            # Check minimum momentum
            if self.acceleration_metrics.momentum_score < 0.4:
                return False
            
            # Check time since last compound
            time_since_compound = datetime.now() - self.last_compound_time
            min_interval = timedelta(seconds=self.accel_config.get("compound_frequency", 300))
            
            if time_since_compound < min_interval:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error checking acceleration conditions: {e}")
            return False
    
    def _execute_compound_reinvestment(self) -> Dict[str, Any]:
        """Execute compound reinvestment strategy"""
        try:
            if not self.acceleration_metrics:
                return {"success": False, "reason": "No metrics available"}
            
            # Calculate reinvestment amount
            user_state = self.api.get_user_state()
            if not user_state:
                return {"success": False, "reason": "Cannot get account state"}
            
            current_balance = safe_float(user_state.get("marginSummary", {}).get("accountValue", "0"))
            profit_since_start = current_balance - self.session_start_balance
            
            if profit_since_start <= 0:
                return {"success": False, "reason": "No profit to reinvest"}
            
            # Calculate reinvestment amount
            reinvestment_amount = profit_since_start * self.acceleration_metrics.reinvestment_rate
            
            compound_event = {
                "timestamp": datetime.now(),
                "profit_available": profit_since_start,
                "reinvestment_rate": self.acceleration_metrics.reinvestment_rate,
                "reinvestment_amount": reinvestment_amount,
                "balance_before": current_balance,
                "compound_factor": self.acceleration_metrics.compound_rate
            }
            
            self.compound_events.append(compound_event)
            self.last_compound_time = datetime.now()
            
            self.logger.info(f"💰 [ACCEL] Compound reinvestment: ${reinvestment_amount:.2f} "
                           f"({self.acceleration_metrics.reinvestment_rate:.1%})")
            
            return {
                "success": True,
                "reinvestment_amount": reinvestment_amount,
                "reinvestment_rate": self.acceleration_metrics.reinvestment_rate,
                "profit_reinvested": profit_since_start
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error executing compound reinvestment: {e}")
            return {"success": False, "reason": f"Compound error: {e}"}
    
    def _execute_dynamic_sizing(self) -> Dict[str, Any]:
        """Execute dynamic position sizing based on acceleration"""
        try:
            if not self.acceleration_metrics:
                return {"success": False, "reason": "No metrics available"}
            
            # This would integrate with the portfolio manager to adjust position sizes
            # For now, we'll log the sizing decision
            
            self.logger.info(f"📏 [ACCEL] Dynamic sizing - Multiplier: {self.acceleration_metrics.position_size_multiplier:.2f}")
            
            return {
                "success": True,
                "position_multiplier": self.acceleration_metrics.position_size_multiplier,
                "mode": self.acceleration_metrics.current_mode.value
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error executing dynamic sizing: {e}")
            return {"success": False, "reason": f"Sizing error: {e}"}
    
    def _get_default_metrics(self) -> AccelerationMetrics:
        """Get default acceleration metrics"""
        return AccelerationMetrics(
            current_mode=AccelerationMode.CONSERVATIVE,
            acceleration_factor=1.1,
            compound_rate=0.8,
            profit_velocity=0.0,
            risk_adjusted_return=0.0,
            momentum_score=0.3,
            reinvestment_rate=0.8,
            position_size_multiplier=1.0,
            success_probability=0.6,
            target_achieved=False
        )
    
    def _log_acceleration_analysis(self, momentum: MarketMomentum) -> None:
        """Log detailed acceleration analysis"""
        try:
            if not self.acceleration_metrics:
                return
            
            analysis = {
                "mode": self.acceleration_metrics.current_mode.value,
                "acceleration_factor": f"{self.acceleration_metrics.acceleration_factor:.2f}",
                "momentum_score": f"{self.acceleration_metrics.momentum_score:.3f}",
                "profit_velocity": f"{self.acceleration_metrics.profit_velocity:.6f}",
                "success_probability": f"{self.acceleration_metrics.success_probability:.2%}",
                "compound_rate": f"{self.acceleration_metrics.compound_rate:.1%}",
                "position_multiplier": f"{self.acceleration_metrics.position_size_multiplier:.2f}",
                "market_momentum": f"{momentum.overall_momentum:.3f}",
                "momentum_confidence": f"{momentum.momentum_confidence:.3f}"
            }
            
            self.logger.info(f"📈 [ACCEL] Analysis: {json.dumps(analysis, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error logging acceleration analysis: {e}")
    
    def get_acceleration_status(self) -> Dict[str, Any]:
        """Get current acceleration status"""
        try:
            return {
                "current_metrics": self.acceleration_metrics.__dict__ if self.acceleration_metrics else {},
                "recent_momentum": [m.__dict__ for m in self.momentum_history[-10:]] if self.momentum_history else [],
                "acceleration_decisions_count": len(self.acceleration_decisions),
                "compound_events_count": len(self.compound_events),
                "session_duration": str(datetime.now() - (self.acceleration_start_time or datetime.now())),
                "last_compound": str(datetime.now() - self.last_compound_time),
                "profit_snapshots_count": len(self.profit_snapshots)
            }
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error getting acceleration status: {e}")
            return {}
    
    def save_acceleration_data(self) -> None:
        """Save acceleration performance data"""
        try:
            data = {
                "session_start_balance": self.session_start_balance,
                "peak_balance": self.peak_balance,
                "current_mode": self.current_mode.value,
                "performance_history": self.performance_history,
                "acceleration_decisions": self.acceleration_decisions,
                "compound_events": self.compound_events,
                "timestamp": datetime.now().isoformat()
            }
            
            filename = f"data/acceleration_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error saving acceleration data: {e}")

    def start_acceleration_engine(self) -> Dict[str, Any]:
        """Start the profit acceleration engine"""
        try:
            self.logger.info("🚀 [ACCEL] Starting profit acceleration engine...")
            
            # Initialize acceleration tracking
            self.acceleration_start_time = datetime.now()
            
            # Analyze initial opportunity
            metrics = self.analyze_acceleration_opportunity()
            
            # Execute initial acceleration if conditions are met
            if self._should_execute_acceleration():
                result = self.execute_profit_acceleration()
                self.logger.info(f"✅ [ACCEL] Initial acceleration executed: {result}")
                return result
            else:
                self.logger.info("⏳ [ACCEL] Waiting for optimal acceleration conditions")
                return {
                    "status": "waiting",
                    "message": "Monitoring for acceleration opportunities",
                    "metrics": metrics.__dict__
                }
                
        except Exception as e:
            self.logger.error(f"❌ [ACCEL] Error starting acceleration engine: {e}")
            return {"status": "error", "message": str(e)} 