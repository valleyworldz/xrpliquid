#!/usr/bin/env python3
"""
📊 REAL-TIME AUTONOMOUS TRADING MONITOR
=====================================

Advanced real-time monitoring system for supreme autonomous trading:
- Live performance tracking
- Risk monitoring and alerts
- Market condition analysis
- System health monitoring
- Performance optimization alerts
- Emergency detection
"""

from src.core.utils.decimal_boundary_guard import safe_float
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager
from core.api.hyperliquid_api import HyperliquidAPI

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    total_balance: float
    unrealized_pnl: float
    realized_pnl: float
    daily_return: float
    session_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    volatility: float
    return_distribution: List[float]

@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    timestamp: datetime
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    current_drawdown: float
    max_drawdown: float
    position_concentration: float
    leverage_ratio: float
    correlation_risk: float
    liquidity_risk: float
    tail_risk: float
    stress_test_result: float
    risk_score: float

@dataclass
class MarketMetrics:
    """Real-time market metrics"""
    timestamp: datetime
    market_volatility: float
    trend_strength: float
    momentum: float
    volume_profile: float
    liquidity_depth: float
    bid_ask_spread: float
    order_flow_imbalance: float
    funding_rates: Dict[str, float]
    market_sentiment: str
    regime_confidence: float
    correlation_matrix: Dict[str, Dict[str, float]]

@dataclass
class SystemMetrics:
    """Real-time system metrics"""
    timestamp: datetime
    api_latency: float
    order_fill_rate: float
    execution_slippage: float
    system_uptime: float
    memory_usage: float
    cpu_usage: float
    thread_health: Dict[str, bool]
    error_rate: float
    alert_count: int
    last_optimization: datetime
    strategies_active: int
    positions_open: int

class RealTimeMonitor:
    """Advanced real-time monitoring system"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = Logger()
        self.api = HyperliquidAPI(testnet=False)
        
        # Monitoring configuration
        self.monitoring_config = self.config.get("monitoring", {
            "update_interval": 5,
            "alert_thresholds": {
                "max_drawdown": 0.03,
                "max_daily_loss": 0.05,
                "min_sharpe": 1.0,
                "max_var_95": 0.02,
                "max_correlation": 0.8,
                "min_liquidity": 0.1,
                "max_latency": 1000,
                "min_fill_rate": 0.95
            }
        })
        
        # State tracking
        self.running = False
        self.performance_history = []
        self.risk_history = []
        self.market_history = []
        self.system_history = []
        self.alert_history = []
        self.last_balance = 0.0
        self.session_start_balance = 0.0
        self.session_start_time = datetime.now()
        
        # Performance calculation
        self.returns_buffer = []
        self.trade_history = []
        self.drawdown_tracker = []
        
        # Threading
        self.monitor_thread = None
        
        self.logger.info("📊 [MONITOR] Real-time monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        try:
            self.running = True
            self.session_start_time = datetime.now()
            
            # Get initial balance
            user_state = self.api.get_user_state()
            if user_state:
                self.session_start_balance = safe_float(user_state.get("marginSummary", {}).get("accountValue", "0"))
                self.last_balance = self.session_start_balance
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("🚀 [MONITOR] Real-time monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error starting monitoring: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("🛑 [MONITOR] Real-time monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect all metrics
                performance_metrics = self._calculate_performance_metrics()
                risk_metrics = self._calculate_risk_metrics()
                market_metrics = self._calculate_market_metrics()
                system_metrics = self._calculate_system_metrics()
                
                # Store metrics
                if performance_metrics:
                    self.performance_history.append(performance_metrics)
                if risk_metrics:
                    self.risk_history.append(risk_metrics)
                if market_metrics:
                    self.market_history.append(market_metrics)
                if system_metrics:
                    self.system_history.append(system_metrics)
                
                # Check for alerts
                self._check_alerts(performance_metrics, risk_metrics, market_metrics, system_metrics)
                
                # Log comprehensive status
                self._log_status_summary(performance_metrics, risk_metrics, market_metrics, system_metrics)
                
                # Cleanup old data
                self._cleanup_history()
                
                # Sleep until next update
                time.sleep(self.monitoring_config.get("update_interval", 5))
                
            except Exception as e:
                self.logger.error(f"❌ [MONITOR] Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _calculate_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Calculate real-time performance metrics"""
        try:
            # Get current account state
            user_state = self.api.get_user_state()
            if not user_state:
                return None
            
            current_balance = safe_float(user_state.get("marginSummary", {}).get("accountValue", "0"))
            unrealized_pnl = safe_float(user_state.get("marginSummary", {}).get("unrealizedPnl", "0"))
            
            # Calculate returns
            if self.last_balance > 0:
                period_return = (current_balance - self.last_balance) / self.last_balance
                self.returns_buffer.append(period_return)
                
                # Keep only recent returns for calculations
                if len(self.returns_buffer) > 1000:
                    self.returns_buffer = self.returns_buffer[-500:]
            
            # Session return
            session_return = 0.0
            if self.session_start_balance > 0:
                session_return = (current_balance - self.session_start_balance) / self.session_start_balance
            
            # Daily return (approximate)
            daily_return = session_return  # Simplified for now
            
            # Calculate Sharpe ratio
            sharpe_ratio = 0.0
            if len(self.returns_buffer) > 10:
                returns_array = np.array(self.returns_buffer)
                if np.std(returns_array) > 0:
                    sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252 * 24 * 12)  # Annualized 5-min intervals
            
            # Calculate drawdown
            if current_balance > 0:
                peak_balance = max([h.total_balance for h in self.performance_history[-100:]] + [current_balance])
                current_drawdown = (peak_balance - current_balance) / peak_balance
                self.drawdown_tracker.append(current_drawdown)
                max_drawdown = max(self.drawdown_tracker[-100:]) if self.drawdown_tracker else 0.0
            else:
                current_drawdown = 0.0
                max_drawdown = 0.0
            
            # Calculate trade statistics (simplified)
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get("pnl", 0) > 0)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Profit factor
            total_wins = sum(trade.get("pnl", 0) for trade in self.trade_history if trade.get("pnl", 0) > 0)
            total_losses = abs(sum(trade.get("pnl", 0) for trade in self.trade_history if trade.get("pnl", 0) < 0))
            profit_factor = total_wins / total_losses if total_losses > 0 else safe_float('inf')
            
            # Largest win/loss
            pnls = [trade.get("pnl", 0) for trade in self.trade_history]
            largest_win = max(pnls) if pnls else 0.0
            largest_loss = min(pnls) if pnls else 0.0
            
            # Volatility
            volatility = np.std(self.returns_buffer) if len(self.returns_buffer) > 1 else 0.0
            
            # Update last balance
            self.last_balance = current_balance
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                total_balance=current_balance,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=current_balance - self.session_start_balance - unrealized_pnl,
                daily_return=daily_return,
                session_return=session_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_duration=0.0,  # To be calculated
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                largest_win=largest_win,
                largest_loss=largest_loss,
                volatility=volatility,
                return_distribution=self.returns_buffer[-100:] if len(self.returns_buffer) >= 100 else []
            )
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error calculating performance metrics: {e}")
            return None
    
    def _calculate_risk_metrics(self) -> Optional[RiskMetrics]:
        """Calculate real-time risk metrics"""
        try:
            if len(self.returns_buffer) < 10:
                return None
            
            returns_array = np.array(self.returns_buffer[-252:])  # Last 252 periods
            
            # Value at Risk calculations
            var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0.0
            var_99 = np.percentile(returns_array, 1) if len(returns_array) > 0 else 0.0
            
            # Expected Shortfall (Conditional VaR)
            es_threshold = np.percentile(returns_array, 5)
            tail_returns = returns_array[returns_array <= es_threshold]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0.0
            
            # Current drawdown
            current_drawdown = self.drawdown_tracker[-1] if self.drawdown_tracker else 0.0
            max_drawdown = max(self.drawdown_tracker) if self.drawdown_tracker else 0.0
            
            # Risk score (composite)
            risk_factors = [
                abs(var_95) * 5,  # VaR impact
                abs(expected_shortfall) * 3,  # Tail risk impact
                current_drawdown * 4,  # Drawdown impact
                np.std(returns_array) * 2  # Volatility impact
            ]
            risk_score = sum(risk_factors)
            
            return RiskMetrics(
                timestamp=datetime.now(),
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                position_concentration=0.0,  # To be calculated
                leverage_ratio=0.0,  # To be calculated
                correlation_risk=0.0,  # To be calculated
                liquidity_risk=0.0,  # To be calculated
                tail_risk=abs(expected_shortfall),
                stress_test_result=0.0,  # To be calculated
                risk_score=risk_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error calculating risk metrics: {e}")
            return None
    
    def _calculate_market_metrics(self) -> Optional[MarketMetrics]:
        """Calculate real-time market metrics"""
        try:
            # Get market data for major tokens
            tokens = self.config.get("trading.default_tokens", ["BTC", "ETH", "SOL"])
            market_data = {}
            
            for token in tokens:
                data = self.api.get_market_data(token)
                if data:
                    market_data[token] = data
            
            if not market_data:
                return None
            
            # Calculate aggregate market metrics
            volatilities = []
            volumes = []
            spreads = []
            
            for token, data in market_data.items():
                # Volatility
                if "price_history" in data and len(data["price_history"]) > 1:
                    prices = np.array(data["price_history"][-20:])
                    if len(prices) > 1:
                        returns = np.diff(prices) / prices[:-1]
                        volatility = np.std(returns)
                        volatilities.append(volatility)
                
                # Volume
                volume = data.get("volume", 0)
                if volume > 0:
                    volumes.append(volume)
                
                # Spread (if available)
                spread = data.get("spread", 0)
                if spread > 0:
                    spreads.append(spread)
            
            # Aggregate metrics
            market_volatility = np.mean(volatilities) if volatilities else 0.0
            avg_volume = np.mean(volumes) if volumes else 0.0
            avg_spread = np.mean(spreads) if spreads else 0.0
            
            # Determine market sentiment
            if market_volatility > 0.03:
                market_sentiment = "high_volatility"
            elif market_volatility < 0.01:
                market_sentiment = "low_volatility"
            else:
                market_sentiment = "normal"
            
            return MarketMetrics(
                timestamp=datetime.now(),
                market_volatility=market_volatility,
                trend_strength=0.0,  # To be calculated
                momentum=0.0,  # To be calculated
                volume_profile=avg_volume,
                liquidity_depth=0.0,  # To be calculated
                bid_ask_spread=avg_spread,
                order_flow_imbalance=0.0,  # To be calculated
                funding_rates={},  # To be collected
                market_sentiment=market_sentiment,
                regime_confidence=0.8,  # Default
                correlation_matrix={}  # To be calculated
            )
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error calculating market metrics: {e}")
            return None
    
    def _calculate_system_metrics(self) -> Optional[SystemMetrics]:
        """Calculate real-time system metrics"""
        try:
            # API latency test
            start_time = time.time()
            user_state = self.api.get_user_state()
            api_latency = (time.time() - start_time) * 1000  # milliseconds
            
            # System uptime
            system_uptime = (datetime.now() - self.session_start_time).total_seconds()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                api_latency=api_latency,
                order_fill_rate=0.95,  # Default/estimated
                execution_slippage=0.001,  # Default/estimated
                system_uptime=system_uptime,
                memory_usage=0.0,  # To be calculated
                cpu_usage=0.0,  # To be calculated
                thread_health={},  # To be monitored
                error_rate=0.0,  # To be calculated
                alert_count=len(self.alert_history),
                last_optimization=datetime.now(),
                strategies_active=4,  # Number of enabled strategies
                positions_open=0  # To be calculated
            )
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error calculating system metrics: {e}")
            return None
    
    def _check_alerts(self, performance: Optional[PerformanceMetrics], risk: Optional[RiskMetrics], 
                     market: Optional[MarketMetrics], system: Optional[SystemMetrics]) -> None:
        """Check for alert conditions"""
        try:
            alerts = []
            thresholds = self.monitoring_config.get("alert_thresholds", {})
            
            # Performance alerts
            if performance:
                if performance.max_drawdown > thresholds.get("max_drawdown", 0.03):
                    alerts.append({
                        "type": "CRITICAL",
                        "category": "RISK",
                        "message": f"Maximum drawdown exceeded: {performance.max_drawdown:.3f}",
                        "timestamp": datetime.now()
                    })
                
                if performance.daily_return < -thresholds.get("max_daily_loss", 0.05):
                    alerts.append({
                        "type": "WARNING",
                        "category": "PERFORMANCE",
                        "message": f"Daily loss threshold exceeded: {performance.daily_return:.3f}",
                        "timestamp": datetime.now()
                    })
                
                if performance.sharpe_ratio < thresholds.get("min_sharpe", 1.0) and len(self.returns_buffer) > 50:
                    alerts.append({
                        "type": "INFO",
                        "category": "PERFORMANCE",
                        "message": f"Sharpe ratio below target: {performance.sharpe_ratio:.2f}",
                        "timestamp": datetime.now()
                    })
            
            # Risk alerts
            if risk:
                if abs(risk.var_95) > thresholds.get("max_var_95", 0.02):
                    alerts.append({
                        "type": "WARNING",
                        "category": "RISK",
                        "message": f"VaR 95% exceeded: {risk.var_95:.3f}",
                        "timestamp": datetime.now()
                    })
                
                if risk.risk_score > 0.1:
                    alerts.append({
                        "type": "WARNING",
                        "category": "RISK",
                        "message": f"Risk score elevated: {risk.risk_score:.3f}",
                        "timestamp": datetime.now()
                    })
            
            # System alerts
            if system:
                if system.api_latency > thresholds.get("max_latency", 1000):
                    alerts.append({
                        "type": "WARNING",
                        "category": "SYSTEM",
                        "message": f"High API latency: {system.api_latency:.1f}ms",
                        "timestamp": datetime.now()
                    })
            
            # Process alerts
            for alert in alerts:
                self.alert_history.append(alert)
                self._send_alert(alert)
            
            # Keep alert history manageable
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]
                
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error checking alerts: {e}")
    
    def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert notification"""
        try:
            alert_level = alert["type"]
            category = alert["category"]
            message = alert["message"]
            
            if alert_level == "CRITICAL":
                self.logger.error(f"🚨 [{category}] CRITICAL ALERT: {message}")
            elif alert_level == "WARNING":
                self.logger.warning(f"⚠️ [{category}] WARNING: {message}")
            else:
                self.logger.info(f"ℹ️ [{category}] INFO: {message}")
                
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error sending alert: {e}")
    
    def _log_status_summary(self, performance: Optional[PerformanceMetrics], risk: Optional[RiskMetrics],
                           market: Optional[MarketMetrics], system: Optional[SystemMetrics]) -> None:
        """Log comprehensive status summary"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "session_duration": str(datetime.now() - self.session_start_time),
            }
            
            if performance:
                summary.update({
                    "balance": f"${performance.total_balance:.2f}",
                    "session_return": f"{performance.session_return:.3f}%",
                    "unrealized_pnl": f"${performance.unrealized_pnl:.2f}",
                    "sharpe_ratio": f"{performance.sharpe_ratio:.2f}",
                    "max_drawdown": f"{performance.max_drawdown:.3f}%",
                    "win_rate": f"{performance.win_rate:.1%}",
                    "total_trades": performance.total_trades
                })
            
            if risk:
                summary.update({
                    "var_95": f"{risk.var_95:.3f}",
                    "risk_score": f"{risk.risk_score:.3f}",
                    "current_drawdown": f"{risk.current_drawdown:.3f}%"
                })
            
            if market:
                summary.update({
                    "market_volatility": f"{market.market_volatility:.3f}",
                    "market_sentiment": market.market_sentiment,
                    "avg_spread": f"{market.bid_ask_spread:.4f}"
                })
            
            if system:
                summary.update({
                    "api_latency": f"{system.api_latency:.1f}ms",
                    "uptime": f"{system.system_uptime:.0f}s",
                    "alerts": len(self.alert_history)
                })
            
            self.logger.info(f"📊 [MONITOR] Status Summary: {json.dumps(summary, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error logging status summary: {e}")
    
    def _cleanup_history(self) -> None:
        """Clean up historical data to prevent memory issues"""
        try:
            max_history = 10000
            
            if len(self.performance_history) > max_history:
                self.performance_history = self.performance_history[-max_history//2:]
            
            if len(self.risk_history) > max_history:
                self.risk_history = self.risk_history[-max_history//2:]
            
            if len(self.market_history) > max_history:
                self.market_history = self.market_history[-max_history//2:]
            
            if len(self.system_history) > max_history:
                self.system_history = self.system_history[-max_history//2:]
                
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error cleaning up history: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current comprehensive status"""
        try:
            return {
                "performance": asdict(self.performance_history[-1]) if self.performance_history else {},
                "risk": asdict(self.risk_history[-1]) if self.risk_history else {},
                "market": asdict(self.market_history[-1]) if self.market_history else {},
                "system": asdict(self.system_history[-1]) if self.system_history else {},
                "recent_alerts": self.alert_history[-10:],
                "session_duration": str(datetime.now() - self.session_start_time),
                "data_points": {
                    "performance": len(self.performance_history),
                    "risk": len(self.risk_history),
                    "market": len(self.market_history),
                    "system": len(self.system_history)
                }
            }
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error getting current status: {e}")
            return {}
    
    def save_monitoring_data(self) -> None:
        """Save all monitoring data to files"""
        try:
            import os
            os.makedirs('logs/monitoring', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save performance data
            with open(f'logs/monitoring/performance_{timestamp}.json', 'w') as f:
                json.dump([asdict(p) for p in self.performance_history], f, indent=2, default=str)
            
            # Save risk data
            with open(f'logs/monitoring/risk_{timestamp}.json', 'w') as f:
                json.dump([asdict(r) for r in self.risk_history], f, indent=2, default=str)
            
            # Save market data
            with open(f'logs/monitoring/market_{timestamp}.json', 'w') as f:
                json.dump([asdict(m) for m in self.market_history], f, indent=2, default=str)
            
            # Save system data
            with open(f'logs/monitoring/system_{timestamp}.json', 'w') as f:
                json.dump([asdict(s) for s in self.system_history], f, indent=2, default=str)
            
            # Save alerts
            with open(f'logs/monitoring/alerts_{timestamp}.json', 'w') as f:
                json.dump(self.alert_history, f, indent=2, default=str)
            
            self.logger.info(f"💾 [MONITOR] Monitoring data saved to logs/monitoring/")
            
        except Exception as e:
            self.logger.error(f"❌ [MONITOR] Error saving monitoring data: {e}") 