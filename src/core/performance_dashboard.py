#!/usr/bin/env python3
"""
📊 REAL-TIME PERFORMANCE DASHBOARD
===================================

Comprehensive dashboard for monitoring trading performance in real-time.
"""

from src.core.utils.decimal_boundary_guard import safe_float
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.logger import Logger
from core.utils.config_manager import ConfigManager
from core.api.hyperliquid_api import HyperliquidAPI


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    total_balance: float
    available_balance: float
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    open_positions: int
    total_volume: float
    roi: float
    average_win: float
    average_loss: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    risk_reward_ratio: float
    exposure: float


class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, config: ConfigManager, api: HyperliquidAPI):
        self.config = config
        self.api = api
        self.logger = Logger()
        
        # Dashboard state
        self.running = False
        self.update_interval = 5  # seconds
        
        # Performance tracking
        self.current_metrics = None
        self.historical_metrics = []
        self.equity_curve = []
        self.trade_history = []
        
        # Statistics
        self.daily_stats = {}
        self.weekly_stats = {}
        self.monthly_stats = {}
        
        # Alerts
        self.performance_alerts = []
        self.risk_alerts = []
        
        self.logger.info("📊 [DASHBOARD] Performance dashboard initialized")
    
    def start_dashboard(self):
        """Start the performance dashboard"""
        try:
            self.running = True
            self.logger.info("🚀 [DASHBOARD] Starting performance dashboard...")
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            monitor_thread.start()
            
            # Start display thread
            display_thread = threading.Thread(target=self._display_loop, daemon=True)
            display_thread.start()
            
            self.logger.info("✅ [DASHBOARD] Dashboard started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ [DASHBOARD] Error starting dashboard: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        try:
            while self.running:
                try:
                    # Update metrics
                    self._update_metrics()
                    
                    # Check for alerts
                    self._check_alerts()
                    
                    # Update statistics
                    self._update_statistics()
                    
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitor loop: {e}")
                    time.sleep(10)
                    
        except Exception as e:
            self.logger.error(f"Critical error in monitor loop: {e}")
    
    def _update_metrics(self):
        """Update current performance metrics"""
        try:
            # Get account info
            account_info = self.api.get_account_info()
            if not account_info:
                return
            
            # Get positions
            positions = self.api.get_positions()
            open_positions = [p for p in positions if p.get('szi', 0) != 0]
            
            # Calculate P&L
            unrealized_pnl = sum(safe_float(p.get('unrealizedPnl', 0)) for p in open_positions)
            
            # Get trade history (last 100 trades)
            trades = self._get_recent_trades()
            
            # Calculate metrics
            total_balance = safe_float(account_info.get('marginSummary', {}).get('accountValue', 0))
            available_balance = safe_float(account_info.get('marginSummary', {}).get('availableMargin', 0))
            
            # Trade statistics
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in trades if t.get('pnl', 0) <= 0])
            total_trades = len(trades)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate other metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                total_balance=total_balance,
                available_balance=available_balance,
                total_pnl=self._calculate_total_pnl(trades),
                daily_pnl=self._calculate_daily_pnl(trades),
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self._calculate_realized_pnl(trades),
                win_rate=win_rate,
                profit_factor=self._calculate_profit_factor(trades),
                sharpe_ratio=self._calculate_sharpe_ratio(),
                max_drawdown=self._calculate_max_drawdown(),
                current_drawdown=self._calculate_current_drawdown(),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                open_positions=len(open_positions),
                total_volume=self._calculate_total_volume(trades),
                roi=self._calculate_roi(),
                average_win=self._calculate_average_win(trades),
                average_loss=self._calculate_average_loss(trades),
                best_trade=self._find_best_trade(trades),
                worst_trade=self._find_worst_trade(trades),
                consecutive_wins=self._count_consecutive_wins(trades),
                consecutive_losses=self._count_consecutive_losses(trades),
                risk_reward_ratio=self._calculate_risk_reward_ratio(trades),
                exposure=self._calculate_exposure(open_positions, total_balance)
            )
            
            # Update current metrics
            self.current_metrics = metrics
            
            # Store historical metrics
            self.historical_metrics.append(metrics)
            if len(self.historical_metrics) > 10000:
                self.historical_metrics = self.historical_metrics[-5000:]
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'balance': total_balance,
                'pnl': metrics.total_pnl
            })
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def _get_recent_trades(self) -> List[Dict]:
        """Get recent trades with P&L info"""
        try:
            # This would normally fetch from API
            # For now, return sample data
            return self.trade_history[-100:] if self.trade_history else []
            
        except Exception:
            return []
    
    def _calculate_total_pnl(self, trades: List[Dict]) -> float:
        """Calculate total P&L from trades"""
        try:
            return sum(t.get('pnl', 0) for t in trades)
        except Exception:
            return 0.0
    
    def _calculate_daily_pnl(self, trades: List[Dict]) -> float:
        """Calculate today's P&L"""
        try:
            today = datetime.now().date()
            daily_trades = [t for t in trades if datetime.fromisoformat(t.get('timestamp', '')).date() == today]
            return sum(t.get('pnl', 0) for t in daily_trades)
        except Exception:
            return 0.0
    
    def _calculate_realized_pnl(self, trades: List[Dict]) -> float:
        """Calculate realized P&L"""
        try:
            closed_trades = [t for t in trades if t.get('status') == 'closed']
            return sum(t.get('pnl', 0) for t in closed_trades)
        except Exception:
            return 0.0
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
            gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
            
            return gross_profit / gross_loss if gross_loss > 0 else safe_float('inf')
            
        except Exception:
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from equity curve"""
        try:
            if len(self.equity_curve) < 20:
                return 0.0
            
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_balance = self.equity_curve[i-1]['balance']
                curr_balance = self.equity_curve[i]['balance']
                if prev_balance > 0:
                    returns.append((curr_balance - prev_balance) / prev_balance)
            
            if not returns:
                return 0.0
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Annualized Sharpe ratio (assuming 252 trading days)
            return np.sqrt(252) * (avg_return / std_return) if std_return > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        try:
            if len(self.equity_curve) < 2:
                return 0.0
            
            balances = [e['balance'] for e in self.equity_curve]
            peak = balances[0]
            max_dd = 0.0
            
            for balance in balances[1:]:
                if balance > peak:
                    peak = balance
                else:
                    dd = (peak - balance) / peak
                    max_dd = max(max_dd, dd)
            
            return max_dd * 100  # Return as percentage
            
        except Exception:
            return 0.0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        try:
            if len(self.equity_curve) < 2:
                return 0.0
            
            balances = [e['balance'] for e in self.equity_curve]
            peak = max(balances)
            current = balances[-1]
            
            return ((peak - current) / peak * 100) if peak > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_roi(self) -> float:
        """Calculate return on investment"""
        try:
            if not self.equity_curve:
                return 0.0
            
            initial_balance = self.equity_curve[0]['balance']
            current_balance = self.equity_curve[-1]['balance']
            
            return ((current_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_average_win(self, trades: List[Dict]) -> float:
        """Calculate average winning trade"""
        try:
            winning_trades = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
            return np.mean(winning_trades) if winning_trades else 0.0
        except Exception:
            return 0.0
    
    def _calculate_average_loss(self, trades: List[Dict]) -> float:
        """Calculate average losing trade"""
        try:
            losing_trades = [abs(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) < 0]
            return np.mean(losing_trades) if losing_trades else 0.0
        except Exception:
            return 0.0
    
    def _find_best_trade(self, trades: List[Dict]) -> float:
        """Find best trade P&L"""
        try:
            pnls = [t.get('pnl', 0) for t in trades]
            return max(pnls) if pnls else 0.0
        except Exception:
            return 0.0
    
    def _find_worst_trade(self, trades: List[Dict]) -> float:
        """Find worst trade P&L"""
        try:
            pnls = [t.get('pnl', 0) for t in trades]
            return min(pnls) if pnls else 0.0
        except Exception:
            return 0.0
    
    def _count_consecutive_wins(self, trades: List[Dict]) -> int:
        """Count current consecutive wins"""
        try:
            if not trades:
                return 0
            
            consecutive = 0
            for trade in reversed(trades):
                if trade.get('pnl', 0) > 0:
                    consecutive += 1
                else:
                    break
            
            return consecutive
            
        except Exception:
            return 0
    
    def _count_consecutive_losses(self, trades: List[Dict]) -> int:
        """Count current consecutive losses"""
        try:
            if not trades:
                return 0
            
            consecutive = 0
            for trade in reversed(trades):
                if trade.get('pnl', 0) <= 0:
                    consecutive += 1
                else:
                    break
            
            return consecutive
            
        except Exception:
            return 0
    
    def _calculate_risk_reward_ratio(self, trades: List[Dict]) -> float:
        """Calculate average risk/reward ratio"""
        try:
            avg_win = self._calculate_average_win(trades)
            avg_loss = self._calculate_average_loss(trades)
            
            return avg_win / avg_loss if avg_loss > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_exposure(self, positions: List[Dict], total_balance: float) -> float:
        """Calculate current market exposure"""
        try:
            if total_balance <= 0:
                return 0.0
            
            total_position_value = sum(
                abs(safe_float(p.get('szi', 0)) * safe_float(p.get('markPx', 0)))
                for p in positions
            )
            
            return (total_position_value / total_balance * 100) if total_balance > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _check_alerts(self):
        """Check for performance and risk alerts"""
        try:
            if not self.current_metrics:
                return
            
            # Check drawdown alert
            if self.current_metrics.current_drawdown > 10:
                self._add_alert("risk", f"High drawdown: {self.current_metrics.current_drawdown:.1f}%")
            
            # Check consecutive losses
            if self.current_metrics.consecutive_losses >= 3:
                self._add_alert("performance", f"Consecutive losses: {self.current_metrics.consecutive_losses}")
            
            # Check win rate
            if self.current_metrics.win_rate < 0.4 and self.current_metrics.total_trades > 10:
                self._add_alert("performance", f"Low win rate: {self.current_metrics.win_rate:.1%}")
            
            # Check exposure
            if self.current_metrics.exposure > 80:
                self._add_alert("risk", f"High exposure: {self.current_metrics.exposure:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add alert to appropriate list"""
        try:
            alert = {
                'timestamp': datetime.now(),
                'type': alert_type,
                'message': message
            }
            
            if alert_type == "risk":
                self.risk_alerts.append(alert)
                self.logger.warning(f"🚨 [RISK ALERT] {message}")
            else:
                self.performance_alerts.append(alert)
                self.logger.info(f"📊 [PERFORMANCE ALERT] {message}")
            
            # Keep alerts manageable
            if len(self.risk_alerts) > 100:
                self.risk_alerts = self.risk_alerts[-50:]
            if len(self.performance_alerts) > 100:
                self.performance_alerts = self.performance_alerts[-50:]
            
        except Exception as e:
            self.logger.error(f"Error adding alert: {e}")
    
    def _update_statistics(self):
        """Update daily/weekly/monthly statistics"""
        try:
            if not self.current_metrics:
                return
            
            now = datetime.now()
            
            # Daily stats
            today_key = now.strftime("%Y-%m-%d")
            if today_key not in self.daily_stats:
                self.daily_stats[today_key] = {
                    'trades': 0,
                    'pnl': 0,
                    'win_rate': 0,
                    'volume': 0
                }
            
            self.daily_stats[today_key]['pnl'] = self.current_metrics.daily_pnl
            self.daily_stats[today_key]['trades'] = self.current_metrics.total_trades
            self.daily_stats[today_key]['win_rate'] = self.current_metrics.win_rate
            self.daily_stats[today_key]['volume'] = self.current_metrics.total_volume
            
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")
    
    def _display_loop(self):
        """Display dashboard information"""
        try:
            while self.running:
                try:
                    if self.current_metrics:
                        self._display_dashboard()
                    
                    time.sleep(10)  # Update display every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in display loop: {e}")
                    time.sleep(30)
                    
        except Exception as e:
            self.logger.error(f"Critical error in display loop: {e}")
    
    def _display_dashboard(self):
        """Display formatted dashboard"""
        try:
            m = self.current_metrics
            
            dashboard = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    PERFORMANCE DASHBOARD                          ║
╠══════════════════════════════════════════════════════════════════╣
║ ACCOUNT SUMMARY                                                   ║
║ Total Balance: ${m.total_balance:,.2f}                           ║
║ Available: ${m.available_balance:,.2f}                           ║
║ Exposure: {m.exposure:.1f}%                                      ║
╠══════════════════════════════════════════════════════════════════╣
║ P&L METRICS                                                       ║
║ Total P&L: ${m.total_pnl:,.2f}                                  ║
║ Daily P&L: ${m.daily_pnl:,.2f}                                  ║
║ Unrealized: ${m.unrealized_pnl:,.2f}                            ║
║ ROI: {m.roi:.1f}%                                               ║
╠══════════════════════════════════════════════════════════════════╣
║ TRADING STATISTICS                                                ║
║ Total Trades: {m.total_trades}                                   ║
║ Win Rate: {m.win_rate:.1%}                                      ║
║ Profit Factor: {m.profit_factor:.2f}                            ║
║ Sharpe Ratio: {m.sharpe_ratio:.2f}                              ║
╠══════════════════════════════════════════════════════════════════╣
║ RISK METRICS                                                      ║
║ Max Drawdown: {m.max_drawdown:.1f}%                             ║
║ Current DD: {m.current_drawdown:.1f}%                           ║
║ Risk/Reward: {m.risk_reward_ratio:.2f}                          ║
║ Open Positions: {m.open_positions}                               ║
╚══════════════════════════════════════════════════════════════════╝
"""
            self.logger.info(dashboard)
            
        except Exception as e:
            self.logger.error(f"Error displaying dashboard: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        try:
            if not self.current_metrics:
                return {}
            
            return asdict(self.current_metrics)
            
        except Exception:
            return {}
    
    def stop_dashboard(self):
        """Stop the performance dashboard"""
        self.running = False
        self.logger.info("📊 [DASHBOARD] Dashboard stopped") 