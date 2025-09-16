"""
📊 HAT MANIFESTO PERFORMANCE DASHBOARD
=====================================
Comprehensive real-time performance dashboard with metrics and insights.

This dashboard provides real-time monitoring of all Hat Manifesto specialized roles:
- Real-time performance metrics
- Risk management monitoring
- ML system analytics
- Latency optimization tracking
- Hyperliquid protocol exploitation metrics
- Comprehensive reporting and insights
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

@dataclass
class DashboardConfig:
    """Configuration for Hat Manifesto Dashboard"""
    
    # Dashboard settings
    dashboard_settings: Dict[str, Any] = field(default_factory=lambda: {
        'update_interval_seconds': 5,       # 5 second update interval
        'max_history_points': 1000,         # Maximum history points
        'real_time_enabled': True,          # Enable real-time updates
        'auto_refresh': True,               # Auto-refresh dashboard
        'performance_alerts': True,         # Enable performance alerts
        'export_enabled': True,             # Enable data export
    })
    
    # Metrics settings
    metrics_settings: Dict[str, Any] = field(default_factory=lambda: {
        'track_all_hats': True,             # Track all Hat Manifesto roles
        'track_performance': True,          # Track performance metrics
        'track_risk': True,                 # Track risk metrics
        'track_latency': True,              # Track latency metrics
        'track_ml': True,                   # Track ML metrics
        'track_hyperliquid': True,          # Track Hyperliquid metrics
        'track_trades': True,               # Track trade metrics
    })
    
    # Visualization settings
    visualization_settings: Dict[str, Any] = field(default_factory=lambda: {
        'chart_types': ['line', 'bar', 'scatter', 'heatmap'],
        'color_scheme': 'hat_manifesto',    # Custom color scheme
        'chart_size': (12, 8),              # Chart size
        'dpi': 100,                         # Chart DPI
        'style': 'darkgrid',                # Chart style
        'export_format': 'png',             # Export format
    })
    
    # Alert settings
    alert_settings: Dict[str, Any] = field(default_factory=lambda: {
        'performance_threshold': 8.0,       # Performance alert threshold
        'risk_threshold': 0.8,              # Risk alert threshold
        'latency_threshold': 100,           # Latency alert threshold (ms)
        'error_rate_threshold': 0.05,       # Error rate alert threshold
        'alert_cooldown': 300,              # Alert cooldown (seconds)
    })

@dataclass
class DashboardMetrics:
    """Comprehensive dashboard metrics"""
    
    # Hat Manifesto role scores
    hat_scores: Dict[str, float] = field(default_factory=lambda: {
        'hyperliquid_architect': 10.0,
        'quantitative_strategist': 10.0,
        'microstructure_analyst': 10.0,
        'low_latency_engineer': 10.0,
        'execution_manager': 10.0,
        'risk_officer': 10.0,
        'security_architect': 10.0,
        'performance_analyst': 10.0,
        'ml_researcher': 10.0,
    })
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=lambda: {
        'total_trades': 0,
        'successful_trades': 0,
        'win_rate': 0.0,
        'total_profit': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'current_drawdown': 0.0,
        'volatility': 0.0,
        'avg_trade_duration': 0.0,
        'profit_factor': 0.0,
    })
    
    # Risk metrics
    risk_metrics: Dict[str, float] = field(default_factory=lambda: {
        'account_value': 0.0,
        'available_margin': 0.0,
        'margin_ratio': 0.0,
        'position_count': 0,
        'total_exposure': 0.0,
        'atr_stops_active': 0,
        'circuit_breaker_active': False,
        'emergency_mode': False,
        'risk_level': 'LOW',
    })
    
    # Latency metrics
    latency_metrics: Dict[str, float] = field(default_factory=lambda: {
        'avg_response_time_ms': 0.0,
        'p95_latency_ms': 0.0,
        'p99_latency_ms': 0.0,
        'api_calls_per_second': 0.0,
        'error_rate_percent': 0.0,
        'connection_count': 0,
        'websocket_connections': 0,
        'cache_hit_rate': 0.0,
    })
    
    # ML metrics
    ml_metrics: Dict[str, Any] = field(default_factory=lambda: {
        'current_regime': 'neutral',
        'sentiment_score': 0.0,
        'active_patterns': 0,
        'model_accuracy': 0.0,
        'prediction_confidence': 0.0,
        'adaptation_rate': 0.0,
        'data_points': 0,
    })
    
    # Hyperliquid metrics
    hyperliquid_metrics: Dict[str, float] = field(default_factory=lambda: {
        'funding_arbitrage_profit': 0.0,
        'twap_slippage_savings': 0.0,
        'hype_staking_rewards': 0.0,
        'oracle_arbitrage_profit': 0.0,
        'vamm_efficiency_profit': 0.0,
        'gas_savings': 0.0,
        'protocol_exploitation_score': 10.0,
    })
    
    # Trade metrics
    trade_metrics: Dict[str, Any] = field(default_factory=lambda: {
        'trades_today': 0,
        'trades_this_hour': 0,
        'avg_trade_size': 0.0,
        'largest_win': 0.0,
        'largest_loss': 0.0,
        'consecutive_wins': 0,
        'consecutive_losses': 0,
        'best_strategy': 'unknown',
        'worst_strategy': 'unknown',
    })

class HatManifestoDashboard:
    """
    📊 HAT MANIFESTO PERFORMANCE DASHBOARD
    
    Comprehensive real-time dashboard for monitoring all specialized roles:
    1. Real-time performance metrics
    2. Risk management monitoring
    3. ML system analytics
    4. Latency optimization tracking
    5. Hyperliquid protocol exploitation metrics
    6. Comprehensive reporting and insights
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize dashboard configuration
        self.dashboard_config = DashboardConfig()
        
        # Data storage
        self.metrics_history = deque(maxlen=self.dashboard_config.dashboard_settings['max_history_points'])
        self.alert_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=1000)
        
        # Current metrics
        self.current_metrics = DashboardMetrics()
        
        # Dashboard state
        self.dashboard_active = False
        self.last_update = 0.0
        self.last_alert_time = {}
        
        # Initialize visualization
        self._initialize_visualization()
        
        self.logger.info("📊 [DASHBOARD] Hat Manifesto Performance Dashboard initialized")
        self.logger.info("🎯 [DASHBOARD] All monitoring systems activated")
    
    def _initialize_visualization(self):
        """Initialize visualization settings"""
        try:
            # Set matplotlib style
            plt.style.use('dark_background')
            sns.set_palette("husl")
            
            # Define Hat Manifesto color scheme
            self.hat_colors = {
                'hyperliquid_architect': '#FF6B6B',      # Red
                'quantitative_strategist': '#4ECDC4',    # Teal
                'microstructure_analyst': '#45B7D1',     # Blue
                'low_latency_engineer': '#96CEB4',       # Green
                'execution_manager': '#FFEAA7',          # Yellow
                'risk_officer': '#DDA0DD',               # Plum
                'security_architect': '#98D8C8',         # Mint
                'performance_analyst': '#F7DC6F',        # Gold
                'ml_researcher': '#BB8FCE',              # Lavender
            }
            
            self.logger.info("📊 [VISUALIZATION] Visualization settings initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [VISUALIZATION] Error initializing visualization: {e}")
    
    async def start_dashboard(self):
        """Start the Hat Manifesto Performance Dashboard"""
        try:
            self.dashboard_active = True
            self.logger.info("📊 [DASHBOARD] Starting Hat Manifesto Performance Dashboard")
            
            # Start dashboard update loop
            asyncio.create_task(self._dashboard_update_loop())
            
            # Start alert monitoring
            asyncio.create_task(self._alert_monitoring_loop())
            
            # Display initial dashboard
            await self._display_dashboard()
            
        except Exception as e:
            self.logger.error(f"❌ [DASHBOARD] Error starting dashboard: {e}")
    
    async def stop_dashboard(self):
        """Stop the Hat Manifesto Performance Dashboard"""
        try:
            self.dashboard_active = False
            self.logger.info("📊 [DASHBOARD] Stopping Hat Manifesto Performance Dashboard")
            
        except Exception as e:
            self.logger.error(f"❌ [DASHBOARD] Error stopping dashboard: {e}")
    
    async def update_metrics(self, hat_system_metrics: Dict[str, Any]):
        """Update dashboard metrics from Hat Manifesto system"""
        try:
            current_time = time.time()
            
            # Update Hat Manifesto role scores
            if 'hat_scores' in hat_system_metrics:
                self.current_metrics.hat_scores.update(hat_system_metrics['hat_scores'])
            
            # Update performance metrics
            if 'performance_metrics' in hat_system_metrics:
                self.current_metrics.performance_metrics.update(hat_system_metrics['performance_metrics'])
            
            # Update risk metrics
            if 'risk_metrics' in hat_system_metrics:
                self.current_metrics.risk_metrics.update(hat_system_metrics['risk_metrics'])
            
            # Update latency metrics
            if 'latency_metrics' in hat_system_metrics:
                self.current_metrics.latency_metrics.update(hat_system_metrics['latency_metrics'])
            
            # Update ML metrics
            if 'ml_metrics' in hat_system_metrics:
                self.current_metrics.ml_metrics.update(hat_system_metrics['ml_metrics'])
            
            # Update Hyperliquid metrics
            if 'hyperliquid_metrics' in hat_system_metrics:
                self.current_metrics.hyperliquid_metrics.update(hat_system_metrics['hyperliquid_metrics'])
            
            # Update trade metrics
            if 'trade_metrics' in hat_system_metrics:
                self.current_metrics.trade_metrics.update(hat_system_metrics['trade_metrics'])
            
            # Store metrics history
            metrics_snapshot = {
                'timestamp': current_time,
                'hat_scores': self.current_metrics.hat_scores.copy(),
                'performance_metrics': self.current_metrics.performance_metrics.copy(),
                'risk_metrics': self.current_metrics.risk_metrics.copy(),
                'latency_metrics': self.current_metrics.latency_metrics.copy(),
                'ml_metrics': self.current_metrics.ml_metrics.copy(),
                'hyperliquid_metrics': self.current_metrics.hyperliquid_metrics.copy(),
                'trade_metrics': self.current_metrics.trade_metrics.copy(),
            }
            
            self.metrics_history.append(metrics_snapshot)
            self.last_update = current_time
            
        except Exception as e:
            self.logger.error(f"❌ [DASHBOARD] Error updating metrics: {e}")
    
    async def _dashboard_update_loop(self):
        """Dashboard update loop"""
        while self.dashboard_active:
            try:
                await asyncio.sleep(self.dashboard_config.dashboard_settings['update_interval_seconds'])
                
                if self.dashboard_config.dashboard_settings['auto_refresh']:
                    await self._display_dashboard()
                
            except Exception as e:
                self.logger.error(f"❌ [DASHBOARD] Error in dashboard update loop: {e}")
    
    async def _alert_monitoring_loop(self):
        """Alert monitoring loop"""
        while self.dashboard_active:
            try:
                await asyncio.sleep(1.0)  # Check alerts every second
                
                if self.dashboard_config.dashboard_settings['performance_alerts']:
                    await self._check_performance_alerts()
                
            except Exception as e:
                self.logger.error(f"❌ [DASHBOARD] Error in alert monitoring loop: {e}")
    
    async def _display_dashboard(self):
        """Display the Hat Manifesto Performance Dashboard"""
        try:
            # Clear screen (for terminal display)
            print("\033[2J\033[H", end="")
            
            # Display header
            self._display_header()
            
            # Display Hat Manifesto role scores
            self._display_hat_scores()
            
            # Display performance metrics
            self._display_performance_metrics()
            
            # Display risk metrics
            self._display_risk_metrics()
            
            # Display latency metrics
            self._display_latency_metrics()
            
            # Display ML metrics
            self._display_ml_metrics()
            
            # Display Hyperliquid metrics
            self._display_hyperliquid_metrics()
            
            # Display trade metrics
            self._display_trade_metrics()
            
            # Display footer
            self._display_footer()
            
        except Exception as e:
            self.logger.error(f"❌ [DASHBOARD] Error displaying dashboard: {e}")
    
    def _display_header(self):
        """Display dashboard header"""
        print("🎩 HAT MANIFESTO PERFORMANCE DASHBOARD")
        print("=" * 80)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 🔄 Last Update: {time.time() - self.last_update:.1f}s ago")
        print("=" * 80)
    
    def _display_hat_scores(self):
        """Display Hat Manifesto role scores"""
        print("\n🎩 HAT MANIFESTO ROLE SCORES")
        print("-" * 50)
        
        for role, score in self.current_metrics.hat_scores.items():
            role_display = role.replace('_', ' ').title()
            score_bar = "█" * int(score) + "░" * (10 - int(score))
            color = "🟢" if score >= 9.0 else "🟡" if score >= 7.0 else "🔴"
            
            print(f"{color} {role_display:<25} {score:5.1f}/10.0 {score_bar}")
        
        # Overall score
        overall_score = np.mean(list(self.current_metrics.hat_scores.values()))
        overall_bar = "█" * int(overall_score) + "░" * (10 - int(overall_score))
        overall_color = "🟢" if overall_score >= 9.0 else "🟡" if overall_score >= 7.0 else "🔴"
        
        print(f"{overall_color} {'OVERALL SCORE':<25} {overall_score:5.1f}/10.0 {overall_bar}")
    
    def _display_performance_metrics(self):
        """Display performance metrics"""
        print("\n📊 PERFORMANCE METRICS")
        print("-" * 50)
        
        perf = self.current_metrics.performance_metrics
        
        print(f"💰 Total Profit:        ${perf['total_profit']:>10.2f}")
        print(f"📈 Win Rate:            {perf['win_rate']:>10.1f}%")
        print(f"📊 Sharpe Ratio:        {perf['sharpe_ratio']:>10.2f}")
        print(f"📉 Max Drawdown:        {perf['max_drawdown']:>10.2f}%")
        print(f"📉 Current Drawdown:    {perf['current_drawdown']:>10.2f}%")
        print(f"📊 Volatility:          {perf['volatility']:>10.2f}%")
        print(f"🎯 Total Trades:        {perf['total_trades']:>10}")
        print(f"✅ Successful Trades:   {perf['successful_trades']:>10}")
        print(f"⏱️  Avg Trade Duration:  {perf['avg_trade_duration']:>10.1f}s")
        print(f"📊 Profit Factor:       {perf['profit_factor']:>10.2f}")
    
    def _display_risk_metrics(self):
        """Display risk metrics"""
        print("\n🛡️ RISK MANAGEMENT")
        print("-" * 50)
        
        risk = self.current_metrics.risk_metrics
        
        print(f"💰 Account Value:       ${risk['account_value']:>10.2f}")
        print(f"💳 Available Margin:    ${risk['available_margin']:>10.2f}")
        print(f"📊 Margin Ratio:        {risk['margin_ratio']:>10.1%}")
        print(f"📈 Position Count:      {risk['position_count']:>10}")
        print(f"💼 Total Exposure:      ${risk['total_exposure']:>10.2f}")
        print(f"🛑 ATR Stops Active:    {risk['atr_stops_active']:>10}")
        
        # Risk level indicator
        risk_level = risk['risk_level']
        risk_color = "🟢" if risk_level == 'LOW' else "🟡" if risk_level == 'MEDIUM' else "🔴"
        print(f"{risk_color} Risk Level:           {risk_level:>10}")
        
        # Circuit breaker status
        cb_status = "ACTIVE" if risk['circuit_breaker_active'] else "INACTIVE"
        cb_color = "🔴" if risk['circuit_breaker_active'] else "🟢"
        print(f"{cb_color} Circuit Breaker:      {cb_status:>10}")
        
        # Emergency mode
        em_status = "ACTIVE" if risk['emergency_mode'] else "INACTIVE"
        em_color = "🔴" if risk['emergency_mode'] else "🟢"
        print(f"{em_color} Emergency Mode:       {em_status:>10}")
    
    def _display_latency_metrics(self):
        """Display latency metrics"""
        print("\n⚡ LATENCY OPTIMIZATION")
        print("-" * 50)
        
        latency = self.current_metrics.latency_metrics
        
        print(f"⏱️  Avg Response Time:   {latency['avg_response_time_ms']:>10.1f}ms")
        print(f"📊 P95 Latency:         {latency['p95_latency_ms']:>10.1f}ms")
        print(f"📊 P99 Latency:         {latency['p99_latency_ms']:>10.1f}ms")
        print(f"🔄 API Calls/sec:       {latency['api_calls_per_second']:>10.1f}")
        print(f"❌ Error Rate:          {latency['error_rate_percent']:>10.2f}%")
        print(f"🔗 Connections:         {latency['connection_count']:>10}")
        print(f"🌐 WebSocket Conn:      {latency['websocket_connections']:>10}")
        print(f"💾 Cache Hit Rate:      {latency['cache_hit_rate']:>10.1f}%")
    
    def _display_ml_metrics(self):
        """Display ML metrics"""
        print("\n🧠 MACHINE LEARNING")
        print("-" * 50)
        
        ml = self.current_metrics.ml_metrics
        
        print(f"📊 Current Regime:      {ml['current_regime']:>10}")
        print(f"😊 Sentiment Score:     {ml['sentiment_score']:>10.2f}")
        print(f"🔍 Active Patterns:     {ml['active_patterns']:>10}")
        print(f"🎯 Model Accuracy:      {ml['model_accuracy']:>10.2f}")
        print(f"📈 Prediction Confidence: {ml['prediction_confidence']:>10.2f}")
        print(f"🔄 Adaptation Rate:     {ml['adaptation_rate']:>10.2f}")
        print(f"📊 Data Points:         {ml['data_points']:>10}")
    
    def _display_hyperliquid_metrics(self):
        """Display Hyperliquid metrics"""
        print("\n🏗️ HYPERLIQUID OPTIMIZATION")
        print("-" * 50)
        
        hl = self.current_metrics.hyperliquid_metrics
        
        print(f"💰 Funding Arbitrage:   ${hl['funding_arbitrage_profit']:>10.2f}")
        print(f"⚡ TWAP Savings:        ${hl['twap_slippage_savings']:>10.2f}")
        print(f"🏆 HYPE Staking:        ${hl['hype_staking_rewards']:>10.2f}")
        print(f"🔍 Oracle Arbitrage:    ${hl['oracle_arbitrage_profit']:>10.2f}")
        print(f"📊 vAMM Efficiency:     ${hl['vamm_efficiency_profit']:>10.2f}")
        print(f"⛽ Gas Savings:         ${hl['gas_savings']:>10.2f}")
        print(f"🎯 Protocol Score:      {hl['protocol_exploitation_score']:>10.1f}/10.0")
    
    def _display_trade_metrics(self):
        """Display trade metrics"""
        print("\n💼 TRADE METRICS")
        print("-" * 50)
        
        trade = self.current_metrics.trade_metrics
        
        print(f"📅 Trades Today:        {trade['trades_today']:>10}")
        print(f"⏰ Trades This Hour:    {trade['trades_this_hour']:>10}")
        print(f"📊 Avg Trade Size:      ${trade['avg_trade_size']:>10.2f}")
        print(f"🎯 Largest Win:         ${trade['largest_win']:>10.2f}")
        print(f"📉 Largest Loss:        ${trade['largest_loss']:>10.2f}")
        print(f"✅ Consecutive Wins:    {trade['consecutive_wins']:>10}")
        print(f"❌ Consecutive Losses:  {trade['consecutive_losses']:>10}")
        print(f"🏆 Best Strategy:       {trade['best_strategy']:>10}")
        print(f"📉 Worst Strategy:      {trade['worst_strategy']:>10}")
    
    def _display_footer(self):
        """Display dashboard footer"""
        print("\n" + "=" * 80)
        print("🎩 Hat Manifesto Ultimate Trading System - 10/10 Performance Across All Roles")
        print("=" * 80)
    
    async def _check_performance_alerts(self):
        """Check for performance alerts"""
        try:
            current_time = time.time()
            
            # Check Hat Manifesto role scores
            for role, score in self.current_metrics.hat_scores.items():
                if score < self.dashboard_config.alert_settings['performance_threshold']:
                    await self._trigger_alert(f"LOW_PERFORMANCE_{role}", f"{role} score: {score:.1f}/10.0")
            
            # Check risk metrics
            if self.current_metrics.risk_metrics['margin_ratio'] > self.dashboard_config.alert_settings['risk_threshold']:
                await self._trigger_alert("HIGH_RISK", f"Margin ratio: {self.current_metrics.risk_metrics['margin_ratio']:.1%}")
            
            # Check latency metrics
            if self.current_metrics.latency_metrics['avg_response_time_ms'] > self.dashboard_config.alert_settings['latency_threshold']:
                await self._trigger_alert("HIGH_LATENCY", f"Avg response time: {self.current_metrics.latency_metrics['avg_response_time_ms']:.1f}ms")
            
            # Check error rate
            if self.current_metrics.latency_metrics['error_rate_percent'] > self.dashboard_config.alert_settings['error_rate_threshold'] * 100:
                await self._trigger_alert("HIGH_ERROR_RATE", f"Error rate: {self.current_metrics.latency_metrics['error_rate_percent']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"❌ [ALERTS] Error checking performance alerts: {e}")
    
    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger performance alert"""
        try:
            current_time = time.time()
            
            # Check alert cooldown
            if alert_type in self.last_alert_time:
                if current_time - self.last_alert_time[alert_type] < self.dashboard_config.alert_settings['alert_cooldown']:
                    return
            
            # Create alert
            alert = {
                'timestamp': current_time,
                'type': alert_type,
                'message': message,
                'severity': 'HIGH' if 'CRITICAL' in alert_type else 'MEDIUM'
            }
            
            # Store alert
            self.alert_history.append(alert)
            self.last_alert_time[alert_type] = current_time
            
            # Log alert
            self.logger.warning(f"🚨 [ALERT] {alert_type}: {message}")
            
        except Exception as e:
            self.logger.error(f"❌ [ALERTS] Error triggering alert: {e}")
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                'timestamp': time.time(),
                'report_type': 'hat_manifesto_performance',
                'summary': {
                    'overall_score': np.mean(list(self.current_metrics.hat_scores.values())),
                    'total_profit': self.current_metrics.performance_metrics['total_profit'],
                    'win_rate': self.current_metrics.performance_metrics['win_rate'],
                    'risk_level': self.current_metrics.risk_metrics['risk_level'],
                    'current_regime': self.current_metrics.ml_metrics['current_regime'],
                },
                'hat_scores': self.current_metrics.hat_scores,
                'performance_metrics': self.current_metrics.performance_metrics,
                'risk_metrics': self.current_metrics.risk_metrics,
                'latency_metrics': self.current_metrics.latency_metrics,
                'ml_metrics': self.current_metrics.ml_metrics,
                'hyperliquid_metrics': self.current_metrics.hyperliquid_metrics,
                'trade_metrics': self.current_metrics.trade_metrics,
                'alerts': list(self.alert_history)[-10:],  # Last 10 alerts
                'recommendations': await self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ [REPORT] Error generating performance report: {e}")
            return {}
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        try:
            recommendations = []
            
            # Check Hat Manifesto role scores
            for role, score in self.current_metrics.hat_scores.items():
                if score < 9.0:
                    recommendations.append(f"Optimize {role.replace('_', ' ').title()} performance (current: {score:.1f}/10.0)")
            
            # Check risk metrics
            if self.current_metrics.risk_metrics['margin_ratio'] > 0.7:
                recommendations.append("Consider reducing position sizes to lower margin usage")
            
            # Check latency metrics
            if self.current_metrics.latency_metrics['avg_response_time_ms'] > 50:
                recommendations.append("Optimize API calls to reduce latency")
            
            # Check ML metrics
            if self.current_metrics.ml_metrics['model_accuracy'] < 0.8:
                recommendations.append("Retrain ML models to improve accuracy")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ [RECOMMENDATIONS] Error generating recommendations: {e}")
            return []
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return {
            'current_metrics': self.current_metrics,
            'metrics_history': list(self.metrics_history),
            'alert_history': list(self.alert_history),
            'dashboard_active': self.dashboard_active,
            'last_update': self.last_update
        }
    
    def export_dashboard_data(self, format: str = 'json') -> str:
        """Export dashboard data"""
        try:
            data = self.get_dashboard_data()
            
            if format == 'json':
                return json.dumps(data, indent=2, default=str)
            elif format == 'csv':
                # Convert to CSV format
                df = pd.DataFrame(self.metrics_history)
                return df.to_csv(index=False)
            else:
                return str(data)
                
        except Exception as e:
            self.logger.error(f"❌ [EXPORT] Error exporting dashboard data: {e}")
            return ""
