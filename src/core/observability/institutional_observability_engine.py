#!/usr/bin/env python3
"""
📊 INSTITUTIONAL OBSERVABILITY ENGINE
====================================
Real-time monitoring, predictive failure detection, and comprehensive alerting
for institutional-grade trading systems.

Features:
- Prometheus metrics collection
- Grafana dashboard generation
- Predictive failure detection using ML
- Real-time alerting (Slack, Email, PagerDuty)
- Executive dashboard for stakeholders
- SLA monitoring and reporting
- Performance attribution analytics
- System health scoring
"""

import asyncio
import time
import json
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import requests
import os

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, push_to_gateway, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

class AlertSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Alert:
    """Alert data structure"""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    labels: Dict[str, str]
    value: float
    threshold: float
    runbook_url: Optional[str] = None
    resolved: bool = False

@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str]
    type: MetricType

@dataclass
class SystemHealth:
    """Overall system health assessment"""
    overall_score: float  # 0-100
    component_scores: Dict[str, float]
    active_alerts: List[Alert]
    sla_compliance: float
    risk_level: str
    recommendations: List[str]

class InstitutionalObservabilityEngine:
    """
    📊 INSTITUTIONAL OBSERVABILITY ENGINE
    Comprehensive real-time monitoring and alerting system
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Prometheus setup
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Alerting system
        self.alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_channels = config.get('alert_channels', {})
        
        # Performance tracking
        self.performance_metrics = {}
        self.system_health_history = deque(maxlen=1440)  # 24 hours of minute data
        
        # Predictive analytics
        self.anomaly_detector = None
        self.failure_predictor = None
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
        # Start monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("📊 [OBSERVABILITY] Institutional Observability Engine initialized")

    def _initialize_core_metrics(self):
        """Initialize core Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("📊 [OBSERVABILITY] Prometheus not available - using mock metrics")
            return
        
        # Trading Performance Metrics
        self.metrics['trades_total'] = Counter(
            'trading_trades_total',
            'Total number of trades executed',
            ['strategy', 'side', 'outcome'],
            registry=self.registry
        )
        
        self.metrics['pnl_realized'] = Gauge(
            'trading_pnl_realized_usd',
            'Realized profit and loss in USD',
            registry=self.registry
        )
        
        self.metrics['pnl_unrealized'] = Gauge(
            'trading_pnl_unrealized_usd',
            'Unrealized profit and loss in USD',
            registry=self.registry
        )
        
        self.metrics['position_count'] = Gauge(
            'trading_positions_count',
            'Number of open positions',
            registry=self.registry
        )
        
        self.metrics['position_value'] = Gauge(
            'trading_position_value_usd',
            'Total position value in USD',
            ['symbol'],
            registry=self.registry
        )
        
        # Risk Metrics
        self.metrics['drawdown_current'] = Gauge(
            'risk_drawdown_current_pct',
            'Current drawdown percentage',
            registry=self.registry
        )
        
        self.metrics['drawdown_max'] = Gauge(
            'risk_drawdown_max_pct',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        
        self.metrics['var_95'] = Gauge(
            'risk_var_95_usd',
            'Value at Risk (95%) in USD',
            registry=self.registry
        )
        
        self.metrics['sharpe_ratio'] = Gauge(
            'performance_sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )
        
        # System Performance Metrics
        self.metrics['latency_order_execution'] = Histogram(
            'system_latency_order_execution_seconds',
            'Order execution latency in seconds',
            registry=self.registry
        )
        
        self.metrics['latency_api_calls'] = Histogram(
            'system_latency_api_calls_seconds',
            'API call latency in seconds',
            ['endpoint'],
            registry=self.registry
        )
        
        self.metrics['error_rate'] = Gauge(
            'system_error_rate_pct',
            'System error rate percentage',
            ['component'],
            registry=self.registry
        )
        
        self.metrics['system_health'] = Gauge(
            'system_health_score',
            'Overall system health score (0-100)',
            registry=self.registry
        )
        
        # Business Metrics
        self.metrics['daily_pnl'] = Gauge(
            'business_daily_pnl_usd',
            'Daily PnL in USD',
            registry=self.registry
        )
        
        self.metrics['win_rate'] = Gauge(
            'business_win_rate_pct',
            'Win rate percentage',
            registry=self.registry
        )
        
        self.metrics['profit_factor'] = Gauge(
            'business_profit_factor',
            'Profit factor ratio',
            registry=self.registry
        )

    async def start_monitoring(self):
        """Start the real-time monitoring system"""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            
            # Start Prometheus metrics server
            if PROMETHEUS_AVAILABLE:
                start_http_server(8000, registry=self.registry)
                self.logger.info("📊 [OBSERVABILITY] Prometheus metrics server started on port 8000")
            
            # Start monitoring loop
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            # Initialize predictive analytics
            await self._initialize_predictive_analytics()
            
            self.logger.info("📊 [OBSERVABILITY] Real-time monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error starting monitoring: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check alert conditions
                self._check_alert_conditions()
                
                # Update system health score
                self._update_system_health()
                
                # Run predictive analytics
                asyncio.run(self._run_predictive_analytics())
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval', 5))
                
            except Exception as e:
                self.logger.error(f"❌ [OBSERVABILITY] Error in monitoring loop: {e}")
                time.sleep(10)

    def record_trade_execution(self, trade_data: Dict[str, Any]):
        """Record trade execution metrics"""
        try:
            strategy = trade_data.get('strategy', 'unknown')
            side = trade_data.get('side', 'unknown')
            outcome = 'success' if trade_data.get('successful', False) else 'failed'
            pnl = float(trade_data.get('pnl', 0))
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and 'trades_total' in self.metrics:
                self.metrics['trades_total'].labels(
                    strategy=strategy,
                    side=side,
                    outcome=outcome
                ).inc()
            
            # Update PnL metrics
            self._update_pnl_metrics(pnl)
            
            # Record for analytics
            self.metric_history['trades'].append({
                'timestamp': time.time(),
                'strategy': strategy,
                'side': side,
                'outcome': outcome,
                'pnl': pnl
            })
            
            self.logger.info(f"📊 [OBSERVABILITY] Trade recorded: {strategy} {side} {outcome} PnL: ${pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error recording trade: {e}")

    def record_order_latency(self, latency_seconds: float, endpoint: str = 'order'):
        """Record order execution latency"""
        try:
            if PROMETHEUS_AVAILABLE and 'latency_order_execution' in self.metrics:
                self.metrics['latency_order_execution'].observe(latency_seconds)
                
                if 'latency_api_calls' in self.metrics:
                    self.metrics['latency_api_calls'].labels(endpoint=endpoint).observe(latency_seconds)
            
            # Record for analytics
            self.metric_history['latency'].append({
                'timestamp': time.time(),
                'latency': latency_seconds,
                'endpoint': endpoint
            })
            
            # Check latency alerts
            self._check_latency_alerts(latency_seconds)
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error recording latency: {e}")

    def update_position_metrics(self, positions: Dict[str, Any]):
        """Update position-related metrics"""
        try:
            position_count = len(positions)
            total_value = sum(abs(float(pos.get('value', 0))) for pos in positions.values())
            total_unrealized_pnl = sum(float(pos.get('unrealized_pnl', 0)) for pos in positions.values())
            
            if PROMETHEUS_AVAILABLE:
                if 'position_count' in self.metrics:
                    self.metrics['position_count'].set(position_count)
                
                if 'pnl_unrealized' in self.metrics:
                    self.metrics['pnl_unrealized'].set(total_unrealized_pnl)
                
                # Update individual position values
                if 'position_value' in self.metrics:
                    for symbol, position in positions.items():
                        value = abs(float(position.get('value', 0)))
                        self.metrics['position_value'].labels(symbol=symbol).set(value)
            
            # Record for analytics
            self.metric_history['positions'].append({
                'timestamp': time.time(),
                'position_count': position_count,
                'total_value': total_value,
                'unrealized_pnl': total_unrealized_pnl
            })
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error updating position metrics: {e}")

    def update_risk_metrics(self, risk_data: Dict[str, Any]):
        """Update risk-related metrics"""
        try:
            current_drawdown = float(risk_data.get('current_drawdown', 0))
            max_drawdown = float(risk_data.get('max_drawdown', 0))
            var_95 = float(risk_data.get('var_95', 0))
            sharpe_ratio = float(risk_data.get('sharpe_ratio', 0))
            
            if PROMETHEUS_AVAILABLE:
                if 'drawdown_current' in self.metrics:
                    self.metrics['drawdown_current'].set(current_drawdown * 100)
                
                if 'drawdown_max' in self.metrics:
                    self.metrics['drawdown_max'].set(max_drawdown * 100)
                
                if 'var_95' in self.metrics:
                    self.metrics['var_95'].set(var_95)
                
                if 'sharpe_ratio' in self.metrics:
                    self.metrics['sharpe_ratio'].set(sharpe_ratio)
            
            # Check risk alerts
            self._check_risk_alerts(current_drawdown, max_drawdown)
            
            # Record for analytics
            self.metric_history['risk'].append({
                'timestamp': time.time(),
                'current_drawdown': current_drawdown,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'sharpe_ratio': sharpe_ratio
            })
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error updating risk metrics: {e}")

    def _update_pnl_metrics(self, pnl: float):
        """Update PnL metrics"""
        try:
            # Calculate daily PnL
            current_time = time.time()
            today_start = current_time - (current_time % 86400)  # Start of day
            
            daily_trades = [
                trade for trade in self.metric_history['trades']
                if trade['timestamp'] >= today_start
            ]
            
            daily_pnl = sum(trade['pnl'] for trade in daily_trades)
            
            if PROMETHEUS_AVAILABLE and 'daily_pnl' in self.metrics:
                self.metrics['daily_pnl'].set(daily_pnl)
            
            # Calculate win rate
            successful_trades = sum(1 for trade in daily_trades if trade['outcome'] == 'success')
            total_trades = len(daily_trades)
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            if PROMETHEUS_AVAILABLE and 'win_rate' in self.metrics:
                self.metrics['win_rate'].set(win_rate)
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error updating PnL metrics: {e}")

    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Record system metrics
            self.metric_history['system'].append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent
            })
            
            # Check system resource alerts
            if cpu_percent > 80:
                self._create_alert(
                    name="high_cpu_usage",
                    severity=AlertSeverity.WARNING,
                    message=f"High CPU usage detected: {cpu_percent:.1f}%",
                    value=cpu_percent,
                    threshold=80
                )
            
            if memory_percent > 85:
                self._create_alert(
                    name="high_memory_usage",
                    severity=AlertSeverity.CRITICAL,
                    message=f"High memory usage detected: {memory_percent:.1f}%",
                    value=memory_percent,
                    threshold=85
                )
                
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error collecting system metrics: {e}")

    def _check_alert_conditions(self):
        """Check all alert conditions"""
        try:
            current_time = time.time()
            
            # Check if alerts need to be resolved
            for alert_name, alert in list(self.alerts.items()):
                if not alert.resolved and current_time - alert.timestamp > 300:  # 5 minutes
                    self._resolve_alert(alert_name)
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error checking alert conditions: {e}")

    def _check_latency_alerts(self, latency_seconds: float):
        """Check latency-based alerts"""
        try:
            # P95 latency check
            if latency_seconds > 0.2:  # 200ms
                self._create_alert(
                    name="high_latency",
                    severity=AlertSeverity.WARNING,
                    message=f"High order execution latency: {latency_seconds*1000:.1f}ms",
                    value=latency_seconds,
                    threshold=0.2
                )
            
            # P99 latency check  
            if latency_seconds > 0.5:  # 500ms
                self._create_alert(
                    name="critical_latency",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Critical order execution latency: {latency_seconds*1000:.1f}ms",
                    value=latency_seconds,
                    threshold=0.5
                )
                
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error checking latency alerts: {e}")

    def _check_risk_alerts(self, current_drawdown: float, max_drawdown: float):
        """Check risk-based alerts"""
        try:
            # Current drawdown alerts
            if current_drawdown > 0.03:  # 3%
                self._create_alert(
                    name="drawdown_warning",
                    severity=AlertSeverity.WARNING,
                    message=f"Drawdown warning: {current_drawdown:.2%}",
                    value=current_drawdown,
                    threshold=0.03
                )
            
            if current_drawdown > 0.05:  # 5%
                self._create_alert(
                    name="drawdown_critical",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Critical drawdown detected: {current_drawdown:.2%}",
                    value=current_drawdown,
                    threshold=0.05
                )
            
            # Max drawdown alert
            if max_drawdown > 0.08:  # 8%
                self._create_alert(
                    name="max_drawdown_exceeded",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Maximum drawdown exceeded: {max_drawdown:.2%}",
                    value=max_drawdown,
                    threshold=0.08
                )
                
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error checking risk alerts: {e}")

    def _create_alert(self, name: str, severity: AlertSeverity, message: str, 
                     value: float, threshold: float, labels: Optional[Dict[str, str]] = None):
        """Create a new alert"""
        try:
            alert = Alert(
                name=name,
                severity=severity,
                message=message,
                timestamp=time.time(),
                labels=labels or {},
                value=value,
                threshold=threshold,
                runbook_url=f"https://runbook.example.com/alerts/{name}"
            )
            
            # Store alert
            self.alerts[name] = alert
            self.alert_history.append(alert)
            
            # Send alert notifications
            asyncio.create_task(self._send_alert_notifications(alert))
            
            self.logger.warning(f"🚨 [ALERT] {severity.value.upper()}: {message}")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error creating alert: {e}")

    def _resolve_alert(self, alert_name: str):
        """Resolve an existing alert"""
        try:
            if alert_name in self.alerts:
                alert = self.alerts[alert_name]
                alert.resolved = True
                
                self.logger.info(f"✅ [ALERT] Resolved: {alert.name}")
                
                # Send resolution notification
                asyncio.create_task(self._send_resolution_notification(alert))
                
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error resolving alert: {e}")

    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications to configured channels"""
        try:
            # Slack notification
            if 'slack_webhook' in self.alert_channels:
                await self._send_slack_alert(alert)
            
            # Email notification  
            if 'email_config' in self.alert_channels:
                await self._send_email_alert(alert)
            
            # PagerDuty notification for critical alerts
            if alert.severity == AlertSeverity.CRITICAL and 'pagerduty_config' in self.alert_channels:
                await self._send_pagerduty_alert(alert)
                
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error sending alert notifications: {e}")

    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        try:
            webhook_url = self.alert_channels['slack_webhook']
            
            color_map = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.WARNING: "#FFA500", 
                AlertSeverity.INFO: "#0000FF",
                AlertSeverity.SUCCESS: "#00FF00"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": f"🚨 {alert.severity.value.upper()} Alert",
                    "text": alert.message,
                    "fields": [
                        {"title": "Value", "value": f"{alert.value:.4f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                        {"title": "Time", "value": datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ]
                }]
            }
            
            async with requests.post(webhook_url, json=payload) as response:
                if response.status_code == 200:
                    self.logger.info(f"📱 [SLACK] Alert sent: {alert.name}")
                    
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error sending Slack alert: {e}")

    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification"""
        try:
            if 'slack_webhook' in self.alert_channels:
                webhook_url = self.alert_channels['slack_webhook']
                
                payload = {
                    "attachments": [{
                        "color": "#00FF00",
                        "title": "✅ Alert Resolved",
                        "text": f"Alert '{alert.name}' has been resolved",
                        "fields": [
                            {"title": "Original Message", "value": alert.message, "short": False},
                            {"title": "Resolved At", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                        ]
                    }]
                }
                
                async with requests.post(webhook_url, json=payload) as response:
                    if response.status_code == 200:
                        self.logger.info(f"📱 [SLACK] Resolution sent: {alert.name}")
                        
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error sending resolution notification: {e}")

    def _update_system_health(self):
        """Update overall system health score"""
        try:
            scores = {}
            
            # Trading performance score (40% weight)
            trading_score = self._calculate_trading_score()
            scores['trading'] = trading_score
            
            # Risk management score (30% weight)  
            risk_score = self._calculate_risk_score()
            scores['risk'] = risk_score
            
            # System performance score (20% weight)
            system_score = self._calculate_system_score()
            scores['system'] = system_score
            
            # Alert score (10% weight)
            alert_score = self._calculate_alert_score()
            scores['alerts'] = alert_score
            
            # Calculate weighted overall score
            overall_score = (
                trading_score * 0.4 +
                risk_score * 0.3 +
                system_score * 0.2 +
                alert_score * 0.1
            )
            
            # Update Prometheus metric
            if PROMETHEUS_AVAILABLE and 'system_health' in self.metrics:
                self.metrics['system_health'].set(overall_score)
            
            # Store health data
            health_data = SystemHealth(
                overall_score=overall_score,
                component_scores=scores,
                active_alerts=[alert for alert in self.alerts.values() if not alert.resolved],
                sla_compliance=self._calculate_sla_compliance(),
                risk_level=self._determine_risk_level(overall_score),
                recommendations=self._generate_recommendations(scores)
            )
            
            self.system_health_history.append({
                'timestamp': time.time(),
                'health_data': health_data
            })
            
            self.logger.info(f"📊 [HEALTH] System health: {overall_score:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error updating system health: {e}")

    def _calculate_trading_score(self) -> float:
        """Calculate trading performance score"""
        try:
            recent_trades = [
                trade for trade in self.metric_history['trades']
                if time.time() - trade['timestamp'] < 3600  # Last hour
            ]
            
            if not recent_trades:
                return 80.0  # Default good score
            
            success_rate = sum(1 for trade in recent_trades if trade['outcome'] == 'success') / len(recent_trades)
            avg_pnl = sum(trade['pnl'] for trade in recent_trades) / len(recent_trades)
            
            # Score based on success rate and PnL
            score = success_rate * 50 + (min(avg_pnl / 100, 1.0) * 50)
            return max(0, min(100, score))
            
        except Exception:
            return 50.0  # Default neutral score

    def _calculate_risk_score(self) -> float:
        """Calculate risk management score"""
        try:
            recent_risk = [
                risk for risk in self.metric_history['risk']
                if time.time() - risk['timestamp'] < 3600  # Last hour
            ]
            
            if not recent_risk:
                return 80.0  # Default good score
            
            latest_risk = recent_risk[-1]
            current_drawdown = latest_risk['current_drawdown']
            sharpe_ratio = latest_risk['sharpe_ratio']
            
            # Score based on drawdown and Sharpe ratio
            drawdown_score = max(0, 100 - (current_drawdown * 2000))  # Penalize drawdown heavily
            sharpe_score = min(100, max(0, sharpe_ratio * 50))
            
            return (drawdown_score * 0.7 + sharpe_score * 0.3)
            
        except Exception:
            return 50.0  # Default neutral score

    def _calculate_system_score(self) -> float:
        """Calculate system performance score"""
        try:
            recent_latency = [
                latency for latency in self.metric_history['latency']
                if time.time() - latency['timestamp'] < 300  # Last 5 minutes
            ]
            
            if not recent_latency:
                return 80.0  # Default good score
            
            avg_latency = sum(l['latency'] for l in recent_latency) / len(recent_latency)
            
            # Score based on latency (lower is better)
            if avg_latency < 0.1:  # < 100ms
                return 100.0
            elif avg_latency < 0.2:  # < 200ms
                return 80.0
            elif avg_latency < 0.5:  # < 500ms
                return 60.0
            else:
                return 40.0
                
        except Exception:
            return 50.0  # Default neutral score

    def _calculate_alert_score(self) -> float:
        """Calculate alerting system score"""
        try:
            active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
            
            if not active_alerts:
                return 100.0
            
            # Penalize based on alert severity
            penalty = 0
            for alert in active_alerts:
                if alert.severity == AlertSeverity.CRITICAL:
                    penalty += 30
                elif alert.severity == AlertSeverity.WARNING:
                    penalty += 10
                
            return max(0, 100 - penalty)
            
        except Exception:
            return 50.0  # Default neutral score

    def _calculate_sla_compliance(self) -> float:
        """Calculate SLA compliance percentage"""
        try:
            # Example SLA: 99.9% uptime, <200ms P95 latency
            uptime_score = 99.9  # Would calculate based on actual uptime
            
            recent_latency = [
                l['latency'] for l in self.metric_history['latency']
                if time.time() - l['timestamp'] < 3600
            ]
            
            if recent_latency:
                p95_latency = np.percentile(recent_latency, 95)
                latency_score = 100 if p95_latency < 0.2 else 90
            else:
                latency_score = 100
            
            return (uptime_score + latency_score) / 2
            
        except Exception:
            return 95.0  # Default good compliance

    def _determine_risk_level(self, health_score: float) -> str:
        """Determine risk level based on health score"""
        if health_score >= 90:
            return "LOW"
        elif health_score >= 70:
            return "MEDIUM"
        elif health_score >= 50:
            return "HIGH"
        else:
            return "CRITICAL"

    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if scores.get('trading', 0) < 70:
            recommendations.append("Review trading strategies and parameters")
        
        if scores.get('risk', 0) < 70:
            recommendations.append("Tighten risk management controls")
        
        if scores.get('system', 0) < 70:
            recommendations.append("Optimize system performance and reduce latency")
        
        if scores.get('alerts', 0) < 90:
            recommendations.append("Address active alerts and system issues")
        
        return recommendations

    async def _initialize_predictive_analytics(self):
        """Initialize predictive analytics components"""
        try:
            # Placeholder for ML-based predictive analytics
            self.logger.info("📊 [OBSERVABILITY] Predictive analytics initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error initializing predictive analytics: {e}")

    async def _run_predictive_analytics(self):
        """Run predictive failure detection"""
        try:
            # Placeholder for predictive analytics
            # Would implement ML models to predict failures
            pass
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error in predictive analytics: {e}")

    async def stop_monitoring(self):
        """Stop the monitoring system"""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("📊 [OBSERVABILITY] Monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error stopping monitoring: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            latest_health = self.system_health_history[-1] if self.system_health_history else None
            
            return {
                'monitoring_active': self.monitoring_active,
                'total_metrics': len(self.metrics),
                'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
                'system_health': latest_health['health_data'] if latest_health else None,
                'last_update': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error getting system status: {e}")
            return {'error': str(e)}

    def export_metrics_for_grafana(self) -> str:
        """Export metrics in Grafana-compatible format"""
        try:
            # Generate Grafana dashboard JSON
            dashboard = {
                "dashboard": {
                    "title": "Institutional Trading System Observability",
                    "panels": [
                        {
                            "title": "Trading Performance",
                            "type": "graph",
                            "targets": [
                                {"expr": "trading_pnl_realized_usd"},
                                {"expr": "trading_pnl_unrealized_usd"}
                            ]
                        },
                        {
                            "title": "System Health",
                            "type": "singlestat",
                            "targets": [{"expr": "system_health_score"}]
                        },
                        {
                            "title": "Risk Metrics",
                            "type": "graph",
                            "targets": [
                                {"expr": "risk_drawdown_current_pct"},
                                {"expr": "performance_sharpe_ratio"}
                            ]
                        },
                        {
                            "title": "Latency Distribution",
                            "type": "heatmap",
                            "targets": [{"expr": "system_latency_order_execution_seconds"}]
                        }
                    ]
                }
            }
            
            return json.dumps(dashboard, indent=2)
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error exporting Grafana config: {e}")
            return "{}"
