#!/usr/bin/env python3
"""
📊 OBSERVABILITY ENGINEER (Prometheus, Grafana)
===============================================

Comprehensive observability system providing real-time monitoring, predictive failure detection,
and performance metrics for the trading system. Implements Prometheus/Grafana-style monitoring
with advanced alerting and failure prediction capabilities.

Features:
- Real-time performance metrics
- Predictive failure detection
- Comprehensive alerting system
- Performance dashboards
- System health monitoring
- Latency tracking
- Error rate monitoring
- Resource utilization tracking
"""

import time
import logging
import json
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import numpy as np

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str]
    type: str  # 'counter', 'gauge', 'histogram', 'summary'

@dataclass
class Alert:
    """Alert data structure"""
    name: str
    severity: str  # 'info', 'warning', 'critical'
    message: str
    timestamp: float
    status: str  # 'firing', 'resolved'
    labels: Dict[str, str]

@dataclass
class SystemHealth:
    """System health status"""
    overall_status: str  # 'healthy', 'degraded', 'critical'
    components: Dict[str, str]
    last_check: float
    uptime: float
    error_rate: float
    latency_p95: float
    memory_usage: float
    cpu_usage: float

class ObservabilityEngine:
    """Advanced observability system with predictive monitoring"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.alerts = deque(maxlen=1000)
        self.health_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_metrics = {
            'trading_performance': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0
            },
            'system_performance': {
                'uptime': 0.0,
                'error_rate': 0.0,
                'latency_p50': 0.0,
                'latency_p95': 0.0,
                'latency_p99': 0.0,
                'throughput': 0.0
            },
            'risk_metrics': {
                'current_drawdown': 0.0,
                'var_95': 0.0,
                'volatility': 0.0,
                'leverage_ratio': 0.0,
                'concentration_risk': 0.0
            }
        }
        
        # Alerting configuration
        self.alert_rules = {
            'high_drawdown': {
                'threshold': 0.05,  # 5% drawdown
                'severity': 'critical',
                'message': 'High drawdown detected: {value:.2%}'
            },
            'high_error_rate': {
                'threshold': 0.05,  # 5% error rate
                'severity': 'warning',
                'message': 'High error rate detected: {value:.2%}'
            },
            'high_latency': {
                'threshold': 1.0,  # 1 second
                'severity': 'warning',
                'message': 'High latency detected: {value:.2f}s'
            },
            'low_win_rate': {
                'threshold': 0.4,  # 40% win rate
                'severity': 'warning',
                'message': 'Low win rate detected: {value:.2%}'
            },
            'system_down': {
                'threshold': 300,  # 5 minutes
                'severity': 'critical',
                'message': 'System appears to be down for {value:.0f}s'
            }
        }
        
        # Predictive monitoring
        self.failure_prediction = {
            'enabled': True,
            'prediction_window': 300,  # 5 minutes
            'confidence_threshold': 0.8,
            'anomaly_threshold': 3.0  # Standard deviations
        }
        
        # Monitoring intervals
        self.monitoring_config = {
            'metrics_collection_interval': 1.0,  # 1 second
            'health_check_interval': 5.0,  # 5 seconds
            'alert_check_interval': 10.0,  # 10 seconds
            'performance_analysis_interval': 60.0,  # 1 minute
            'failure_prediction_interval': 30.0,  # 30 seconds
            'dashboard_update_interval': 5.0  # 5 seconds
        }
        
        # State tracking
        self.start_time = time.time()
        self.is_running = False
        self.monitoring_threads = {}
        self.last_metrics_update = time.time()
        self.last_health_check = time.time()
        self.last_alert_check = time.time()
        
        # Anomaly detection
        self.anomaly_detectors = {}
        self.baseline_metrics = {}
        
        self.logger.info("📊 [OBSERVABILITY] Observability Engine initialized")
        self.logger.info(f"📊 [OBSERVABILITY] Alert rules: {len(self.alert_rules)} configured")
        self.logger.info(f"📊 [OBSERVABILITY] Failure prediction: {'enabled' if self.failure_prediction['enabled'] else 'disabled'}")
    
    def start_monitoring(self) -> None:
        """Start all monitoring threads"""
        try:
            self.is_running = True
            
            # Start metrics collection thread
            self.monitoring_threads['metrics_collection'] = threading.Thread(
                target=self._metrics_collection_loop,
                daemon=True
            )
            self.monitoring_threads['metrics_collection'].start()
            
            # Start health check thread
            self.monitoring_threads['health_check'] = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self.monitoring_threads['health_check'].start()
            
            # Start alert checking thread
            self.monitoring_threads['alert_check'] = threading.Thread(
                target=self._alert_check_loop,
                daemon=True
            )
            self.monitoring_threads['alert_check'].start()
            
            # Start performance analysis thread
            self.monitoring_threads['performance_analysis'] = threading.Thread(
                target=self._performance_analysis_loop,
                daemon=True
            )
            self.monitoring_threads['performance_analysis'].start()
            
            # Start failure prediction thread
            if self.failure_prediction['enabled']:
                self.monitoring_threads['failure_prediction'] = threading.Thread(
                    target=self._failure_prediction_loop,
                    daemon=True
                )
                self.monitoring_threads['failure_prediction'].start()
            
            self.logger.info("📊 [OBSERVABILITY] All monitoring threads started")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error starting monitoring: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring threads"""
        try:
            self.is_running = False
            
            # Wait for threads to finish
            for thread_name, thread in self.monitoring_threads.items():
                if thread.is_alive():
                    thread.join(timeout=5.0)
                    self.logger.info(f"📊 [OBSERVABILITY] Thread {thread_name} stopped")
            
            self.logger.info("📊 [OBSERVABILITY] All monitoring threads stopped")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error stopping monitoring: {e}")
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None, 
                     metric_type: str = 'gauge') -> None:
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                type=metric_type
            )
            
            self.metrics[name].append(metric)
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error recording metric {name}: {e}")
    
    def record_trading_metric(self, metric_name: str, value: float) -> None:
        """Record a trading-specific metric"""
        try:
            labels = {'component': 'trading', 'metric_type': 'performance'}
            self.record_metric(f"trading_{metric_name}", value, labels)
            
            # Update performance metrics
            if metric_name in self.performance_metrics['trading_performance']:
                self.performance_metrics['trading_performance'][metric_name] = value
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error recording trading metric {metric_name}: {e}")
    
    def record_system_metric(self, metric_name: str, value: float) -> None:
        """Record a system-specific metric"""
        try:
            labels = {'component': 'system', 'metric_type': 'performance'}
            self.record_metric(f"system_{metric_name}", value, labels)
            
            # Update performance metrics
            if metric_name in self.performance_metrics['system_performance']:
                self.performance_metrics['system_performance'][metric_name] = value
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error recording system metric {metric_name}: {e}")
    
    def record_risk_metric(self, metric_name: str, value: float) -> None:
        """Record a risk-specific metric"""
        try:
            labels = {'component': 'risk', 'metric_type': 'risk'}
            self.record_metric(f"risk_{metric_name}", value, labels)
            
            # Update performance metrics
            if metric_name in self.performance_metrics['risk_metrics']:
                self.performance_metrics['risk_metrics'][metric_name] = value
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error recording risk metric {metric_name}: {e}")
    
    def create_alert(self, name: str, severity: str, message: str, 
                    labels: Dict[str, str] = None) -> None:
        """Create an alert"""
        try:
            alert = Alert(
                name=name,
                severity=severity,
                message=message,
                timestamp=time.time(),
                status='firing',
                labels=labels or {}
            )
            
            self.alerts.append(alert)
            
            # Log alert
            if severity == 'critical':
                self.logger.critical(f"🚨 [OBSERVABILITY] CRITICAL ALERT: {name} - {message}")
            elif severity == 'warning':
                self.logger.warning(f"⚠️ [OBSERVABILITY] WARNING ALERT: {name} - {message}")
            else:
                self.logger.info(f"ℹ️ [OBSERVABILITY] INFO ALERT: {name} - {message}")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error creating alert {name}: {e}")
    
    def get_metrics(self, metric_name: str, time_window: float = 3600) -> List[Metric]:
        """Get metrics for a specific name within a time window"""
        try:
            current_time = time.time()
            cutoff_time = current_time - time_window
            
            if metric_name not in self.metrics:
                return []
            
            return [
                metric for metric in self.metrics[metric_name]
                if metric.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error getting metrics for {metric_name}: {e}")
            return []
    
    def get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """Get the latest metric for a specific name"""
        try:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None
            
            return self.metrics[metric_name][-1]
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error getting latest metric for {metric_name}: {e}")
            return None
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (firing) alerts"""
        try:
            return [alert for alert in self.alerts if alert.status == 'firing']
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error getting active alerts: {e}")
            return []
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate error rate
            error_metrics = self.get_metrics('system_error_rate', 300)  # Last 5 minutes
            error_rate = np.mean([m.value for m in error_metrics]) if error_metrics else 0.0
            
            # Calculate latency
            latency_metrics = self.get_metrics('system_latency', 300)  # Last 5 minutes
            if latency_metrics:
                latencies = [m.value for m in latency_metrics]
                latency_p95 = np.percentile(latencies, 95)
            else:
                latency_p95 = 0.0
            
            # Determine overall status
            if error_rate > 0.1 or latency_p95 > 5.0:  # 10% error rate or 5s latency
                overall_status = 'critical'
            elif error_rate > 0.05 or latency_p95 > 2.0:  # 5% error rate or 2s latency
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'
            
            # Component status (simplified)
            components = {
                'trading_engine': 'healthy' if error_rate < 0.05 else 'degraded',
                'risk_management': 'healthy' if error_rate < 0.05 else 'degraded',
                'data_feeds': 'healthy' if latency_p95 < 1.0 else 'degraded',
                'order_execution': 'healthy' if latency_p95 < 2.0 else 'degraded'
            }
            
            health = SystemHealth(
                overall_status=overall_status,
                components=components,
                last_check=current_time,
                uptime=uptime,
                error_rate=error_rate,
                latency_p95=latency_p95,
                memory_usage=0.0,  # Would need system monitoring
                cpu_usage=0.0  # Would need system monitoring
            )
            
            self.health_history.append(health)
            return health
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error getting system health: {e}")
            return self._empty_system_health()
    
    def predict_failures(self) -> List[Dict[str, Any]]:
        """Predict potential system failures"""
        try:
            predictions = []
            current_time = time.time()
            
            # Check for anomalies in key metrics
            key_metrics = [
                'trading_win_rate',
                'risk_current_drawdown',
                'system_error_rate',
                'system_latency'
            ]
            
            for metric_name in key_metrics:
                metrics = self.get_metrics(metric_name, 3600)  # Last hour
                if len(metrics) < 10:
                    continue
                
                values = [m.value for m in metrics]
                
                # Calculate baseline (mean of first 80% of data)
                baseline_size = int(len(values) * 0.8)
                baseline = np.mean(values[:baseline_size])
                baseline_std = np.std(values[:baseline_size])
                
                # Check recent values for anomalies
                recent_values = values[baseline_size:]
                if recent_values:
                    recent_mean = np.mean(recent_values)
                    anomaly_score = abs(recent_mean - baseline) / baseline_std if baseline_std > 0 else 0
                    
                    if anomaly_score > self.failure_prediction['anomaly_threshold']:
                        confidence = min(anomaly_score / 5.0, 1.0)  # Normalize to 0-1
                        
                        if confidence >= self.failure_prediction['confidence_threshold']:
                            predictions.append({
                                'metric': metric_name,
                                'anomaly_score': anomaly_score,
                                'confidence': confidence,
                                'baseline': baseline,
                                'current_value': recent_mean,
                                'prediction': f"Anomaly detected in {metric_name}",
                                'timestamp': current_time
                            })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error predicting failures: {e}")
            return []
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard data for real-time monitoring"""
        try:
            current_time = time.time()
            
            # Get latest metrics
            latest_metrics = {}
            for metric_name in self.metrics.keys():
                latest = self.get_latest_metric(metric_name)
                if latest:
                    latest_metrics[metric_name] = {
                        'value': latest.value,
                        'timestamp': latest.timestamp,
                        'labels': latest.labels
                    }
            
            # Get system health
            health = self.get_system_health()
            
            # Get active alerts
            active_alerts = self.get_active_alerts()
            
            # Get failure predictions
            failure_predictions = self.predict_failures()
            
            # Performance summary
            performance_summary = {
                'uptime': health.uptime,
                'overall_status': health.overall_status,
                'error_rate': health.error_rate,
                'latency_p95': health.latency_p95,
                'active_alerts': len(active_alerts),
                'failure_predictions': len(failure_predictions)
            }
            
            return {
                'timestamp': current_time,
                'performance_summary': performance_summary,
                'latest_metrics': latest_metrics,
                'system_health': asdict(health),
                'active_alerts': [asdict(alert) for alert in active_alerts],
                'failure_predictions': failure_predictions,
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error generating dashboard data: {e}")
            return {}
    
    def _metrics_collection_loop(self) -> None:
        """Metrics collection loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Record system metrics
                self.record_system_metric('uptime', current_time - self.start_time)
                
                # Record performance metrics
                if self.performance_metrics['trading_performance']['total_trades'] > 0:
                    win_rate = (self.performance_metrics['trading_performance']['winning_trades'] / 
                              self.performance_metrics['trading_performance']['total_trades'])
                    self.record_trading_metric('win_rate', win_rate)
                
                # Record risk metrics
                self.record_risk_metric('current_drawdown', 
                                      self.performance_metrics['risk_metrics']['current_drawdown'])
                
                self.last_metrics_update = current_time
                
                # Sleep for collection interval
                time.sleep(self.monitoring_config['metrics_collection_interval'])
                
            except Exception as e:
                self.logger.error(f"❌ [OBSERVABILITY] Error in metrics collection loop: {e}")
                time.sleep(5.0)
    
    def _health_check_loop(self) -> None:
        """Health check loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Perform health check
                health = self.get_system_health()
                
                # Check for system down condition
                if current_time - self.last_metrics_update > self.alert_rules['system_down']['threshold']:
                    self.create_alert(
                        'system_down',
                        'critical',
                        self.alert_rules['system_down']['message'].format(
                            value=current_time - self.last_metrics_update
                        )
                    )
                
                self.last_health_check = current_time
                
                # Sleep for health check interval
                time.sleep(self.monitoring_config['health_check_interval'])
                
            except Exception as e:
                self.logger.error(f"❌ [OBSERVABILITY] Error in health check loop: {e}")
                time.sleep(5.0)
    
    def _alert_check_loop(self) -> None:
        """Alert checking loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check alert rules
                for alert_name, rule in self.alert_rules.items():
                    metric_name = f"risk_{alert_name.split('_', 1)[1]}"  # Map to metric name
                    latest_metric = self.get_latest_metric(metric_name)
                    
                    if latest_metric and latest_metric.value > rule['threshold']:
                        # Check if alert is already firing
                        active_alerts = [a for a in self.alerts if a.name == alert_name and a.status == 'firing']
                        
                        if not active_alerts:
                            self.create_alert(
                                alert_name,
                                rule['severity'],
                                rule['message'].format(value=latest_metric.value),
                                {'metric': metric_name, 'threshold': rule['threshold']}
                            )
                
                self.last_alert_check = current_time
                
                # Sleep for alert check interval
                time.sleep(self.monitoring_config['alert_check_interval'])
                
            except Exception as e:
                self.logger.error(f"❌ [OBSERVABILITY] Error in alert check loop: {e}")
                time.sleep(5.0)
    
    def _performance_analysis_loop(self) -> None:
        """Performance analysis loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Analyze performance trends
                self._analyze_performance_trends()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for performance analysis interval
                time.sleep(self.monitoring_config['performance_analysis_interval'])
                
            except Exception as e:
                self.logger.error(f"❌ [OBSERVABILITY] Error in performance analysis loop: {e}")
                time.sleep(5.0)
    
    def _failure_prediction_loop(self) -> None:
        """Failure prediction loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Predict failures
                predictions = self.predict_failures()
                
                # Create alerts for high-confidence predictions
                for prediction in predictions:
                    if prediction['confidence'] >= self.failure_prediction['confidence_threshold']:
                        self.create_alert(
                            'failure_prediction',
                            'warning',
                            f"Failure predicted: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})",
                            {'metric': prediction['metric'], 'confidence': prediction['confidence']}
                        )
                
                # Sleep for failure prediction interval
                time.sleep(self.monitoring_config['failure_prediction_interval'])
                
            except Exception as e:
                self.logger.error(f"❌ [OBSERVABILITY] Error in failure prediction loop: {e}")
                time.sleep(5.0)
    
    def _analyze_performance_trends(self) -> None:
        """Analyze performance trends"""
        try:
            # Analyze trading performance trends
            win_rate_metrics = self.get_metrics('trading_win_rate', 3600)  # Last hour
            if len(win_rate_metrics) > 10:
                recent_win_rate = np.mean([m.value for m in win_rate_metrics[-10:]])
                if recent_win_rate < 0.4:  # 40% win rate
                    self.create_alert(
                        'low_win_rate_trend',
                        'warning',
                        f"Declining win rate trend: {recent_win_rate:.2%}"
                    )
            
            # Analyze drawdown trends
            drawdown_metrics = self.get_metrics('risk_current_drawdown', 3600)  # Last hour
            if len(drawdown_metrics) > 10:
                recent_drawdown = np.mean([m.value for m in drawdown_metrics[-10:]])
                if recent_drawdown > 0.03:  # 3% drawdown
                    self.create_alert(
                        'increasing_drawdown_trend',
                        'warning',
                        f"Increasing drawdown trend: {recent_drawdown:.2%}"
                    )
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error analyzing performance trends: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        try:
            # Update trading performance metrics
            total_trades = len(self.get_metrics('trading_total_trades', 86400))  # Last 24 hours
            if total_trades > 0:
                self.performance_metrics['trading_performance']['total_trades'] = total_trades
            
            # Update system performance metrics
            error_metrics = self.get_metrics('system_error_rate', 300)  # Last 5 minutes
            if error_metrics:
                error_rate = np.mean([m.value for m in error_metrics])
                self.performance_metrics['system_performance']['error_rate'] = error_rate
            
            # Update risk metrics
            drawdown_metric = self.get_latest_metric('risk_current_drawdown')
            if drawdown_metric:
                self.performance_metrics['risk_metrics']['current_drawdown'] = drawdown_metric.value
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error updating performance metrics: {e}")
    
    def _empty_system_health(self) -> SystemHealth:
        """Return empty system health"""
        return SystemHealth(
            overall_status='unknown',
            components={},
            last_check=time.time(),
            uptime=0.0,
            error_rate=0.0,
            latency_p95=0.0,
            memory_usage=0.0,
            cpu_usage=0.0
        )
    
    def log_observability_status(self) -> None:
        """Log current observability status"""
        try:
            dashboard_data = self.generate_dashboard_data()
            
            self.logger.info("📊 [OBSERVABILITY] === OBSERVABILITY STATUS ===")
            self.logger.info(f"📊 [OBSERVABILITY] Overall Status: {dashboard_data.get('performance_summary', {}).get('overall_status', 'unknown')}")
            self.logger.info(f"📊 [OBSERVABILITY] Uptime: {dashboard_data.get('performance_summary', {}).get('uptime', 0):.0f}s")
            self.logger.info(f"📊 [OBSERVABILITY] Error Rate: {dashboard_data.get('performance_summary', {}).get('error_rate', 0):.2%}")
            self.logger.info(f"📊 [OBSERVABILITY] Latency P95: {dashboard_data.get('performance_summary', {}).get('latency_p95', 0):.2f}s")
            self.logger.info(f"📊 [OBSERVABILITY] Active Alerts: {dashboard_data.get('performance_summary', {}).get('active_alerts', 0)}")
            self.logger.info(f"📊 [OBSERVABILITY] Failure Predictions: {dashboard_data.get('performance_summary', {}).get('failure_predictions', 0)}")
            self.logger.info(f"📊 [OBSERVABILITY] Metrics Tracked: {len(self.metrics)}")
            self.logger.info("📊 [OBSERVABILITY] ================================")
            
        except Exception as e:
            self.logger.error(f"❌ [OBSERVABILITY] Error logging observability status: {e}")
