#!/usr/bin/env python3
"""
📊 ULTIMATE PERFORMANCE MONITOR
"Comprehensive monitoring and optimization for 10/10 performance."

This module implements:
- Real-time performance tracking across all 9 specialized roles
- Advanced analytics and trend analysis
- Performance prediction and optimization recommendations
- Comprehensive reporting and alerting
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
import json
from collections import deque
import statistics

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    overall_score: float
    system_health: float
    total_profit: float
    win_rate: float
    hat_scores: Dict[str, float]
    active_trades: int
    total_trades: int
    confidence: float
    position_size: float

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_type: str  # 'critical', 'warning', 'info'
    message: str
    timestamp: datetime
    hat_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None

class UltimatePerformanceMonitor:
    """
    Ultimate Performance Monitor - Comprehensive 10/10 Performance Tracking
    
    Features:
    1. Real-time performance monitoring
    2. Advanced analytics and trend analysis
    3. Performance prediction and optimization
    4. Comprehensive alerting system
    5. Performance reporting and insights
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Monitor configuration
        self.monitor_config = {
            'snapshot_frequency': 10,  # cycles
            'history_size': 1000,
            'alert_thresholds': {
                'critical_score': 5.0,
                'warning_score': 7.0,
                'optimal_score': 9.0,
                'max_drawdown': 0.10,
                'min_win_rate': 0.60
            },
            'performance_targets': {
                'overall_score': 10.0,
                'system_health': 1.0,
                'daily_profit_target': 0.05,
                'win_rate_target': 0.85
            }
        }
        
        # Data storage
        self.performance_history = deque(maxlen=self.monitor_config['history_size'])
        self.performance_alerts = deque(maxlen=100)
        self.performance_trends = {}
        self.performance_insights = {}
        
        # Performance tracking
        self.peak_performance = 0.0
        self.peak_profit = 0.0
        self.performance_metrics = {}
        self.optimization_recommendations = []
        
        # Monitoring state
        self.monitoring_active = True
        self.last_snapshot = None
        self.performance_baseline = {}
        
        self.logger.info("📊 [ULTIMATE_MONITOR] Ultimate Performance Monitor initialized")
        self.logger.info(f"📊 [ULTIMATE_MONITOR] Snapshot frequency: {self.monitor_config['snapshot_frequency']} cycles")
        self.logger.info(f"📊 [ULTIMATE_MONITOR] History size: {self.monitor_config['history_size']}")
    
    def capture_performance_snapshot(self, metrics: Dict[str, Any]) -> PerformanceSnapshot:
        """Capture a performance snapshot"""
        try:
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                overall_score=metrics.get('overall_score', 0.0),
                system_health=metrics.get('system_health', 0.0),
                total_profit=metrics.get('total_profit', 0.0),
                win_rate=metrics.get('win_rate', 0.0),
                hat_scores=metrics.get('hat_scores', {}),
                active_trades=metrics.get('active_trades', 0),
                total_trades=metrics.get('total_trades', 0),
                confidence=metrics.get('confidence', 0.0),
                position_size=metrics.get('position_size', 0.0)
            )
            
            # Store snapshot
            self.performance_history.append(snapshot)
            self.last_snapshot = snapshot
            
            # Update peaks
            if snapshot.overall_score > self.peak_performance:
                self.peak_performance = snapshot.overall_score
            if snapshot.total_profit > self.peak_profit:
                self.peak_profit = snapshot.total_profit
            
            # Analyze performance
            self._analyze_performance(snapshot)
            
            # Check for alerts
            self._check_performance_alerts(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Error capturing snapshot: {e}")
            return None
    
    def _analyze_performance(self, snapshot: PerformanceSnapshot):
        """Analyze performance and generate insights"""
        try:
            # Calculate performance trends
            if len(self.performance_history) >= 10:
                recent_snapshots = list(self.performance_history)[-10:]
                
                # Overall score trend
                recent_scores = [s.overall_score for s in recent_snapshots]
                score_trend = self._calculate_trend(recent_scores)
                
                # Profit trend
                recent_profits = [s.total_profit for s in recent_snapshots]
                profit_trend = self._calculate_trend(recent_profits)
                
                # Win rate trend
                recent_win_rates = [s.win_rate for s in recent_snapshots]
                win_rate_trend = self._calculate_trend(recent_win_rates)
                
                self.performance_trends = {
                    'score_trend': score_trend,
                    'profit_trend': profit_trend,
                    'win_rate_trend': win_rate_trend,
                    'trend_strength': abs(score_trend) + abs(profit_trend) + abs(win_rate_trend)
                }
            
            # Generate performance insights
            self._generate_performance_insights(snapshot)
            
            # Generate optimization recommendations
            self._generate_optimization_recommendations(snapshot)
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Error analyzing performance: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and strength"""
        try:
            if len(values) < 2:
                return 0.0
            
            # Simple linear trend calculation
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize trend
            trend = slope / (max(values) - min(values)) if max(values) != min(values) else 0.0
            
            return trend
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Error calculating trend: {e}")
            return 0.0
    
    def _generate_performance_insights(self, snapshot: PerformanceSnapshot):
        """Generate performance insights"""
        try:
            insights = []
            
            # Overall performance insight
            if snapshot.overall_score >= 9.5:
                insights.append("🏆 EXCELLENT: System achieving near-perfect performance")
            elif snapshot.overall_score >= 8.5:
                insights.append("🟢 GOOD: System performing well with room for optimization")
            elif snapshot.overall_score >= 7.0:
                insights.append("🟡 FAIR: System needs optimization to reach peak performance")
            else:
                insights.append("🔴 POOR: System requires immediate optimization")
            
            # Profit insight
            if snapshot.total_profit > 0.05:  # 5% profit
                insights.append(f"💰 PROFITABLE: Total profit at {snapshot.total_profit*100:.2f}%")
            elif snapshot.total_profit > 0:
                insights.append(f"📈 POSITIVE: Small profit at {snapshot.total_profit*100:.2f}%")
            else:
                insights.append("📉 LOSS: System currently at a loss")
            
            # Win rate insight
            if snapshot.win_rate >= 0.8:
                insights.append(f"🎯 HIGH WIN RATE: {snapshot.win_rate*100:.1f}% success rate")
            elif snapshot.win_rate >= 0.6:
                insights.append(f"📊 MODERATE WIN RATE: {snapshot.win_rate*100:.1f}% success rate")
            else:
                insights.append(f"⚠️ LOW WIN RATE: {snapshot.win_rate*100:.1f}% success rate")
            
            # Hat performance insights
            top_hats = sorted(snapshot.hat_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            bottom_hats = sorted(snapshot.hat_scores.items(), key=lambda x: x[1])[:3]
            
            if top_hats:
                insights.append(f"⭐ TOP PERFORMERS: {', '.join([f'{hat}: {score:.1f}' for hat, score in top_hats])}")
            
            if bottom_hats:
                insights.append(f"🔧 NEEDS OPTIMIZATION: {', '.join([f'{hat}: {score:.1f}' for hat, score in bottom_hats])}")
            
            self.performance_insights = {
                'timestamp': snapshot.timestamp,
                'insights': insights,
                'overall_assessment': self._get_overall_assessment(snapshot)
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Error generating insights: {e}")
    
    def _get_overall_assessment(self, snapshot: PerformanceSnapshot) -> str:
        """Get overall performance assessment"""
        try:
            score = snapshot.overall_score
            profit = snapshot.total_profit
            win_rate = snapshot.win_rate
            
            if score >= 9.5 and profit > 0.02 and win_rate >= 0.8:
                return "🏆 EXCEPTIONAL: System operating at peak performance"
            elif score >= 8.5 and profit > 0 and win_rate >= 0.7:
                return "🟢 EXCELLENT: System performing very well"
            elif score >= 7.5 and win_rate >= 0.6:
                return "🟡 GOOD: System performing adequately with optimization potential"
            elif score >= 6.0:
                return "🟠 FAIR: System needs optimization"
            else:
                return "🔴 POOR: System requires immediate attention"
                
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Error getting assessment: {e}")
            return "❓ UNKNOWN: Unable to assess performance"
    
    def _generate_optimization_recommendations(self, snapshot: PerformanceSnapshot):
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            # Overall score recommendations
            if snapshot.overall_score < 8.0:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Overall Performance',
                    'recommendation': 'Focus on optimizing underperforming hats',
                    'expected_improvement': 1.0
                })
            
            # Individual hat recommendations
            for hat_name, score in snapshot.hat_scores.items():
                if score < 7.0:
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': f'{hat_name.replace("_", " ").title()}',
                        'recommendation': f'Optimize {hat_name} algorithms and parameters',
                        'expected_improvement': 10.0 - score
                    })
                elif score < 8.5:
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': f'{hat_name.replace("_", " ").title()}',
                        'recommendation': f'Fine-tune {hat_name} performance',
                        'expected_improvement': 10.0 - score
                    })
            
            # Profit optimization recommendations
            if snapshot.total_profit < 0.01:  # Less than 1% profit
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Profitability',
                    'recommendation': 'Increase trading frequency and position sizing',
                    'expected_improvement': 0.02
                })
            
            # Win rate optimization recommendations
            if snapshot.win_rate < 0.7:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Win Rate',
                    'recommendation': 'Improve signal quality and risk management',
                    'expected_improvement': 0.1
                })
            
            self.optimization_recommendations = recommendations
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Error generating recommendations: {e}")
    
    def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance alerts"""
        try:
            alerts = []
            thresholds = self.monitor_config['alert_thresholds']
            
            # Overall score alerts
            if snapshot.overall_score < thresholds['critical_score']:
                alerts.append(PerformanceAlert(
                    alert_type='critical',
                    message=f"Critical performance: {snapshot.overall_score:.1f}/10",
                    timestamp=snapshot.timestamp,
                    current_value=snapshot.overall_score,
                    threshold=thresholds['critical_score']
                ))
            elif snapshot.overall_score < thresholds['warning_score']:
                alerts.append(PerformanceAlert(
                    alert_type='warning',
                    message=f"Performance warning: {snapshot.overall_score:.1f}/10",
                    timestamp=snapshot.timestamp,
                    current_value=snapshot.overall_score,
                    threshold=thresholds['warning_score']
                ))
            
            # Win rate alerts
            if snapshot.win_rate < thresholds['min_win_rate']:
                alerts.append(PerformanceAlert(
                    alert_type='warning',
                    message=f"Low win rate: {snapshot.win_rate*100:.1f}%",
                    timestamp=snapshot.timestamp,
                    current_value=snapshot.win_rate,
                    threshold=thresholds['min_win_rate']
                ))
            
            # Individual hat alerts
            for hat_name, score in snapshot.hat_scores.items():
                if score < thresholds['critical_score']:
                    alerts.append(PerformanceAlert(
                        alert_type='critical',
                        message=f"{hat_name} critical performance: {score:.1f}/10",
                        timestamp=snapshot.timestamp,
                        hat_name=hat_name,
                        current_value=score,
                        threshold=thresholds['critical_score']
                    ))
                elif score < thresholds['warning_score']:
                    alerts.append(PerformanceAlert(
                        alert_type='warning',
                        message=f"{hat_name} performance warning: {score:.1f}/10",
                        timestamp=snapshot.timestamp,
                        hat_name=hat_name,
                        current_value=score,
                        threshold=thresholds['warning_score']
                    ))
            
            # Store alerts
            for alert in alerts:
                self.performance_alerts.append(alert)
                
                # Log critical alerts
                if alert.alert_type == 'critical':
                    self.logger.critical(f"🚨 [ULTIMATE_MONITOR] {alert.message}")
                elif alert.alert_type == 'warning':
                    self.logger.warning(f"⚠️ [ULTIMATE_MONITOR] {alert.message}")
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Error checking alerts: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            if not self.last_snapshot:
                return {}
            
            return {
                'current_performance': {
                    'overall_score': self.last_snapshot.overall_score,
                    'system_health': self.last_snapshot.system_health,
                    'total_profit': self.last_snapshot.total_profit,
                    'win_rate': self.last_snapshot.win_rate,
                    'active_trades': self.last_snapshot.active_trades,
                    'total_trades': self.last_snapshot.total_trades,
                    'confidence': self.last_snapshot.confidence,
                    'position_size': self.last_snapshot.position_size
                },
                'performance_peaks': {
                    'peak_performance': self.peak_performance,
                    'peak_profit': self.peak_profit
                },
                'performance_trends': self.performance_trends,
                'performance_insights': self.performance_insights,
                'optimization_recommendations': self.optimization_recommendations,
                'recent_alerts': [
                    {
                        'type': alert.alert_type,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'hat_name': alert.hat_name,
                        'current_value': alert.current_value,
                        'threshold': alert.threshold
                    }
                    for alert in list(self.performance_alerts)[-10:]  # Last 10 alerts
                ],
                'performance_history_size': len(self.performance_history),
                'monitoring_status': {
                    'monitoring_active': self.monitoring_active,
                    'last_snapshot': self.last_snapshot.timestamp.isoformat() if self.last_snapshot else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Error getting performance report: {e}")
            return {}
    
    def shutdown(self):
        """Gracefully shutdown the performance monitor"""
        try:
            self.monitoring_active = False
            
            # Log final performance report
            final_report = self.get_performance_report()
            self.logger.info(f"📊 [ULTIMATE_MONITOR] Final performance report: {final_report}")
            
            self.logger.info("📊 [ULTIMATE_MONITOR] Ultimate Performance Monitor shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ [ULTIMATE_MONITOR] Shutdown error: {e}")

# Export the main class
__all__ = ['UltimatePerformanceMonitor', 'PerformanceSnapshot', 'PerformanceAlert']
