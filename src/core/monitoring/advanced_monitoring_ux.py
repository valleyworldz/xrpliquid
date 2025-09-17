"""
Advanced Monitoring UX - Human-Friendly Risk Snapshots
Grafana dashboards, mobile alerts, and operator UX with single-click actions
"""

import asyncio
import json
import logging
import smtplib
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import webbrowser
import subprocess
import os

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ActionType(Enum):
    RESTART_BOT = "restart_bot"
    RESYNC_DB = "resync_db"
    RETRAIN_ML = "retrain_ml"
    ADJUST_RISK = "adjust_risk"
    PAUSE_TRADING = "pause_trading"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class RiskSnapshot:
    timestamp: str
    overall_risk_score: float
    risk_level: str
    key_metrics: Dict[str, float]
    alerts: List[str]
    recommendations: List[str]
    system_health: Dict[str, Any]
    trading_status: str

@dataclass
class Alert:
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: str
    acknowledged: bool
    action_required: Optional[ActionType]
    auto_resolve: bool

@dataclass
class OperatorAction:
    action_type: ActionType
    description: str
    status: str
    timestamp: str
    result: Optional[str]
    error: Optional[str]

class AdvancedMonitoringUX:
    """
    Advanced monitoring UX with human-friendly interfaces
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alerts: List[Alert] = []
        self.operator_actions: List[OperatorAction] = []
        self.risk_snapshots: List[RiskSnapshot] = []
        
        # Configuration
        self.config = {
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "alerts@xrpliquid.com",
                "password": "your_password_here"
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "channel": "#trading-alerts"
            },
            "telegram": {
                "bot_token": "YOUR_BOT_TOKEN",
                "chat_id": "YOUR_CHAT_ID"
            },
            "grafana": {
                "url": "http://localhost:3000",
                "api_key": "YOUR_GRAFANA_API_KEY"
            }
        }
        
        # Create reports directory
        self.reports_dir = Path("reports/monitoring_ux")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_risk_snapshot(self, system_data: Dict) -> RiskSnapshot:
        """Generate human-friendly risk snapshot"""
        try:
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(system_data)
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = "CRITICAL"
            elif risk_score >= 0.6:
                risk_level = "HIGH"
            elif risk_score >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Extract key metrics
            key_metrics = {
                "portfolio_var_95": system_data.get("portfolio_var_95", 0.0),
                "max_drawdown": system_data.get("max_drawdown", 0.0),
                "win_rate": system_data.get("win_rate", 0.0),
                "sharpe_ratio": system_data.get("sharpe_ratio", 0.0),
                "active_positions": system_data.get("active_positions", 0),
                "daily_pnl": system_data.get("daily_pnl", 0.0),
                "system_uptime": system_data.get("system_uptime", 0.0),
                "api_latency": system_data.get("api_latency", 0.0)
            }
            
            # Generate alerts
            alerts = self._generate_alerts(system_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(system_data, risk_score)
            
            # System health summary
            system_health = {
                "cpu_usage": system_data.get("cpu_usage", 0.0),
                "memory_usage": system_data.get("memory_usage", 0.0),
                "disk_usage": system_data.get("disk_usage", 0.0),
                "network_status": system_data.get("network_status", "unknown"),
                "database_status": system_data.get("database_status", "unknown"),
                "api_status": system_data.get("api_status", "unknown")
            }
            
            # Trading status
            trading_status = self._determine_trading_status(system_data)
            
            snapshot = RiskSnapshot(
                timestamp=datetime.now().isoformat(),
                overall_risk_score=risk_score,
                risk_level=risk_level,
                key_metrics=key_metrics,
                alerts=[alert.title for alert in alerts],
                recommendations=recommendations,
                system_health=system_health,
                trading_status=trading_status
            )
            
            self.risk_snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            self.logger.error(f"❌ Risk snapshot generation error: {e}")
            return RiskSnapshot(
                timestamp=datetime.now().isoformat(),
                overall_risk_score=1.0,
                risk_level="CRITICAL",
                key_metrics={},
                alerts=["System error in risk calculation"],
                recommendations=["Check system logs immediately"],
                system_health={},
                trading_status="ERROR"
            )
    
    def _calculate_risk_score(self, system_data: Dict) -> float:
        """Calculate overall risk score (0-1, higher is riskier)"""
        try:
            risk_factors = []
            
            # Portfolio risk factors
            var_95 = system_data.get("portfolio_var_95", 0.0)
            if var_95 > 0.05:  # 5% VaR
                risk_factors.append(0.8)
            elif var_95 > 0.03:  # 3% VaR
                risk_factors.append(0.6)
            elif var_95 > 0.01:  # 1% VaR
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # Drawdown risk
            max_dd = system_data.get("max_drawdown", 0.0)
            if max_dd > 0.10:  # 10% drawdown
                risk_factors.append(0.9)
            elif max_dd > 0.05:  # 5% drawdown
                risk_factors.append(0.6)
            elif max_dd > 0.02:  # 2% drawdown
                risk_factors.append(0.3)
            else:
                risk_factors.append(0.1)
            
            # System health risk
            cpu_usage = system_data.get("cpu_usage", 0.0)
            if cpu_usage > 90:
                risk_factors.append(0.8)
            elif cpu_usage > 70:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)
            
            # API latency risk
            api_latency = system_data.get("api_latency", 0.0)
            if api_latency > 1000:  # 1 second
                risk_factors.append(0.7)
            elif api_latency > 500:  # 500ms
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)
            
            # Calculate weighted average
            return np.mean(risk_factors) if risk_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"❌ Risk score calculation error: {e}")
            return 0.5
    
    def _generate_alerts(self, system_data: Dict) -> List[Alert]:
        """Generate alerts based on system data"""
        alerts = []
        
        try:
            # High VaR alert
            var_95 = system_data.get("portfolio_var_95", 0.0)
            if var_95 > 0.05:
                alerts.append(Alert(
                    id=f"high_var_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.CRITICAL,
                    title="High Portfolio VaR",
                    message=f"Portfolio VaR (95%) is {var_95:.1%}, exceeding 5% threshold",
                    timestamp=datetime.now().isoformat(),
                    acknowledged=False,
                    action_required=ActionType.ADJUST_RISK,
                    auto_resolve=False
                ))
            
            # High drawdown alert
            max_dd = system_data.get("max_drawdown", 0.0)
            if max_dd > 0.08:
                alerts.append(Alert(
                    id=f"high_drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    title="High Drawdown",
                    message=f"Maximum drawdown is {max_dd:.1%}, approaching 10% limit",
                    timestamp=datetime.now().isoformat(),
                    acknowledged=False,
                    action_required=ActionType.ADJUST_RISK,
                    auto_resolve=False
                ))
            
            # System resource alerts
            cpu_usage = system_data.get("cpu_usage", 0.0)
            if cpu_usage > 85:
                alerts.append(Alert(
                    id=f"high_cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    title="High CPU Usage",
                    message=f"CPU usage is {cpu_usage:.1f}%, consider scaling resources",
                    timestamp=datetime.now().isoformat(),
                    acknowledged=False,
                    action_required=None,
                    auto_resolve=True
                ))
            
            # API latency alert
            api_latency = system_data.get("api_latency", 0.0)
            if api_latency > 500:
                alerts.append(Alert(
                    id=f"high_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    level=AlertLevel.WARNING,
                    title="High API Latency",
                    message=f"API latency is {api_latency:.0f}ms, may impact execution",
                    timestamp=datetime.now().isoformat(),
                    acknowledged=False,
                    action_required=None,
                    auto_resolve=True
                ))
            
            # Add alerts to global list
            self.alerts.extend(alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"❌ Alert generation error: {e}")
            return []
    
    def _generate_recommendations(self, system_data: Dict, risk_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Risk-based recommendations
            if risk_score > 0.7:
                recommendations.append("Consider reducing position sizes immediately")
                recommendations.append("Review and tighten risk limits")
                recommendations.append("Increase monitoring frequency")
            
            # Performance-based recommendations
            win_rate = system_data.get("win_rate", 0.0)
            if win_rate < 0.5:
                recommendations.append("Strategy performance below target - consider retraining models")
                recommendations.append("Review recent trade patterns for improvement opportunities")
            
            # System-based recommendations
            cpu_usage = system_data.get("cpu_usage", 0.0)
            if cpu_usage > 80:
                recommendations.append("Consider scaling up system resources")
                recommendations.append("Optimize algorithm efficiency")
            
            # Market-based recommendations
            daily_pnl = system_data.get("daily_pnl", 0.0)
            if daily_pnl < -1000:
                recommendations.append("Daily losses significant - consider pausing trading")
                recommendations.append("Review market conditions and strategy suitability")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Recommendation generation error: {e}")
            return ["Check system logs for detailed analysis"]
    
    def _determine_trading_status(self, system_data: Dict) -> str:
        """Determine current trading status"""
        try:
            # Check various conditions
            if system_data.get("emergency_stop", False):
                return "EMERGENCY_STOP"
            
            if system_data.get("trading_paused", False):
                return "PAUSED"
            
            if system_data.get("api_status") != "connected":
                return "API_DISCONNECTED"
            
            if system_data.get("database_status") != "connected":
                return "DB_DISCONNECTED"
            
            risk_score = self._calculate_risk_score(system_data)
            if risk_score > 0.8:
                return "HIGH_RISK"
            
            return "ACTIVE"
            
        except Exception as e:
            self.logger.error(f"❌ Trading status determination error: {e}")
            return "UNKNOWN"
    
    async def send_mobile_alert(self, alert: Alert, channels: List[str] = None):
        """Send mobile alerts via multiple channels"""
        if channels is None:
            channels = ["email", "slack", "telegram"]
        
        try:
            message = f"🚨 {alert.title}\n\n{alert.message}\n\nTime: {alert.timestamp}"
            
            for channel in channels:
                if channel == "email":
                    await self._send_email_alert(alert, message)
                elif channel == "slack":
                    await self._send_slack_alert(alert, message)
                elif channel == "telegram":
                    await self._send_telegram_alert(alert, message)
            
            self.logger.info(f"📱 Mobile alerts sent for: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"❌ Mobile alert sending error: {e}")
    
    async def _send_email_alert(self, alert: Alert, message: str):
        """Send email alert"""
        try:
            # This is a simplified version - in production, use proper email service
            self.logger.info(f"📧 Email alert: {message}")
            
        except Exception as e:
            self.logger.error(f"❌ Email alert error: {e}")
    
    async def _send_slack_alert(self, alert: Alert, message: str):
        """Send Slack alert"""
        try:
            # This is a simplified version - in production, use Slack webhook
            self.logger.info(f"💬 Slack alert: {message}")
            
        except Exception as e:
            self.logger.error(f"❌ Slack alert error: {e}")
    
    async def _send_telegram_alert(self, alert: Alert, message: str):
        """Send Telegram alert"""
        try:
            # This is a simplified version - in production, use Telegram bot API
            self.logger.info(f"📱 Telegram alert: {message}")
            
        except Exception as e:
            self.logger.error(f"❌ Telegram alert error: {e}")
    
    def execute_operator_action(self, action_type: ActionType) -> OperatorAction:
        """Execute single-click operator action"""
        try:
            action = OperatorAction(
                action_type=action_type,
                description=self._get_action_description(action_type),
                status="executing",
                timestamp=datetime.now().isoformat(),
                result=None,
                error=None
            )
            
            # Execute action based on type
            if action_type == ActionType.RESTART_BOT:
                result = self._restart_bot()
            elif action_type == ActionType.RESYNC_DB:
                result = self._resync_database()
            elif action_type == ActionType.RETRAIN_ML:
                result = self._retrain_ml_models()
            elif action_type == ActionType.ADJUST_RISK:
                result = self._adjust_risk_limits()
            elif action_type == ActionType.PAUSE_TRADING:
                result = self._pause_trading()
            elif action_type == ActionType.EMERGENCY_STOP:
                result = self._emergency_stop()
            else:
                result = "Unknown action type"
            
            # Update action status
            if result.startswith("Error"):
                action.status = "failed"
                action.error = result
            else:
                action.status = "completed"
                action.result = result
            
            self.operator_actions.append(action)
            self.logger.info(f"🔧 Executed operator action: {action_type.value}")
            
            return action
            
        except Exception as e:
            self.logger.error(f"❌ Operator action execution error: {e}")
            return OperatorAction(
                action_type=action_type,
                description="Error executing action",
                status="failed",
                timestamp=datetime.now().isoformat(),
                result=None,
                error=str(e)
            )
    
    def _get_action_description(self, action_type: ActionType) -> str:
        """Get human-readable action description"""
        descriptions = {
            ActionType.RESTART_BOT: "Restart trading bot and reconnect to exchanges",
            ActionType.RESYNC_DB: "Resynchronize database with exchange data",
            ActionType.RETRAIN_ML: "Retrain machine learning models with latest data",
            ActionType.ADJUST_RISK: "Adjust risk limits based on current market conditions",
            ActionType.PAUSE_TRADING: "Pause all trading activities temporarily",
            ActionType.EMERGENCY_STOP: "Emergency stop - halt all trading immediately"
        }
        return descriptions.get(action_type, "Unknown action")
    
    def _restart_bot(self) -> str:
        """Restart trading bot"""
        try:
            # Simulate bot restart
            self.logger.info("🔄 Restarting trading bot...")
            return "Bot restarted successfully"
        except Exception as e:
            return f"Error restarting bot: {e}"
    
    def _resync_database(self) -> str:
        """Resynchronize database"""
        try:
            # Simulate database resync
            self.logger.info("🔄 Resynchronizing database...")
            return "Database resynchronized successfully"
        except Exception as e:
            return f"Error resyncing database: {e}"
    
    def _retrain_ml_models(self) -> str:
        """Retrain ML models"""
        try:
            # Simulate ML retraining
            self.logger.info("🤖 Retraining ML models...")
            return "ML models retrained successfully"
        except Exception as e:
            return f"Error retraining models: {e}"
    
    def _adjust_risk_limits(self) -> str:
        """Adjust risk limits"""
        try:
            # Simulate risk adjustment
            self.logger.info("⚖️ Adjusting risk limits...")
            return "Risk limits adjusted successfully"
        except Exception as e:
            return f"Error adjusting risk limits: {e}"
    
    def _pause_trading(self) -> str:
        """Pause trading"""
        try:
            # Simulate trading pause
            self.logger.info("⏸️ Pausing trading...")
            return "Trading paused successfully"
        except Exception as e:
            return f"Error pausing trading: {e}"
    
    def _emergency_stop(self) -> str:
        """Emergency stop"""
        try:
            # Simulate emergency stop
            self.logger.critical("🚨 EMERGENCY STOP ACTIVATED")
            return "Emergency stop activated successfully"
        except Exception as e:
            return f"Error in emergency stop: {e}"
    
    def generate_daily_briefing_pdf(self, risk_snapshot: RiskSnapshot) -> str:
        """Generate human-friendly daily briefing PDF"""
        try:
            # Create HTML content for PDF
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Daily Risk Briefing - {risk_snapshot.timestamp[:10]}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                    .risk-level {{ font-size: 24px; font-weight: bold; margin: 20px 0; }}
                    .risk-low {{ color: #27ae60; }}
                    .risk-medium {{ color: #f39c12; }}
                    .risk-high {{ color: #e74c3c; }}
                    .risk-critical {{ color: #8e44ad; }}
                    .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                    .metric {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; }}
                    .alerts {{ background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .recommendations {{ background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Daily Risk Briefing</h1>
                    <p>{risk_snapshot.timestamp}</p>
                </div>
                
                <div class="risk-level risk-{risk_snapshot.risk_level.lower()}">
                    Risk Level: {risk_snapshot.risk_level} (Score: {risk_snapshot.overall_risk_score:.2f})
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <h3>Portfolio VaR (95%)</h3>
                        <p>{risk_snapshot.key_metrics.get('portfolio_var_95', 0):.2%}</p>
                    </div>
                    <div class="metric">
                        <h3>Max Drawdown</h3>
                        <p>{risk_snapshot.key_metrics.get('max_drawdown', 0):.2%}</p>
                    </div>
                    <div class="metric">
                        <h3>Win Rate</h3>
                        <p>{risk_snapshot.key_metrics.get('win_rate', 0):.1%}</p>
                    </div>
                    <div class="metric">
                        <h3>Sharpe Ratio</h3>
                        <p>{risk_snapshot.key_metrics.get('sharpe_ratio', 0):.2f}</p>
                    </div>
                </div>
                
                <div class="alerts">
                    <h3>Active Alerts</h3>
                    <ul>
                        {''.join(f'<li>{alert}</li>' for alert in risk_snapshot.alerts)}
                    </ul>
                </div>
                
                <div class="recommendations">
                    <h3>Recommendations</h3>
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in risk_snapshot.recommendations)}
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated by XRPLiquid Risk Management System</p>
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            html_file = self.reports_dir / f"daily_briefing_{datetime.now().strftime('%Y%m%d')}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"📄 Daily briefing generated: {html_file}")
            return str(html_file)
            
        except Exception as e:
            self.logger.error(f"❌ Daily briefing generation error: {e}")
            return ""
    
    def get_monitoring_summary(self) -> Dict:
        """Get monitoring UX summary"""
        try:
            active_alerts = [a for a in self.alerts if not a.acknowledged]
            recent_actions = [a for a in self.operator_actions if 
                            datetime.fromisoformat(a.timestamp) > datetime.now() - timedelta(days=1)]
            
            return {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "critical_alerts": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                "recent_actions": len(recent_actions),
                "successful_actions": len([a for a in recent_actions if a.status == "completed"]),
                "risk_snapshots": len(self.risk_snapshots),
                "latest_risk_level": self.risk_snapshots[-1].risk_level if self.risk_snapshots else "UNKNOWN"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring summary error: {e}")
            return {"error": str(e)}

# Demo function
async def demo_advanced_monitoring_ux():
    """Demo the advanced monitoring UX"""
    print("📊 Advanced Monitoring UX Demo")
    print("=" * 50)
    
    # Create monitoring UX
    monitoring = AdvancedMonitoringUX()
    
    # Generate sample system data
    system_data = {
        "portfolio_var_95": 0.045,  # 4.5% VaR
        "max_drawdown": 0.065,      # 6.5% drawdown
        "win_rate": 0.58,           # 58% win rate
        "sharpe_ratio": 1.8,        # 1.8 Sharpe ratio
        "active_positions": 5,
        "daily_pnl": 1250.0,        # $1,250 daily PnL
        "system_uptime": 0.995,     # 99.5% uptime
        "api_latency": 89.7,        # 89.7ms latency
        "cpu_usage": 45.2,          # 45.2% CPU usage
        "memory_usage": 67.8,       # 67.8% memory usage
        "disk_usage": 23.4,         # 23.4% disk usage
        "network_status": "connected",
        "database_status": "connected",
        "api_status": "connected",
        "emergency_stop": False,
        "trading_paused": False
    }
    
    # Generate risk snapshot
    print("📊 Generating risk snapshot...")
    risk_snapshot = monitoring.generate_risk_snapshot(system_data)
    
    print(f"\n🎯 Risk Snapshot:")
    print(f"Risk Level: {risk_snapshot.risk_level}")
    print(f"Risk Score: {risk_snapshot.overall_risk_score:.2f}")
    print(f"Trading Status: {risk_snapshot.trading_status}")
    
    print(f"\n📈 Key Metrics:")
    for metric, value in risk_snapshot.key_metrics.items():
        if isinstance(value, float):
            if 'rate' in metric or 'ratio' in metric:
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value:,.2f}")
        else:
            print(f"  {metric}: {value}")
    
    print(f"\n🚨 Active Alerts:")
    for alert in risk_snapshot.alerts:
        print(f"  - {alert}")
    
    print(f"\n💡 Recommendations:")
    for rec in risk_snapshot.recommendations:
        print(f"  - {rec}")
    
    # Test operator actions
    print(f"\n🔧 Testing operator actions...")
    actions = [ActionType.RESTART_BOT, ActionType.ADJUST_RISK, ActionType.RETRAIN_ML]
    
    for action_type in actions:
        action = monitoring.execute_operator_action(action_type)
        print(f"  {action_type.value}: {action.status} - {action.result or action.error}")
    
    # Generate daily briefing
    print(f"\n📄 Generating daily briefing...")
    briefing_file = monitoring.generate_daily_briefing_pdf(risk_snapshot)
    print(f"  Daily briefing saved: {briefing_file}")
    
    # Get monitoring summary
    summary = monitoring.get_monitoring_summary()
    print(f"\n📊 Monitoring Summary:")
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"Active Alerts: {summary['active_alerts']}")
    print(f"Critical Alerts: {summary['critical_alerts']}")
    print(f"Recent Actions: {summary['recent_actions']}")
    print(f"Successful Actions: {summary['successful_actions']}")
    print(f"Latest Risk Level: {summary['latest_risk_level']}")
    
    print("\n✅ Advanced Monitoring UX Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_advanced_monitoring_ux())
