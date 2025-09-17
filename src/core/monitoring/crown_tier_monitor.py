"""
Crown Tier Monitor - Real-time monitoring for crown tier operational status
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import os
import sys

class CrownTierMonitor:
    """
    Real-time monitor for crown tier operational status
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.metrics = {
            'decimal_errors': 0,
            'engine_failures': 0,
            'feasibility_blocks': 0,
            'guardian_invocations': 0,
            'orders_submitted': 0,
            'orders_blocked': 0,
            'performance_score': 0.0,
            'crown_tier_status': 'UNKNOWN'
        }
        self.alerts = []
        self.thresholds = {
            'max_decimal_errors': 0,
            'max_engine_failures': 0,
            'max_guardian_invocations': 0,
            'min_performance_score': 8.0,
            'max_feasibility_block_rate': 0.1  # 10%
        }
    
    def log_decimal_error(self, error_msg: str, context: Dict[str, Any] = None):
        """Log a decimal error occurrence"""
        self.metrics['decimal_errors'] += 1
        
        alert = {
            'type': 'DECIMAL_ERROR',
            'timestamp': datetime.now().isoformat(),
            'message': error_msg,
            'context': context or {},
            'severity': 'CRITICAL'
        }
        self.alerts.append(alert)
        
        self.logger.error(f"🚨 DECIMAL_ERROR: {error_msg}")
        self._check_crown_tier_status()
    
    def log_engine_failure(self, engine_name: str, error_msg: str):
        """Log an engine failure"""
        self.metrics['engine_failures'] += 1
        
        alert = {
            'type': 'ENGINE_FAILURE',
            'timestamp': datetime.now().isoformat(),
            'engine': engine_name,
            'message': error_msg,
            'severity': 'CRITICAL'
        }
        self.alerts.append(alert)
        
        self.logger.error(f"🚨 ENGINE_FAILURE: {engine_name} - {error_msg}")
        self._check_crown_tier_status()
    
    def log_feasibility_block(self, symbol: str, reason: str):
        """Log a feasibility block"""
        self.metrics['feasibility_blocks'] += 1
        
        alert = {
            'type': 'FEASIBILITY_BLOCK',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'reason': reason,
            'severity': 'WARNING'
        }
        self.alerts.append(alert)
        
        self.logger.warning(f"🚫 FEASIBILITY_BLOCK: {symbol} - {reason}")
        self._check_crown_tier_status()
    
    def log_guardian_invocation(self, reason: str, context: Dict[str, Any] = None):
        """Log a Guardian invocation (should not happen with proper feasibility gates)"""
        self.metrics['guardian_invocations'] += 1
        
        alert = {
            'type': 'GUARDIAN_INVOCATION',
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'context': context or {},
            'severity': 'CRITICAL'
        }
        self.alerts.append(alert)
        
        self.logger.critical(f"🚨 GUARDIAN_INVOCATION: {reason}")
        self._check_crown_tier_status()
    
    def log_order_submitted(self, symbol: str, side: str, size: Decimal, price: Decimal):
        """Log a successful order submission"""
        self.metrics['orders_submitted'] += 1
        
        self.logger.info(f"✅ ORDER_SUBMITTED: {side} {size} {symbol} @ {price}")
        self._check_crown_tier_status()
    
    def log_order_blocked(self, symbol: str, reason: str):
        """Log an order that was blocked"""
        self.metrics['orders_blocked'] += 1
        
        self.logger.warning(f"🚫 ORDER_BLOCKED: {symbol} - {reason}")
        self._check_crown_tier_status()
    
    def update_performance_score(self, score: float):
        """Update the performance score"""
        self.metrics['performance_score'] = score
        self._check_crown_tier_status()
    
    def _check_crown_tier_status(self):
        """Check if system maintains crown tier status"""
        try:
            # Calculate block rate
            total_orders = self.metrics['orders_submitted'] + self.metrics['orders_blocked']
            block_rate = self.metrics['orders_blocked'] / total_orders if total_orders > 0 else 0
            
            # Check all thresholds
            violations = []
            
            if self.metrics['decimal_errors'] > self.thresholds['max_decimal_errors']:
                violations.append(f"Decimal errors: {self.metrics['decimal_errors']} > {self.thresholds['max_decimal_errors']}")
            
            if self.metrics['engine_failures'] > self.thresholds['max_engine_failures']:
                violations.append(f"Engine failures: {self.metrics['engine_failures']} > {self.thresholds['max_engine_failures']}")
            
            if self.metrics['guardian_invocations'] > self.thresholds['max_guardian_invocations']:
                violations.append(f"Guardian invocations: {self.metrics['guardian_invocations']} > {self.thresholds['max_guardian_invocations']}")
            
            if self.metrics['performance_score'] < self.thresholds['min_performance_score']:
                violations.append(f"Performance score: {self.metrics['performance_score']:.2f} < {self.thresholds['min_performance_score']}")
            
            if block_rate > self.thresholds['max_feasibility_block_rate']:
                violations.append(f"Block rate: {block_rate:.1%} > {self.thresholds['max_feasibility_block_rate']:.1%}")
            
            # Determine crown tier status
            if len(violations) == 0:
                self.metrics['crown_tier_status'] = 'CROWN_TIER'
                status_emoji = '🏆'
            elif len(violations) <= 2:
                self.metrics['crown_tier_status'] = 'INSTITUTION_READY'
                status_emoji = '✅'
            else:
                self.metrics['crown_tier_status'] = 'DEGRADED'
                status_emoji = '⚠️'
            
            # Log status update
            self.logger.info(f"{status_emoji} CROWN_TIER_STATUS: {self.metrics['crown_tier_status']}")
            
            if violations:
                self.logger.warning(f"⚠️ CROWN_TIER_VIOLATIONS: {', '.join(violations)}")
            
        except Exception as e:
            self.logger.error(f"Error checking crown tier status: {e}")
    
    def get_crown_tier_report(self) -> Dict[str, Any]:
        """Get comprehensive crown tier status report"""
        try:
            uptime = datetime.now() - self.start_time
            total_orders = self.metrics['orders_submitted'] + self.metrics['orders_blocked']
            block_rate = self.metrics['orders_blocked'] / total_orders if total_orders > 0 else 0
            
            # Calculate health score
            health_score = 100.0
            health_score -= self.metrics['decimal_errors'] * 10  # -10 per decimal error
            health_score -= self.metrics['engine_failures'] * 20  # -20 per engine failure
            health_score -= self.metrics['guardian_invocations'] * 30  # -30 per guardian invocation
            health_score -= (self.thresholds['min_performance_score'] - self.metrics['performance_score']) * 5  # -5 per performance point below threshold
            health_score -= block_rate * 100  # -1 per 1% block rate
            health_score = max(0, health_score)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime.total_seconds(),
                'crown_tier_status': self.metrics['crown_tier_status'],
                'health_score': health_score,
                'metrics': self.metrics.copy(),
                'thresholds': self.thresholds.copy(),
                'operational_stats': {
                    'total_orders': total_orders,
                    'orders_submitted': self.metrics['orders_submitted'],
                    'orders_blocked': self.metrics['orders_blocked'],
                    'block_rate_percent': block_rate * 100,
                    'feasibility_blocks': self.metrics['feasibility_blocks']
                },
                'error_stats': {
                    'decimal_errors': self.metrics['decimal_errors'],
                    'engine_failures': self.metrics['engine_failures'],
                    'guardian_invocations': self.metrics['guardian_invocations']
                },
                'recent_alerts': self.alerts[-10:],  # Last 10 alerts
                'threshold_violations': self._get_threshold_violations()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating crown tier report: {e}")
            return {'error': str(e)}
    
    def _get_threshold_violations(self) -> List[str]:
        """Get list of current threshold violations"""
        violations = []
        
        if self.metrics['decimal_errors'] > self.thresholds['max_decimal_errors']:
            violations.append(f"Decimal errors: {self.metrics['decimal_errors']} > {self.thresholds['max_decimal_errors']}")
        
        if self.metrics['engine_failures'] > self.thresholds['max_engine_failures']:
            violations.append(f"Engine failures: {self.metrics['engine_failures']} > {self.thresholds['max_engine_failures']}")
        
        if self.metrics['guardian_invocations'] > self.thresholds['max_guardian_invocations']:
            violations.append(f"Guardian invocations: {self.metrics['guardian_invocations']} > {self.thresholds['max_guardian_invocations']}")
        
        if self.metrics['performance_score'] < self.thresholds['min_performance_score']:
            violations.append(f"Performance score: {self.metrics['performance_score']:.2f} < {self.thresholds['min_performance_score']}")
        
        total_orders = self.metrics['orders_submitted'] + self.metrics['orders_blocked']
        block_rate = self.metrics['orders_blocked'] / total_orders if total_orders > 0 else 0
        if block_rate > self.thresholds['max_feasibility_block_rate']:
            violations.append(f"Block rate: {block_rate:.1%} > {self.thresholds['max_feasibility_block_rate']:.1%}")
        
        return violations
    
    def export_crown_tier_report(self, filepath: str = None):
        """Export crown tier report to file"""
        try:
            if filepath is None:
                filepath = f"reports/crown_tier_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            report = self.get_crown_tier_report()
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"📊 Crown tier report exported to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting crown tier report: {e}")
            return None

# Global monitor instance
_crown_tier_monitor = CrownTierMonitor()

def log_decimal_error(error_msg: str, context: Dict[str, Any] = None):
    """Global function to log decimal errors"""
    _crown_tier_monitor.log_decimal_error(error_msg, context)

def log_engine_failure(engine_name: str, error_msg: str):
    """Global function to log engine failures"""
    _crown_tier_monitor.log_engine_failure(engine_name, error_msg)

def log_feasibility_block(symbol: str, reason: str):
    """Global function to log feasibility blocks"""
    _crown_tier_monitor.log_feasibility_block(symbol, reason)

def log_guardian_invocation(reason: str, context: Dict[str, Any] = None):
    """Global function to log Guardian invocations"""
    _crown_tier_monitor.log_guardian_invocation(reason, context)

def log_order_submitted(symbol: str, side: str, size: Decimal, price: Decimal):
    """Global function to log order submissions"""
    _crown_tier_monitor.log_order_submitted(symbol, side, size, price)

def log_order_blocked(symbol: str, reason: str):
    """Global function to log order blocks"""
    _crown_tier_monitor.log_order_blocked(symbol, reason)

def update_performance_score(score: float):
    """Global function to update performance score"""
    _crown_tier_monitor.update_performance_score(score)

def get_crown_tier_report() -> Dict[str, Any]:
    """Global function to get crown tier report"""
    return _crown_tier_monitor.get_crown_tier_report()

def export_crown_tier_report(filepath: str = None) -> Optional[str]:
    """Global function to export crown tier report"""
    return _crown_tier_monitor.export_crown_tier_report(filepath)

# Demo function
def demo_crown_tier_monitor():
    """Demo the crown tier monitor"""
    print("🏆 Crown Tier Monitor Demo")
    print("=" * 50)
    
    monitor = CrownTierMonitor()
    
    # Simulate some events
    print("\n🔍 Simulating operational events...")
    
    # Good events
    monitor.log_order_submitted("XRP/USD", "BUY", Decimal('1000'), Decimal('0.52'))
    monitor.log_order_submitted("XRP/USD", "SELL", Decimal('1000'), Decimal('0.53'))
    monitor.update_performance_score(8.5)
    
    # Some feasibility blocks (normal)
    monitor.log_feasibility_block("XRP/USD", "Insufficient market depth")
    monitor.log_feasibility_block("XRP/USD", "Slippage too high")
    
    # Check status
    report = monitor.get_crown_tier_report()
    print(f"\n📊 Crown Tier Status Report:")
    print(f"  Status: {report['crown_tier_status']}")
    print(f"  Health Score: {report['health_score']:.1f}")
    print(f"  Orders Submitted: {report['metrics']['orders_submitted']}")
    print(f"  Orders Blocked: {report['metrics']['orders_blocked']}")
    print(f"  Performance Score: {report['metrics']['performance_score']}")
    
    # Simulate some problems
    print(f"\n⚠️ Simulating problems...")
    monitor.log_decimal_error("TypeError: unsupported operand type(s) for -: 'float' and 'decimal.Decimal'")
    monitor.log_engine_failure("RealTimeRiskEngine", "Engine not available")
    monitor.log_guardian_invocation("TP/SL activation failed")
    
    # Check status again
    report = monitor.get_crown_tier_report()
    print(f"\n📊 Updated Crown Tier Status Report:")
    print(f"  Status: {report['crown_tier_status']}")
    print(f"  Health Score: {report['health_score']:.1f}")
    print(f"  Violations: {len(report['threshold_violations'])}")
    
    if report['threshold_violations']:
        print(f"  Violations:")
        for violation in report['threshold_violations']:
            print(f"    - {violation}")
    
    # Export report
    filepath = monitor.export_crown_tier_report()
    if filepath:
        print(f"\n📁 Report exported to: {filepath}")
    
    print(f"\n✅ Crown Tier Monitor Demo Complete")

if __name__ == "__main__":
    demo_crown_tier_monitor()
