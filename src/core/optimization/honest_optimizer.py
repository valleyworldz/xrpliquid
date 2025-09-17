"""
Honest Optimizer - Surfaces "not enough data" messages and gates runs behind minimum trades
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
import numpy as np

class OptimizationStatus(Enum):
    SUFFICIENT_DATA = "sufficient_data"
    INSUFFICIENT_DATA = "insufficient_data"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    OPTIMIZATION_FAILED = "optimization_failed"

@dataclass
class OptimizationResult:
    status: OptimizationStatus
    message: str
    confidence: float
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    data_quality: Dict[str, Any]
    recommendations: List[str]

class HonestOptimizer:
    """
    Honest optimizer that surfaces data quality issues and gates optimization behind minimum requirements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Minimum data requirements
        self.min_trades = 50
        self.min_sessions = 10
        self.min_days = 7
        self.min_confidence = 0.7
        
        # Data quality thresholds
        self.min_win_rate = 0.3
        self.max_drawdown_threshold = 0.2
        self.min_sharpe_ratio = 0.5
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        
    def check_data_sufficiency(self, trade_data: List[Dict[str, Any]], 
                             session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if there's sufficient data for optimization
        """
        try:
            # Count trades
            total_trades = len(trade_data)
            
            # Count sessions
            total_sessions = len(session_data)
            
            # Calculate days
            if trade_data:
                first_trade = min(trade['timestamp'] for trade in trade_data)
                last_trade = max(trade['timestamp'] for trade in trade_data)
                days_span = (last_trade - first_trade).days
            else:
                days_span = 0
            
            # Calculate basic metrics
            if total_trades > 0:
                winning_trades = sum(1 for trade in trade_data if trade.get('pnl', 0) > 0)
                win_rate = winning_trades / total_trades
                
                total_pnl = sum(trade.get('pnl', 0) for trade in trade_data)
                avg_pnl = total_pnl / total_trades
                
                # Calculate Sharpe ratio (simplified)
                pnl_values = [trade.get('pnl', 0) for trade in trade_data]
                if len(pnl_values) > 1:
                    sharpe_ratio = np.mean(pnl_values) / (np.std(pnl_values) + 1e-8)
                else:
                    sharpe_ratio = 0
            else:
                win_rate = 0
                avg_pnl = 0
                sharpe_ratio = 0
            
            # Check sufficiency
            sufficient_trades = total_trades >= self.min_trades
            sufficient_sessions = total_sessions >= self.min_sessions
            sufficient_days = days_span >= self.min_days
            sufficient_quality = win_rate >= self.min_win_rate and sharpe_ratio >= self.min_sharpe_ratio
            
            overall_sufficient = sufficient_trades and sufficient_sessions and sufficient_days and sufficient_quality
            
            return {
                'sufficient': overall_sufficient,
                'total_trades': total_trades,
                'total_sessions': total_sessions,
                'days_span': days_span,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'sharpe_ratio': sharpe_ratio,
                'checks': {
                    'sufficient_trades': sufficient_trades,
                    'sufficient_sessions': sufficient_sessions,
                    'sufficient_days': sufficient_days,
                    'sufficient_quality': sufficient_quality
                },
                'requirements': {
                    'min_trades': self.min_trades,
                    'min_sessions': self.min_sessions,
                    'min_days': self.min_days,
                    'min_win_rate': self.min_win_rate,
                    'min_sharpe_ratio': self.min_sharpe_ratio
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error checking data sufficiency: {e}")
            return {
                'sufficient': False,
                'error': str(e),
                'total_trades': 0,
                'total_sessions': 0,
                'days_span': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'sharpe_ratio': 0
            }
    
    def optimize_parameters(self, trade_data: List[Dict[str, Any]], 
                          session_data: List[Dict[str, Any]],
                          current_params: Dict[str, Any]) -> OptimizationResult:
        """
        Optimize parameters with honest data quality assessment
        """
        try:
            # Check data sufficiency first
            data_check = self.check_data_sufficiency(trade_data, session_data)
            
            if not data_check['sufficient']:
                # Generate honest message about insufficient data
                reasons = []
                if not data_check['checks']['sufficient_trades']:
                    reasons.append(f"Need {self.min_trades - data_check['total_trades']} more trades")
                if not data_check['checks']['sufficient_sessions']:
                    reasons.append(f"Need {self.min_sessions - data_check['total_sessions']} more sessions")
                if not data_check['checks']['sufficient_days']:
                    reasons.append(f"Need {self.min_days - data_check['days_span']} more days")
                if not data_check['checks']['sufficient_quality']:
                    if data_check['win_rate'] < self.min_win_rate:
                        reasons.append(f"Win rate {data_check['win_rate']:.1%} below {self.min_win_rate:.1%} threshold")
                    if data_check['sharpe_ratio'] < self.min_sharpe_ratio:
                        reasons.append(f"Sharpe ratio {data_check['sharpe_ratio']:.2f} below {self.min_sharpe_ratio:.2f} threshold")
                
                message = f"REASON=insufficient_data: {', '.join(reasons)}"
                
                # Log explicit no-op status
                self.logger.warning(f"📊 OPTIMIZER_NO_OP: {message}")
                
                return OptimizationResult(
                    status=OptimizationStatus.INSUFFICIENT_DATA,
                    message=message,
                    confidence=0.0,
                    parameters=current_params,  # Return unchanged parameters
                    metrics=data_check,
                    data_quality=data_check,
                    recommendations=[
                        "Continue trading to gather more data",
                        f"Target {self.min_trades} trades minimum",
                        f"Target {self.min_sessions} sessions minimum",
                        f"Target {self.min_days} days minimum",
                        "Focus on improving win rate and Sharpe ratio"
                    ]
                )
            
            # Data is sufficient, proceed with optimization
            self.logger.info(f"✅ Sufficient data for optimization: {data_check['total_trades']} trades, {data_check['total_sessions']} sessions, {data_check['days_span']} days")
            
            # Perform optimization (simplified example)
            optimized_params = self._perform_optimization(trade_data, session_data, current_params)
            
            # Calculate confidence based on data quality
            confidence = min(0.95, data_check['win_rate'] * data_check['sharpe_ratio'] * 2)
            
            return OptimizationResult(
                status=OptimizationStatus.OPTIMIZATION_COMPLETE,
                message=f"Optimization complete with {data_check['total_trades']} trades",
                confidence=confidence,
                parameters=optimized_params,
                metrics=data_check,
                data_quality=data_check,
                recommendations=[
                    "Optimization successful",
                    f"Confidence: {confidence:.1%}",
                    "Monitor performance and re-optimize weekly"
                ]
            )
            
        except Exception as e:
            self.logger.error(f"❌ Optimization error: {e}")
            return OptimizationResult(
                status=OptimizationStatus.OPTIMIZATION_FAILED,
                message=f"Optimization failed: {e}",
                confidence=0.0,
                parameters=current_params,
                metrics={},
                data_quality={},
                recommendations=["Check data quality and try again"]
            )
    
    def _perform_optimization(self, trade_data: List[Dict[str, Any]], 
                            session_data: List[Dict[str, Any]], 
                            current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform actual parameter optimization
        """
        # Simplified optimization logic
        optimized_params = current_params.copy()
        
        # Analyze trade performance
        if trade_data:
            pnl_values = [trade.get('pnl', 0) for trade in trade_data]
            avg_pnl = np.mean(pnl_values)
            std_pnl = np.std(pnl_values)
            
            # Adjust position size based on performance
            if avg_pnl > 0 and std_pnl > 0:
                # Good performance, can increase position size slightly
                if 'position_size' in optimized_params:
                    optimized_params['position_size'] *= 1.1
            elif avg_pnl < 0:
                # Poor performance, reduce position size
                if 'position_size' in optimized_params:
                    optimized_params['position_size'] *= 0.9
            
            # Adjust stop loss based on volatility
            if std_pnl > 0:
                volatility_adjusted_sl = std_pnl * 2
                if 'stop_loss' in optimized_params:
                    optimized_params['stop_loss'] = min(optimized_params['stop_loss'], volatility_adjusted_sl)
        
        return optimized_params
    
    def log_optimization_result(self, result: OptimizationResult):
        """
        Log optimization result with honest assessment
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'status': result.status.value,
                'message': result.message,
                'confidence': result.confidence,
                'data_quality': result.data_quality,
                'recommendations': result.recommendations
            }
            
            self.optimization_history.append(log_entry)
            
            # Log based on status
            if result.status == OptimizationStatus.INSUFFICIENT_DATA:
                self.logger.warning(f"📊 OPTIMIZATION_SKIPPED: {result.message}")
                self.logger.info(f"📊 Data Quality: {result.data_quality['total_trades']} trades, {result.data_quality['total_sessions']} sessions, {result.data_quality['days_span']} days")
            elif result.status == OptimizationStatus.OPTIMIZATION_COMPLETE:
                self.logger.info(f"✅ OPTIMIZATION_COMPLETE: {result.message}")
                self.logger.info(f"📊 Confidence: {result.confidence:.1%}")
            elif result.status == OptimizationStatus.OPTIMIZATION_FAILED:
                self.logger.error(f"❌ OPTIMIZATION_FAILED: {result.message}")
            
            # Log recommendations
            for recommendation in result.recommendations:
                self.logger.info(f"💡 Recommendation: {recommendation}")
                
        except Exception as e:
            self.logger.error(f"❌ Error logging optimization result: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization history
        """
        try:
            if not self.optimization_history:
                return {
                    'total_optimizations': 0,
                    'successful_optimizations': 0,
                    'insufficient_data_count': 0,
                    'failed_optimizations': 0,
                    'average_confidence': 0.0,
                    'last_optimization': None
                }
            
            total = len(self.optimization_history)
            successful = sum(1 for entry in self.optimization_history if entry['status'] == 'optimization_complete')
            insufficient = sum(1 for entry in self.optimization_history if entry['status'] == 'insufficient_data')
            failed = sum(1 for entry in self.optimization_history if entry['status'] == 'optimization_failed')
            
            avg_confidence = np.mean([entry['confidence'] for entry in self.optimization_history])
            
            return {
                'total_optimizations': total,
                'successful_optimizations': successful,
                'insufficient_data_count': insufficient,
                'failed_optimizations': failed,
                'average_confidence': avg_confidence,
                'last_optimization': self.optimization_history[-1] if self.optimization_history else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting optimization summary: {e}")
            return {}

# Demo function
def demo_honest_optimizer():
    """Demo the honest optimizer"""
    print("📊 Honest Optimizer Demo")
    print("=" * 50)
    
    optimizer = HonestOptimizer()
    
    # Test with insufficient data
    print("🔍 Testing with insufficient data:")
    insufficient_trades = [{'timestamp': datetime.now(), 'pnl': 10} for _ in range(5)]  # Only 5 trades
    insufficient_sessions = [{'timestamp': datetime.now(), 'pnl': 50} for _ in range(2)]  # Only 2 sessions
    
    result1 = optimizer.optimize_parameters(insufficient_trades, insufficient_sessions, {'position_size': 100})
    optimizer.log_optimization_result(result1)
    
    # Test with sufficient data
    print(f"\n🔍 Testing with sufficient data:")
    sufficient_trades = [{'timestamp': datetime.now(), 'pnl': 10} for _ in range(60)]  # 60 trades
    sufficient_sessions = [{'timestamp': datetime.now(), 'pnl': 50} for _ in range(15)]  # 15 sessions
    
    result2 = optimizer.optimize_parameters(sufficient_trades, sufficient_sessions, {'position_size': 100})
    optimizer.log_optimization_result(result2)
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    print(f"\n📊 Optimization Summary:")
    print(f"  Total optimizations: {summary['total_optimizations']}")
    print(f"  Successful: {summary['successful_optimizations']}")
    print(f"  Insufficient data: {summary['insufficient_data_count']}")
    print(f"  Failed: {summary['failed_optimizations']}")
    print(f"  Average confidence: {summary['average_confidence']:.1%}")
    
    print("\n✅ Honest Optimizer Demo Complete")

if __name__ == "__main__":
    demo_honest_optimizer()
