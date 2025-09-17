"""
Optimizer Honesty Surface - NO-OP with exit=78 when data < thresholds
"""

import logging
import json
from typing import Dict, Any, List
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import os

class OptimizerStatus(Enum):
    SUFFICIENT_DATA = "sufficient_data"
    NO_OP = "no_op"
    ERROR = "error"

@dataclass
class DataSufficiencyCheck:
    check_name: str
    required_threshold: int
    actual_value: int
    is_sufficient: bool
    reason: str

@dataclass
class OptimizerResult:
    status: OptimizerStatus
    exit_code: int
    message: str
    data_checks: List[DataSufficiencyCheck]
    should_proceed: bool

class OptimizerHonestySurface:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.thresholds = {
            'min_trades': int(os.getenv('OPTIMIZER_MIN_TRADES', '50')),
            'min_days': int(os.getenv('OPTIMIZER_MIN_DAYS', '7')),
            'min_volume': int(os.getenv('OPTIMIZER_MIN_VOLUME', '1000000'))
        }
        
        self.EXIT_CODE_NO_OP = 78
        self.EXIT_CODE_SUCCESS = 0
        self.EXIT_CODE_ERROR = 1
        
        self.no_op_count = 0
        self.success_count = 0
    
    def check_data_sufficiency(self, data: Dict[str, Any]) -> List[DataSufficiencyCheck]:
        checks = []
        
        try:
            # Check trade count
            trade_count = len(data.get('trades', []))
            trade_sufficient = trade_count >= self.thresholds['min_trades']
            checks.append(DataSufficiencyCheck(
                check_name='trade_count',
                required_threshold=self.thresholds['min_trades'],
                actual_value=trade_count,
                is_sufficient=trade_sufficient,
                reason=f"Need {self.thresholds['min_trades']} trades, have {trade_count}"
            ))
            
            # Check volume
            total_volume = sum(trade.get('volume', 0) for trade in data.get('trades', []))
            volume_sufficient = total_volume >= self.thresholds['min_volume']
            checks.append(DataSufficiencyCheck(
                check_name='total_volume',
                required_threshold=self.thresholds['min_volume'],
                actual_value=int(total_volume),
                is_sufficient=volume_sufficient,
                reason=f"Need {self.thresholds['min_volume']} volume, have {int(total_volume)}"
            ))
            
        except Exception as e:
            self.logger.error(f"❌ Error checking data sufficiency: {e}")
            checks.append(DataSufficiencyCheck(
                check_name='error',
                required_threshold=0,
                actual_value=0,
                is_sufficient=False,
                reason=f"Error checking data: {e}"
            ))
        
        return checks
    
    def should_proceed_with_optimization(self, data: Dict[str, Any]) -> OptimizerResult:
        try:
            checks = self.check_data_sufficiency(data)
            all_sufficient = all(check.is_sufficient for check in checks)
            insufficient_checks = [check for check in checks if not check.is_sufficient]
            
            if all_sufficient:
                self.success_count += 1
                return OptimizerResult(
                    status=OptimizerStatus.SUFFICIENT_DATA,
                    exit_code=self.EXIT_CODE_SUCCESS,
                    message="Data sufficient for optimization",
                    data_checks=checks,
                    should_proceed=True
                )
            else:
                self.no_op_count += 1
                reasons = [check.reason for check in insufficient_checks]
                
                return OptimizerResult(
                    status=OptimizerStatus.NO_OP,
                    exit_code=self.EXIT_CODE_NO_OP,
                    message=f"NO-OP: Insufficient data - {', '.join(reasons)}",
                    data_checks=checks,
                    should_proceed=False
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error in optimizer honesty surface: {e}")
            return OptimizerResult(
                status=OptimizerStatus.ERROR,
                exit_code=self.EXIT_CODE_ERROR,
                message=f"Error: {e}",
                data_checks=[],
                should_proceed=False
            )
    
    def get_optimizer_statistics(self) -> Dict[str, Any]:
        total_attempts = self.no_op_count + self.success_count
        no_op_rate = (self.no_op_count / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            "total_attempts": total_attempts,
            "no_op_count": self.no_op_count,
            "success_count": self.success_count,
            "no_op_rate_percent": round(no_op_rate, 2),
            "thresholds": self.thresholds,
            "last_updated": datetime.now().isoformat()
        }

def demo_optimizer_honesty_surface():
    print("🎯 Optimizer Honesty Surface Demo")
    print("=" * 50)
    
    honesty_surface = OptimizerHonestySurface()
    
    # Test 1: Sufficient data
    print("🔍 Test 1: Sufficient data")
    sufficient_data = {
        'trades': [{'volume': 1000} for _ in range(100)]
    }
    
    result = honesty_surface.should_proceed_with_optimization(sufficient_data)
    print(f"  Status: {result.status.value}")
    print(f"  Exit Code: {result.exit_code}")
    print(f"  Should Proceed: {result.should_proceed}")
    
    # Test 2: Insufficient data
    print(f"\n🔍 Test 2: Insufficient data")
    insufficient_data = {
        'trades': [{'volume': 100}]
    }
    
    result = honesty_surface.should_proceed_with_optimization(insufficient_data)
    print(f"  Status: {result.status.value}")
    print(f"  Exit Code: {result.exit_code}")
    print(f"  Should Proceed: {result.should_proceed}")
    
    # Show statistics
    stats = honesty_surface.get_optimizer_statistics()
    print(f"\n📊 Honesty Surface Statistics: {stats}")
    
    print(f"\n✅ Optimizer Honesty Surface Demo Complete")

if __name__ == "__main__":
    demo_optimizer_honesty_surface()
