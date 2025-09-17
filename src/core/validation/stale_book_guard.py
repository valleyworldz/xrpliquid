"""
Stale Book Guard - Fail-closed on stale L2 data
"""

import logging
import time
import os
import json
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class BookStatus(Enum):
    FRESH = "fresh"
    STALE = "stale"
    INSUFFICIENT = "insufficient"

@dataclass
class BookValidationResult:
    status: BookStatus
    reason: str
    book_age_ms: int
    level_count: int
    should_proceed: bool
    requires_aggressive: bool

class StaleBookGuard:
    """
    Fail-closed guard for stale or insufficient L2 data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_book_age_ms = int(os.getenv('MAX_BOOK_AGE_MS', '5000'))  # 5 seconds
        self.min_levels = int(os.getenv('MIN_BOOK_LEVELS', '5'))
        self.aggressive_allowed = os.getenv('AGGRESSIVE_ALLOWED', 'false').lower() == 'true'
        self.blocked_orders = 0
        self.allowed_orders = 0
        
    def validate_book_freshness(self, book_data: Dict[str, Any]) -> BookValidationResult:
        """
        Validate if book data is fresh and sufficient
        """
        try:
            current_time = time.time() * 1000  # Convert to milliseconds
            
            # Check if book has timestamp
            if 'timestamp' not in book_data:
                return BookValidationResult(
                    status=BookStatus.STALE,
                    reason="Book data missing timestamp",
                    book_age_ms=999999,
                    level_count=0,
                    should_proceed=False,
                    requires_aggressive=True
                )
            
            # Calculate book age
            book_timestamp = book_data['timestamp']
            if isinstance(book_timestamp, str):
                # Parse timestamp if it's a string
                try:
                    book_time = datetime.fromisoformat(book_timestamp.replace('Z', '+00:00'))
                    book_timestamp = book_time.timestamp() * 1000
                except:
                    book_timestamp = current_time - self.max_book_age_ms - 1000  # Force stale
            
            book_age_ms = int(current_time - book_timestamp)
            
            # Check book age
            if book_age_ms > self.max_book_age_ms:
                return BookValidationResult(
                    status=BookStatus.STALE,
                    reason=f"Book age {book_age_ms}ms exceeds limit {self.max_book_age_ms}ms",
                    book_age_ms=book_age_ms,
                    level_count=0,
                    should_proceed=False,
                    requires_aggressive=True
                )
            
            # Check level count
            bids = book_data.get('bids', [])
            asks = book_data.get('asks', [])
            level_count = len(bids) + len(asks)
            
            if level_count < self.min_levels:
                return BookValidationResult(
                    status=BookStatus.INSUFFICIENT,
                    reason=f"Book has {level_count} levels, minimum {self.min_levels} required",
                    book_age_ms=book_age_ms,
                    level_count=level_count,
                    should_proceed=False,
                    requires_aggressive=True
                )
            
            # Book is fresh and sufficient
            return BookValidationResult(
                status=BookStatus.FRESH,
                reason="Book is fresh and sufficient",
                book_age_ms=book_age_ms,
                level_count=level_count,
                should_proceed=True,
                requires_aggressive=False
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error validating book freshness: {e}")
            return BookValidationResult(
                status=BookStatus.STALE,
                reason=f"Book validation error: {e}",
                book_age_ms=999999,
                level_count=0,
                should_proceed=False,
                requires_aggressive=True
            )
    
    def should_proceed_with_order(self, book_data: Dict[str, Any], order_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Main gate function - returns (should_proceed, reason)
        """
        try:
            validation = self.validate_book_freshness(book_data)
            
            if validation.should_proceed:
                self.allowed_orders += 1
                self.logger.info(f"✅ BOOK_VALIDATION_PASSED: {validation.reason}")
                return True, validation.reason
            else:
                # Check if aggressive mode is allowed
                if validation.requires_aggressive and self.aggressive_allowed:
                    self.allowed_orders += 1
                    self.logger.warning(f"⚠️ BOOK_VALIDATION_AGGRESSIVE: {validation.reason} (AGGRESSIVE_ALLOWED=true)")
                    return True, f"{validation.reason} (AGGRESSIVE_ALLOWED=true)"
                else:
                    self.blocked_orders += 1
                    self.logger.warning(f"❌ BOOK_VALIDATION_BLOCKED: {validation.reason}")
                    
                    # Log structured JSON event
                    self.log_book_validation_failure(validation, order_data)
                    
                    return False, validation.reason
                    
        except Exception as e:
            self.logger.error(f"❌ Error in book validation gate: {e}")
            self.blocked_orders += 1
            return False, f"Book validation gate error: {e}"
    
    def log_book_validation_failure(self, validation: BookValidationResult, order_data: Dict[str, Any]):
        """
        Log structured JSON event for book validation failure
        """
        try:
            failure_event = {
                "event": "book_validation_failed",
                "timestamp": datetime.now().isoformat(),
                "status": validation.status.value,
                "reason": validation.reason,
                "book_age_ms": validation.book_age_ms,
                "level_count": validation.level_count,
                "requires_aggressive": validation.requires_aggressive,
                "aggressive_allowed": self.aggressive_allowed,
                "order_data": {
                    "side": order_data.get('side'),
                    "size": str(order_data.get('size', 0)),
                    "price": str(order_data.get('price', 0))
                },
                "blocked_orders": self.blocked_orders,
                "allowed_orders": self.allowed_orders
            }
            
            # Log as structured JSON
            self.logger.warning(f"🚫 BOOK_VALIDATION_FAILED: {json.dumps(failure_event)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging book validation failure: {e}")
    
    def get_guard_statistics(self) -> Dict[str, Any]:
        """
        Get book validation guard statistics
        """
        total_orders = self.blocked_orders + self.allowed_orders
        block_rate = (self.blocked_orders / total_orders * 100) if total_orders > 0 else 0
        
        return {
            "total_orders": total_orders,
            "blocked_orders": self.blocked_orders,
            "allowed_orders": self.allowed_orders,
            "block_rate_percent": round(block_rate, 2),
            "max_book_age_ms": self.max_book_age_ms,
            "min_levels": self.min_levels,
            "aggressive_allowed": self.aggressive_allowed,
            "last_updated": datetime.now().isoformat()
        }
    
    def log_guard_statistics(self):
        """
        Log current guard statistics
        """
        stats = self.get_guard_statistics()
        self.logger.info(f"📊 BOOK_VALIDATION_GUARD_STATS: {json.dumps(stats)}")

# Global guard instance
_book_guard = StaleBookGuard()

def should_proceed_with_order(book_data: Dict[str, Any], order_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Global function to check if order should proceed based on book validation
    """
    return _book_guard.should_proceed_with_order(book_data, order_data)

def get_book_guard() -> StaleBookGuard:
    """
    Get the global book guard instance
    """
    return _book_guard

# Demo function
def demo_stale_book_guard():
    """Demo the stale book guard"""
    print("📖 Stale Book Guard Demo")
    print("=" * 50)
    
    guard = StaleBookGuard()
    
    # Test 1: Fresh book
    print("🔍 Test 1: Fresh book")
    fresh_book = {
        'timestamp': datetime.now().isoformat(),
        'bids': [['0.5234', '1000'], ['0.5233', '1500'], ['0.5232', '2000']],
        'asks': [['0.5236', '1000'], ['0.5237', '1500'], ['0.5238', '2000']]
    }
    
    order_data = {
        'side': 'BUY',
        'size': 100,
        'price': 0.5235
    }
    
    should_proceed, reason = guard.should_proceed_with_order(fresh_book, order_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    
    # Test 2: Stale book
    print(f"\n🔍 Test 2: Stale book")
    stale_book = {
        'timestamp': (datetime.now() - timedelta(seconds=10)).isoformat(),  # 10 seconds old
        'bids': [['0.5234', '1000']],
        'asks': [['0.5236', '1000']]
    }
    
    should_proceed, reason = guard.should_proceed_with_order(stale_book, order_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    
    # Test 3: Insufficient levels
    print(f"\n🔍 Test 3: Insufficient levels")
    thin_book = {
        'timestamp': datetime.now().isoformat(),
        'bids': [['0.5234', '1000']],  # Only 1 level
        'asks': [['0.5236', '1000']]   # Only 1 level
    }
    
    should_proceed, reason = guard.should_proceed_with_order(thin_book, order_data)
    print(f"  Result: {'✅ ALLOWED' if should_proceed else '❌ BLOCKED'}")
    print(f"  Reason: {reason}")
    
    # Show statistics
    print(f"\n📊 Guard Statistics:")
    stats = guard.get_guard_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Stale Book Guard Demo Complete")

if __name__ == "__main__":
    demo_stale_book_guard()
