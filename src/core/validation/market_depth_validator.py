"""
Market Depth Validation
Pre-trade book depth sanity checks to fail fast before generating TP/SLs.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class DepthValidationResult:
    """Result of market depth validation."""
    is_valid: bool
    confidence_score: float
    warnings: List[str]
    errors: List[str]
    depth_metrics: Dict[str, float]
    recommendation: str

@dataclass
class OrderBookLevel:
    """Individual order book level."""
    price: float
    quantity: float
    side: str
    timestamp: datetime

class MarketDepthValidator:
    """Validates market depth before trade execution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.validation_history = []
        
        # Depth thresholds
        self.min_depth_ratio = self.config.get('min_depth_ratio', 0.1)
        self.max_spread_ratio = self.config.get('max_spread_ratio', 0.02)
        self.min_liquidity_threshold = self.config.get('min_liquidity_threshold', 1000)
        self.max_price_deviation = self.config.get('max_price_deviation', 0.05)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'min_depth_ratio': 0.1,  # 10% of order size
            'max_spread_ratio': 0.02,  # 2% max spread
            'min_liquidity_threshold': 1000,  # Minimum liquidity
            'max_price_deviation': 0.05,  # 5% max price deviation
            'depth_levels_to_check': 5,  # Check top 5 levels
            'min_confidence_score': 0.7,  # Minimum confidence for execution
            'enable_staleness_check': True,
            'max_staleness_seconds': 30
        }
    
    def validate_order_book(self, order_book: Dict[str, List[Dict]], 
                          order_size: float, order_side: str) -> DepthValidationResult:
        """Validate order book depth for a specific order."""
        
        warnings = []
        errors = []
        depth_metrics = {}
        
        try:
            # Extract bids and asks
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                errors.append("Incomplete order book: missing bids or asks")
                return DepthValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    warnings=warnings,
                    errors=errors,
                    depth_metrics=depth_metrics,
                    recommendation="Do not execute - incomplete order book"
                )
            
            # Check order book staleness
            if self.config.get('enable_staleness_check', True):
                staleness_check = self._check_order_book_staleness(order_book)
                if not staleness_check['is_fresh']:
                    errors.append(f"Order book is stale: {staleness_check['age_seconds']}s old")
            
            # Calculate basic metrics
            best_bid = bids[0]['price'] if bids else 0
            best_ask = asks[0]['price'] if asks else 0
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            depth_metrics.update({
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_ratio': spread / mid_price if mid_price > 0 else 0,
                'mid_price': mid_price
            })
            
            # Validate spread
            if depth_metrics['spread_ratio'] > self.max_spread_ratio:
                errors.append(f"Spread too wide: {depth_metrics['spread_ratio']:.4f} > {self.max_spread_ratio}")
            
            # Validate depth for order side
            if order_side == 'buy':
                depth_validation = self._validate_buy_depth(asks, order_size, mid_price)
            else:
                depth_validation = self._validate_sell_depth(bids, order_size, mid_price)
            
            warnings.extend(depth_validation['warnings'])
            errors.extend(depth_validation['errors'])
            depth_metrics.update(depth_validation['metrics'])
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(depth_metrics, warnings, errors)
            
            # Determine recommendation
            recommendation = self._get_recommendation(confidence_score, errors, warnings)
            
            # Create validation result
            result = DepthValidationResult(
                is_valid=len(errors) == 0 and confidence_score >= self.config.get('min_confidence_score', 0.7),
                confidence_score=confidence_score,
                warnings=warnings,
                errors=errors,
                depth_metrics=depth_metrics,
                recommendation=recommendation
            )
            
            # Store validation history
            self.validation_history.append({
                'timestamp': datetime.now(),
                'order_size': order_size,
                'order_side': order_side,
                'result': result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Market depth validation failed: {e}")
            return DepthValidationResult(
                is_valid=False,
                confidence_score=0.0,
                warnings=warnings,
                errors=[f"Validation error: {str(e)}"],
                depth_metrics=depth_metrics,
                recommendation="Do not execute - validation error"
            )
    
    def _check_order_book_staleness(self, order_book: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Check if order book data is stale."""
        
        # Get timestamp from order book
        timestamp = order_book.get('timestamp')
        if not timestamp:
            return {'is_fresh': False, 'age_seconds': None}
        
        # Calculate age
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        age_seconds = (datetime.now() - timestamp).total_seconds()
        max_age = self.config.get('max_staleness_seconds', 30)
        
        return {
            'is_fresh': age_seconds <= max_age,
            'age_seconds': age_seconds
        }
    
    def _validate_buy_depth(self, asks: List[Dict], order_size: float, mid_price: float) -> Dict[str, Any]:
        """Validate depth for buy orders."""
        
        warnings = []
        errors = []
        metrics = {}
        
        # Calculate cumulative depth
        cumulative_depth = 0
        levels_checked = 0
        max_levels = self.config.get('depth_levels_to_check', 5)
        
        for level in asks[:max_levels]:
            cumulative_depth += level['quantity']
            levels_checked += 1
            
            # Check if we have enough depth
            if cumulative_depth >= order_size:
                break
        
        # Calculate depth ratio
        depth_ratio = cumulative_depth / order_size if order_size > 0 else 0
        metrics['depth_ratio'] = depth_ratio
        metrics['cumulative_depth'] = cumulative_depth
        metrics['levels_checked'] = levels_checked
        
        # Validate depth ratio
        if depth_ratio < self.min_depth_ratio:
            errors.append(f"Insufficient depth: {depth_ratio:.4f} < {self.min_depth_ratio}")
        
        # Check for depth concentration
        if levels_checked > 0:
            first_level_ratio = asks[0]['quantity'] / order_size if order_size > 0 else 0
            if first_level_ratio > 0.8:  # 80% of order on first level
                warnings.append(f"High depth concentration: {first_level_ratio:.4f} on first level")
        
        # Check price impact
        if levels_checked > 1:
            price_impact = (asks[levels_checked - 1]['price'] - mid_price) / mid_price
            metrics['price_impact'] = price_impact
            
            if price_impact > self.max_price_deviation:
                warnings.append(f"High price impact: {price_impact:.4f} > {self.max_price_deviation}")
        
        return {
            'warnings': warnings,
            'errors': errors,
            'metrics': metrics
        }
    
    def _validate_sell_depth(self, bids: List[Dict], order_size: float, mid_price: float) -> Dict[str, Any]:
        """Validate depth for sell orders."""
        
        warnings = []
        errors = []
        metrics = {}
        
        # Calculate cumulative depth
        cumulative_depth = 0
        levels_checked = 0
        max_levels = self.config.get('depth_levels_to_check', 5)
        
        for level in bids[:max_levels]:
            cumulative_depth += level['quantity']
            levels_checked += 1
            
            # Check if we have enough depth
            if cumulative_depth >= order_size:
                break
        
        # Calculate depth ratio
        depth_ratio = cumulative_depth / order_size if order_size > 0 else 0
        metrics['depth_ratio'] = depth_ratio
        metrics['cumulative_depth'] = cumulative_depth
        metrics['levels_checked'] = levels_checked
        
        # Validate depth ratio
        if depth_ratio < self.min_depth_ratio:
            errors.append(f"Insufficient depth: {depth_ratio:.4f} < {self.min_depth_ratio}")
        
        # Check for depth concentration
        if levels_checked > 0:
            first_level_ratio = bids[0]['quantity'] / order_size if order_size > 0 else 0
            if first_level_ratio > 0.8:  # 80% of order on first level
                warnings.append(f"High depth concentration: {first_level_ratio:.4f} on first level")
        
        # Check price impact
        if levels_checked > 1:
            price_impact = (mid_price - bids[levels_checked - 1]['price']) / mid_price
            metrics['price_impact'] = price_impact
            
            if price_impact > self.max_price_deviation:
                warnings.append(f"High price impact: {price_impact:.4f} > {self.max_price_deviation}")
        
        return {
            'warnings': warnings,
            'errors': errors,
            'metrics': metrics
        }
    
    def _calculate_confidence_score(self, depth_metrics: Dict[str, float], 
                                  warnings: List[str], errors: List[str]) -> float:
        """Calculate confidence score based on depth metrics."""
        
        # Start with base score
        confidence = 1.0
        
        # Penalize for errors
        confidence -= len(errors) * 0.3
        
        # Penalize for warnings
        confidence -= len(warnings) * 0.1
        
        # Adjust based on depth ratio
        depth_ratio = depth_metrics.get('depth_ratio', 0)
        if depth_ratio < 0.5:
            confidence -= 0.2
        elif depth_ratio < 1.0:
            confidence -= 0.1
        
        # Adjust based on spread
        spread_ratio = depth_metrics.get('spread_ratio', 0)
        if spread_ratio > 0.01:  # 1%
            confidence -= 0.1
        elif spread_ratio > 0.005:  # 0.5%
            confidence -= 0.05
        
        # Adjust based on price impact
        price_impact = depth_metrics.get('price_impact', 0)
        if price_impact > 0.02:  # 2%
            confidence -= 0.15
        elif price_impact > 0.01:  # 1%
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _get_recommendation(self, confidence_score: float, errors: List[str], warnings: List[str]) -> str:
        """Get execution recommendation based on validation results."""
        
        if errors:
            return "Do not execute - critical issues detected"
        
        if confidence_score >= 0.9:
            return "Execute with confidence"
        elif confidence_score >= 0.7:
            return "Execute with caution"
        elif confidence_score >= 0.5:
            return "Consider reducing order size"
        else:
            return "Do not execute - low confidence"
    
    def validate_tp_sl_feasibility(self, order_book: Dict[str, List[Dict]], 
                                 tp_price: float, sl_price: float, 
                                 order_side: str) -> Dict[str, Any]:
        """Validate feasibility of TP/SL orders."""
        
        # Get current price
        best_bid = order_book.get('bids', [{}])[0].get('price', 0)
        best_ask = order_book.get('asks', [{}])[0].get('price', 0)
        current_price = (best_bid + best_ask) / 2
        
        # Validate TP/SL logic
        if order_side == 'buy':
            if tp_price <= current_price:
                return {'feasible': False, 'reason': 'TP price must be above current price for buy orders'}
            if sl_price >= current_price:
                return {'feasible': False, 'reason': 'SL price must be below current price for buy orders'}
        else:  # sell
            if tp_price >= current_price:
                return {'feasible': False, 'reason': 'TP price must be below current price for sell orders'}
            if sl_price <= current_price:
                return {'feasible': False, 'reason': 'SL price must be above current price for sell orders'}
        
        # Check if TP/SL prices are within reasonable range
        tp_distance = abs(tp_price - current_price) / current_price
        sl_distance = abs(sl_price - current_price) / current_price
        
        if tp_distance > 0.1:  # 10%
            return {'feasible': False, 'reason': 'TP price too far from current price'}
        
        if sl_distance > 0.05:  # 5%
            return {'feasible': False, 'reason': 'SL price too far from current price'}
        
        return {'feasible': True, 'reason': 'TP/SL orders are feasible'}
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        
        if not self.validation_history:
            return {'total_validations': 0}
        
        total_validations = len(self.validation_history)
        successful_validations = len([v for v in self.validation_history if v['result'].is_valid])
        avg_confidence = np.mean([v['result'].confidence_score for v in self.validation_history])
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
            'avg_confidence_score': avg_confidence,
            'config': self.config
        }

def main():
    """Demonstrate market depth validation."""
    
    # Initialize validator
    validator = MarketDepthValidator()
    
    print("🧪 Testing Market Depth Validation")
    print("=" * 50)
    
    # Create sample order book
    sample_order_book = {
        'bids': [
            {'price': 0.519, 'quantity': 1000, 'timestamp': datetime.now()},
            {'price': 0.518, 'quantity': 2000, 'timestamp': datetime.now()},
            {'price': 0.517, 'quantity': 1500, 'timestamp': datetime.now()},
            {'price': 0.516, 'quantity': 3000, 'timestamp': datetime.now()},
            {'price': 0.515, 'quantity': 1000, 'timestamp': datetime.now()}
        ],
        'asks': [
            {'price': 0.521, 'quantity': 800, 'timestamp': datetime.now()},
            {'price': 0.522, 'quantity': 1200, 'timestamp': datetime.now()},
            {'price': 0.523, 'quantity': 2000, 'timestamp': datetime.now()},
            {'price': 0.524, 'quantity': 1500, 'timestamp': datetime.now()},
            {'price': 0.525, 'quantity': 1000, 'timestamp': datetime.now()}
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    # Test buy order validation
    buy_result = validator.validate_order_book(sample_order_book, 500, 'buy')
    print(f"Buy Order (500 XRP):")
    print(f"  Valid: {buy_result.is_valid}")
    print(f"  Confidence: {buy_result.confidence_score:.3f}")
    print(f"  Recommendation: {buy_result.recommendation}")
    if buy_result.warnings:
        print(f"  Warnings: {buy_result.warnings}")
    if buy_result.errors:
        print(f"  Errors: {buy_result.errors}")
    
    # Test sell order validation
    sell_result = validator.validate_order_book(sample_order_book, 1000, 'sell')
    print(f"\nSell Order (1000 XRP):")
    print(f"  Valid: {sell_result.is_valid}")
    print(f"  Confidence: {sell_result.confidence_score:.3f}")
    print(f"  Recommendation: {sell_result.recommendation}")
    if sell_result.warnings:
        print(f"  Warnings: {sell_result.warnings}")
    if sell_result.errors:
        print(f"  Errors: {sell_result.errors}")
    
    # Test TP/SL feasibility
    tp_sl_result = validator.validate_tp_sl_feasibility(
        sample_order_book, 0.53, 0.51, 'buy'
    )
    print(f"\nTP/SL Feasibility (TP: 0.53, SL: 0.51, Buy):")
    print(f"  Feasible: {tp_sl_result['feasible']}")
    print(f"  Reason: {tp_sl_result['reason']}")
    
    # Get validation stats
    stats = validator.get_validation_stats()
    print(f"\n📊 Validation Stats: {stats}")
    
    return 0

if __name__ == "__main__":
    exit(main())
