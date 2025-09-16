"""
Microstructure Router
Implements maker/taker routing with spread/depth policy switches.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrostructureRouter:
    """Routes orders based on market microstructure conditions."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.reports_dir = self.repo_root / "reports"
        
        # Routing parameters
        self.spread_threshold_bps = 2.0
        self.depth_threshold_usd = 10000
        self.volatility_threshold = 0.02
        
        # Routing statistics
        self.routing_stats = {
            'post_only_attempts': 0,
            'post_only_successes': 0,
            'taker_fallbacks': 0,
            'urgency_overrides': 0,
            'policy_switches': 0
        }
        
        # Load impact model
        self.impact_model = self.load_impact_model()
        
        # Load opportunity cost data
        self.opportunity_cost_data = self.load_opportunity_cost_data()
    
    def load_impact_model(self) -> Dict:
        """Load impact model from reports."""
        impact_file = self.reports_dir / "microstructure" / "impact_residuals.json"
        
        if impact_file.exists():
            with open(impact_file, 'r') as f:
                return json.load(f)
        else:
            # Default impact model
            return {
                'impact_analysis': {
                    'participation_rate_bps': [1, 5, 10, 20, 50, 100],
                    'expected_impact_bps': [0.1, 0.3, 0.6, 1.2, 2.5, 4.8]
                }
            }
    
    def load_opportunity_cost_data(self) -> Dict:
        """Load opportunity cost data from reports."""
        opportunity_file = self.reports_dir / "maker_taker" / "opportunity_cost.json"
        
        if opportunity_file.exists():
            with open(opportunity_file, 'r') as f:
                return json.load(f)
        else:
            # Default opportunity cost data
            return {
                'maker_taker_analysis': {
                    'maker_ratio': 0.70
                },
                'rebate_analysis': {
                    'maker_rebate_bps': 0.5
                }
            }
    
    def calculate_market_impact(self, notional: float, participation_rate: float) -> float:
        """Calculate expected market impact for a trade."""
        impact_analysis = self.impact_model.get('impact_analysis', {})
        participation_rates = impact_analysis.get('participation_rate_bps', [])
        expected_impacts = impact_analysis.get('expected_impact_bps', [])
        
        if not participation_rates or not expected_impacts:
            return 0.5  # Default impact
        
        # Interpolate impact based on participation rate
        for i in range(len(participation_rates) - 1):
            if participation_rates[i] <= participation_rate <= participation_rates[i + 1]:
                # Linear interpolation
                x1, y1 = participation_rates[i], expected_impacts[i]
                x2, y2 = participation_rates[i + 1], expected_impacts[i + 1]
                impact = y1 + (y2 - y1) * (participation_rate - x1) / (x2 - x1)
                return impact
        
        # Extrapolate if outside range
        if participation_rate < participation_rates[0]:
            return expected_impacts[0]
        else:
            return expected_impacts[-1]
    
    def assess_market_conditions(self, order_book: Dict, volatility: float) -> Dict:
        """Assess current market conditions for routing decisions."""
        best_bid = order_book.get('best_bid', 0)
        best_ask = order_book.get('best_ask', 0)
        bid_size = order_book.get('bid_size', 0)
        ask_size = order_book.get('ask_size', 0)
        
        # Calculate spread
        spread_bps = ((best_ask - best_bid) / best_bid) * 10000 if best_bid > 0 else 0
        
        # Calculate depth
        total_depth = (bid_size + ask_size) * best_bid
        
        # Assess conditions
        conditions = {
            'spread_bps': spread_bps,
            'total_depth_usd': total_depth,
            'volatility': volatility,
            'spread_wide': spread_bps > self.spread_threshold_bps,
            'depth_shallow': total_depth < self.depth_threshold_usd,
            'volatility_high': volatility > self.volatility_threshold
        }
        
        return conditions
    
    def determine_routing_strategy(self, order: Dict, market_conditions: Dict) -> str:
        """Determine optimal routing strategy for an order."""
        order_side = order.get('side', 'buy')
        order_size = order.get('size', 0)
        urgency = order.get('urgency', 'normal')
        
        # Default to post-only (maker)
        strategy = 'post_only'
        
        # Check for urgency override
        if urgency == 'high':
            strategy = 'taker'
            self.routing_stats['urgency_overrides'] += 1
            logger.info(f"🚨 Urgency override: routing to taker")
        
        # Check market conditions for maker viability
        elif market_conditions['spread_wide'] or market_conditions['depth_shallow']:
            strategy = 'taker'
            self.routing_stats['taker_fallbacks'] += 1
            logger.info(f"📉 Market conditions poor for maker: routing to taker")
        
        # Check volatility regime
        elif market_conditions['volatility_high']:
            # In high volatility, prefer taker for immediate execution
            strategy = 'taker'
            self.routing_stats['taker_fallbacks'] += 1
            logger.info(f"📊 High volatility: routing to taker")
        
        return strategy
    
    def execute_routing_decision(self, order: Dict, strategy: str) -> Dict:
        """Execute the routing decision and track results."""
        order_id = order.get('order_id', 'unknown')
        
        if strategy == 'post_only':
            self.routing_stats['post_only_attempts'] += 1
            
            # Simulate post-only execution (in real implementation, this would be actual order placement)
            success = self.simulate_post_only_execution(order)
            
            if success:
                self.routing_stats['post_only_successes'] += 1
                logger.info(f"✅ Post-only success for order {order_id}")
            else:
                logger.info(f"❌ Post-only failed for order {order_id}")
        
        elif strategy == 'taker':
            # Simulate taker execution
            success = self.simulate_taker_execution(order)
            logger.info(f"⚡ Taker execution for order {order_id}: {'success' if success else 'failed'}")
        
        return {
            'order_id': order_id,
            'strategy': strategy,
            'timestamp': datetime.now().isoformat(),
            'success': success if 'success' in locals() else False
        }
    
    def simulate_post_only_execution(self, order: Dict) -> bool:
        """Simulate post-only order execution."""
        # In real implementation, this would place a post-only order
        # For simulation, assume 82% success rate
        import random
        return random.random() < 0.82
    
    def simulate_taker_execution(self, order: Dict) -> bool:
        """Simulate taker order execution."""
        # In real implementation, this would place a market order
        # For simulation, assume 95% success rate
        import random
        return random.random() < 0.95
    
    def log_policy_switch(self, old_conditions: Dict, new_conditions: Dict, reason: str):
        """Log when routing policy switches."""
        self.routing_stats['policy_switches'] += 1
        
        switch_log = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'old_conditions': old_conditions,
            'new_conditions': new_conditions,
            'switch_count': self.routing_stats['policy_switches']
        }
        
        logger.info(f"🔄 Policy switch: {reason}")
        
        # Save switch log
        switches_dir = self.reports_dir / "microstructure"
        switches_dir.mkdir(exist_ok=True)
        switch_file = switches_dir / "policy_switches.jsonl"
        
        with open(switch_file, 'a') as f:
            f.write(json.dumps(switch_log) + '\n')
    
    def get_routing_statistics(self) -> Dict:
        """Get current routing statistics."""
        total_attempts = self.routing_stats['post_only_attempts']
        success_rate = (self.routing_stats['post_only_successes'] / total_attempts) if total_attempts > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'routing_stats': self.routing_stats,
            'post_only_success_rate': success_rate,
            'taker_fallback_rate': (self.routing_stats['taker_fallbacks'] / total_attempts) if total_attempts > 0 else 0,
            'urgency_override_rate': (self.routing_stats['urgency_overrides'] / total_attempts) if total_attempts > 0 else 0
        }
    
    def save_routing_statistics(self) -> Path:
        """Save routing statistics to file."""
        stats = self.get_routing_statistics()
        
        stats_dir = self.reports_dir / "maker_taker"
        stats_dir.mkdir(exist_ok=True)
        stats_file = stats_dir / "routing_statistics.json"
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"💾 Routing statistics saved: {stats_file}")
        return stats_file


def main():
    """Main function to demonstrate microstructure routing."""
    router = MicrostructureRouter()
    
    # Simulate some orders
    orders = [
        {'order_id': 'order_001', 'side': 'buy', 'size': 1000, 'urgency': 'normal'},
        {'order_id': 'order_002', 'side': 'sell', 'size': 500, 'urgency': 'high'},
        {'order_id': 'order_003', 'side': 'buy', 'size': 2000, 'urgency': 'normal'}
    ]
    
    # Simulate market conditions
    market_conditions = {
        'best_bid': 0.50,
        'best_ask': 0.501,
        'bid_size': 5000,
        'ask_size': 5000,
        'volatility': 0.015
    }
    
    # Process orders
    for order in orders:
        conditions = router.assess_market_conditions(market_conditions, market_conditions['volatility'])
        strategy = router.determine_routing_strategy(order, conditions)
        result = router.execute_routing_decision(order, strategy)
        print(f"Order {order['order_id']}: {strategy} -> {result['success']}")
    
    # Save statistics
    router.save_routing_statistics()
    print("✅ Microstructure routing demonstration completed")


if __name__ == "__main__":
    main()
