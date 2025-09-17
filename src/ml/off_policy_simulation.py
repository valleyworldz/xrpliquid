"""
Off-Policy Simulation Harness
Light simulation system for what-if analysis and policy learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class ExecutionPolicy(Enum):
    """Execution policies for simulation."""
    MAKER_ONLY = "maker_only"
    TAKER_ONLY = "taker_only"
    ADAPTIVE = "adaptive"
    TWAP = "twap"
    VWAP = "vwap"

@dataclass
class SimulatedTrade:
    """Simulated trade result."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    policy: ExecutionPolicy
    fill_price: float
    fill_quantity: float
    slippage_bps: float
    fill_time_ms: float
    rebate_earned: float
    success: bool

@dataclass
class SimulationResult:
    """Result of off-policy simulation."""
    policy: ExecutionPolicy
    total_trades: int
    successful_trades: int
    total_pnl: float
    avg_slippage_bps: float
    avg_fill_time_ms: float
    total_rebates: float
    success_rate: float
    sharpe_ratio: float

class MarketSimulator:
    """Simulates market conditions for off-policy evaluation."""
    
    def __init__(self, tick_data: pd.DataFrame = None):
        self.tick_data = tick_data
        self.current_time = datetime.now()
        self.current_price = 0.52
        self.current_spread = 0.001
        self.current_volatility = 0.02
        
        # Market impact parameters
        self.impact_alpha = 0.5
        self.impact_beta = 0.8
        
        # Liquidity parameters
        self.base_liquidity = 10000
        self.liquidity_volatility = 0.3
    
    def simulate_market_conditions(self, timestamp: datetime) -> Dict:
        """Simulate market conditions at a given timestamp."""
        
        # Simple market model
        time_factor = (timestamp.hour - 9) / 8.0  # 9 AM to 5 PM
        volatility_factor = 1.0 + 0.5 * np.sin(time_factor * np.pi)
        
        # Simulate price movement
        price_change = np.random.normal(0, self.current_volatility * volatility_factor)
        self.current_price *= (1 + price_change)
        
        # Simulate spread
        spread_change = np.random.normal(0, 0.0001)
        self.current_spread = max(0.0001, self.current_spread + spread_change)
        
        # Simulate liquidity
        liquidity = self.base_liquidity * (1 + np.random.normal(0, self.liquidity_volatility))
        
        return {
            'timestamp': timestamp,
            'price': self.current_price,
            'spread': self.current_spread,
            'volatility': self.current_volatility * volatility_factor,
            'liquidity': max(1000, liquidity),
            'bid': self.current_price - self.current_spread / 2,
            'ask': self.current_price + self.current_spread / 2
        }
    
    def simulate_execution(self, order_side: str, quantity: float, 
                          policy: ExecutionPolicy, market_conditions: Dict) -> SimulatedTrade:
        """Simulate order execution under given policy."""
        
        base_price = market_conditions['price']
        spread = market_conditions['spread']
        liquidity = market_conditions['liquidity']
        
        # Simulate execution based on policy
        if policy == ExecutionPolicy.MAKER_ONLY:
            fill_price, fill_quantity, fill_time, rebate = self._simulate_maker_execution(
                order_side, quantity, base_price, spread, liquidity
            )
        elif policy == ExecutionPolicy.TAKER_ONLY:
            fill_price, fill_quantity, fill_time, rebate = self._simulate_taker_execution(
                order_side, quantity, base_price, spread, liquidity
            )
        elif policy == ExecutionPolicy.ADAPTIVE:
            fill_price, fill_quantity, fill_time, rebate = self._simulate_adaptive_execution(
                order_side, quantity, base_price, spread, liquidity
            )
        elif policy == ExecutionPolicy.TWAP:
            fill_price, fill_quantity, fill_time, rebate = self._simulate_twap_execution(
                order_side, quantity, base_price, spread, liquidity
            )
        else:  # VWAP
            fill_price, fill_quantity, fill_time, rebate = self._simulate_vwap_execution(
                order_side, quantity, base_price, spread, liquidity
            )
        
        # Calculate slippage
        expected_price = base_price
        slippage_bps = abs(fill_price - expected_price) / expected_price * 10000
        
        # Determine success
        success = fill_quantity >= quantity * 0.95  # 95% fill rate threshold
        
        return SimulatedTrade(
            timestamp=market_conditions['timestamp'],
            symbol='XRP',
            side=order_side,
            quantity=quantity,
            price=base_price,
            policy=policy,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            slippage_bps=slippage_bps,
            fill_time_ms=fill_time,
            rebate_earned=rebate,
            success=success
        )
    
    def _simulate_maker_execution(self, side: str, quantity: float, 
                                 base_price: float, spread: float, liquidity: float) -> Tuple[float, float, float, float]:
        """Simulate maker-only execution."""
        
        # Maker orders get better prices but may not fill
        if side == 'buy':
            fill_price = base_price - spread / 2
        else:
            fill_price = base_price + spread / 2
        
        # Fill probability based on liquidity
        fill_probability = min(0.9, liquidity / (quantity * 10))
        fill_quantity = quantity if np.random.random() < fill_probability else quantity * 0.7
        
        # Longer fill time for maker orders
        fill_time = np.random.uniform(200, 1000)
        
        # Rebate for maker orders
        rebate = fill_quantity * fill_price * 0.0005
        
        return fill_price, fill_quantity, fill_time, rebate
    
    def _simulate_taker_execution(self, side: str, quantity: float, 
                                 base_price: float, spread: float, liquidity: float) -> Tuple[float, float, float, float]:
        """Simulate taker-only execution."""
        
        # Taker orders pay spread but fill quickly
        if side == 'buy':
            fill_price = base_price + spread / 2
        else:
            fill_price = base_price - spread / 2
        
        # High fill probability for taker orders
        fill_quantity = quantity * np.random.uniform(0.95, 1.0)
        
        # Fast fill time
        fill_time = np.random.uniform(50, 200)
        
        # No rebate for taker orders
        rebate = 0.0
        
        return fill_price, fill_quantity, fill_time, rebate
    
    def _simulate_adaptive_execution(self, side: str, quantity: float, 
                                   base_price: float, spread: float, liquidity: float) -> Tuple[float, float, float, float]:
        """Simulate adaptive execution (mix of maker/taker)."""
        
        # Adaptive policy: use maker if spread is wide, taker if narrow
        if spread > 0.002:  # Wide spread
            return self._simulate_maker_execution(side, quantity, base_price, spread, liquidity)
        else:  # Narrow spread
            return self._simulate_taker_execution(side, quantity, base_price, spread, liquidity)
    
    def _simulate_twap_execution(self, side: str, quantity: float, 
                               base_price: float, spread: float, liquidity: float) -> Tuple[float, float, float, float]:
        """Simulate TWAP execution."""
        
        # TWAP: split order over time
        num_slices = min(5, int(quantity / 100))
        slice_quantity = quantity / num_slices
        
        # Average execution over slices
        total_fill_price = 0
        total_fill_quantity = 0
        total_fill_time = 0
        total_rebate = 0
        
        for i in range(num_slices):
            # Simulate each slice
            if side == 'buy':
                fill_price = base_price + spread / 2 * (1 + i * 0.1)
            else:
                fill_price = base_price - spread / 2 * (1 + i * 0.1)
            
            fill_quantity = slice_quantity * np.random.uniform(0.9, 1.0)
            fill_time = np.random.uniform(100, 300)
            rebate = fill_quantity * fill_price * 0.0002  # Lower rebate for TWAP
            
            total_fill_price += fill_price * fill_quantity
            total_fill_quantity += fill_quantity
            total_fill_time += fill_time
            total_rebate += rebate
        
        avg_fill_price = total_fill_price / total_fill_quantity if total_fill_quantity > 0 else base_price
        
        return avg_fill_price, total_fill_quantity, total_fill_time, total_rebate
    
    def _simulate_vwap_execution(self, side: str, quantity: float, 
                               base_price: float, spread: float, liquidity: float) -> Tuple[float, float, float, float]:
        """Simulate VWAP execution."""
        
        # VWAP: execute at volume-weighted average price
        # Simplified: assume we get close to VWAP
        vwap_price = base_price + np.random.normal(0, spread * 0.1)
        
        fill_quantity = quantity * np.random.uniform(0.92, 0.98)
        fill_time = np.random.uniform(150, 400)
        rebate = fill_quantity * vwap_price * 0.0003
        
        return vwap_price, fill_quantity, fill_time, rebate

class OffPolicySimulator:
    """Main off-policy simulation harness."""
    
    def __init__(self, market_simulator: MarketSimulator):
        self.market_simulator = market_simulator
        self.simulation_results = {}
        self.trade_history = []
    
    def run_policy_comparison(self, orders: List[Dict], 
                            policies: List[ExecutionPolicy]) -> Dict:
        """Run comparison simulation across multiple policies."""
        
        results = {}
        
        for policy in policies:
            print(f"Simulating policy: {policy.value}")
            
            policy_results = []
            total_pnl = 0.0
            total_slippage = 0.0
            total_fill_time = 0.0
            total_rebates = 0.0
            successful_trades = 0
            
            for order in orders:
                # Simulate market conditions
                market_conditions = self.market_simulator.simulate_market_conditions(
                    order['timestamp']
                )
                
                # Simulate execution
                simulated_trade = self.market_simulator.simulate_execution(
                    order['side'], order['quantity'], policy, market_conditions
                )
                
                # Calculate PnL
                if simulated_trade.success:
                    pnl = self._calculate_trade_pnl(simulated_trade, market_conditions)
                    total_pnl += pnl
                    successful_trades += 1
                
                total_slippage += simulated_trade.slippage_bps
                total_fill_time += simulated_trade.fill_time_ms
                total_rebates += simulated_trade.rebate_earned
                
                policy_results.append(simulated_trade)
            
            # Calculate summary statistics
            num_trades = len(orders)
            success_rate = successful_trades / num_trades if num_trades > 0 else 0.0
            avg_slippage = total_slippage / num_trades if num_trades > 0 else 0.0
            avg_fill_time = total_fill_time / num_trades if num_trades > 0 else 0.0
            
            # Calculate Sharpe ratio (simplified)
            returns = [self._calculate_trade_pnl(trade, market_conditions) 
                      for trade in policy_results if trade.success]
            sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0
            
            results[policy.value] = SimulationResult(
                policy=policy,
                total_trades=num_trades,
                successful_trades=successful_trades,
                total_pnl=total_pnl,
                avg_slippage_bps=avg_slippage,
                avg_fill_time_ms=avg_fill_time,
                total_rebates=total_rebates,
                success_rate=success_rate,
                sharpe_ratio=sharpe_ratio
            )
        
        self.simulation_results = results
        return results
    
    def _calculate_trade_pnl(self, trade: SimulatedTrade, market_conditions: Dict) -> float:
        """Calculate PnL for a simulated trade."""
        
        # Simple PnL calculation
        price_change = trade.fill_price - trade.price
        pnl = trade.fill_quantity * price_change * (1 if trade.side == 'buy' else -1)
        
        # Add rebate
        pnl += trade.rebate_earned
        
        # Subtract slippage cost
        slippage_cost = trade.fill_quantity * trade.price * trade.slippage_bps / 10000
        pnl -= slippage_cost
        
        return pnl
    
    def generate_comparison_report(self) -> str:
        """Generate HTML comparison report."""
        
        if not self.simulation_results:
            return "<p>No simulation results available</p>"
        
        # Create comparison table
        table_rows = []
        for policy_name, result in self.simulation_results.items():
            table_rows.append(f"""
                <tr>
                    <td>{policy_name}</td>
                    <td>{result.success_rate:.1%}</td>
                    <td>${result.total_pnl:.2f}</td>
                    <td>{result.avg_slippage_bps:.2f} bps</td>
                    <td>{result.avg_fill_time_ms:.0f} ms</td>
                    <td>${result.total_rebates:.2f}</td>
                    <td>{result.sharpe_ratio:.2f}</td>
                </tr>
            """)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Off-Policy Simulation Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
                .worst {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>📊 Off-Policy Simulation Results</h1>
            
            <table>
                <thead>
                    <tr>
                        <th>Policy</th>
                        <th>Success Rate</th>
                        <th>Total PnL</th>
                        <th>Avg Slippage</th>
                        <th>Avg Fill Time</th>
                        <th>Total Rebates</th>
                        <th>Sharpe Ratio</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
            
            <h3>Key Insights</h3>
            <ul>
                <li><strong>Best Success Rate:</strong> {max(self.simulation_results.items(), key=lambda x: x[1].success_rate)[0]}</li>
                <li><strong>Best PnL:</strong> {max(self.simulation_results.items(), key=lambda x: x[1].total_pnl)[0]}</li>
                <li><strong>Lowest Slippage:</strong> {min(self.simulation_results.items(), key=lambda x: x[1].avg_slippage_bps)[0]}</li>
                <li><strong>Fastest Execution:</strong> {min(self.simulation_results.items(), key=lambda x: x[1].avg_fill_time_ms)[0]}</li>
            </ul>
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = Path("reports/ml/off_policy_simulation_report.html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Off-policy simulation report saved to {html_path}")
        return html
    
    def save_simulation_results(self, filepath: str = "reports/ml/simulation_results.json"):
        """Save simulation results to JSON."""
        
        results_data = {}
        for policy_name, result in self.simulation_results.items():
            results_data[policy_name] = {
                'policy': result.policy.value,
                'total_trades': result.total_trades,
                'successful_trades': result.successful_trades,
                'total_pnl': result.total_pnl,
                'avg_slippage_bps': result.avg_slippage_bps,
                'avg_fill_time_ms': result.avg_fill_time_ms,
                'total_rebates': result.total_rebates,
                'success_rate': result.success_rate,
                'sharpe_ratio': result.sharpe_ratio
            }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results_data
            }, f, indent=2)
        
        logger.info(f"Simulation results saved to {filepath}")

def main():
    """Demonstrate off-policy simulation harness."""
    
    # Initialize market simulator
    market_simulator = MarketSimulator()
    
    # Initialize off-policy simulator
    simulator = OffPolicySimulator(market_simulator)
    
    # Create sample orders
    orders = []
    base_time = datetime.now()
    
    for i in range(100):
        order = {
            'timestamp': base_time + timedelta(minutes=i),
            'side': 'buy' if i % 2 == 0 else 'sell',
            'quantity': np.random.uniform(100, 1000)
        }
        orders.append(order)
    
    # Define policies to test
    policies = [
        ExecutionPolicy.MAKER_ONLY,
        ExecutionPolicy.TAKER_ONLY,
        ExecutionPolicy.ADAPTIVE,
        ExecutionPolicy.TWAP,
        ExecutionPolicy.VWAP
    ]
    
    # Run simulation
    print("🧪 Running Off-Policy Simulation")
    print("=" * 50)
    
    results = simulator.run_policy_comparison(orders, policies)
    
    # Print results
    for policy_name, result in results.items():
        print(f"\n{policy_name.upper()}:")
        print(f"  Success Rate: {result.success_rate:.1%}")
        print(f"  Total PnL: ${result.total_pnl:.2f}")
        print(f"  Avg Slippage: {result.avg_slippage_bps:.2f} bps")
        print(f"  Avg Fill Time: {result.avg_fill_time_ms:.0f} ms")
        print(f"  Total Rebates: ${result.total_rebates:.2f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    # Generate reports
    simulator.generate_comparison_report()
    simulator.save_simulation_results()
    
    print("\n✅ Off-policy simulation completed")
    print("   HTML report: reports/ml/off_policy_simulation_report.html")
    print("   JSON results: reports/ml/simulation_results.json")
    
    return 0

if __name__ == "__main__":
    exit(main())
