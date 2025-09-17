"""
Auto-Optimization Bootstrap
Bootstrapping with historical synthetic trades to feed optimizer more data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SyntheticTradeGenerator:
    """Generates synthetic historical trades for optimization bootstrap."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for synthetic trade generation."""
        return {
            'trade_types': ['momentum', 'mean_reversion', 'funding_arb', 'breakout'],
            'success_rates': {
                'momentum': 0.65,
                'mean_reversion': 0.58,
                'funding_arb': 0.72,
                'breakout': 0.62
            },
            'avg_returns': {
                'momentum': 0.002,
                'mean_reversion': 0.0015,
                'funding_arb': 0.0008,
                'breakout': 0.003
            },
            'volatility_multipliers': {
                'momentum': 1.2,
                'mean_reversion': 0.8,
                'funding_arb': 0.5,
                'breakout': 1.5
            },
            'trade_frequency': {
                'momentum': 0.3,
                'mean_reversion': 0.25,
                'funding_arb': 0.2,
                'breakout': 0.25
            }
        }
    
    def generate_synthetic_trades(self, market_data: pd.DataFrame, 
                                num_trades: int = 1000) -> pd.DataFrame:
        """Generate synthetic historical trades."""
        
        logger.info(f"Generating {num_trades} synthetic trades")
        
        trades = []
        trade_id = 0
        
        for _ in range(num_trades):
            # Select trade type based on frequency
            trade_type = np.random.choice(
                list(self.config['trade_types']),
                p=[self.config['trade_frequency'][t] for t in self.config['trade_types']]
            )
            
            # Generate trade parameters
            trade = self._generate_single_trade(trade_id, trade_type, market_data)
            trades.append(trade)
            trade_id += 1
        
        trades_df = pd.DataFrame(trades)
        
        # Add derived metrics
        trades_df = self._add_derived_metrics(trades_df)
        
        logger.info(f"Generated {len(trades_df)} synthetic trades")
        
        return trades_df
    
    def _generate_single_trade(self, trade_id: int, trade_type: str, 
                             market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a single synthetic trade."""
        
        # Random timestamp
        timestamp = np.random.choice(market_data.index)
        
        # Get market context at timestamp
        market_context = self._get_market_context(market_data, timestamp)
        
        # Generate trade parameters based on type
        if trade_type == 'momentum':
            trade = self._generate_momentum_trade(trade_id, timestamp, market_context)
        elif trade_type == 'mean_reversion':
            trade = self._generate_mean_reversion_trade(trade_id, timestamp, market_context)
        elif trade_type == 'funding_arb':
            trade = self._generate_funding_arb_trade(trade_id, timestamp, market_context)
        elif trade_type == 'breakout':
            trade = self._generate_breakout_trade(trade_id, timestamp, market_context)
        else:
            trade = self._generate_generic_trade(trade_id, timestamp, market_context)
        
        # Add market context
        trade.update(market_context)
        
        return trade
    
    def _get_market_context(self, market_data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Get market context at a specific timestamp."""
        
        # Get data around timestamp
        window_data = market_data.loc[:timestamp].tail(20)
        
        if len(window_data) < 5:
            return {
                'volatility': 0.02,
                'trend': 0.0,
                'volume_ratio': 1.0,
                'spread': 0.001,
                'liquidity': 1000
            }
        
        # Calculate market metrics
        volatility = window_data['close'].pct_change().std()
        trend = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
        volume_ratio = window_data['volume'].iloc[-1] / window_data['volume'].mean()
        spread = (window_data['high'] - window_data['low']).mean()
        liquidity = window_data['volume'].mean()
        
        return {
            'volatility': volatility,
            'trend': trend,
            'volume_ratio': volume_ratio,
            'spread': spread,
            'liquidity': liquidity
        }
    
    def _generate_momentum_trade(self, trade_id: int, timestamp: pd.Timestamp, 
                               market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate momentum trade."""
        
        success_rate = self.config['success_rates']['momentum']
        avg_return = self.config['avg_returns']['momentum']
        vol_mult = self.config['volatility_multipliers']['momentum']
        
        # Momentum trades are more successful in trending markets
        trend_boost = max(0, market_context['trend'] * 0.1)
        adjusted_success_rate = min(0.9, success_rate + trend_boost)
        
        # Generate outcome
        is_successful = np.random.random() < adjusted_success_rate
        
        if is_successful:
            # Positive return with momentum characteristics
            base_return = avg_return * (1 + market_context['trend'] * 0.5)
            volatility = market_context['volatility'] * vol_mult
            return_pct = np.random.normal(base_return, volatility)
        else:
            # Negative return
            volatility = market_context['volatility'] * vol_mult
            return_pct = np.random.normal(-avg_return * 0.5, volatility)
        
        return {
            'trade_id': trade_id,
            'timestamp': timestamp,
            'trade_type': 'momentum',
            'side': 'buy' if np.random.random() > 0.5 else 'sell',
            'quantity': np.random.uniform(100, 1000),
            'entry_price': 0.52 + np.random.normal(0, 0.001),
            'exit_price': 0.52 + np.random.normal(0, 0.001),
            'return_pct': return_pct,
            'is_successful': is_successful,
            'duration_minutes': np.random.exponential(30),
            'slippage_bps': np.random.exponential(2),
            'fee_pct': 0.001
        }
    
    def _generate_mean_reversion_trade(self, trade_id: int, timestamp: pd.Timestamp, 
                                     market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mean reversion trade."""
        
        success_rate = self.config['success_rates']['mean_reversion']
        avg_return = self.config['avg_returns']['mean_reversion']
        vol_mult = self.config['volatility_multipliers']['mean_reversion']
        
        # Mean reversion trades are more successful in volatile, non-trending markets
        volatility_boost = min(0.1, market_context['volatility'] * 2)
        trend_penalty = abs(market_context['trend']) * 0.1
        adjusted_success_rate = min(0.9, success_rate + volatility_boost - trend_penalty)
        
        # Generate outcome
        is_successful = np.random.random() < adjusted_success_rate
        
        if is_successful:
            # Positive return with mean reversion characteristics
            base_return = avg_return * (1 + market_context['volatility'] * 0.3)
            volatility = market_context['volatility'] * vol_mult
            return_pct = np.random.normal(base_return, volatility)
        else:
            # Negative return
            volatility = market_context['volatility'] * vol_mult
            return_pct = np.random.normal(-avg_return * 0.6, volatility)
        
        return {
            'trade_id': trade_id,
            'timestamp': timestamp,
            'trade_type': 'mean_reversion',
            'side': 'buy' if np.random.random() > 0.5 else 'sell',
            'quantity': np.random.uniform(100, 1000),
            'entry_price': 0.52 + np.random.normal(0, 0.001),
            'exit_price': 0.52 + np.random.normal(0, 0.001),
            'return_pct': return_pct,
            'is_successful': is_successful,
            'duration_minutes': np.random.exponential(45),
            'slippage_bps': np.random.exponential(1.5),
            'fee_pct': 0.001
        }
    
    def _generate_funding_arb_trade(self, trade_id: int, timestamp: pd.Timestamp, 
                                  market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate funding arbitrage trade."""
        
        success_rate = self.config['success_rates']['funding_arb']
        avg_return = self.config['avg_returns']['funding_arb']
        vol_mult = self.config['volatility_multipliers']['funding_arb']
        
        # Funding arb trades are more successful in stable markets
        stability_boost = max(0, 0.1 - market_context['volatility'] * 2)
        adjusted_success_rate = min(0.9, success_rate + stability_boost)
        
        # Generate outcome
        is_successful = np.random.random() < adjusted_success_rate
        
        if is_successful:
            # Positive return with funding characteristics
            base_return = avg_return * (1 + np.random.uniform(0, 0.2))
            volatility = market_context['volatility'] * vol_mult
            return_pct = np.random.normal(base_return, volatility)
        else:
            # Small negative return
            volatility = market_context['volatility'] * vol_mult
            return_pct = np.random.normal(-avg_return * 0.3, volatility)
        
        return {
            'trade_id': trade_id,
            'timestamp': timestamp,
            'trade_type': 'funding_arb',
            'side': 'buy' if np.random.random() > 0.5 else 'sell',
            'quantity': np.random.uniform(100, 1000),
            'entry_price': 0.52 + np.random.normal(0, 0.001),
            'exit_price': 0.52 + np.random.normal(0, 0.001),
            'return_pct': return_pct,
            'is_successful': is_successful,
            'duration_minutes': np.random.exponential(60),
            'slippage_bps': np.random.exponential(1),
            'fee_pct': 0.001
        }
    
    def _generate_breakout_trade(self, trade_id: int, timestamp: pd.Timestamp, 
                               market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate breakout trade."""
        
        success_rate = self.config['success_rates']['breakout']
        avg_return = self.config['avg_returns']['breakout']
        vol_mult = self.config['volatility_multipliers']['breakout']
        
        # Breakout trades are more successful in high volume, trending markets
        volume_boost = min(0.1, (market_context['volume_ratio'] - 1) * 0.05)
        trend_boost = max(0, abs(market_context['trend']) * 0.1)
        adjusted_success_rate = min(0.9, success_rate + volume_boost + trend_boost)
        
        # Generate outcome
        is_successful = np.random.random() < adjusted_success_rate
        
        if is_successful:
            # High positive return with breakout characteristics
            base_return = avg_return * (1 + market_context['volume_ratio'] * 0.2)
            volatility = market_context['volatility'] * vol_mult
            return_pct = np.random.normal(base_return, volatility)
        else:
            # Large negative return (failed breakout)
            volatility = market_context['volatility'] * vol_mult
            return_pct = np.random.normal(-avg_return * 0.8, volatility)
        
        return {
            'trade_id': trade_id,
            'timestamp': timestamp,
            'trade_type': 'breakout',
            'side': 'buy' if np.random.random() > 0.5 else 'sell',
            'quantity': np.random.uniform(100, 1000),
            'entry_price': 0.52 + np.random.normal(0, 0.001),
            'exit_price': 0.52 + np.random.normal(0, 0.001),
            'return_pct': return_pct,
            'is_successful': is_successful,
            'duration_minutes': np.random.exponential(20),
            'slippage_bps': np.random.exponential(3),
            'fee_pct': 0.001
        }
    
    def _generate_generic_trade(self, trade_id: int, timestamp: pd.Timestamp, 
                              market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic trade."""
        
        return {
            'trade_id': trade_id,
            'timestamp': timestamp,
            'trade_type': 'generic',
            'side': 'buy' if np.random.random() > 0.5 else 'sell',
            'quantity': np.random.uniform(100, 1000),
            'entry_price': 0.52 + np.random.normal(0, 0.001),
            'exit_price': 0.52 + np.random.normal(0, 0.001),
            'return_pct': np.random.normal(0, market_context['volatility']),
            'is_successful': np.random.random() > 0.5,
            'duration_minutes': np.random.exponential(30),
            'slippage_bps': np.random.exponential(2),
            'fee_pct': 0.001
        }
    
    def _add_derived_metrics(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics to trades."""
        
        # PnL
        trades_df['pnl'] = trades_df['quantity'] * trades_df['entry_price'] * trades_df['return_pct']
        
        # Risk-adjusted return
        trades_df['risk_adjusted_return'] = trades_df['return_pct'] / (trades_df['slippage_bps'] / 10000 + 0.001)
        
        # Trade efficiency
        trades_df['trade_efficiency'] = trades_df['pnl'] / (trades_df['quantity'] * trades_df['entry_price'] * trades_df['fee_pct'])
        
        # Market impact
        trades_df['market_impact'] = trades_df['slippage_bps'] / 10000
        
        return trades_df

class AutoOptimizationBootstrap:
    """Auto-optimization with bootstrap data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.synthetic_generator = SyntheticTradeGenerator()
        self.optimization_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default optimization configuration."""
        return {
            'optimization_methods': ['differential_evolution', 'minimize'],
            'objective_functions': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
            'parameter_bounds': {
                'position_size': (0.01, 0.1),
                'stop_loss': (0.005, 0.05),
                'take_profit': (0.01, 0.1),
                'max_trades_per_day': (1, 20),
                'volatility_threshold': (0.01, 0.05)
            },
            'bootstrap_samples': 5,
            'synthetic_trades_per_sample': 2000,
            'validation_split': 0.2,
            'max_iterations': 100
        }
    
    def bootstrap_optimization(self, market_data: pd.DataFrame, 
                             current_strategy_params: Dict[str, float]) -> Dict[str, Any]:
        """Run bootstrap optimization with synthetic data."""
        
        logger.info("Starting bootstrap optimization")
        
        # Generate synthetic trades
        synthetic_trades = self.synthetic_generator.generate_synthetic_trades(
            market_data, 
            self.config['synthetic_trades_per_sample']
        )
        
        # Run multiple optimization samples
        optimization_results = []
        
        for sample in range(self.config['bootstrap_samples']):
            logger.info(f"Running optimization sample {sample + 1}/{self.config['bootstrap_samples']}")
            
            # Create bootstrap sample
            bootstrap_trades = synthetic_trades.sample(
                n=len(synthetic_trades), 
                replace=True
            ).reset_index(drop=True)
            
            # Run optimization
            result = self._optimize_strategy(bootstrap_trades, current_strategy_params)
            optimization_results.append(result)
        
        # Aggregate results
        aggregated_result = self._aggregate_optimization_results(optimization_results)
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'results': aggregated_result,
            'synthetic_trades_count': len(synthetic_trades)
        })
        
        logger.info("Bootstrap optimization completed")
        
        return aggregated_result
    
    def _optimize_strategy(self, trades_df: pd.DataFrame, 
                          current_params: Dict[str, float]) -> Dict[str, Any]:
        """Optimize strategy parameters on synthetic trades."""
        
        # Split data for validation
        split_idx = int(len(trades_df) * (1 - self.config['validation_split']))
        train_trades = trades_df.iloc[:split_idx]
        val_trades = trades_df.iloc[split_idx:]
        
        # Define objective function
        def objective(params):
            return -self._evaluate_strategy_performance(train_trades, params)
        
        # Run optimization
        best_result = None
        best_score = -np.inf
        
        for method in self.config['optimization_methods']:
            try:
                if method == 'differential_evolution':
                    result = differential_evolution(
                        objective,
                        bounds=list(self.config['parameter_bounds'].values()),
                        maxiter=self.config['max_iterations'],
                        seed=42
                    )
                else:  # minimize
                    x0 = list(current_params.values())
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=list(self.config['parameter_bounds'].values())
                    )
                
                if result.fun > best_score:
                    best_score = result.fun
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Optimization method {method} failed: {e}")
                continue
        
        if best_result is None:
            return {
                'success': False,
                'error': 'All optimization methods failed',
                'parameters': current_params,
                'score': 0.0
            }
        
        # Extract optimized parameters
        param_names = list(self.config['parameter_bounds'].keys())
        optimized_params = dict(zip(param_names, best_result.x))
        
        # Validate on validation set
        val_score = self._evaluate_strategy_performance(val_trades, optimized_params)
        
        return {
            'success': True,
            'parameters': optimized_params,
            'train_score': -best_score,
            'val_score': val_score,
            'optimization_method': method,
            'iterations': best_result.nit if hasattr(best_result, 'nit') else 0
        }
    
    def _evaluate_strategy_performance(self, trades_df: pd.DataFrame, 
                                     params: Dict[str, float]) -> float:
        """Evaluate strategy performance with given parameters."""
        
        # Apply strategy parameters to trades
        filtered_trades = self._apply_strategy_filters(trades_df, params)
        
        if len(filtered_trades) == 0:
            return 0.0
        
        # Calculate performance metrics
        returns = filtered_trades['return_pct']
        
        # Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Calmar ratio
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = returns.mean() / max_drawdown if max_drawdown > 0 else 0
        
        # Combined score
        combined_score = (sharpe_ratio * 0.4 + sortino_ratio * 0.3 + calmar_ratio * 0.3)
        
        return combined_score
    
    def _apply_strategy_filters(self, trades_df: pd.DataFrame, 
                              params: Dict[str, float]) -> pd.DataFrame:
        """Apply strategy filters based on parameters."""
        
        filtered_trades = trades_df.copy()
        
        # Position size filter
        max_position_size = params.get('position_size', 0.05)
        filtered_trades = filtered_trades[filtered_trades['quantity'] <= max_position_size * 10000]
        
        # Stop loss filter
        stop_loss = params.get('stop_loss', 0.02)
        filtered_trades = filtered_trades[filtered_trades['return_pct'] >= -stop_loss]
        
        # Take profit filter
        take_profit = params.get('take_profit', 0.05)
        filtered_trades = filtered_trades[filtered_trades['return_pct'] <= take_profit]
        
        # Volatility filter
        volatility_threshold = params.get('volatility_threshold', 0.03)
        filtered_trades = filtered_trades[filtered_trades['volatility'] <= volatility_threshold]
        
        # Max trades per day filter
        max_trades_per_day = params.get('max_trades_per_day', 10)
        daily_trades = filtered_trades.groupby(filtered_trades['timestamp'].dt.date).size()
        valid_days = daily_trades[daily_trades <= max_trades_per_day].index
        filtered_trades = filtered_trades[filtered_trades['timestamp'].dt.date.isin(valid_days)]
        
        return filtered_trades
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _aggregate_optimization_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple optimization samples."""
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'success': False,
                'error': 'No successful optimizations',
                'aggregated_parameters': {},
                'confidence_score': 0.0
            }
        
        # Aggregate parameters
        param_names = list(self.config['parameter_bounds'].keys())
        aggregated_params = {}
        
        for param_name in param_names:
            values = [r['parameters'][param_name] for r in successful_results]
            aggregated_params[param_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Calculate confidence score
        param_stability = []
        for param_name in param_names:
            values = [r['parameters'][param_name] for r in successful_results]
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 1
            param_stability.append(1 / (1 + cv))  # Higher stability = lower CV
        
        confidence_score = np.mean(param_stability)
        
        # Aggregate scores
        train_scores = [r['train_score'] for r in successful_results]
        val_scores = [r['val_score'] for r in successful_results]
        
        return {
            'success': True,
            'aggregated_parameters': aggregated_params,
            'confidence_score': confidence_score,
            'performance_metrics': {
                'avg_train_score': np.mean(train_scores),
                'std_train_score': np.std(train_scores),
                'avg_val_score': np.mean(val_scores),
                'std_val_score': np.std(val_scores),
                'best_train_score': np.max(train_scores),
                'best_val_score': np.max(val_scores)
            },
            'optimization_stats': {
                'successful_samples': len(successful_results),
                'total_samples': len(results),
                'success_rate': len(successful_results) / len(results)
            }
        }
    
    def get_optimization_recommendations(self, aggregated_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization recommendations."""
        
        if not aggregated_result['success']:
            return {
                'recommendation': 'No optimization recommendations available',
                'confidence': 0.0,
                'parameters': {}
            }
        
        # Extract recommended parameters
        recommended_params = {}
        for param_name, stats in aggregated_result['aggregated_parameters'].items():
            # Use median as recommendation (more robust than mean)
            recommended_params[param_name] = stats['median']
        
        # Determine confidence level
        confidence_score = aggregated_result['confidence_score']
        if confidence_score > 0.8:
            confidence_level = 'High'
        elif confidence_score > 0.6:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        # Generate recommendation
        if confidence_score > 0.7:
            recommendation = f"Optimization completed with {confidence_level} confidence. Recommended parameter updates available."
        else:
            recommendation = f"Optimization completed with {confidence_level} confidence. Consider collecting more data before implementing changes."
        
        return {
            'recommendation': recommendation,
            'confidence': confidence_score,
            'confidence_level': confidence_level,
            'parameters': recommended_params,
            'parameter_stability': aggregated_result['aggregated_parameters']
        }
    
    def save_optimization_results(self, results: Dict[str, Any], 
                                filepath: str = "reports/ml/optimization_results.json"):
        """Save optimization results."""
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")

def main():
    """Demonstrate auto-optimization bootstrap."""
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    base_price = 0.52
    returns = np.random.normal(0, 0.01, 1000)
    prices = base_price * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 1000))),
        'close': prices,
        'volume': np.random.lognormal(8, 1, 1000)
    }, index=dates)
    
    # Initialize auto-optimization
    auto_opt = AutoOptimizationBootstrap()
    
    # Current strategy parameters
    current_params = {
        'position_size': 0.05,
        'stop_loss': 0.02,
        'take_profit': 0.05,
        'max_trades_per_day': 10,
        'volatility_threshold': 0.03
    }
    
    print("🧪 Testing Auto-Optimization Bootstrap")
    print("=" * 50)
    
    # Run bootstrap optimization
    results = auto_opt.bootstrap_optimization(market_data, current_params)
    
    # Get recommendations
    recommendations = auto_opt.get_optimization_recommendations(results)
    
    print(f"Optimization Success: {results['success']}")
    print(f"Confidence Score: {results['confidence_score']:.3f}")
    print(f"Recommendation: {recommendations['recommendation']}")
    
    if results['success']:
        print(f"\nRecommended Parameters:")
        for param, value in recommendations['parameters'].items():
            print(f"  {param}: {value:.4f}")
        
        print(f"\nPerformance Metrics:")
        perf_metrics = results['performance_metrics']
        print(f"  Average Train Score: {perf_metrics['avg_train_score']:.4f}")
        print(f"  Average Val Score: {perf_metrics['avg_val_score']:.4f}")
        print(f"  Best Train Score: {perf_metrics['best_train_score']:.4f}")
        print(f"  Best Val Score: {perf_metrics['best_val_score']:.4f}")
        
        print(f"\nOptimization Stats:")
        opt_stats = results['optimization_stats']
        print(f"  Successful Samples: {opt_stats['successful_samples']}")
        print(f"  Success Rate: {opt_stats['success_rate']:.1%}")
    
    # Save results
    auto_opt.save_optimization_results(results)
    
    print("\n✅ Auto-optimization bootstrap completed")
    
    return 0

if __name__ == "__main__":
    exit(main())
