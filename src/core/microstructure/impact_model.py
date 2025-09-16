"""
Market Impact Model and Microstructure Analysis
Implements impact model calibration, adverse selection metrics, and queue position awareness.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


class MarketImpactModel:
    """Market impact model for execution cost estimation and optimization."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.microstructure_dir = self.reports_dir / "microstructure"
        self.microstructure_dir.mkdir(parents=True, exist_ok=True)
    
    def calibrate_impact_model(self, 
                             trade_data: pd.DataFrame,
                             participation_rates: List[float] = None) -> Dict[str, Any]:
        """Calibrate empirical impact model from trade data."""
        
        if participation_rates is None:
            # Calculate participation rates from trade data
            participation_rates = self._calculate_participation_rates(trade_data)
        
        # Calculate realized impact for each trade
        impacts = self._calculate_realized_impact(trade_data)
        
        # Fit impact model
        model_params = self._fit_impact_curve(participation_rates, impacts)
        
        # Calculate model residuals
        predicted_impacts = self._predict_impact(participation_rates, model_params)
        residuals = impacts - predicted_impacts
        
        # Model validation
        r2 = r2_score(impacts, predicted_impacts)
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        
        impact_model = {
            "model_type": "power_law",
            "parameters": model_params,
            "calibration_data": {
                "n_trades": len(trade_data),
                "participation_rate_range": [min(participation_rates), max(participation_rates)],
                "impact_range": [min(impacts), max(impacts)]
            },
            "model_quality": {
                "r2_score": r2,
                "mae": mae,
                "rmse": rmse,
                "residual_std": np.std(residuals)
            },
            "residual_analysis": {
                "residuals": residuals.tolist(),
                "residual_histogram": np.histogram(residuals, bins=20)[0].tolist(),
                "residual_bins": np.histogram(residuals, bins=20)[1].tolist()
            }
        }
        
        return impact_model
    
    def _calculate_participation_rates(self, trade_data: pd.DataFrame) -> List[float]:
        """Calculate participation rates from trade data."""
        participation_rates = []
        
        for _, trade in trade_data.iterrows():
            # Participation rate = trade_size / market_volume
            if 'volume' in trade_data.columns and trade['volume'] > 0:
                participation_rate = trade['qty'] / trade['volume']
            else:
                # Estimate based on typical market depth
                participation_rate = trade['qty'] / 10000  # Assume 10k XRP typical depth
            
            participation_rates.append(min(participation_rate, 1.0))  # Cap at 100%
        
        return participation_rates
    
    def _calculate_realized_impact(self, trade_data: pd.DataFrame) -> List[float]:
        """Calculate realized market impact for each trade."""
        impacts = []
        
        for _, trade in trade_data.iterrows():
            if 'expected_price' in trade_data.columns and 'fill_price' in trade_data.columns:
                # Impact = (fill_price - expected_price) / expected_price
                impact = (trade['fill_price'] - trade['expected_price']) / trade['expected_price']
            elif 'mid_price' in trade_data.columns and 'fill_price' in trade_data.columns:
                # Use mid-price as proxy for expected price
                impact = (trade['fill_price'] - trade['mid_price']) / trade['mid_price']
            else:
                # Estimate impact from slippage
                impact = trade.get('slippage_bps', 0) / 10000  # Convert bps to decimal
            
            impacts.append(impact)
        
        return impacts
    
    def _fit_impact_curve(self, participation_rates: List[float], impacts: List[float]) -> Dict[str, float]:
        """Fit power law impact model: Impact = A * (Participation)^B."""
        
        # Convert to numpy arrays
        x = np.array(participation_rates)
        y = np.array(impacts)
        
        # Remove zeros and negative values for log fitting
        valid_mask = (x > 0) & (y > 0)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 2:
            return {"A": 0.0, "B": 1.0, "fit_success": False}
        
        try:
            # Log-linear fit: log(y) = log(A) + B * log(x)
            log_x = np.log(x_valid)
            log_y = np.log(y_valid)
            
            # Linear regression
            model = LinearRegression()
            model.fit(log_x.reshape(-1, 1), log_y)
            
            # Extract parameters
            B = model.coef_[0]
            log_A = model.intercept_
            A = np.exp(log_A)
            
            return {
                "A": float(A),
                "B": float(B),
                "fit_success": True,
                "r2_score": float(model.score(log_x.reshape(-1, 1), log_y))
            }
            
        except Exception as e:
            return {"A": 0.0, "B": 1.0, "fit_success": False, "error": str(e)}
    
    def _predict_impact(self, participation_rates: List[float], model_params: Dict[str, float]) -> List[float]:
        """Predict market impact using calibrated model."""
        A = model_params.get("A", 0.0)
        B = model_params.get("B", 1.0)
        
        predicted_impacts = []
        for pr in participation_rates:
            if pr > 0:
                impact = A * (pr ** B)
            else:
                impact = 0.0
            predicted_impacts.append(impact)
        
        return predicted_impacts
    
    def calculate_adverse_selection(self, 
                                  trade_data: pd.DataFrame,
                                  lookforward_seconds: int = 30) -> Dict[str, Any]:
        """Calculate post-trade alpha (adverse selection metric)."""
        
        adverse_selection_results = []
        
        for i, trade in trade_data.iterrows():
            if i >= len(trade_data) - 1:
                continue
            
            # Get trade price and timestamp
            trade_price = trade['price']
            trade_time = trade.get('ts', i)
            
            # Find future price after lookforward period
            future_trades = trade_data[trade_data['ts'] > trade_time + lookforward_seconds]
            
            if len(future_trades) > 0:
                future_price = future_trades.iloc[0]['price']
                
                # Calculate post-trade alpha
                if trade['side'] == 'buy':
                    post_trade_alpha = (future_price - trade_price) / trade_price
                else:  # sell
                    post_trade_alpha = (trade_price - future_price) / trade_price
                
                adverse_selection_results.append({
                    "trade_index": i,
                    "trade_price": trade_price,
                    "future_price": future_price,
                    "post_trade_alpha": post_trade_alpha,
                    "lookforward_seconds": lookforward_seconds
                })
        
        if not adverse_selection_results:
            return {"error": "Insufficient data for adverse selection calculation"}
        
        # Calculate statistics
        post_trade_alphas = [r['post_trade_alpha'] for r in adverse_selection_results]
        
        adverse_selection_analysis = {
            "lookforward_seconds": lookforward_seconds,
            "n_trades_analyzed": len(adverse_selection_results),
            "post_trade_alpha_stats": {
                "mean": float(np.mean(post_trade_alphas)),
                "std": float(np.std(post_trade_alphas)),
                "median": float(np.median(post_trade_alphas)),
                "min": float(np.min(post_trade_alphas)),
                "max": float(np.max(post_trade_alphas))
            },
            "adverse_selection_detected": np.mean(post_trade_alphas) < -0.001,  # -0.1% threshold
            "adverse_selection_severity": "high" if np.mean(post_trade_alphas) < -0.005 else "medium" if np.mean(post_trade_alphas) < -0.001 else "low"
        }
        
        return adverse_selection_analysis
    
    def estimate_queue_position(self, 
                              order_data: pd.DataFrame,
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate queue position and priority loss."""
        
        queue_analysis = []
        
        for _, order in order_data.iterrows():
            if order['order_type'] != 'limit':
                continue
            
            # Get market data at order time
            order_time = order['ts']
            market_snapshot = market_data[market_data['ts'] <= order_time].iloc[-1] if len(market_data[market_data['ts'] <= order_time]) > 0 else None
            
            if market_snapshot is None:
                continue
            
            # Estimate queue position based on price level
            order_price = order['price']
            best_bid = market_snapshot.get('best_bid', 0)
            best_ask = market_snapshot.get('best_ask', 0)
            
            if order['side'] == 'buy':
                if order_price >= best_ask:
                    queue_position = "market_order"  # Will execute immediately
                    priority_loss = 0.0
                elif order_price == best_bid:
                    queue_position = "at_best"
                    priority_loss = 0.5  # 50% chance of execution
                else:
                    queue_position = "behind_best"
                    priority_loss = 0.1  # Low priority
            else:  # sell
                if order_price <= best_bid:
                    queue_position = "market_order"
                    priority_loss = 0.0
                elif order_price == best_ask:
                    queue_position = "at_best"
                    priority_loss = 0.5
                else:
                    queue_position = "behind_best"
                    priority_loss = 0.1
            
            queue_analysis.append({
                "order_id": order.get('order_id', 'unknown'),
                "side": order['side'],
                "price": order_price,
                "queue_position": queue_position,
                "priority_loss": priority_loss,
                "best_bid": best_bid,
                "best_ask": best_ask
            })
        
        # Calculate statistics
        if queue_analysis:
            priority_losses = [q['priority_loss'] for q in queue_analysis]
            queue_positions = [q['queue_position'] for q in queue_analysis]
            
            queue_stats = {
                "n_orders_analyzed": len(queue_analysis),
                "average_priority_loss": float(np.mean(priority_losses)),
                "queue_position_distribution": {
                    pos: queue_positions.count(pos) for pos in set(queue_positions)
                },
                "priority_loss_stats": {
                    "mean": float(np.mean(priority_losses)),
                    "std": float(np.std(priority_losses)),
                    "min": float(np.min(priority_losses)),
                    "max": float(np.max(priority_losses))
                }
            }
        else:
            queue_stats = {"error": "No limit orders found for queue analysis"}
        
        return queue_stats
    
    def calculate_maker_taker_opportunity_cost(self, 
                                             maker_trades: pd.DataFrame,
                                             taker_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate opportunity cost of maker vs taker routing."""
        
        # Calculate rebate gains from maker trades
        maker_rebate_rate = 0.00005  # 0.5 bps rebate
        maker_rebates = []
        
        for _, trade in maker_trades.iterrows():
            rebate = trade['qty'] * trade['price'] * maker_rebate_rate
            maker_rebates.append(rebate)
        
        total_maker_rebates = sum(maker_rebates)
        
        # Calculate opportunity cost of missed fills
        # This is a simplified calculation - in practice, you'd need more sophisticated modeling
        missed_fill_opportunity_cost = 0.0  # Placeholder
        
        # Calculate taker costs
        taker_fee_rate = 0.0005  # 5 bps taker fee
        taker_costs = []
        
        for _, trade in taker_trades.iterrows():
            cost = trade['qty'] * trade['price'] * taker_fee_rate
            taker_costs.append(cost)
        
        total_taker_costs = sum(taker_costs)
        
        # Calculate net benefit
        net_benefit = total_maker_rebates - total_taker_costs - missed_fill_opportunity_cost
        
        opportunity_cost_analysis = {
            "maker_trades": {
                "count": len(maker_trades),
                "total_rebates": total_maker_rebates,
                "average_rebate": total_maker_rebates / len(maker_trades) if len(maker_trades) > 0 else 0
            },
            "taker_trades": {
                "count": len(taker_trades),
                "total_costs": total_taker_costs,
                "average_cost": total_taker_costs / len(taker_trades) if len(taker_trades) > 0 else 0
            },
            "opportunity_cost": {
                "missed_fill_cost": missed_fill_opportunity_cost,
                "net_benefit": net_benefit,
                "maker_ratio": len(maker_trades) / (len(maker_trades) + len(taker_trades)) if (len(maker_trades) + len(taker_trades)) > 0 else 0
            }
        }
        
        # Save to file
        opportunity_file = self.microstructure_dir / "opportunity_cost.json"
        with open(opportunity_file, 'w') as f:
            json.dump(opportunity_cost_analysis, f, indent=2)
        
        return opportunity_cost_analysis
    
    def spread_regime_analysis(self, 
                             market_data: pd.DataFrame,
                             spread_threshold_ticks: float = 2.0,
                             depth_threshold_notional: float = 1000.0) -> Dict[str, Any]:
        """Analyze spread regimes and switching policies."""
        
        spread_analysis = []
        
        for _, snapshot in market_data.iterrows():
            best_bid = snapshot.get('best_bid', 0)
            best_ask = snapshot.get('best_ask', 0)
            bid_size = snapshot.get('bid_size', 0)
            ask_size = snapshot.get('ask_size', 0)
            
            if best_bid > 0 and best_ask > 0:
                spread = best_ask - best_bid
                spread_bps = (spread / best_bid) * 10000
                
                # Calculate depth
                bid_depth_notional = bid_size * best_bid
                ask_depth_notional = ask_size * best_ask
                total_depth = bid_depth_notional + ask_depth_notional
                
                # Determine regime
                if spread_bps <= spread_threshold_ticks and total_depth >= depth_threshold_notional:
                    regime = "tight_liquid"
                    recommended_action = "aggressive_maker"
                elif spread_bps <= spread_threshold_ticks and total_depth < depth_threshold_notional:
                    regime = "tight_illiquid"
                    recommended_action = "cautious_maker"
                elif spread_bps > spread_threshold_ticks and total_depth >= depth_threshold_notional:
                    regime = "wide_liquid"
                    recommended_action = "patient_maker"
                else:
                    regime = "wide_illiquid"
                    recommended_action = "avoid_or_taker"
                
                spread_analysis.append({
                    "timestamp": snapshot.get('ts', 0),
                    "spread_bps": spread_bps,
                    "total_depth_notional": total_depth,
                    "regime": regime,
                    "recommended_action": recommended_action
                })
        
        if spread_analysis:
            # Calculate regime statistics
            regimes = [s['regime'] for s in spread_analysis]
            regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
            
            spread_stats = {
                "n_snapshots": len(spread_analysis),
                "regime_distribution": regime_counts,
                "regime_percentages": {regime: count/len(spread_analysis)*100 for regime, count in regime_counts.items()},
                "thresholds": {
                    "spread_threshold_ticks": spread_threshold_ticks,
                    "depth_threshold_notional": depth_threshold_notional
                },
                "policy_recommendations": {
                    "tight_liquid": "Use aggressive maker routing for maximum rebates",
                    "tight_illiquid": "Use cautious maker routing with quick fallback to taker",
                    "wide_liquid": "Use patient maker routing, wait for better prices",
                    "wide_illiquid": "Avoid maker routing, use taker or wait for better conditions"
                }
            }
        else:
            spread_stats = {"error": "No market data available for spread analysis"}
        
        return spread_stats


def main():
    """Test market impact model functionality."""
    impact_model = MarketImpactModel()
    
    # Generate sample trade data
    np.random.seed(42)
    n_trades = 1000
    
    trade_data = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=n_trades, freq='1min'),
        'qty': np.random.uniform(10, 1000, n_trades),
        'price': np.random.uniform(0.4, 0.6, n_trades),
        'side': np.random.choice(['buy', 'sell'], n_trades),
        'volume': np.random.uniform(5000, 50000, n_trades),
        'expected_price': np.random.uniform(0.4, 0.6, n_trades),
        'fill_price': np.random.uniform(0.4, 0.6, n_trades),
        'slippage_bps': np.random.uniform(-5, 5, n_trades)
    })
    
    # Test impact model calibration
    impact_model_result = impact_model.calibrate_impact_model(trade_data)
    print(f"✅ Impact model calibrated: R² = {impact_model_result['model_quality']['r2_score']:.3f}")
    
    # Test adverse selection
    adverse_selection = impact_model.calculate_adverse_selection(trade_data)
    print(f"✅ Adverse selection analysis: {adverse_selection.get('n_trades_analyzed', 0)} trades analyzed")
    
    # Test opportunity cost
    maker_trades = trade_data[trade_data['side'] == 'buy'].head(100)
    taker_trades = trade_data[trade_data['side'] == 'sell'].head(100)
    
    opportunity_cost = impact_model.calculate_maker_taker_opportunity_cost(maker_trades, taker_trades)
    print(f"✅ Opportunity cost analysis: Net benefit = ${opportunity_cost['opportunity_cost']['net_benefit']:.2f}")
    
    print("✅ Market impact model testing completed")


if __name__ == "__main__":
    main()
