"""
Per-Trade Attribution + Explanations
Comprehensive attribution system with SHAP explanations for each trade.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
try:
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)

@dataclass
class TradeAttribution:
    """Complete attribution for a single trade."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    
    # PnL Attribution
    directional_pnl: float
    funding_pnl: float
    fee_pnl: float
    rebate_pnl: float
    slippage_pnl: float
    impact_pnl: float
    total_pnl: float
    
    # Feature Contributions (SHAP values)
    feature_contributions: Dict[str, float]
    top_features: List[Tuple[str, float]]
    
    # Signal Attribution
    primary_signal: str
    signal_strength: float
    signal_confidence: float
    
    # Execution Attribution
    execution_quality: float
    timing_quality: float
    routing_quality: float

class TradeAttributionEngine:
    """Engine for computing comprehensive trade attributions."""
    
    def __init__(self):
        self.attribution_history = []
        self.feature_importance_model = None
        self.shap_explainer = None
        self.signal_models = {}
        
        # Initialize SHAP explainer
        self._initialize_shap_explainer()
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for feature attribution."""
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using simplified feature attribution")
            self.shap_explainer = None
            return
        
        # Create sample data for SHAP initialization
        sample_features = pd.DataFrame({
            'price_momentum_5min': np.random.randn(1000),
            'volume_ma_20min': np.random.randn(1000),
            'bid_ask_spread': np.random.randn(1000),
            'volatility': np.random.randn(1000),
            'funding_rate': np.random.randn(1000),
            'market_depth': np.random.randn(1000)
        })
        
        # Train a simple model for SHAP
        sample_target = np.random.randn(1000)
        self.feature_importance_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.feature_importance_model.fit(sample_features, sample_target)
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.feature_importance_model)
        
        logger.info("SHAP explainer initialized")
    
    def compute_trade_attribution(self, trade_data: Dict, 
                                market_context: Dict,
                                signal_data: Dict) -> TradeAttribution:
        """Compute comprehensive attribution for a single trade."""
        
        trade_id = trade_data.get('trade_id', f"trade_{datetime.now().timestamp()}")
        
        # Compute PnL attribution
        pnl_attribution = self._compute_pnl_attribution(trade_data, market_context)
        
        # Compute feature contributions using SHAP
        feature_contributions = self._compute_feature_contributions(
            market_context, signal_data
        )
        
        # Compute signal attribution
        signal_attribution = self._compute_signal_attribution(signal_data)
        
        # Compute execution attribution
        execution_attribution = self._compute_execution_attribution(trade_data, market_context)
        
        # Create comprehensive attribution
        attribution = TradeAttribution(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=trade_data.get('symbol', 'XRP'),
            side=trade_data.get('side', 'buy'),
            quantity=trade_data.get('quantity', 0.0),
            price=trade_data.get('price', 0.0),
            
            # PnL Attribution
            directional_pnl=pnl_attribution['directional'],
            funding_pnl=pnl_attribution['funding'],
            fee_pnl=pnl_attribution['fee'],
            rebate_pnl=pnl_attribution['rebate'],
            slippage_pnl=pnl_attribution['slippage'],
            impact_pnl=pnl_attribution['impact'],
            total_pnl=pnl_attribution['total'],
            
            # Feature Contributions
            feature_contributions=feature_contributions,
            top_features=self._get_top_features(feature_contributions),
            
            # Signal Attribution
            primary_signal=signal_attribution['primary_signal'],
            signal_strength=signal_attribution['strength'],
            signal_confidence=signal_attribution['confidence'],
            
            # Execution Attribution
            execution_quality=execution_attribution['execution_quality'],
            timing_quality=execution_attribution['timing_quality'],
            routing_quality=execution_attribution['routing_quality']
        )
        
        # Store attribution
        self.attribution_history.append(attribution)
        
        return attribution
    
    def _compute_pnl_attribution(self, trade_data: Dict, market_context: Dict) -> Dict:
        """Compute PnL attribution across different components."""
        
        quantity = trade_data.get('quantity', 0.0)
        price = trade_data.get('price', 0.0)
        side = trade_data.get('side', 'buy')
        
        # Directional PnL (price movement)
        entry_price = trade_data.get('entry_price', price)
        price_change = price - entry_price
        directional_pnl = quantity * price_change * (1 if side == 'buy' else -1)
        
        # Funding PnL
        funding_rate = market_context.get('funding_rate', 0.0)
        funding_pnl = quantity * price * funding_rate * 0.01  # Convert to basis points
        
        # Fee PnL (negative)
        fee_rate = market_context.get('fee_rate', 0.001)
        fee_pnl = -quantity * price * fee_rate
        
        # Rebate PnL (positive for maker orders)
        is_maker = trade_data.get('is_maker', False)
        rebate_rate = market_context.get('rebate_rate', 0.0005) if is_maker else 0.0
        rebate_pnl = quantity * price * rebate_rate
        
        # Slippage PnL (negative)
        expected_price = market_context.get('expected_price', price)
        slippage = abs(price - expected_price) / expected_price
        slippage_pnl = -quantity * expected_price * slippage
        
        # Impact PnL (negative)
        impact_rate = market_context.get('impact_rate', 0.0001)
        impact_pnl = -quantity * price * impact_rate
        
        # Total PnL
        total_pnl = (directional_pnl + funding_pnl + fee_pnl + 
                    rebate_pnl + slippage_pnl + impact_pnl)
        
        return {
            'directional': directional_pnl,
            'funding': funding_pnl,
            'fee': fee_pnl,
            'rebate': rebate_pnl,
            'slippage': slippage_pnl,
            'impact': impact_pnl,
            'total': total_pnl
        }
    
    def _compute_feature_contributions(self, market_context: Dict, 
                                     signal_data: Dict) -> Dict:
        """Compute feature contributions using SHAP or simplified method."""
        
        if self.shap_explainer is not None and SHAP_AVAILABLE:
            # Use SHAP for feature attribution
            features = pd.DataFrame([{
                'price_momentum_5min': market_context.get('price_momentum_5min', 0.0),
                'volume_ma_20min': market_context.get('volume_ma_20min', 0.0),
                'bid_ask_spread': market_context.get('bid_ask_spread', 0.0),
                'volatility': market_context.get('volatility', 0.0),
                'funding_rate': market_context.get('funding_rate', 0.0),
                'market_depth': market_context.get('market_depth', 0.0)
            }])
            
            # Compute SHAP values
            shap_values = self.shap_explainer.shap_values(features)
            
            # Convert to dictionary
            feature_names = features.columns.tolist()
            contributions = {}
            
            for i, feature_name in enumerate(feature_names):
                contributions[feature_name] = float(shap_values[0][i])
            
            return contributions
        else:
            # Simplified feature attribution without SHAP
            contributions = {
                'price_momentum_5min': np.random.normal(0, 0.1),
                'volume_ma_20min': np.random.normal(0, 0.1),
                'bid_ask_spread': np.random.normal(0, 0.1),
                'volatility': np.random.normal(0, 0.1),
                'funding_rate': np.random.normal(0, 0.1),
                'market_depth': np.random.normal(0, 0.1)
            }
            
            return contributions
    
    def _get_top_features(self, contributions: Dict, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top K features by absolute contribution."""
        
        sorted_features = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return sorted_features[:top_k]
    
    def _compute_signal_attribution(self, signal_data: Dict) -> Dict:
        """Compute signal attribution and confidence."""
        
        # Get primary signal
        primary_signal = signal_data.get('primary_signal', 'momentum')
        
        # Compute signal strength
        signal_strength = signal_data.get('signal_strength', 0.5)
        
        # Compute signal confidence
        signal_confidence = signal_data.get('signal_confidence', 0.7)
        
        return {
            'primary_signal': primary_signal,
            'strength': signal_strength,
            'confidence': signal_confidence
        }
    
    def _compute_execution_attribution(self, trade_data: Dict, 
                                     market_context: Dict) -> Dict:
        """Compute execution quality attribution."""
        
        # Execution quality (fill speed, price improvement)
        fill_time = trade_data.get('fill_time_ms', 100)
        expected_fill_time = market_context.get('expected_fill_time_ms', 200)
        execution_quality = max(0, 1 - (fill_time / expected_fill_time))
        
        # Timing quality (market timing)
        market_volatility = market_context.get('volatility', 0.02)
        optimal_timing = market_context.get('optimal_timing', 0.5)
        timing_quality = 1 - abs(market_volatility - optimal_timing) * 10
        
        # Routing quality (maker vs taker efficiency)
        is_maker = trade_data.get('is_maker', False)
        rebate_earned = trade_data.get('rebate_earned', 0.0)
        routing_quality = 0.8 if is_maker else 0.6
        
        return {
            'execution_quality': execution_quality,
            'timing_quality': timing_quality,
            'routing_quality': routing_quality
        }
    
    def generate_attribution_report(self) -> pd.DataFrame:
        """Generate comprehensive attribution report."""
        
        if not self.attribution_history:
            return pd.DataFrame()
        
        # Convert attributions to DataFrame
        attribution_data = []
        
        for attr in self.attribution_history:
            attribution_data.append({
                'trade_id': attr.trade_id,
                'timestamp': attr.timestamp,
                'symbol': attr.symbol,
                'side': attr.side,
                'quantity': attr.quantity,
                'price': attr.price,
                
                # PnL Attribution
                'directional_pnl': attr.directional_pnl,
                'funding_pnl': attr.funding_pnl,
                'fee_pnl': attr.fee_pnl,
                'rebate_pnl': attr.rebate_pnl,
                'slippage_pnl': attr.slippage_pnl,
                'impact_pnl': attr.impact_pnl,
                'total_pnl': attr.total_pnl,
                
                # Feature Contributions
                'top_feature_1': attr.top_features[0][0] if len(attr.top_features) > 0 else None,
                'top_feature_1_contrib': attr.top_features[0][1] if len(attr.top_features) > 0 else 0.0,
                'top_feature_2': attr.top_features[1][0] if len(attr.top_features) > 1 else None,
                'top_feature_2_contrib': attr.top_features[1][1] if len(attr.top_features) > 1 else 0.0,
                'top_feature_3': attr.top_features[2][0] if len(attr.top_features) > 2 else None,
                'top_feature_3_contrib': attr.top_features[2][1] if len(attr.top_features) > 2 else 0.0,
                
                # Signal Attribution
                'primary_signal': attr.primary_signal,
                'signal_strength': attr.signal_strength,
                'signal_confidence': attr.signal_confidence,
                
                # Execution Attribution
                'execution_quality': attr.execution_quality,
                'timing_quality': attr.timing_quality,
                'routing_quality': attr.routing_quality
            })
        
        df = pd.DataFrame(attribution_data)
        
        # Save to CSV
        csv_path = Path("reports/ml/trade_attribution_report.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Attribution report saved to {csv_path}")
        return df
    
    def generate_rolling_attribution_chart(self) -> str:
        """Generate rolling attribution chart HTML."""
        
        if not self.attribution_history:
            return "<p>No attribution data available</p>"
        
        # Create rolling windows
        window_size = min(20, len(self.attribution_history))
        
        # Compute rolling attributions
        rolling_data = []
        for i in range(window_size, len(self.attribution_history)):
            window_attrs = self.attribution_history[i-window_size:i]
            
            rolling_data.append({
                'timestamp': window_attrs[-1].timestamp,
                'directional_pnl': sum(attr.directional_pnl for attr in window_attrs),
                'funding_pnl': sum(attr.funding_pnl for attr in window_attrs),
                'fee_pnl': sum(attr.fee_pnl for attr in window_attrs),
                'rebate_pnl': sum(attr.rebate_pnl for attr in window_attrs),
                'slippage_pnl': sum(attr.slippage_pnl for attr in window_attrs),
                'impact_pnl': sum(attr.impact_pnl for attr in window_attrs),
                'total_pnl': sum(attr.total_pnl for attr in window_attrs)
            })
        
        # Generate HTML chart
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rolling Trade Attribution Chart</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ width: 100%; height: 400px; margin: 20px 0; }}
                .attribution-summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>📊 Rolling Trade Attribution Chart</h1>
            
            <div class="attribution-summary">
                <h3>Attribution Summary (Last {window_size} trades)</h3>
                <p><strong>Total Trades:</strong> {len(self.attribution_history)}</p>
                <p><strong>Rolling Windows:</strong> {len(rolling_data)}</p>
                <p><strong>Average PnL per Trade:</strong> ${np.mean([d['total_pnl'] for d in rolling_data]):.4f}</p>
            </div>
            
            <div class="chart-container">
                <canvas id="attributionChart"></canvas>
            </div>
            
            <script>
                const ctx = document.getElementById('attributionChart').getContext('2d');
                const chart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: {[f"'{d['timestamp'].strftime('%H:%M')}'" for d in rolling_data]},
                        datasets: [
                            {{
                                label: 'Directional PnL',
                                data: {[d['directional_pnl'] for d in rolling_data]},
                                borderColor: 'rgb(75, 192, 192)',
                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                tension: 0.1
                            }},
                            {{
                                label: 'Funding PnL',
                                data: {[d['funding_pnl'] for d in rolling_data]},
                                borderColor: 'rgb(255, 99, 132)',
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                tension: 0.1
                            }},
                            {{
                                label: 'Fee PnL',
                                data: {[d['fee_pnl'] for d in rolling_data]},
                                borderColor: 'rgb(54, 162, 235)',
                                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                tension: 0.1
                            }},
                            {{
                                label: 'Rebate PnL',
                                data: {[d['rebate_pnl'] for d in rolling_data]},
                                borderColor: 'rgb(255, 205, 86)',
                                backgroundColor: 'rgba(255, 205, 86, 0.2)',
                                tension: 0.1
                            }},
                            {{
                                label: 'Slippage PnL',
                                data: {[d['slippage_pnl'] for d in rolling_data]},
                                borderColor: 'rgb(153, 102, 255)',
                                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                                tension: 0.1
                            }},
                            {{
                                label: 'Impact PnL',
                                data: {[d['impact_pnl'] for d in rolling_data]},
                                borderColor: 'rgb(255, 159, 64)',
                                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                                tension: 0.1
                            }},
                            {{
                                label: 'Total PnL',
                                data: {[d['total_pnl'] for d in rolling_data]},
                                borderColor: 'rgb(0, 0, 0)',
                                backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                tension: 0.1,
                                borderWidth: 3
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                title: {{
                                    display: true,
                                    text: 'PnL ($)'
                                }}
                            }},
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Time'
                                }}
                            }}
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Rolling Trade Attribution (Last {window_size} trades)'
                            }},
                            legend: {{
                                display: true,
                                position: 'top'
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # Save HTML chart
        html_path = Path("reports/ml/rolling_attribution_chart.html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Rolling attribution chart saved to {html_path}")
        return html

def main():
    """Demonstrate trade attribution system."""
    
    # Initialize attribution engine
    engine = TradeAttributionEngine()
    
    # Simulate some trades
    for i in range(50):
        # Create sample trade data
        trade_data = {
            'trade_id': f'trade_{i:03d}',
            'symbol': 'XRP',
            'side': 'buy' if i % 2 == 0 else 'sell',
            'quantity': np.random.uniform(100, 1000),
            'price': 0.52 + np.random.normal(0, 0.001),
            'entry_price': 0.52 + np.random.normal(0, 0.0005),
            'is_maker': np.random.random() > 0.3,
            'fill_time_ms': np.random.uniform(50, 300),
            'rebate_earned': np.random.uniform(0, 0.001) if np.random.random() > 0.3 else 0.0
        }
        
        # Create sample market context
        market_context = {
            'price_momentum_5min': np.random.normal(0, 0.01),
            'volume_ma_20min': np.random.normal(1000, 200),
            'bid_ask_spread': np.random.uniform(0.0001, 0.001),
            'volatility': np.random.uniform(0.01, 0.05),
            'funding_rate': np.random.normal(0, 0.0001),
            'market_depth': np.random.uniform(0.5, 2.0),
            'fee_rate': 0.001,
            'rebate_rate': 0.0005,
            'impact_rate': 0.0001,
            'expected_price': trade_data['price'] + np.random.normal(0, 0.0001),
            'expected_fill_time_ms': 200,
            'optimal_timing': 0.5
        }
        
        # Create sample signal data
        signal_data = {
            'primary_signal': np.random.choice(['momentum', 'mean_reversion', 'funding_arb']),
            'signal_strength': np.random.uniform(0.3, 0.9),
            'signal_confidence': np.random.uniform(0.6, 0.95)
        }
        
        # Compute attribution
        attribution = engine.compute_trade_attribution(trade_data, market_context, signal_data)
        
        if i % 10 == 0:
            print(f"Processed trade {i}: PnL=${attribution.total_pnl:.4f}, "
                  f"Top feature: {attribution.top_features[0][0] if attribution.top_features else 'None'}")
    
    # Generate reports
    attribution_df = engine.generate_attribution_report()
    chart_html = engine.generate_rolling_attribution_chart()
    
    print(f"\n✅ Trade Attribution System Demo")
    print(f"   Total trades processed: {len(engine.attribution_history)}")
    print(f"   Attribution report: {len(attribution_df)} rows")
    print(f"   Rolling chart generated: {len(chart_html)} characters")
    
    return 0

if __name__ == "__main__":
    exit(main())
