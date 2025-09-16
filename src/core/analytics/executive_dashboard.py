"""
Executive Dashboard Generator
Combines equity curves, latency histograms, risk events, and attribution in one comprehensive artifact.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import base64
from io import BytesIO
import webbrowser


class ExecutiveDashboard:
    """Generates comprehensive executive summary combining all key metrics."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_data(self):
        """Load all report data for dashboard generation."""
        data = {}
        
        # Load trade ledger
        try:
            ledger_path = self.reports_dir / "ledgers" / "trades.parquet"
            if ledger_path.exists():
                data['trades'] = pd.read_parquet(ledger_path)
            else:
                data['trades'] = pd.read_csv(self.reports_dir / "ledgers" / "trades.csv")
        except Exception as e:
            print(f"Warning: Could not load trade ledger: {e}")
            data['trades'] = pd.DataFrame()
            
        # Load risk events
        try:
            risk_path = self.reports_dir / "risk_events" / "risk_events.json"
            if risk_path.exists():
                with open(risk_path, 'r') as f:
                    data['risk_events'] = json.load(f)
            else:
                data['risk_events'] = []
        except Exception as e:
            print(f"Warning: Could not load risk events: {e}")
            data['risk_events'] = []
            
        # Load latency analysis
        try:
            latency_path = self.reports_dir / "latency" / "latency_analysis.json"
            if latency_path.exists():
                with open(latency_path, 'r') as f:
                    data['latency'] = json.load(f)
            else:
                data['latency'] = {}
        except Exception as e:
            print(f"Warning: Could not load latency data: {e}")
            data['latency'] = {}
            
        # Load regime analysis
        try:
            regime_path = self.reports_dir / "regime" / "regime_analysis.json"
            if regime_path.exists():
                with open(regime_path, 'r') as f:
                    data['regime'] = json.load(f)
            else:
                data['regime'] = {}
        except Exception as e:
            print(f"Warning: Could not load regime data: {e}")
            data['regime'] = {}
            
        # Load maker/taker analysis
        try:
            routing_path = self.reports_dir / "maker_taker" / "routing_analysis.json"
            if routing_path.exists():
                with open(routing_path, 'r') as f:
                    data['routing'] = json.load(f)
            else:
                data['routing'] = {}
        except Exception as e:
            print(f"Warning: Could not load routing data: {e}")
            data['routing'] = {}
            
        return data
    
    def generate_equity_curve(self, trades_df: pd.DataFrame):
        """Generate equity curve visualization."""
        if trades_df.empty:
            return go.Figure()
            
        # Calculate cumulative P&L
        trades_df['cumulative_pnl'] = trades_df['pnl_realized'].cumsum()
        trades_df['portfolio_value'] = 10000 + trades_df['cumulative_pnl']  # Starting with $10K
        
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=trades_df.index,
            y=trades_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2E8B57', width=3)
        ))
        
        # Add drawdown overlay
        peak = trades_df['portfolio_value'].expanding().max()
        drawdown = (trades_df['portfolio_value'] - peak) / peak * 100
        
        fig.add_trace(go.Scatter(
            x=trades_df.index,
            y=drawdown,
            mode='lines',
            name='Drawdown %',
            yaxis='y2',
            line=dict(color='#DC143C', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve & Drawdown',
            xaxis_title='Trade Number',
            yaxis_title='Portfolio Value ($)',
            yaxis2=dict(title='Drawdown (%)', overlaying='y', side='right'),
            height=400,
            showlegend=True
        )
        
        return fig
    
    def generate_latency_histogram(self, latency_data: dict):
        """Generate latency distribution histogram."""
        if not latency_data or 'latency_samples' not in latency_data:
            return go.Figure()
            
        samples = latency_data['latency_samples']
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=samples,
            nbinsx=50,
            name='Latency Distribution',
            marker_color='#4169E1',
            opacity=0.7
        ))
        
        # Add percentile lines
        p50 = latency_data.get('p50_latency_ms', 0)
        p95 = latency_data.get('p95_latency_ms', 0)
        p99 = latency_data.get('p99_latency_ms', 0)
        
        for percentile, value, color in [(50, p50, '#FFD700'), (95, p95, '#FF8C00'), (99, p99, '#FF0000')]:
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color=color,
                annotation_text=f"P{percentile}: {value:.1f}ms",
                annotation_position="top"
            )
        
        fig.update_layout(
            title='Execution Latency Distribution',
            xaxis_title='Latency (ms)',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig
    
    def generate_risk_events_timeline(self, risk_events: list):
        """Generate risk events timeline."""
        if not risk_events:
            return go.Figure()
            
        # Convert to DataFrame for easier manipulation
        events_df = pd.DataFrame(risk_events)
        
        fig = go.Figure()
        
        # Color code by severity
        severity_colors = {
            'WARNING': '#FFD700',
            'ALERT': '#FF8C00', 
            'CRITICAL': '#FF0000',
            'KILL_SWITCH': '#8B0000'
        }
        
        for severity, color in severity_colors.items():
            severity_events = events_df[events_df['severity'] == severity]
            if not severity_events.empty:
                fig.add_trace(go.Scatter(
                    x=severity_events['timestamp'],
                    y=[severity] * len(severity_events),
                    mode='markers',
                    name=severity,
                    marker=dict(color=color, size=10),
                    text=severity_events['message'],
                    hovertemplate='%{text}<br>Time: %{x}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Risk Events Timeline',
            xaxis_title='Time',
            yaxis_title='Severity',
            height=300,
            showlegend=True
        )
        
        return fig
    
    def generate_attribution_analysis(self, trades_df: pd.DataFrame):
        """Generate strategy attribution analysis."""
        if trades_df.empty:
            return go.Figure()
            
        # Group by strategy and calculate metrics
        attribution = trades_df.groupby('strategy_name').agg({
            'pnl_realized': ['sum', 'count', 'mean'],
            'fee': 'sum',
            'funding': 'sum'
        }).round(2)
        
        attribution.columns = ['Total_PnL', 'Trade_Count', 'Avg_PnL', 'Total_Fees', 'Total_Funding']
        attribution['Net_PnL'] = attribution['Total_PnL'] - attribution['Total_Fees'] + attribution['Total_Funding']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=attribution.index,
            y=attribution['Net_PnL'],
            name='Net P&L',
            marker_color='#2E8B57'
        ))
        
        fig.add_trace(go.Bar(
            x=attribution.index,
            y=attribution['Total_Fees'],
            name='Fees Paid',
            marker_color='#DC143C'
        ))
        
        fig.add_trace(go.Bar(
            x=attribution.index,
            y=attribution['Total_Funding'],
            name='Funding Received',
            marker_color='#4169E1'
        ))
        
        fig.update_layout(
            title='Strategy Attribution Analysis',
            xaxis_title='Strategy',
            yaxis_title='P&L ($)',
            barmode='group',
            height=400
        )
        
        return fig
    
    def generate_performance_metrics(self, trades_df: pd.DataFrame, latency_data: dict, routing_data: dict):
        """Generate key performance metrics summary."""
        if trades_df.empty:
            return {}
            
        # Calculate key metrics
        total_return = trades_df['pnl_realized'].sum()
        total_trades = len(trades_df)
        win_rate = (trades_df['pnl_realized'] > 0).mean() * 100
        avg_trade = trades_df['pnl_realized'].mean()
        
        # Risk metrics
        returns = trades_df['pnl_realized']
        sharpe_ratio = returns.mean() / returns.std() * (252**0.5) if returns.std() > 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        # Latency metrics
        p95_latency = latency_data.get('p95_latency_ms', 0)
        p99_latency = latency_data.get('p99_latency_ms', 0)
        
        # Routing metrics
        maker_ratio = routing_data.get('maker_ratio', 0) * 100
        avg_slippage = routing_data.get('avg_slippage_bps', 0)
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'maker_ratio': maker_ratio,
            'avg_slippage': avg_slippage
        }
    
    def generate_dashboard(self, output_format: str = 'html'):
        """Generate comprehensive executive dashboard."""
        print("📊 Generating Executive Dashboard...")
        
        # Load all data
        data = self.load_data()
        
        # Generate visualizations
        equity_fig = self.generate_equity_curve(data['trades'])
        latency_fig = self.generate_latency_histogram(data['latency'])
        risk_fig = self.generate_risk_events_timeline(data['risk_events'])
        attribution_fig = self.generate_attribution_analysis(data['trades'])
        
        # Calculate performance metrics
        metrics = self.generate_performance_metrics(
            data['trades'], 
            data['latency'], 
            data['routing']
        )
        
        if output_format == 'html':
            return self._generate_html_dashboard(
                equity_fig, latency_fig, risk_fig, attribution_fig, metrics
            )
        else:
            return self._generate_pdf_dashboard(
                equity_fig, latency_fig, risk_fig, attribution_fig, metrics
            )
    
    def _generate_html_dashboard(self, equity_fig, latency_fig, risk_fig, attribution_fig, metrics):
        """Generate HTML dashboard."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hat Manifesto Executive Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                               gap: 15px; margin-bottom: 20px; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 8px; 
                              box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2E8B57; }}
                .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
                .chart-container {{ background: white; padding: 20px; border-radius: 8px; 
                                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .negative {{ color: #DC143C; }}
                .positive {{ color: #2E8B57; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎩 Hat Manifesto Executive Dashboard</h1>
                <p>Comprehensive Performance Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {'positive' if metrics.get('total_return', 0) >= 0 else 'negative'}">
                        ${metrics.get('total_return', 0):,.2f}
                    </div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('total_trades', 0):,}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('win_rate', 0):.1f}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value negative">{metrics.get('max_drawdown', 0):.2f}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('p95_latency', 0):.1f}ms</div>
                    <div class="metric-label">P95 Latency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('maker_ratio', 0):.1f}%</div>
                    <div class="metric-label">Maker Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('avg_slippage', 0):.1f}bps</div>
                    <div class="metric-label">Avg Slippage</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Portfolio Equity Curve & Drawdown</h3>
                <div id="equity-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3>Execution Latency Distribution</h3>
                <div id="latency-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3>Risk Events Timeline</h3>
                <div id="risk-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3>Strategy Attribution Analysis</h3>
                <div id="attribution-chart"></div>
            </div>
            
            <script>
                Plotly.newPlot('equity-chart', {equity_fig.to_json()});
                Plotly.newPlot('latency-chart', {latency_fig.to_json()});
                Plotly.newPlot('risk-chart', {risk_fig.to_json()});
                Plotly.newPlot('attribution-chart', {attribution_fig.to_json()});
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        output_path = self.reports_dir / "executive_dashboard.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ Executive Dashboard saved to: {output_path}")
        return output_path
    
    def _generate_pdf_dashboard(self, equity_fig, latency_fig, risk_fig, attribution_fig, metrics):
        """Generate PDF dashboard (placeholder for future implementation)."""
        # This would require additional PDF generation libraries
        print("📄 PDF generation not yet implemented - using HTML format")
        return self._generate_html_dashboard(equity_fig, latency_fig, risk_fig, attribution_fig, metrics)


def main():
    """Generate executive dashboard."""
    dashboard = ExecutiveDashboard()
    output_path = dashboard.generate_dashboard('html')
    
    # Open in browser
    try:
        webbrowser.open(f"file://{output_path.absolute()}")
        print("🌐 Dashboard opened in browser")
    except Exception as e:
        print(f"Could not open browser: {e}")
        print(f"Manual access: {output_path.absolute()}")


if __name__ == "__main__":
    main()
