"""
Microstructure Dashboard Enhancement
Adds impact residuals and maker/taker opportunity cost tiles to the executive dashboard.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MicrostructureDashboardEnhancer:
    """Enhances the executive dashboard with microstructure metrics."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.impact_file = self.reports_dir / "microstructure" / "impact_residuals.json"
        self.opportunity_file = self.reports_dir / "maker_taker" / "opportunity_cost.json"
    
    def load_microstructure_data(self):
        """Load microstructure analysis data."""
        data = {}
        
        # Load impact residuals
        if self.impact_file.exists():
            try:
                with open(self.impact_file, 'r') as f:
                    data['impact'] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading impact residuals: {e}")
                data['impact'] = None
        else:
            data['impact'] = None
        
        # Load opportunity cost
        if self.opportunity_file.exists():
            try:
                with open(self.opportunity_file, 'r') as f:
                    data['opportunity'] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading opportunity cost: {e}")
                data['opportunity'] = None
        else:
            data['opportunity'] = None
        
        return data
    
    def generate_microstructure_tiles(self, data):
        """Generate HTML tiles for microstructure metrics."""
        
        tiles_html = ""
        
        # Impact Residuals Tile
        if data['impact']:
            impact = data['impact']
            r_squared = impact['impact_model']['r_squared']
            avg_impact = impact['execution_quality_metrics']['avg_implementation_shortfall']
            residual_std = impact['residuals_analysis']['std_residual']
            
            tiles_html += f"""
            <div class="metric-card">
                <div class="metric-value positive">
                    {r_squared:.3f}
                </div>
                <div class="metric-label">Impact Model R²</div>
                <div class="data-source">Source: impact_residuals.json</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value {'positive' if avg_impact < 1.0 else 'negative'}">
                    {avg_impact:.2f} bps
                </div>
                <div class="metric-label">Avg Implementation Shortfall</div>
                <div class="data-source">Source: impact_residuals.json</div>
            </div>
            """
        
        # Maker/Taker Opportunity Cost Tile
        if data['opportunity']:
            opp = data['opportunity']
            maker_ratio = opp['maker_taker_summary']['maker_ratio']
            net_cost = opp['net_opportunity_cost']['net_opportunity_cost']
            rebates = opp['rebate_analysis']['total_rebates_earned']
            
            tiles_html += f"""
            <div class="metric-card">
                <div class="metric-value positive">
                    {maker_ratio:.1%}
                </div>
                <div class="metric-label">Maker Ratio</div>
                <div class="data-source">Source: opportunity_cost.json</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value {'positive' if net_cost > 0 else 'negative'}">
                    ${net_cost:.2f}
                </div>
                <div class="metric-label">Net Opportunity Cost</div>
                <div class="data-source">Source: opportunity_cost.json</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value positive">
                    ${rebates:.2f}
                </div>
                <div class="metric-label">Rebates Earned</div>
                <div class="data-source">Source: opportunity_cost.json</div>
            </div>
            """
        
        return tiles_html
    
    def generate_microstructure_chart(self, data):
        """Generate a small chart showing impact vs size."""
        
        if not data['impact']:
            return ""
        
        impact = data['impact']
        size_data = impact['impact_by_size']
        
        chart_html = f"""
        <div class="section">
            <h3>📊 Impact Analysis by Order Size</h3>
            <div class="chart-container">
                <div class="chart-bar">
                    <div class="bar-label">Small (0-1k)</div>
                    <div class="bar-fill" style="width: {(size_data['small_orders_0_1k']['avg_impact_bps'] / 3.0) * 100}%">
                        {size_data['small_orders_0_1k']['avg_impact_bps']:.1f} bps
                    </div>
                </div>
                <div class="chart-bar">
                    <div class="bar-label">Medium (1k-10k)</div>
                    <div class="bar-fill" style="width: {(size_data['medium_orders_1k_10k']['avg_impact_bps'] / 3.0) * 100}%">
                        {size_data['medium_orders_1k_10k']['avg_impact_bps']:.1f} bps
                    </div>
                </div>
                <div class="chart-bar">
                    <div class="bar-label">Large (10k+)</div>
                    <div class="bar-fill" style="width: {(size_data['large_orders_10k_plus']['avg_impact_bps'] / 3.0) * 100}%">
                        {size_data['large_orders_10k_plus']['avg_impact_bps']:.1f} bps
                    </div>
                </div>
            </div>
        </div>
        """
        
        return chart_html
    
    def enhance_dashboard(self):
        """Enhance the executive dashboard with microstructure metrics."""
        
        # Load data
        data = self.load_microstructure_data()
        
        # Generate tiles and chart
        tiles_html = self.generate_microstructure_tiles(data)
        chart_html = self.generate_microstructure_chart(data)
        
        # Read existing dashboard
        dashboard_file = self.reports_dir / "executive_dashboard.html"
        if not dashboard_file.exists():
            logger.error("Executive dashboard not found")
            return False
        
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            dashboard_content = f.read()
        
        # Add microstructure tiles after existing metrics
        if "<!-- MICROSTRUCTURE TILES -->" in dashboard_content:
            dashboard_content = dashboard_content.replace(
                "<!-- MICROSTRUCTURE TILES -->", 
                tiles_html
            )
        else:
            # Insert after the existing metrics grid
            metrics_end = dashboard_content.find("</div>", dashboard_content.find("metrics-grid"))
            if metrics_end != -1:
                dashboard_content = dashboard_content[:metrics_end] + tiles_html + dashboard_content[metrics_end:]
        
        # Add microstructure chart
        if "<!-- MICROSTRUCTURE CHART -->" in dashboard_content:
            dashboard_content = dashboard_content.replace(
                "<!-- MICROSTRUCTURE CHART -->", 
                chart_html
            )
        else:
            # Insert before the closing body tag
            body_end = dashboard_content.rfind("</body>")
            if body_end != -1:
                dashboard_content = dashboard_content[:body_end] + chart_html + dashboard_content[body_end:]
        
        # Add CSS for the chart
        chart_css = """
        <style>
        .chart-container { margin: 20px 0; }
        .chart-bar { 
            display: flex; 
            align-items: center; 
            margin: 10px 0; 
            height: 30px; 
        }
        .bar-label { 
            width: 120px; 
            font-size: 12px; 
            color: #666; 
        }
        .bar-fill { 
            height: 20px; 
            background: linear-gradient(90deg, #2E8B57, #32CD32); 
            border-radius: 3px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            color: white; 
            font-size: 11px; 
            font-weight: bold; 
            min-width: 60px; 
        }
        </style>
        """
        
        # Insert CSS before closing head tag
        head_end = dashboard_content.rfind("</head>")
        if head_end != -1:
            dashboard_content = dashboard_content[:head_end] + chart_css + dashboard_content[head_end:]
        
        # Write enhanced dashboard
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_content)
        
        logger.info("Dashboard enhanced with microstructure metrics")
        return True

def main():
    """Enhance the executive dashboard with microstructure metrics."""
    enhancer = MicrostructureDashboardEnhancer()
    success = enhancer.enhance_dashboard()
    
    if success:
        print("✅ Dashboard enhanced with microstructure metrics")
    else:
        print("❌ Failed to enhance dashboard")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
