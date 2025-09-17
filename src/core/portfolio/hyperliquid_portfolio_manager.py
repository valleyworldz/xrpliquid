#!/usr/bin/env python3
"""
Hyperliquid Portfolio Manager
============================
Multi-market portfolio risk management across Hyperliquid perps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os

class HyperliquidPortfolioManager:
    """
    Manages portfolio risk across multiple Hyperliquid perpetual markets
    with VaR/ES calculations and factor exposure analysis.
    """
    
    def __init__(self):
        self.markets = ["XRP", "BTC", "ETH", "SOL", "ARB", "AVAX"]
        self.reports_dir = "reports/portfolio"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def calculate_portfolio_risk(self, positions: Dict[str, float], 
                               mark_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio VaR/ES and factor exposures"""
        try:
            # Calculate portfolio value
            portfolio_value = sum(positions[market] * mark_prices[market] 
                                for market in positions.keys())
            
            # Calculate individual market risks
            market_risks = {}
            for market in positions.keys():
                if market in mark_prices:
                    market_risks[market] = {
                        "position": positions[market],
                        "mark_price": mark_prices[market],
                        "notional": positions[market] * mark_prices[market],
                        "weight": (positions[market] * mark_prices[market]) / portfolio_value
                    }
            
            # Calculate portfolio VaR (simplified)
            portfolio_var_95 = self._calculate_portfolio_var(market_risks, 0.95)
            portfolio_var_99 = self._calculate_portfolio_var(market_risks, 0.99)
            
            # Calculate Expected Shortfall
            portfolio_es_95 = portfolio_var_95 * 1.2  # Simplified ES calculation
            portfolio_es_99 = portfolio_var_99 * 1.3
            
            # Calculate factor exposures
            factor_exposures = self._calculate_factor_exposures(market_risks)
            
            # Generate correlation heatmap
            correlation_matrix = self._generate_correlation_matrix(market_risks)
            
            risk_metrics = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
                "market_risks": market_risks,
                "portfolio_var_95": portfolio_var_95,
                "portfolio_var_99": portfolio_var_99,
                "portfolio_es_95": portfolio_es_95,
                "portfolio_es_99": portfolio_es_99,
                "factor_exposures": factor_exposures,
                "correlation_matrix": correlation_matrix
            }
            
            # Save risk report
            self._save_risk_report(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            print(f"Error calculating portfolio risk: {e}")
            return {}
    
    def _calculate_portfolio_var(self, market_risks: Dict[str, Any], confidence: float) -> float:
        """Calculate portfolio VaR"""
        # Simplified VaR calculation
        total_notional = sum(risk["notional"] for risk in market_risks.values())
        
        # Assume 2% daily volatility for crypto
        daily_volatility = 0.02
        
        # VaR calculation
        if confidence == 0.95:
            z_score = 1.645
        elif confidence == 0.99:
            z_score = 2.326
        else:
            z_score = 1.645
        
        var = total_notional * daily_volatility * z_score
        return var
    
    def _calculate_factor_exposures(self, market_risks: Dict[str, Any]) -> Dict[str, float]:
        """Calculate factor exposures (crypto, DeFi, etc.)"""
        factor_exposures = {
            "crypto_beta": 0.0,
            "defi_exposure": 0.0,
            "layer1_exposure": 0.0,
            "layer2_exposure": 0.0
        }
        
        for market, risk in market_risks.items():
            weight = risk["weight"]
            
            # Categorize by market type
            if market in ["BTC", "ETH"]:
                factor_exposures["crypto_beta"] += weight * 0.8
                factor_exposures["layer1_exposure"] += weight
            elif market in ["SOL", "AVAX"]:
                factor_exposures["crypto_beta"] += weight * 0.7
                factor_exposures["layer1_exposure"] += weight
            elif market in ["ARB"]:
                factor_exposures["layer2_exposure"] += weight
            elif market in ["XRP"]:
                factor_exposures["crypto_beta"] += weight * 0.6
        
        return factor_exposures
    
    def _generate_correlation_matrix(self, market_risks: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Generate correlation matrix between markets"""
        markets = list(market_risks.keys())
        correlation_matrix = {}
        
        # Simplified correlation matrix (in practice, use historical data)
        base_correlations = {
            "BTC": {"BTC": 1.0, "ETH": 0.8, "SOL": 0.7, "XRP": 0.6, "ARB": 0.5, "AVAX": 0.6},
            "ETH": {"BTC": 0.8, "ETH": 1.0, "SOL": 0.8, "XRP": 0.5, "ARB": 0.7, "AVAX": 0.7},
            "SOL": {"BTC": 0.7, "ETH": 0.8, "SOL": 1.0, "XRP": 0.4, "ARB": 0.6, "AVAX": 0.8},
            "XRP": {"BTC": 0.6, "ETH": 0.5, "SOL": 0.4, "XRP": 1.0, "ARB": 0.3, "AVAX": 0.4},
            "ARB": {"BTC": 0.5, "ETH": 0.7, "SOL": 0.6, "XRP": 0.3, "ARB": 1.0, "AVAX": 0.5},
            "AVAX": {"BTC": 0.6, "ETH": 0.7, "SOL": 0.8, "XRP": 0.4, "ARB": 0.5, "AVAX": 1.0}
        }
        
        for market1 in markets:
            correlation_matrix[market1] = {}
            for market2 in markets:
                correlation_matrix[market1][market2] = base_correlations.get(market1, {}).get(market2, 0.5)
        
        return correlation_matrix
    
    def _save_risk_report(self, risk_metrics: Dict[str, Any]):
        """Save portfolio risk report"""
        try:
            filename = f"portfolio_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.reports_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(risk_metrics, f, indent=2)
            
            # Also save latest
            latest_filepath = os.path.join(self.reports_dir, "portfolio_risk_latest.json")
            with open(latest_filepath, 'w') as f:
                json.dump(risk_metrics, f, indent=2)
                
        except Exception as e:
            print(f"Error saving risk report: {e}")
