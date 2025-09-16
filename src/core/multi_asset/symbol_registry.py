"""
Symbol Registry - Multi-Asset Trading Support
Implements symbol registry, position netting, and throughput optimization.
"""

import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class AssetType(Enum):
    """Asset type enumeration."""
    PERPETUAL = "perpetual"
    SPOT = "spot"
    FUTURE = "future"
    OPTION = "option"


@dataclass
class SymbolConfig:
    """Symbol configuration for multi-asset trading."""
    symbol: str
    asset_type: AssetType
    base_asset: str
    quote_asset: str
    tick_size: float
    lot_size: float
    min_notional: float
    max_position_size: float
    funding_interval_hours: int
    maker_fee: float
    taker_fee: float
    maker_rebate: float
    risk_bands: Dict[str, float]
    correlation_groups: List[str]


class SymbolRegistry:
    """Registry for multi-asset trading symbols and configurations."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.multi_asset_dir = self.reports_dir / "multi_asset"
        self.multi_asset_dir.mkdir(parents=True, exist_ok=True)
        
        # Symbol registry
        self.symbols: Dict[str, SymbolConfig] = {}
        
        # Position netting
        self.position_netting = {}
        
        # Throughput monitoring
        self.throughput_metrics = {}
        
        # Load registry
        self._load_registry()
    
    def register_symbol(self, config: SymbolConfig) -> bool:
        """Register a new symbol in the registry."""
        
        # Validate configuration
        validation_result = self._validate_symbol_config(config)
        if not validation_result["valid"]:
            print(f"Symbol registration failed: {validation_result['errors']}")
            return False
        
        # Register symbol
        self.symbols[config.symbol] = config
        
        # Save registry
        self._save_registry()
        
        return True
    
    def _validate_symbol_config(self, config: SymbolConfig) -> Dict[str, Any]:
        """Validate symbol configuration."""
        
        errors = []
        
        # Check required fields
        if not config.symbol:
            errors.append("Symbol name is required")
        
        if not config.base_asset or not config.quote_asset:
            errors.append("Base and quote assets are required")
        
        if config.tick_size <= 0:
            errors.append("Tick size must be positive")
        
        if config.lot_size <= 0:
            errors.append("Lot size must be positive")
        
        if config.min_notional <= 0:
            errors.append("Minimum notional must be positive")
        
        if config.max_position_size <= 0:
            errors.append("Maximum position size must be positive")
        
        # Check fee structure
        if config.maker_fee < 0 or config.taker_fee < 0:
            errors.append("Fees cannot be negative")
        
        if config.maker_rebate < 0:
            errors.append("Maker rebate cannot be negative")
        
        # Check risk bands
        if not config.risk_bands:
            errors.append("Risk bands are required")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Get symbol configuration."""
        return self.symbols.get(symbol)
    
    def list_symbols(self, 
                    asset_type: AssetType = None,
                    base_asset: str = None,
                    quote_asset: str = None) -> List[str]:
        """List symbols matching criteria."""
        
        matching_symbols = []
        
        for symbol, config in self.symbols.items():
            if asset_type and config.asset_type != asset_type:
                continue
            
            if base_asset and config.base_asset != base_asset:
                continue
            
            if quote_asset and config.quote_asset != quote_asset:
                continue
            
            matching_symbols.append(symbol)
        
        return matching_symbols
    
    def calculate_position_netting(self, 
                                 positions: Dict[str, float],
                                 correlation_threshold: float = 0.7) -> Dict[str, Any]:
        """Calculate position netting across correlated assets."""
        
        netting_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_positions": len(positions),
            "netting_groups": [],
            "net_exposure": {},
            "correlation_analysis": {}
        }
        
        # Group positions by correlation
        processed_symbols = set()
        
        for symbol, position in positions.items():
            if symbol in processed_symbols:
                continue
            
            if symbol not in self.symbols:
                continue
            
            config = self.symbols[symbol]
            
            # Find correlated symbols
            correlated_symbols = [symbol]
            for other_symbol, other_position in positions.items():
                if other_symbol in processed_symbols or other_symbol == symbol:
                    continue
                
                if other_symbol in self.symbols:
                    other_config = self.symbols[other_symbol]
                    
                    # Check if symbols are in same correlation group
                    if any(group in other_config.correlation_groups for group in config.correlation_groups):
                        correlated_symbols.append(other_symbol)
            
            # Calculate net position for group
            if len(correlated_symbols) > 1:
                group_positions = {s: positions[s] for s in correlated_symbols}
                net_position = sum(group_positions.values())
                
                netting_result["netting_groups"].append({
                    "group_id": f"group_{len(netting_result['netting_groups'])}",
                    "symbols": correlated_symbols,
                    "individual_positions": group_positions,
                    "net_position": net_position,
                    "netting_ratio": abs(net_position) / sum(abs(p) for p in group_positions.values()) if sum(abs(p) for p in group_positions.values()) > 0 else 0
                })
                
                # Add to net exposure
                for s in correlated_symbols:
                    netting_result["net_exposure"][s] = net_position / len(correlated_symbols)
                
                processed_symbols.update(correlated_symbols)
            else:
                # Single symbol, no netting
                netting_result["net_exposure"][symbol] = position
                processed_symbols.add(symbol)
        
        return netting_result
    
    def optimize_throughput(self, 
                          symbols: List[str],
                          target_latency_ms: float = 250.0) -> Dict[str, Any]:
        """Optimize throughput for multiple symbols."""
        
        optimization_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_latency_ms": target_latency_ms,
            "symbols_analyzed": len(symbols),
            "optimization_recommendations": [],
            "throughput_limits": {}
        }
        
        # Analyze each symbol
        for symbol in symbols:
            if symbol not in self.symbols:
                continue
            
            config = self.symbols[symbol]
            
            # Calculate theoretical throughput limits
            tick_processing_time = 0.1  # ms per tick
            order_processing_time = 5.0  # ms per order
            
            # Estimate throughput based on symbol characteristics
            if config.asset_type == AssetType.PERPETUAL:
                # Perpetuals have higher throughput requirements
                max_orders_per_second = 1000 / (tick_processing_time + order_processing_time)
            else:
                # Spot has lower throughput requirements
                max_orders_per_second = 500 / (tick_processing_time + order_processing_time)
            
            # Adjust based on tick size and lot size
            complexity_factor = 1.0 + (1.0 / config.tick_size) * 0.01 + (1.0 / config.lot_size) * 0.01
            adjusted_throughput = max_orders_per_second / complexity_factor
            
            optimization_result["throughput_limits"][symbol] = {
                "max_orders_per_second": adjusted_throughput,
                "complexity_factor": complexity_factor,
                "estimated_latency_ms": (tick_processing_time + order_processing_time) * complexity_factor
            }
            
            # Check if latency target is achievable
            if optimization_result["throughput_limits"][symbol]["estimated_latency_ms"] > target_latency_ms:
                optimization_result["optimization_recommendations"].append({
                    "symbol": symbol,
                    "issue": "latency_target_exceeded",
                    "current_latency_ms": optimization_result["throughput_limits"][symbol]["estimated_latency_ms"],
                    "target_latency_ms": target_latency_ms,
                    "recommendation": "Reduce tick size or lot size complexity"
                })
        
        return optimization_result
    
    def create_correlation_matrix(self, 
                                symbols: List[str],
                                lookback_days: int = 30) -> Dict[str, Any]:
        """Create correlation matrix for symbol analysis."""
        
        correlation_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols,
            "lookback_days": lookback_days,
            "correlation_matrix": {},
            "high_correlation_pairs": [],
            "correlation_groups": []
        }
        
        # Generate sample correlation data (in practice, this would use real price data)
        import numpy as np
        np.random.seed(42)
        
        # Create correlation matrix
        n_symbols = len(symbols)
        correlation_matrix = np.random.rand(n_symbols, n_symbols)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
        
        # Store correlation matrix
        for i, symbol1 in enumerate(symbols):
            correlation_result["correlation_matrix"][symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                correlation_result["correlation_matrix"][symbol1][symbol2] = float(correlation_matrix[i, j])
        
        # Find high correlation pairs
        threshold = 0.7
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j and abs(correlation_matrix[i, j]) > threshold:
                    correlation_result["high_correlation_pairs"].append({
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "correlation": float(correlation_matrix[i, j])
                    })
        
        # Group highly correlated symbols
        processed_symbols = set()
        for pair in correlation_result["high_correlation_pairs"]:
            if pair["symbol1"] not in processed_symbols and pair["symbol2"] not in processed_symbols:
                group = [pair["symbol1"], pair["symbol2"]]
                correlation_result["correlation_groups"].append(group)
                processed_symbols.update(group)
        
        return correlation_result
    
    def generate_multi_asset_report(self) -> Dict[str, Any]:
        """Generate comprehensive multi-asset report."""
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_symbols": len(self.symbols),
            "asset_type_breakdown": {},
            "risk_band_summary": {},
            "throughput_summary": {},
            "correlation_summary": {}
        }
        
        # Asset type breakdown
        for symbol, config in self.symbols.items():
            asset_type = config.asset_type.value
            if asset_type not in report["asset_type_breakdown"]:
                report["asset_type_breakdown"][asset_type] = 0
            report["asset_type_breakdown"][asset_type] += 1
        
        # Risk band summary
        all_risk_bands = set()
        for config in self.symbols.values():
            all_risk_bands.update(config.risk_bands.keys())
        
        for band in all_risk_bands:
            band_values = [config.risk_bands.get(band, 0) for config in self.symbols.values()]
            report["risk_band_summary"][band] = {
                "min": min(band_values),
                "max": max(band_values),
                "mean": sum(band_values) / len(band_values)
            }
        
        # Save report
        report_file = self.multi_asset_dir / f"multi_asset_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _save_registry(self):
        """Save symbol registry to disk."""
        
        registry_data = {
            "symbols": {
                symbol: {
                    "symbol": config.symbol,
                    "asset_type": config.asset_type.value,
                    "base_asset": config.base_asset,
                    "quote_asset": config.quote_asset,
                    "tick_size": config.tick_size,
                    "lot_size": config.lot_size,
                    "min_notional": config.min_notional,
                    "max_position_size": config.max_position_size,
                    "funding_interval_hours": config.funding_interval_hours,
                    "maker_fee": config.maker_fee,
                    "taker_fee": config.taker_fee,
                    "maker_rebate": config.maker_rebate,
                    "risk_bands": config.risk_bands,
                    "correlation_groups": config.correlation_groups
                }
                for symbol, config in self.symbols.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        registry_file = self.multi_asset_dir / "symbol_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def _load_registry(self):
        """Load symbol registry from disk."""
        
        registry_file = self.multi_asset_dir / "symbol_registry.json"
        if not registry_file.exists():
            # Create default registry with XRP
            self._create_default_registry()
            return
        
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            for symbol, config_data in registry_data["symbols"].items():
                config = SymbolConfig(
                    symbol=config_data["symbol"],
                    asset_type=AssetType(config_data["asset_type"]),
                    base_asset=config_data["base_asset"],
                    quote_asset=config_data["quote_asset"],
                    tick_size=config_data["tick_size"],
                    lot_size=config_data["lot_size"],
                    min_notional=config_data["min_notional"],
                    max_position_size=config_data["max_position_size"],
                    funding_interval_hours=config_data["funding_interval_hours"],
                    maker_fee=config_data["maker_fee"],
                    taker_fee=config_data["taker_fee"],
                    maker_rebate=config_data["maker_rebate"],
                    risk_bands=config_data["risk_bands"],
                    correlation_groups=config_data["correlation_groups"]
                )
                self.symbols[symbol] = config
                
        except Exception as e:
            print(f"Warning: Could not load symbol registry: {e}")
            self._create_default_registry()
    
    def _create_default_registry(self):
        """Create default symbol registry with XRP."""
        
        xrp_config = SymbolConfig(
            symbol="XRP",
            asset_type=AssetType.PERPETUAL,
            base_asset="XRP",
            quote_asset="USD",
            tick_size=0.0001,
            lot_size=0.1,
            min_notional=1.0,
            max_position_size=100000.0,
            funding_interval_hours=1,
            maker_fee=0.0001,
            taker_fee=0.0005,
            maker_rebate=0.00005,
            risk_bands={
                "low": 0.02,
                "medium": 0.05,
                "high": 0.10
            },
            correlation_groups=["crypto_majors"]
        )
        
        self.symbols["XRP"] = xrp_config
        self._save_registry()


def main():
    """Test symbol registry functionality."""
    registry = SymbolRegistry()
    
    # Test symbol registration
    btc_config = SymbolConfig(
        symbol="BTC",
        asset_type=AssetType.PERPETUAL,
        base_asset="BTC",
        quote_asset="USD",
        tick_size=0.01,
        lot_size=0.001,
        min_notional=10.0,
        max_position_size=1000000.0,
        funding_interval_hours=1,
        maker_fee=0.0001,
        taker_fee=0.0005,
        maker_rebate=0.00005,
        risk_bands={"low": 0.01, "medium": 0.03, "high": 0.05},
        correlation_groups=["crypto_majors"]
    )
    
    success = registry.register_symbol(btc_config)
    print(f"✅ BTC registration: {success}")
    
    # Test symbol listing
    crypto_symbols = registry.list_symbols(asset_type=AssetType.PERPETUAL)
    print(f"✅ Crypto symbols: {crypto_symbols}")
    
    # Test position netting
    positions = {"XRP": 100.0, "BTC": -50.0}
    netting_result = registry.calculate_position_netting(positions)
    print(f"✅ Position netting: {len(netting_result['netting_groups'])} groups")
    
    # Test throughput optimization
    throughput_result = registry.optimize_throughput(["XRP", "BTC"])
    print(f"✅ Throughput optimization: {len(throughput_result['optimization_recommendations'])} recommendations")
    
    # Test correlation matrix
    correlation_result = registry.create_correlation_matrix(["XRP", "BTC"])
    print(f"✅ Correlation matrix: {len(correlation_result['high_correlation_pairs'])} high correlation pairs")
    
    # Generate report
    report = registry.generate_multi_asset_report()
    print(f"✅ Multi-asset report: {report['total_symbols']} symbols")
    
    print("✅ Symbol registry testing completed")


if __name__ == "__main__":
    main()
