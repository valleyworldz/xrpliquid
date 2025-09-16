"""
PnL Taxonomy - Complete Profit & Loss Attribution
Implements comprehensive PnL categorization and reconciliation.
"""

import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class PnLCategory(Enum):
    """PnL category enumeration."""
    DIRECTIONAL = "directional"
    FUNDING = "funding"
    FEES = "fees"
    REBATES = "rebates"
    SLIPPAGE = "slippage"
    BORROWING = "borrowing"
    INVENTORY = "inventory"
    OTHER = "other"


@dataclass
class PnLEntry:
    """Represents a PnL entry."""
    timestamp: datetime
    category: PnLCategory
    amount: float
    currency: str
    trade_id: Optional[str] = None
    position_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = None


class PnLTaxonomy:
    """Comprehensive PnL taxonomy and reconciliation system."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.accounting_dir = self.reports_dir / "accounting"
        self.accounting_dir.mkdir(parents=True, exist_ok=True)
        
        # PnL entries
        self.pnl_entries: List[PnLEntry] = []
        
        # Category definitions
        self.categories = self._define_categories()
        
        # Reconciliation rules
        self.reconciliation_rules = self._define_reconciliation_rules()
    
    def _define_categories(self) -> Dict[str, Dict[str, Any]]:
        """Define PnL categories and their characteristics."""
        
        categories = {
            "directional": {
                "description": "PnL from directional price movements",
                "subcategories": ["long_pnl", "short_pnl", "realized_pnl", "unrealized_pnl"],
                "tax_treatment": "capital_gains",
                "reconciliation_required": True
            },
            "funding": {
                "description": "Funding payments and receipts",
                "subcategories": ["funding_paid", "funding_received", "funding_arbitrage"],
                "tax_treatment": "ordinary_income",
                "reconciliation_required": True
            },
            "fees": {
                "description": "Trading fees and commissions",
                "subcategories": ["maker_fees", "taker_fees", "withdrawal_fees", "other_fees"],
                "tax_treatment": "expense",
                "reconciliation_required": True
            },
            "rebates": {
                "description": "Maker rebates and incentives",
                "subcategories": ["maker_rebates", "volume_rebates", "referral_rebates"],
                "tax_treatment": "ordinary_income",
                "reconciliation_required": True
            },
            "slippage": {
                "description": "Execution slippage costs",
                "subcategories": ["market_impact", "timing_slippage", "liquidity_slippage"],
                "tax_treatment": "expense",
                "reconciliation_required": False
            },
            "borrowing": {
                "description": "Borrowing costs and interest",
                "subcategories": ["margin_interest", "borrowing_fees", "liquidation_costs"],
                "tax_treatment": "expense",
                "reconciliation_required": True
            },
            "inventory": {
                "description": "Inventory valuation changes",
                "subcategories": ["inventory_valuation", "inventory_adjustments"],
                "tax_treatment": "capital_gains",
                "reconciliation_required": True
            },
            "other": {
                "description": "Other PnL items",
                "subcategories": ["adjustments", "corrections", "miscellaneous"],
                "tax_treatment": "varies",
                "reconciliation_required": False
            }
        }
        
        return categories
    
    def _define_reconciliation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define reconciliation rules for PnL categories."""
        
        rules = {
            "directional": {
                "must_match_equity_change": True,
                "tolerance_percent": 0.01,  # 1% tolerance
                "required_fields": ["trade_id", "position_id", "amount"]
            },
            "funding": {
                "must_match_exchange_records": True,
                "tolerance_percent": 0.001,  # 0.1% tolerance
                "required_fields": ["timestamp", "amount", "currency"]
            },
            "fees": {
                "must_match_exchange_records": True,
                "tolerance_percent": 0.001,  # 0.1% tolerance
                "required_fields": ["trade_id", "amount", "currency"]
            },
            "rebates": {
                "must_match_exchange_records": True,
                "tolerance_percent": 0.001,  # 0.1% tolerance
                "required_fields": ["timestamp", "amount", "currency"]
            }
        }
        
        return rules
    
    def add_pnl_entry(self, 
                     timestamp: datetime,
                     category: PnLCategory,
                     amount: float,
                     currency: str = "USD",
                     trade_id: str = None,
                     position_id: str = None,
                     description: str = "",
                     metadata: Dict[str, Any] = None) -> PnLEntry:
        """Add a PnL entry to the taxonomy."""
        
        entry = PnLEntry(
            timestamp=timestamp,
            category=category,
            amount=amount,
            currency=currency,
            trade_id=trade_id,
            position_id=position_id,
            description=description,
            metadata=metadata or {}
        )
        
        self.pnl_entries.append(entry)
        
        # Validate entry
        validation_result = self._validate_pnl_entry(entry)
        if not validation_result["valid"]:
            print(f"Warning: PnL entry validation failed: {validation_result['errors']}")
        
        return entry
    
    def _validate_pnl_entry(self, entry: PnLEntry) -> Dict[str, Any]:
        """Validate a PnL entry against rules."""
        
        errors = []
        
        # Check required fields based on category
        category_name = entry.category.value
        if category_name in self.reconciliation_rules:
            rule = self.reconciliation_rules[category_name]
            required_fields = rule["required_fields"]
            
            for field in required_fields:
                if field == "trade_id" and not entry.trade_id:
                    errors.append("trade_id is required for this category")
                elif field == "position_id" and not entry.position_id:
                    errors.append("position_id is required for this category")
                elif field == "amount" and entry.amount == 0:
                    errors.append("amount cannot be zero")
                elif field == "currency" and not entry.currency:
                    errors.append("currency is required")
                elif field == "timestamp" and not entry.timestamp:
                    errors.append("timestamp is required")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def calculate_category_pnl(self, 
                             category: PnLCategory,
                             start_date: datetime = None,
                             end_date: datetime = None) -> Dict[str, Any]:
        """Calculate PnL for a specific category."""
        
        # Filter entries by category and date range
        filtered_entries = [
            entry for entry in self.pnl_entries
            if entry.category == category
            and (start_date is None or entry.timestamp >= start_date)
            and (end_date is None or entry.timestamp <= end_date)
        ]
        
        if not filtered_entries:
            return {
                "category": category.value,
                "total_pnl": 0.0,
                "entry_count": 0,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        
        # Calculate statistics
        amounts = [entry.amount for entry in filtered_entries]
        total_pnl = sum(amounts)
        
        # Group by currency
        currency_breakdown = {}
        for entry in filtered_entries:
            currency = entry.currency
            if currency not in currency_breakdown:
                currency_breakdown[currency] = 0.0
            currency_breakdown[currency] += entry.amount
        
        return {
            "category": category.value,
            "total_pnl": total_pnl,
            "entry_count": len(filtered_entries),
            "currency_breakdown": currency_breakdown,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "min_amount": min(amounts),
            "max_amount": max(amounts),
            "avg_amount": total_pnl / len(amounts)
        }
    
    def calculate_total_pnl(self, 
                          start_date: datetime = None,
                          end_date: datetime = None) -> Dict[str, Any]:
        """Calculate total PnL across all categories."""
        
        total_pnl = 0.0
        category_breakdown = {}
        
        for category in PnLCategory:
            category_pnl = self.calculate_category_pnl(category, start_date, end_date)
            category_breakdown[category.value] = category_pnl
            total_pnl += category_pnl["total_pnl"]
        
        return {
            "total_pnl": total_pnl,
            "category_breakdown": category_breakdown,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "total_entries": sum(cat["entry_count"] for cat in category_breakdown.values())
        }
    
    def reconcile_with_exchange(self, 
                              exchange_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reconcile PnL entries with exchange data."""
        
        reconciliation_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_exchange_records": len(exchange_data),
            "total_pnl_entries": len(self.pnl_entries),
            "reconciliation_status": "pending",
            "discrepancies": [],
            "category_reconciliations": {}
        }
        
        # Reconcile each category
        for category in PnLCategory:
            category_name = category.value
            
            if category_name in self.reconciliation_rules:
                rule = self.reconciliation_rules[category_name]
                
                if rule["must_match_exchange_records"]:
                    category_result = self._reconcile_category_with_exchange(
                        category, exchange_data, rule
                    )
                    reconciliation_results["category_reconciliations"][category_name] = category_result
                    
                    if category_result["discrepancies"]:
                        reconciliation_results["discrepancies"].extend(category_result["discrepancies"])
        
        # Determine overall reconciliation status
        if reconciliation_results["discrepancies"]:
            reconciliation_results["reconciliation_status"] = "discrepancies_found"
        else:
            reconciliation_results["reconciliation_status"] = "reconciled"
        
        return reconciliation_results
    
    def _reconcile_category_with_exchange(self, 
                                        category: PnLCategory,
                                        exchange_data: List[Dict[str, Any]],
                                        rule: Dict[str, Any]) -> Dict[str, Any]:
        """Reconcile a specific category with exchange data."""
        
        # Filter PnL entries for this category
        category_entries = [entry for entry in self.pnl_entries if entry.category == category]
        
        # Filter exchange data for this category (simplified)
        exchange_records = [record for record in exchange_data if record.get("category") == category.value]
        
        discrepancies = []
        
        # Simple reconciliation logic (can be made more sophisticated)
        pnl_total = sum(entry.amount for entry in category_entries)
        exchange_total = sum(record.get("amount", 0) for record in exchange_records)
        
        tolerance = rule["tolerance_percent"]
        difference = abs(pnl_total - exchange_total)
        max_difference = abs(pnl_total) * tolerance
        
        if difference > max_difference:
            discrepancies.append({
                "category": category.value,
                "type": "amount_mismatch",
                "pnl_total": pnl_total,
                "exchange_total": exchange_total,
                "difference": difference,
                "tolerance": max_difference,
                "severity": "high" if difference > max_difference * 2 else "medium"
            })
        
        return {
            "category": category.value,
            "pnl_entries_count": len(category_entries),
            "exchange_records_count": len(exchange_records),
            "pnl_total": pnl_total,
            "exchange_total": exchange_total,
            "difference": difference,
            "within_tolerance": difference <= max_difference,
            "discrepancies": discrepancies
        }
    
    def generate_tax_report(self, 
                          year: int,
                          currency: str = "USD") -> Dict[str, Any]:
        """Generate tax report for a specific year."""
        
        start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        
        # Filter entries for the year
        year_entries = [
            entry for entry in self.pnl_entries
            if start_date <= entry.timestamp <= end_date
            and entry.currency == currency
        ]
        
        # Group by tax treatment
        tax_groups = {
            "capital_gains": [],
            "ordinary_income": [],
            "expense": [],
            "varies": []
        }
        
        for entry in year_entries:
            category_info = self.categories.get(entry.category.value, {})
            tax_treatment = category_info.get("tax_treatment", "varies")
            tax_groups[tax_treatment].append(entry)
        
        # Calculate totals by tax treatment
        tax_totals = {}
        for treatment, entries in tax_groups.items():
            tax_totals[treatment] = sum(entry.amount for entry in entries)
        
        return {
            "year": year,
            "currency": currency,
            "total_entries": len(year_entries),
            "tax_totals": tax_totals,
            "category_breakdown": {
                category.value: self.calculate_category_pnl(category, start_date, end_date)
                for category in PnLCategory
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def export_to_csv(self, output_file: str = None) -> str:
        """Export PnL entries to CSV format."""
        
        if output_file is None:
            output_file = self.accounting_dir / f"pnl_entries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Convert entries to DataFrame
        data = []
        for entry in self.pnl_entries:
            data.append({
                "timestamp": entry.timestamp.isoformat(),
                "category": entry.category.value,
                "amount": entry.amount,
                "currency": entry.currency,
                "trade_id": entry.trade_id or "",
                "position_id": entry.position_id or "",
                "description": entry.description,
                "metadata": json.dumps(entry.metadata) if entry.metadata else ""
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        return str(output_file)
    
    def save_taxonomy_report(self) -> str:
        """Save comprehensive PnL taxonomy report."""
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(self.pnl_entries),
            "categories_defined": len(self.categories),
            "reconciliation_rules": len(self.reconciliation_rules),
            "total_pnl_summary": self.calculate_total_pnl(),
            "category_definitions": self.categories,
            "reconciliation_rules": self.reconciliation_rules
        }
        
        report_file = self.accounting_dir / f"pnl_taxonomy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(report_file)


def main():
    """Test PnL taxonomy functionality."""
    taxonomy = PnLTaxonomy()
    
    # Add sample PnL entries
    now = datetime.now(timezone.utc)
    
    # Directional PnL
    taxonomy.add_pnl_entry(
        timestamp=now,
        category=PnLCategory.DIRECTIONAL,
        amount=150.0,
        trade_id="trade_001",
        description="Long XRP position profit"
    )
    
    # Funding PnL
    taxonomy.add_pnl_entry(
        timestamp=now,
        category=PnLCategory.FUNDING,
        amount=25.0,
        description="Funding payment received"
    )
    
    # Fees
    taxonomy.add_pnl_entry(
        timestamp=now,
        category=PnLCategory.FEES,
        amount=-5.0,
        trade_id="trade_001",
        description="Trading fees"
    )
    
    # Rebates
    taxonomy.add_pnl_entry(
        timestamp=now,
        category=PnLCategory.REBATES,
        amount=2.5,
        description="Maker rebate"
    )
    
    # Calculate totals
    total_pnl = taxonomy.calculate_total_pnl()
    print(f"✅ Total PnL: ${total_pnl['total_pnl']:.2f}")
    
    # Calculate by category
    directional_pnl = taxonomy.calculate_category_pnl(PnLCategory.DIRECTIONAL)
    print(f"✅ Directional PnL: ${directional_pnl['total_pnl']:.2f}")
    
    # Generate tax report
    tax_report = taxonomy.generate_tax_report(2024)
    print(f"✅ Tax report generated: {len(tax_report['tax_totals'])} tax categories")
    
    # Export to CSV
    csv_file = taxonomy.export_to_csv()
    print(f"✅ PnL entries exported to: {csv_file}")
    
    print("✅ PnL taxonomy testing completed")


if __name__ == "__main__":
    main()
