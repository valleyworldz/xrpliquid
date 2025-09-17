"""
Update Hash Manifest - Add all new crown-tier artifacts to hash manifest
"""

import os
import json
import hashlib
from datetime import datetime

def update_hash_manifest():
    """
    Update hash manifest with all new crown-tier artifacts
    """
    print("🔗 Updating Hash Manifest")
    print("=" * 50)
    
    # Read existing manifest
    manifest_path = "reports/hash_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {
            "created_at": datetime.now().isoformat(),
            "artifacts": {}
        }
    
    # New artifacts to add
    new_artifacts = [
        "reports/risk/var_es.json",
        "reports/reconciliation/exchange_vs_ledger.json",
        "reports/crown/arbitrage/trades.csv",
        "reports/crown/arbitrage/pnl_summary.json",
        "reports/crown/arbitrage/methodology.md",
        "reports/crown/capacity/slippage_curves.csv",
        "reports/crown/capacity/orderbook_snapshots.json",
        "reports/crown/capacity/config.json",
        "reports/crown/audits/quant_fund_auditor_report.html",
        "reports/crown/attribution/per_trade_ledger.csv",
        "reports/crown/attribution/six_component_breakdown.json",
        "reports/crown/attribution/reconciliation_proof.py",
        "reports/latency/trace/raw_traces.csv",
        "reports/latency/trace/summary.json",
        "reports/tests/decimal_discipline_proof.txt"
    ]
    
    added_count = 0
    
    for artifact_path in new_artifacts:
        if os.path.exists(artifact_path):
            try:
                with open(artifact_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                manifest["artifacts"][artifact_path] = {
                    "sha256": file_hash,
                    "size_bytes": os.path.getsize(artifact_path),
                    "created_at": datetime.fromtimestamp(os.path.getctime(artifact_path)).isoformat()
                }
                
                print(f"  ✅ Added: {artifact_path}")
                added_count += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {artifact_path}: {e}")
        else:
            print(f"  ⚠️ File not found: {artifact_path}")
    
    # Update manifest timestamp
    manifest["last_updated"] = datetime.now().isoformat()
    manifest["total_artifacts"] = len(manifest["artifacts"])
    
    # Write updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📊 Hash Manifest Updated")
    print(f"  Total artifacts: {manifest['total_artifacts']}")
    print(f"  New artifacts added: {added_count}")
    print(f"  Manifest file: {manifest_path}")

if __name__ == "__main__":
    update_hash_manifest()
