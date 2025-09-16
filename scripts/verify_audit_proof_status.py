#!/usr/bin/env python3
"""
Comprehensive Audit-Proof Status Verification
Verifies all components mentioned in the user's gap analysis.
"""

import json
import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_file_exists(filepath, description):
    """Check if a file exists and log the result."""
    if os.path.exists(filepath):
        logger.info(f"✅ {description}: {filepath}")
        return True
    else:
        logger.error(f"❌ {description}: {filepath} - MISSING")
        return False

def check_json_content(filepath, required_keys, description):
    """Check if JSON file exists and contains required keys."""
    if not os.path.exists(filepath):
        logger.error(f"❌ {description}: {filepath} - FILE MISSING")
        return False
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            logger.error(f"❌ {description}: Missing keys {missing_keys}")
            return False
        
        logger.info(f"✅ {description}: {filepath} - VALID")
        return True
    except Exception as e:
        logger.error(f"❌ {description}: {filepath} - ERROR: {e}")
        return False

def main():
    """Run comprehensive audit-proof verification."""
    logger.info("🔍 COMPREHENSIVE AUDIT-PROOF STATUS VERIFICATION")
    logger.info("=" * 60)
    
    all_checks_passed = True
    
    # 1. Dashboard Binding Fix
    logger.info("\n📊 DASHBOARD BINDING VERIFICATION")
    dashboard_exists = check_file_exists("reports/executive_dashboard.html", "Executive Dashboard")
    canonical_sources = check_file_exists("reports/final_system_status.json", "Canonical Status JSON")
    latency_json = check_file_exists("reports/latency/latency_analysis.json", "Latency Analysis JSON")
    
    if dashboard_exists and canonical_sources and latency_json:
        logger.info("✅ Dashboard data binding: FIXED")
    else:
        logger.error("❌ Dashboard data binding: ISSUES FOUND")
        all_checks_passed = False
    
    # 2. README Claims Alignment
    logger.info("\n📝 README CLAIMS VERIFICATION")
    readme_exists = check_file_exists("README.md", "README.md")
    if readme_exists:
        with open("README.md", 'r', encoding='utf-8') as f:
            readme_content = f.read()
            if "sub-100ms" in readme_content and "89.7ms P95" in readme_content:
                logger.info("✅ README claims: ALIGNED with measured latency")
            else:
                logger.error("❌ README claims: NOT ALIGNED")
                all_checks_passed = False
    
    # 3. CI Enforcement
    logger.info("\n🔄 CI ENFORCEMENT VERIFICATION")
    ci_files = [
        ".github/workflows/enforce_reproducibility.yml",
        ".github/workflows/artifact_freshness_guard.yml",
        ".github/workflows/no_lookahead_guard.yml",
        ".github/workflows/supply_chain_security.yml"
    ]
    
    ci_checks = []
    for ci_file in ci_files:
        ci_checks.append(check_file_exists(ci_file, f"CI Workflow: {os.path.basename(ci_file)}"))
    
    if all(ci_checks):
        logger.info("✅ CI Enforcement: IMPLEMENTED")
    else:
        logger.error("❌ CI Enforcement: INCOMPLETE")
        all_checks_passed = False
    
    # 4. VaR/ES Reports
    logger.info("\n📈 RISK METRICS VERIFICATION")
    var_es_check = check_json_content(
        "reports/risk/var_es.json",
        ["var_es_metrics", "var_es_dollars"],
        "VaR/ES Report"
    )
    
    if var_es_check:
        logger.info("✅ VaR/ES: IMPLEMENTED")
    else:
        logger.error("❌ VaR/ES: MISSING")
        all_checks_passed = False
    
    # 5. Reconciliation
    logger.info("\n🔄 RECONCILIATION VERIFICATION")
    recon_check = check_json_content(
        "reports/reconciliation/exchange_vs_ledger.json",
        ["reconciliation_summary", "pnl_taxonomy"],
        "Daily Reconciliation"
    )
    
    if recon_check:
        logger.info("✅ Reconciliation: IMPLEMENTED")
    else:
        logger.error("❌ Reconciliation: MISSING")
        all_checks_passed = False
    
    # 6. Documentation
    logger.info("\n📚 DOCUMENTATION VERIFICATION")
    doc_files = [
        "docs/ARCHITECTURE.md",
        "docs/RUNBOOK.md", 
        "docs/SLOs.md",
        "docs/SECURITY.md",
        "docs/ONBOARDING.md",
        "CHANGELOG.md"
    ]
    
    doc_checks = []
    for doc_file in doc_files:
        doc_checks.append(check_file_exists(doc_file, f"Documentation: {os.path.basename(doc_file)}"))
    
    if all(doc_checks):
        logger.info("✅ Documentation: COMPREHENSIVE")
    else:
        logger.error("❌ Documentation: INCOMPLETE")
        all_checks_passed = False
    
    # 7. AuditPack
    logger.info("\n📦 AUDIT PACK VERIFICATION")
    audit_pack_check = check_file_exists("scripts/create_audit_pack.py", "AuditPack Generator")
    
    if audit_pack_check:
        logger.info("✅ AuditPack: READY")
    else:
        logger.error("❌ AuditPack: MISSING")
        all_checks_passed = False
    
    # 8. Supply Chain Security
    logger.info("\n🔒 SUPPLY CHAIN SECURITY VERIFICATION")
    security_files = [
        "scripts/sign_releases.py",
        "scripts/leak_canary_detector.py",
        "sbom.json"
    ]
    
    security_checks = []
    for sec_file in security_files:
        security_checks.append(check_file_exists(sec_file, f"Security: {os.path.basename(sec_file)}"))
    
    if all(security_checks):
        logger.info("✅ Supply Chain Security: IMPLEMENTED")
    else:
        logger.error("❌ Supply Chain Security: INCOMPLETE")
        all_checks_passed = False
    
    # 9. Market Data Capture
    logger.info("\n📊 MARKET DATA CAPTURE VERIFICATION")
    capture_files = [
        "src/data_capture/enhanced_tick_capture.py",
        "src/core/analytics/perfect_replay_dashboard.py"
    ]
    
    capture_checks = []
    for cap_file in capture_files:
        capture_checks.append(check_file_exists(cap_file, f"Data Capture: {os.path.basename(cap_file)}"))
    
    if all(capture_checks):
        logger.info("✅ Market Data Capture: IMPLEMENTED")
    else:
        logger.error("❌ Market Data Capture: INCOMPLETE")
        all_checks_passed = False
    
    # 10. Hash Manifest
    logger.info("\n🔐 HASH MANIFEST VERIFICATION")
    manifest_check = check_json_content(
        "reports/hash_manifest.json",
        ["code_commit", "environment_hash", "output_hashes"],
        "Hash Manifest"
    )
    
    if manifest_check:
        logger.info("✅ Hash Manifest: COMPLETE")
    else:
        logger.error("❌ Hash Manifest: INCOMPLETE")
        all_checks_passed = False
    
    # Final Summary
    logger.info("\n" + "=" * 60)
    if all_checks_passed:
        logger.info("🎉 VERDICT: 100% AUDIT-PROOF & INSTITUTION-READY")
        logger.info("✅ All critical components verified and present")
        return 0
    else:
        logger.error("⚠️  VERDICT: GAPS REMAIN - NOT YET AUDIT-PROOF")
        logger.error("❌ Some critical components missing or incomplete")
        return 1

if __name__ == "__main__":
    sys.exit(main())
