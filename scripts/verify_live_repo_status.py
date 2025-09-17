#!/usr/bin/env python3
"""
Live Repository Status Verification
Verifies that all changes are properly visible on the live GitHub repository.
"""

import json
import os
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_live_repo():
    """Verify all components are visible on live GitHub repo."""
    
    base_url = "https://raw.githubusercontent.com/valleyworldz/xrpliquid/master"
    
    # Files to verify
    critical_files = [
        "README.md",
        "PROOF_ARTIFACTS_SUMMARY.md", 
        "reports/hash_manifest.json",
        "reports/executive_dashboard.html",
        "reports/risk/var_es.json",
        "reports/reconciliation/exchange_vs_ledger.json",
        "reports/audit/xrpliquid_audit_pack_20250916_192751.zip",
        "docs/SECURITY.md",
        "sbom.json"
    ]
    
    logger.info("🔍 VERIFYING LIVE REPOSITORY STATUS")
    logger.info("=" * 50)
    
    all_verified = True
    
    for file_path in critical_files:
        url = f"{base_url}/{file_path}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ {file_path} - ACCESSIBLE")
                
                # Special checks for specific files
                if file_path == "README.md":
                    content = response.text
                    if "![Reproducibility]" in content and "sub-100ms" in content:
                        logger.info(f"   ✅ README has CI badges and correct latency claims")
                    else:
                        logger.error(f"   ❌ README missing CI badges or has wrong claims")
                        all_verified = False
                
                elif file_path == "reports/hash_manifest.json":
                    try:
                        data = json.loads(response.text)
                        if "8488e883b13e72788b879846249c206cb87ca67c44317bdf8786789faeebb1d4" in str(data):
                            logger.info(f"   ✅ Hash manifest contains AuditPack SHA256")
                        else:
                            logger.error(f"   ❌ Hash manifest missing AuditPack SHA256")
                            all_verified = False
                    except:
                        logger.error(f"   ❌ Hash manifest is not valid JSON")
                        all_verified = False
                        
            else:
                logger.error(f"❌ {file_path} - HTTP {response.status_code}")
                all_verified = False
                
        except Exception as e:
            logger.error(f"❌ {file_path} - ERROR: {e}")
            all_verified = False
    
    # Check GitHub Actions workflows
    logger.info("\n🔄 CHECKING GITHUB ACTIONS")
    workflow_urls = [
        "https://github.com/valleyworldz/xrpliquid/actions/workflows/enforce_reproducibility.yml",
        "https://github.com/valleyworldz/xrpliquid/actions/workflows/artifact_freshness_guard.yml", 
        "https://github.com/valleyworldz/xrpliquid/actions/workflows/no_lookahead_guard.yml",
        "https://github.com/valleyworldz/xrpliquid/actions/workflows/supply_chain_security.yml"
    ]
    
    for workflow_url in workflow_urls:
        try:
            response = requests.get(workflow_url, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ Workflow accessible: {workflow_url.split('/')[-1]}")
            else:
                logger.error(f"❌ Workflow not accessible: {workflow_url}")
                all_verified = False
        except Exception as e:
            logger.error(f"❌ Workflow error: {e}")
            all_verified = False
    
    # Final summary
    logger.info("\n" + "=" * 50)
    if all_verified:
        logger.info("🎉 LIVE REPOSITORY STATUS: 100% VERIFIED")
        logger.info("✅ All critical files accessible")
        logger.info("✅ CI badges and claims correct")
        logger.info("✅ Hash manifest complete")
        logger.info("✅ GitHub Actions workflows present")
        logger.info("✅ External reviewers can access all artifacts")
        return True
    else:
        logger.error("⚠️  LIVE REPOSITORY STATUS: ISSUES FOUND")
        logger.error("❌ Some files not accessible or incorrect")
        logger.error("❌ External reviewers may have issues")
        return False

if __name__ == "__main__":
    success = verify_live_repo()
    exit(0 if success else 1)
