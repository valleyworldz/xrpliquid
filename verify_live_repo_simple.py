#!/usr/bin/env python3
"""
Simple Live Repository Verification
Run this script to verify what's actually present on the live GitHub repository.
"""

import requests
import json

def check_live_repo():
    """Check what's actually present on the live repository."""
    
    base_url = "https://raw.githubusercontent.com/valleyworldz/xrpliquid/master"
    
    print("🔍 LIVE REPOSITORY VERIFICATION")
    print("=" * 50)
    print(f"Repository: https://github.com/valleyworldz/xrpliquid")
    print()
    
    # 1. Check README for CI badges
    print("1️⃣ CI BADGES IN README")
    try:
        response = requests.get(f"{base_url}/README.md", timeout=10)
        if response.status_code == 200:
            content = response.text
            badge_count = content.count("[![")
            print(f"   Status: ✅ README accessible (HTTP 200)")
            print(f"   Badges found: {badge_count}")
            if badge_count >= 4:
                print("   ✅ CI badges present in README")
            else:
                print("   ❌ CI badges missing from README")
        else:
            print(f"   ❌ README not accessible: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Check README for problematic claims
    print("\n2️⃣ README CLAIMS CHECK")
    try:
        response = requests.get(f"{base_url}/README.md", timeout=10)
        if response.status_code == 200:
            content = response.text.lower()
            problematic = ["sub-millisecond", "0.5-second"]
            found_problematic = [claim for claim in problematic if claim in content]
            
            if not found_problematic:
                print("   ✅ No problematic claims found")
            else:
                print(f"   ❌ Problematic claims found: {found_problematic}")
        else:
            print(f"   ❌ README not accessible: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Check PROOF_ARTIFACTS_SUMMARY.md
    print("\n3️⃣ PROOF_ARTIFACTS_SUMMARY.md")
    try:
        response = requests.get(f"{base_url}/PROOF_ARTIFACTS_SUMMARY.md", timeout=10)
        if response.status_code == 200:
            print("   ✅ PROOF_ARTIFACTS_SUMMARY.md accessible")
            print(f"   Size: {len(response.text)} characters")
        else:
            print(f"   ❌ PROOF_ARTIFACTS_SUMMARY.md not accessible: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. Check hash manifest for AuditPack hash
    print("\n4️⃣ HASH MANIFEST WITH AUDITPACK SHA256")
    try:
        response = requests.get(f"{base_url}/reports/hash_manifest.json", timeout=10)
        if response.status_code == 200:
            data = json.loads(response.text)
            auditpack_hash = "8488e883b13e72788b879846249c206cb87ca67c44317bdf8786789faeebb1d4"
            
            if auditpack_hash in json.dumps(data):
                print("   ✅ AuditPack SHA256 present in hash manifest")
            else:
                print("   ❌ AuditPack SHA256 missing from hash manifest")
        else:
            print(f"   ❌ Hash manifest not accessible: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Check critical artifacts
    print("\n5️⃣ CRITICAL ARTIFACTS")
    artifacts = [
        "reports/executive_dashboard.html",
        "reports/risk/var_es.json",
        "reports/reconciliation/exchange_vs_ledger.json",
        "reports/audit/xrpliquid_audit_pack_20250916_192751.zip"
    ]
    
    accessible_count = 0
    for artifact in artifacts:
        try:
            response = requests.get(f"{base_url}/{artifact}", timeout=10)
            if response.status_code == 200:
                accessible_count += 1
                print(f"   ✅ {artifact}")
            else:
                print(f"   ❌ {artifact}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {artifact}: Error {e}")
    
    print(f"\n   Summary: {accessible_count}/{len(artifacts)} artifacts accessible")
    
    # 6. Check GitHub Actions workflows
    print("\n6️⃣ GITHUB ACTIONS WORKFLOWS")
    workflows = [
        "enforce_reproducibility.yml",
        "artifact_freshness_guard.yml",
        "no_lookahead_guard.yml",
        "supply_chain_security.yml"
    ]
    
    accessible_workflows = 0
    for workflow in workflows:
        try:
            response = requests.get(f"{base_url}/.github/workflows/{workflow}", timeout=10)
            if response.status_code == 200:
                accessible_workflows += 1
                print(f"   ✅ {workflow}")
            else:
                print(f"   ❌ {workflow}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {workflow}: Error {e}")
    
    print(f"\n   Summary: {accessible_workflows}/{len(workflows)} workflows accessible")
    
    print("\n" + "=" * 50)
    print("🔗 DIRECT LINKS TO VERIFY:")
    print(f"README: {base_url}/README.md")
    print(f"Artifact Summary: {base_url}/PROOF_ARTIFACTS_SUMMARY.md")
    print(f"Hash Manifest: {base_url}/reports/hash_manifest.json")
    print(f"Executive Dashboard: {base_url}/reports/executive_dashboard.html")

if __name__ == "__main__":
    check_live_repo()
