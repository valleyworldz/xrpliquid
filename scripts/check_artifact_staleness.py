"""
Artifact Staleness Check
Ensures all artifacts are fresh and up-to-date with code/config changes.
"""

import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys


def get_git_commit_timestamp():
    """Get the timestamp of the latest commit."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError:
        return 0


def get_file_timestamp(file_path):
    """Get the modification timestamp of a file."""
    try:
        return int(os.path.getmtime(file_path))
    except OSError:
        return 0


def check_artifact_staleness():
    """Check if artifacts are stale compared to code/config changes."""
    
    print("🔍 Checking artifact staleness...")
    
    # Get latest commit timestamp
    commit_timestamp = get_git_commit_timestamp()
    commit_time = datetime.fromtimestamp(commit_timestamp, tz=timezone.utc)
    
    print(f"📅 Latest commit: {commit_time.isoformat()}")
    
    # Define critical artifacts that must be fresh
    critical_artifacts = [
        "reports/tearsheets/comprehensive_tearsheet.html",
        "reports/ledgers/trades.parquet",
        "reports/ledgers/trades.csv",
        "reports/backtest_results.json",
        "reports/executive_dashboard.html",
        "reports/hash_manifest.json",
        "reports/latency/latency_analysis.json",
        "reports/risk_events/risk_events.json",
        "reports/regime/regime_analysis.json",
        "reports/maker_taker/routing_analysis.json",
        "reports/microstructure/impact_residuals.json",
        "reports/microstructure/opportunity_cost.json",
        "reports/risk/var_es.json",
        "reports/reconciliation/exchange_vs_ledger.json"
    ]
    
    stale_artifacts = []
    
    for artifact_path in critical_artifacts:
        if not os.path.exists(artifact_path):
            print(f"❌ Missing artifact: {artifact_path}")
            stale_artifacts.append(artifact_path)
            continue
        
        artifact_timestamp = get_file_timestamp(artifact_path)
        artifact_time = datetime.fromtimestamp(artifact_timestamp, tz=timezone.utc)
        
        # Check if artifact is older than latest commit
        if artifact_timestamp < commit_timestamp:
            print(f"❌ Stale artifact: {artifact_path}")
            print(f"   Artifact time: {artifact_time.isoformat()}")
            print(f"   Commit time:   {commit_time.isoformat()}")
            stale_artifacts.append(artifact_path)
        else:
            print(f"✅ Fresh artifact: {artifact_path}")
    
    # Check for config changes that should trigger regeneration
    config_files = [
        "config/",
        "requirements.txt",
        ".pre-commit-config.yaml",
        "src/core/config/"
    ]
    
    config_changed = False
    for config_path in config_files:
        if os.path.exists(config_path):
            if os.path.isdir(config_path):
                # Check if any file in directory is newer than artifacts
                for root, dirs, files in os.walk(config_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if get_file_timestamp(file_path) > commit_timestamp:
                            config_changed = True
                            break
            else:
                if get_file_timestamp(config_path) > commit_timestamp:
                    config_changed = True
    
    if config_changed:
        print("⚠️  Config files changed - artifacts may need regeneration")
    
    # Summary
    if stale_artifacts:
        print(f"\n❌ STALENESS CHECK FAILED")
        print(f"📊 {len(stale_artifacts)} stale artifacts found:")
        for artifact in stale_artifacts:
            print(f"   - {artifact}")
        print(f"\n🔧 Run 'python scripts/regenerate_artifacts.py' to fix")
        return False
    else:
        print(f"\n✅ STALENESS CHECK PASSED")
        print(f"📊 All {len(critical_artifacts)} artifacts are fresh")
        return True


def main():
    """Main function."""
    success = check_artifact_staleness()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
