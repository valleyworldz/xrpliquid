#!/usr/bin/env python3
"""
Create Initial GitHub Issues for Public Trace
============================================

This script creates initial GitHub issues to establish public trace
of QA tasks, risk sign-offs, and pending features for auditability.

Usage:
    python scripts/create_initial_issues.py [--dry-run]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def create_issue_template(title: str, body: str, labels: List[str]) -> Dict[str, Any]:
    """Create a GitHub issue template"""
    return {
        "title": title,
        "body": body,
        "labels": labels
    }

def get_crown_tier_verification_issues() -> List[Dict[str, Any]]:
    """Get crown-tier verification issues"""
    return [
        create_issue_template(
            "[CROWN-TIER] Verify Sharpe Ratio Consistency",
            """## 🏆 Crown-Tier Verification: Sharpe Ratio Consistency

### 📋 Task Description
Verify that all Sharpe ratio references in documentation are consistent and accurate.

### 🎯 Verification Criteria
- [ ] README.md Sharpe ratio matches tearsheet_latest.json
- [ ] All performance claims are consistent across documentation
- [ ] Proof artifacts reflect actual backtest results

### 📊 Current Status
- README.md: Sharpe 2.1 ✅
- tearsheet_latest.json: Sharpe 2.1 ✅
- Performance Metrics section: Sharpe 2.1 ✅

### ✅ Acceptance Criteria
- [ ] All Sharpe ratio references are consistent
- [ ] Documentation updated to reflect accurate metrics
- [ ] Proof artifacts regenerated and verified

### 🔗 Related
- Fixes documentation inconsistency
- Ensures crown-tier proof credibility""",
            ["crown-tier", "verification", "documentation", "performance"]
        ),
        
        create_issue_template(
            "[CROWN-TIER] Regenerate All Proof Artifacts",
            """## 🏆 Crown-Tier Verification: Proof Artifacts Regeneration

### 📋 Task Description
Regenerate all proof artifacts from raw inputs to ensure reproducibility and transparency.

### 🎯 Verification Criteria
- [ ] Tearsheet JSON with complete performance metrics
- [ ] Latency analysis with histogram data
- [ ] Portfolio risk analysis (VaR/ES)
- [ ] Correlation heatmaps
- [ ] Capacity analysis
- [ ] Stress testing results
- [ ] Execution slippage analysis
- [ ] ML drift monitoring
- [ ] Security penetration test results

### 🔍 Verification Steps
1. [ ] Run `python scripts/regenerate_proof_artifacts.py`
2. [ ] Verify all artifacts are generated correctly
3. [ ] Check artifact consistency and accuracy
4. [ ] Update CI/CD to auto-regenerate on tags

### ✅ Acceptance Criteria
- [ ] All proof artifacts regenerated successfully
- [ ] Artifacts are consistent and accurate
- [ ] CI/CD pipeline updated for automatic regeneration
- [ ] Documentation links to verified artifacts

### 🔗 Related
- Ensures artifact reproducibility
- Maintains crown-tier proof transparency""",
            ["crown-tier", "verification", "artifacts", "automation"]
        ),
        
        create_issue_template(
            "[CROWN-TIER] Validate Risk Management Controls",
            """## 🏆 Crown-Tier Verification: Risk Management Controls

### 📋 Task Description
Validate all risk management controls and circuit breakers are functioning correctly.

### 🎯 Verification Criteria
- [ ] Kill-switch logic prevents false triggers
- [ ] Decimal/float type mixing resolved
- [ ] Market order price handling correct
- [ ] RR/ATR calculations accurate
- [ ] Fee defaults prevent divide-by-zero

### 🔍 Verification Steps
1. [ ] Test kill-switch with profitable positions (should not trigger)
2. [ ] Test kill-switch with losing positions (should trigger)
3. [ ] Verify market orders don't show $0.00 values
4. [ ] Check RR/ATR calculations for accuracy
5. [ ] Validate fee calculations with non-zero defaults

### ✅ Acceptance Criteria
- [ ] All risk controls function correctly
- [ ] No false positive kill-switch triggers
- [ ] Market orders execute properly
- [ ] Financial calculations are accurate
- [ ] System fails closed on invalid inputs

### 🔗 Related
- Critical for trading system stability
- Ensures proper risk management""",
            ["crown-tier", "verification", "risk-management", "critical"]
        )
    ]

def get_security_audit_issues() -> List[Dict[str, Any]]:
    """Get security audit issues"""
    return [
        create_issue_template(
            "[SECURITY] Implement Fail-Closed Secrets Loader",
            """## 🔐 Security Audit: Fail-Closed Secrets Loader

### 📋 Task Description
Implement and validate fail-closed secrets loader to ensure system cannot operate with missing or invalid credentials.

### 🎯 Security Areas to Audit
- [ ] Credential validation and format checking
- [ ] Fail-closed behavior on missing credentials
- [ ] Audit logging for all credential access
- [ ] Environment variable security
- [ ] File-based credential security

### 🔍 Audit Checklist
- [ ] Secrets loader fails hard on missing required credentials
- [ ] Credential format validation prevents invalid inputs
- [ ] All credential access is logged for audit
- [ ] Environment variables are properly sanitized
- [ ] Credential files are properly protected

### 🛡️ Security Controls to Verify
- [ ] **Authentication**: Private key validation
- [ ] **Authorization**: Address format validation
- [ ] **Encryption**: Secure credential storage
- [ ] **Monitoring**: Audit trail for credential access
- [ ] **Incident Response**: Fail-closed behavior

### ✅ Acceptance Criteria
- [ ] System exits immediately on missing credentials
- [ ] Invalid credential formats are rejected
- [ ] Complete audit trail of credential access
- [ ] No credentials logged in plain text
- [ ] Fail-closed behavior validated

### 🔗 Related
- Critical for credential security
- Ensures system security posture""",
            ["security", "audit", "credentials", "critical"]
        ),
        
        create_issue_template(
            "[SECURITY] Validate .gitignore for Credentials",
            """## 🔐 Security Audit: Credential File Protection

### 📋 Task Description
Validate that all credential files are properly excluded from version control.

### 🎯 Security Areas to Audit
- [ ] .gitignore excludes config/secure_creds.env
- [ ] No credential files in repository history
- [ ] CI/CD prevents credential commits
- [ ] Documentation warns about credential security

### 🔍 Audit Checklist
- [ ] .gitignore contains config/secure_creds.env
- [ ] No credential files in git history
- [ ] CI/CD scans for credential leaks
- [ ] Documentation includes security warnings
- [ ] Example files use placeholder values

### 🛡️ Security Controls to Verify
- [ ] **Prevention**: .gitignore protection
- [ ] **Detection**: CI/CD credential scanning
- [ ] **Response**: Automated credential leak detection
- [ ] **Education**: Security documentation

### ✅ Acceptance Criteria
- [ ] All credential files excluded from version control
- [ ] No credential leaks in repository
- [ ] CI/CD prevents future credential commits
- [ ] Security documentation is comprehensive
- [ ] Example files are safe to commit

### 🔗 Related
- Prevents credential leaks
- Ensures repository security""",
            ["security", "audit", "gitignore", "credentials"]
        )
    ]

def get_qa_issues() -> List[Dict[str, Any]]:
    """Get QA and testing issues"""
    return [
        create_issue_template(
            "[QA] Comprehensive Decimal/Float Type Testing",
            """## 🧪 QA: Decimal/Float Type System Testing

### 📋 Task Description
Comprehensive testing of the decimal/float type system to ensure no type mixing errors.

### 🎯 Testing Areas
- [ ] RR/ATR calculations with various input types
- [ ] TP/SL adjustments with Decimal precision
- [ ] Fee calculations with non-zero defaults
- [ ] Regime reconfiguration type handling
- [ ] Price history decimal/float boundaries

### 🔍 Test Cases
1. [ ] Test RR/ATR with Decimal prices and float ATR
2. [ ] Test TP/SL with mixed Decimal/float inputs
3. [ ] Test fee calculations with zero and non-zero fees
4. [ ] Test regime reconfiguration with various types
5. [ ] Test price history with float arrays and Decimal calculations

### ✅ Acceptance Criteria
- [ ] No unsupported operand type errors
- [ ] All financial calculations use proper precision
- [ ] Type boundaries are clearly defined
- [ ] Error handling for type mismatches
- [ ] Performance impact is minimal

### 🔗 Related
- Critical for financial calculation accuracy
- Prevents runtime type errors""",
            ["qa", "testing", "decimal", "type-safety"]
        ),
        
        create_issue_template(
            "[QA] End-to-End Trading System Testing",
            """## 🧪 QA: End-to-End Trading System Testing

### 📋 Task Description
Comprehensive end-to-end testing of the trading system with real market conditions.

### 🎯 Testing Areas
- [ ] Live trading with real Hyperliquid API
- [ ] Order placement and execution
- [ ] Position management and tracking
- [ ] Risk management and circuit breakers
- [ ] Performance monitoring and logging

### 🔍 Test Scenarios
1. [ ] Test with small position sizes on testnet
2. [ ] Verify order execution and fills
3. [ ] Test risk management triggers
4. [ ] Validate performance metrics collection
5. [ ] Test system recovery from errors

### ✅ Acceptance Criteria
- [ ] System executes trades successfully
- [ ] Risk management functions correctly
- [ ] Performance metrics are accurate
- [ ] Error handling is robust
- [ ] System recovers from failures

### 🔗 Related
- Validates complete trading system
- Ensures production readiness""",
            ["qa", "testing", "end-to-end", "trading"]
        )
    ]

def get_performance_issues() -> List[Dict[str, Any]]:
    """Get performance optimization issues"""
    return [
        create_issue_template(
            "[PERFORMANCE] Latency Optimization Review",
            """## ⚡ Performance: Latency Optimization Review

### 📋 Task Description
Review and optimize system latency to maintain sub-100ms trading cycles.

### 🎯 Performance Areas
- [ ] API call optimization
- [ ] WebSocket connection efficiency
- [ ] Data processing speed
- [ ] Memory usage optimization
- [ ] CPU utilization efficiency

### 🔍 Performance Metrics
- [ ] P50 latency: Target < 50ms
- [ ] P95 latency: Target < 100ms
- [ ] P99 latency: Target < 200ms
- [ ] Memory usage: Monitor for leaks
- [ ] CPU usage: Optimize for efficiency

### ✅ Acceptance Criteria
- [ ] All latency targets met
- [ ] No memory leaks detected
- [ ] CPU usage optimized
- [ ] Performance monitoring active
- [ ] Optimization documented

### 🔗 Related
- Critical for trading performance
- Ensures competitive advantage""",
            ["performance", "optimization", "latency", "trading"]
        )
    ]

def main():
    """Main function to create initial GitHub issues"""
    parser = argparse.ArgumentParser(description='Create initial GitHub issues for public trace')
    parser.add_argument('--dry-run', action='store_true', help='Show issues that would be created without creating them')
    args = parser.parse_args()
    
    print("🏆 Creating initial GitHub issues for public trace...")
    
    all_issues = []
    all_issues.extend(get_crown_tier_verification_issues())
    all_issues.extend(get_security_audit_issues())
    all_issues.extend(get_qa_issues())
    all_issues.extend(get_performance_issues())
    
    print(f"📋 Generated {len(all_issues)} issues:")
    
    for i, issue in enumerate(all_issues, 1):
        print(f"\n{i}. {issue['title']}")
        print(f"   Labels: {', '.join(issue['labels'])}")
        if args.dry_run:
            print(f"   Body preview: {issue['body'][:100]}...")
    
    if args.dry_run:
        print(f"\n🔍 Dry run complete. {len(all_issues)} issues would be created.")
        print("Run without --dry-run to create the issues.")
        return 0
    
    # In a real implementation, you would use the GitHub API to create these issues
    # For now, we'll create a JSON file that can be used to create issues
    issues_file = Path("scripts/generated_issues.json")
    with open(issues_file, 'w') as f:
        json.dump(all_issues, f, indent=2)
    
    print(f"\n✅ Issues saved to {issues_file}")
    print("📝 To create these issues on GitHub:")
    print("1. Use the GitHub web interface")
    print("2. Use the GitHub CLI: gh issue create --title '...' --body '...' --label '...'")
    print("3. Use the GitHub API with the JSON file")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
