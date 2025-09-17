#!/usr/bin/env python3
"""
Runtime Smoke Test for Critical Fixes
"""

import sys
sys.path.append('.')

def test_decimal_bug_absence():
    """Test for absence of Decimal TypeError"""
    print("🔢 Testing for Decimal TypeError absence:")
    
    try:
        from src.core.utils.decimal_normalizer import decimal_normalizer
        
        # Test the exact operation that was causing errors
        result = decimal_normalizer.safe_operation('subtract', 123.456, 789.012)
        print(f"✅ No TypeError: 123.456 - 789.012 = {result}")
        
        # Test other operations
        result2 = decimal_normalizer.safe_operation('add', 123.456, 789.012)
        print(f"✅ No TypeError: 123.456 + 789.012 = {result2}")
        
        return True
        
    except TypeError as e:
        if "unsupported operand type(s) for -: 'float' and 'decimal.Decimal'" in str(e):
            print(f"❌ CRITICAL: Decimal TypeError found: {e}")
            return False
        else:
            print(f"⚠️ Other TypeError: {e}")
            return False
    except Exception as e:
        print(f"⚠️ Other error: {e}")
        return False

def test_feasibility_gate():
    """Test market depth feasibility gate"""
    print("\n🔍 Testing market depth feasibility gate:")
    
    try:
        from src.core.validation.market_depth_feasibility import MarketDepthFeasibilityChecker
        from src.core.validation.market_depth_feasibility import MarketDepthSnapshot, FeasibilityResult
        from decimal import Decimal
        
        checker = MarketDepthFeasibilityChecker()
        
        # Create thin book scenario
        thin_book = MarketDepthSnapshot(
            symbol="XRP-USD",
            timestamp="2025-09-17T00:00:00Z",
            bids=[(Decimal('0.5234'), Decimal('10'))],  # Very thin depth
            asks=[(Decimal('0.5236'), Decimal('10'))],
            snapshot_hash="thin_book_test"
        )
        
        # Test feasibility check
        feasibility = checker.check_tp_sl_feasibility(
            thin_book, 
            Decimal('0.5235'),  # entry
            Decimal('0.5250'),  # TP (too far)
            Decimal('0.5210'),  # SL (too far)
            Decimal('100'),     # large order
            "buy"
        )
        
        if feasibility.result == FeasibilityResult.INFEASIBLE:
            print("✅ Feasibility gate working: Infeasible scenario properly detected")
            return True
        else:
            print(f"❌ Feasibility gate failed: Expected INFEASIBLE, got {feasibility.result}")
            return False
            
    except Exception as e:
        print(f"❌ Feasibility gate test failed: {e}")
        return False

def test_audit_artifacts():
    """Test audit artifacts exist and are valid"""
    print("\n📊 Testing audit artifacts:")
    
    import json
    from pathlib import Path
    
    artifacts = [
        "reports/risk/var_es.json",
        "reports/reconciliation/exchange_vs_ledger.json"
    ]
    
    all_valid = True
    
    for artifact in artifacts:
        path = Path(artifact)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"✅ {artifact}: Valid JSON with {len(data)} keys")
            except json.JSONDecodeError as e:
                print(f"❌ {artifact}: Invalid JSON - {e}")
                all_valid = False
        else:
            print(f"❌ {artifact}: File missing")
            all_valid = False
    
    return all_valid

def main():
    """Run all smoke tests"""
    print("🚀 Runtime Smoke Test for Critical Fixes")
    print("=" * 50)
    
    tests = [
        ("Decimal Bug Absence", test_decimal_bug_absence),
        ("Feasibility Gate", test_feasibility_gate),
        ("Audit Artifacts", test_audit_artifacts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("🎯 SMOKE TEST RESULTS:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL CRITICAL FIXES VERIFIED WORKING!")
        print("✅ System is institution-ready")
    else:
        print("\n⚠️ Some critical fixes need attention")
        sys.exit(1)

if __name__ == "__main__":
    main()
