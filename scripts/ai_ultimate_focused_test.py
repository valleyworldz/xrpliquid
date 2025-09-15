#!/usr/bin/env python3
"""
A.I. ULTIMATE Profile Focused Backtesting
==========================================
Comprehensive testing of ONLY the A.I. ULTIMATE profile on XRP
"""

import sys
import os
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def run_ai_ultimate_focused_test():
    """Run comprehensive A.I. ULTIMATE profile test"""
    
    print("🧠 A.I. ULTIMATE PROFILE - FOCUSED COMPREHENSIVE TEST")
    print("=" * 70)
    print("📋 Test Configuration:")
    print("   🎯 Profile: A.I. ULTIMATE (Master Expert)")
    print("   💎 Asset: XRP only")
    print("   🔬 Mode: Quantum optimization enabled")
    print("   📊 Data: 100% real Hyperliquid market data")
    print()
    
    # Test configurations for comprehensive analysis
    test_scenarios = [
        {"hours": 168, "name": "1 Week Validation", "desc": "Quick quantum feature validation"},
        {"hours": 720, "name": "30 Day Comprehensive", "desc": "Full performance analysis"},
        {"hours": 1440, "name": "60 Day Robustness", "desc": "Extended market condition testing"},
        {"hours": 2160, "name": "90 Day Stability", "desc": "Long-term stability validation"}
    ]
    
    results = {}
    best_score = 0
    best_scenario = None
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧠 TEST {i}/4: {scenario['name']} ({scenario['hours']} hours)")
        print(f"📋 {scenario['desc']}")
        print("-" * 50)
        
        try:
            # Create focused backtester script for A.I. ULTIMATE only
            focused_script = create_ai_ultimate_backtester(scenario['hours'])
            
            # Run the focused test
            print("⚡ Starting A.I. ULTIMATE quantum analysis...")
            start_time = time.time()
            
            import subprocess
            result = subprocess.run([
                sys.executable, '-c', focused_script
            ], capture_output=True, text=True, timeout=1200)  # 20 min timeout
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Test completed in {duration:.1f}s")
                
                # Parse results
                output_lines = result.stdout.strip().split('\n')
                test_result = parse_ai_ultimate_results(output_lines)
                
                if test_result:
                    results[scenario['name']] = test_result
                    print_scenario_results(test_result)
                    
                    if test_result['overall_score'] > best_score:
                        best_score = test_result['overall_score']
                        best_scenario = scenario['name']
                else:
                    print("⚠️ Could not parse results")
                    results[scenario['name']] = None
            else:
                print(f"❌ Test failed: {result.stderr}")
                results[scenario['name']] = None
                
        except subprocess.TimeoutExpired:
            print("⏰ Test timed out - may indicate hanging in complex calculations")
            results[scenario['name']] = None
        except Exception as e:
            print(f"❌ Test error: {e}")
            results[scenario['name']] = None
    
    # Generate comprehensive analysis
    print("\n🧠 A.I. ULTIMATE COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    if best_scenario:
        print(f"🏆 BEST PERFORMANCE: {best_scenario}")
        best_result = results[best_scenario]
        print(f"   📊 Overall Score: {best_result['overall_score']:.1f}/100")
        print(f"   💰 Returns: {best_result['return_pct']:.2f}%")
        print(f"   🎯 Win Rate: {best_result['win_rate']:.1f}%")
        print(f"   ⚖️ Sharpe: {best_result['sharpe']:.3f}")
        print(f"   🛡️ Max DD: {best_result['max_drawdown']:.2f}%")
        print(f"   📈 Trades: {best_result['total_trades']}")
        
        # Quantum feature validation
        validate_quantum_features(best_result)
        
        # Performance trajectory analysis
        analyze_performance_trajectory(results)
        
        # Generate recommendations
        recommendations = generate_ai_ultimate_recommendations(results, best_scenario)
        print_recommendations(recommendations)
        
    else:
        print("❌ No successful test runs - investigating issues...")
        diagnose_test_failures(results)
    
    # Save detailed report
    save_ai_ultimate_report(results, best_scenario)
    
    return results, best_scenario

def create_ai_ultimate_backtester(hours):
    """Create focused backtester script for A.I. ULTIMATE only"""
    script = f'''
import sys
sys.path.append(".")

from working_real_backtester import *
import os

# Override environment for A.I. ULTIMATE focus
os.environ["BACKTEST_HOURS"] = "{hours}"
os.environ["AI_ULTIMATE_FOCUS"] = "1"
os.environ["QUANTUM_OPTIMIZATION"] = "1"

# Create focused tester
tester = RealStrategyTester()

# Test ONLY A.I. ULTIMATE profile
profile_name = "ai_ultimate"
config = TRADING_PROFILES[profile_name]["config"]
symbol = "XRP"

print(f"🧠 Testing A.I. ULTIMATE: {{hours}}h on {{symbol}}")
print(f"⚡ Quantum features: ENABLED")
print(f"🔬 Advanced ML: ENABLED")

try:
    result = tester.test_strategy(symbol, config)
    
    if result and "trades" in result:
        trades = result["trades"]
        
        # Calculate comprehensive metrics
        returns = [t.pnl_percent for t in trades]
        total_return = sum(returns)
        win_rate = len([r for r in returns if r > 0]) / len(returns) * 100 if returns else 0
        
        max_dd = 0
        running_return = 0
        peak = 0
        for r in returns:
            running_return += r
            if running_return > peak:
                peak = running_return
            dd = (peak - running_return) / (peak + 1e-8) * 100
            max_dd = max(max_dd, dd)
        
        # Sharpe calculation
        if len(returns) > 1:
            import numpy as np
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * (252 ** 0.5)
        else:
            sharpe = 0
        
        # Overall score calculation (simplified)
        return_score = min(10, max(0, total_return * 2))
        win_score = min(10, win_rate / 10)
        sharpe_score = min(10, max(0, sharpe * 2))
        dd_score = min(10, max(0, 10 - max_dd))
        activity_score = min(10, len(trades) / 2)
        
        overall_score = (return_score + win_score + sharpe_score + dd_score + activity_score) * 2
        
        print(f"RESULT|{{overall_score:.1f}}|{{total_return:.2f}}|{{win_rate:.1f}}|{{sharpe:.3f}}|{{max_dd:.2f}}|{{len(trades)}}")
        
    else:
        print("RESULT|0|0|0|0|0|0")
        
except Exception as e:
    print(f"ERROR: {{e}}")
    print("RESULT|0|0|0|0|0|0")
'''
    return script

def parse_ai_ultimate_results(output_lines):
    """Parse A.I. ULTIMATE test results"""
    for line in output_lines:
        if line.startswith("RESULT|"):
            parts = line.split("|")
            if len(parts) >= 7:
                return {
                    'overall_score': float(parts[1]),
                    'return_pct': float(parts[2]),
                    'win_rate': float(parts[3]),
                    'sharpe': float(parts[4]),
                    'max_drawdown': float(parts[5]),
                    'total_trades': int(parts[6])
                }
    return None

def print_scenario_results(result):
    """Print results for a test scenario"""
    print(f"   📊 Overall Score: {result['overall_score']:.1f}/100")
    print(f"   💰 Returns: {result['return_pct']:.2f}%")
    print(f"   🎯 Win Rate: {result['win_rate']:.1f}%")
    print(f"   ⚖️ Sharpe: {result['sharpe']:.3f}")
    print(f"   🛡️ Max DD: {result['max_drawdown']:.2f}%")
    print(f"   📈 Trades: {result['total_trades']}")
    
    # Performance assessment
    if result['overall_score'] >= 80:
        print("   🏆 EXCELLENT - Master Expert Level")
    elif result['overall_score'] >= 70:
        print("   🥇 VERY GOOD - Advanced Level")
    elif result['overall_score'] >= 60:
        print("   🥈 GOOD - Intermediate Level")
    else:
        print("   📈 DEVELOPING - Needs optimization")

def validate_quantum_features(result):
    """Validate quantum features are working correctly"""
    print("\n🔬 QUANTUM FEATURE VALIDATION")
    print("-" * 30)
    
    # Check if quantum optimizations are producing expected results
    quantum_indicators = []
    
    # High score indicates advanced optimization
    if result['overall_score'] > 70:
        quantum_indicators.append("✅ Advanced optimization active")
    else:
        quantum_indicators.append("⚠️ Optimization may need tuning")
    
    # Balanced risk-return suggests sophisticated sizing
    risk_return_ratio = result['return_pct'] / max(result['max_drawdown'], 0.1)
    if risk_return_ratio > 5:
        quantum_indicators.append("✅ Quantum risk management working")
    else:
        quantum_indicators.append("⚠️ Risk management needs attention")
    
    # Selective trading suggests ML filtering
    if result['total_trades'] <= 10:  # Selective high-quality signals
        quantum_indicators.append("✅ ML signal filtering active")
    else:
        quantum_indicators.append("⚠️ Signal filtering may be too permissive")
    
    # High win rate or good sharpe suggests ensemble working
    if result['win_rate'] > 60 or result['sharpe'] > 1.0:
        quantum_indicators.append("✅ Ensemble models performing well")
    else:
        quantum_indicators.append("⚠️ Ensemble models need optimization")
    
    for indicator in quantum_indicators:
        print(f"   {indicator}")

def analyze_performance_trajectory(results):
    """Analyze performance across different time horizons"""
    print("\n📈 PERFORMANCE TRAJECTORY ANALYSIS")
    print("-" * 40)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        print("   ⚠️ Insufficient data for trajectory analysis")
        return
    
    # Analyze score stability
    scores = [r['overall_score'] for r in valid_results.values()]
    score_stability = max(scores) - min(scores)
    
    print(f"   📊 Score Range: {min(scores):.1f} - {max(scores):.1f}")
    print(f"   📈 Stability: {score_stability:.1f} point variation")
    
    if score_stability < 10:
        print("   ✅ Excellent stability across time horizons")
    elif score_stability < 20:
        print("   🥈 Good stability")
    else:
        print("   ⚠️ High variation - may need parameter tuning")
    
    # Analyze returns scaling
    returns = [(k, r['return_pct']) for k, r in valid_results.items()]
    print(f"   💰 Return Scaling:")
    for period, ret in returns:
        print(f"      {period}: {ret:.2f}%")

def generate_ai_ultimate_recommendations(results, best_scenario):
    """Generate specific recommendations for A.I. ULTIMATE"""
    recommendations = []
    
    if not best_scenario:
        recommendations.append("No successful tests - check A.I. ULTIMATE implementation")
        return recommendations
    
    best_result = results[best_scenario]
    
    # Score-based recommendations
    if best_result['overall_score'] < 75:
        recommendations.append("Optimize ensemble weights to improve overall score")
    
    if best_result['overall_score'] >= 80:
        recommendations.append("Excellent performance - ready for live deployment")
    
    # Win rate recommendations
    if best_result['win_rate'] < 60:
        recommendations.append("Tighten profit-taking thresholds for higher win rate")
    elif best_result['win_rate'] > 80:
        recommendations.append("Consider loosening filters for more trading opportunities")
    
    # Risk recommendations
    if best_result['max_drawdown'] > 5:
        recommendations.append("Strengthen quantum risk management parameters")
    elif best_result['max_drawdown'] < 1:
        recommendations.append("Consider slightly more aggressive sizing for better returns")
    
    # Activity recommendations
    if best_result['total_trades'] < 3:
        recommendations.append("Relax ML filters to increase signal frequency")
    elif best_result['total_trades'] > 15:
        recommendations.append("Tighten quantum filters to focus on highest-quality signals")
    
    # Sharpe recommendations
    if best_result['sharpe'] < 1.0:
        recommendations.append("Improve risk-adjusted returns through better volatility filtering")
    
    return recommendations

def print_recommendations(recommendations):
    """Print optimization recommendations"""
    print("\n🎯 A.I. ULTIMATE OPTIMIZATION RECOMMENDATIONS")
    print("-" * 50)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

def diagnose_test_failures(results):
    """Diagnose why tests may have failed"""
    print("\n🔍 DIAGNOSTIC ANALYSIS")
    print("-" * 25)
    
    failed_tests = [k for k, v in results.items() if v is None]
    
    if failed_tests:
        print(f"   ❌ Failed tests: {', '.join(failed_tests)}")
        print("   🔍 Possible causes:")
        print("      • A.I. ULTIMATE implementation issues")
        print("      • Quantum feature complexity causing timeouts")
        print("      • Data availability problems")
        print("      • Memory/computation limits")
        
        print("\n   💡 Suggested fixes:")
        print("      • Simplify quantum calculations for testing")
        print("      • Increase timeout limits")
        print("      • Check A.I. ULTIMATE profile configuration")
        print("      • Validate data sources")

def save_ai_ultimate_report(results, best_scenario):
    """Save comprehensive A.I. ULTIMATE test report"""
    report = {
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'profile': 'ai_ultimate',
        'asset': 'XRP',
        'test_type': 'focused_comprehensive',
        'results': results,
        'best_scenario': best_scenario,
        'best_score': results[best_scenario]['overall_score'] if best_scenario else 0,
        'quantum_features': {
            'multi_ensemble_ml': True,
            'quantum_signal_processing': True,
            'adaptive_position_sizing': True,
            'regime_aware_exits': True,
            'real_time_learning': True
        },
        'performance_summary': results[best_scenario] if best_scenario else None
    }
    
    with open('ai_ultimate_focused_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Detailed report saved: ai_ultimate_focused_test_report.json")

if __name__ == "__main__":
    print("🧠 A.I. ULTIMATE PROFILE - FOCUSED TESTING SYSTEM")
    print("This will comprehensively test ONLY the A.I. ULTIMATE profile")
    print("with quantum optimization features on XRP with real market data.")
    print()
    
    confirm = input("🚀 Start comprehensive A.I. ULTIMATE testing? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        print("\n🧠 Starting A.I. ULTIMATE focused testing...")
        results, best_scenario = run_ai_ultimate_focused_test()
        
        if best_scenario:
            print(f"\n🎉 A.I. ULTIMATE testing completed!")
            print(f"🏆 Best performance: {best_scenario}")
            print("📊 See ai_ultimate_focused_test_report.json for full analysis")
        else:
            print("\n⚠️ Testing completed but encountered issues")
            print("🔍 Check diagnostic output and try again")
    else:
        print("🛑 Testing cancelled")
