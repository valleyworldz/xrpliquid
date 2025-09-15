#!/usr/bin/env python3
"""
A.I. ULTIMATE Profile Testing and Optimization
==============================================
Test the new A.I. ULTIMATE profile with quantum-enhanced features
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_ai_ultimate_profile():
    """Test the A.I. ULTIMATE profile"""
    
    print("🧠 TESTING A.I. ULTIMATE PROFILE")
    print("=" * 60)
    
    # Test different time horizons
    test_configs = [
        {"hours": 168, "name": "1 Week Test"},  # 1 week
        {"hours": 720, "name": "30 Day Test"},  # 30 days
        {"hours": 2160, "name": "90 Day Test"}, # 90 days
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n🧠 Running {config['name']} ({config['hours']} hours)")
        print("-" * 40)
        
        # Set environment variables for A.I. ULTIMATE testing
        env = os.environ.copy()
        env['BACKTEST_HOURS'] = str(config['hours'])
        env['AI_ULTIMATE_MODE'] = '1'
        env['QUANTUM_OPTIMIZATION'] = '1'
        env['ENSEMBLE_LEARNING'] = '1'
        env['ADAPTIVE_THRESHOLDS'] = '1'
        
        try:
            # Run the backtester
            result = subprocess.run([
                sys.executable, '-X', 'utf8', 'working_real_backtester.py'
            ], env=env, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                print("✅ Backtest completed successfully")
                
                # Load results
                try:
                    with open('real_backtest_summary.json', 'r') as f:
                        summary = json.load(f)
                    
                    # Extract A.I. ULTIMATE results
                    ai_ultimate_results = summary.get('profiles', {}).get('ai_ultimate', {})
                    
                    if ai_ultimate_results:
                        meta = ai_ultimate_results.get('meta', {})
                        results[config['name']] = {
                            'overall_score': meta.get('overall_score', 0),
                            'return_pct': meta.get('return_pct', 0),
                            'win_rate_pct': meta.get('win_rate_pct', 0),
                            'sharpe': meta.get('sharpe', 0),
                            'max_dd_pct': meta.get('max_dd_pct', 0),
                            'total_trades': meta.get('total_trades', 0)
                        }
                        
                        print(f"📊 Overall Score: {meta.get('overall_score', 0):.1f}/100")
                        print(f"💰 Return: {meta.get('return_pct', 0):.2f}%")
                        print(f"🎯 Win Rate: {meta.get('win_rate_pct', 0):.1f}%")
                        print(f"⚖️ Sharpe: {meta.get('sharpe', 0):.3f}")
                        print(f"🛡️ Max DD: {meta.get('max_dd_pct', 0):.2f}%")
                        print(f"📈 Trades: {meta.get('total_trades', 0)}")
                    else:
                        print("⚠️ A.I. ULTIMATE profile not found in results")
                        results[config['name']] = None
                        
                except Exception as e:
                    print(f"❌ Error reading results: {e}")
                    results[config['name']] = None
                    
            else:
                print(f"❌ Backtest failed: {result.stderr}")
                results[config['name']] = None
                
        except subprocess.TimeoutExpired:
            print("⏰ Backtest timed out")
            results[config['name']] = None
        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            results[config['name']] = None
    
    # Generate comprehensive report
    print("\n🧠 A.I. ULTIMATE PERFORMANCE REPORT")
    print("=" * 60)
    
    best_score = 0
    best_config = None
    
    for test_name, result in results.items():
        if result:
            print(f"\n📊 {test_name}:")
            print(f"   Overall Score: {result['overall_score']:.1f}/100")
            print(f"   Return: {result['return_pct']:.2f}%")
            print(f"   Win Rate: {result['win_rate_pct']:.1f}%")
            print(f"   Sharpe: {result['sharpe']:.3f}")
            print(f"   Max DD: {result['max_dd_pct']:.2f}%")
            print(f"   Trades: {result['total_trades']}")
            
            if result['overall_score'] > best_score:
                best_score = result['overall_score']
                best_config = test_name
        else:
            print(f"\n❌ {test_name}: No valid results")
    
    if best_config:
        print(f"\n🏆 BEST PERFORMANCE: {best_config}")
        print(f"   Champion Score: {best_score:.1f}/100")
        best_result = results[best_config]
        print(f"   📈 Performance Summary:")
        print(f"      Return: {best_result['return_pct']:.2f}%")
        print(f"      Win Rate: {best_result['win_rate_pct']:.1f}%")
        print(f"      Sharpe: {best_result['sharpe']:.3f}")
        print(f"      Max DD: {best_result['max_dd_pct']:.2f}%")
        
        # Evaluate against targets
        targets = {
            'score': 95.0,
            'win_rate': 80.0,
            'sharpe': 3.0,
            'max_dd': 3.0,
            'return': 5.0  # For the respective period
        }
        
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        score_achievement = (best_score / targets['score']) * 100
        win_rate_achievement = (best_result['win_rate_pct'] / targets['win_rate']) * 100
        sharpe_achievement = (best_result['sharpe'] / targets['sharpe']) * 100 if targets['sharpe'] > 0 else 0
        dd_achievement = (targets['max_dd'] / max(best_result['max_dd_pct'], 0.1)) * 100
        
        print(f"   Score Achievement: {score_achievement:.1f}% ({best_score:.1f}/{targets['score']})")
        print(f"   Win Rate Achievement: {win_rate_achievement:.1f}% ({best_result['win_rate_pct']:.1f}%/{targets['win_rate']}%)")
        print(f"   Sharpe Achievement: {sharpe_achievement:.1f}% ({best_result['sharpe']:.3f}/{targets['sharpe']})")
        print(f"   Drawdown Control: {dd_achievement:.1f}% ({best_result['max_dd_pct']:.2f}%/{targets['max_dd']}%)")
        
        overall_achievement = (score_achievement + win_rate_achievement + sharpe_achievement + dd_achievement) / 4
        print(f"   🏆 Overall Achievement: {overall_achievement:.1f}%")
        
        if overall_achievement >= 90:
            print("   🎉 MASTER EXPERT LEVEL ACHIEVED!")
        elif overall_achievement >= 75:
            print("   🥇 ADVANCED LEVEL ACHIEVED!")
        elif overall_achievement >= 60:
            print("   🥈 INTERMEDIATE LEVEL ACHIEVED!")
        else:
            print("   📈 DEVELOPING - Continue optimization")
        
    else:
        print("\n❌ No successful test runs")
    
    # Save detailed results
    detailed_report = {
        'test_timestamp': '2025-01-26',
        'profile': 'ai_ultimate',
        'test_results': results,
        'best_configuration': best_config,
        'best_score': best_score,
        'achievement_level': overall_achievement if best_config else 0,
        'recommendations': generate_recommendations(results, best_config)
    }
    
    with open('ai_ultimate_test_report.json', 'w') as f:
        json.dump(detailed_report, f, indent=2)
    
    print(f"\n📁 Detailed report saved to: ai_ultimate_test_report.json")
    
    return results, best_config

def generate_recommendations(results, best_config):
    """Generate optimization recommendations"""
    recommendations = []
    
    if not best_config:
        recommendations.append("No successful test runs - check configuration and data availability")
        return recommendations
    
    best_result = results[best_config]
    
    # Score recommendations
    if best_result['overall_score'] < 90:
        recommendations.append("Optimize signal quality and ensemble weighting to improve overall score")
    
    # Win rate recommendations
    if best_result['win_rate_pct'] < 75:
        recommendations.append("Implement tighter profit-taking and more selective entry criteria")
    
    # Sharpe ratio recommendations
    if best_result['sharpe'] < 2.0:
        recommendations.append("Improve risk-adjusted returns through better position sizing and volatility filtering")
    
    # Drawdown recommendations
    if best_result['max_dd_pct'] > 5.0:
        recommendations.append("Strengthen risk management with dynamic stop losses and correlation limits")
    
    # Trade frequency recommendations
    if best_result['total_trades'] < 5:
        recommendations.append("Relax entry filters to increase trade frequency while maintaining quality")
    elif best_result['total_trades'] > 50:
        recommendations.append("Apply stricter filters to reduce overtrading and transaction costs")
    
    if not recommendations:
        recommendations.append("Excellent performance - consider live deployment with gradual scaling")
    
    return recommendations

if __name__ == "__main__":
    print("🧠 A.I. ULTIMATE PROFILE TESTING SYSTEM")
    print("This will test the quantum-enhanced A.I. ULTIMATE profile")
    print("across multiple time horizons with real market data.\n")
    
    confirm = input("Continue with testing? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        results, best_config = test_ai_ultimate_profile()
        
        if best_config:
            print(f"\n🎉 Testing completed successfully!")
            print(f"Best configuration: {best_config}")
            print("See ai_ultimate_test_report.json for detailed analysis")
        else:
            print("\n⚠️ Testing completed but no successful runs")
            print("Check configuration and try again")
    else:
        print("Testing cancelled")
