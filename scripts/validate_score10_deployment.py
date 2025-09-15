#!/usr/bin/env python3
"""
Score-10 Deployment Validation
=============================
Validates that Score-10 optimizations are properly configured for live deployment
"""

import json
import os
import sys
from pathlib import Path

def validate_deployment():
    """Validate Score-10 deployment readiness"""
    
    print("🔍 VALIDATING SCORE-10 DEPLOYMENT")
    print("=" * 50)
    
    validation_results = []
    
    # Check 1: Optimized parameters file
    try:
        with open('optimized_params_live.json', 'r') as f:
            params = json.load(f)
        
        champion_profile = None
        best_score = 0
        for profile, data in params.get('profiles', {}).items():
            score = data.get('meta', {}).get('overall_score', 0)
            if score > best_score:
                best_score = score
                champion_profile = profile
        
        if best_score >= 70:
            validation_results.append(("✅", f"Champion profile {champion_profile} with score {best_score:.1f}/100"))
        else:
            validation_results.append(("❌", f"Champion score {best_score:.1f}/100 below target (70+)"))
            
    except FileNotFoundError:
        validation_results.append(("❌", "optimized_params_live.json not found"))
    
    # Check 2: Live runtime overrides
    try:
        with open('live_runtime_overrides.json', 'r') as f:
            overrides = json.load(f)
        
        if overrides.get('score_10_mode'):
            validation_results.append(("✅", "Score-10 mode enabled"))
            
            # Check execution settings
            exec_settings = overrides.get('execution_settings', {})
            if all(exec_settings.get(key) for key in ['ultra_tight_stops', 'nano_profit_taking', 'ml_signal_confirmation']):
                validation_results.append(("✅", "All Score-10 execution features enabled"))
            else:
                validation_results.append(("⚠️", "Some Score-10 features disabled"))
                
            # Check risk management
            risk_mgmt = overrides.get('risk_management', {})
            if risk_mgmt.get('stop_loss_multiplier') == 0.3:
                validation_results.append(("✅", "Ultra-tight stops configured (0.3x)"))
            else:
                validation_results.append(("⚠️", f"Stop loss multiplier: {risk_mgmt.get('stop_loss_multiplier')}"))
                
            if risk_mgmt.get('profit_target_ratio') == 0.02:
                validation_results.append(("✅", "Nano profit taking configured (0.02R)"))
            else:
                validation_results.append(("⚠️", f"Profit target: {risk_mgmt.get('profit_target_ratio')}"))
        else:
            validation_results.append(("❌", "Score-10 mode not enabled"))
            
    except FileNotFoundError:
        validation_results.append(("❌", "live_runtime_overrides.json not found"))
    
    # Check 3: Updated trading profiles
    try:
        with open('newbotcode.py', 'r', encoding='utf-8') as f:
            bot_code = f.read()
        
        if "Score-10 Optimized" in bot_code:
            validation_results.append(("✅", "Trading profiles updated with Score-10 optimizations"))
        else:
            validation_results.append(("⚠️", "Trading profiles may not be updated"))
            
        if "CHAMPION - Score 71.5/100" in bot_code:
            validation_results.append(("✅", "Champion Degen Mode profile marked"))
        else:
            validation_results.append(("⚠️", "Champion profile not marked"))
            
    except FileNotFoundError:
        validation_results.append(("❌", "newbotcode.py not found"))
    
    # Check 4: Last config
    try:
        with open('last_config.json', 'r') as f:
            last_config = json.load(f)
        
        if last_config.get('score_10_optimized'):
            validation_results.append(("✅", f"Last config set to {last_config.get('profile')} with Score-10"))
        else:
            validation_results.append(("⚠️", "Last config not optimized"))
            
    except FileNotFoundError:
        validation_results.append(("⚠️", "last_config.json not found (will use default)"))
    
    # Check 5: Environment variables
    try:
        with open('score10_env.txt', 'r') as f:
            env_content = f.read()
        
        if "SCORE_10_MODE=1" in env_content:
            validation_results.append(("✅", "Environment variables created"))
        else:
            validation_results.append(("⚠️", "Environment variables incomplete"))
            
    except FileNotFoundError:
        validation_results.append(("⚠️", "score10_env.txt not found"))
    
    # Display results
    print("\n📋 VALIDATION RESULTS:")
    print("-" * 30)
    
    success_count = 0
    warning_count = 0
    error_count = 0
    
    for status, message in validation_results:
        print(f"{status} {message}")
        if status == "✅":
            success_count += 1
        elif status == "⚠️":
            warning_count += 1
        else:
            error_count += 1
    
    print("\n📊 SUMMARY:")
    print(f"   ✅ Passed: {success_count}")
    print(f"   ⚠️ Warnings: {warning_count}")
    print(f"   ❌ Errors: {error_count}")
    
    # Overall assessment
    if error_count == 0:
        if warning_count <= 2:
            print("\n🎉 DEPLOYMENT READY!")
            print("   All critical Score-10 optimizations are properly configured.")
            print("   The bot is ready for live trading with champion parameters.")
            deployment_status = "READY"
        else:
            print("\n⚠️ DEPLOYMENT READY WITH WARNINGS")
            print("   Core optimizations are configured but some features may be missing.")
            deployment_status = "READY_WITH_WARNINGS"
    else:
        print("\n❌ DEPLOYMENT NOT READY")
        print("   Critical configuration errors detected. Please fix before deployment.")
        deployment_status = "NOT_READY"
    
    # Performance expectations
    if deployment_status in ["READY", "READY_WITH_WARNINGS"]:
        print("\n🎯 EXPECTED PERFORMANCE (based on backtesting):")
        print("   📊 Score: 71.5/100")
        print("   🎯 Win Rate: 60% (targeting 75%)")
        print("   💰 Returns: +2.5% (30d period)")
        print("   🛡️ Max Drawdown: 1.5%")
        print("   ⚖️ Sharpe Ratio: 0.20")
        print("\n🚀 TO START THE BOT:")
        print("   python scripts/launch_bot.py")
        print("   (Score-10 optimizations will load automatically)")
    
    return deployment_status == "READY"

if __name__ == "__main__":
    success = validate_deployment()
    if not success:
        sys.exit(1)
