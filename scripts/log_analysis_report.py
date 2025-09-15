#!/usr/bin/env python3
"""
LOG ANALYSIS REPORT
Comprehensive analysis of bot logs across all hats and job titles
"""

import re
from datetime import datetime
from collections import defaultdict, Counter

class LogAnalysisReport:
    def __init__(self, log_content):
        self.log_content = log_content
        self.analysis_results = {}
        
    def analyze_startup_sequence(self):
        """Analyze bot startup sequence"""
        print("🚀 STARTUP SEQUENCE ANALYSIS")
        print("=" * 60)
        
        startup_events = []
        lines = self.log_content.split('\n')
        
        for line in lines:
            if any(keyword in line for keyword in [
                'MULTI-ASSET TRADING BOT',
                'PROFESSIONAL TRADING PROFILES',
                'A.I. ULTIMATE',
                'CHAMPION +213%',
                'XRP FORCED',
                'Score-10 champion configuration'
            ]):
                startup_events.append(line.strip())
        
        print("✅ SUCCESSFUL STARTUP EVENTS:")
        for event in startup_events[:10]:  # Show first 10 events
            print(f"   {event}")
        
        return {
            'startup_successful': len(startup_events) > 0,
            'profile_selected': 'A.I. ULTIMATE' in self.log_content,
            'xrp_forced': 'XRP FORCED' in self.log_content,
            'champion_config_loaded': 'Score-10 champion configuration' in self.log_content
        }
    
    def analyze_emergency_issues(self):
        """Analyze emergency and error issues"""
        print("\n🚨 EMERGENCY ISSUES ANALYSIS")
        print("=" * 60)
        
        emergency_patterns = {
            'drawdown_exceeded': r'EMERGENCY: (\d+)% drawdown exceeded',
            'risk_check_failed': r'EMERGENCY: Risk check failed',
            'regime_reconfigure_failed': r'Mid-session regime reconfigure failed',
            'holographic_storage_failed': r'Holographic storage initialization failed',
            'no_fee_tiers': r'No fee tiers found in meta'
        }
        
        emergency_counts = {}
        for pattern_name, pattern in emergency_patterns.items():
            matches = re.findall(pattern, self.log_content)
            emergency_counts[pattern_name] = len(matches)
            if matches:
                print(f"⚠️ {pattern_name.upper()}: {len(matches)} occurrences")
                if pattern_name == 'drawdown_exceeded':
                    print(f"   Latest drawdown: {matches[-1]}%")
        
        return emergency_counts
    
    def analyze_performance_metrics(self):
        """Analyze performance metrics from logs"""
        print("\n📊 PERFORMANCE METRICS ANALYSIS")
        print("=" * 60)
        
        # Extract performance scores
        performance_scores = re.findall(r'PERFORMANCE SCORE: (\d+\.\d+)/10\.0', self.log_content)
        if performance_scores:
            latest_score = float(performance_scores[-1])
            print(f"📊 Latest Performance Score: {latest_score}/10.0")
        
        # Extract individual component scores
        component_scores = {}
        components = ['Win Rate', 'Profit Factor', 'Drawdown Control', 'Signal Quality', 'Risk Management', 'Market Adaptation']
        
        for component in components:
            pattern = f'{component}: (\\d+\\.\\d+)/10\\.0'
            matches = re.findall(pattern, self.log_content)
            if matches:
                component_scores[component] = float(matches[-1])
                print(f"📊 {component}: {matches[-1]}/10.0")
        
        # Extract account values
        account_values = re.findall(r'Account values - Withdrawable: \$(\d+\.\d+), Free Collateral: \$(\d+\.\d+), Account Value: \$(\d+\.\d+)', self.log_content)
        if account_values:
            latest_values = account_values[-1]
            print(f"💰 Account Values:")
            print(f"   Withdrawable: ${latest_values[0]}")
            print(f"   Free Collateral: ${latest_values[1]}")
            print(f"   Account Value: ${latest_values[2]}")
        
        return {
            'performance_score': latest_score if performance_scores else None,
            'component_scores': component_scores,
            'account_values': latest_values if account_values else None
        }
    
    def analyze_optimization_attempts(self):
        """Analyze optimization attempts and results"""
        print("\n🔧 OPTIMIZATION ANALYSIS")
        print("=" * 60)
        
        optimization_events = []
        lines = self.log_content.split('\n')
        
        for line in lines:
            if any(keyword in line for keyword in [
                'AUTO-OPTIMIZATION START',
                'OPTIMIZATION COMPLETE',
                'NO OPTIMIZATION IMPROVEMENT',
                'WIN RATE OPTIMIZATION',
                'GUARDIAN OPTIMIZATION'
            ]):
                optimization_events.append(line.strip())
        
        print("🔧 OPTIMIZATION EVENTS:")
        for event in optimization_events[-10:]:  # Show last 10 events
            print(f"   {event}")
        
        # Count optimization attempts
        optimization_attempts = len([line for line in lines if 'AUTO-OPTIMIZATION START' in line])
        failed_optimizations = len([line for line in lines if 'NO OPTIMIZATION IMPROVEMENT' in line])
        
        print(f"\n📊 Optimization Summary:")
        print(f"   Total Attempts: {optimization_attempts}")
        print(f"   Failed Attempts: {failed_optimizations}")
        print(f"   Success Rate: {((optimization_attempts - failed_optimizations) / optimization_attempts * 100) if optimization_attempts > 0 else 0:.1f}%")
        
        return {
            'total_attempts': optimization_attempts,
            'failed_attempts': failed_optimizations,
            'success_rate': ((optimization_attempts - failed_optimizations) / optimization_attempts * 100) if optimization_attempts > 0 else 0
        }
    
    def analyze_system_health(self):
        """Analyze overall system health"""
        print("\n🏥 SYSTEM HEALTH ANALYSIS")
        print("=" * 60)
        
        # Count different log levels
        log_levels = Counter()
        lines = self.log_content.split('\n')
        
        for line in lines:
            if 'INFO:' in line:
                log_levels['INFO'] += 1
            elif 'WARNING:' in line:
                log_levels['WARNING'] += 1
            elif 'ERROR:' in line:
                log_levels['ERROR'] += 1
            elif 'DEBUG:' in line:
                log_levels['DEBUG'] += 1
        
        print("📊 LOG LEVEL DISTRIBUTION:")
        for level, count in log_levels.items():
            print(f"   {level}: {count}")
        
        # Calculate health score
        total_logs = sum(log_levels.values())
        error_rate = (log_levels['ERROR'] / total_logs * 100) if total_logs > 0 else 0
        warning_rate = (log_levels['WARNING'] / total_logs * 100) if total_logs > 0 else 0
        
        health_score = max(0, 100 - (error_rate * 2) - (warning_rate * 0.5))
        
        print(f"\n🏥 SYSTEM HEALTH SCORE: {health_score:.1f}/100")
        print(f"   Error Rate: {error_rate:.1f}%")
        print(f"   Warning Rate: {warning_rate:.1f}%")
        
        return {
            'log_levels': dict(log_levels),
            'health_score': health_score,
            'error_rate': error_rate,
            'warning_rate': warning_rate
        }
    
    def analyze_trading_activity(self):
        """Analyze trading activity and patterns"""
        print("\n📈 TRADING ACTIVITY ANALYSIS")
        print("=" * 60)
        
        # Check for trading signals
        trading_signals = len([line for line in self.log_content.split('\n') if 'trading signal' in line.lower()])
        
        # Check for position changes
        position_changes = len([line for line in self.log_content.split('\n') if 'position' in line.lower() and ('open' in line.lower() or 'close' in line.lower())])
        
        # Check for market data updates
        market_updates = len([line for line in self.log_content.split('\n') if 'snapshot' in line or 'bid' in line or 'ask' in line])
        
        print(f"📊 Trading Activity Summary:")
        print(f"   Trading Signals: {trading_signals}")
        print(f"   Position Changes: {position_changes}")
        print(f"   Market Updates: {market_updates}")
        
        # Check if bot is actively trading
        is_trading = position_changes > 0 or trading_signals > 0
        print(f"   Active Trading: {'✅ YES' if is_trading else '❌ NO'}")
        
        return {
            'trading_signals': trading_signals,
            'position_changes': position_changes,
            'market_updates': market_updates,
            'is_trading': is_trading
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive log analysis report"""
        print("🚀 COMPREHENSIVE LOG ANALYSIS REPORT")
        print("=" * 60)
        print("🎯 ANALYZING BOT LOGS ACROSS ALL HATS AND JOB TITLES")
        print("=" * 60)
        
        # Run all analyses
        startup_analysis = self.analyze_startup_sequence()
        emergency_analysis = self.analyze_emergency_issues()
        performance_analysis = self.analyze_performance_metrics()
        optimization_analysis = self.analyze_optimization_attempts()
        health_analysis = self.analyze_system_health()
        trading_analysis = self.analyze_trading_activity()
        
        # Compile comprehensive report
        comprehensive_report = {
            'startup': startup_analysis,
            'emergency': emergency_analysis,
            'performance': performance_analysis,
            'optimization': optimization_analysis,
            'health': health_analysis,
            'trading': trading_analysis
        }
        
        print(f"\n🎉 COMPREHENSIVE LOG ANALYSIS COMPLETE!")
        print(f"📊 System Health Score: {health_analysis['health_score']:.1f}/100")
        print(f"📈 Performance Score: {performance_analysis['performance_score']}/10.0" if performance_analysis['performance_score'] else "📈 Performance Score: N/A")
        print(f"🔧 Optimization Success Rate: {optimization_analysis['success_rate']:.1f}%")
        print(f"📊 Active Trading: {'✅ YES' if trading_analysis['is_trading'] else '❌ NO'}")
        
        return comprehensive_report

def main():
    # This would be called with the actual log content
    log_content = """
    # Log content would be passed here
    """
    
    analyzer = LogAnalysisReport(log_content)
    report = analyzer.generate_comprehensive_report()
    return report

if __name__ == "__main__":
    main()
