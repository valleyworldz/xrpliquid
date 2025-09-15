#!/usr/bin/env python3
"""
Comprehensive XRP Bot Monitor - 25 Cycles
Utilizing ALL available tools and optimizations for extended monitoring
"""

import os
import json
import time
import subprocess
import pandas as pd
from datetime import datetime, timedelta

class ComprehensiveMonitor:
    def __init__(self, max_cycles=25):
        self.start_time = time.time()
        self.max_cycles = max_cycles
        self.current_cycle = 0
        self.monitoring_active = True
        self.performance_history = []
        
    def check_environment_variables(self):
        """Monitor all critical environment variables"""
        print("🔧 ENVIRONMENT VARIABLES MONITOR")
        print("=" * 50)
        
        critical_vars = {
            "BOT_CONFIDENCE_THRESHOLD": "0.005",
            "V8_MICROSTRUCTURE_SPREAD_CAP": "0.0025",
            "V8_MICROSTRUCTURE_IMBALANCE_GATE": "0.15",
            "BOT_AGGRESSIVE_MODE": "true",
            "EMERGENCY_MICROSTRUCTURE_BYPASS": "false",
            "V8_POSITION_LOSS_THRESHOLD": "0.05"
        }
        
        all_good = True
        for var, expected in critical_vars.items():
            actual = os.environ.get(var)
            if actual == expected:
                print(f"✅ {var} = {actual}")
            else:
                print(f"❌ {var} = {actual} (expected {expected})")
                all_good = False
        
        return all_good
    
    def check_ml_engine_status(self):
        """Monitor ML engine configuration and performance"""
        print("\n🧠 ML ENGINE MONITOR")
        print("=" * 50)
        
        try:
            with open('ml_engine_state.json', 'r') as f:
                config = json.load(f)
            
            print(f"Confidence Threshold: {config['current_params']['confidence_threshold']}")
            print(f"Position Size Multiplier: {config['current_params']['position_size_multiplier']}")
            print(f"Total Trades: {config['performance_metrics']['total_trades']}")
            print(f"Win Rate: {config['performance_metrics']['win_rate']:.2%}")
            print(f"Sharpe Ratio: {config['performance_metrics']['sharpe_ratio']:.3f}")
            
            return config
        except Exception as e:
            print(f"❌ ML Engine status error: {e}")
            return None
    
    def check_bot_process_status(self):
        """Monitor bot process health"""
        print("\n🤖 BOT PROCESS MONITOR")
        print("=" * 50)
        
        try:
            result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                                  capture_output=True, text=True)
            
            if 'python.exe' in result.stdout:
                print("✅ Python processes running")
                process_count = result.stdout.count('python.exe')
                print(f"Active Python processes: {process_count}")
                return True, process_count
            else:
                print("❌ No Python processes found")
                return False, 0
        except Exception as e:
            print(f"⚠️ Process check error: {e}")
            return False, 0
    
    def analyze_trade_history(self):
        """Analyze recent trade performance"""
        print("\n📊 TRADE HISTORY ANALYSIS")
        print("=" * 50)
        
        try:
            if os.path.exists('trades_log.csv'):
                df = pd.read_csv('trades_log.csv')
                
                if len(df) > 0:
                    print(f"Total Trades: {len(df)}")
                    print(f"BUY Trades: {len(df[df['side'] == 'BUY'])}")
                    print(f"SELL Trades: {len(df[df['side'] == 'SELL'])}")
                    
                    # Recent performance
                    recent_trades = df.tail(10)
                    if len(recent_trades) > 0:
                        print(f"\nRecent 10 Trades:")
                        for _, trade in recent_trades.iterrows():
                            timestamp = trade.get('timestamp', 'N/A')
                            side = trade.get('side', 'N/A')
                            pnl = trade.get('pnl', 0)
                            print(f"  {timestamp}: {side} | PnL: {pnl}")
                    
                    return df
                else:
                    print("No trades recorded yet")
                    return None
            else:
                print("No trades log found")
                return None
        except Exception as e:
            print(f"❌ Trade analysis error: {e}")
            return None
    
    def check_cooldown_status(self):
        """Monitor bot cooldown state"""
        print("\n⏳ COOLDOWN STATUS MONITOR")
        print("=" * 50)
        
        try:
            if os.path.exists('cooldown_state.json'):
                with open('cooldown_state.json', 'r') as f:
                    cooldown = json.load(f)
                
                until_ts = cooldown.get('until_ts', 0)
                if until_ts > 0:
                    remaining = until_ts - time.time()
                    if remaining > 0:
                        print(f"🔄 Cooldown active: {remaining:.1f}s remaining")
                    else:
                        print("✅ Cooldown expired - bot ready for trading")
                else:
                    print("✅ No cooldown active")
                
                return cooldown
            else:
                print("No cooldown state file found")
                return None
        except Exception as e:
            print(f"❌ Cooldown check error: {e}")
            return None
    
    def calculate_performance_score(self):
        """Calculate real-time performance score"""
        print("\n🎯 PERFORMANCE SCORE CALCULATION")
        print("=" * 50)
        
        try:
            df = self.analyze_trade_history()
            if df is None or len(df) == 0:
                print("Insufficient trade data for scoring")
                return 0.0
            
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = df['pnl'].sum()
            pnl_std = df['pnl'].std()
            
            win_rate_score = min(10.0, win_rate * 10)
            pnl_score = min(10.0, max(0, (total_pnl + 100) / 20))
            consistency_score = min(10.0, max(0, 10 - pnl_std / 10))
            
            overall_score = (win_rate_score + pnl_score + consistency_score) / 3
            
            print(f"Win Rate Score: {win_rate_score:.2f}/10.0")
            print(f"PnL Score: {pnl_score:.2f}/10.0")
            print(f"Consistency Score: {consistency_score:.3f}/10.0")
            print(f"🎯 OVERALL PERFORMANCE SCORE: {overall_score:.2f}/10.0")
            
            return overall_score
            
        except Exception as e:
            print(f"❌ Performance scoring error: {e}")
            return 0.0
    
    def monitor_real_time_activity(self):
        """Monitor real-time bot activity"""
        print("\n🔍 REAL-TIME ACTIVITY MONITOR")
        print("=" * 50)
        
        try:
            current_time = time.time()
            recent_files = []
            
            for file in ['trades_log.csv', 'ml_engine_state.json', 'cooldown_state.json']:
                if os.path.exists(file):
                    mod_time = os.path.getmtime(file)
                    if current_time - mod_time < 300:  # Last 5 minutes
                        recent_files.append(file)
            
            if recent_files:
                print("✅ Recent activity detected:")
                for file in recent_files:
                    mod_time = os.path.getmtime(file)
                    age = current_time - mod_time
                    print(f"  {file}: modified {age:.1f}s ago")
            else:
                print("⚠️ No recent activity detected")
            
            return len(recent_files) > 0
            
        except Exception as e:
            print(f"❌ Real-time monitoring error: {e}")
            return False
    
    def generate_optimization_recommendations(self):
        """Generate optimization recommendations"""
        print("\n💡 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = []
        
        if os.path.exists('trades_log.csv'):
            df = pd.read_csv('trades_log.csv')
            recent_trades = len(df.tail(20))
            if recent_trades < 5:
                recommendations.append("🔧 Increase signal sensitivity - too few trades")
        
        ml_config = self.check_ml_engine_status()
        if ml_config and ml_config['performance_metrics']['total_trades'] == 0:
            recommendations.append("🧠 ML engine needs trade data - consider lowering thresholds")
        
        cooldown = self.check_cooldown_status()
        if cooldown and cooldown.get('until_ts', 0) > time.time():
            recommendations.append("⏳ Cooldown active - consider adjusting risk parameters")
        
        if not recommendations:
            recommendations.append("✅ Current configuration appears optimal")
        
        for rec in recommendations:
            print(rec)
        
        return recommendations
    
    def analyze_trends(self, performance_history):
        """Analyze performance trends"""
        if len(performance_history) < 2:
            return
        
        print("\n📈 PERFORMANCE TREND ANALYSIS")
        print("=" * 50)
        
        recent_scores = [cycle['performance_score'] for cycle in performance_history[-5:]]
        if len(recent_scores) > 1:
            score_change = recent_scores[-1] - recent_scores[0]
            print(f"Score Change (Last 5 cycles): {score_change:+.2f}")
            
            if score_change > 0.5:
                print("🚀 SIGNIFICANT IMPROVEMENT DETECTED!")
            elif score_change > 0.1:
                print("📈 Gradual improvement detected")
            elif score_change < -0.1:
                print("📉 Performance declining - investigate")
            else:
                print("➡️ Performance stable")
    
    def generate_final_report(self, performance_history):
        """Generate final monitoring report"""
        print("\n" + "="*60)
        print("🎯 FINAL 25-CYCLE MONITORING REPORT")
        print("="*60)
        
        if not performance_history:
            print("No performance data collected")
            return
        
        # Performance analysis
        scores = [cycle['score'] for cycle in performance_history]
        initial_score = scores[0]
        final_score = scores[-1]
        score_change = final_score - initial_score
        
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Initial Score: {initial_score:.2f}/10.0")
        print(f"   Final Score: {final_score:.2f}/10.0")
        print(f"   Total Change: {score_change:+.2f}")
        
        # Activity analysis
        active_cycles = sum(1 for cycle in performance_history if cycle['activity'])
        activity_rate = (active_cycles / len(performance_history)) * 100
        
        print(f"\n🔍 ACTIVITY ANALYSIS:")
        print(f"   Total Cycles: {len(performance_history)}")
        print(f"   Active Cycles: {active_cycles}")
        print(f"   Activity Rate: {activity_rate:.1f}%")
        
        # V8 optimization effectiveness
        env_ok_rate = (sum(1 for cycle in performance_history if cycle['env_ok']) / len(performance_history)) * 100
        print(f"\n🔧 V8 OPTIMIZATION STATUS:")
        print(f"   Environment Variables: {env_ok_rate:.1f}% cycles OK")
        
        # Final recommendations
        print(f"\n💡 FINAL RECOMMENDATIONS:")
        if score_change > 1.0:
            print("🚀 EXCELLENT: V8 optimizations working perfectly!")
        elif score_change > 0.5:
            print("✅ GOOD: V8 optimizations showing improvement")
        elif score_change > 0:
            print("📈 POSITIVE: V8 optimizations starting to work")
        else:
            print("⚠️ NEEDS ATTENTION: V8 optimizations may need adjustment")
        
        print(f"\n🎯 MONITORING COMPLETE: {len(performance_history)} cycles analyzed")
        print("Next step: Continue monitoring for trade execution improvements")
    
    def run_extended_monitoring(self):
        """Run extended monitoring for specified cycles"""
        print("🚨 25-CYCLE COMPREHENSIVE XRP BOT MONITOR")
        print("=" * 60)
        print(f"Monitoring started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target cycles: {self.max_cycles}")
        print("This monitor will track V8 optimizations and trade execution improvements")
        
        while self.monitoring_active and self.current_cycle < self.max_cycles:
            self.current_cycle += 1
            cycle_start_time = time.time()
            
            print(f"\n🔄 MONITORING CYCLE #{self.current_cycle}/{self.max_cycles}")
            print("=" * 60)
            print(f"Cycle start: {datetime.now().strftime('%H:%M:%S')}")
            
            # Run all monitoring functions
            env_ok = self.check_environment_variables()
            ml_status = self.check_ml_engine_status()
            process_ok, process_count = self.check_bot_process_status()
            trade_data = self.analyze_trade_history()
            cooldown_status = self.check_cooldown_status()
            performance_score = self.calculate_performance_score()
            real_time_activity = self.monitor_real_time_activity()
            recommendations = self.generate_optimization_recommendations()
            
            # Store cycle data
            cycle_data = {
                'cycle': self.current_cycle,
                'timestamp': time.time(),
                'performance_score': performance_score,
                'process_count': process_count,
                'real_time_activity': real_time_activity,
                'environment_ok': env_ok
            }
            self.performance_history.append(cycle_data)
            
            # Summary
            print(f"\n📊 CYCLE #{self.current_cycle} SUMMARY")
            print("=" * 40)
            print(f"Environment Variables: {'✅' if env_ok else '❌'}")
            print(f"ML Engine: {'✅' if ml_status else '❌'}")
            print(f"Bot Process: {'✅' if process_ok else '❌'} ({process_count} processes)")
            print(f"Trade Data: {'✅' if trade_data is not None else '❌'}")
            print(f"Performance Score: {performance_score:.2f}/10.0")
            print(f"Real-time Activity: {'✅' if real_time_activity else '❌'}")
            
            # Progress tracking
            progress = (self.current_cycle / self.max_cycles) * 100
            print(f"\n📊 MONITORING PROGRESS: {progress:.1f}% ({self.current_cycle}/{self.max_cycles})")
            
            # Analyze trends every 5 cycles
            if self.current_cycle % 5 == 0:
                self.analyze_trends(self.performance_history)
            
            # Check if we've reached the target
            if self.current_cycle >= self.max_cycles:
                print(f"\n🎯 TARGET REACHED: {self.max_cycles} cycles completed!")
                self.generate_final_report(self.performance_history)
                break
            
            # Wait before next cycle
            cycle_duration = time.time() - cycle_start_time
            wait_time = max(30 - cycle_duration, 5)  # Ensure minimum 5s wait
            
            print(f"\n⏰ Next cycle in {wait_time:.1f} seconds...")
            print("Press Ctrl+C to stop monitoring early")
            
            try:
                time.sleep(wait_time)
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                self.generate_final_report(self.performance_history)
                break
        
        # Final analysis
        if self.current_cycle >= self.max_cycles:
            self.generate_final_report(self.performance_history)
    
    def start_monitoring(self):
        """Start the extended monitoring"""
        try:
            self.run_extended_monitoring()
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        finally:
            self.monitoring_active = False

def main():
    """Main monitoring routine"""
    print("🚨 25-CYCLE COMPREHENSIVE XRP BOT MONITOR")
    print("=" * 60)
    print("This extended monitor will track:")
    print("• V8 emergency fixes effectiveness")
    print("• Trade execution improvements")
    print("• Performance score trends")
    print("• ML engine learning progress")
    print("• Real-time activity patterns")
    print("\nStarting 25-cycle comprehensive monitoring...")
    
    monitor = ComprehensiveMonitor(max_cycles=25)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
