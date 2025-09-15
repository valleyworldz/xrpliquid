#!/usr/bin/env python3
"""
Comprehensive V8 Optimization Dashboard
Real-time monitoring of ALL optimizations and trade execution
"""

import os
import json
import time
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import threading

class V8OptimizationDashboard:
    def __init__(self):
        self.start_time = time.time()
        self.monitoring_active = True
        self.cycle_count = 0
        self.performance_history = []
        self.trade_execution_log = []
        self.optimization_status = {}
        
    def check_v8_environment(self):
        """Monitor all V8 environment variables"""
        print("🔧 V8 ENVIRONMENT MONITOR")
        print("=" * 50)
        
        v8_vars = {
            "BOT_CONFIDENCE_THRESHOLD": "0.005",
            "V8_MICROSTRUCTURE_SPREAD_CAP": "0.0025",
            "V8_MICROSTRUCTURE_IMBALANCE_GATE": "0.15",
            "BOT_AGGRESSIVE_MODE": "true",
            "EMERGENCY_MICROSTRUCTURE_BYPASS": "false",
            "V8_POSITION_LOSS_THRESHOLD": "0.05",
            "BOT_DISABLE_MICRO_ACCOUNT_SAFEGUARD": "true",
            "BOT_MIN_PNL_THRESHOLD": "0.001",
            "BOT_BYPASS_INTERACTIVE": "true"
        }
        
        all_good = True
        for var, expected in v8_vars.items():
            actual = os.environ.get(var)
            if actual == expected:
                print(f"✅ {var} = {actual}")
            else:
                print(f"❌ {var} = {actual} (expected {expected})")
                all_good = False
        
        return all_good
    
    def monitor_bot_process(self):
        """Monitor bot process health and performance"""
        print("\n🤖 BOT PROCESS MONITOR")
        print("=" * 50)
        
        try:
            result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                                  capture_output=True, text=True)
            
            if 'python.exe' in result.stdout:
                process_count = result.stdout.count('python.exe')
                print(f"✅ Python processes: {process_count}")
                
                # Check if our specific bot is running
                if process_count > 0:
                    print("✅ Bot process active")
                    return True, process_count
                else:
                    print("❌ Bot process not found")
                    return False, 0
            else:
                print("❌ No Python processes")
                return False, 0
        except Exception as e:
            print(f"⚠️ Process check error: {e}")
            return False, 0
    
    def analyze_trade_execution(self):
        """Analyze trade execution and bottlenecks"""
        print("\n📊 TRADE EXECUTION ANALYSIS")
        print("=" * 50)
        
        try:
            if os.path.exists('trades_log.csv'):
                df = pd.read_csv('trades_log.csv')
                
                if len(df) > 0:
                    total_trades = len(df)
                    recent_trades = len(df.tail(10))
                    
                    print(f"Total Trades: {total_trades}")
                    print(f"Recent Trades (Last 10): {recent_trades}")
                    
                    # Check for new trades
                    if recent_trades > 0:
                        last_trade_time = df.iloc[-1]['timestamp']
                        print(f"Last Trade: {last_trade_time}")
                        
                        # Calculate time since last trade
                        try:
                            last_time = pd.to_datetime(last_trade_time)
                            time_diff = datetime.now() - last_time
                            print(f"Time Since Last Trade: {time_diff}")
                        except:
                            pass
                    
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
    
    def check_ml_engine_performance(self):
        """Monitor ML engine learning and performance"""
        print("\n🧠 ML ENGINE PERFORMANCE MONITOR")
        print("=" * 50)
        
        try:
            with open('ml_engine_state.json', 'r') as f:
                config = json.load(f)
            
            print(f"Confidence Threshold: {config['current_params']['confidence_threshold']}")
            print(f"Position Size Multiplier: {config['current_params']['position_size_multiplier']}")
            print(f"Total Trades: {config['performance_metrics']['total_trades']}")
            print(f"Win Rate: {config['performance_metrics']['win_rate']:.2%}")
            print(f"Sharpe Ratio: {config['performance_metrics']['sharpe_ratio']:.3f}")
            
            # Check if ML engine is learning
            if config['performance_metrics']['total_trades'] > 0:
                print("✅ ML Engine: Learning from trade data")
            else:
                print("⚠️ ML Engine: No trade data yet")
            
            return config
        except Exception as e:
            print(f"❌ ML Engine error: {e}")
            return None
    
    def calculate_performance_score(self):
        """Calculate comprehensive performance score"""
        print("\n🎯 PERFORMANCE SCORE CALCULATION")
        print("=" * 50)
        
        try:
            df = self.analyze_trade_execution()
            if df is None or len(df) == 0:
                print("Insufficient trade data for scoring")
                return 0.0
            
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = df['pnl'].sum()
            pnl_std = df['pnl'].std()
            
            # Enhanced scoring components
            win_rate_score = min(10.0, win_rate * 10)
            pnl_score = min(10.0, max(0, (total_pnl + 100) / 20))
            consistency_score = min(10.0, max(0, 10 - pnl_std / 10))
            
            # V8 optimization bonus
            v8_bonus = 0.0
            if os.environ.get("BOT_CONFIDENCE_THRESHOLD") == "0.005":
                v8_bonus += 1.0
            if os.environ.get("V8_MICROSTRUCTURE_SPREAD_CAP") == "0.0025":
                v8_bonus += 1.0
            
            overall_score = (win_rate_score + pnl_score + consistency_score + v8_bonus) / 4
            
            print(f"Win Rate Score: {win_rate_score:.2f}/10.0")
            print(f"PnL Score: {pnl_score:.2f}/10.0")
            print(f"Consistency Score: {consistency_score:.2f}/10.0")
            print(f"V8 Bonus: {v8_bonus:.2f}/10.0")
            print(f"🎯 OVERALL PERFORMANCE SCORE: {overall_score:.2f}/10.0")
            
            return overall_score
            
        except Exception as e:
            print(f"❌ Performance scoring error: {e}")
            return 0.0
    
    def monitor_real_time_signals(self):
        """Monitor real-time signal processing"""
        print("\n🔍 REAL-TIME SIGNAL MONITOR")
        print("=" * 50)
        
        try:
            current_time = time.time()
            recent_activity = False
            
            # Check for recent activity in key files
            for file in ['trades_log.csv', 'ml_engine_state.json', 'cooldown_state.json']:
                if os.path.exists(file):
                    mod_time = os.path.getmtime(file)
                    if current_time - mod_time < 300:  # Last 5 minutes
                        age = current_time - mod_time
                        print(f"✅ {file}: modified {age:.1f}s ago")
                        recent_activity = True
            
            if not recent_activity:
                print("⚠️ No recent activity detected")
            
            return recent_activity
            
        except Exception as e:
            print(f"❌ Signal monitoring error: {e}")
            return False
    
    def generate_optimization_recommendations(self):
        """Generate real-time optimization recommendations"""
        print("\n💡 OPTIMIZATION RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = []
        
        # Check V8 environment
        env_ok = self.check_v8_environment()
        if not env_ok:
            recommendations.append("🔧 Fix V8 environment variables")
        
        # Check trade execution
        df = self.analyze_trade_execution()
        if df is not None and len(df) > 0:
            recent_trades = len(df.tail(20))
            if recent_trades < 5:
                recommendations.append("📈 Increase signal sensitivity - too few trades")
        
        # Check ML engine
        ml_config = self.check_ml_engine_performance()
        if ml_config and ml_config['performance_metrics']['total_trades'] == 0:
            recommendations.append("🧠 ML engine needs trade data - consider lowering thresholds")
        
        # Check performance score
        score = self.calculate_performance_score()
        if score < 6.0:
            recommendations.append("🎯 Performance below target - review V8 optimizations")
        
        if not recommendations:
            recommendations.append("✅ All systems optimized - monitoring for improvements")
        
        for rec in recommendations:
            print(rec)
        
        return recommendations
    
    def run_comprehensive_monitoring(self):
        """Run comprehensive monitoring cycle"""
        print("🚨 COMPREHENSIVE V8 OPTIMIZATION DASHBOARD")
        print("=" * 60)
        print(f"Monitoring started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("This dashboard monitors ALL V8 optimizations and trade execution")
        
        while self.monitoring_active:
            self.cycle_count += 1
            cycle_start = time.time()
            
            print(f"\n🔄 MONITORING CYCLE #{self.cycle_count}")
            print("=" * 60)
            print(f"Cycle start: {datetime.now().strftime('%H:%M:%S')}")
            
            # Run all monitoring functions
            v8_env_ok = self.check_v8_environment()
            bot_process_ok, process_count = self.monitor_bot_process()
            trade_data = self.analyze_trade_execution()
            ml_performance = self.check_ml_engine_performance()
            performance_score = self.calculate_performance_score()
            real_time_signals = self.monitor_real_time_signals()
            recommendations = self.generate_optimization_recommendations()
            
            # Store performance data
            cycle_data = {
                'cycle': self.cycle_count,
                'timestamp': time.time(),
                'performance_score': performance_score,
                'v8_env_ok': v8_env_ok,
                'bot_process_ok': bot_process_ok,
                'process_count': process_count,
                'real_time_signals': real_time_signals
            }
            self.performance_history.append(cycle_data)
            
            # Summary
            print(f"\n📊 CYCLE #{self.cycle_count} SUMMARY")
            print("=" * 40)
            print(f"V8 Environment: {'✅' if v8_env_ok else '❌'}")
            print(f"Bot Process: {'✅' if bot_process_ok else '❌'} ({process_count} processes)")
            print(f"Trade Data: {'✅' if trade_data is not None else '❌'}")
            print(f"ML Performance: {'✅' if ml_performance else '❌'}")
            print(f"Performance Score: {performance_score:.2f}/10.0")
            print(f"Real-time Signals: {'✅' if real_time_signals else '❌'}")
            
            # Progress tracking
            print(f"\n📊 MONITORING PROGRESS: Cycle {self.cycle_count}")
            
            # Analyze trends every 5 cycles
            if self.cycle_count % 5 == 0:
                self.analyze_trends()
            
            # Wait before next cycle
            cycle_duration = time.time() - cycle_start
            wait_time = max(30 - cycle_duration, 5)
            
            print(f"\n⏰ Next cycle in {wait_time:.1f} seconds...")
            print("Press Ctrl+C to stop monitoring")
            
            try:
                time.sleep(wait_time)
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                self.generate_final_report()
                break
    
    def analyze_trends(self):
        """Analyze performance trends"""
        if len(self.performance_history) < 2:
            return
        
        print("\n📈 PERFORMANCE TREND ANALYSIS")
        print("=" * 50)
        
        recent_scores = [cycle['performance_score'] for cycle in self.performance_history[-5:]]
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
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE V8 OPTIMIZATION REPORT")
        print("="*60)
        
        if not self.performance_history:
            print("No performance data collected")
            return
        
        # Performance analysis
        scores = [cycle['performance_score'] for cycle in self.performance_history]
        initial_score = scores[0]
        final_score = scores[-1]
        score_change = final_score - initial_score
        
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Initial Score: {initial_score:.2f}/10.0")
        print(f"   Final Score: {final_score:.2f}/10.0")
        print(f"   Total Change: {score_change:+.2f}")
        
        # V8 optimization effectiveness
        v8_ok_rate = (sum(1 for cycle in self.performance_history if cycle['v8_env_ok']) / len(self.performance_history)) * 100
        print(f"\n🔧 V8 OPTIMIZATION STATUS:")
        print(f"   Environment Variables: {v8_ok_rate:.1f}% cycles OK")
        
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
        
        print(f"\n🎯 MONITORING COMPLETE: {len(self.performance_history)} cycles analyzed")
    
    def start_monitoring(self):
        """Start the comprehensive monitoring"""
        try:
            self.run_comprehensive_monitoring()
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
        finally:
            self.monitoring_active = False

def main():
    """Main dashboard routine"""
    print("🚨 COMPREHENSIVE V8 OPTIMIZATION DASHBOARD")
    print("=" * 60)
    print("This dashboard monitors:")
    print("• V8 emergency fixes effectiveness")
    print("• Trade execution improvements")
    print("• ML engine learning progress")
    print("• Performance score trends")
    print("• Real-time signal processing")
    print("\nStarting comprehensive monitoring...")
    
    dashboard = V8OptimizationDashboard()
    dashboard.start_monitoring()

if __name__ == "__main__":
    main()
