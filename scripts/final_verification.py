#!/usr/bin/env python3
"""
Final Verification - Check for any remaining trade-blocking logic
"""

import os
import time
import csv
import psutil

def final_verification():
    """Final verification of trade execution readiness"""
    print("🚨 FINAL VERIFICATION - MAXIMUM TRADE EXECUTION READINESS")
    print("=" * 70)
    
    # Check if bot is running
    bot_running = False
    bot_pid = None
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and 'newbotcode.py' in ' '.join(proc.info['cmdline'] or []):
                bot_running = True
                bot_pid = proc.info['pid']
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not bot_running:
        print("❌ Bot is NOT running!")
        return False
    
    print(f"✅ Bot is running (PID: {bot_pid})")
    
    # Check bot resources
    try:
        process = psutil.Process(bot_pid)
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        print(f"\n💻 Bot Resource Usage:")
        print(f"   CPU: {cpu_percent:.2f}%")
        print(f"   Memory: {memory_mb:.1f} MB")
        print(f"   Status: {'🟢 ACTIVE' if cpu_percent > 0 else '🟡 IDLE'}")
        
    except psutil.NoSuchProcess:
        print("❌ Bot process not found")
        return False
    
    # Check for any remaining trade-blocking logic in source code
    source_file = "newbotcode.py"
    if os.path.exists(source_file):
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for remaining trade-blocking patterns
        blocking_patterns = [
            "expected_pnl < threshold",
            "expected_pnl >= threshold",
            "if expected_pnl <",
            "if expected_pnl >=",
            "🚫 Skipping",
            "❌ Skipping",
            "Trade blocked",
            "return False",
            "skip_trade = True",
            "block_trade = True",
            "apply_veto = True",
            "if confidence < threshold",
            "if rsi < 30",
            "if rsi > 70",
            "if abs(momentum) <",
            "if fee > max_fee",
            "if position_size > max_size",
            "if risk_level > max_risk",
            "if drawdown > max_drawdown",
            "time.sleep(cooldown",
            "if signal == \"HOLD\"",
        ]
        
        print(f"\n🔍 FINAL TRADE-BLOCKING LOGIC VERIFICATION:")
        print("-" * 60)
        
        remaining_blocks = 0
        for pattern in blocking_patterns:
            if pattern in content:
                print(f"⚠️  REMAINING BLOCK: {pattern}")
                remaining_blocks += 1
            else:
                print(f"✅ REMOVED: {pattern}")
        
        print(f"\n📊 Blocking Logic Status: {remaining_blocks}/{len(blocking_patterns)} patterns found")
        
        if remaining_blocks == 0:
            print("🎯 EXCELLENT: ALL trade-blocking logic has been removed!")
        else:
            print(f"⚠️  WARNING: {remaining_blocks} trade-blocking patterns still exist")
    
    # Check trades log for recent activity
    trades_file = "trades_log.csv"
    if os.path.exists(trades_file):
        with open(trades_file, 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        
        if trades:
            recent_trades = trades[-3:]  # Last 3 trades
            print(f"\n📊 Recent Trades (Last 3):")
            print("-" * 80)
            
            for trade in recent_trades:
                timestamp = trade.get('timestamp', 'N/A')
                side = trade.get('side', 'N/A')
                size = trade.get('size', 'N/A')
                confidence = trade.get('confidence', 'N/A')
                
                print(f"🕒 {timestamp} | {side} | Size: {size} | Conf: {confidence}")
            
            # Check if new trades are being generated
            latest_time = trades[-1]['timestamp'] if trades else None
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n⏰ Trade Timing Analysis:")
            print(f"   Latest trade: {latest_time}")
            print(f"   Current time: {current_time}")
            
            if latest_time:
                # Parse latest trade time
                try:
                    from datetime import datetime
                    latest_dt = datetime.strptime(latest_time, "%Y-%m-%d %H:%M:%S")
                    current_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                    time_diff = (current_dt - latest_dt).total_seconds()
                    
                    if time_diff < 300:  # Less than 5 minutes
                        print(f"   ✅ Recent trading activity (last trade: {time_diff:.0f}s ago)")
                    else:
                        print(f"   ⚠️  No recent trades (last trade: {time_diff/60:.1f} minutes ago)")
                except:
                    print("   ⚠️  Could not parse trade timing")
    
    # Final status assessment
    print(f"\n🎯 FINAL TRADE EXECUTION READINESS ASSESSMENT:")
    print("=" * 60)
    print(f"   Bot Running: {'✅ YES' if bot_running else '❌ NO'}")
    
    # Check each restriction individually
    pnl_removed = 'expected_pnl < threshold' not in content
    confidence_removed = 'if confidence < threshold' not in content
    rsi_removed = 'if rsi < 30' not in content
    momentum_removed = 'if abs(momentum) <' not in content
    fee_removed = 'if fee > max_fee' not in content
    position_removed = 'if position_size > max_size' not in content
    risk_removed = 'if risk_level > max_risk' not in content
    drawdown_removed = 'if drawdown > max_drawdown' not in content
    cooldown_removed = 'time.sleep(cooldown' not in content
    hold_removed = 'if signal == "HOLD"' not in content
    
    print(f"   PnL Thresholds: {'✅ REMOVED' if pnl_removed else '❌ STILL EXIST'}")
    print(f"   Confidence Filters: {'✅ REMOVED' if confidence_removed else '❌ STILL EXIST'}")
    print(f"   RSI Filters: {'✅ REMOVED' if rsi_removed else '❌ STILL EXIST'}")
    print(f"   Momentum Filters: {'✅ REMOVED' if momentum_removed else '❌ STILL EXIST'}")
    print(f"   Fee Thresholds: {'✅ REMOVED' if fee_removed else '❌ STILL EXIST'}")
    print(f"   Position Limits: {'✅ REMOVED' if position_removed else '❌ STILL EXIST'}")
    print(f"   Risk Checks: {'✅ REMOVED' if risk_removed else '❌ STILL EXIST'}")
    print(f"   Drawdown Throttle: {'✅ REMOVED' if drawdown_removed else '❌ STILL EXIST'}")
    print(f"   Cooldown Periods: {'✅ REMOVED' if cooldown_removed else '❌ STILL EXIST'}")
    print(f"   HOLD Signals: {'✅ REMOVED' if hold_removed else '❌ STILL EXIST'}")
    
    # Overall readiness score
    total_checks = 11
    passed_checks = sum([
        pnl_removed,
        confidence_removed,
        rsi_removed,
        momentum_removed,
        fee_removed,
        position_removed,
        risk_removed,
        drawdown_removed,
        cooldown_removed,
        hold_removed,
        bot_running
    ])
    
    readiness_score = (passed_checks / total_checks) * 100
    
    print(f"\n📊 OVERALL READINESS SCORE: {readiness_score:.1f}%")
    
    if readiness_score >= 90:
        print("🎯 EXCELLENT: Bot is ready for MAXIMUM TRADE EXECUTION!")
    elif readiness_score >= 70:
        print("✅ GOOD: Bot is mostly ready for high-frequency trading")
    elif readiness_score >= 50:
        print("⚠️  MODERATE: Some restrictions remain, trading may be limited")
    else:
        print("❌ POOR: Many restrictions remain, trading will be severely limited")
    
    return True

if __name__ == "__main__":
    final_verification()
