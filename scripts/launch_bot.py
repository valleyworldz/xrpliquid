#!/usr/bin/env python3
"""
Trading Bot Launcher
Launches the autonomous trading engine with proper configuration
"""

import sys
import os
import asyncio
import json
from typing import Any, Dict

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def load_config():
    """Load the enhanced trading configuration"""
    try:
        with open('config/enhanced_trading_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Enhanced config not found, using default config")
        return {}

def load_json(path: str) -> Dict[str, Any]:
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(data, f, indent=2, ensure_ascii=False)


async def main():
    """Main launcher function"""
    print("🚀 LAUNCHING SCORE-10 OPTIMIZED TRADING BOT...")
    print("=" * 60)
    
    # Check for Score-10 optimizations
    score_10_config = None
    try:
        with open('live_runtime_overrides.json', 'r') as f:
            score_10_config = json.load(f)
        
        if score_10_config.get('score_10_mode'):
            print("🏆 SCORE-10 OPTIMIZATIONS DETECTED!")
            print(f"   Champion Profile: {score_10_config['champion_profile']}")
            print(f"   Champion Score: {score_10_config['champion_score']:.1f}/100")
            print(f"   Target Win Rate: {score_10_config['performance_targets']['target_win_rate']}%")
            print(f"   Expected Returns: Based on backtesting")
            print(f"   Risk Control: Ultra-tight stops + nano profit taking")
            print("   Features: ML signals + regime detection + smart sizing")
            print()
    except FileNotFoundError:
        print("ℹ️ No Score-10 optimizations found, using standard configuration")
    
    # Load base configuration
    config = load_config()
    print(f"✅ Configuration loaded: {len(config)} sections")
    
    try:
        # Import and start the trading engine
        from core.autonomous_master_engine import SupremeAutonomousEngine
        from core.utils.config_manager import ConfigManager
        
        # Initialize config manager
        config_manager = ConfigManager()
        config_manager.update_config(config)
        
        # Create and start the engine
        engine = SupremeAutonomousEngine(config_manager)
        print("✅ Trading engine initialized")
        
        # Start autonomous trading
        await engine.start_autonomous_trading()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check that all dependencies are installed")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        import traceback
        traceback.print_exc()

def main() -> int:
	base = os.getcwd()
	live_params_path = os.path.join(base, 'optimized_params_live.json')
	last_cfg_path = os.path.join(base, 'last_config.json')
	if not os.path.exists(live_params_path):
		print('optimized_params_live.json not found. Run export_live_params.py first.')
		return 1
	if not os.path.exists(last_cfg_path):
		print('last_config.json not found. Launch your bot once to create it.')
		return 1

	live_params = load_json(live_params_path)
	last_cfg = load_json(last_cfg_path)

	# Choose profile with highest overall_score
	best_key = None
	best_score = float('-inf')
	for k, v in live_params.get('profiles', {}).items():
		try:
			score = float(v.get('meta', {}).get('overall_score', 0.0))
			if score > best_score:
				best_score = score
				best_key = k
		except Exception:
			continue
	if not best_key:
		print('No profiles found in optimized_params_live.json')
		return 1

	best = live_params['profiles'][best_key]
	params = best.get('params', {})
	stop = best.get('stop', None)
	tp_mult = best.get('tp_mult', None)
	exec_cfg = live_params.get('execution', {})

	# Patch last_config.json startup_cfg fields based on best profile
	startup = last_cfg.get('startup_cfg', {})
	# Map stop -> stop_loss_type if present
	if stop in ('tight', 'normal', 'wide'):
		startup['stop_loss_type'] = stop
	# Optional: adjust trading_mode by profile
	if best_key == 'day_trader':
		startup['trading_mode'] = 'scalping'
	elif best_key == 'swing_trader':
		startup['trading_mode'] = 'swing'
	elif best_key == 'hodl_king':
		startup['trading_mode'] = 'position'
	else:
		startup['trading_mode'] = startup.get('trading_mode', 'swing')

	last_cfg['startup_cfg'] = startup

	# Save patched last_config.json
	save_json(last_cfg_path, last_cfg)

	# Write a separate live_runtime_overrides.json for the strategy params and execution thresholds
	overrides = {
		'params': params,
		'tp_mult': tp_mult,
		'execution': exec_cfg,
		'profile': best_key,
	}
	over_path = os.path.join(base, 'live_runtime_overrides.json')
	save_json(over_path, overrides)
	print(f'Updated last_config.json with best profile "{best_key}" and wrote live_runtime_overrides.json')
	return 0


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc() 
    raise SystemExit(main()) 