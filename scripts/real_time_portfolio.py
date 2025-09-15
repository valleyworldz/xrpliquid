#!/usr/bin/env python3
"""
REAL-TIME PORTFOLIO MONITOR
Monitors portfolio status and trading performance
"""

import os
import sys
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load credentials from .env (no hardcoded wallet)
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)

# Map primary vars to aliases used by core API if present
addr = os.getenv('WALLET_ADDRESS') or os.getenv('HYPERLIQUID_ADDRESS') or os.getenv('HL_ADDRESS')
pk = os.getenv('PRIVATE_KEY') or os.getenv('HYPERLIQUID_PRIVATE_KEY') or os.getenv('HL_PRIVATE_KEY')
if addr:
    os.environ['HYPERLIQUID_ADDRESS'] = addr
if pk:
    # Some paths expect raw hex without 0x; keep both to be safe
    os.environ['HYPERLIQUID_PRIVATE_KEY'] = pk[2:] if pk.startswith('0x') and len(pk) == 66 else pk

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.api.hyperliquid_api import HyperliquidAPI

def safe_float_convert(value, default=0.0):
    """Safely convert value to float"""
    try:
        if isinstance(value, dict):
            # If it's a dict, try to extract the value
            if 'value' in value:
                return float(value['value'])
            else:
                return default
        else:
            return float(value)
    except (ValueError, TypeError):
        return default

def get_portfolio_status():
    """Get current portfolio status"""
    try:
        api = HyperliquidAPI(testnet=False)
        user_state = api.get_user_state()
        
        if not user_state:
            return None
        
        margin_summary = user_state.get('marginSummary', {})
        
        # Safely extract values
        account_value = safe_float_convert(margin_summary.get('accountValue', 0))
        free_collateral = safe_float_convert(margin_summary.get('freeCollateral', 0))
        used_collateral = safe_float_convert(margin_summary.get('usedCollateral', 0))
        
        # Calculate margin ratio safely
        margin_ratio = 0.0
        if account_value > 0:
            margin_ratio = used_collateral / account_value
        
        # Calculate utilization
        utilization = 0.0
        if account_value > 0:
            utilization = (used_collateral / account_value) * 100
        
        return {
            'account_value': account_value,
            'free_collateral': free_collateral,
            'used_collateral': used_collateral,
            'margin_ratio': margin_ratio,
            'utilization': utilization,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"Error getting portfolio status: {e}")
        return None

def display_portfolio_status(status):
    """Display portfolio status"""
    if not status:
        print("Unable to retrieve portfolio status")
        return
    
    print("ULTIMATE MASTER BOT - REAL-TIME PORTFOLIO MONITOR")
    print("="*70)
    print(f"Last Updated: {status['timestamp']}")
    print("="*70)
    print("ACCOUNT OVERVIEW")
    print("-" * 40)
    print(f"Account Value:     $      {status['account_value']:8.2f}")
    print(f"Available Margin:  $      {status['free_collateral']:8.2f}")
    print(f"Used Margin:       $      {status['used_collateral']:8.2f}")
    print(f"Margin Ratio:            {status['margin_ratio']:8.4f}")
    print(f"Utilization:            {status['utilization']:8.2f}%")
    
    # Check for active positions
    try:
        api = HyperliquidAPI(testnet=False)
        positions = api.get_positions()
        
        if positions and len(positions) > 0:
            print("\nACTIVE POSITIONS")
            print("-" * 40)
            for pos in positions:
                if isinstance(pos, dict):
                    symbol = pos.get('symbol', 'Unknown')
                    size = pos.get('size', 0)
                    if abs(float(size)) > 0:
                        print(f"{symbol}: {size}")
        else:
            print("\nACTIVE POSITIONS (0)")
            print("-" * 40)
            print("No active positions")
            
    except Exception as e:
        print(f"\nError checking positions: {e}")
    
    print("="*70)
    print("Auto-refreshing every 10 seconds...")
    print("Press Ctrl+C to stop")

def main():
    """Main monitoring loop"""
    print("Starting Real-Time Portfolio Monitor...")
    print("="*50)
    
    try:
        while True:
            status = get_portfolio_status()
            display_portfolio_status(status)
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nPortfolio monitor stopped by user")
    except Exception as e:
        print(f"Error in portfolio monitor: {e}")

if __name__ == "__main__":
    main() 