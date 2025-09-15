#!/usr/bin/env python3
"""
DYNAMIC TOKEN SELECTOR
======================
Intelligent token selection and rotation based on market conditions
"""

import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..analytics.market_scanner import AdvancedMarketScanner, TokenMetrics

@dataclass
class TradingToken:
    """Trading token with position information"""
    symbol: str
    score: float
    position_size: float
    entry_price: float = 0.0
    side: str = ""
    entry_time: datetime = None
    reasoning: str = ""

class DynamicTokenSelector:
    def __init__(self, api_client, max_positions: int = 5, rotation_interval: int = 3600):
        self.api = api_client
        self.scanner = AdvancedMarketScanner(api_client)
        self.max_positions = max_positions
        self.rotation_interval = rotation_interval  # 1 hour default
        
        # Current trading tokens
        self.active_tokens: List[TradingToken] = []
        self.token_history: List[TradingToken] = []
        
        # Rotation tracking
        self.last_rotation = datetime.now()
        self.rotation_thread = None
        self.is_running = False
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        
    def start_rotation(self):
        """Start automatic token rotation"""
        if self.rotation_thread and self.rotation_thread.is_alive():
            return
        
        self.is_running = True
        self.rotation_thread = threading.Thread(target=self._rotation_loop, daemon=True)
        self.rotation_thread.start()
        print("🔄 Token rotation started")
    
    def stop_rotation(self):
        """Stop automatic token rotation"""
        self.is_running = False
        if self.rotation_thread:
            self.rotation_thread.join(timeout=5)
        print("⏹️ Token rotation stopped")
    
    def _rotation_loop(self):
        """Main rotation loop"""
        while self.is_running:
            try:
                self.rotate_tokens()
                time.sleep(self.rotation_interval)
            except Exception as e:
                print(f"❌ Error in token rotation: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def rotate_tokens(self):
        """Rotate tokens based on current market conditions"""
        print(f"\n🔄 Token rotation at {datetime.now().strftime('%H:%M:%S')}")
        
        # Get current recommendations
        recommendations = self.scanner.get_trading_recommendations(self.max_positions * 2)
        
        if not recommendations.get('recommendations'):
            print("⚠️ No trading recommendations available")
            return
        
        # Get current active tokens
        current_symbols = {token.symbol for token in self.active_tokens}
        
        # Find new opportunities
        new_opportunities = []
        for rec in recommendations['recommendations']:
            if rec['symbol'] not in current_symbols:
                new_opportunities.append(rec)
        
        # Check if we should rotate any tokens
        should_rotate = self._should_rotate_tokens(recommendations['recommendations'])
        
        if should_rotate and new_opportunities:
            self._execute_rotation(new_opportunities[:2])  # Rotate max 2 tokens at a time
        else:
            print("✅ Current tokens still optimal")
    
    def _should_rotate_tokens(self, recommendations: List[Dict]) -> bool:
        """Determine if we should rotate tokens"""
        if not self.active_tokens:
            return True
        
        # Check if any current tokens have significantly dropped in ranking
        current_scores = {token.symbol: token.score for token in self.active_tokens}
        
        for rec in recommendations:
            symbol = rec['symbol']
            new_score = rec['score']
            
            if symbol in current_scores:
                old_score = current_scores[symbol]
                score_drop = (old_score - new_score) / old_score if old_score > 0 else 0
                
                # Rotate if score dropped by more than 20%
                if score_drop > 0.2:
                    print(f"🔄 {symbol} score dropped by {score_drop*100:.1f}%, considering rotation")
                    return True
        
        # Check if we've been holding positions too long
        for token in self.active_tokens:
            if token.entry_time:
                holding_time = datetime.now() - token.entry_time
                if holding_time > timedelta(hours=4):  # Rotate after 4 hours
                    print(f"🔄 {token.symbol} held for {holding_time}, considering rotation")
                    return True
        
        return False
    
    def _execute_rotation(self, new_opportunities: List[Dict]):
        """Execute token rotation"""
        print(f"🔄 Executing token rotation with {len(new_opportunities)} new opportunities")
        
        # Close worst performing positions
        if self.active_tokens:
            # Sort by score (worst first)
            self.active_tokens.sort(key=lambda x: x.score)
            
            # Close bottom 25% of positions
            positions_to_close = max(1, len(self.active_tokens) // 4)
            
            for i in range(positions_to_close):
                if i < len(self.active_tokens):
                    token_to_close = self.active_tokens[i]
                    print(f"🔄 Closing {token_to_close.symbol} (score: {token_to_close.score:.3f})")
                    self._close_position(token_to_close)
        
        # Add new opportunities
        for opportunity in new_opportunities:
            if len(self.active_tokens) < self.max_positions:
                new_token = TradingToken(
                    symbol=opportunity['symbol'],
                    score=opportunity['score'],
                    position_size=opportunity['position_size'],
                    reasoning=opportunity['reasoning']
                )
                self.active_tokens.append(new_token)
                print(f"✅ Added {new_token.symbol} (score: {new_token.score:.3f})")
    
    def _close_position(self, token: TradingToken):
        """Close a trading position"""
        try:
            # This would integrate with your order execution system
            print(f"🔄 Closing position for {token.symbol}")
            
            # Remove from active tokens
            if token in self.active_tokens:
                self.active_tokens.remove(token)
            
            # Add to history
            self.token_history.append(token)
            
        except Exception as e:
            print(f"❌ Error closing position for {token.symbol}: {e}")
    
    def get_current_tokens(self) -> List[TradingToken]:
        """Get current active trading tokens"""
        return self.active_tokens.copy()
    
    def get_token_recommendations(self) -> Dict[str, Any]:
        """Get current token recommendations"""
        return self.scanner.get_trading_recommendations(self.max_positions)
    
    def add_position(self, symbol: str, side: str, price: float, size: float):
        """Add a new trading position"""
        # Find the token in active tokens
        for token in self.active_tokens:
            if token.symbol == symbol:
                token.entry_price = price
                token.side = side
                token.entry_time = datetime.now()
                print(f"✅ Position opened: {symbol} {side} at ${price:.4f}")
                return
        
        print(f"⚠️ Token {symbol} not found in active tokens")
    
    def update_position(self, symbol: str, current_price: float):
        """Update position with current price"""
        for token in self.active_tokens:
            if token.symbol == symbol and token.entry_price > 0:
                # Calculate P&L
                if token.side == 'buy':
                    pnl_pct = (current_price - token.entry_price) / token.entry_price
                else:
                    pnl_pct = (token.entry_price - current_price) / token.entry_price
                
                return {
                    'symbol': symbol,
                    'side': token.side,
                    'entry_price': token.entry_price,
                    'current_price': current_price,
                    'pnl_pct': pnl_pct,
                    'score': token.score
                }
        
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'active_tokens': len(self.active_tokens),
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_profit': self.total_profit,
            'last_rotation': self.last_rotation.isoformat(),
            'active_tokens_list': [
                {
                    'symbol': token.symbol,
                    'score': token.score,
                    'side': token.side,
                    'entry_price': token.entry_price,
                    'reasoning': token.reasoning
                }
                for token in self.active_tokens
            ]
        }
    
    def print_status(self):
        """Print current status"""
        print(f"\n📊 DYNAMIC TOKEN SELECTOR STATUS")
        print("=" * 50)
        print(f"Active Tokens: {len(self.active_tokens)}/{self.max_positions}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Success Rate: {self.successful_trades/self.total_trades*100:.1f}%" if self.total_trades > 0 else "N/A")
        print(f"Total Profit: ${self.total_profit:.4f}")
        print(f"Last Rotation: {self.last_rotation.strftime('%H:%M:%S')}")
        
        if self.active_tokens:
            print(f"\n🏆 ACTIVE TOKENS:")
            for token in self.active_tokens:
                print(f"  {token.symbol:8s} | Score: {token.score:.3f} | "
                      f"Side: {token.side:4s} | "
                      f"Entry: ${token.entry_price:.4f} | "
                      f"Reason: {token.reasoning}") 