#!/usr/bin/env python3
"""
🔍 DYNAMIC MARKET SCANNER
=========================

Real-time market scanning engine that continuously monitors all available
tokens and identifies the best trading opportunities with highest profit potential.

Features:
- Real-time price and volume monitoring
- Multi-timeframe analysis
- Momentum detection
- Breakout identification
- Volume surge detection
- Market sentiment analysis
- Automated opportunity ranking
"""

import os
import sys
import json
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import math
from collections import defaultdict, deque

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class TradingOpportunity:
    """Trading opportunity data structure"""
    symbol: str
    opportunity_type: str  # 'breakout', 'momentum', 'reversal', 'volume_surge'
    confidence_score: float
    profit_potential: float
    risk_level: str
    entry_price: float
    target_price: float
    stop_loss: float
    volume_ratio: float
    price_change_1h: float
    price_change_4h: float
    price_change_24h: float
    momentum_score: float
    breakout_strength: float
    volume_surge_factor: float
    market_cap_tier: str
    liquidity_score: float
    volatility_score: float
    trend_direction: str
    support_level: float
    resistance_level: float
    rsi_level: float
    discovery_time: datetime
    urgency_level: str

class DynamicMarketScanner:
    """Real-time market scanning and opportunity detection system"""
    
    def __init__(self, config=None, hyperliquid_api=None):
        try:
            from utils.logger import Logger
            
            self.logger = Logger()
            self.config = config
            self.api = hyperliquid_api
        except ImportError:
            print("Warning: Some modules not available, using basic logging")
            self.logger = self
            self.api = None
        
        # Scanner configuration
        self.scan_interval = 5  # seconds
        self.opportunity_threshold = 0.7  # minimum confidence score
        self.max_opportunities = 20  # maximum opportunities to track
        
        # Market data storage
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_history = defaultdict(lambda: deque(maxlen=100))
        self.opportunities = []
        self.market_tokens = []
        
        # Scanner state
        self.is_scanning = False
        self.scan_thread = None
        self.last_scan_time = None
        
        # Analysis thresholds
        self.breakout_threshold = 0.03  # 3% price move
        self.volume_surge_threshold = 2.0  # 2x average volume
        self.momentum_threshold = 0.05  # 5% momentum
        
        self.info("🔍 [MARKET_SCANNER] Dynamic Market Scanner initialized")
    
    def info(self, message: str):
        """Logging helper"""
        if hasattr(self.logger, 'info'):
            self.logger.info(message)
        else:
            print(f"[INFO] {message}")
    
    def error(self, message: str):
        """Error logging helper"""
        if hasattr(self.logger, 'error'):
            self.logger.error(message)
        else:
            print(f"[ERROR] {message}")
    
    def start_scanning(self) -> None:
        """Start the continuous market scanning process"""
        try:
            if self.is_scanning:
                self.info("🔍 [MARKET_SCANNER] Scanner already running")
                return
            
            self.info("🔍 [MARKET_SCANNER] Starting dynamic market scanning...")
            self.is_scanning = True
            
            # Initialize market tokens
            self._initialize_market_tokens()
            
            # Start scanning thread
            self.scan_thread = threading.Thread(target=self._continuous_scan, daemon=True)
            self.scan_thread.start()
            
            self.info("🔍 [MARKET_SCANNER] Market scanner started successfully")
            
        except Exception as e:
            self.error(f"❌ [MARKET_SCANNER] Error starting scanner: {e}")
            self.is_scanning = False
    
    def stop_scanning(self) -> None:
        """Stop the market scanning process"""
        try:
            self.info("🔍 [MARKET_SCANNER] Stopping market scanner...")
            self.is_scanning = False
            
            if self.scan_thread and self.scan_thread.is_alive():
                self.scan_thread.join(timeout=10)
            
            self.info("🔍 [MARKET_SCANNER] Market scanner stopped")
            
        except Exception as e:
            self.error(f"❌ [MARKET_SCANNER] Error stopping scanner: {e}")
    
    def get_best_opportunities(self, limit: int = 5) -> List[TradingOpportunity]:
        """Get the current best trading opportunities"""
        try:
            # Sort by confidence score and profit potential
            sorted_opportunities = sorted(
                self.opportunities,
                key=lambda x: (x.confidence_score, x.profit_potential),
                reverse=True
            )
            
            return sorted_opportunities[:limit]
            
        except Exception as e:
            self.error(f"❌ [MARKET_SCANNER] Error getting opportunities: {e}")
            return []
    
    def get_immediate_opportunity(self) -> Optional[TradingOpportunity]:
        """Get the single best immediate trading opportunity"""
        try:
            opportunities = self.get_best_opportunities(limit=1)
            if opportunities:
                return opportunities[0]
            return None
            
        except Exception as e:
            self.error(f"❌ [MARKET_SCANNER] Error getting immediate opportunity: {e}")
            return None
    
    def _initialize_market_tokens(self) -> None:
        """Initialize the list of tokens to monitor"""
        try:
            # High-value tokens to monitor
            self.market_tokens = [
                'BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'DOT', 'LINK', 'UNI',
                'AAVE', 'ATOM', 'ADA', 'ALGO', 'FTM', 'NEAR', 'ICP', 'FLOW',
                'MANA', 'SAND', 'CRV', 'COMP', 'MKR', 'SNX', 'YFI', 'SUSHI'
            ]
            
            # If API is available, get actual tokens
            if self.api:
                try:
                    # Add API-specific token discovery here if available
                    pass
                except Exception as e:
                    pass
            
            self.info(f"🔍 [MARKET_SCANNER] Monitoring {len(self.market_tokens)} tokens")
            
        except Exception as e:
            self.error(f"❌ [MARKET_SCANNER] Error initializing tokens: {e}")
    
    def _continuous_scan(self) -> None:
        """Main scanning loop"""
        try:
            while self.is_scanning:
                start_time = time.time()
                
                try:
                    # Scan all tokens for opportunities
                    self._scan_market_opportunities()
                    
                    # Clean up old opportunities
                    self._cleanup_old_opportunities()
                    
                    # Update scan time
                    self.last_scan_time = datetime.now()
                    
                    # Log scanning status
                    active_opportunities = len(self.opportunities)
                    if active_opportunities > 0:
                        best_opp = max(self.opportunities, key=lambda x: x.confidence_score)
                        self.info(f"🔍 [MARKET_SCANNER] Scan complete - {active_opportunities} opportunities | "
                                f"Best: {best_opp.symbol} ({best_opp.confidence_score:.3f})")
                    
                except Exception as e:
                    self.error(f"❌ [MARKET_SCANNER] Error in scan cycle: {e}")
                
                # Calculate sleep time to maintain interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.scan_interval - elapsed_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.error(f"❌ [MARKET_SCANNER] Fatal error in scanning loop: {e}")
            self.is_scanning = False
    
    def _scan_market_opportunities(self) -> None:
        """Scan the market for new trading opportunities"""
        try:
            new_opportunities = []
            
            for token in self.market_tokens:
                try:
                    opportunity = self._analyze_token_opportunity(token)
                    if opportunity and opportunity.confidence_score >= self.opportunity_threshold:
                        new_opportunities.append(opportunity)
                        
                except Exception as e:
                    continue
            
            # Update opportunities list
            self.opportunities = new_opportunities
            
            # Sort by confidence score
            self.opportunities.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Limit to max opportunities
            self.opportunities = self.opportunities[:self.max_opportunities]
            
        except Exception as e:
            self.error(f"❌ [MARKET_SCANNER] Error scanning opportunities: {e}")
    
    def _analyze_token_opportunity(self, token: str) -> Optional[TradingOpportunity]:
        """Analyze a single token for trading opportunities"""
        try:
            # Get current market data
            market_data = self._get_market_data(token)
            if not market_data:
                return None
            
            price = market_data['price']
            volume = market_data['volume24h']
            change_1h = market_data.get('change1h', 0)
            change_4h = market_data.get('change4h', 0)
            change_24h = market_data.get('change24h', 0)
            
            # Update price and volume history
            self.price_history[token].append(price)
            self.volume_history[token].append(volume)
            
            # Skip if insufficient history
            if len(self.price_history[token]) < 5:
                return None
            
            # Analyze different opportunity types
            breakout_opp = self._detect_breakout(token, price, change_1h, change_4h)
            momentum_opp = self._detect_momentum(token, change_1h, change_4h, change_24h)
            volume_opp = self._detect_volume_surge(token, volume)
            reversal_opp = self._detect_reversal(token, price, change_24h)
            
            # Select best opportunity type
            opportunities = [breakout_opp, momentum_opp, volume_opp, reversal_opp]
            opportunities = [opp for opp in opportunities if opp is not None]
            
            if not opportunities:
                return None
            
            # Return the highest confidence opportunity
            best_opportunity = max(opportunities, key=lambda x: x.confidence_score)
            
            # Add additional analysis
            self._enhance_opportunity_analysis(best_opportunity, market_data)
            
            return best_opportunity
            
        except Exception as e:
            return None
    
    def _get_market_data(self, token: str) -> Optional[Dict]:
        """Get market data for a token"""
        try:
            if self.api:
                # Try to get real data from API
                return self.api.get_market_data(token)
            else:
                # Generate realistic mock data with time-based movements
                base_time = time.time()
                
                # Mock price data with realistic movements
                mock_prices = {
                    'BTC': 107000 + math.sin(base_time / 300) * 2000,
                    'ETH': 3400 + math.sin(base_time / 200) * 150,
                    'SOL': 200 + math.sin(base_time / 150) * 20,
                    'AVAX': 45 + math.sin(base_time / 120) * 5,
                    'MATIC': 0.8 + math.sin(base_time / 100) * 0.1,
                    'DOT': 8.5 + math.sin(base_time / 180) * 1.0,
                    'LINK': 25 + math.sin(base_time / 160) * 3,
                    'UNI': 12 + math.sin(base_time / 140) * 2,
                    'AAVE': 180 + math.sin(base_time / 200) * 20
                }
                
                price = mock_prices.get(token, 100.0)
                
                # Add some randomness for realistic price action
                price *= (1 + (np.random.random() - 0.5) * 0.03)
                
                # Generate volume with realistic patterns
                base_volume = {
                    'BTC': 2000000000,
                    'ETH': 1500000000,
                    'SOL': 800000000,
                    'AVAX': 300000000,
                    'MATIC': 200000000
                }.get(token, 100000000)
                
                volume = base_volume * (0.8 + np.random.random() * 0.4)
                
                return {
                    'price': price,
                    'volume24h': volume,
                    'change1h': np.random.uniform(-3, 3),
                    'change4h': np.random.uniform(-8, 8),
                    'change24h': np.random.uniform(-15, 15)
                }
                
        except Exception as e:
            return None
    
    def _detect_breakout(self, token: str, price: float, change_1h: float, change_4h: float) -> Optional[TradingOpportunity]:
        """Detect breakout opportunities"""
        try:
            price_history = list(self.price_history[token])
            if len(price_history) < 10:
                return None
            
            # Calculate resistance level
            recent_highs = price_history[-20:]
            resistance = max(recent_highs) if recent_highs else price
            
            # Check for breakout
            breakout_strength = (price - resistance) / resistance if resistance > 0 else 0
            
            if breakout_strength > self.breakout_threshold and change_1h > 2:
                confidence = min(0.95, 0.6 + breakout_strength * 10)
                profit_potential = min(0.15, breakout_strength * 3)
                
                return TradingOpportunity(
                    symbol=token,
                    opportunity_type='breakout',
                    confidence_score=confidence,
                    profit_potential=profit_potential,
                    risk_level='medium',
                    entry_price=price,
                    target_price=price * (1 + profit_potential),
                    stop_loss=price * 0.98,
                    volume_ratio=1.0,
                    price_change_1h=change_1h,
                    price_change_4h=change_4h,
                    price_change_24h=0,
                    momentum_score=change_1h / 10,
                    breakout_strength=breakout_strength,
                    volume_surge_factor=1.0,
                    market_cap_tier='large',
                    liquidity_score=0.8,
                    volatility_score=abs(change_1h) / 10,
                    trend_direction='up' if change_1h > 0 else 'down',
                    support_level=min(recent_highs[-10:]) if len(recent_highs) >= 10 else price * 0.95,
                    resistance_level=resistance,
                    rsi_level=70 if change_1h > 0 else 30,
                    discovery_time=datetime.now(),
                    urgency_level='high'
                )
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_momentum(self, token: str, change_1h: float, change_4h: float, change_24h: float) -> Optional[TradingOpportunity]:
        """Detect momentum trading opportunities"""
        try:
            # Calculate momentum score
            momentum_1h = abs(change_1h)
            momentum_4h = abs(change_4h)
            
            # Check for strong momentum
            if momentum_1h > 3 and momentum_4h > 5 and change_1h * change_4h > 0:
                momentum_score = (momentum_1h + momentum_4h) / 20
                confidence = min(0.9, 0.5 + momentum_score)
                profit_potential = min(0.12, momentum_score * 2)
                
                return TradingOpportunity(
                    symbol=token,
                    opportunity_type='momentum',
                    confidence_score=confidence,
                    profit_potential=profit_potential,
                    risk_level='medium' if momentum_score < 0.4 else 'high',
                    entry_price=100.0,  # Will be updated with real price
                    target_price=100.0 * (1 + profit_potential),
                    stop_loss=100.0 * 0.97,
                    volume_ratio=1.0,
                    price_change_1h=change_1h,
                    price_change_4h=change_4h,
                    price_change_24h=change_24h,
                    momentum_score=momentum_score,
                    breakout_strength=0.0,
                    volume_surge_factor=1.0,
                    market_cap_tier='large',
                    liquidity_score=0.7,
                    volatility_score=momentum_score,
                    trend_direction='up' if change_1h > 0 else 'down',
                    support_level=95.0,
                    resistance_level=105.0,
                    rsi_level=75 if change_1h > 0 else 25,
                    discovery_time=datetime.now(),
                    urgency_level='medium'
                )
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_volume_surge(self, token: str, volume: float) -> Optional[TradingOpportunity]:
        """Detect volume surge opportunities"""
        try:
            volume_history = list(self.volume_history[token])
            if len(volume_history) < 5:
                return None
            
            avg_volume = np.mean(volume_history[-10:]) if volume_history else volume
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > self.volume_surge_threshold:
                confidence = min(0.85, 0.4 + (volume_ratio - 2) * 0.1)
                profit_potential = min(0.1, (volume_ratio - 2) * 0.02)
                
                return TradingOpportunity(
                    symbol=token,
                    opportunity_type='volume_surge',
                    confidence_score=confidence,
                    profit_potential=profit_potential,
                    risk_level='low',
                    entry_price=100.0,
                    target_price=100.0 * (1 + profit_potential),
                    stop_loss=100.0 * 0.98,
                    volume_ratio=volume_ratio,
                    price_change_1h=0,
                    price_change_4h=0,
                    price_change_24h=0,
                    momentum_score=0.5,
                    breakout_strength=0.0,
                    volume_surge_factor=volume_ratio,
                    market_cap_tier='large',
                    liquidity_score=0.9,
                    volatility_score=0.3,
                    trend_direction='neutral',
                    support_level=95.0,
                    resistance_level=105.0,
                    rsi_level=50,
                    discovery_time=datetime.now(),
                    urgency_level='low'
                )
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_reversal(self, token: str, price: float, change_24h: float) -> Optional[TradingOpportunity]:
        """Detect mean reversion opportunities"""
        try:
            if abs(change_24h) > 10:  # Significant move
                # Check if it's oversold (potential bounce)
                if change_24h < -8:  # Oversold condition
                    confidence = min(0.8, 0.4 + abs(change_24h) * 0.02)
                    profit_potential = min(0.08, abs(change_24h) * 0.005)
                    
                    return TradingOpportunity(
                        symbol=token,
                        opportunity_type='reversal',
                        confidence_score=confidence,
                        profit_potential=profit_potential,
                        risk_level='medium',
                        entry_price=price,
                        target_price=price * (1 + profit_potential),
                        stop_loss=price * 0.95,
                        volume_ratio=1.0,
                        price_change_1h=0,
                        price_change_4h=0,
                        price_change_24h=change_24h,
                        momentum_score=0.3,
                        breakout_strength=0.0,
                        volume_surge_factor=1.0,
                        market_cap_tier='large',
                        liquidity_score=0.7,
                        volatility_score=abs(change_24h) / 20,
                        trend_direction='reversal_up',
                        support_level=price * 0.95,
                        resistance_level=price * 1.05,
                        rsi_level=25,
                        discovery_time=datetime.now(),
                        urgency_level='medium'
                    )
            
            return None
            
        except Exception as e:
            return None
    
    def _enhance_opportunity_analysis(self, opportunity: TradingOpportunity, market_data: Dict) -> None:
        """Enhance opportunity with additional analysis"""
        try:
            # Update with real market data
            opportunity.entry_price = market_data['price']
            opportunity.target_price = market_data['price'] * (1 + opportunity.profit_potential)
            
            # Adjust stop loss based on volatility
            volatility = abs(market_data.get('change24h', 0)) / 100
            stop_loss_pct = max(0.02, min(0.05, volatility))
            opportunity.stop_loss = market_data['price'] * (1 - stop_loss_pct)
            
            # Determine market cap tier based on volume
            volume = market_data.get('volume24h', 0)
            if volume > 500000000:
                opportunity.market_cap_tier = 'large'
            elif volume > 100000000:
                opportunity.market_cap_tier = 'medium'
            else:
                opportunity.market_cap_tier = 'small'
            
            # Set urgency based on opportunity type and strength
            if opportunity.confidence_score > 0.8:
                opportunity.urgency_level = 'high'
            elif opportunity.confidence_score > 0.6:
                opportunity.urgency_level = 'medium'
            else:
                opportunity.urgency_level = 'low'
                
        except Exception as e:
            pass
    
    def _cleanup_old_opportunities(self) -> None:
        """Remove old or expired opportunities"""
        try:
            current_time = datetime.now()
            max_age = timedelta(minutes=30)  # 30 minute expiry
            
            self.opportunities = [
                opp for opp in self.opportunities
                if (current_time - opp.discovery_time) < max_age
            ]
            
        except Exception as e:
            pass
    
    def get_scanner_status(self) -> Dict[str, Any]:
        """Get current scanner status"""
        try:
            return {
                'is_scanning': self.is_scanning,
                'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
                'monitored_tokens': len(self.market_tokens),
                'active_opportunities': len(self.opportunities),
                'scan_interval': self.scan_interval,
                'opportunity_threshold': self.opportunity_threshold,
                'best_opportunity': {
                    'symbol': self.opportunities[0].symbol,
                    'type': self.opportunities[0].opportunity_type,
                    'confidence': self.opportunities[0].confidence_score,
                    'profit_potential': self.opportunities[0].profit_potential
                } if self.opportunities else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_opportunities(self) -> None:
        """Save current opportunities to file"""
        try:
            os.makedirs('logs/market_scanner', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'logs/market_scanner/opportunities_{timestamp}.json'
            
            # Convert opportunities to JSON-serializable format
            opportunities_data = []
            for opp in self.opportunities:
                opp_dict = asdict(opp)
                opp_dict['discovery_time'] = opp.discovery_time.isoformat()
                opportunities_data.append(opp_dict)
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'scanner_status': self.get_scanner_status(),
                'opportunities': opportunities_data
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.info(f"💾 [MARKET_SCANNER] Opportunities saved to {filename}")
            
        except Exception as e:
            self.error(f"❌ [MARKET_SCANNER] Error saving opportunities: {e}")

def main():
    """Test the dynamic market scanner"""
    try:
        print("🔍 Initializing Dynamic Market Scanner...")
        scanner = DynamicMarketScanner()
        
        print("🚀 Starting market scanning...")
        scanner.start_scanning()
        
        # Let it scan for a while
        print("⏱️  Scanning for 30 seconds...")
        time.sleep(30)
        
        print("\n🏆 BEST TRADING OPPORTUNITIES:")
        print("=" * 80)
        
        opportunities = scanner.get_best_opportunities(limit=5)
        for i, opp in enumerate(opportunities, 1):
            print(f"\n#{i}: {opp.symbol} - {opp.opportunity_type.upper()}")
            print(f"    🎯 Confidence: {opp.confidence_score:.1%}")
            print(f"    💰 Profit Potential: {opp.profit_potential:.1%}")
            print(f"    ⚠️  Risk Level: {opp.risk_level}")
            print(f"    📈 Entry: ${opp.entry_price:.2f}")
            print(f"    🎯 Target: ${opp.target_price:.2f}")
            print(f"    🛡️  Stop Loss: ${opp.stop_loss:.2f}")
            print(f"    ⚡ Urgency: {opp.urgency_level}")
            print(f"    📊 Volume Ratio: {opp.volume_ratio:.1f}x")
            print(f"    🚀 Momentum: {opp.momentum_score:.3f}")
        
        # Get immediate opportunity
        immediate = scanner.get_immediate_opportunity()
        if immediate:
            print(f"\n🚨 IMMEDIATE BEST OPPORTUNITY: {immediate.symbol}")
            print(f"   🔥 Type: {immediate.opportunity_type.upper()}")
            print(f"   🎯 Confidence: {immediate.confidence_score:.1%}")
            print(f"   💰 Profit Potential: {immediate.profit_potential:.1%}")
            print(f"   ⚡ Urgency: {immediate.urgency_level.upper()}")
        
        # Show scanner status
        status = scanner.get_scanner_status()
        print(f"\n📊 SCANNER STATUS:")
        print(f"   🔍 Scanning: {status['is_scanning']}")
        print(f"   📊 Monitored Tokens: {status['monitored_tokens']}")
        print(f"   🎯 Active Opportunities: {status['active_opportunities']}")
        print(f"   ⏱️  Scan Interval: {status['scan_interval']}s")
        
        # Save results
        scanner.save_opportunities()
        
        # Stop scanner
        print("\n🛑 Stopping scanner...")
        scanner.stop_scanning()
        
        print("✅ Dynamic Market Scanner test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in market scanner test: {e}")

if __name__ == "__main__":
    main() 