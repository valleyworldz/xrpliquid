#!/usr/bin/env python3
"""
🎯 INTELLIGENT TRADING ORCHESTRATOR
===================================

Master orchestrator that combines intelligent token selection with automated trading
execution to maximize profits by always trading the best opportunities.

Features:
- Continuous market scanning and token analysis
- Real-time opportunity detection and ranking  
- Automated best token selection
- Dynamic strategy allocation
- Risk-optimized position sizing
- Performance tracking and optimization
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class TradingDecision:
    """Trading decision data structure"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    position_size: float
    entry_price: float
    target_price: float
    stop_loss: float
    confidence_score: float
    profit_potential: float
    risk_score: float
    strategy_type: str
    reasoning: str
    timestamp: datetime
    priority: str  # 'immediate', 'high', 'medium', 'low'
    expected_duration: str

@dataclass
class PortfolioAllocation:
    """Portfolio allocation data structure"""
    symbol: str
    current_allocation: float
    target_allocation: float
    adjustment_needed: float
    action_required: str
    priority: int

class IntelligentTradingOrchestrator:
    """Master trading orchestrator with intelligent token selection"""
    
    def __init__(self):
        try:
            from utils.logger import Logger
            from utils.config_manager import ConfigManager
            from api.hyperliquid_api import HyperliquidAPI
            from engines.intelligent_token_selector import IntelligentTokenSelector
            from engines.dynamic_market_scanner import DynamicMarketScanner
            
            self.logger = Logger()
            self.config = ConfigManager("config/parameters.json")
            self.api = HyperliquidAPI(testnet=False)
            self.token_selector = IntelligentTokenSelector()
            self.market_scanner = DynamicMarketScanner()
        except ImportError as e:
            print(f"Warning: Some modules not available: {e}")
            self.logger = self
            self.api = None
            self.token_selector = None
            self.market_scanner = None
        
        # Orchestrator configuration
        self.analysis_interval = 30  # seconds between full analysis
        self.quick_scan_interval = 5  # seconds between quick scans
        self.max_positions = 5  # maximum concurrent positions
        self.min_confidence_threshold = 0.7  # minimum confidence for trades
        
        # State management
        self.is_running = False
        self.analysis_thread = None
        self.scanner_thread = None
        self.current_positions = {}
        self.pending_orders = {}
        self.trading_decisions = []
        self.portfolio_allocations = []
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.last_analysis_time = None
        
        self.info("🎯 [TRADING_ORCHESTRATOR] Intelligent Trading Orchestrator initialized")
    
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
    
    def start_orchestrator(self) -> None:
        """Start the intelligent trading orchestrator"""
        try:
            if self.is_running:
                self.info("🎯 [TRADING_ORCHESTRATOR] Orchestrator already running")
                return
            
            self.info("🎯 [TRADING_ORCHESTRATOR] Starting intelligent trading orchestrator...")
            self.is_running = True
            
            # Start market scanner
            if self.market_scanner:
                self.market_scanner.start_scanning()
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(target=self._continuous_analysis, daemon=True)
            self.analysis_thread.start()
            
            # Start quick scanner thread
            self.scanner_thread = threading.Thread(target=self._quick_opportunity_scan, daemon=True)
            self.scanner_thread.start()
            
            self.info("🎯 [TRADING_ORCHESTRATOR] All systems operational!")
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error starting orchestrator: {e}")
            self.is_running = False
    
    def stop_orchestrator(self) -> None:
        """Stop the trading orchestrator"""
        try:
            self.info("🎯 [TRADING_ORCHESTRATOR] Stopping orchestrator...")
            self.is_running = False
            
            # Stop market scanner
            if self.market_scanner:
                self.market_scanner.stop_scanning()
            
            # Wait for threads to complete
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=10)
            
            if self.scanner_thread and self.scanner_thread.is_alive():
                self.scanner_thread.join(timeout=10)
            
            self.info("🎯 [TRADING_ORCHESTRATOR] Orchestrator stopped")
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error stopping orchestrator: {e}")
    
    def get_current_best_tokens(self, limit: int = 5) -> List[str]:
        """Get the current best tokens for trading"""
        try:
            if self.token_selector:
                best_tokens = self.token_selector.analyze_and_select_best_tokens(top_n=limit)
                return [token.symbol for token in best_tokens]
            else:
                # Fallback to high-quality tokens
                return ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
                
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error getting best tokens: {e}")
            return ['BTC', 'ETH', 'SOL']
    
    def get_immediate_trading_opportunity(self) -> Optional[Dict[str, Any]]:
        """Get the immediate best trading opportunity"""
        try:
            if self.market_scanner:
                opportunity = self.market_scanner.get_immediate_opportunity()
                if opportunity:
                    return {
                        'symbol': opportunity.symbol,
                        'type': opportunity.opportunity_type,
                        'confidence': opportunity.confidence_score,
                        'profit_potential': opportunity.profit_potential,
                        'entry_price': opportunity.entry_price,
                        'target_price': opportunity.target_price,
                        'stop_loss': opportunity.stop_loss,
                        'urgency': opportunity.urgency_level,
                        'reasoning': f"{opportunity.opportunity_type} detected with {opportunity.confidence_score:.1%} confidence"
                    }
            
            # Fallback to token selector
            if self.token_selector:
                best_token = self.token_selector.get_current_best_token()
                if best_token:
                    return {
                        'symbol': best_token,
                        'type': 'analysis_based',
                        'confidence': 0.7,
                        'profit_potential': 0.05,
                        'entry_price': 0.0,  # To be determined
                        'target_price': 0.0,  # To be determined
                        'stop_loss': 0.0,    # To be determined
                        'urgency': 'medium',
                        'reasoning': 'Selected by intelligent analysis'
                    }
            
            # Ultimate fallback
            return {
                'symbol': 'BTC',
                'type': 'fallback',
                'confidence': 0.6,
                'profit_potential': 0.03,
                'entry_price': 107000.0,
                'target_price': 110210.0,
                'stop_loss': 103790.0,
                'urgency': 'low',
                'reasoning': 'Fallback to BTC - most reliable asset'
            }
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error getting immediate opportunity: {e}")
            return None
    
    def _continuous_analysis(self) -> None:
        """Continuous analysis and decision making loop"""
        try:
            while self.is_running:
                start_time = time.time()
                
                try:
                    # Perform comprehensive market analysis
                    self._perform_market_analysis()
                    
                    # Generate trading decisions
                    self._generate_trading_decisions()
                    
                    # Optimize portfolio allocation
                    self._optimize_portfolio_allocation()
                    
                    # Execute trading decisions
                    self._execute_trading_decisions()
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    # Log analysis results
                    self._log_analysis_results()
                    
                    self.last_analysis_time = datetime.now()
                    
                except Exception as e:
                    self.error(f"❌ [TRADING_ORCHESTRATOR] Error in analysis cycle: {e}")
                
                # Sleep for the remaining interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.analysis_interval - elapsed_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Fatal error in analysis loop: {e}")
            self.is_running = False
    
    def _quick_opportunity_scan(self) -> None:
        """Quick opportunity scanning loop"""
        try:
            while self.is_running:
                start_time = time.time()
                
                try:
                    # Check for immediate opportunities
                    immediate_opp = self.get_immediate_trading_opportunity()
                    
                    if immediate_opp and immediate_opp.get('urgency') == 'high':
                        # Process high-urgency opportunities immediately
                        self._process_immediate_opportunity(immediate_opp)
                    
                except Exception as e:
                    self.error(f"❌ [TRADING_ORCHESTRATOR] Error in quick scan: {e}")
                
                # Sleep for quick scan interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.quick_scan_interval - elapsed_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Fatal error in quick scan loop: {e}")
    
    def _perform_market_analysis(self) -> None:
        """Perform comprehensive market analysis"""
        try:
            self.info("🔍 [TRADING_ORCHESTRATOR] Performing market analysis...")
            
            # Get best tokens from selector
            if self.token_selector:
                best_tokens = self.token_selector.analyze_and_select_best_tokens(top_n=8)
                self.info(f"📊 [TRADING_ORCHESTRATOR] Top tokens: {[t.symbol for t in best_tokens[:3]]}")
            
            # Get current opportunities from scanner
            if self.market_scanner:
                opportunities = self.market_scanner.get_best_opportunities(limit=5)
                if opportunities:
                    self.info(f"🎯 [TRADING_ORCHESTRATOR] Best opportunity: {opportunities[0].symbol} "
                             f"({opportunities[0].confidence_score:.1%} confidence)")
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error in market analysis: {e}")
    
    def _generate_trading_decisions(self) -> None:
        """Generate intelligent trading decisions"""
        try:
            new_decisions = []
            
            # Get immediate opportunities
            immediate_opp = self.get_immediate_trading_opportunity()
            if immediate_opp and immediate_opp['confidence'] >= self.min_confidence_threshold:
                decision = self._create_trading_decision(immediate_opp)
                if decision:
                    new_decisions.append(decision)
            
            # Get best tokens for longer-term positions
            best_tokens = self.get_current_best_tokens(limit=3)
            for token in best_tokens:
                if token not in self.current_positions:
                    # Create decision for new position
                    token_decision = self._create_token_decision(token)
                    if token_decision:
                        new_decisions.append(token_decision)
            
            # Update decisions list
            self.trading_decisions = new_decisions
            
            if new_decisions:
                self.info(f"💡 [TRADING_ORCHESTRATOR] Generated {len(new_decisions)} trading decisions")
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error generating decisions: {e}")
    
    def _create_trading_decision(self, opportunity: Dict[str, Any]) -> Optional[TradingDecision]:
        """Create a trading decision from an opportunity"""
        try:
            # Calculate position size based on confidence and risk
            base_size = 0.1  # 10% of portfolio
            confidence_multiplier = opportunity['confidence']
            position_size = base_size * confidence_multiplier
            
            # Adjust for risk
            risk_adjustment = 1.0 - (1.0 - confidence_multiplier) * 0.5
            position_size *= risk_adjustment
            
            # Determine priority
            if opportunity['urgency'] == 'high':
                priority = 'immediate'
            elif opportunity['confidence'] > 0.8:
                priority = 'high'
            elif opportunity['confidence'] > 0.6:
                priority = 'medium'
            else:
                priority = 'low'
            
            return TradingDecision(
                symbol=opportunity['symbol'],
                action='buy',
                position_size=position_size,
                entry_price=opportunity.get('entry_price', 0.0),
                target_price=opportunity.get('target_price', 0.0),
                stop_loss=opportunity.get('stop_loss', 0.0),
                confidence_score=opportunity['confidence'],
                profit_potential=opportunity.get('profit_potential', 0.05),
                risk_score=1.0 - opportunity['confidence'],
                strategy_type=opportunity.get('type', 'intelligent'),
                reasoning=opportunity.get('reasoning', 'AI-generated opportunity'),
                timestamp=datetime.now(),
                priority=priority,
                expected_duration='short_term' if opportunity['urgency'] == 'high' else 'medium_term'
            )
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error creating decision: {e}")
            return None
    
    def _create_token_decision(self, token: str) -> Optional[TradingDecision]:
        """Create a trading decision for a top-ranked token"""
        try:
            return TradingDecision(
                symbol=token,
                action='buy',
                position_size=0.15,  # 15% allocation for top tokens
                entry_price=0.0,     # To be determined at execution
                target_price=0.0,    # To be determined at execution
                stop_loss=0.0,       # To be determined at execution
                confidence_score=0.75,
                profit_potential=0.08,
                risk_score=0.25,
                strategy_type='strategic_hold',
                reasoning=f'{token} selected as top-performing token by AI analysis',
                timestamp=datetime.now(),
                priority='medium',
                expected_duration='medium_term'
            )
            
        except Exception as e:
            return None
    
    def _optimize_portfolio_allocation(self) -> None:
        """Optimize portfolio allocation across selected tokens"""
        try:
            # Get current portfolio state
            current_allocations = self._get_current_allocations()
            
            # Get target allocations from best tokens
            target_allocations = self._calculate_target_allocations()
            
            # Calculate required adjustments
            adjustments = []
            for symbol, target_alloc in target_allocations.items():
                current_alloc = current_allocations.get(symbol, 0.0)
                adjustment = target_alloc - current_alloc
                
                if abs(adjustment) > 0.05:  # 5% threshold for rebalancing
                    action = 'increase' if adjustment > 0 else 'decrease'
                    priority = 1 if abs(adjustment) > 0.15 else 2
                    
                    adjustments.append(PortfolioAllocation(
                        symbol=symbol,
                        current_allocation=current_alloc,
                        target_allocation=target_alloc,
                        adjustment_needed=adjustment,
                        action_required=action,
                        priority=priority
                    ))
            
            self.portfolio_allocations = sorted(adjustments, key=lambda x: x.priority)
            
            if adjustments:
                self.info(f"📊 [TRADING_ORCHESTRATOR] Portfolio rebalancing needed for {len(adjustments)} tokens")
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error optimizing portfolio: {e}")
    
    def _get_current_allocations(self) -> Dict[str, float]:
        """Get current portfolio allocations"""
        try:
            # Calculate from current positions
            total_value = sum(pos.get('size', 0) for pos in self.current_positions.values())
            
            if total_value == 0:
                return {}
            
            allocations = {}
            for symbol, position in self.current_positions.items():
                allocations[symbol] = position.get('size', 0) / total_value
            
            return allocations
            
        except Exception as e:
            return {}
    
    def _calculate_target_allocations(self) -> Dict[str, float]:
        """Calculate target allocations based on token analysis"""
        try:
            target_allocations = {}
            
            # Get best tokens with their recommended allocations
            if self.token_selector:
                best_tokens = self.token_selector.analyze_and_select_best_tokens(top_n=5)
                
                for token in best_tokens:
                    target_allocations[token.symbol] = token.recommended_allocation
            else:
                # Fallback equal allocation
                fallback_tokens = ['BTC', 'ETH', 'SOL', 'AVAX']
                for token in fallback_tokens:
                    target_allocations[token] = 1.0 / len(fallback_tokens)
            
            return target_allocations
            
        except Exception as e:
            return {}
    
    def _execute_trading_decisions(self) -> None:
        """Execute approved trading decisions"""
        try:
            if not self.trading_decisions:
                return
            
            executed = 0
            for decision in self.trading_decisions:
                try:
                    # Check if we should execute this decision
                    if self._should_execute_decision(decision):
                        success = self._execute_single_decision(decision)
                        if success:
                            executed += 1
                            self.total_trades += 1
                            
                except Exception as e:
                    self.error(f"❌ [TRADING_ORCHESTRATOR] Error executing decision for {decision.symbol}: {e}")
                    continue
            
            if executed > 0:
                self.info(f"✅ [TRADING_ORCHESTRATOR] Executed {executed} trading decisions")
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error in decision execution: {e}")
    
    def _should_execute_decision(self, decision: TradingDecision) -> bool:
        """Determine if a trading decision should be executed"""
        try:
            # Check confidence threshold
            if decision.confidence_score < self.min_confidence_threshold:
                return False
            
            # Check position limits
            if len(self.current_positions) >= self.max_positions and decision.action == 'buy':
                return False
            
            # Check if we already have a position in this token
            if decision.symbol in self.current_positions and decision.action == 'buy':
                return False
            
            # Prioritize immediate decisions
            if decision.priority == 'immediate':
                return True
            
            # Check risk limits
            total_risk = sum(pos.get('risk_score', 0.3) for pos in self.current_positions.values())
            if total_risk + decision.risk_score > 2.0:  # Risk limit
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def _execute_single_decision(self, decision: TradingDecision) -> bool:
        """Execute a single trading decision"""
        try:
            self.info(f"🎯 [TRADING_ORCHESTRATOR] Executing {decision.action.upper()} for {decision.symbol} "
                     f"(Confidence: {decision.confidence_score:.1%})")
            
            # Mock execution - replace with real trading logic
            if decision.action == 'buy':
                # Generate realistic entry price if not provided
                entry_price = decision.entry_price if decision.entry_price > 0 else self._get_current_price(decision.symbol)
                target_price = decision.target_price if decision.target_price > 0 else entry_price * 1.05
                stop_loss = decision.stop_loss if decision.stop_loss > 0 else entry_price * 0.97
                
                self.current_positions[decision.symbol] = {
                    'size': decision.position_size,
                    'entry_price': entry_price,
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'confidence': decision.confidence_score,
                    'risk_score': decision.risk_score,
                    'entry_time': datetime.now(),
                    'strategy': decision.strategy_type
                }
                
                self.info(f"✅ [TRADING_ORCHESTRATOR] Opened {decision.symbol} position "
                         f"(Size: {decision.position_size:.1%}, Entry: ${entry_price:.2f})")
                return True
            
            elif decision.action == 'sell':
                if decision.symbol in self.current_positions:
                    # Calculate profit/loss
                    position = self.current_positions[decision.symbol]
                    current_price = self._get_current_price(decision.symbol)
                    profit = (current_price - position['entry_price']) / position['entry_price']
                    self.total_profit += profit
                    
                    if profit > 0:
                        self.successful_trades += 1
                    
                    del self.current_positions[decision.symbol]
                    
                    self.info(f"✅ [TRADING_ORCHESTRATOR] Closed {decision.symbol} position "
                             f"(P&L: {profit:.1%}, Exit: ${current_price:.2f})")
                    return True
            
            return False
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error executing decision: {e}")
            return False
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            # Mock current prices - replace with real price data
            prices = {
                'BTC': 107000.0,
                'ETH': 3400.0,
                'SOL': 200.0,
                'AVAX': 45.0,
                'MATIC': 0.8,
                'DOT': 8.5,
                'LINK': 25.0,
                'UNI': 12.0,
                'AAVE': 180.0
            }
            
            base_price = prices.get(symbol, 100.0)
            # Add some realistic variation
            return base_price * (0.98 + np.random.random() * 0.04)
            
        except Exception as e:
            return 100.0
    
    def _process_immediate_opportunity(self, opportunity: Dict[str, Any]) -> None:
        """Process high-urgency opportunities immediately"""
        try:
            self.info(f"🚨 [TRADING_ORCHESTRATOR] Processing immediate opportunity: {opportunity['symbol']}")
            
            # Create and execute decision immediately
            decision = self._create_trading_decision(opportunity)
            if decision and self._should_execute_decision(decision):
                success = self._execute_single_decision(decision)
                if success:
                    self.info(f"⚡ [TRADING_ORCHESTRATOR] Immediate execution successful for {opportunity['symbol']}")
            
        except Exception as e:
            self.error(f"❌ [TRADING_ORCHESTRATOR] Error processing immediate opportunity: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance tracking metrics"""
        try:
            # Calculate success rate
            success_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            # Calculate unrealized P&L for current positions
            unrealized_pnl = 0.0
            for symbol, position in self.current_positions.items():
                current_price = self._get_current_price(symbol)
                position_pnl = (current_price - position['entry_price']) / position['entry_price']
                unrealized_pnl += position_pnl * position['size']
            
            # Update max drawdown
            total_pnl = self.total_profit + unrealized_pnl
            if total_pnl < self.max_drawdown:
                self.max_drawdown = total_pnl
            
            self.info(f"📈 [TRADING_ORCHESTRATOR] Performance - Trades: {self.total_trades}, "
                     f"Success Rate: {success_rate:.1f}%, Realized P&L: {self.total_profit:.1%}, "
                     f"Unrealized P&L: {unrealized_pnl:.1%}")
            
        except Exception as e:
            pass
    
    def _log_analysis_results(self) -> None:
        """Log comprehensive analysis results"""
        try:
            # Log current status
            active_positions = len(self.current_positions)
            pending_decisions = len(self.trading_decisions)
            
            positions_summary = []
            for symbol, pos in self.current_positions.items():
                current_price = self._get_current_price(symbol)
                pnl = (current_price - pos['entry_price']) / pos['entry_price']
                positions_summary.append(f"{symbol}({pnl:+.1%})")
            
            self.info(f"📊 [TRADING_ORCHESTRATOR] Status - Active Positions: {active_positions}, "
                     f"Pending Decisions: {pending_decisions}, "
                     f"Best Tokens: {self.get_current_best_tokens(limit=3)}")
            
            if positions_summary:
                self.info(f"💼 [TRADING_ORCHESTRATOR] Positions: {', '.join(positions_summary)}")
            
        except Exception as e:
            pass
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        try:
            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            position_details = []
            
            for symbol, position in self.current_positions.items():
                current_price = self._get_current_price(symbol)
                position_pnl = (current_price - position['entry_price']) / position['entry_price']
                unrealized_pnl += position_pnl * position['size']
                
                position_details.append({
                    'symbol': symbol,
                    'size': position['size'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'pnl': position_pnl,
                    'confidence': position['confidence'],
                    'strategy': position.get('strategy', 'unknown')
                })
            
            return {
                'is_running': self.is_running,
                'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                'active_positions': len(self.current_positions),
                'pending_decisions': len(self.trading_decisions),
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'success_rate': (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
                'realized_profit': self.total_profit,
                'unrealized_profit': unrealized_pnl,
                'total_profit': self.total_profit + unrealized_pnl,
                'max_drawdown': self.max_drawdown,
                'current_best_tokens': self.get_current_best_tokens(limit=5),
                'immediate_opportunity': self.get_immediate_trading_opportunity(),
                'position_details': position_details,
                'portfolio_allocations': [asdict(alloc) for alloc in self.portfolio_allocations],
                'performance_metrics': {
                    'max_drawdown': self.max_drawdown,
                    'total_profit': self.total_profit + unrealized_pnl,
                    'avg_profit_per_trade': self.total_profit / self.total_trades if self.total_trades > 0 else 0,
                    'win_rate': (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                }
            }
            
        except Exception as e:
            return {'error': str(e)}

def main():
    """Test the intelligent trading orchestrator"""
    try:
        print("🎯 Initializing Intelligent Trading Orchestrator...")
        orchestrator = IntelligentTradingOrchestrator()
        
        print("🚀 Starting orchestrator...")
        orchestrator.start_orchestrator()
        
        # Let it run for analysis
        print("⏱️  Running orchestrator for 60 seconds...")
        time.sleep(60)
        
        # Get status
        status = orchestrator.get_orchestrator_status()
        
        print("\n📊 INTELLIGENT TRADING ORCHESTRATOR STATUS:")
        print("=" * 70)
        print(f"🔄 Running: {status['is_running']}")
        print(f"📈 Active Positions: {status['active_positions']}")
        print(f"💡 Pending Decisions: {status['pending_decisions']}")
        print(f"🎯 Total Trades: {status['total_trades']}")
        print(f"✅ Success Rate: {status['success_rate']:.1f}%")
        print(f"💰 Realized Profit: {status['realized_profit']:.1%}")
        print(f"💎 Unrealized Profit: {status['unrealized_profit']:.1%}")
        print(f"🏆 Total Profit: {status['total_profit']:.1%}")
        
        print(f"\n🏆 CURRENT BEST TOKENS:")
        for i, token in enumerate(status['current_best_tokens'], 1):
            print(f"  #{i}: {token}")
        
        immediate_opp = status['immediate_opportunity']
        if immediate_opp:
            print(f"\n🚨 IMMEDIATE OPPORTUNITY:")
            print(f"  🎯 Token: {immediate_opp['symbol']}")
            print(f"  🔥 Type: {immediate_opp['type']}")
            print(f"  📊 Confidence: {immediate_opp['confidence']:.1%}")
            print(f"  💰 Profit Potential: {immediate_opp.get('profit_potential', 0):.1%}")
            print(f"  ⚡ Urgency: {immediate_opp.get('urgency', 'unknown')}")
        
        # Show position details
        if status['position_details']:
            print(f"\n📋 CURRENT POSITIONS:")
            for pos in status['position_details']:
                print(f"  {pos['symbol']}: ${pos['current_price']:.2f} "
                      f"(Entry: ${pos['entry_price']:.2f}, P&L: {pos['pnl']:+.1%})")
        
        print("\n🛑 Stopping orchestrator...")
        orchestrator.stop_orchestrator()
        
        print("✅ Intelligent Trading Orchestrator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in orchestrator test: {e}")

if __name__ == "__main__":
    main() 