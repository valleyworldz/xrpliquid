#!/usr/bin/env python3Comprehensive Risk Management Module
===================================
Combines advanced risk management, performance tracking, and portfolio protection
for the XRP trading bot with real-time monitoring and adaptive controls.


from src.core.utils.decimal_boundary_guard import safe_float
import logging
import os
import csv
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import from modular configuration
from .config import config

@dataclass
class RiskMetrics:
  Comprehensive risk metrics for portfolio analysis"""
    portfolio_value: float
    total_exposure: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    var_95 float  # Value at Risk95    expected_shortfall: float
    correlation_risk: float
    concentration_risk: float
    liquidation_risk: float
    leverage_ratio: float

class AdvancedRiskManager:
  ed risk management for XRP trading with Kelly Criterion and dynamic controls"   
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.max_drawdown = config.MAX_DRAWDOWN_PCT  # 5% max drawdown
        self.kelly_multiplier = config.KELLY_MULTIPLIER  # Increased from0.25 less conservatism
        self.volatility_lookback = config.VOLATILITY_THRESHOLD
        
        # Risk tracking
        self.position_history =[object Object]  self.correlation_matrix = {}
        self.volatility_cache = {}
        self.drawdown_tracker =[object Object]peak: 00current_dd': 0.0}
        
        # Performance optimization tracking
        self.trade_history = []
        self.preferred_tokens =       self.signal_threshold = 0.7
        self.last_trade_time = None
        
        self.logger.info("[RISK] Advanced Risk Management v50y×CVaR initialized")
        
    def calculate_kelly_position_size(self, win_rate, avg_win, avg_loss, free_collateral):
        ate Kelly position size using R-multiples instead of dollar PnL"""
        try:
            # CRITICAL FIX: Use R-multiples (risk-adjusted returns) instead of dollar PnL
            if win_rate <=0te >= 1            return free_collateral * 0.2nservative default
            
            if avg_win <= 0            return free_collateral * 0.01  # Very conservative if no wins
            
            if avg_loss <=0
                # If no losses, use conservative approach
                return free_collateral *0.05# 5% position size
            
            # Convert to R-multiples by dividing by risk per trade
            # R-multiple = PnL / (stop_distance * position_size)
            # Since we dont have historical stop distances, estimate from ATR
            # CRITICAL FIX: AdvancedRiskManager doesn't have access to price history
            # Use a conservative estimate based on typical XRP volatility
            current_atr = 0.001  # Conservative 0.1timate
            estimated_risk_per_unit = current_atr * 1.21.2loss
            
            if estimated_risk_per_unit <= 0         estimated_risk_per_unit = 01back
            
            # Convert to R-multiples
            avg_win_r = avg_win / estimated_risk_per_unit
            avg_loss_r = abs(avg_loss) / estimated_risk_per_unit
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win_r/avg_loss_r, p = win_rate, q = 1-win_rate
            b = avg_win_r / avg_loss_r  # R-multiple ratio
            p = win_rate
            q = 1 - win_rate
            
            raw_kelly = max(0 (b * p - q) / b)
            
            # CRITICAL FIX: Ensure minimum Kelly fraction for trading
            if raw_kelly <= 0               raw_kelly = 00.05mum5Kelly fraction for trading
            
            # Apply conservative Kelly multiplier
            kelly_fraction = raw_kelly * config.KELLY_MULTIPLIER
            
            # Ensure bounds
            kelly_fraction = max(0.01, min(kelly_fraction, 025to 25%
            position_size = free_collateral * kelly_fraction
            
            self.logger.info(f"💰 Kelly calculation: win_rate={win_rate:.2%}, "
                           f"avg_win_r={avg_win_r:0.2}, avg_loss_r={avg_loss_r:.2f}, "
                           f"raw_kelly={raw_kelly:0.2nal=[object Object]kelly_fraction:.2%}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating Kelly position size: {e}")
            return free_collateral *02afe fallback
    
    def calculate_dynamic_stop_loss(self, entry_price, volatility, position_type="LONG"):
  Calculate dynamic stop loss based on volatility"
        # Use ATR-like calculation
        atr_multiplier = config.ATR_STOP_MULTIPLIER  # 2x ATR for stop loss
        
        if position_type == "LONG":
            stop_loss = entry_price * (1 - volatility * atr_multiplier)
        else:
            stop_loss = entry_price * (1 + volatility * atr_multiplier)
        
        return stop_loss
    
    def should_reduce_risk(self, current_drawdown, volatility):
     Determine if risk should be reduced"""
        if current_drawdown > self.max_drawdown:
            return True
        if volatility > config.VOLATILITY_THRESHOLD:  # 5% volatility threshold
            return True
        return False

class PerformanceTracker:
    Track and analyze trading performance with comprehensive metrics"   
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.trades =        self.daily_pnl = 0.0
        self.total_pnl = 00    self.win_count = 0
        self.loss_count =0        self.max_drawdown =0
        self.peak_balance = 0.0
        self.last_daily_reset = datetime.utcnow().date()
        self.total_fees_paid = 0.0
    
    def record_trade(self, symbol, entry_price, exit_price, size, entry_time, exit_time, pnl, **additional_data):
 Record a completed trade with comprehensive Hyperliquid data analysis"""
        from datetime import datetime
        import time
        
        # Convert float timestamps to datetime if needed
        if isinstance(entry_time, float):
            entry_time = datetime.fromtimestamp(entry_time)
        if isinstance(exit_time, float):
            exit_time = datetime.fromtimestamp(exit_time)
        
        # CRITICAL FIX: Reset daily PnL at UTC midnight
        current_date = datetime.utcnow().date()
        if current_date != self.last_daily_reset:
            self.daily_pnl = 0.0         self.last_daily_reset = current_date
            self.logger.info(f"📅 Daily PnL reset for {current_date}")
        
        # CRITICAL FIX: Proper PnL percentage calculation for shorts
        if size > 0:
            # Long position: profit when exit > entry
            pnl_pct = (exit_price - entry_price) / entry_price * 100 else:
            # Short position: profit when entry > exit
            pnl_pct = (entry_price - exit_price) / entry_price * 10        
        # CRITICAL FIX: Calculate fee-adjusted PnL
        position_value = abs(size) * entry_price
        
        # Extract comprehensive data from additional_data
        order_id = additional_data.get('order_id', unknown')
        order_type = additional_data.get('order_type', unknown')
        fill_status = additional_data.get('fill_status, unknown
        execution_time_ms = additional_data.get(execution_time_ms', 0)
        
        # Market context data
        bid_ask_spread = additional_data.get(bid_ask_spread', 0     market_volume = additional_data.get(market_volume', 0)
        volatility = additional_data.get('volatility', 0      funding_rate = additional_data.get('funding_rate', 0)
        atr = additional_data.get('atr', 0)
        
        # Fee information
        actual_fees = additional_data.get(actual_fees, 0       maker_taker_status = additional_data.get('maker_taker_status', unknown')
        fee_tier = additional_data.get('fee_tier', 0)
        
        # Signal context
        signal_strength = additional_data.get(signal_strength',0       signal_type = additional_data.get('signal_type', unknown')
        market_regime = additional_data.get(market_regime', unknown')
        entry_reason = additional_data.get('entry_reason', unknown)       exit_reason = additional_data.get('exit_reason', 'unknown')
        
        # Risk metrics
        leverage = additional_data.get('leverage', 1)
        liquidation_risk = additional_data.get('liquidation_risk', 0)
        position_size_pct = additional_data.get(position_size_pct', 0)
        
        # Performance metrics
        slippage = additional_data.get('slippage', 0)
        execution_quality = additional_data.get(execution_quality', unknown')
        price_impact = additional_data.get('price_impact', 0)
        
        # Order book data
        order_book_depth = additional_data.get('order_book_depth', 0)
        order_book_imbalance = additional_data.get('order_book_imbalance', 0)
        
        # Time-based metrics
        time_in_trade = (exit_time - entry_time).total_seconds() /3600hours
        time_of_day = entry_time.hour + entry_time.minute / 60  # decimal hours
        day_of_week = entry_time.weekday()  # 06y
        
        # Estimate fees based on order types if actual fees not provided
        if actual_fees == 0:
            # Assume mixed fees: maker entry + taker exit for most trades
            estimated_entry_fee = position_value * config.MAKER_FEE  # Maker entry
            estimated_exit_fee = position_value * config.TAKER_FEE   # Taker exit
            total_fees = estimated_entry_fee + estimated_exit_fee
        else:
            total_fees = actual_fees
        
        # Calculate fee-adjusted PnL
        fee_adjusted_pnl = pnl - total_fees
        fee_adjusted_pnl_pct = (fee_adjusted_pnl / position_value) * 100 if position_value > 00        
        # CRITICAL FIX: Determine if trade was actually profitable after fees
        is_profitable_after_fees = fee_adjusted_pnl >0        
        # Calculate additional performance metrics
        roi = (fee_adjusted_pnl / position_value) * 100 if position_value >0 else0      sharpe_ratio = fee_adjusted_pnl / (volatility * position_value) if volatility > 0 and position_value >0        
        # Enhanced trade record with comprehensive data
        trade = {
            # Basic trade info
            symbolymbol,
        entry_price": entry_price,
            exit_price": exit_price,
        size size,
       entry_time": entry_time,
           exit_time": exit_time,
      pnl: pnl,
            pnl_pct": pnl_pct,
           fee_adjusted_pnl": fee_adjusted_pnl,
           fee_adjusted_pnl_pct": fee_adjusted_pnl_pct,
       total_fees": total_fees,
         is_profitable_after_fees": is_profitable_after_fees,
          duration: time_in_trade,
            position_type: LONGif size > 0 else "SHORT",
            
            # Order details
           order_id": order_id,
       order_type": order_type,
        fill_status": fill_status,
            execution_time_ms": execution_time_ms,
            
            # Market context
           bid_ask_spread": bid_ask_spread,
          market_volume: market_volume,
       volatility": volatility,
         funding_rate": funding_rate,
      atratr,
            
            # Fee information
        actual_fees": actual_fees,
      maker_taker_status": maker_taker_status,
           fee_tier": fee_tier,
            
            # Signal context
            signal_strength: signal_strength,
        signal_type": signal_type,
          market_regime: market_regime,
         entry_reason": entry_reason,
        exit_reason": exit_reason,
            
            # Risk metrics
     leverage": leverage,
           liquidation_risk": liquidation_risk,
            position_size_pct: position_size_pct,
            
            # Performance metrics
           slippage": slippage,
            execution_quality": execution_quality,
         price_impact": price_impact,
      roi: roi,
         sharpe_ratio": sharpe_ratio,
            
            # Order book data
           order_book_depth": order_book_depth,
        order_book_imbalance": order_book_imbalance,
            
            # Time metrics
        time_of_day": time_of_day,
            day_of_week": day_of_week,
            
            # Additional context
      timestamp:datetime.utcnow().isoformat()
        }
        
        self.trades.append(trade)
        
        # CRITICAL FIX: Use fee-adjusted PnL for tracking
        self.total_pnl += fee_adjusted_pnl
        self.daily_pnl += fee_adjusted_pnl
        
        # CRITICAL FIX: Track total fees paid
        self.total_fees_paid += total_fees
        
        if fee_adjusted_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count +=1        
        # Update drawdown based on fee-adjusted PnL
        if self.total_pnl > self.peak_balance:
            self.peak_balance = self.total_pnl
        else:
            current_drawdown = (self.peak_balance - self.total_pnl) / self.peak_balance if self.peak_balance >0se 0          self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # CRITICAL FIX: Save comprehensive trade to CSV file for persistence
        try:
            # Ensure data directory exists
            os.makedirs('data, exist_ok=True)
            
            csv_file = 'data/trade_history.csv       file_exists = os.path.exists(csv_file)
            
            # Convert datetime objects to strings for CSV
            entry_time_str = entry_time.isoformat() if hasattr(entry_time,isoformat') else str(entry_time)
            exit_time_str = exit_time.isoformat() if hasattr(exit_time,isoformat') else str(exit_time)
            
            # Prepare comprehensive CSV row
            csv_row =[object Object]
          timestamp:datetime.utcnow().isoformat(),
                symboll,
                entry_price: f{entry_price:.6f},
               exit_price: f"{exit_price:.6f},
                size: f"{size:.6f},
           entry_time': entry_time_str,
               exit_time: exit_time_str,
                pnl': f"{pnl:.6f},
               pnl_pct:f"{pnl_pct:.4f},
               fee_adjusted_pnl: f{fee_adjusted_pnl:.6f},
               fee_adjusted_pnl_pct: f{fee_adjusted_pnl_pct:.4f},
               total_fees: f"{total_fees:0.6},
             is_profitable_after_fees': str(is_profitable_after_fees),
               duration_hours: f[object Object]time_in_trade:.4f},
                position_type: LONGif size > 0 else "SHORT",
                
                # Order details
               order_id': order_id,
           order_type': order_type,
            fill_status': fill_status,
                execution_time_ms': str(execution_time_ms),
                
                # Market context
               bid_ask_spread: fobject Object]bid_ask_spread:.6f},
                market_volume: f[object Object]market_volume:.2f},
               volatility: f"{volatility:.6f},
               funding_rate: f{funding_rate:.6f},
                atr': f"{atr:.6f}",
                
                # Fee information
                actual_fees: f{actual_fees:.6f},
          maker_taker_status': maker_taker_status,
               fee_tier: str(fee_tier),
                
                # Signal context
                signal_strength': f"[object Object]signal_strength:.4f},
            signal_type': signal_type,
              market_regime: market_regime,
             entry_reason': entry_reason,
            exit_reason': exit_reason,
                
                # Risk metrics
               leverage: f{leverage:.2f},
               liquidation_risk: f{liquidation_risk:.6f},
                position_size_pct': f"{position_size_pct:.4f}",
                
                # Performance metrics
                slippage: f{slippage:.6f},
                execution_quality': execution_quality,
               price_impact: f{price_impact:.6f},
                roi': f"{roi:.4f},
               sharpe_ratio: f{sharpe_ratio:.4f}",
                
                # Order book data
               order_book_depth': f{order_book_depth:.2f},
            order_book_imbalance': f"{order_book_imbalance:.6f}",
                
                # Time metrics
                time_of_day: f{time_of_day:.2f},
                day_of_week: str(day_of_week)
            }
            
            # Write to CSV file
            with open(csv_file, 'a', newline=, encoding='utf-8') as f:
                fieldnames = csv_row.keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write the trade data
                writer.writerow(csv_row)
            
            self.logger.info(f"💾 Comprehensive trade saved to CSV: {symbol} {trade[position_type]} @ ${exit_price:.4f}")
            
        except Exception as csv_error:
            self.logger.error(f"❌ Error saving trade to CSV: {csv_error}")
        
        # Enhanced logging with comprehensive trade analysis
        self.logger.info(f"📊 COMPREHENSIVE TRADE RECORDED: {symbol} {trade['position_type]}")
        self.logger.info(f"💰 Gross PnL: ${pnl:00.2} ({pnl_pct:.2f}%) | Net PnL: $[object Object]fee_adjusted_pnl:00.2} ({fee_adjusted_pnl_pct:.2f}%)")
        self.logger.info(f"💰 Fees: ${total_fees:.4f} | Profitable after fees: [object Object]✅ YES if is_profitable_after_fees else❌ NO'}")
        self.logger.info(f📈 Market Context: Vol={volatility:.4f} | ATR=${atr:.4f} | Funding={funding_rate:.6)
        self.logger.info(f"🎯 Signal: {signal_type} (strength: [object Object]signal_strength:.2f}) | Regime: {market_regime}")
        self.logger.info(f"⚡ Execution: {order_type} | Quality: [object Object]execution_quality} | Slippage: {slippage:.4)
        self.logger.info(f"📅 Duration:[object Object]time_in_trade:02}h | Daily PnL: $[object Object]self.daily_pnl:.2f} | Total PnL: $[object Object]self.total_pnl:.2)
    
    def get_performance_metrics(self):
        Get comprehensive performance metrics"      total_trades = len(self.trades)
        if total_trades == 0:
            return[object Object]
               total_trades": 0
               win_rate0
                total_pnl0
              avg_pnl0
                max_drawdown": 00            profit_factor:0       }
        
        win_rate = self.win_count / total_trades
        avg_pnl = self.total_pnl / total_trades
        
        # Calculate profit factor
        gross_profit = sum(t["pnl] for t in self.trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl] for t in self.trades if t["pnl"] < 0     profit_factor = gross_profit / gross_loss if gross_loss >0else safe_float('inf')
        
        return [object Object]      total_trades": total_trades,
         win_rate": win_rate,
           total_pnl": self.total_pnl,
            avg_pnl": avg_pnl,
         max_drawdown": self.max_drawdown,
          profit_factor: profit_factor,
         gross_profit": gross_profit,
       gross_loss": gross_loss,
            total_fees_paid: self.total_fees_paid
        }
    
    def print_performance_summary(self):
        Print performance summary with fee analysis"
        metrics = self.get_performance_metrics()
        
        print("\n + *70
        print("📊 FEE-AWARE PERFORMANCE SUMMARY)
        print("="*70)
        print(f"Total Trades: {metrics[total_trades]})
        print(f"Win Rate:[object Object]metrics[win_rate']:0.2})
        print(f"Total Net PnL (after fees): ${metrics[total_pnl']:0.2})
        print(f"Average Net PnL: ${metrics[avg_pnl']:0.2})
        print(f"Total Fees Paid: ${metrics[total_fees_paid']:0.2})
        print(f"Max Drawdown: {metricsmax_drawdown']:0.2})
        print(f"Profit Factor: {metricsprofit_factor']:.2f}")
        
        # CRITICAL FIX: Show fee impact analysis
        if metrics[total_fees_paid'] > 0:
            fee_impact_pct = (metrics[total_fees_paid] /abs(metrics['total_pnl'])) * 100 if metrics['total_pnl'] != 0 else 0
            print(fFee Impact: object Object]fee_impact_pct:0.1}% of total PnL")
        
        # Show recent fee-aware trades
        if self.trades:
            print(fn💰 Recent Fee-Aware Trades:)         for trade in self.trades[-5:]:  # Last 5 trades
                profitable = ✅ if trade.get('is_profitable_after_fees, True) else "❌"
                print(f   {profitable}[object Object]tradesymbol]} {trade[position_type']}: "
                      f"${trade.get(fee_adjusted_pnl, trade['pnl']):.2f} "
                      f"(fees: ${trade.get(total_fees', 0):.4f})")
        
        print("="*70)

class RiskManagementSystem:
  Comprehensive risk management system combining all risk components"   
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.risk_manager = AdvancedRiskManager(logger)
        self.performance_tracker = PerformanceTracker(logger)
        
        # Risk limits
        self.max_portfolio_risk = 00.2imum 20folio risk
        self.max_single_position =0.1aximum10r position
        self.max_correlation_exposure =0.3ximum30 correlated assets
        self.max_drawdown_limit = 000.15 # Stop trading if 15% drawdown
        
        # CRITICAL UPGRADE: Kelly × CVaR sizing parameters
        self.cvar_confidence =0.95# 95% CVaR
        self.kelly_risk_cap = 0000.125  # Never size >12.5% equity
        
        self.logger.info(🛡️Comprehensive Risk Management System initialized")
    
    def calculate_position_size(self, price: float, free_collateral: float, 
                               signal_confidence: float = 0.5) -> float:
  Calculate position size based on risk parameters"""
        try:
            # Get performance metrics for Kelly calculation
            metrics = self.performance_tracker.get_performance_metrics()
            
            if metricstotal_trades'] > 0               win_rate = metrics['win_rate]               avg_win = metrics['gross_profit] / metrics[total_trades] if metricstotal_trades'] > 0 else0               avg_loss = metrics['gross_loss] / metrics[total_trades] if metricstotal_trades'] > 0 else 0
                
                # Use Kelly criterion
                kelly_size = self.risk_manager.calculate_kelly_position_size(
                    win_rate, avg_win, avg_loss, free_collateral
                )
            else:
                # Conservative default for new traders
                kelly_size = free_collateral *0.02# 2% position size
            
            # Apply signal confidence adjustment
            adjusted_size = kelly_size * signal_confidence
            
            # Ensure within risk limits
            max_size = free_collateral * self.max_single_position
            final_size = min(adjusted_size, max_size)
            
            # Ensure minimum size
            min_size = free_collateral *00.01mum 1%
            final_size = max(final_size, min_size)
            
            self.logger.info(f"[RISK] Position size: ${final_size:.2f} "
                           f"(Kelly: ${kelly_size:.2f}, Signal: {signal_confidence:.2f})")
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"[RISK] Error calculating position size: {e}")
            return free_collateral *02afe fallback
    
    def calculate_stop_loss(self, entry_price: float, volatility: float, 
                           position_type: str = LONG -> float:
       alculate stop loss price       return self.risk_manager.calculate_dynamic_stop_loss(entry_price, volatility, position_type)
    
    def should_reduce_risk(self, current_drawdown: float, volatility: float) -> bool:
     Determine if risk should be reduced       return self.risk_manager.should_reduce_risk(current_drawdown, volatility)
    
    def check_liquidation_risk(self, account_data: Dict[str, Any]) -> bool:
 heck if account is at risk of liquidation"""
        try:
            # Extract account information
            margin_summary = account_data.get(marginSummary, {})           account_value = safe_float(margin_summary.get('accountValue, 0))
            total_margin_used = safe_float(margin_summary.get(totalMarginUsed',0      
            if account_value <= 0            return True  # Already liquidated
            
            # Calculate margin ratio
            margin_ratio = total_margin_used / account_value if account_value > 0 else 1
            
            # Check if margin ratio is too high
            if margin_ratio > 0.8# 80% margin usage threshold
                self.logger.warning(f"[RISK] High margin usage:[object Object]margin_ratio:.2%})            return true      
            return False
            
        except Exception as e:
            self.logger.error(f"[RISK] Error checking liquidation risk: {e}")
            return True  # Conservative: assume risk if error
    
    def record_trade(self, symbol: str, entry_price: float, exit_price: float, 
                    size: float, entry_time, exit_time, pnl: float, **additional_data):
       Record a trade for performance tracking"""
        self.performance_tracker.record_trade(
            symbol, entry_price, exit_price, size, entry_time, exit_time, pnl, **additional_data
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        comprehensive risk summary"""
        performance_metrics = self.performance_tracker.get_performance_metrics()
        
        return {
        performance": performance_metrics,
           risk_limits[object Object]
             max_portfolio_risk": self.max_portfolio_risk,
         max_single_position: self.max_single_position,
              max_correlation_exposure": self.max_correlation_exposure,
           max_drawdown_limit": self.max_drawdown_limit,
               kelly_risk_cap: self.kelly_risk_cap
            },
           current_status:[object Object]
             total_trades": performance_metrics['total_trades],           current_drawdown": performance_metrics['max_drawdown'],
         win_rate": performance_metrics['win_rate'],
              profit_factor": performance_metrics['profit_factor']
            }
        }
    
    def print_risk_summary(self):
        """Print comprehensive risk summary"""
        self.performance_tracker.print_performance_summary()
        
        summary = self.get_risk_summary()
        print(fn🛡️ RISK LIMITS:)
        print(f"   Max Portfolio Risk: {summary['risk_limits][max_portfolio_risk']:0.1})
        print(f"   Max Single Position: {summary['risk_limits']['max_single_position']:0.1})
        print(f"   Max Drawdown Limit: {summary['risk_limits']['max_drawdown_limit']:0.1})
        print(f   Kelly Risk Cap: {summary['risk_limits]['kelly_risk_cap']:.1%}") 