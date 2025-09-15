import logging
import os
import csv
from datetime import datetime
from typing import Optional, Dict, Any

# Import configuration
try:
    from src.core.config import config
except ImportError:
    class FallbackConfig:
        MAKER_FEE = 0.00015
        TAKER_FEE = 0.00045
    config = FallbackConfig()

class PerformanceTracker:
    """Track and analyze trading performance"""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.trades = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.last_daily_reset = datetime.utcnow().date()

    def record_trade(self, symbol: str, entry_price: float, exit_price: float, size: float, 
                    entry_time, exit_time, pnl: float, **additional_data):
        """Record a completed trade with comprehensive Hyperliquid data analysis"""
        # Convert float timestamps to datetime if needed
        if isinstance(entry_time, float):
            entry_time = datetime.fromtimestamp(entry_time)
        if isinstance(exit_time, float):
            exit_time = datetime.fromtimestamp(exit_time)
        
        # Reset daily PnL at UTC midnight
        current_date = datetime.utcnow().date()
        if current_date != self.last_daily_reset:
            self.daily_pnl = 0.0
            self.last_daily_reset = current_date
            self.logger.info(f"📅 Daily PnL reset for {current_date}")
        
        # Proper PnL percentage calculation for shorts
        if size > 0:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        # Calculate fee-adjusted PnL
        position_value = abs(size) * entry_price
        
        # Extract comprehensive data from additional_data
        order_id = additional_data.get('order_id', 'unknown')
        order_type = additional_data.get('order_type', 'unknown')
        fill_status = additional_data.get('fill_status', 'unknown')
        execution_time_ms = additional_data.get('execution_time_ms', 0)
        
        # Market context data
        bid_ask_spread = additional_data.get('bid_ask_spread', 0)
        market_volume = additional_data.get('market_volume', 0)
        volatility = additional_data.get('volatility', 0)
        funding_rate = additional_data.get('funding_rate', 0)
        atr = additional_data.get('atr', 0)
        
        # Fee information
        actual_fees = additional_data.get('actual_fees', 0)
        maker_taker_status = additional_data.get('maker_taker_status', 'unknown')
        fee_tier = additional_data.get('fee_tier', 0)
        
        # Signal context
        signal_strength = additional_data.get('signal_strength', 0)
        signal_type = additional_data.get('signal_type', 'unknown')
        market_regime = additional_data.get('market_regime', 'unknown')
        entry_reason = additional_data.get('entry_reason', 'unknown')
        exit_reason = additional_data.get('exit_reason', 'unknown')
        
        # Risk metrics
        leverage = additional_data.get('leverage', 1)
        liquidation_risk = additional_data.get('liquidation_risk', 0)
        position_size_pct = additional_data.get('position_size_pct', 0)
        
        # Performance metrics
        slippage = additional_data.get('slippage', 0)
        execution_quality = additional_data.get('execution_quality', 'unknown')
        price_impact = additional_data.get('price_impact', 0)
        
        # Order book data
        order_book_depth = additional_data.get('order_book_depth', 0)
        order_book_imbalance = additional_data.get('order_book_imbalance', 0)
        
        # Time-based metrics
        time_in_trade = (exit_time - entry_time).total_seconds() / 3600  # hours
        time_of_day = entry_time.hour + entry_time.minute / 60  # decimal hours
        day_of_week = entry_time.weekday()  # 0=Monday, 6=Sunday
        
        # Estimate fees based on order types if actual fees not provided
        if actual_fees == 0:
            estimated_entry_fee = position_value * getattr(config, 'MAKER_FEE', 0.00015)
            estimated_exit_fee = position_value * getattr(config, 'TAKER_FEE', 0.00045)
            total_fees = estimated_entry_fee + estimated_exit_fee
        else:
            total_fees = actual_fees
        
        # Calculate fee-adjusted PnL
        fee_adjusted_pnl = pnl - total_fees
        fee_adjusted_pnl_pct = (fee_adjusted_pnl / position_value) * 100 if position_value > 0 else 0.0
        
        # Determine if trade was actually profitable after fees
        is_profitable_after_fees = fee_adjusted_pnl > 0
        
        # Calculate additional performance metrics
        roi = (fee_adjusted_pnl / position_value) * 100 if position_value > 0 else 0
        sharpe_ratio = fee_adjusted_pnl / (volatility * position_value) if volatility > 0 and position_value > 0 else 0
        
        # Enhanced trade record with comprehensive data
        trade = {
            # Basic trade info
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "fee_adjusted_pnl": fee_adjusted_pnl,
            "fee_adjusted_pnl_pct": fee_adjusted_pnl_pct,
            "total_fees": total_fees,
            "is_profitable_after_fees": is_profitable_after_fees,
            "duration": time_in_trade,
            "position_type": "LONG" if size > 0 else "SHORT",
            
            # Order details
            "order_id": order_id,
            "order_type": order_type,
            "fill_status": fill_status,
            "execution_time_ms": execution_time_ms,
            
            # Market context
            "bid_ask_spread": bid_ask_spread,
            "market_volume": market_volume,
            "volatility": volatility,
            "funding_rate": funding_rate,
            "atr": atr,
            
            # Fee information
            "actual_fees": actual_fees,
            "maker_taker_status": maker_taker_status,
            "fee_tier": fee_tier,
            
            # Signal context
            "signal_strength": signal_strength,
            "signal_type": signal_type,
            "market_regime": market_regime,
            "entry_reason": entry_reason,
            "exit_reason": exit_reason,
            
            # Risk metrics
            "leverage": leverage,
            "liquidation_risk": liquidation_risk,
            "position_size_pct": position_size_pct,
            
            # Performance metrics
            "slippage": slippage,
            "execution_quality": execution_quality,
            "price_impact": price_impact,
            "roi": roi,
            "sharpe_ratio": sharpe_ratio,
            
            # Order book data
            "order_book_depth": order_book_depth,
            "order_book_imbalance": order_book_imbalance,
            
            # Time metrics
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            
            # Additional context
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.trades.append(trade)
        
        # Use fee-adjusted PnL for tracking
        self.total_pnl += fee_adjusted_pnl
        self.daily_pnl += fee_adjusted_pnl
        
        # Track total fees paid
        if not hasattr(self, 'total_fees_paid'):
            self.total_fees_paid = 0.0
        self.total_fees_paid += total_fees
        
        if fee_adjusted_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Update drawdown based on fee-adjusted PnL
        if self.total_pnl > self.peak_balance:
            self.peak_balance = self.total_pnl
        else:
            current_drawdown = (self.peak_balance - self.total_pnl) / self.peak_balance if self.peak_balance > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Save comprehensive trade to CSV file for persistence
        try:
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            csv_file = 'data/trade_history.csv'
            file_exists = os.path.exists(csv_file)
            
            # Convert datetime objects to strings for CSV
            entry_time_str = entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time)
            exit_time_str = exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time)
            
            # Prepare comprehensive CSV row
            csv_row = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'entry_price': f"{entry_price:.6f}",
                'exit_price': f"{exit_price:.6f}",
                'size': f"{size:.6f}",
                'entry_time': entry_time_str,
                'exit_time': exit_time_str,
                'pnl': f"{pnl:.6f}",
                'pnl_pct': f"{pnl_pct:.4f}",
                'fee_adjusted_pnl': f"{fee_adjusted_pnl:.6f}",
                'fee_adjusted_pnl_pct': f"{fee_adjusted_pnl_pct:.4f}",
                'total_fees': f"{total_fees:.6f}",
                'is_profitable_after_fees': str(is_profitable_after_fees),
                'duration_hours': f"{time_in_trade:.4f}",
                'position_type': "LONG" if size > 0 else "SHORT",
                
                # Order details
                'order_id': order_id,
                'order_type': order_type,
                'fill_status': fill_status,
                'execution_time_ms': str(execution_time_ms),
                
                # Market context
                'bid_ask_spread': f"{bid_ask_spread:.6f}",
                'market_volume': f"{market_volume:.2f}",
                'volatility': f"{volatility:.6f}",
                'funding_rate': f"{funding_rate:.6f}",
                'atr': f"{atr:.6f}",
                
                # Fee information
                'actual_fees': f"{actual_fees:.6f}",
                'maker_taker_status': maker_taker_status,
                'fee_tier': str(fee_tier),
                
                # Signal context
                'signal_strength': f"{signal_strength:.4f}",
                'signal_type': signal_type,
                'market_regime': market_regime,
                'entry_reason': entry_reason,
                'exit_reason': exit_reason,
                
                # Risk metrics
                'leverage': f"{leverage:.2f}",
                'liquidation_risk': f"{liquidation_risk:.6f}",
                'position_size_pct': f"{position_size_pct:.4f}",
                
                # Performance metrics
                'slippage': f"{slippage:.6f}",
                'execution_quality': execution_quality,
                'price_impact': f"{price_impact:.6f}",
                'roi': f"{roi:.4f}",
                'sharpe_ratio': f"{sharpe_ratio:.4f}",
                
                # Order book data
                'order_book_depth': f"{order_book_depth:.2f}",
                'order_book_imbalance': f"{order_book_imbalance:.6f}",
                
                # Time metrics
                'time_of_day': f"{time_of_day:.2f}",
                'day_of_week': str(day_of_week)
            }
            
            # Write to CSV file
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = csv_row.keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write the trade data
                writer.writerow(csv_row)
            
            self.logger.info(f"💾 Comprehensive trade saved to CSV: {symbol} {trade['position_type']} @ ${exit_price:.4f}")
            
        except Exception as csv_error:
            self.logger.error(f"❌ Error saving trade to CSV: {csv_error}")
        
        # Enhanced logging with comprehensive trade analysis
        self.logger.info(f"📊 COMPREHENSIVE TRADE RECORDED: {symbol} {trade['position_type']}")
        self.logger.info(f"💰 Gross PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Net PnL: ${fee_adjusted_pnl:.2f} ({fee_adjusted_pnl_pct:.2f}%)")
        self.logger.info(f"💰 Fees: ${total_fees:.4f} | Profitable after fees: {'✅ YES' if is_profitable_after_fees else '❌ NO'}")
        self.logger.info(f"📈 Market Context: Vol={volatility:.4f} | ATR=${atr:.4f} | Funding={funding_rate:.6f}")
        self.logger.info(f"🎯 Signal: {signal_type} (strength: {signal_strength:.2f}) | Regime: {market_regime}")
        self.logger.info(f"⚡ Execution: {order_type} | Quality: {execution_quality} | Slippage: {slippage:.4f}")
        self.logger.info(f"📅 Duration: {time_in_trade:.2f}h | Daily PnL: ${self.daily_pnl:.2f} | Total PnL: ${self.total_pnl:.2f}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0
            }
        
        win_rate = self.win_count / total_trades
        avg_pnl = self.total_pnl / total_trades
        
        # Calculate profit factor
        gross_profit = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl": avg_pnl,
            "max_drawdown": self.max_drawdown,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss
        }

    def print_performance_summary(self):
        """Print performance summary with fee analysis"""
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*70)
        print("📊 FEE-AWARE PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Net PnL (after fees): ${metrics['total_pnl']:.2f}")
        print(f"Average Net PnL: ${metrics['avg_pnl']:.2f}")
        print(f"Total Fees Paid: ${getattr(self, 'total_fees_paid', 0.0):.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Show fee impact analysis
        if hasattr(self, 'total_fees_paid') and self.total_fees_paid > 0:
            fee_impact_pct = (self.total_fees_paid / abs(metrics['total_pnl'])) * 100 if metrics['total_pnl'] != 0 else 0
            print(f"Fee Impact: {fee_impact_pct:.1f}% of total PnL")
        
        # Show recent fee-aware trades
        if self.trades:
            print(f"\n💰 Recent Fee-Aware Trades:")
            for trade in self.trades[-5:]:  # Last 5 trades
                profitable = "✅" if trade.get('is_profitable_after_fees', True) else "❌"
                print(f"   {profitable} {trade['symbol']} {trade['position_type']}: "
                      f"${trade.get('fee_adjusted_pnl', trade['pnl']):.2f} "
                      f"(fees: ${trade.get('total_fees', 0):.4f})")
        
        print("="*70) 