"""
🎯 FUNDING ARBITRAGE SCHEDULER
=============================
Intelligent scheduling system for funding arbitrage opportunities
with optimal timing and execution coordination.
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.strategies.funding_arbitrage import FundingArbitrageStrategy, FundingArbitrageConfig
from src.core.utils.logger import Logger

@dataclass
class ScheduleConfig:
    """Configuration for funding arbitrage scheduler"""
    
    # Timing parameters
    funding_rate_check_interval: int = 300      # Check every 5 minutes
    execution_window_minutes: int = 30          # 30-minute execution window
    pre_funding_check_minutes: int = 15         # Check 15 minutes before funding
    post_funding_check_minutes: int = 15        # Check 15 minutes after funding
    
    # Funding schedule (UTC times when funding payments occur)
    funding_times_utc: List[str] = field(default_factory=lambda: [
        "00:00", "08:00", "16:00"  # Every 8 hours
    ])
    
    # Execution parameters
    max_concurrent_positions: int = 3           # Maximum concurrent positions
    min_time_between_trades: int = 60           # Minimum 1 minute between trades
    max_daily_trades: int = 10                  # Maximum trades per day
    
    # Risk management
    max_daily_loss: float = 500.0               # Maximum daily loss in USD
    max_position_duration_hours: int = 24       # Maximum position duration
    emergency_stop_loss_percent: float = 10.0   # Emergency stop loss
    
    # Market conditions
    min_volume_threshold: float = 1000000.0     # Minimum 24h volume
    max_spread_threshold: float = 0.001         # Maximum spread (0.1%)
    min_liquidity_threshold: float = 50000.0    # Minimum liquidity

@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    
    task_id: str
    task_type: str  # 'funding_check', 'execution', 'monitoring', 'cleanup'
    scheduled_time: float
    priority: int = 1  # 1 = highest, 5 = lowest
    data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)

class FundingArbitrageScheduler:
    """
    🎯 FUNDING ARBITRAGE SCHEDULER
    Intelligent scheduling system for optimal funding arbitrage execution
    """
    
    def __init__(self, 
                 strategy: FundingArbitrageStrategy,
                 config: ScheduleConfig,
                 logger: Optional[Logger] = None):
        self.strategy = strategy
        self.config = config
        self.logger = logger or Logger()
        
        # Scheduler state
        self.running = False
        self.scheduled_tasks: List[ScheduledTask] = []
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_counter = 0
        
        # Performance tracking
        self.daily_stats = {
            'trades_executed': 0,
            'total_pnl': 0.0,
            'start_time': time.time(),
            'last_reset': time.time()
        }
        
        # Funding time tracking
        self.next_funding_times = self._calculate_next_funding_times()
        
        self.logger.info("🎯 [FUNDING_SCHEDULER] Funding Arbitrage Scheduler initialized")
        self.logger.info(f"📊 [FUNDING_SCHEDULER] Check interval: {self.config.funding_rate_check_interval}s")
        self.logger.info(f"📊 [FUNDING_SCHEDULER] Max concurrent positions: {self.config.max_concurrent_positions}")
        self.logger.info(f"📊 [FUNDING_SCHEDULER] Funding times: {self.config.funding_times_utc}")
    
    def _calculate_next_funding_times(self) -> List[float]:
        """Calculate next funding payment times"""
        now = datetime.utcnow()
        funding_times = []
        
        for time_str in self.config.funding_times_utc:
            hour, minute = map(int, time_str.split(':'))
            funding_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If time has passed today, schedule for next occurrence
            if funding_time <= now:
                funding_time += timedelta(hours=8)  # Next 8-hour cycle
            
            funding_times.append(funding_time.timestamp())
        
        return sorted(funding_times)
    
    def _get_next_task_id(self) -> str:
        """Generate next task ID"""
        self.task_counter += 1
        return f"task_{self.task_counter}_{int(time.time())}"
    
    def schedule_task(self, 
                     task_type: str,
                     scheduled_time: float,
                     priority: int = 1,
                     data: Optional[Dict[str, Any]] = None) -> str:
        """Schedule a new task"""
        
        task_id = self._get_next_task_id()
        task = ScheduledTask(
            task_id=task_id,
            task_type=task_type,
            scheduled_time=scheduled_time,
            priority=priority,
            data=data or {}
        )
        
        self.scheduled_tasks.append(task)
        self.scheduled_tasks.sort(key=lambda x: (x.scheduled_time, x.priority))
        
        self.logger.debug(f"📅 [SCHEDULER] Scheduled {task_type} task {task_id} for {datetime.fromtimestamp(scheduled_time)}")
        
        return task_id
    
    def schedule_funding_checks(self):
        """Schedule funding rate checks around funding times"""
        
        current_time = time.time()
        
        for funding_time in self.next_funding_times:
            # Schedule pre-funding check
            pre_check_time = funding_time - (self.config.pre_funding_check_minutes * 60)
            if pre_check_time > current_time:
                self.schedule_task(
                    task_type='funding_check',
                    scheduled_time=pre_check_time,
                    priority=1,
                    data={'funding_time': funding_time, 'check_type': 'pre_funding'}
                )
            
            # Schedule funding time check
            if funding_time > current_time:
                self.schedule_task(
                    task_type='funding_check',
                    scheduled_time=funding_time,
                    priority=1,
                    data={'funding_time': funding_time, 'check_type': 'funding_time'}
                )
            
            # Schedule post-funding check
            post_check_time = funding_time + (self.config.post_funding_check_minutes * 60)
            if post_check_time > current_time:
                self.schedule_task(
                    task_type='funding_check',
                    scheduled_time=post_check_time,
                    priority=2,
                    data={'funding_time': funding_time, 'check_type': 'post_funding'}
                )
    
    def schedule_regular_monitoring(self):
        """Schedule regular monitoring tasks"""
        
        current_time = time.time()
        
        # Schedule regular funding rate checks
        next_check = current_time + self.config.funding_rate_check_interval
        self.schedule_task(
            task_type='funding_check',
            scheduled_time=next_check,
            priority=3,
            data={'check_type': 'regular'}
        )
        
        # Schedule position monitoring
        next_monitor = current_time + 60  # Every minute
        self.schedule_task(
            task_type='position_monitoring',
            scheduled_time=next_monitor,
            priority=2,
            data={}
        )
        
        # Schedule cleanup tasks
        next_cleanup = current_time + 3600  # Every hour
        self.schedule_task(
            task_type='cleanup',
            scheduled_time=next_cleanup,
            priority=4,
            data={}
        )
    
    async def execute_task(self, task: ScheduledTask) -> bool:
        """Execute a scheduled task"""
        
        try:
            self.logger.debug(f"🔄 [SCHEDULER] Executing {task.task_type} task {task.task_id}")
            
            if task.task_type == 'funding_check':
                return await self._execute_funding_check(task)
            elif task.task_type == 'position_monitoring':
                return await self._execute_position_monitoring(task)
            elif task.task_type == 'execution':
                return await self._execute_trade_execution(task)
            elif task.task_type == 'cleanup':
                return await self._execute_cleanup(task)
            else:
                self.logger.warning(f"⚠️ [SCHEDULER] Unknown task type: {task.task_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [SCHEDULER] Error executing task {task.task_id}: {e}")
            return False
    
    async def _execute_funding_check(self, task: ScheduledTask) -> bool:
        """Execute funding rate check"""
        
        try:
            check_type = task.data.get('check_type', 'regular')
            funding_time = task.data.get('funding_time')
            
            self.logger.info(f"📊 [SCHEDULER] Executing {check_type} funding check")
            
            # Check if we can execute new trades
            if not self._can_execute_new_trade():
                self.logger.info("⏸️ [SCHEDULER] Cannot execute new trades (limits reached)")
                return True
            
            # Monitor funding rates for opportunities
            opportunities = await self.strategy.monitor_funding_rates()
            
            if opportunities:
                self.logger.info(f"🎯 [SCHEDULER] Found {len(opportunities)} funding arbitrage opportunities")
                
                # Execute best opportunity
                best_opportunity = opportunities[0]  # Already sorted by expected value
                
                # Schedule execution
                execution_time = time.time() + self.config.min_time_between_trades
                self.schedule_task(
                    task_type='execution',
                    scheduled_time=execution_time,
                    priority=1,
                    data={'opportunity': best_opportunity.to_dict()}
                )
                
                self.logger.info(f"📅 [SCHEDULER] Scheduled execution for {best_opportunity.symbol}")
            else:
                self.logger.info("📊 [SCHEDULER] No funding arbitrage opportunities found")
            
            # Schedule next check
            next_check_time = time.time() + self.config.funding_rate_check_interval
            self.schedule_task(
                task_type='funding_check',
                scheduled_time=next_check_time,
                priority=3,
                data={'check_type': 'regular'}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [SCHEDULER] Error in funding check: {e}")
            return False
    
    async def _execute_position_monitoring(self, task: ScheduledTask) -> bool:
        """Execute position monitoring"""
        
        try:
            self.logger.debug("📊 [SCHEDULER] Monitoring active positions")
            
            current_time = time.time()
            positions_to_close = []
            
            # Check each active position
            for symbol, position in self.strategy.active_positions.items():
                position_age = current_time - position['entry_time']
                position_age_hours = position_age / 3600
                
                # Check if position should be closed
                if position_age_hours > self.config.max_position_duration_hours:
                    positions_to_close.append(symbol)
                    self.logger.info(f"⏰ [SCHEDULER] Position {symbol} exceeded max duration ({position_age_hours:.1f}h)")
                
                # Check for emergency stop loss
                # This would require current PnL calculation
                # For now, just log position status
                self.logger.debug(f"📊 [SCHEDULER] Position {symbol}: {position_age_hours:.1f}h old")
            
            # Close positions that need to be closed
            for symbol in positions_to_close:
                await self._close_position(symbol, "max_duration_exceeded")
            
            # Schedule next monitoring
            next_monitor_time = time.time() + 60  # Every minute
            self.schedule_task(
                task_type='position_monitoring',
                scheduled_time=next_monitor_time,
                priority=2,
                data={}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [SCHEDULER] Error in position monitoring: {e}")
            return False
    
    async def _execute_trade_execution(self, task: ScheduledTask) -> bool:
        """Execute trade execution"""
        
        try:
            opportunity_data = task.data.get('opportunity', {})
            
            if not opportunity_data:
                self.logger.error("❌ [SCHEDULER] No opportunity data for execution")
                return False
            
            # Recreate opportunity object
            from src.core.strategies.funding_arbitrage import FundingArbitrageOpportunity
            opportunity = FundingArbitrageOpportunity(**opportunity_data)
            
            self.logger.info(f"🎯 [SCHEDULER] Executing funding arbitrage for {opportunity.symbol}")
            
            # Execute the trade
            result = await self.strategy.execute_funding_arbitrage(opportunity)
            
            if result['success']:
                self.daily_stats['trades_executed'] += 1
                self.logger.info(f"✅ [SCHEDULER] Trade executed successfully: {result['trade_id']}")
            else:
                self.logger.error(f"❌ [SCHEDULER] Trade execution failed: {result.get('error')}")
            
            return result['success']
            
        except Exception as e:
            self.logger.error(f"❌ [SCHEDULER] Error in trade execution: {e}")
            return False
    
    async def _execute_cleanup(self, task: ScheduledTask) -> bool:
        """Execute cleanup tasks"""
        
        try:
            self.logger.debug("🧹 [SCHEDULER] Executing cleanup tasks")
            
            # Clean up old tasks
            current_time = time.time()
            self.scheduled_tasks = [
                task for task in self.scheduled_tasks 
                if task.scheduled_time > current_time - 3600  # Keep tasks from last hour
            ]
            
            # Reset daily stats if needed
            if current_time - self.daily_stats['last_reset'] > 86400:  # 24 hours
                self.daily_stats = {
                    'trades_executed': 0,
                    'total_pnl': 0.0,
                    'start_time': current_time,
                    'last_reset': current_time
                }
                self.logger.info("📊 [SCHEDULER] Daily stats reset")
            
            # Schedule next cleanup
            next_cleanup_time = time.time() + 3600  # Every hour
            self.schedule_task(
                task_type='cleanup',
                scheduled_time=next_cleanup_time,
                priority=4,
                data={}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [SCHEDULER] Error in cleanup: {e}")
            return False
    
    def _can_execute_new_trade(self) -> bool:
        """Check if we can execute a new trade"""
        
        # Check daily trade limit
        if self.daily_stats['trades_executed'] >= self.config.max_daily_trades:
            return False
        
        # Check concurrent position limit
        if len(self.strategy.active_positions) >= self.config.max_concurrent_positions:
            return False
        
        # Check daily loss limit
        if self.daily_stats['total_pnl'] <= -self.config.max_daily_loss:
            return False
        
        return True
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        
        try:
            self.logger.info(f"🔄 [SCHEDULER] Closing position {symbol} - {reason}")
            
            # This would implement actual position closing logic
            # For now, just remove from active positions
            if symbol in self.strategy.active_positions:
                del self.strategy.active_positions[symbol]
                self.logger.info(f"✅ [SCHEDULER] Position {symbol} closed")
            
        except Exception as e:
            self.logger.error(f"❌ [SCHEDULER] Error closing position {symbol}: {e}")
    
    async def start(self):
        """Start the scheduler"""
        
        self.logger.info("🚀 [SCHEDULER] Starting Funding Arbitrage Scheduler")
        
        self.running = True
        
        # Schedule initial tasks
        self.schedule_funding_checks()
        self.schedule_regular_monitoring()
        
        # Main scheduler loop
        while self.running:
            try:
                current_time = time.time()
                
                # Execute ready tasks
                ready_tasks = [
                    task for task in self.scheduled_tasks 
                    if task.scheduled_time <= current_time
                ]
                
                for task in ready_tasks:
                    # Remove from scheduled tasks
                    self.scheduled_tasks.remove(task)
                    
                    # Execute task
                    success = await self.execute_task(task)
                    
                    if not success and task.retry_count < task.max_retries:
                        # Retry task
                        task.retry_count += 1
                        task.scheduled_time = current_time + (60 * task.retry_count)  # Exponential backoff
                        self.scheduled_tasks.append(task)
                        self.scheduled_tasks.sort(key=lambda x: (x.scheduled_time, x.priority))
                        
                        self.logger.warning(f"⚠️ [SCHEDULER] Retrying task {task.task_id} (attempt {task.retry_count})")
                
                # Sleep for a short time
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ [SCHEDULER] Error in scheduler loop: {e}")
                await asyncio.sleep(5)
    
    def stop(self):
        """Stop the scheduler"""
        
        self.logger.info("🛑 [SCHEDULER] Stopping Funding Arbitrage Scheduler")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        
        current_time = time.time()
        
        return {
            'running': self.running,
            'scheduled_tasks': len(self.scheduled_tasks),
            'active_positions': len(self.strategy.active_positions),
            'daily_stats': self.daily_stats.copy(),
            'next_funding_times': [
                datetime.fromtimestamp(ft).strftime('%Y-%m-%d %H:%M:%S UTC') 
                for ft in self.next_funding_times
            ],
            'next_scheduled_task': (
                datetime.fromtimestamp(self.scheduled_tasks[0].scheduled_time).strftime('%Y-%m-%d %H:%M:%S UTC')
                if self.scheduled_tasks else None
            )
        }
