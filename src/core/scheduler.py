#!/usr/bin/env python3
"""
⏰ AUTOMATED TRADING SCHEDULER
==============================

24/7 automated trading scheduler with periodic tasks.
Orchestrates all trading operations and maintenance tasks.

Features:
- Portfolio rebalancing
- Hyperparameter optimization
- Risk monitoring
- Market data updates
- Performance reporting
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from core.utils.config_manager import ConfigManager
from core.utils.logger import Logger
from core.hpo_manager import HPOManager
from core.portfolio_manager import PortfolioManager
from core.engines.risk_management import RiskManagement
from core.api.hyperliquid_api import HyperliquidAPI

class TradingScheduler:
    """
    Automated Trading Scheduler for 24/7 Operations
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = Logger()
        self.api = HyperliquidAPI()
        
        # Initialize managers
        self.hpo_manager = HPOManager(config)
        self.risk_manager = RiskManagement(config)
        
        # Portfolio manager will be initialized after strategy manager
        self.portfolio_manager = None
        
        # Scheduler state
        self.running = False
        self.tasks = {}
        self.task_threads = {}
        self.last_execution = {}
        
        # Task intervals (in seconds)
        self.intervals = {
            "risk_monitoring": config.get("scheduler.risk_monitoring_seconds", 30),
            "portfolio_rebalancing": config.get("scheduler.portfolio_rebalancing_minutes", 15) * 60,
            "hpo_optimization": config.get("scheduler.hpo_optimization_hours", 4) * 3600,
            "market_data_update": config.get("scheduler.market_data_seconds", 10),
            "performance_report": config.get("scheduler.performance_report_hours", 1) * 3600,
            "health_check": config.get("scheduler.health_check_minutes", 5) * 60
        }
        
        self.logger.info("[SCHEDULER] Trading scheduler initialized")
    
    def set_portfolio_manager(self, portfolio_manager: PortfolioManager):
        """Set portfolio manager after initialization"""
        self.portfolio_manager = portfolio_manager
        self.logger.info("[SCHEDULER] Portfolio manager set")
    
    def start(self):
        """Start the automated trading scheduler"""
        try:
            if self.running:
                self.logger.warning("[SCHEDULER] Scheduler is already running")
                return
            
            self.running = True
            self.logger.info("[SCHEDULER] Starting automated trading scheduler")
            
            # Start all scheduled tasks
            self._start_task("risk_monitoring", self._risk_monitoring_task)
            self._start_task("portfolio_rebalancing", self._portfolio_rebalancing_task)
            self._start_task("hpo_optimization", self._hpo_optimization_task)
            self._start_task("market_data_update", self._market_data_update_task)
            self._start_task("performance_report", self._performance_report_task)
            self._start_task("health_check", self._health_check_task)
            
            self.logger.info("[SCHEDULER] All tasks started successfully")
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error starting scheduler: {e}")
            self.running = False
    
    def stop(self):
        """Stop the automated trading scheduler"""
        try:
            if not self.running:
                self.logger.warning("[SCHEDULER] Scheduler is not running")
                return
            
            self.running = False
            self.logger.info("[SCHEDULER] Stopping automated trading scheduler")
            
            # Stop all task threads
            for task_name, thread in self.task_threads.items():
                if thread and thread.is_alive():
                    thread.join(timeout=5)
                    self.logger.info(f"[SCHEDULER] Task {task_name} stopped")
            
            self.task_threads.clear()
            self.logger.info("[SCHEDULER] All tasks stopped")
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error stopping scheduler: {e}")
    
    def _start_task(self, task_name: str, task_function: Callable):
        """Start a specific task in a separate thread"""
        try:
            if task_name in self.task_threads and self.task_threads[task_name].is_alive():
                self.logger.warning(f"[SCHEDULER] Task {task_name} is already running")
                return
            
            thread = threading.Thread(
                target=self._task_runner,
                args=(task_name, task_function),
                daemon=True,
                name=f"Scheduler-{task_name}"
            )
            
            self.task_threads[task_name] = thread
            thread.start()
            
            self.logger.info(f"[SCHEDULER] Task {task_name} started")
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error starting task {task_name}: {e}")
    
    def _task_runner(self, task_name: str, task_function: Callable):
        """Run a task in a loop with specified interval"""
        try:
            interval = self.intervals.get(task_name, 60)
            
            while self.running:
                try:
                    # Execute task
                    start_time = time.time()
                    task_function()
                    execution_time = time.time() - start_time
                    
                    # Update last execution time
                    self.last_execution[task_name] = datetime.now()
                    
                    # Log execution
                    self.logger.debug(f"[SCHEDULER] Task {task_name} completed in {execution_time:.2f}s")
                    
                    # Sleep for remaining interval
                    sleep_time = max(0, interval - execution_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"[SCHEDULER] Error in task {task_name}: {e}")
                    # Sleep for a shorter interval on error
                    time.sleep(min(interval, 60))
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Fatal error in task runner {task_name}: {e}")
    
    def _risk_monitoring_task(self):
        """Continuous risk monitoring task"""
        try:
            risk_summary = self.risk_manager.monitor_portfolio_risk()
            
            if risk_summary.get("status") == "critical":
                self.logger.critical(f"[SCHEDULER] Critical risk detected: {risk_summary}")
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error in risk monitoring task: {e}")
    
    def _portfolio_rebalancing_task(self):
        """Portfolio rebalancing task"""
        try:
            if not self.portfolio_manager:
                self.logger.warning("[SCHEDULER] Portfolio manager not available")
                return
            
            self.logger.info("[SCHEDULER] Starting portfolio rebalancing")
            results = self.portfolio_manager.rebalance_portfolio()
            
            if results:
                self.logger.info(f"[SCHEDULER] Portfolio rebalancing completed: {len(results)} tokens processed")
            else:
                self.logger.warning("[SCHEDULER] Portfolio rebalancing failed or no changes needed")
                
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error in portfolio rebalancing task: {e}")
    
    def _hpo_optimization_task(self):
        """Hyperparameter optimization task"""
        try:
            self.logger.info("[SCHEDULER] Starting hyperparameter optimization")
            
            # Run optimization for all strategies
            results = self.hpo_manager.run_all_optimizations()
            
            if results:
                self.logger.info(f"[SCHEDULER] HPO completed: {len(results)} strategies optimized")
                for strategy, result in results.items():
                    if result:
                        self.logger.info(f"[SCHEDULER] {strategy}: Sharpe {result.get('best_sharpe', 0):.4f}")
            else:
                self.logger.warning("[SCHEDULER] HPO failed or no strategies to optimize")
                
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error in HPO task: {e}")
    
    def _market_data_update_task(self):
        """Market data update task"""
        try:
            # Update market data for all tracked tokens
            tokens = self.config.get("portfolio.tokens", ["DOGE", "ETH", "SOL", "BTC", "TRUMP"])
            
            for token in tokens:
                try:
                    market_data = self.api.get_market_data(token)
                    if market_data:
                        self.logger.debug(f"[SCHEDULER] Updated market data for {token}")
                    else:
                        self.logger.warning(f"[SCHEDULER] Failed to get market data for {token}")
                except Exception as e:
                    self.logger.error(f"[SCHEDULER] Error updating market data for {token}: {e}")
                    
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error in market data update task: {e}")
    
    def _performance_report_task(self):
        """Performance reporting task"""
        try:
            self.logger.info("[SCHEDULER] Generating performance report")
            
            # Get portfolio summary
            if self.portfolio_manager:
                portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                self.logger.info(f"[SCHEDULER] Portfolio value: {portfolio_summary.get('portfolio_value', 0):.2f}")
            
            # Get risk summary
            risk_summary = self.risk_manager.get_risk_summary()
            if risk_summary and "risk_metrics" in risk_summary:
                metrics = risk_summary["risk_metrics"]
                self.logger.info(f"[SCHEDULER] Risk status: {metrics.get('status', 'unknown')}")
                self.logger.info(f"[SCHEDULER] Drawdown: {metrics.get('overall_drawdown', 0):.2%}")
            
            # Get HPO history
            hpo_history = self.hpo_manager.get_optimization_history()
            if hpo_history:
                self.logger.info(f"[SCHEDULER] HPO runs completed: {len(hpo_history)}")
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error in performance report task: {e}")
    
    def _health_check_task(self):
        """System health check task"""
        try:
            # Check API connectivity
            try:
                user_state = self.api.get_user_state()
                if user_state:
                    self.logger.debug("[SCHEDULER] API connectivity: OK")
                else:
                    self.logger.warning("[SCHEDULER] API connectivity: Failed to get user state")
            except Exception as e:
                self.logger.error(f"[SCHEDULER] API connectivity error: {e}")
            
            # Check task health
            for task_name, thread in self.task_threads.items():
                if thread and thread.is_alive():
                    self.logger.debug(f"[SCHEDULER] Task {task_name}: Running")
                else:
                    self.logger.warning(f"[SCHEDULER] Task {task_name}: Not running")
            
            # Check memory usage (simplified)
            import psutil
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 80:
                self.logger.warning(f"[SCHEDULER] High memory usage: {memory_usage:.1f}%")
            else:
                self.logger.debug(f"[SCHEDULER] Memory usage: {memory_usage:.1f}%")
                
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error in health check task: {e}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        try:
            status = {
                "running": self.running,
                "tasks": {},
                "last_execution": {},
                "intervals": self.intervals
            }
            
            # Task status
            for task_name, thread in self.task_threads.items():
                status["tasks"][task_name] = {
                    "running": thread.is_alive() if thread else False,
                    "daemon": thread.daemon if thread else False
                }
            
            # Last execution times
            for task_name, last_time in self.last_execution.items():
                status["last_execution"][task_name] = last_time.isoformat()
            
            return status
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error getting scheduler status: {e}")
            return {"error": str(e)}
    
    def add_custom_task(self, task_name: str, task_function: Callable, interval_seconds: int):
        """Add a custom task to the scheduler"""
        try:
            if task_name in self.intervals:
                self.logger.warning(f"[SCHEDULER] Task {task_name} already exists")
                return False
            
            self.intervals[task_name] = interval_seconds
            self._start_task(task_name, task_function)
            
            self.logger.info(f"[SCHEDULER] Custom task {task_name} added with {interval_seconds}s interval")
            return True
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error adding custom task {task_name}: {e}")
            return False
    
    def remove_task(self, task_name: str):
        """Remove a task from the scheduler"""
        try:
            if task_name not in self.intervals:
                self.logger.warning(f"[SCHEDULER] Task {task_name} does not exist")
                return False
            
            # Stop the task thread
            if task_name in self.task_threads:
                thread = self.task_threads[task_name]
                if thread and thread.is_alive():
                    # Note: We can't easily stop a thread, so we'll just remove it from tracking
                    pass
                del self.task_threads[task_name]
            
            # Remove from intervals
            del self.intervals[task_name]
            
            # Remove from last execution tracking
            if task_name in self.last_execution:
                del self.last_execution[task_name]
            
            self.logger.info(f"[SCHEDULER] Task {task_name} removed")
            return True
            
        except Exception as e:
            self.logger.error(f"[SCHEDULER] Error removing task {task_name}: {e}")
            return False
    
    def wait(self):
        """Wait for the scheduler to complete (for main thread)"""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("[SCHEDULER] Received interrupt signal")
            self.stop()
