#!/usr/bin/env python3
"""
🔧 DYNAMIC HYPERPARAMETER OPTIMIZATION MANAGER
==============================================

Automated hyperparameter tuning for trading strategies using Optuna.
Continuously optimizes strategy parameters based on historical performance.

Features:
- Multi-strategy optimization (scalping, grid, mean reversion, RL)
- Paper trading simulation for parameter validation
- Automated parameter updates in config
- Performance tracking and logging
"""

from src.core.utils.decimal_boundary_guard import safe_float
import optuna
import numpy as np
from typing import Dict, Any, Tuple, Optional
from core.utils.config_manager import ConfigManager
from core.utils.logger import Logger
from core.paper_simulator import PaperSimulator

class HPOManager:
    """
    Dynamic Hyperparameter Optimization Manager
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = Logger()
        self.simulator = PaperSimulator()
        self.study = None
        self.optimization_history = []
        
    def _objective_scalping(self, trial) -> float:
        """Objective function for scalping strategy optimization"""
        try:
            # Suggest scalping parameters
            params = {
                "spread_threshold": trial.suggest_float("spread_threshold", 0.0001, 0.0020),
                "tp_pips": trial.suggest_int("tp_pips", 1, 10),
                "sl_pips": trial.suggest_int("sl_pips", 1, 10),
                "min_volume": trial.suggest_float("min_volume", 0.001, 0.01),
                "max_orders": trial.suggest_int("max_orders", 3, 10)
            }
            
            # Run simulation
            pnl, volatility, sharpe = self.simulator.simulate(
                strategy="scalping",
                params=params,
                lookback_minutes=self.config.get("hpo.lookback_minutes", 240),
                token=self.config.get("hpo.test_token", "DOGE")
            )
            
            # Log trial results
            self.logger.info(f"[HPO] Scalping trial - PnL: {pnl:.4f}, Vol: {volatility:.4f}, Sharpe: {sharpe:.4f}")
            
            return sharpe if not np.isnan(sharpe) else -safe_float("inf")
            
        except Exception as e:
            self.logger.error(f"[HPO] Error in scalping objective: {e}")
            return -safe_float("inf")
    
    def _objective_grid_trading(self, trial) -> float:
        """Objective function for grid trading strategy optimization"""
        try:
            params = {
                "grid_size": trial.suggest_float("grid_size", 0.0005, 0.005),
                "num_grids": trial.suggest_int("num_grids", 5, 20),
                "order_quantity": trial.suggest_float("order_quantity", 0.001, 0.01),
                "grid_spread": trial.suggest_float("grid_spread", 0.001, 0.01)
            }
            
            pnl, volatility, sharpe = self.simulator.simulate(
                strategy="grid_trading",
                params=params,
                lookback_minutes=self.config.get("hpo.lookback_minutes", 240),
                token=self.config.get("hpo.test_token", "DOGE")
            )
            
            self.logger.info(f"[HPO] Grid trial - PnL: {pnl:.4f}, Vol: {volatility:.4f}, Sharpe: {sharpe:.4f}")
            
            return sharpe if not np.isnan(sharpe) else -safe_float("inf")
            
        except Exception as e:
            self.logger.error(f"[HPO] Error in grid objective: {e}")
            return -safe_float("inf")
    
    def _objective_mean_reversion(self, trial) -> float:
        """Objective function for mean reversion strategy optimization"""
        try:
            params = {
                "entry_deviation": trial.suggest_float("entry_deviation", 0.0001, 0.002),
                "exit_deviation": trial.suggest_float("exit_deviation", 0.0002, 0.003),
                "max_position_size": trial.suggest_float("max_position_size", 0.005, 0.05),
                "standard_deviations": trial.suggest_float("standard_deviations", 1.5, 3.0),
                "lookback_period": trial.suggest_int("lookback_period", 20, 100)
            }
            
            pnl, volatility, sharpe = self.simulator.simulate(
                strategy="mean_reversion",
                params=params,
                lookback_minutes=self.config.get("hpo.lookback_minutes", 240),
                token=self.config.get("hpo.test_token", "DOGE")
            )
            
            self.logger.info(f"[HPO] Mean reversion trial - PnL: {pnl:.4f}, Vol: {volatility:.4f}, Sharpe: {sharpe:.4f}")
            
            return sharpe if not np.isnan(sharpe) else -safe_float("inf")
            
        except Exception as e:
            self.logger.error(f"[HPO] Error in mean reversion objective: {e}")
            return -safe_float("inf")
    
    def run_scalping_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization for scalping strategy"""
        self.logger.info("[HPO] Starting scalping hyperparameter optimization")
        
        try:
            self.study = optuna.create_study(
                direction="maximize",
                study_name="scalping_hpo",
                storage=None  # In-memory storage for simplicity
            )
            
            self.study.optimize(
                self._objective_scalping,
                n_trials=self.config.get("hpo.scalping_trials", 20),
                timeout=self.config.get("hpo.timeout_minutes", 30) * 60
            )
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            # Update config with best parameters
            self.config.set("strategies.scalping.params", best_params)
            self.config.set("strategies.scalping.optimized_sharpe", best_value)
            
            # Log optimization results
            self.logger.info(f"[HPO] Scalping optimization complete")
            self.logger.info(f"[HPO] Best Sharpe: {best_value:.4f}")
            self.logger.info(f"[HPO] Best params: {best_params}")
            
            # Store optimization history
            self.optimization_history.append({
                "strategy": "scalping",
                "best_params": best_params,
                "best_sharpe": best_value,
                "n_trials": len(self.study.trials)
            })
            
            return {
                "strategy": "scalping",
                "best_params": best_params,
                "best_sharpe": best_value,
                "n_trials": len(self.study.trials)
            }
            
        except Exception as e:
            self.logger.error(f"[HPO] Error in scalping optimization: {e}")
            return {}
    
    def run_grid_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization for grid trading strategy"""
        self.logger.info("[HPO] Starting grid trading hyperparameter optimization")
        
        try:
            self.study = optuna.create_study(
                direction="maximize",
                study_name="grid_hpo",
                storage=None
            )
            
            self.study.optimize(
                self._objective_grid_trading,
                n_trials=self.config.get("hpo.grid_trials", 20),
                timeout=self.config.get("hpo.timeout_minutes", 30) * 60
            )
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            # Update config
            self.config.set("strategies.grid_trading.params", best_params)
            self.config.set("strategies.grid_trading.optimized_sharpe", best_value)
            
            self.logger.info(f"[HPO] Grid optimization complete - Best Sharpe: {best_value:.4f}")
            
            return {
                "strategy": "grid_trading",
                "best_params": best_params,
                "best_sharpe": best_value,
                "n_trials": len(self.study.trials)
            }
            
        except Exception as e:
            self.logger.error(f"[HPO] Error in grid optimization: {e}")
            return {}
    
    def run_mean_reversion_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization for mean reversion strategy"""
        self.logger.info("[HPO] Starting mean reversion hyperparameter optimization")
        
        try:
            self.study = optuna.create_study(
                direction="maximize",
                study_name="mean_reversion_hpo",
                storage=None
            )
            
            self.study.optimize(
                self._objective_mean_reversion,
                n_trials=self.config.get("hpo.mean_reversion_trials", 20),
                timeout=self.config.get("hpo.timeout_minutes", 30) * 60
            )
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            # Update config
            self.config.set("strategies.mean_reversion.params", best_params)
            self.config.set("strategies.mean_reversion.optimized_sharpe", best_value)
            
            self.logger.info(f"[HPO] Mean reversion optimization complete - Best Sharpe: {best_value:.4f}")
            
            return {
                "strategy": "mean_reversion",
                "best_params": best_params,
                "best_sharpe": best_value,
                "n_trials": len(self.study.trials)
            }
            
        except Exception as e:
            self.logger.error(f"[HPO] Error in mean reversion optimization: {e}")
            return {}
    
    def run_all_optimizations(self) -> Dict[str, Any]:
        """Run optimization for all strategies"""
        self.logger.info("[HPO] Starting comprehensive strategy optimization")
        
        results = {}
        
        # Run optimizations for each strategy
        if self.config.get("strategies.scalping.enabled", True):
            results["scalping"] = self.run_scalping_optimization()
        
        if self.config.get("strategies.grid_trading.enabled", True):
            results["grid_trading"] = self.run_grid_optimization()
        
        if self.config.get("strategies.mean_reversion.enabled", False):
            results["mean_reversion"] = self.run_mean_reversion_optimization()
        
        # Log summary
        self.logger.info("[HPO] All optimizations complete")
        for strategy, result in results.items():
            if result:
                self.logger.info(f"[HPO] {strategy}: Sharpe {result.get('best_sharpe', 0):.4f}")
        
        return results
    
    def get_optimization_history(self) -> list:
        """Get optimization history"""
        return self.optimization_history
    
    def get_best_params(self, strategy: str) -> Dict[str, Any]:
        """Get best parameters for a specific strategy"""
        return self.config.get(f"strategies.{strategy}.params", {})
    
    def get_best_sharpe(self, strategy: str) -> float:
        """Get best Sharpe ratio for a specific strategy"""
        return self.config.get(f"strategies.{strategy}.optimized_sharpe", 0.0)
