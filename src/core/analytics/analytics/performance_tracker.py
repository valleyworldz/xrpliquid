class PerformanceTracker:
    def __init__(self, config=None):
        """Initialize PerformanceTracker with optional config parameter"""
        self.config = config
        
    def track_metrics(self, trades):
        """
        Calculates and tracks key performance indicators (KPIs) from a list of trades.

        Args:
            trades (list): A list of executed trade objects or dictionaries.

        Returns:
            dict: A dictionary containing calculated metrics like win rate, profit factor,
                  Sharpe ratio, max drawdown, etc.
        """
        print("Tracking performance metrics from trades:", trades)
        total_trades = len(trades)
        profitable_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0) # Sum of negative PnL
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            # ... other metrics like Sharpe Ratio, Max Drawdown (more complex to calculate here)
        }

    def analyze_equity_curve(self, historical_pnl):
        """
        Analyzes the equity curve based on historical PnL data.

        Args:
            historical_pnl (list): A list of cumulative PnL values over time.

        Returns:
            dict: Metrics related to the equity curve, such as its slope, maximum drawdown, etc.
        """
        print("Analyzing equity curve...")
        if not historical_pnl:
            return {"slope": 0, "max_drawdown": 0}

        # Simple slope calculation (end PnL - start PnL) / number of points
        slope = (historical_pnl[-1] - historical_pnl[0]) / len(historical_pnl)

        # Max Drawdown calculation
        max_drawdown = 0
        peak = historical_pnl[0]
        for pnl in historical_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {"slope": slope, "max_drawdown": max_drawdown}

    def track_slippage(self, expected_price, executed_price, quantity):
        """
        Tracks slippage for a given trade.

        Args:
            expected_price (float): The price at which the order was expected to be filled.
            executed_price (float): The actual price at which the order was filled.
            quantity (float): The quantity of the asset traded.

        Returns:
            float: The slippage amount in currency terms.
        """
        slippage_per_unit = abs(executed_price - expected_price)
        total_slippage = slippage_per_unit * quantity
        print(f"Slippage tracked: {total_slippage}")
        return total_slippage

    def auto_adjust_strategy_params(self, strategy_name, win_rate, threshold=0.5):
        """
        Placeholder for automatically adjusting strategy parameters based on win rate.

        Args:
            strategy_name (str): The name of the strategy.
            win_rate (float): The current win rate of the strategy.
            threshold (float): The win rate threshold below which parameters should be adjusted.
        """
        if win_rate < threshold:
            print(f"Win rate for {strategy_name} ({win_rate:.2f}) is below threshold ({threshold:.2f}). Adjusting parameters...")
            # This would involve loading the strategy's config, modifying it,
            # and potentially reloading the strategy or notifying a higher-level manager.
            # Example: Increase stop-loss, decrease position size, or switch to a different sub-strategy.
        else:
            print(f"Win rate for {strategy_name} ({win_rate:.2f}) is healthy.")


