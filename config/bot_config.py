from dataclasses import dataclass

@dataclass
class BotConfig:
    """Centralized configuration for XRP trading bot."""
    min_xrp: int = 10  # Minimum XRP position size
    min_notional: float = 10.0  # Minimum notional value per trade
    risk_per_trade: float = 0.02  # Fraction of free collateral to risk per trade
    confidence_threshold: float = 0.7  # Minimum confidence to enter a trade
    atr_period: int = 14  # ATR lookback period
    atr_multiplier_tp: float = 2.0  # Take-profit = entry + N*ATR
    atr_multiplier_sl: float = 1.0  # Stop-loss = entry - N*ATR
    post_trade_cooldown_minutes: float = 5.0  # Cooldown after each trade
    funding_rate_buffer: float = 0.0005  # Funding rate buffer for skipping trades
    funding_rate_threshold: float = 0.0005  # Funding rate threshold for skipping trades
    stop_loss_pct: float = 0.03  # Maximum stop loss per trade (as a fraction, e.g., 0.03 for 3%)
    profit_target_pct: float = 0.035  # Take profit percentage (as a fraction, e.g., 0.035 for 3.5%)
    max_daily_loss_pct: float = 0.05  # Max daily loss as % of starting capital
    max_drawdown_pct: float = 0.15  # Max drawdown before pausing trading
    max_consecutive_losses: int = 5  # Max consecutive losses before pausing
    min_hold_time_minutes: float = 15.0  # Minimum hold time for a position
    max_hold_time_hours: float = 2.0  # Maximum hold time for a position (in hours)
    volatility_threshold: float = 0.02  # Threshold for high/low volatility regime
    atr_period_high_vol: int = 7  # ATR period in high volatility
    atr_period_low_vol: int = 21  # ATR period in low volatility
    volume_threshold: float = 1000000.0  # Minimum 24h volume required to trade
    # Add more tunables as needed 