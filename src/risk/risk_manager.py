import logging
from typing import Optional

# Import configuration
try:
    from src.core.config import config
except ImportError:
    class FallbackConfig:
        MAX_DRAWDOWN_PCT = 0.03
        KELLY_MULTIPLIER = 0.3
        VOLATILITY_THRESHOLD = 0.005
        ATR_STOP_MULTIPLIER = 0.8
    config = FallbackConfig()

class AdvancedRiskManager:
    """Advanced risk management for XRP trading"""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.max_drawdown = getattr(config, 'MAX_DRAWDOWN_PCT', 0.03)
        self.kelly_multiplier = getattr(config, 'KELLY_MULTIPLIER', 0.3)
        self.volatility_lookback = getattr(config, 'VOLATILITY_THRESHOLD', 0.005)

    def calculate_kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float, free_collateral: float) -> float:
        """Calculate Kelly position size using R-multiples instead of dollar PnL"""
        try:
            if win_rate <= 0 or win_rate >= 1:
                return free_collateral * 0.02  # Conservative default
            if avg_win <= 0:
                return free_collateral * 0.01  # Very conservative if no wins
            if avg_loss <= 0:
                return free_collateral * 0.05  # 5% position size
            # Conservative ATR estimate
            current_atr = 0.001
            estimated_risk_per_unit = current_atr * 1.2
            if estimated_risk_per_unit <= 0:
                estimated_risk_per_unit = 0.001
            avg_win_r = avg_win / estimated_risk_per_unit
            avg_loss_r = abs(avg_loss) / estimated_risk_per_unit
            b = avg_win_r / avg_loss_r
            p = win_rate
            q = 1 - win_rate
            raw_kelly = max(0, (b * p - q) / b)
            if raw_kelly <= 0:
                raw_kelly = 0.05
            kelly_fraction = raw_kelly * self.kelly_multiplier
            kelly_fraction = max(0.01, min(kelly_fraction, 0.25))
            position_size = free_collateral * kelly_fraction
            self.logger.info(f"💰 Kelly calculation: win_rate={win_rate:.2%}, avg_win_r={avg_win_r:.2f}, avg_loss_r={avg_loss_r:.2f}, raw_kelly={raw_kelly:.2%}, final={kelly_fraction:.2%}")
            return position_size
        except Exception as e:
            self.logger.error(f"❌ Error calculating Kelly position size: {e}")
            return free_collateral * 0.02

    def calculate_dynamic_stop_loss(self, entry_price: float, volatility: float, position_type: str = "LONG") -> float:
        """Calculate dynamic stop loss based on volatility"""
        atr_multiplier = getattr(config, 'ATR_STOP_MULTIPLIER', 0.8)
        if position_type == "LONG":
            stop_loss = entry_price * (1 - volatility * atr_multiplier)
        else:
            stop_loss = entry_price * (1 + volatility * atr_multiplier)
        return stop_loss

    def should_reduce_risk(self, current_drawdown: float, volatility: float) -> bool:
        """Determine if risk should be reduced"""
        if current_drawdown > self.max_drawdown:
            return True
        if volatility > self.volatility_lookback:
            return True
        return False 