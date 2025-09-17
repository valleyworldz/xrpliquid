"""
Enhanced Feature Engineering
Expanded feature set beyond MACD/EMA to address low confidence and weak win rate.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """Enhanced feature engineering with expanded feature set."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.feature_names = []
        self.feature_importance = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration."""
        return {
            'price_features': {
                'sma_periods': [5, 10, 20, 50, 100],
                'ema_periods': [5, 10, 20, 50],
                'rsi_periods': [14, 21],
                'macd_params': [(12, 26, 9), (5, 35, 5)],
                'bollinger_periods': [20, 50],
                'bollinger_std': [2, 2.5],
                'atr_periods': [14, 21],
                'stoch_params': [(14, 3, 3), (21, 5, 5)],
                'williams_r_periods': [14, 21],
                'cci_periods': [14, 20],
                'mfi_periods': [14, 21],
                'obv_enabled': True,
                'adx_periods': [14, 21],
                'parabolic_sar': True,
                'tema_periods': [10, 20],
                'kama_periods': [10, 20],
                'mama_params': [(0.5, 0.05), (0.3, 0.1)]
            },
            'volume_features': {
                'volume_sma_periods': [5, 10, 20],
                'volume_ratio_periods': [5, 10, 20],
                'vwap_periods': [10, 20, 50],
                'volume_price_trend': True,
                'ease_of_movement': True,
                'force_index_periods': [13, 21],
                'money_flow_index': True,
                'volume_oscillator': True
            },
            'volatility_features': {
                'volatility_periods': [10, 20, 30],
                'garch_volatility': True,
                'parkinson_volatility': True,
                'garman_klass_volatility': True,
                'rogers_satchell_volatility': True,
                'yang_zhang_volatility': True
            },
            'momentum_features': {
                'roc_periods': [5, 10, 20],
                'momentum_periods': [5, 10, 20],
                'rate_of_change': True,
                'ultimate_oscillator': True,
                'awesome_oscillator': True,
                'fisher_transform': True,
                'ichimoku_cloud': True,
                'supertrend': True
            },
            'market_microstructure': {
                'bid_ask_spread': True,
                'spread_ratio': True,
                'depth_imbalance': True,
                'order_flow_imbalance': True,
                'trade_size_distribution': True,
                'tick_direction': True,
                'volume_weighted_price': True,
                'time_weighted_price': True
            },
            'regime_features': {
                'volatility_regime': True,
                'trend_regime': True,
                'volume_regime': True,
                'market_state': True,
                'regime_transition': True
            },
            'cross_asset_features': {
                'correlation_features': True,
                'relative_strength': True,
                'cross_momentum': True,
                'spread_features': True
            },
            'time_features': {
                'hour_of_day': True,
                'day_of_week': True,
                'month_of_year': True,
                'is_weekend': True,
                'is_holiday': True,
                'time_since_open': True,
                'time_to_close': True
            },
            'feature_selection': {
                'n_features': 50,
                'selection_method': 'f_regression',
                'correlation_threshold': 0.95
            }
        }
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive feature set."""
        
        logger.info("Starting enhanced feature engineering")
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create feature dataframe
        features_df = pd.DataFrame(index=data.index)
        
        # Price-based features
        features_df = self._engineer_price_features(features_df, data)
        
        # Volume-based features
        features_df = self._engineer_volume_features(features_df, data)
        
        # Volatility features
        features_df = self._engineer_volatility_features(features_df, data)
        
        # Momentum features
        features_df = self._engineer_momentum_features(features_df, data)
        
        # Market microstructure features
        features_df = self._engineer_microstructure_features(features_df, data)
        
        # Regime features
        features_df = self._engineer_regime_features(features_df, data)
        
        # Cross-asset features
        features_df = self._engineer_cross_asset_features(features_df, data)
        
        # Time features
        features_df = self._engineer_time_features(features_df, data)
        
        # Advanced features
        features_df = self._engineer_advanced_features(features_df, data)
        
        # Clean features
        features_df = self._clean_features(features_df)
        
        # Feature selection
        features_df = self._select_features(features_df)
        
        logger.info(f"Feature engineering completed. Generated {len(features_df.columns)} features")
        
        return features_df
    
    def _engineer_price_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer price-based features."""
        
        config = self.config['price_features']
        
        # Simple Moving Averages
        for period in config['sma_periods']:
            features_df[f'sma_{period}'] = talib.SMA(data['close'], timeperiod=period)
            features_df[f'sma_ratio_{period}'] = data['close'] / features_df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in config['ema_periods']:
            features_df[f'ema_{period}'] = talib.EMA(data['close'], timeperiod=period)
            features_df[f'ema_ratio_{period}'] = data['close'] / features_df[f'ema_{period}']
        
        # RSI
        for period in config['rsi_periods']:
            features_df[f'rsi_{period}'] = talib.RSI(data['close'], timeperiod=period)
        
        # MACD
        for fast, slow, signal in config['macd_params']:
            macd, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
            features_df[f'macd_{fast}_{slow}_{signal}'] = macd
            features_df[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            features_df[f'macd_hist_{fast}_{slow}_{signal}'] = macd_hist
        
        # Bollinger Bands
        for period in config['bollinger_periods']:
            for std in config['bollinger_std']:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'], timeperiod=period, nbdevup=std, nbdevdn=std)
                features_df[f'bb_upper_{period}_{std}'] = bb_upper
                features_df[f'bb_middle_{period}_{std}'] = bb_middle
                features_df[f'bb_lower_{period}_{std}'] = bb_lower
                features_df[f'bb_width_{period}_{std}'] = (bb_upper - bb_lower) / bb_middle
                features_df[f'bb_position_{period}_{std}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        for period in config['atr_periods']:
            features_df[f'atr_{period}'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
            features_df[f'atr_ratio_{period}'] = features_df[f'atr_{period}'] / data['close']
        
        # Stochastic
        for k_period, d_period, smooth in config['stoch_params']:
            slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'], 
                                      fastk_period=k_period, slowk_period=smooth, slowd_period=d_period)
            features_df[f'stoch_k_{k_period}_{d_period}_{smooth}'] = slowk
            features_df[f'stoch_d_{k_period}_{d_period}_{smooth}'] = slowd
        
        # Williams %R
        for period in config['williams_r_periods']:
            features_df[f'williams_r_{period}'] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=period)
        
        # CCI
        for period in config['cci_periods']:
            features_df[f'cci_{period}'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=period)
        
        # MFI
        for period in config['mfi_periods']:
            features_df[f'mfi_{period}'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=period)
        
        # OBV
        if config['obv_enabled']:
            features_df['obv'] = talib.OBV(data['close'], data['volume'])
            features_df['obv_sma'] = talib.SMA(features_df['obv'], timeperiod=10)
            features_df['obv_ratio'] = features_df['obv'] / features_df['obv_sma']
        
        # ADX
        for period in config['adx_periods']:
            features_df[f'adx_{period}'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=period)
            features_df[f'plus_di_{period}'] = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=period)
            features_df[f'minus_di_{period}'] = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=period)
        
        # Parabolic SAR
        if config['parabolic_sar']:
            features_df['sar'] = talib.SAR(data['high'], data['low'])
            features_df['sar_signal'] = np.where(data['close'] > features_df['sar'], 1, -1)
        
        # TEMA
        for period in config['tema_periods']:
            features_df[f'tema_{period}'] = talib.TEMA(data['close'], timeperiod=period)
        
        # KAMA
        for period in config['kama_periods']:
            features_df[f'kama_{period}'] = talib.KAMA(data['close'], timeperiod=period)
        
        # MAMA
        for fast, slow in config['mama_params']:
            mama, fama = talib.MAMA(data['close'], fastlimit=fast, slowlimit=slow)
            features_df[f'mama_{fast}_{slow}'] = mama
            features_df[f'fama_{fast}_{slow}'] = fama
        
        return features_df
    
    def _engineer_volume_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer volume-based features."""
        
        config = self.config['volume_features']
        
        # Volume SMAs
        for period in config['volume_sma_periods']:
            features_df[f'volume_sma_{period}'] = talib.SMA(data['volume'], timeperiod=period)
            features_df[f'volume_ratio_{period}'] = data['volume'] / features_df[f'volume_sma_{period}']
        
        # VWAP
        for period in config['vwap_periods']:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()
            features_df[f'vwap_{period}'] = vwap
            features_df[f'vwap_ratio_{period}'] = data['close'] / vwap
        
        # Volume Price Trend
        if config['volume_price_trend']:
            features_df['vpt'] = talib.OBV(data['close'], data['volume'])
            features_df['vpt_sma'] = talib.SMA(features_df['vpt'], timeperiod=10)
            features_df['vpt_ratio'] = features_df['vpt'] / features_df['vpt_sma']
        
        # Ease of Movement
        if config['ease_of_movement']:
            distance = (data['high'] + data['low']) / 2 - (data['high'].shift(1) + data['low'].shift(1)) / 2
            box_height = data['volume'] / (data['high'] - data['low'])
            features_df['eom'] = distance / box_height
            features_df['eom_sma'] = talib.SMA(features_df['eom'], timeperiod=14)
        
        # Force Index
        for period in config['force_index_periods']:
            features_df[f'force_index_{period}'] = talib.AD(data['high'], data['low'], data['close'], data['volume'])
        
        # Money Flow Index
        if config['money_flow_index']:
            features_df['mfi'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=14)
        
        # Volume Oscillator
        if config['volume_oscillator']:
            vol_sma_short = talib.SMA(data['volume'], timeperiod=5)
            vol_sma_long = talib.SMA(data['volume'], timeperiod=20)
            features_df['volume_oscillator'] = (vol_sma_short - vol_sma_long) / vol_sma_long
        
        return features_df
    
    def _engineer_volatility_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer volatility-based features."""
        
        config = self.config['volatility_features']
        
        # Simple volatility
        for period in config['volatility_periods']:
            returns = data['close'].pct_change()
            features_df[f'volatility_{period}'] = returns.rolling(period).std()
            features_df[f'volatility_ratio_{period}'] = features_df[f'volatility_{period}'] / features_df[f'volatility_{period}'].rolling(50).mean()
        
        # GARCH volatility (simplified)
        if config['garch_volatility']:
            returns = data['close'].pct_change().dropna()
            # Simple GARCH(1,1) approximation
            features_df['garch_vol'] = returns.rolling(20).std() * np.sqrt(252)
        
        # Parkinson volatility
        if config['parkinson_volatility']:
            features_df['parkinson_vol'] = np.sqrt(0.25 * np.log(data['high'] / data['low']) ** 2)
        
        # Garman-Klass volatility
        if config['garman_klass_volatility']:
            features_df['gk_vol'] = np.sqrt(0.5 * np.log(data['high'] / data['low']) ** 2 - 
                                          (2 * np.log(2) - 1) * np.log(data['close'] / data['open']) ** 2)
        
        # Rogers-Satchell volatility
        if config['rogers_satchell_volatility']:
            features_df['rs_vol'] = np.sqrt(np.log(data['high'] / data['close']) * np.log(data['high'] / data['open']) +
                                          np.log(data['low'] / data['close']) * np.log(data['low'] / data['open']))
        
        # Yang-Zhang volatility
        if config['yang_zhang_volatility']:
            features_df['yz_vol'] = np.sqrt(features_df['rs_vol'] + 
                                          np.log(data['open'] / data['close'].shift(1)) ** 2)
        
        return features_df
    
    def _engineer_momentum_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer momentum-based features."""
        
        config = self.config['momentum_features']
        
        # Rate of Change
        for period in config['roc_periods']:
            features_df[f'roc_{period}'] = talib.ROC(data['close'], timeperiod=period)
        
        # Momentum
        for period in config['momentum_periods']:
            features_df[f'momentum_{period}'] = talib.MOM(data['close'], timeperiod=period)
        
        # Ultimate Oscillator
        if config['ultimate_oscillator']:
            features_df['ultimate_oscillator'] = talib.ULTOSC(data['high'], data['low'], data['close'])
        
        # Awesome Oscillator
        if config['awesome_oscillator']:
            try:
                features_df['awesome_oscillator'] = talib.AO(data['high'], data['low'])
            except AttributeError:
                # Fallback if AO not available
                sma_5 = talib.SMA((data['high'] + data['low']) / 2, timeperiod=5)
                sma_34 = talib.SMA((data['high'] + data['low']) / 2, timeperiod=34)
                features_df['awesome_oscillator'] = sma_5 - sma_34
        
        # Fisher Transform
        if config['fisher_transform']:
            features_df['fisher_transform'] = talib.FISHER(data['high'], data['low'])
        
        # Ichimoku Cloud
        if config['ichimoku_cloud']:
            try:
                tenkan, kijun, senkou_a, senkou_b, chikou = talib.ICHIMOKU(data['high'], data['low'], data['close'])
                features_df['tenkan'] = tenkan
                features_df['kijun'] = kijun
                features_df['senkou_a'] = senkou_a
                features_df['senkou_b'] = senkou_b
                features_df['chikou'] = chikou
                features_df['cloud_position'] = np.where(data['close'] > np.maximum(senkou_a, senkou_b), 1, 
                                                       np.where(data['close'] < np.minimum(senkou_a, senkou_b), -1, 0))
            except AttributeError:
                # Fallback if ICHIMOKU not available
                features_df['tenkan'] = talib.SMA((data['high'] + data['low']) / 2, timeperiod=9)
                features_df['kijun'] = talib.SMA((data['high'] + data['low']) / 2, timeperiod=26)
                features_df['senkou_a'] = (features_df['tenkan'] + features_df['kijun']) / 2
                features_df['senkou_b'] = talib.SMA((data['high'] + data['low']) / 2, timeperiod=52)
                features_df['chikou'] = data['close'].shift(-26)
                features_df['cloud_position'] = np.where(data['close'] > np.maximum(features_df['senkou_a'], features_df['senkou_b']), 1, 
                                                       np.where(data['close'] < np.minimum(features_df['senkou_a'], features_df['senkou_b']), -1, 0))
        
        # Supertrend
        if config['supertrend']:
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
            hl2 = (data['high'] + data['low']) / 2
            upper_band = hl2 + (2 * atr)
            lower_band = hl2 - (2 * atr)
            features_df['supertrend'] = np.where(data['close'] > upper_band.shift(1), lower_band, upper_band)
            features_df['supertrend_signal'] = np.where(data['close'] > features_df['supertrend'], 1, -1)
        
        return features_df
    
    def _engineer_microstructure_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer market microstructure features."""
        
        config = self.config['market_microstructure']
        
        # Bid-ask spread (simplified)
        if config['bid_ask_spread']:
            features_df['spread'] = data['high'] - data['low']
            features_df['spread_ratio'] = features_df['spread'] / data['close']
        
        # Depth imbalance (simplified)
        if config['depth_imbalance']:
            features_df['depth_imbalance'] = (data['high'] - data['close']) / (data['close'] - data['low'])
        
        # Order flow imbalance (simplified)
        if config['order_flow_imbalance']:
            features_df['order_flow_imbalance'] = (data['close'] - data['open']) / (data['high'] - data['low'])
        
        # Trade size distribution (simplified)
        if config['trade_size_distribution']:
            features_df['trade_size_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Tick direction
        if config['tick_direction']:
            features_df['tick_direction'] = np.where(data['close'] > data['close'].shift(1), 1, -1)
            features_df['tick_direction_sma'] = talib.SMA(features_df['tick_direction'], timeperiod=10)
        
        # Volume weighted price
        if config['volume_weighted_price']:
            features_df['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
            features_df['vwap_ratio'] = data['close'] / features_df['vwap']
        
        # Time weighted price
        if config['time_weighted_price']:
            features_df['twap'] = data['close'].rolling(20).mean()
            features_df['twap_ratio'] = data['close'] / features_df['twap']
        
        return features_df
    
    def _engineer_regime_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer regime-based features."""
        
        config = self.config['regime_features']
        
        # Volatility regime
        if config['volatility_regime']:
            volatility = data['close'].pct_change().rolling(20).std()
            vol_percentile = volatility.rolling(100).rank(pct=True)
            features_df['volatility_regime'] = np.where(vol_percentile > 0.8, 'high',
                                                      np.where(vol_percentile < 0.2, 'low', 'normal'))
        
        # Trend regime
        if config['trend_regime']:
            sma_short = talib.SMA(data['close'], timeperiod=20)
            sma_long = talib.SMA(data['close'], timeperiod=50)
            features_df['trend_regime'] = np.where(sma_short > sma_long, 'uptrend',
                                                 np.where(sma_short < sma_long, 'downtrend', 'sideways'))
        
        # Volume regime
        if config['volume_regime']:
            volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
            features_df['volume_regime'] = np.where(volume_ratio > 1.5, 'high',
                                                  np.where(volume_ratio < 0.5, 'low', 'normal'))
        
        # Market state
        if config['market_state']:
            features_df['market_state'] = np.where(
                (features_df['volatility_regime'] == 'high') & (features_df['trend_regime'] == 'uptrend'), 'bull_volatile',
                np.where(
                    (features_df['volatility_regime'] == 'high') & (features_df['trend_regime'] == 'downtrend'), 'bear_volatile',
                    np.where(
                        (features_df['volatility_regime'] == 'low') & (features_df['trend_regime'] == 'uptrend'), 'bull_calm',
                        np.where(
                            (features_df['volatility_regime'] == 'low') & (features_df['trend_regime'] == 'downtrend'), 'bear_calm',
                            'sideways'
                        )
                    )
                )
            )
        
        # Regime transition
        if config['regime_transition']:
            features_df['regime_transition'] = (features_df['market_state'] != features_df['market_state'].shift(1)).astype(int)
        
        return features_df
    
    def _engineer_cross_asset_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer cross-asset features."""
        
        config = self.config['cross_asset_features']
        
        # Correlation features (simplified)
        if config['correlation_features']:
            returns = data['close'].pct_change()
            features_df['correlation_short'] = returns.rolling(10).corr(returns.shift(1))
            features_df['correlation_long'] = returns.rolling(50).corr(returns.shift(1))
        
        # Relative strength
        if config['relative_strength']:
            features_df['relative_strength'] = data['close'] / data['close'].rolling(50).mean()
        
        # Cross momentum
        if config['cross_momentum']:
            features_df['cross_momentum'] = (data['close'] / data['close'].shift(10)) / (data['close'].shift(10) / data['close'].shift(20))
        
        # Spread features
        if config['spread_features']:
            features_df['price_spread'] = data['high'] - data['low']
            features_df['volume_spread'] = data['volume'] / data['volume'].rolling(20).mean()
        
        return features_df
    
    def _engineer_time_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based features."""
        
        config = self.config['time_features']
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return features_df
        
        # Hour of day
        if config['hour_of_day']:
            features_df['hour_of_day'] = data.index.hour
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)
        
        # Day of week
        if config['day_of_week']:
            features_df['day_of_week'] = data.index.dayofweek
            features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        
        # Month of year
        if config['month_of_year']:
            features_df['month_of_year'] = data.index.month
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month_of_year'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month_of_year'] / 12)
        
        # Weekend
        if config['is_weekend']:
            features_df['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # Time since open
        if config['time_since_open']:
            features_df['time_since_open'] = (data.index.hour * 60 + data.index.minute) / (24 * 60)
        
        # Time to close
        if config['time_to_close']:
            features_df['time_to_close'] = 1 - features_df['time_since_open']
        
        return features_df
    
    def _engineer_advanced_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features."""
        
        # Fractal dimension
        features_df['fractal_dimension'] = self._calculate_fractal_dimension(data['close'])
        
        # Hurst exponent
        features_df['hurst_exponent'] = self._calculate_hurst_exponent(data['close'])
        
        # Detrended fluctuation analysis
        features_df['dfa'] = self._calculate_dfa(data['close'])
        
        # Approximate entropy
        features_df['approximate_entropy'] = self._calculate_approximate_entropy(data['close'])
        
        # Sample entropy
        features_df['sample_entropy'] = self._calculate_sample_entropy(data['close'])
        
        return features_df
    
    def _calculate_fractal_dimension(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate fractal dimension using box-counting method."""
        # Simplified implementation
        returns = series.pct_change().dropna()
        return returns.rolling(window).apply(lambda x: len(np.unique(np.round(x, 2))) / len(x))
    
    def _calculate_hurst_exponent(self, series: pd.Series, window: int = 50) -> pd.Series:
        """Calculate Hurst exponent."""
        # Simplified implementation
        returns = series.pct_change().dropna()
        return returns.rolling(window).apply(lambda x: 0.5 + 0.5 * np.corrcoef(x[:-1], x[1:])[0, 1])
    
    def _calculate_dfa(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Detrended Fluctuation Analysis."""
        # Simplified implementation
        returns = series.pct_change().dropna()
        return returns.rolling(window).std()
    
    def _calculate_approximate_entropy(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate approximate entropy."""
        # Simplified implementation
        returns = series.pct_change().dropna()
        return returns.rolling(window).apply(lambda x: -np.sum(x * np.log(np.abs(x) + 1e-10)))
    
    def _calculate_sample_entropy(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate sample entropy."""
        # Simplified implementation
        returns = series.pct_change().dropna()
        return returns.rolling(window).apply(lambda x: np.std(x))
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values
        features_df = features_df.fillna(method='ffill')
        
        # Drop rows with all NaN values
        features_df = features_df.dropna(how='all')
        
        # Remove constant features
        constant_features = features_df.columns[features_df.nunique() <= 1]
        if len(constant_features) > 0:
            logger.warning(f"Removing {len(constant_features)} constant features")
            features_df = features_df.drop(columns=constant_features)
        
        return features_df
    
    def _select_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Select most important features."""
        
        config = self.config['feature_selection']
        n_features = config['n_features']
        
        if len(features_df.columns) <= n_features:
            return features_df
        
        # Remove highly correlated features
        corr_matrix = features_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > config['correlation_threshold'])]
        
        if len(to_drop) > 0:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            features_df = features_df.drop(columns=to_drop)
        
        # Select top features using variance
        if len(features_df.columns) > n_features:
            feature_selector = SelectKBest(f_regression, k=n_features)
            # Use dummy target for feature selection
            dummy_target = np.random.randn(len(features_df))
            feature_selector.fit(features_df.fillna(0), dummy_target)
            
            selected_features = features_df.columns[feature_selector.get_support()]
            features_df = features_df[selected_features]
            
            self.feature_selector = feature_selector
        
        return features_df
    
    def fit_scaler(self, features_df: pd.DataFrame):
        """Fit the scaler on training data."""
        self.scaler.fit(features_df.fillna(0))
        self.feature_names = features_df.columns.tolist()
    
    def transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        # Ensure same columns as training data
        missing_cols = set(self.feature_names) - set(features_df.columns)
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            for col in missing_cols:
                features_df[col] = 0
        
        # Reorder columns
        features_df = features_df[self.feature_names]
        
        # Transform
        scaled_features = self.scaler.transform(features_df.fillna(0))
        
        return pd.DataFrame(scaled_features, index=features_df.index, columns=self.feature_names)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance
    
    def save_feature_config(self, filepath: str = "config/feature_engineering_config.json"):
        """Save feature engineering configuration."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Feature engineering configuration saved to {filepath}")

def main():
    """Demonstrate enhanced feature engineering."""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # Generate realistic OHLCV data
    base_price = 0.52
    returns = np.random.normal(0, 0.01, 1000)
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 1000))),
        'close': prices,
        'volume': np.random.lognormal(8, 1, 1000)
    }, index=dates)
    
    # Initialize feature engineer
    feature_engineer = EnhancedFeatureEngineer()
    
    print("🧪 Testing Enhanced Feature Engineering")
    print("=" * 50)
    
    # Engineer features
    features_df = feature_engineer.engineer_features(sample_data)
    
    print(f"Generated {len(features_df.columns)} features")
    print(f"Feature shape: {features_df.shape}")
    
    # Show feature categories
    feature_categories = {
        'Price': [col for col in features_df.columns if any(x in col for x in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr'])],
        'Volume': [col for col in features_df.columns if 'volume' in col or 'vwap' in col or 'obv' in col],
        'Volatility': [col for col in features_df.columns if 'vol' in col or 'volatility' in col],
        'Momentum': [col for col in features_df.columns if any(x in col for x in ['roc', 'momentum', 'oscillator'])],
        'Microstructure': [col for col in features_df.columns if any(x in col for x in ['spread', 'imbalance', 'tick'])],
        'Time': [col for col in features_df.columns if any(x in col for x in ['hour', 'day', 'month', 'weekend'])],
        'Advanced': [col for col in features_df.columns if any(x in col for x in ['fractal', 'hurst', 'entropy'])]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"\n{category} Features ({len(features)}):")
            for feature in features[:5]:  # Show first 5
                print(f"  - {feature}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")
    
    # Fit scaler
    feature_engineer.fit_scaler(features_df)
    
    # Transform features
    scaled_features = feature_engineer.transform_features(features_df)
    
    print(f"\nScaled features shape: {scaled_features.shape}")
    print(f"Feature statistics:")
    print(f"  Mean: {scaled_features.mean().mean():.4f}")
    print(f"  Std: {scaled_features.std().mean():.4f}")
    print(f"  Min: {scaled_features.min().min():.4f}")
    print(f"  Max: {scaled_features.max().max():.4f}")
    
    # Save configuration
    feature_engineer.save_feature_config()
    
    print("\n✅ Enhanced feature engineering completed")
    
    return 0

if __name__ == "__main__":
    exit(main())
