"""
Simplified Feature Engineering
Expanded feature set beyond MACD/EMA with robust fallbacks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SimplifiedFeatureEngineer:
    """Simplified feature engineering with robust fallbacks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.feature_names = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration."""
        return {
            'price_features': {
                'sma_periods': [5, 10, 20, 50],
                'ema_periods': [5, 10, 20, 50],
                'rsi_periods': [14, 21],
                'macd_params': [(12, 26, 9)],
                'bollinger_periods': [20],
                'bollinger_std': [2],
                'atr_periods': [14, 21]
            },
            'volume_features': {
                'volume_sma_periods': [5, 10, 20],
                'volume_ratio_periods': [5, 10, 20],
                'vwap_periods': [10, 20]
            },
            'volatility_features': {
                'volatility_periods': [10, 20, 30]
            },
            'momentum_features': {
                'roc_periods': [5, 10, 20],
                'momentum_periods': [5, 10, 20]
            },
            'time_features': {
                'hour_of_day': True,
                'day_of_week': True,
                'is_weekend': True
            },
            'feature_selection': {
                'n_features': 30,
                'correlation_threshold': 0.95
            }
        }
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive feature set."""
        
        logger.info("Starting simplified feature engineering")
        
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
            features_df[f'sma_{period}'] = data['close'].rolling(period).mean()
            features_df[f'sma_ratio_{period}'] = data['close'] / features_df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in config['ema_periods']:
            features_df[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            features_df[f'ema_ratio_{period}'] = data['close'] / features_df[f'ema_{period}']
        
        # RSI
        for period in config['rsi_periods']:
            features_df[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
        
        # MACD
        for fast, slow, signal in config['macd_params']:
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_hist = macd - macd_signal
            
            features_df[f'macd_{fast}_{slow}_{signal}'] = macd
            features_df[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            features_df[f'macd_hist_{fast}_{slow}_{signal}'] = macd_hist
        
        # Bollinger Bands
        for period in config['bollinger_periods']:
            for std in config['bollinger_std']:
                sma = data['close'].rolling(period).mean()
                std_dev = data['close'].rolling(period).std()
                
                bb_upper = sma + (std_dev * std)
                bb_lower = sma - (std_dev * std)
                
                features_df[f'bb_upper_{period}_{std}'] = bb_upper
                features_df[f'bb_middle_{period}_{std}'] = sma
                features_df[f'bb_lower_{period}_{std}'] = bb_lower
                features_df[f'bb_width_{period}_{std}'] = (bb_upper - bb_lower) / sma
                features_df[f'bb_position_{period}_{std}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        for period in config['atr_periods']:
            features_df[f'atr_{period}'] = self._calculate_atr(data, period)
            features_df[f'atr_ratio_{period}'] = features_df[f'atr_{period}'] / data['close']
        
        return features_df
    
    def _engineer_volume_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer volume-based features."""
        
        config = self.config['volume_features']
        
        # Volume SMAs
        for period in config['volume_sma_periods']:
            features_df[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
            features_df[f'volume_ratio_{period}'] = data['volume'] / features_df[f'volume_sma_{period}']
        
        # VWAP
        for period in config['vwap_periods']:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()
            features_df[f'vwap_{period}'] = vwap
            features_df[f'vwap_ratio_{period}'] = data['close'] / vwap
        
        # Volume Price Trend
        features_df['vpt'] = (data['close'].pct_change() * data['volume']).cumsum()
        features_df['vpt_sma'] = features_df['vpt'].rolling(10).mean()
        features_df['vpt_ratio'] = features_df['vpt'] / features_df['vpt_sma']
        
        # On-Balance Volume
        features_df['obv'] = (data['volume'] * np.sign(data['close'].diff())).cumsum()
        features_df['obv_sma'] = features_df['obv'].rolling(10).mean()
        features_df['obv_ratio'] = features_df['obv'] / features_df['obv_sma']
        
        return features_df
    
    def _engineer_volatility_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer volatility-based features."""
        
        config = self.config['volatility_features']
        
        # Simple volatility
        for period in config['volatility_periods']:
            returns = data['close'].pct_change()
            features_df[f'volatility_{period}'] = returns.rolling(period).std()
            features_df[f'volatility_ratio_{period}'] = features_df[f'volatility_{period}'] / features_df[f'volatility_{period}'].rolling(50).mean()
        
        # Parkinson volatility
        features_df['parkinson_vol'] = np.sqrt(0.25 * np.log(data['high'] / data['low']) ** 2)
        
        # Garman-Klass volatility
        features_df['gk_vol'] = np.sqrt(0.5 * np.log(data['high'] / data['low']) ** 2 - 
                                      (2 * np.log(2) - 1) * np.log(data['close'] / data['open']) ** 2)
        
        return features_df
    
    def _engineer_momentum_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer momentum-based features."""
        
        config = self.config['momentum_features']
        
        # Rate of Change
        for period in config['roc_periods']:
            features_df[f'roc_{period}'] = data['close'].pct_change(period)
        
        # Momentum
        for period in config['momentum_periods']:
            features_df[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
        
        # Stochastic Oscillator
        features_df['stoch_k'] = self._calculate_stochastic_k(data, 14)
        features_df['stoch_d'] = features_df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        features_df['williams_r'] = self._calculate_williams_r(data, 14)
        
        # Commodity Channel Index
        features_df['cci'] = self._calculate_cci(data, 20)
        
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
        
        # Weekend
        if config['is_weekend']:
            features_df['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        return features_df
    
    def _engineer_advanced_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features."""
        
        # Price patterns
        features_df['hammer'] = self._detect_hammer(data)
        features_df['doji'] = self._detect_doji(data)
        features_df['engulfing'] = self._detect_engulfing(data)
        
        # Market microstructure
        features_df['spread'] = data['high'] - data['low']
        features_df['spread_ratio'] = features_df['spread'] / data['close']
        features_df['body_size'] = abs(data['close'] - data['open'])
        features_df['body_ratio'] = features_df['body_size'] / features_df['spread']
        
        # Trend strength
        features_df['trend_strength'] = self._calculate_trend_strength(data)
        
        # Support/Resistance levels
        features_df['support_level'] = data['low'].rolling(20).min()
        features_df['resistance_level'] = data['high'].rolling(20).max()
        features_df['support_distance'] = (data['close'] - features_df['support_level']) / data['close']
        features_df['resistance_distance'] = (features_df['resistance_level'] - data['close']) / data['close']
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = data['low'].rolling(period).min()
        highest_high = data['high'].rolling(period).max()
        return 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = data['high'].rolling(period).max()
        lowest_low = data['low'].rolling(period).min()
        return -100 * (highest_high - data['close']) / (highest_high - lowest_low)
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)
    
    def _detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect hammer candlestick pattern."""
        body_size = abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        return ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
    
    def _detect_doji(self, data: pd.DataFrame) -> pd.Series:
        """Detect doji candlestick pattern."""
        body_size = abs(data['close'] - data['open'])
        total_range = data['high'] - data['low']
        
        return (body_size < 0.1 * total_range).astype(int)
    
    def _detect_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect engulfing candlestick pattern."""
        prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
        curr_body = abs(data['close'] - data['open'])
        
        bullish_engulfing = ((data['close'] > data['open']) & 
                           (data['close'].shift(1) < data['open'].shift(1)) & 
                           (curr_body > prev_body))
        
        bearish_engulfing = ((data['close'] < data['open']) & 
                           (data['close'].shift(1) > data['open'].shift(1)) & 
                           (curr_body > prev_body))
        
        return (bullish_engulfing | bearish_engulfing).astype(int)
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength."""
        sma_short = data['close'].rolling(10).mean()
        sma_long = data['close'].rolling(30).mean()
        
        trend_direction = np.where(sma_short > sma_long, 1, -1)
        trend_magnitude = abs(sma_short - sma_long) / sma_long
        
        return trend_direction * trend_magnitude
    
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
            feature_variance = features_df.var()
            top_features = feature_variance.nlargest(n_features).index
            features_df = features_df[top_features]
        
        return features_df
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Get feature engineering summary."""
        
        return {
            'total_features': len(features_df.columns),
            'feature_categories': {
                'price': len([col for col in features_df.columns if any(x in col for x in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr'])]),
                'volume': len([col for col in features_df.columns if 'volume' in col or 'vwap' in col or 'obv' in col]),
                'volatility': len([col for col in features_df.columns if 'vol' in col or 'volatility' in col]),
                'momentum': len([col for col in features_df.columns if any(x in col for x in ['roc', 'momentum', 'stoch', 'williams', 'cci'])]),
                'time': len([col for col in features_df.columns if any(x in col for x in ['hour', 'day', 'weekend'])]),
                'advanced': len([col for col in features_df.columns if any(x in col for x in ['hammer', 'doji', 'engulfing', 'trend', 'support', 'resistance'])])
            },
            'data_shape': features_df.shape,
            'missing_values': features_df.isnull().sum().sum(),
            'constant_features': len(features_df.columns[features_df.nunique() <= 1])
        }

def main():
    """Demonstrate simplified feature engineering."""
    
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
    feature_engineer = SimplifiedFeatureEngineer()
    
    print("🧪 Testing Simplified Feature Engineering")
    print("=" * 50)
    
    # Engineer features
    features_df = feature_engineer.engineer_features(sample_data)
    
    # Get summary
    summary = feature_engineer.get_feature_summary(features_df)
    
    print(f"Generated {summary['total_features']} features")
    print(f"Feature shape: {summary['data_shape']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Constant features: {summary['constant_features']}")
    
    # Show feature categories
    print(f"\nFeature Categories:")
    for category, count in summary['feature_categories'].items():
        print(f"  {category}: {count} features")
    
    # Show sample features
    print(f"\nSample Features:")
    for i, feature in enumerate(features_df.columns[:10]):
        print(f"  {i+1}. {feature}")
    if len(features_df.columns) > 10:
        print(f"  ... and {len(features_df.columns) - 10} more")
    
    print("\n✅ Simplified feature engineering completed")
    
    return 0

if __name__ == "__main__":
    exit(main())
