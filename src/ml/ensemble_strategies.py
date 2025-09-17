"""
Ensemble Strategies - Blending Multiple Models with Meta-Allocator
Implements ensemble of momentum, mean-reversion, and ML classification models
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ML_CLASSIFICATION = "ml_classification"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    FUNDING_ARBITRAGE = "funding_arbitrage"

class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class ModelPrediction:
    strategy_type: StrategyType
    signal: SignalStrength
    confidence: float
    features_used: List[str]
    prediction_time: str
    model_version: str

@dataclass
class EnsemblePrediction:
    final_signal: SignalStrength
    confidence: float
    individual_predictions: List[ModelPrediction]
    meta_weights: Dict[str, float]
    prediction_time: str
    consensus_score: float

@dataclass
class ModelPerformance:
    strategy_type: StrategyType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    last_updated: str

class MetaAllocator:
    """
    Meta-allocator that dynamically weights ensemble models based on performance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_weights: Dict[StrategyType, float] = {}
        self.performance_history: Dict[StrategyType, List[ModelPerformance]] = {}
        self.adaptive_weights: Dict[StrategyType, float] = {}
        
        # Initialize equal weights
        for strategy in StrategyType:
            self.model_weights[strategy] = 1.0 / len(StrategyType)
            self.performance_history[strategy] = []
            self.adaptive_weights[strategy] = 1.0 / len(StrategyType)
    
    def update_weights(self, performance_metrics: Dict[StrategyType, ModelPerformance]):
        """Update model weights based on recent performance"""
        self.logger.info("🔄 Updating ensemble weights based on performance")
        
        # Calculate performance scores
        performance_scores = {}
        for strategy, perf in performance_metrics.items():
            # Composite score: 40% Sharpe, 30% Win Rate, 20% F1, 10% Max DD (inverted)
            score = (
                0.4 * perf.sharpe_ratio +
                0.3 * perf.win_rate +
                0.2 * perf.f1_score +
                0.1 * (1 - perf.max_drawdown)  # Invert drawdown (lower is better)
            )
            performance_scores[strategy] = max(0.01, score)  # Minimum weight
        
        # Normalize weights
        total_score = sum(performance_scores.values())
        for strategy in StrategyType:
            self.adaptive_weights[strategy] = performance_scores[strategy] / total_score
        
        # Smooth weight changes (exponential moving average)
        alpha = 0.3  # Smoothing factor
        for strategy in StrategyType:
            self.model_weights[strategy] = (
                alpha * self.adaptive_weights[strategy] +
                (1 - alpha) * self.model_weights[strategy]
            )
        
        self.logger.info(f"📊 Updated weights: {self.model_weights}")
    
    def get_weights(self) -> Dict[StrategyType, float]:
        """Get current model weights"""
        return self.model_weights.copy()

class MomentumStrategy:
    """
    Momentum-based trading strategy
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lookback_periods = [5, 10, 20, 50]
        self.momentum_threshold = 0.02  # 2% momentum threshold
    
    def predict(self, data: pd.DataFrame) -> ModelPrediction:
        """Generate momentum prediction"""
        try:
            if len(data) < max(self.lookback_periods):
                return ModelPrediction(
                    strategy_type=StrategyType.MOMENTUM,
                    signal=SignalStrength.HOLD,
                    confidence=0.0,
                    features_used=[],
                    prediction_time=datetime.now().isoformat(),
                    model_version="1.0"
                )
            
            # Calculate momentum indicators
            momentum_scores = []
            features_used = []
            
            for period in self.lookback_periods:
                if len(data) >= period:
                    # Price momentum
                    price_momentum = (data['close'].iloc[-1] - data['close'].iloc[-period]) / data['close'].iloc[-period]
                    momentum_scores.append(price_momentum)
                    features_used.append(f"price_momentum_{period}")
                    
                    # Volume momentum
                    if 'volume' in data.columns:
                        vol_momentum = (data['volume'].iloc[-1] - data['volume'].iloc[-period]) / data['volume'].iloc[-period]
                        momentum_scores.append(vol_momentum * 0.5)  # Weight volume less
                        features_used.append(f"volume_momentum_{period}")
            
            # Calculate overall momentum score
            avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.0
            
            # Determine signal
            if avg_momentum > self.momentum_threshold:
                signal = SignalStrength.STRONG_BUY if avg_momentum > self.momentum_threshold * 2 else SignalStrength.BUY
                confidence = min(0.95, abs(avg_momentum) / self.momentum_threshold)
            elif avg_momentum < -self.momentum_threshold:
                signal = SignalStrength.STRONG_SELL if avg_momentum < -self.momentum_threshold * 2 else SignalStrength.SELL
                confidence = min(0.95, abs(avg_momentum) / self.momentum_threshold)
            else:
                signal = SignalStrength.HOLD
                confidence = 0.1
            
            return ModelPrediction(
                strategy_type=StrategyType.MOMENTUM,
                signal=signal,
                confidence=confidence,
                features_used=features_used,
                prediction_time=datetime.now().isoformat(),
                model_version="1.0"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Momentum prediction error: {e}")
            return ModelPrediction(
                strategy_type=StrategyType.MOMENTUM,
                signal=SignalStrength.HOLD,
                confidence=0.0,
                features_used=[],
                prediction_time=datetime.now().isoformat(),
                model_version="1.0"
            )

class MeanReversionStrategy:
    """
    Mean reversion trading strategy
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lookback_periods = [20, 50, 100]
        self.reversion_threshold = 2.0  # 2 standard deviations
    
    def predict(self, data: pd.DataFrame) -> ModelPrediction:
        """Generate mean reversion prediction"""
        try:
            if len(data) < max(self.lookback_periods):
                return ModelPrediction(
                    strategy_type=StrategyType.MEAN_REVERSION,
                    signal=SignalStrength.HOLD,
                    confidence=0.0,
                    features_used=[],
                    prediction_time=datetime.now().isoformat(),
                    model_version="1.0"
                )
            
            # Calculate mean reversion indicators
            reversion_scores = []
            features_used = []
            
            for period in self.lookback_periods:
                if len(data) >= period:
                    # Z-score calculation
                    prices = data['close'].iloc[-period:]
                    mean_price = prices.mean()
                    std_price = prices.std()
                    
                    if std_price > 0:
                        z_score = (data['close'].iloc[-1] - mean_price) / std_price
                        reversion_scores.append(-z_score)  # Negative for mean reversion
                        features_used.append(f"z_score_{period}")
            
            # Calculate overall reversion score
            avg_reversion = np.mean(reversion_scores) if reversion_scores else 0.0
            
            # Determine signal
            if avg_reversion > self.reversion_threshold:
                signal = SignalStrength.STRONG_BUY
                confidence = min(0.95, abs(avg_reversion) / self.reversion_threshold)
            elif avg_reversion < -self.reversion_threshold:
                signal = SignalStrength.STRONG_SELL
                confidence = min(0.95, abs(avg_reversion) / self.reversion_threshold)
            elif abs(avg_reversion) > 1.0:
                signal = SignalStrength.BUY if avg_reversion > 0 else SignalStrength.SELL
                confidence = min(0.8, abs(avg_reversion) / self.reversion_threshold)
            else:
                signal = SignalStrength.HOLD
                confidence = 0.1
            
            return ModelPrediction(
                strategy_type=StrategyType.MEAN_REVERSION,
                signal=signal,
                confidence=confidence,
                features_used=features_used,
                prediction_time=datetime.now().isoformat(),
                model_version="1.0"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Mean reversion prediction error: {e}")
            return ModelPrediction(
                strategy_type=StrategyType.MEAN_REVERSION,
                signal=SignalStrength.HOLD,
                confidence=0.0,
                features_used=[],
                prediction_time=datetime.now().isoformat(),
                model_version="1.0"
            )

class MLClassificationStrategy:
    """
    Machine learning classification strategy
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42)
        }
        self.feature_columns = []
        self.is_trained = False
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            features = pd.DataFrame()
            
            # Price features
            features['price_change_1'] = data['close'].pct_change(1)
            features['price_change_5'] = data['close'].pct_change(5)
            features['price_change_10'] = data['close'].pct_change(10)
            
            # Moving averages
            features['sma_5'] = data['close'].rolling(5).mean()
            features['sma_20'] = data['close'].rolling(20).mean()
            features['sma_50'] = data['close'].rolling(50).mean()
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(data['close'])
            features['bollinger_upper'] = data['close'].rolling(20).mean() + 2 * data['close'].rolling(20).std()
            features['bollinger_lower'] = data['close'].rolling(20).mean() - 2 * data['close'].rolling(20).std()
            features['bollinger_position'] = (data['close'] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'])
            
            # Volume features
            if 'volume' in data.columns:
                features['volume_change'] = data['volume'].pct_change()
                features['volume_sma'] = data['volume'].rolling(20).mean()
                features['volume_ratio'] = data['volume'] / features['volume_sma']
            
            # Volatility
            features['volatility'] = data['close'].rolling(20).std()
            features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Feature preparation error: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def train(self, data: pd.DataFrame, target_column: str = 'future_return'):
        """Train ML models"""
        try:
            self.logger.info("🤖 Training ML classification models")
            
            # Prepare features
            features = self.prepare_features(data)
            
            if len(features) < 100:
                self.logger.warning("⚠️ Insufficient data for training")
                return
            
            # Create target variable (future return)
            if target_column not in data.columns:
                # Create synthetic target based on future price movement
                future_returns = data['close'].shift(-5) / data['close'] - 1
                data[target_column] = future_returns
            
            # Align features with targets
            aligned_data = data.loc[features.index].copy()
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 50:
                self.logger.warning("⚠️ Insufficient aligned data for training")
                return
            
            # Prepare training data
            X = features.loc[aligned_data.index]
            y = (aligned_data[target_column] > 0.01).astype(int)  # Binary classification: >1% return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.feature_columns = X.columns.tolist()
            
            # Train models
            model_predictions = {}
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    self.logger.info(f"📊 {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
                    model_predictions[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                    
                except Exception as e:
                    self.logger.error(f"❌ Error training {name}: {e}")
            
            self.is_trained = True
            self.logger.info("✅ ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
    
    def predict(self, data: pd.DataFrame) -> ModelPrediction:
        """Generate ML prediction"""
        try:
            if not self.is_trained:
                return ModelPrediction(
                    strategy_type=StrategyType.ML_CLASSIFICATION,
                    signal=SignalStrength.HOLD,
                    confidence=0.0,
                    features_used=[],
                    prediction_time=datetime.now().isoformat(),
                    model_version="1.0"
                )
            
            # Prepare features
            features = self.prepare_features(data)
            
            if len(features) == 0 or len(features) < len(self.feature_columns):
                return ModelPrediction(
                    strategy_type=StrategyType.ML_CLASSIFICATION,
                    signal=SignalStrength.HOLD,
                    confidence=0.0,
                    features_used=[],
                    prediction_time=datetime.now().isoformat(),
                    model_version="1.0"
                )
            
            # Get latest features
            latest_features = features.iloc[-1:][self.feature_columns]
            
            # Get predictions from all models
            predictions = []
            confidences = []
            
            for name, model in self.models.items():
                try:
                    pred_proba = model.predict_proba(latest_features)[0]
                    pred_class = model.predict(latest_features)[0]
                    
                    predictions.append(pred_class)
                    confidences.append(max(pred_proba))
                    
                except Exception as e:
                    self.logger.error(f"❌ Prediction error for {name}: {e}")
            
            if not predictions:
                return ModelPrediction(
                    strategy_type=StrategyType.ML_CLASSIFICATION,
                    signal=SignalStrength.HOLD,
                    confidence=0.0,
                    features_used=[],
                    prediction_time=datetime.now().isoformat(),
                    model_version="1.0"
                )
            
            # Ensemble prediction
            avg_prediction = np.mean(predictions)
            avg_confidence = np.mean(confidences)
            
            # Convert to signal
            if avg_prediction > 0.7:
                signal = SignalStrength.STRONG_BUY
            elif avg_prediction > 0.5:
                signal = SignalStrength.BUY
            elif avg_prediction < 0.3:
                signal = SignalStrength.STRONG_SELL
            elif avg_prediction < 0.5:
                signal = SignalStrength.SELL
            else:
                signal = SignalStrength.HOLD
            
            return ModelPrediction(
                strategy_type=StrategyType.ML_CLASSIFICATION,
                signal=signal,
                confidence=avg_confidence,
                features_used=self.feature_columns,
                prediction_time=datetime.now().isoformat(),
                model_version="1.0"
            )
            
        except Exception as e:
            self.logger.error(f"❌ ML prediction error: {e}")
            return ModelPrediction(
                strategy_type=StrategyType.ML_CLASSIFICATION,
                signal=SignalStrength.HOLD,
                confidence=0.0,
                features_used=[],
                prediction_time=datetime.now().isoformat(),
                model_version="1.0"
            )

class EnsembleStrategyManager:
    """
    Manages ensemble of multiple trading strategies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.meta_allocator = MetaAllocator()
        self.strategies = {
            StrategyType.MOMENTUM: MomentumStrategy(),
            StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
            StrategyType.ML_CLASSIFICATION: MLClassificationStrategy()
        }
        self.prediction_history: List[EnsemblePrediction] = []
        
        # Create reports directory
        self.reports_dir = Path("reports/ensemble_strategies")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_ensemble_prediction(self, data: pd.DataFrame) -> EnsemblePrediction:
        """Generate ensemble prediction from all strategies"""
        self.logger.info("🎯 Generating ensemble prediction")
        
        # Get individual predictions
        individual_predictions = []
        for strategy_type, strategy in self.strategies.items():
            try:
                prediction = strategy.predict(data)
                individual_predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"❌ Error in {strategy_type.value}: {e}")
        
        # Get current weights
        weights = self.meta_allocator.get_weights()
        
        # Calculate weighted consensus
        signal_scores = {
            SignalStrength.STRONG_BUY: 2.0,
            SignalStrength.BUY: 1.0,
            SignalStrength.HOLD: 0.0,
            SignalStrength.SELL: -1.0,
            SignalStrength.STRONG_SELL: -2.0
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for prediction in individual_predictions:
            weight = weights.get(prediction.strategy_type, 0.0)
            score = signal_scores[prediction.signal]
            confidence = prediction.confidence
            
            weighted_score += score * weight * confidence
            weighted_confidence += confidence * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weighted_score /= total_weight
            weighted_confidence /= total_weight
        
        # Determine final signal
        if weighted_score > 1.5:
            final_signal = SignalStrength.STRONG_BUY
        elif weighted_score > 0.5:
            final_signal = SignalStrength.BUY
        elif weighted_score < -1.5:
            final_signal = SignalStrength.STRONG_SELL
        elif weighted_score < -0.5:
            final_signal = SignalStrength.SELL
        else:
            final_signal = SignalStrength.HOLD
        
        # Calculate consensus score
        consensus_score = len([p for p in individual_predictions if p.signal == final_signal]) / len(individual_predictions)
        
        # Create ensemble prediction
        ensemble_prediction = EnsemblePrediction(
            final_signal=final_signal,
            confidence=weighted_confidence,
            individual_predictions=individual_predictions,
            meta_weights=weights,
            prediction_time=datetime.now().isoformat(),
            consensus_score=consensus_score
        )
        
        # Store prediction
        self.prediction_history.append(ensemble_prediction)
        
        # Keep only recent predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        self.logger.info(f"🎯 Ensemble prediction: {final_signal.value} (confidence: {weighted_confidence:.3f})")
        return ensemble_prediction
    
    def train_ml_models(self, data: pd.DataFrame):
        """Train ML models in the ensemble"""
        self.logger.info("🤖 Training ensemble ML models")
        
        # Train ML classification strategy
        ml_strategy = self.strategies[StrategyType.ML_CLASSIFICATION]
        ml_strategy.train(data)
    
    def update_performance(self, performance_metrics: Dict[StrategyType, ModelPerformance]):
        """Update strategy performance and rebalance weights"""
        self.logger.info("📊 Updating ensemble performance metrics")
        
        # Update meta-allocator weights
        self.meta_allocator.update_weights(performance_metrics)
        
        # Save performance metrics
        self._save_performance_metrics(performance_metrics)
    
    def _save_performance_metrics(self, performance_metrics: Dict[StrategyType, ModelPerformance]):
        """Save performance metrics to file"""
        try:
            metrics_data = {}
            for strategy_type, perf in performance_metrics.items():
                metrics_data[strategy_type.value] = asdict(perf)
            
            metrics_file = self.reports_dir / "performance_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"💾 Performance metrics saved: {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save performance metrics: {e}")
    
    def get_ensemble_summary(self) -> Dict:
        """Get ensemble strategy summary"""
        if not self.prediction_history:
            return {"message": "No predictions generated yet"}
        
        recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
        
        signal_counts = {}
        for signal in SignalStrength:
            signal_counts[signal.value] = len([p for p in recent_predictions if p.final_signal == signal])
        
        avg_confidence = np.mean([p.confidence for p in recent_predictions])
        avg_consensus = np.mean([p.consensus_score for p in recent_predictions])
        
        return {
            "total_predictions": len(self.prediction_history),
            "recent_predictions": len(recent_predictions),
            "signal_distribution": signal_counts,
            "average_confidence": avg_confidence,
            "average_consensus": avg_consensus,
            "current_weights": {k.value: v for k, v in self.meta_allocator.get_weights().items()}
        }

# Demo function
async def demo_ensemble_strategies():
    """Demo the ensemble strategies system"""
    print("🎯 Ensemble Strategies Demo")
    print("=" * 50)
    
    # Create ensemble manager
    ensemble_manager = EnsembleStrategyManager()
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
    n = len(dates)
    
    # Create realistic price data with trend and volatility
    trend = np.linspace(0.5, 0.6, n)
    noise = np.random.normal(0, 0.01, n)
    prices = trend + noise
    volumes = np.random.lognormal(10, 0.5, n)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        'close': prices,
        'volume': volumes
    })
    
    print(f"📊 Generated {len(data)} data points")
    
    # Train ML models
    print("🤖 Training ML models...")
    ensemble_manager.train_ml_models(data)
    
    # Generate ensemble predictions
    print("🎯 Generating ensemble predictions...")
    predictions = []
    
    for i in range(100, len(data), 50):  # Predict every 50 hours
        subset_data = data.iloc[:i]
        prediction = await ensemble_manager.generate_ensemble_prediction(subset_data)
        predictions.append(prediction)
    
    print(f"✅ Generated {len(predictions)} ensemble predictions")
    
    # Show prediction summary
    summary = ensemble_manager.get_ensemble_summary()
    print(f"\n📈 Ensemble Summary:")
    print(f"Total Predictions: {summary['total_predictions']}")
    print(f"Average Confidence: {summary['average_confidence']:.3f}")
    print(f"Average Consensus: {summary['average_consensus']:.3f}")
    
    print(f"\n🎯 Signal Distribution:")
    for signal, count in summary['signal_distribution'].items():
        print(f"  {signal}: {count}")
    
    print(f"\n⚖️ Current Weights:")
    for strategy, weight in summary['current_weights'].items():
        print(f"  {strategy}: {weight:.3f}")
    
    # Show recent predictions
    print(f"\n🔮 Recent Predictions:")
    for i, pred in enumerate(predictions[-5:]):
        print(f"  {i+1}. {pred.final_signal.value} (confidence: {pred.confidence:.3f}, consensus: {pred.consensus_score:.3f})")
    
    print("\n✅ Ensemble Strategies Demo Complete")

if __name__ == "__main__":
    asyncio.run(demo_ensemble_strategies())
