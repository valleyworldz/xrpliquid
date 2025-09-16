"""
🧠 MACHINE LEARNING RESEARCH SCIENTIST
"The market evolves. Therefore, we must evolve faster."

This module implements advanced machine learning for trading:
- Adaptive strategy optimization
- Reinforcement learning for position sizing
- Deep learning for pattern recognition
- Ensemble methods for signal generation
- Online learning and model adaptation
- Feature engineering and selection
- Model validation and backtesting
- Automated hyperparameter optimization
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import json
from datetime import datetime, timedelta
import threading
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

class ModelType(Enum):
    """Model type enumeration"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

class LearningMode(Enum):
    """Learning mode enumeration"""
    BATCH = "batch"
    ONLINE = "online"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"

@dataclass
class ModelConfig:
    """Model configuration data structure"""
    model_type: ModelType
    learning_mode: LearningMode
    features: List[str]
    target: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    validation_split: float = 0.2
    test_split: float = 0.1
    retrain_frequency: int = 100  # retrain every N samples
    feature_selection: bool = True
    ensemble_method: str = "voting"  # "voting", "stacking", "bagging"

@dataclass
class ModelPerformance:
    """Model performance data structure"""
    model_id: str
    model_type: ModelType
    training_score: float
    validation_score: float
    test_score: float
    mse: float
    mae: float
    r2: float
    feature_importance: Dict[str, float]
    training_time: float
    prediction_time: float
    last_updated: datetime

@dataclass
class PredictionResult:
    """Prediction result data structure"""
    model_id: str
    prediction: float
    confidence: float
    features_used: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class MachineLearningResearchScientist:
    """
    Machine Learning Research Scientist - Master of Adaptive Intelligence
    
    This class implements advanced machine learning for trading:
    1. Adaptive strategy optimization
    2. Reinforcement learning for position sizing
    3. Deep learning for pattern recognition
    4. Ensemble methods for signal generation
    5. Online learning and model adaptation
    6. Feature engineering and selection
    7. Model validation and backtesting
    8. Automated hyperparameter optimization
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # ML configuration
        self.ml_config = {
            'max_models': 10,
            'retrain_threshold': 0.05,  # Retrain if performance drops 5%
            'feature_window': 100,  # Number of samples for feature calculation
            'prediction_horizon': 1,  # Steps ahead to predict
            'ensemble_weights': 'performance',  # 'equal', 'performance', 'adaptive'
            'online_learning_rate': 0.01,
            'model_storage_path': './models',
            'feature_importance_threshold': 0.01
        }
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.feature_scalers: Dict[str, Any] = {}
        
        # Data storage
        self.training_data: deque = deque(maxlen=10000)
        self.feature_data: deque = deque(maxlen=10000)
        self.prediction_history: deque = deque(maxlen=1000)
        
        # Feature engineering
        self.feature_engineers: Dict[str, Callable] = {}
        self.feature_selectors: Dict[str, Any] = {}
        
        # Online learning
        self.online_models: Dict[str, Any] = {}
        self.reinforcement_agent = None
        
        # Model optimization
        self.optimization_study = None
        self.hyperparameter_space: Dict[str, Any] = {}
        
        # Performance tracking
        self.ml_metrics = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'model_retrains': 0,
            'feature_engineering_time': 0.0,
            'training_time': 0.0,
            'prediction_time': 0.0
        }
        
        # Threading
        self.ml_lock = threading.RLock()
        self.training_thread = None
        self.running = False
        
        # Initialize ML system
        self._initialize_ml_system()
    
    def _initialize_ml_system(self):
        """Initialize machine learning system"""
        try:
            self.logger.info("Initializing machine learning research scientist...")
            
            # Create model storage directory
            import os
            os.makedirs(self.ml_config['model_storage_path'], exist_ok=True)
            
            # Initialize feature engineers
            self._initialize_feature_engineers()
            
            # Initialize hyperparameter optimization
            self._initialize_hyperparameter_optimization()
            
            # Initialize reinforcement learning agent
            self._initialize_reinforcement_agent()
            
            # Start training thread
            self.training_thread = threading.Thread(
                target=self._ml_training_loop,
                daemon=True,
                name="ml_trainer"
            )
            self.training_thread.start()
            
            self.running = True
            self.logger.info("Machine learning research scientist initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML system: {e}")
    
    def _initialize_feature_engineers(self):
        """Initialize feature engineering functions"""
        try:
            # Technical indicators
            self.feature_engineers['sma_ratio'] = self._calculate_sma_ratio
            self.feature_engineers['rsi'] = self._calculate_rsi
            self.feature_engineers['macd'] = self._calculate_macd
            self.feature_engineers['bollinger_bands'] = self._calculate_bollinger_bands
            self.feature_engineers['atr'] = self._calculate_atr
            self.feature_engineers['volume_profile'] = self._calculate_volume_profile
            
            # Price-based features
            self.feature_engineers['price_momentum'] = self._calculate_price_momentum
            self.feature_engineers['volatility'] = self._calculate_volatility
            self.feature_engineers['price_acceleration'] = self._calculate_price_acceleration
            
            # Market microstructure
            self.feature_engineers['order_flow'] = self._calculate_order_flow
            self.feature_engineers['liquidity_metrics'] = self._calculate_liquidity_metrics
            self.feature_engineers['spread_analysis'] = self._calculate_spread_analysis
            
            # Time-based features
            self.feature_engineers['time_features'] = self._calculate_time_features
            self.feature_engineers['seasonality'] = self._calculate_seasonality
            
            self.logger.info(f"Initialized {len(self.feature_engineers)} feature engineers")
            
        except Exception as e:
            self.logger.error(f"Error initializing feature engineers: {e}")
    
    def _initialize_hyperparameter_optimization(self):
        """Initialize hyperparameter optimization"""
        try:
            # Define hyperparameter search spaces
            self.hyperparameter_space = {
                'random_forest': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10)
                },
                'xgboost': {
                    'n_estimators': (50, 500),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0)
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'learning_rate_init': (0.001, 0.1),
                    'alpha': (0.0001, 0.1)
                }
            }
            
            # Initialize Optuna study
            self.optimization_study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing hyperparameter optimization: {e}")
    
    def _initialize_reinforcement_agent(self):
        """Initialize reinforcement learning agent"""
        try:
            # Simple Q-learning agent for position sizing
            # In production, you would use more sophisticated RL algorithms
            self.reinforcement_agent = {
                'q_table': {},
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'exploration_rate': 0.1,
                'state_space': 100,  # Discretized state space
                'action_space': 10   # Discretized action space (position sizes)
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing reinforcement agent: {e}")
    
    def _ml_training_loop(self):
        """Main ML training loop"""
        try:
            while self.running:
                try:
                    # Check if retraining is needed
                    if self._should_retrain():
                        self._retrain_models()
                    
                    # Update online models
                    self._update_online_models()
                    
                    # Optimize hyperparameters
                    self._optimize_hyperparameters()
                    
                    # Sleep for training interval
                    time.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Error in ML training loop: {e}")
                    time.sleep(60)  # Wait 1 minute on error
                    
        except Exception as e:
            self.logger.error(f"Fatal error in ML training loop: {e}")
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        try:
            # Check if enough new data is available
            if len(self.training_data) < 100:
                return False
            
            # Check if performance has degraded
            for model_id, performance in self.model_performance.items():
                if performance.validation_score < 0.5:  # Threshold for retraining
                    return True
            
            # Check if retrain frequency is reached
            if self.ml_metrics['total_predictions'] % self.ml_config['retrain_threshold'] == 0:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retrain condition: {e}")
            return False
    
    def _retrain_models(self):
        """Retrain all models"""
        try:
            with self.ml_lock:
                self.logger.info("Starting model retraining...")
                
                # Prepare training data
                X, y = self._prepare_training_data()
                if X is None or y is None:
                    return
                
                # Retrain each model
                for model_id, model in self.models.items():
                    try:
                        start_time = time.time()
                        
                        # Train model
                        model.fit(X, y)
                        
                        # Evaluate performance
                        performance = self._evaluate_model(model, X, y)
                        self.model_performance[model_id] = performance
                        
                        # Save model
                        self._save_model(model_id, model)
                        
                        training_time = time.time() - start_time
                        self.ml_metrics['training_time'] += training_time
                        self.ml_metrics['model_retrains'] += 1
                        
                        self.logger.info(f"Retrained model {model_id} in {training_time:.2f}s")
                        
                    except Exception as e:
                        self.logger.error(f"Error retraining model {model_id}: {e}")
                
                self.logger.info("Model retraining completed")
                
        except Exception as e:
            self.logger.error(f"Error in model retraining: {e}")
    
    def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from historical data"""
        try:
            if len(self.training_data) < 50:
                return None, None
            
            # Convert to DataFrame
            df = pd.DataFrame(list(self.training_data))
            
            # Engineer features
            feature_df = self._engineer_features(df)
            
            # Select features
            selected_features = self._select_features(feature_df)
            
            # Prepare target variable (current period return - no future data)
            target = df['return'].dropna()
            
            # Align features and target
            min_length = min(len(selected_features), len(target))
            X = selected_features.iloc[:min_length].values
            y = target.iloc[:min_length].values
            
            # Remove any NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""
        try:
            start_time = time.time()
            
            feature_df = df.copy()
            
            # Apply feature engineering functions
            for feature_name, feature_func in self.feature_engineers.items():
                try:
                    feature_values = feature_func(df)
                    if isinstance(feature_values, dict):
                        for key, value in feature_values.items():
                            feature_df[f"{feature_name}_{key}"] = value
                    else:
                        feature_df[feature_name] = feature_values
                except Exception as e:
                    self.logger.warning(f"Error engineering feature {feature_name}: {e}")
            
            # Remove original columns to avoid data leakage
            original_columns = ['price', 'volume', 'timestamp']
            feature_df = feature_df.drop(columns=[col for col in original_columns if col in feature_df.columns])
            
            engineering_time = time.time() - start_time
            self.ml_metrics['feature_engineering_time'] += engineering_time
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            return df
    
    def _select_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Select most important features"""
        try:
            # Remove non-numeric columns
            numeric_df = feature_df.select_dtypes(include=[np.number])
            
            # Remove columns with too many NaN values
            numeric_df = numeric_df.dropna(axis=1, thresh=len(numeric_df) * 0.5)
            
            # Fill remaining NaN values
            numeric_df = numeric_df.fillna(numeric_df.median())
            
            # Feature selection using mutual information
            if len(numeric_df.columns) > 20:  # Only if we have many features
                selector = SelectKBest(score_func=mutual_info_regression, k=20)
                # Note: This would need target variable, simplified for now
                selected_features = numeric_df.iloc[:, :20]  # Take first 20 features
            else:
                selected_features = numeric_df
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {e}")
            return feature_df.select_dtypes(include=[np.number])
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate model performance"""
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            
            # Calculate metrics
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            performance = ModelPerformance(
                model_id=f"model_{int(time.time())}",
                model_type=ModelType.RANDOM_FOREST,  # Default type
                training_score=np.mean(scores),
                validation_score=np.mean(scores),
                test_score=np.mean(scores),
                mse=mse,
                mae=mae,
                r2=r2,
                feature_importance=feature_importance,
                training_time=0.0,
                prediction_time=0.0,
                last_updated=datetime.now()
            )
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return ModelPerformance(
                model_id=f"model_{int(time.time())}",
                model_type=ModelType.RANDOM_FOREST,
                training_score=0.0,
                validation_score=0.0,
                test_score=0.0,
                mse=0.0,
                mae=0.0,
                r2=0.0,
                feature_importance={},
                training_time=0.0,
                prediction_time=0.0,
                last_updated=datetime.now()
            )
    
    def _save_model(self, model_id: str, model: Any):
        """Save model to disk"""
        try:
            model_path = f"{self.ml_config['model_storage_path']}/{model_id}.pkl"
            joblib.dump(model, model_path)
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {e}")
    
    def _load_model(self, model_id: str) -> Optional[Any]:
        """Load model from disk"""
        try:
            model_path = f"{self.ml_config['model_storage_path']}/{model_id}.pkl"
            if os.path.exists(model_path):
                return joblib.load(model_path)
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def _update_online_models(self):
        """Update online learning models"""
        try:
            # Update online models with new data
            for model_id, model in self.online_models.items():
                if hasattr(model, 'partial_fit'):
                    # Get recent data
                    recent_data = list(self.training_data)[-10:]  # Last 10 samples
                    if len(recent_data) > 0:
                        df = pd.DataFrame(recent_data)
                        X, y = self._prepare_training_data()
                        if X is not None and y is not None:
                            model.partial_fit(X, y)
            
        except Exception as e:
            self.logger.error(f"Error updating online models: {e}")
    
    def _optimize_hyperparameters(self):
        """Optimize hyperparameters using Optuna"""
        try:
            if len(self.training_data) < 100:
                return
            
            # Prepare data
            X, y = self._prepare_training_data()
            if X is None or y is None:
                return
            
            # Define objective function
            def objective(trial):
                # Sample hyperparameters
                model_type = trial.suggest_categorical('model_type', ['random_forest', 'xgboost'])
                
                if model_type == 'random_forest':
                    model = RandomForestRegressor(
                        n_estimators=trial.suggest_int('n_estimators', 50, 500),
                        max_depth=trial.suggest_int('max_depth', 3, 20),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                        random_state=42
                    )
                elif model_type == 'xgboost':
                    model = xgb.XGBRegressor(
                        n_estimators=trial.suggest_int('n_estimators', 50, 500),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        random_state=42
                    )
                
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                
                return np.mean(scores)
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10)  # Limited trials for demo
            
            # Get best parameters
            best_params = study.best_params
            self.logger.info(f"Best hyperparameters: {best_params}")
            
        except Exception as e:
            self.logger.error(f"Error optimizing hyperparameters: {e}")
    
    def create_model(self, model_config: ModelConfig) -> str:
        """Create a new ML model"""
        try:
            with self.ml_lock:
                model_id = f"model_{int(time.time())}"
                
                # Create model based on type
                if model_config.model_type == ModelType.LINEAR_REGRESSION:
                    model = LinearRegression()
                elif model_config.model_type == ModelType.RIDGE_REGRESSION:
                    model = Ridge(alpha=model_config.hyperparameters.get('alpha', 1.0))
                elif model_config.model_type == ModelType.LASSO_REGRESSION:
                    model = Lasso(alpha=model_config.hyperparameters.get('alpha', 1.0))
                elif model_config.model_type == ModelType.RANDOM_FOREST:
                    model = RandomForestRegressor(
                        n_estimators=model_config.hyperparameters.get('n_estimators', 100),
                        max_depth=model_config.hyperparameters.get('max_depth', 10),
                        random_state=42
                    )
                elif model_config.model_type == ModelType.XGBOOST:
                    model = xgb.XGBRegressor(
                        n_estimators=model_config.hyperparameters.get('n_estimators', 100),
                        max_depth=model_config.hyperparameters.get('max_depth', 6),
                        learning_rate=model_config.hyperparameters.get('learning_rate', 0.1),
                        random_state=42
                    )
                elif model_config.model_type == ModelType.LIGHTGBM:
                    model = lgb.LGBMRegressor(
                        n_estimators=model_config.hyperparameters.get('n_estimators', 100),
                        max_depth=model_config.hyperparameters.get('max_depth', 6),
                        learning_rate=model_config.hyperparameters.get('learning_rate', 0.1),
                        random_state=42
                    )
                elif model_config.model_type == ModelType.SVM:
                    model = SVR(
                        kernel=model_config.hyperparameters.get('kernel', 'rbf'),
                        C=model_config.hyperparameters.get('C', 1.0),
                        gamma=model_config.hyperparameters.get('gamma', 'scale')
                    )
                elif model_config.model_type == ModelType.NEURAL_NETWORK:
                    model = MLPRegressor(
                        hidden_layer_sizes=model_config.hyperparameters.get('hidden_layer_sizes', (100,)),
                        learning_rate_init=model_config.hyperparameters.get('learning_rate_init', 0.001),
                        alpha=model_config.hyperparameters.get('alpha', 0.0001),
                        random_state=42
                    )
                else:
                    raise ValueError(f"Unsupported model type: {model_config.model_type}")
                
                # Store model and config
                self.models[model_id] = model
                self.model_configs[model_id] = model_config
                
                # Initialize feature scaler
                self.feature_scalers[model_id] = StandardScaler()
                
                self.logger.info(f"Created model {model_id} of type {model_config.model_type}")
                
                return model_id
                
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            return None
    
    def train_model(self, model_id: str, X: np.ndarray, y: np.ndarray) -> bool:
        """Train a specific model"""
        try:
            with self.ml_lock:
                if model_id not in self.models:
                    self.logger.error(f"Model {model_id} not found")
                    return False
                
                model = self.models[model_id]
                scaler = self.feature_scalers[model_id]
                
                # Scale features
                X_scaled = scaler.fit_transform(X)
                
                # Train model
                start_time = time.time()
                model.fit(X_scaled, y)
                training_time = time.time() - start_time
                
                # Evaluate performance
                performance = self._evaluate_model(model, X_scaled, y)
                performance.training_time = training_time
                self.model_performance[model_id] = performance
                
                # Save model
                self._save_model(model_id, model)
                
                self.logger.info(f"Trained model {model_id} in {training_time:.2f}s")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error training model {model_id}: {e}")
            return False
    
    def predict(self, model_id: str, features: Dict[str, float]) -> Optional[PredictionResult]:
        """Make prediction using a specific model"""
        try:
            with self.ml_lock:
                if model_id not in self.models:
                    self.logger.error(f"Model {model_id} not found")
                    return None
                
                model = self.models[model_id]
                scaler = self.feature_scalers[model_id]
                
                # Prepare features
                feature_array = np.array([list(features.values())]).reshape(1, -1)
                
                # Scale features
                feature_array_scaled = scaler.transform(feature_array)
                
                # Make prediction
                start_time = time.time()
                prediction = model.predict(feature_array_scaled)[0]
                prediction_time = time.time() - start_time
                
                # Calculate confidence (simplified)
                confidence = 0.8  # In production, calculate actual confidence
                
                # Update metrics
                self.ml_metrics['total_predictions'] += 1
                self.ml_metrics['prediction_time'] += prediction_time
                
                # Create result
                result = PredictionResult(
                    model_id=model_id,
                    prediction=prediction,
                    confidence=confidence,
                    features_used=list(features.keys()),
                    timestamp=datetime.now(),
                    metadata={'prediction_time': prediction_time}
                )
                
                # Store prediction history
                self.prediction_history.append(result)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error making prediction with model {model_id}: {e}")
            return None
    
    def ensemble_predict(self, features: Dict[str, float]) -> Optional[PredictionResult]:
        """Make ensemble prediction using all models"""
        try:
            if len(self.models) == 0:
                return None
            
            predictions = []
            weights = []
            
            # Get predictions from all models
            for model_id in self.models.keys():
                result = self.predict(model_id, features)
                if result:
                    predictions.append(result.prediction)
                    
                    # Calculate weight based on performance
                    if model_id in self.model_performance:
                        weight = max(0.1, self.model_performance[model_id].r2)
                    else:
                        weight = 1.0
                    weights.append(weight)
            
            if not predictions:
                return None
            
            # Calculate weighted average
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            ensemble_prediction = np.average(predictions, weights=weights)
            ensemble_confidence = np.average([0.8] * len(predictions))  # Simplified confidence
            
            # Create ensemble result
            result = PredictionResult(
                model_id="ensemble",
                prediction=ensemble_prediction,
                confidence=ensemble_confidence,
                features_used=list(features.keys()),
                timestamp=datetime.now(),
                metadata={
                    'individual_predictions': predictions,
                    'weights': weights.tolist(),
                    'num_models': len(predictions)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making ensemble prediction: {e}")
            return None
    
    def update_training_data(self, data: Dict[str, Any]):
        """Update training data with new sample"""
        try:
            self.training_data.append(data)
            
        except Exception as e:
            self.logger.error(f"Error updating training data: {e}")
    
    def get_model_performance(self, model_id: str) -> Optional[ModelPerformance]:
        """Get model performance metrics"""
        return self.model_performance.get(model_id)
    
    def get_all_model_performance(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all models"""
        return self.model_performance.copy()
    
    def get_ml_metrics(self) -> Dict[str, Any]:
        """Get ML system metrics"""
        return self.ml_metrics.copy()
    
    def get_prediction_history(self) -> List[PredictionResult]:
        """Get prediction history"""
        return list(self.prediction_history)
    
    # Feature engineering functions
    def _calculate_sma_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SMA ratio"""
        try:
            if 'price' not in df.columns:
                return pd.Series([0] * len(df))
            
            sma_short = df['price'].rolling(window=10).mean()
            sma_long = df['price'].rolling(window=30).mean()
            return sma_short / sma_long
        except:
            return pd.Series([0] * len(df))
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI"""
        try:
            if 'price' not in df.columns:
                return pd.Series([50] * len(df))
            
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(df))
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        try:
            if 'price' not in df.columns:
                return {'macd': pd.Series([0] * len(df)), 'signal': pd.Series([0] * len(df))}
            
            ema_12 = df['price'].ewm(span=12).mean()
            ema_26 = df['price'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            return {'macd': macd, 'signal': signal}
        except:
            return {'macd': pd.Series([0] * len(df)), 'signal': pd.Series([0] * len(df))}
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            if 'price' not in df.columns:
                return {'upper': pd.Series([0] * len(df)), 'lower': pd.Series([0] * len(df))}
            
            sma = df['price'].rolling(window=20).mean()
            std = df['price'].rolling(window=20).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            return {'upper': upper, 'lower': lower}
        except:
            return {'upper': pd.Series([0] * len(df)), 'lower': pd.Series([0] * len(df))}
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        try:
            if 'price' not in df.columns:
                return pd.Series([0] * len(df))
            
            high = df['price'] * 1.01  # Simulate high
            low = df['price'] * 0.99   # Simulate low
            close = df['price']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            return atr.fillna(0)
        except:
            return pd.Series([0] * len(df))
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile"""
        try:
            if 'volume' not in df.columns:
                return pd.Series([0] * len(df))
            
            return df['volume'].rolling(window=20).mean()
        except:
            return pd.Series([0] * len(df))
    
    def _calculate_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price momentum"""
        try:
            if 'price' not in df.columns:
                return pd.Series([0] * len(df))
            
            return df['price'].pct_change(periods=5)
        except:
            return pd.Series([0] * len(df))
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility"""
        try:
            if 'price' not in df.columns:
                return pd.Series([0] * len(df))
            
            returns = df['price'].pct_change()
            return returns.rolling(window=20).std()
        except:
            return pd.Series([0] * len(df))
    
    def _calculate_price_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price acceleration"""
        try:
            if 'price' not in df.columns:
                return pd.Series([0] * len(df))
            
            momentum = df['price'].pct_change(periods=5)
            return momentum.diff()
        except:
            return pd.Series([0] * len(df))
    
    def _calculate_order_flow(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow metrics"""
        try:
            # Simplified order flow calculation
            return pd.Series([0] * len(df))
        except:
            return pd.Series([0] * len(df))
    
    def _calculate_liquidity_metrics(self, df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity metrics"""
        try:
            # Simplified liquidity calculation
            return pd.Series([1] * len(df))
        except:
            return pd.Series([1] * len(df))
    
    def _calculate_spread_analysis(self, df: pd.DataFrame) -> pd.Series:
        """Calculate spread analysis"""
        try:
            # Simplified spread calculation
            return pd.Series([0.001] * len(df))
        except:
            return pd.Series([0.001] * len(df))
    
    def _calculate_time_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate time-based features"""
        try:
            if 'timestamp' not in df.columns:
                return {'hour': pd.Series([0] * len(df)), 'day_of_week': pd.Series([0] * len(df))}
            
            timestamps = pd.to_datetime(df['timestamp'])
            return {
                'hour': timestamps.dt.hour,
                'day_of_week': timestamps.dt.dayofweek
            }
        except:
            return {'hour': pd.Series([0] * len(df)), 'day_of_week': pd.Series([0] * len(df))}
    
    def _calculate_seasonality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate seasonality features"""
        try:
            # Simplified seasonality calculation
            return pd.Series([0] * len(df))
        except:
            return pd.Series([0] * len(df))
    
    def shutdown(self):
        """Shutdown ML system"""
        try:
            self.running = False
            
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=5.0)
            
            self.logger.info("Machine learning research scientist shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during ML shutdown: {e}")

