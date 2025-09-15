import logging
import time
import numpy as np

class AdvancedPatternAnalyzer:
    """Advanced pattern recognition for XRP trading signals with GROK ML integration"""
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.pattern_history = []
        self.confidence_threshold = 0.95
        
        # GROK INTEGRATION: Machine Learning Components
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            self.ml_available = True
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_model_trained = False
            self.training_data = []
            self.training_labels = []
            self.logger.info("✅ GROK ML components initialized")
        except ImportError:
            self.ml_available = False
            self.logger.warning("⚠️ GROK ML components not available, using basic patterns")
        
        # GROK INTEGRATION: Enhanced Pattern Recognition
        self.pattern_weights = {
            'RSI_OVERSOLD': 0.8,
            'RSI_OVERBOUGHT': -0.8,
            'STRONG_UPTREND': 0.9,
            'STRONG_DOWNTREND': -0.9,
            'UPTREND': 0.6,
            'DOWNTREND': -0.6,
            'POSITIVE_MOMENTUM': 0.7,
            'NEGATIVE_MOMENTUM': -0.7,
            'CONSECUTIVE_HIGHER': 0.5,
            'CONSECUTIVE_LOWER': -0.5,
            'HIGH_VOLATILITY': 0.3,
            'VOLUME_CONFIRMED': 0.4,
            'WEAK_VOLUME': -0.2
        }
        
        # GROK INTEGRATION: Performance Tracking
        self.pattern_performance = {}
        self.adaptation_enabled = True
        self.last_adaptation = time.time()
        self.adaptation_interval = 3600  # 1 hour

    def analyze_xrp_patterns(self, price_history, volume_history=None):
        # PATCH ①: EMA50/200 + soft RSI (45/55) filter
        if len(price_history) < 200:
            rsi = self._calculate_rsi(price_history)
            return {"signal": "HOLD", "confidence": 0.0, "patterns": [], "ema50": None, "ema200": None, "rsi": rsi}
        patterns = []
        confidence = 0.0
        # Use numpy for EMA if available
        def ema_np(prices, period):
            prices = np.array(prices)
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            a = np.convolve(prices, weights, mode='full')[:len(prices)]
            a[:period] = a[period]
            return a[-1]
        try:
            ema50 = ema_np(price_history, 50)
            ema200 = ema_np(price_history, 200)
        except Exception:
            # fallback to simple EMA
            def ema(prices, period):
                k = 2 / (period + 1)
                ema_val = prices[0]
                for price in prices[1:]:
                    ema_val = price * k + ema_val * (1 - k)
                return ema_val
            ema50 = ema(price_history, 50)
            ema200 = ema(price_history, 200)
        rsi = self._calculate_rsi(price_history)
        # PATCH ⑧: Adjust RSI thresholds for shallow markets
        if ema50 > ema200 and rsi > 52:
            patterns.append({"type": "EMA_BULLISH_CROSS", "confidence": 1.0})
            patterns.append({"type": "RSI_BULLISH", "confidence": 1.0})
            confidence = 1.0
            signal = "BUY"
        elif ema50 < ema200 and rsi < 48:
            patterns.append({"type": "EMA_BEARISH_CROSS", "confidence": 1.0})
            patterns.append({"type": "RSI_BEARISH", "confidence": 1.0})
            confidence = 1.0
            signal = "SELL"
        else:
            return {"signal": "HOLD", "confidence": 0.0, "patterns": [], "ema50": ema50, "ema200": ema200, "rsi": rsi}
        return {
            "signal": signal,
            "confidence": confidence,
            "patterns": patterns,
            "rsi": rsi,
            "ema50": ema50,
            "ema200": ema200
        }
    
    # GROK INTEGRATION: Enhanced Pattern Analysis Methods
    def _apply_pattern_weights(self, patterns, base_confidence):
        """Apply GROK pattern weights for enhanced confidence calculation"""
        weighted_confidence = base_confidence
        
        for pattern in patterns:
            pattern_type = pattern.get('type', '')
            pattern_confidence = pattern.get('confidence', 0.0)
            weight = self.pattern_weights.get(pattern_type, 0.0)
            
            # Apply weighted contribution
            weighted_confidence += weight * pattern_confidence * 0.1
        
        return weighted_confidence
    
    def _get_ml_prediction(self, price_history, volume_history):
        """Get ML prediction for enhanced signal accuracy"""
        try:
            if len(price_history) < 20:
                return None
            
            # Prepare features
            features = self._extract_ml_features(price_history, volume_history)
            if not features:
                return None
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction
            prediction = self.ml_model.predict_proba(features_scaled)[0]
            
            # Convert to confidence score (-1 to 1)
            confidence = (prediction[1] - prediction[0]) * 2 - 1
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {e}")
            return None
    
    def _extract_ml_features(self, price_history, volume_history):
        """Extract features for ML model"""
        try:
            features = []
            
            # Price-based features
            if len(price_history) >= 20:
                # RSI
                rsi = self._calculate_rsi(price_history)
                features.append(rsi / 100.0)  # Normalize to 0-1
                
                # Momentum
                momentum = self._calculate_momentum(price_history)
                features.append(momentum)
                
                # Volatility
                volatility = self._calculate_volatility(price_history)
                features.append(volatility)
                
                # Moving averages
                sma_10 = sum(price_history[-10:]) / 10
                sma_20 = sum(price_history[-20:]) / 20
                current_price = price_history[-1]
                
                features.append((current_price - sma_10) / sma_10)
                features.append((current_price - sma_20) / sma_20)
                features.append((sma_10 - sma_20) / sma_20)
                
                # Price changes
                price_change_1h = (current_price - price_history[-60]) / price_history[-60] if len(price_history) >= 60 else 0
                price_change_24h = (current_price - price_history[-1440]) / price_history[-1440] if len(price_history) >= 1440 else 0
                
                features.extend([price_change_1h, price_change_24h])
            
            # Volume-based features
            if volume_history and len(volume_history) >= 20:
                current_volume = volume_history[-1]
                avg_volume = sum(volume_history[-20:]) / 20
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                features.append(volume_ratio)
            else:
                features.append(1.0)  # Default volume ratio
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def _calculate_dynamic_threshold(self):
        """Calculate dynamic confidence threshold based on market conditions"""
        base_threshold = self.confidence_threshold
        
        # Adjust based on pattern performance
        if self.pattern_performance:
            avg_performance = sum(self.pattern_performance.values()) / len(self.pattern_performance)
            performance_adjustment = (avg_performance - 0.5) * 0.2  # ±10% adjustment
            base_threshold += performance_adjustment
        
        # Ensure reasonable bounds
        return max(0.7, min(0.98, base_threshold))
    
    def _track_pattern_performance(self, patterns, confidence):
        """Track pattern performance for adaptation"""
        for pattern in patterns:
            pattern_type = pattern.get('type', '')
            if pattern_type not in self.pattern_performance:
                self.pattern_performance[pattern_type] = []
            
            # Store confidence for later evaluation
            self.pattern_performance[pattern_type].append(confidence)
            
            # Keep only recent data
            if len(self.pattern_performance[pattern_type]) > 100:
                self.pattern_performance[pattern_type] = self.pattern_performance[pattern_type][-100:]
    
    def train_ml_model(self, historical_data):
        """Train ML model with historical data"""
        if not self.ml_available:
            return False
        
        try:
            features_list = []
            labels_list = []
            
            # Prepare training data
            for i in range(20, len(historical_data) - 1):
                price_window = historical_data[i-20:i+1]
                volume_window = historical_data[i-20:i+1] if len(historical_data) > i else None
                
                features = self._extract_ml_features(price_window, volume_window)
                if features:
                    features_list.append(features)
                    
                    # Create label (1 for price increase, 0 for decrease)
                    future_price = historical_data[i+1]
                    current_price = historical_data[i]
                    label = 1 if future_price > current_price else 0
                    labels_list.append(label)
            
            if len(features_list) < 50:
                self.logger.warning("Insufficient data for ML training")
                return False
            
            # Train model
            X = np.array(features_list)
            y = np.array(labels_list)
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.ml_model.fit(X_scaled, y)
            
            self.is_model_trained = True
            self.logger.info(f"✅ ML model trained with {len(features_list)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"ML training failed: {e}")
            return False
    
    def adapt_pattern_weights(self):
        """Adapt pattern weights based on performance"""
        if not self.adaptation_enabled or time.time() - self.last_adaptation < self.adaptation_interval:
            return
        
        try:
            for pattern_type, performances in self.pattern_performance.items():
                if len(performances) >= 10:
                    # Calculate average performance
                    avg_performance = sum(performances) / len(performances)
                    
                    # Adjust weight based on performance
                    current_weight = self.pattern_weights.get(pattern_type, 0.0)
                    adjustment = (avg_performance - 0.5) * 0.2  # ±20% adjustment
                    new_weight = current_weight + adjustment
                    
                    # Ensure reasonable bounds
                    new_weight = max(-1.0, min(1.0, new_weight))
                    self.pattern_weights[pattern_type] = new_weight
            
            self.last_adaptation = time.time()
            self.logger.info("✅ Pattern weights adapted based on performance")
            
        except Exception as e:
            self.logger.error(f"Pattern weight adaptation failed: {e}")

    def _calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change); losses.append(0)
            else:
                gains.append(0); losses.append(abs(change))
        if len(gains) < period:
            return 50.0
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_momentum(self, prices, period=5):
        if len(prices) < period + 1:
            return 0.0
        current_price = prices[-1]
        past_price = prices[-period-1]
        return (current_price - past_price) / past_price

    def _calculate_volatility(self, prices, period=20):
        if len(prices) < period:
            return 0.0
        recent = prices[-period:]
        mean_price = sum(recent) / len(recent)
        variance = sum((p - mean_price) ** 2 for p in recent) / len(recent)
        return (variance ** 0.5) / mean_price

    def analyze_volume_confirmation(self, price_change, volume_history):
        if not volume_history or len(volume_history) < 5:
            return True
        recent_volume = volume_history[-5:]
        avg_volume = sum(recent_volume) / len(recent_volume)
        current_volume = recent_volume[-1]
        if abs(price_change) > 0.005:
            return current_volume > avg_volume * 1.5
        elif abs(price_change) > 0.001:
            return current_volume > avg_volume * 1.2
        else:
            return True


