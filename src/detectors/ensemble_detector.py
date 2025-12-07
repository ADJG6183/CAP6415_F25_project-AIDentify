"""
Ensemble detector combining multiple detection methods.

This detector combines:
1. Frequency domain analysis
2. Statistical feature analysis
3. Deep learning CNN detector

The ensemble approach provides more robust and accurate detection
by leveraging the strengths of different methods.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import os

from .frequency_detector import FrequencyDomainDetector
from .statistical_detector import StatisticalDetector
from ..models.cnn_detector import DeepLearningDetector


class EnsembleDetector:
    """
    Ensemble detector combining multiple detection methods.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ml_model_path: Optional[str] = None,
        cnn_model_path: Optional[str] = None,
        use_cnn: bool = True
    ):
        """
        Initialize ensemble detector.

        Args:
            weights: Dictionary of weights for each detector
                    {'frequency': w1, 'statistical': w2, 'cnn': w3}
            ml_model_path: Path to trained ML model for feature-based detection
            cnn_model_path: Path to trained CNN model
            use_cnn: Whether to use CNN detector (requires more resources)
        """
        # Initialize individual detectors
        self.frequency_detector = FrequencyDomainDetector()
        self.statistical_detector = StatisticalDetector()

        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn_detector = DeepLearningDetector(model_path=cnn_model_path)
        else:
            self.cnn_detector = None

        # Set default weights if not provided
        if weights is None:
            if use_cnn:
                self.weights = {
                    'frequency': 0.2,
                    'statistical': 0.2,
                    'cnn': 0.6
                }
            else:
                self.weights = {
                    'frequency': 0.5,
                    'statistical': 0.5,
                    'cnn': 0.0
                }
        else:
            self.weights = weights

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # ML model for feature-based classification
        self.ml_model = None
        if ml_model_path is not None and os.path.exists(ml_model_path):
            with open(ml_model_path, 'rb') as f:
                self.ml_model = pickle.load(f)

    def extract_all_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from all detectors.

        Args:
            image: Input image

        Returns:
            Dictionary of features from each detector
        """
        features = {}

        # Frequency domain features
        features['frequency'] = self.frequency_detector.extract_features(image)

        # Statistical features
        features['statistical'] = self.statistical_detector.extract_features(image)

        # CNN features (if enabled)
        if self.use_cnn and self.cnn_detector is not None:
            features['cnn'] = self.cnn_detector.extract_features(image)

        return features

    def predict_proba(self, image: np.ndarray, method: str = 'weighted_average') -> float:
        """
        Predict probability of image being AI-generated.

        Args:
            image: Input image (RGB numpy array)
            method: Ensemble method ('weighted_average', 'ml_model', 'voting')

        Returns:
            Probability of being AI-generated (0-1)
        """
        if method == 'ml_model' and self.ml_model is not None:
            return self._predict_with_ml_model(image)
        elif method == 'voting':
            return self._predict_with_voting(image)
        else:
            return self._predict_with_weighted_average(image)

    def _predict_with_weighted_average(self, image: np.ndarray) -> float:
        """Predict using weighted average of individual detectors."""
        predictions = {}

        # Get predictions from each detector
        freq_features = self.frequency_detector.extract_features(image)
        predictions['frequency'] = self.frequency_detector.predict_proba(freq_features)

        stat_features = self.statistical_detector.extract_features(image)
        predictions['statistical'] = self.statistical_detector.predict_proba(stat_features)

        if self.use_cnn and self.cnn_detector is not None:
            predictions['cnn'] = self.cnn_detector.predict_proba(image)
        else:
            predictions['cnn'] = 0.0

        # Compute weighted average
        ensemble_score = 0.0
        for detector_name, weight in self.weights.items():
            ensemble_score += weight * predictions[detector_name]

        return ensemble_score

    def _predict_with_voting(self, image: np.ndarray, threshold: float = 0.5) -> float:
        """Predict using majority voting."""
        votes = []

        # Get predictions from each detector
        freq_features = self.frequency_detector.extract_features(image)
        freq_pred = self.frequency_detector.predict_proba(freq_features)
        votes.append(1 if freq_pred > threshold else 0)

        stat_features = self.statistical_detector.extract_features(image)
        stat_pred = self.statistical_detector.predict_proba(stat_features)
        votes.append(1 if stat_pred > threshold else 0)

        if self.use_cnn and self.cnn_detector is not None:
            cnn_pred = self.cnn_detector.predict_proba(image)
            votes.append(1 if cnn_pred > threshold else 0)

        # Return percentage of votes for AI-generated
        return sum(votes) / len(votes)

    def _predict_with_ml_model(self, image: np.ndarray) -> float:
        """Predict using trained ML model on combined features."""
        if self.ml_model is None:
            raise ValueError("ML model not loaded")

        # Extract all features
        all_features = []

        freq_features = self.frequency_detector.extract_features(image)
        all_features.extend(freq_features)

        stat_features = self.statistical_detector.extract_features(image)
        all_features.extend(stat_features)

        # Combine features and ensure consistent dtype
        feature_vector = np.array(all_features, dtype=np.float64).reshape(1, -1)

        # Predict
        if hasattr(self.ml_model, 'predict_proba'):
            proba = self.ml_model.predict_proba(feature_vector)[0, 1]
        else:
            # For models without predict_proba, use decision function
            score = self.ml_model.decision_function(feature_vector)[0]
            # Convert to probability using sigmoid
            proba = 1 / (1 + np.exp(-score))

        return proba

    def predict(self, image: np.ndarray, threshold: float = 0.5, method: str = 'weighted_average') -> Dict:
        """
        Predict whether image is AI-generated.

        Args:
            image: Input image
            threshold: Classification threshold
            method: Ensemble method to use

        Returns:
            Dictionary with prediction results
        """
        # Get probability
        probability = self.predict_proba(image, method=method)

        # Get individual predictions for debugging
        freq_features = self.frequency_detector.extract_features(image)
        freq_prob = self.frequency_detector.predict_proba(freq_features)

        stat_features = self.statistical_detector.extract_features(image)
        stat_prob = self.statistical_detector.predict_proba(stat_features)

        cnn_prob = None
        if self.use_cnn and self.cnn_detector is not None:
            cnn_prob = self.cnn_detector.predict_proba(image)

        return {
            'is_ai_generated': probability > threshold,
            'probability': probability,
            'confidence': abs(probability - 0.5) * 2,  # 0 to 1
            'threshold': threshold,
            'individual_predictions': {
                'frequency': freq_prob,
                'statistical': stat_prob,
                'cnn': cnn_prob
            },
            'method': method
        }

    def save_ml_model(self, path: str):
        """Save the ML model to disk."""
        if self.ml_model is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.ml_model, f)

    def load_ml_model(self, path: str):
        """Load ML model from disk."""
        with open(path, 'rb') as f:
            self.ml_model = pickle.load(f)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from ML model (if available).

        Returns:
            Feature importance array or None
        """
        if self.ml_model is None:
            return None

        if hasattr(self.ml_model, 'feature_importances_'):
            return self.ml_model.feature_importances_
        elif hasattr(self.ml_model, 'coef_'):
            return np.abs(self.ml_model.coef_[0])
        else:
            return None
