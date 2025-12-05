"""
AppScanner Models

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

This module provides:
1. AppScannerNN: Neural network classifier (PyTorch version)
2. AppScannerRF: Random Forest classifier wrapper (original paper approach)
3. AppScannerEnsemble: Hybrid ensemble of NN and RF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AppScannerNN(nn.Module):
    """
    Neural Network classifier for AppScanner.

    Architecture:
    - Input: Statistical features (54 or 40 dimensions)
    - Hidden layers with ReLU, BatchNorm, and Dropout
    - Output: Class probabilities

    This is a PyTorch implementation that achieves similar performance
    to the Random Forest classifier from the original paper.
    """

    def __init__(
        self,
        input_dim: int = 54,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 110,
        dropout: float = 0.3,
    ):
        """
        Initialize AppScanner Neural Network.

        Args:
            input_dim: Number of input features (54 full, 40 selected)
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes (apps)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Build network layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Output layer
        self.classifier = nn.Linear(prev_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        threshold: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with confidence threshold (as in paper Section V-C).

        Args:
            x: Input tensor
            threshold: Confidence threshold for prediction

        Returns:
            predictions: Predicted class indices
            confidences: Prediction confidence values
            is_confident: Boolean mask for confident predictions
        """
        probs = self.predict_proba(x)
        confidences, predictions = probs.max(dim=-1)
        is_confident = confidences >= threshold
        return predictions, confidences, is_confident


class AppScannerDeep(nn.Module):
    """
    Deeper neural network variant with residual connections.

    This variant provides better gradient flow for deeper architectures
    and can capture more complex feature interactions.
    """

    def __init__(
        self,
        input_dim: int = 54,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 110,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with pre-activation."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AppScannerRF:
    """
    Random Forest classifier wrapper (original paper approach).

    This implements Approach 4 from the paper: Single Large Random Forest.
    Achieves 99.6% accuracy with prediction threshold of 0.9.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest classifier")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AppScannerRF':
        """Train the Random Forest classifier."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        return self.model.predict_proba(X)

    def predict_with_threshold(
        self,
        X: np.ndarray,
        threshold: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence threshold.

        Args:
            X: Feature matrix
            threshold: Confidence threshold

        Returns:
            predictions: Predicted labels
            confidences: Confidence values
            is_confident: Boolean mask
        """
        probs = self.predict_proba(X)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        is_confident = confidences >= threshold
        return predictions, confidences, is_confident

    def feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


class AppScannerSVM:
    """
    SVM classifier wrapper (alternative approach from paper).

    Uses RBF kernel as mentioned in the paper.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SVM classifier")

        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AppScannerSVM':
        """Train the SVM classifier."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        return self.model.predict_proba(X)


class AppScannerEnsemble(nn.Module):
    """
    Ensemble classifier combining Neural Network and Random Forest.

    This hybrid approach can leverage the strengths of both methods:
    - NN: Good at learning complex feature interactions
    - RF: Robust to outliers, provides feature importance
    """

    def __init__(
        self,
        nn_model: AppScannerNN,
        rf_model: Optional[AppScannerRF] = None,
        nn_weight: float = 0.5,
    ):
        super().__init__()
        self.nn_model = nn_model
        self.rf_model = rf_model
        self.nn_weight = nn_weight
        self.rf_weight = 1.0 - nn_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through neural network only."""
        return self.nn_model(x)

    def predict_ensemble(
        self,
        x_tensor: torch.Tensor,
        x_numpy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble prediction combining NN and RF.

        Args:
            x_tensor: Input for neural network
            x_numpy: Same input as numpy array for RF

        Returns:
            predictions: Ensemble predictions
            confidences: Ensemble confidence values
        """
        # NN predictions
        self.nn_model.eval()
        with torch.no_grad():
            nn_probs = self.nn_model.predict_proba(x_tensor).cpu().numpy()

        # RF predictions
        if self.rf_model is not None and self.rf_model.is_fitted:
            rf_probs = self.rf_model.predict_proba(x_numpy)
            # Weighted average
            ensemble_probs = self.nn_weight * nn_probs + self.rf_weight * rf_probs
        else:
            ensemble_probs = nn_probs

        predictions = ensemble_probs.argmax(axis=1)
        confidences = ensemble_probs.max(axis=1)

        return predictions, confidences


def build_model(
    model_type: str = 'nn',
    input_dim: int = 54,
    num_classes: int = 110,
    **kwargs
) -> nn.Module:
    """
    Factory function to build AppScanner model.

    Args:
        model_type: 'nn', 'deep', 'rf', 'svm'
        input_dim: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    if model_type == 'nn':
        return AppScannerNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=kwargs.get('hidden_dims', [256, 128, 64]),
            dropout=kwargs.get('dropout', 0.3),
        )
    elif model_type == 'deep':
        return AppScannerDeep(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_layers=kwargs.get('num_layers', 4),
            dropout=kwargs.get('dropout', 0.3),
        )
    elif model_type == 'rf':
        return AppScannerRF(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=kwargs.get('random_state', 42),
        )
    elif model_type == 'svm':
        return AppScannerSVM(
            C=kwargs.get('C', 1.0),
            kernel=kwargs.get('kernel', 'rbf'),
            random_state=kwargs.get('random_state', 42),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test models
    batch_size = 32
    input_dim = 54
    num_classes = 110

    # Test NN
    model = AppScannerNN(input_dim=input_dim, num_classes=num_classes)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"AppScannerNN output shape: {output.shape}")

    # Test with confidence
    preds, confs, confident = model.predict_with_confidence(x, threshold=0.9)
    print(f"Predictions: {preds.shape}, Confident: {confident.sum()}/{len(confident)}")

    # Test Deep
    model_deep = AppScannerDeep(input_dim=input_dim, num_classes=num_classes)
    output_deep = model_deep(x)
    print(f"AppScannerDeep output shape: {output_deep.shape}")

    # Test RF
    if SKLEARN_AVAILABLE:
        rf = AppScannerRF()
        X_train = np.random.randn(100, input_dim)
        y_train = np.random.randint(0, num_classes, 100)
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_train[:10])
        print(f"RF predictions: {preds_rf}")

    print("All model tests passed!")
