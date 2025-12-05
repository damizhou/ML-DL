"""
AppScanner Configuration

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

Key Paper Parameters:
- 54 statistical features (narrowed to 40 most important)
- Features computed for: incoming, outgoing, bidirectional packets
- Random Forest classifier (original), Neural Network (this implementation)
- Burst threshold: 1 second
- Flow length: min 7 packets, max 260 packets
- Best approach: Single Large Random Forest with 99.6% accuracy
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class AppScannerConfig:
    """Configuration for AppScanner model and training."""

    # ==========================================================================
    # Feature Extraction Parameters (from Section III-B)
    # ==========================================================================

    # Burst threshold in seconds (packets within this interval = same burst)
    burst_threshold: float = 1.0

    # Flow length constraints
    min_flow_length: int = 7      # Minimum packets per flow
    max_flow_length: int = 260    # Maximum packets per flow (truncate if longer)

    # Number of statistical features per direction
    # 18 base features * 3 directions = 54 total, narrowed to 40
    num_features: int = 40

    # Feature directions
    directions: List[str] = field(default_factory=lambda: ['incoming', 'outgoing', 'bidirectional'])

    # ==========================================================================
    # Statistical Features (from Table I)
    # ==========================================================================
    # Per direction (incoming/outgoing/bidirectional):
    # - Packet count (1)
    # - Min, Max, Mean, Std, Variance (5)
    # - Skewness, Kurtosis (2)
    # - Median Absolute Deviation (1)
    # - Percentiles: 10, 20, 30, 40, 50, 60, 70, 80, 90 (9)
    # Total per direction: 18 features
    # Total: 18 * 3 = 54 features

    percentiles: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90])

    # ==========================================================================
    # Neural Network Parameters (PyTorch version of Random Forest)
    # ==========================================================================

    # Input dimension (after feature selection)
    input_dim: int = 54  # Full features, can be reduced to 40

    # Hidden layer dimensions
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    # Dropout rate
    dropout: float = 0.3

    # Number of classes (default: 110 apps from paper)
    num_classes: int = 110

    # ==========================================================================
    # Training Parameters
    # ==========================================================================

    # Learning rate
    learning_rate: float = 0.001

    # Weight decay (L2 regularization)
    weight_decay: float = 1e-4

    # Batch size
    batch_size: int = 128

    # Number of epochs
    epochs: int = 100

    # Early stopping patience
    patience: int = 10

    # ==========================================================================
    # Classification Thresholds (from Section V-C)
    # ==========================================================================

    # Prediction probability threshold for high-confidence classification
    prediction_threshold: float = 0.9

    # ==========================================================================
    # Random Forest Parameters (for comparison/hybrid approach)
    # ==========================================================================

    # Number of trees in Random Forest
    n_estimators: int = 100

    # Maximum depth of trees
    max_depth: Optional[int] = None

    # Minimum samples to split
    min_samples_split: int = 2

    # ==========================================================================
    # Data Processing
    # ==========================================================================

    # Train/test split ratio
    test_ratio: float = 0.2

    # Validation ratio (from training set)
    val_ratio: float = 0.1

    # Random seed for reproducibility
    seed: int = 42

    # Number of workers for data loading
    num_workers: int = 4

    # ==========================================================================
    # Paths
    # ==========================================================================

    # Data directory
    data_dir: str = './data'

    # Output directory
    output_dir: str = './output'

    # Model checkpoint path
    checkpoint_path: str = './output/appscanner_model.pth'

    # ==========================================================================
    # Device
    # ==========================================================================

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.min_flow_length >= 1, "min_flow_length must be >= 1"
        assert self.max_flow_length >= self.min_flow_length, "max_flow_length must be >= min_flow_length"
        assert self.burst_threshold > 0, "burst_threshold must be > 0"
        assert 0 < self.prediction_threshold <= 1, "prediction_threshold must be in (0, 1]"
        assert len(self.hidden_dims) >= 1, "hidden_dims must have at least 1 layer"


# Default configuration
def get_config() -> AppScannerConfig:
    """Get default AppScanner configuration."""
    return AppScannerConfig()


# Paper-specific configurations for different approaches
def get_approach_config(approach: int) -> AppScannerConfig:
    """
    Get configuration for specific approach from paper.

    Approaches (from Section IV):
    1. Per-app binary classifier
    2. Per-app one-class classifier
    3. Multi-class classifier with separate training/test sets
    4. Single large multi-class classifier (BEST - 99.6%)
    5. Per-app binary with separate training/test
    6. Multi-class with all data
    """
    config = AppScannerConfig()

    if approach == 4:
        # Single Large Random Forest - Best approach
        config.n_estimators = 100
        config.prediction_threshold = 0.9
    elif approach == 1:
        # Per-app binary classifier
        config.num_classes = 2  # Binary per app
    elif approach == 2:
        # One-class classifier
        config.num_classes = 1

    return config


if __name__ == '__main__':
    config = get_config()
    print("AppScanner Configuration:")
    print(f"  Burst threshold: {config.burst_threshold}s")
    print(f"  Flow length: {config.min_flow_length}-{config.max_flow_length} packets")
    print(f"  Number of features: {config.num_features}")
    print(f"  Hidden dimensions: {config.hidden_dims}")
    print(f"  Prediction threshold: {config.prediction_threshold}")
    print(f"  Device: {config.device}")
