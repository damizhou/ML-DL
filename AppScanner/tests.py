"""
AppScanner Unit Tests

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

Run tests:
    python -m pytest tests.py -v
    python tests.py
"""

import unittest
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from config import AppScannerConfig, get_config, get_approach_config
from models import (
    AppScannerNN,
    AppScannerDeep,
    AppScannerRF,
    ResidualBlock,
    build_model,
    SKLEARN_AVAILABLE,
)
from data import (
    StatisticalFeatureExtractor,
    AppScannerDataset,
    create_dataloaders,
)
from engine import (
    EarlyStopping,
    train_one_epoch,
    evaluate,
)


class TestConfig(unittest.TestCase):
    """Test configuration module."""

    def test_default_config(self):
        """Test default configuration values match paper."""
        config = get_config()

        # Paper parameters
        self.assertEqual(config.burst_threshold, 1.0)  # 1 second burst threshold
        self.assertEqual(config.min_flow_length, 7)    # Minimum 7 packets
        self.assertEqual(config.max_flow_length, 260)  # Maximum 260 packets
        self.assertEqual(config.prediction_threshold, 0.9)  # 90% confidence

    def test_approach_config(self):
        """Test approach-specific configurations."""
        # Approach 4: Single Large Random Forest (best)
        config4 = get_approach_config(4)
        self.assertEqual(config4.n_estimators, 100)
        self.assertEqual(config4.prediction_threshold, 0.9)

    def test_config_validation(self):
        """Test configuration validation."""
        config = get_config()

        # Test valid config
        config.__post_init__()  # Should not raise

        # Test invalid min_flow_length
        config.min_flow_length = 0
        with self.assertRaises(AssertionError):
            config.__post_init__()

    def test_feature_dimensions(self):
        """Test feature dimension calculation."""
        config = get_config()

        # 18 features per direction * 3 directions = 54
        self.assertEqual(len(config.directions), 3)
        self.assertEqual(len(config.percentiles), 9)
        # 18 = 1 (count) + 5 (basic stats) + 2 (shape) + 1 (MAD) + 9 (percentiles)
        expected_per_direction = 1 + 5 + 2 + 1 + 9
        self.assertEqual(expected_per_direction, 18)


class TestStatisticalFeatures(unittest.TestCase):
    """Test statistical feature extraction (paper Section III-B)."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = StatisticalFeatureExtractor()
        np.random.seed(42)

    def test_direction_features_shape(self):
        """Test that direction features have correct shape."""
        lengths = np.array([100, 200, 300, 400, 500])
        features = self.extractor.extract_direction_features(lengths)

        # 18 features per direction
        self.assertEqual(features.shape, (18,))

    def test_direction_features_values(self):
        """Test specific feature values."""
        lengths = np.array([100, 200, 300, 400, 500])
        features = self.extractor.extract_direction_features(lengths)

        # Check basic statistics
        self.assertEqual(features[0], 5)           # Packet count
        self.assertEqual(features[1], 100)         # Min
        self.assertEqual(features[2], 500)         # Max
        self.assertAlmostEqual(features[3], 300)   # Mean

    def test_flow_features_shape(self):
        """Test that flow features have correct shape."""
        lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        directions = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

        features = self.extractor.extract_flow_features(lengths, directions)

        # 54 features total (18 * 3 directions)
        self.assertEqual(features.shape, (54,))

    def test_flow_features_minimum_length(self):
        """Test minimum flow length requirement."""
        # Flow with less than 7 packets should return None
        lengths = [100, 200, 300]
        directions = [1, -1, 1]

        features = self.extractor.extract_flow_features(lengths, directions)
        self.assertIsNone(features)

    def test_flow_features_truncation(self):
        """Test flow truncation to max length."""
        # Create flow with more than 260 packets
        lengths = list(range(300))
        directions = [1 if i % 2 == 0 else -1 for i in range(300)]

        extractor = StatisticalFeatureExtractor(max_packets=260)
        features = extractor.extract_flow_features(lengths, directions)

        # Should still extract features (truncated to 260)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (54,))

    def test_empty_direction(self):
        """Test handling of empty direction."""
        lengths = np.array([])
        features = self.extractor.extract_direction_features(lengths)

        # Should return zeros
        self.assertEqual(features.shape, (18,))
        self.assertTrue(np.all(features == 0))


class TestModels(unittest.TestCase):
    """Test model architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_dim = 54
        self.num_classes = 110
        torch.manual_seed(42)

    def test_appscanner_nn_forward(self):
        """Test AppScannerNN forward pass."""
        model = AppScannerNN(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

        x = torch.randn(self.batch_size, self.input_dim)
        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_appscanner_nn_predict_proba(self):
        """Test probability prediction."""
        model = AppScannerNN(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

        x = torch.randn(self.batch_size, self.input_dim)
        probs = model.predict_proba(x)

        # Check shape
        self.assertEqual(probs.shape, (self.batch_size, self.num_classes))

        # Check probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(self.batch_size)))

    def test_appscanner_nn_confidence_threshold(self):
        """Test prediction with confidence threshold."""
        model = AppScannerNN(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

        x = torch.randn(self.batch_size, self.input_dim)
        predictions, confidences, is_confident = model.predict_with_confidence(
            x, threshold=0.9
        )

        self.assertEqual(predictions.shape, (self.batch_size,))
        self.assertEqual(confidences.shape, (self.batch_size,))
        self.assertEqual(is_confident.shape, (self.batch_size,))
        self.assertTrue(is_confident.dtype == torch.bool)

    def test_appscanner_deep_forward(self):
        """Test AppScannerDeep forward pass."""
        model = AppScannerDeep(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

        x = torch.randn(self.batch_size, self.input_dim)
        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_residual_block(self):
        """Test residual block."""
        block = ResidualBlock(dim=256)

        x = torch.randn(self.batch_size, 256)
        output = block(x)

        self.assertEqual(output.shape, (self.batch_size, 256))

    @unittest.skipIf(not SKLEARN_AVAILABLE, "scikit-learn not installed")
    def test_appscanner_rf(self):
        """Test Random Forest classifier."""
        rf = AppScannerRF(n_estimators=10)

        X_train = np.random.randn(100, self.input_dim)
        y_train = np.random.randint(0, self.num_classes, 100)

        rf.fit(X_train, y_train)
        self.assertTrue(rf.is_fitted)

        predictions = rf.predict(X_train[:10])
        self.assertEqual(predictions.shape, (10,))

        probs = rf.predict_proba(X_train[:10])
        self.assertEqual(probs.shape[0], 10)

    def test_build_model_factory(self):
        """Test model factory function."""
        # NN model
        model_nn = build_model('nn', self.input_dim, self.num_classes)
        self.assertIsInstance(model_nn, AppScannerNN)

        # Deep model
        model_deep = build_model('deep', self.input_dim, self.num_classes)
        self.assertIsInstance(model_deep, AppScannerDeep)

    def test_model_gradients(self):
        """Test that model gradients flow correctly."""
        model = AppScannerNN(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

        x = torch.randn(self.batch_size, self.input_dim)
        y = torch.randint(0, self.num_classes, (self.batch_size,))

        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestDataset(unittest.TestCase):
    """Test dataset and dataloaders."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 54
        self.n_classes = 10

        self.features = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.labels = np.random.randint(0, self.n_classes, self.n_samples)

    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = AppScannerDataset(self.features, self.labels)

        self.assertEqual(len(dataset), self.n_samples)

        x, y = dataset[0]
        self.assertEqual(x.shape, (self.n_features,))
        self.assertIsInstance(y, torch.Tensor)

    def test_dataset_normalization(self):
        """Test feature normalization."""
        dataset = AppScannerDataset(self.features, self.labels, normalize=True)

        mean, std = dataset.get_normalization_params()
        self.assertEqual(mean.shape, (self.n_features,))
        self.assertEqual(std.shape, (self.n_features,))

    def test_create_dataloaders(self):
        """Test dataloader creation."""
        train_loader, val_loader, test_loader, norm_params = create_dataloaders(
            self.features, self.labels,
            batch_size=16,
            test_ratio=0.2,
            val_ratio=0.1,
        )

        # Check loaders exist
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

        # Check batch
        for x, y in train_loader:
            self.assertEqual(x.shape[1], self.n_features)
            break

    def test_dataloader_split_ratios(self):
        """Test that data is split correctly."""
        train_loader, val_loader, test_loader, _ = create_dataloaders(
            self.features, self.labels,
            batch_size=16,
            test_ratio=0.2,
            val_ratio=0.1,
        )

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        total = train_size + val_size + test_size
        self.assertEqual(total, self.n_samples)

        # Test size should be ~20%
        self.assertAlmostEqual(test_size / self.n_samples, 0.2, places=1)


class TestEngine(unittest.TestCase):
    """Test training engine."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)

        self.input_dim = 54
        self.num_classes = 10
        self.batch_size = 16

        # Create dummy data
        features = np.random.randn(100, self.input_dim).astype(np.float32)
        labels = np.random.randint(0, self.num_classes, 100)

        self.train_loader, self.val_loader, _, _ = create_dataloaders(
            features, labels, batch_size=self.batch_size
        )

        self.model = AppScannerNN(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )
        self.device = torch.device('cpu')

    def test_early_stopping(self):
        """Test early stopping logic."""
        es = EarlyStopping(patience=3, mode='max')

        # Improving scores
        self.assertFalse(es(0.5))
        self.assertFalse(es(0.6))
        self.assertFalse(es(0.7))

        # Non-improving scores
        self.assertFalse(es(0.7))
        self.assertFalse(es(0.65))
        self.assertTrue(es(0.6))  # Should stop

    def test_train_one_epoch(self):
        """Test single epoch training."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        metrics = train_one_epoch(
            self.model,
            self.train_loader,
            criterion,
            optimizer,
            self.device,
        )

        self.assertGreater(metrics.loss, 0)
        self.assertGreaterEqual(metrics.accuracy, 0)
        self.assertLessEqual(metrics.accuracy, 1)

    def test_evaluate(self):
        """Test model evaluation."""
        metrics = evaluate(
            self.model,
            self.val_loader,
            self.device,
            prediction_threshold=0.9,
            num_classes=self.num_classes,
        )

        self.assertGreaterEqual(metrics.accuracy, 0)
        self.assertLessEqual(metrics.accuracy, 1)
        self.assertGreaterEqual(metrics.confidence_ratio, 0)
        self.assertEqual(
            metrics.confusion_matrix.shape,
            (self.num_classes, self.num_classes)
        )


class TestPaperReproduction(unittest.TestCase):
    """
    Tests to verify paper parameter reproduction.

    Paper: AppScanner (Euro S&P 2015)
    """

    def test_paper_parameters(self):
        """Verify key parameters from paper."""
        config = get_config()

        # Section III-B: Feature extraction
        self.assertEqual(config.min_flow_length, 7)
        self.assertEqual(config.max_flow_length, 260)
        self.assertEqual(config.burst_threshold, 1.0)

        # Section V-C: Classification threshold
        self.assertEqual(config.prediction_threshold, 0.9)

        # Feature count: 54 total (18 per direction * 3 directions)
        extractor = StatisticalFeatureExtractor()
        lengths = list(range(10, 20))
        directions = [1 if i % 2 == 0 else -1 for i in range(10)]
        features = extractor.extract_flow_features(lengths, directions)
        self.assertEqual(features.shape[0], 54)

    def test_feature_types(self):
        """Verify all feature types from Table I."""
        extractor = StatisticalFeatureExtractor()

        # Test with known values
        lengths = np.array([100, 200, 300, 400, 500, 600, 700])
        features = extractor.extract_direction_features(lengths)

        # Feature indices:
        # 0: count
        # 1: min, 2: max, 3: mean, 4: std, 5: var
        # 6: skew, 7: kurtosis
        # 8: MAD
        # 9-17: percentiles

        self.assertEqual(features[0], 7)      # count
        self.assertEqual(features[1], 100)    # min
        self.assertEqual(features[2], 700)    # max
        self.assertEqual(features[3], 400)    # mean

        # Verify percentiles exist
        self.assertEqual(len(features[9:18]), 9)  # 9 percentiles

    def test_random_forest_settings(self):
        """Test Random Forest settings from paper."""
        config = get_config()

        # Paper uses 100 trees
        self.assertEqual(config.n_estimators, 100)

    def test_approaches(self):
        """Test different classification approaches from Section IV."""
        # Approach 4 should be default (best approach)
        config4 = get_approach_config(4)
        self.assertEqual(config4.prediction_threshold, 0.9)

        # Approach 1: Binary classifier
        config1 = get_approach_config(1)
        self.assertEqual(config1.num_classes, 2)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestPaperReproduction))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
