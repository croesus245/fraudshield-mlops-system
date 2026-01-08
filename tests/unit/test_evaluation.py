"""
Unit tests for evaluation metrics.
"""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_metrics,
    compute_calibration_metrics,
    compute_threshold_analysis,
    compute_precision_at_k,
)


class TestMetrics:
    """Tests for metric computation."""
    
    def test_compute_metrics_returns_expected_keys(self, sample_labels, sample_predictions):
        """Test that compute_metrics returns expected keys."""
        metrics = compute_metrics(sample_labels, sample_predictions)
        
        expected_keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
        for key in expected_keys:
            assert key in metrics
    
    def test_compute_metrics_values_in_range(self, sample_labels, sample_predictions):
        """Test that metrics are in valid ranges."""
        metrics = compute_metrics(sample_labels, sample_predictions)
        
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"{key} = {value} not in [0, 1]"
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics["roc_auc"] > 0.9
    
    def test_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 1000, p=[0.9, 0.1])
        y_pred = np.random.random(1000)
        
        metrics = compute_metrics(y_true, y_pred)
        
        # Random should have ROC-AUC around 0.5
        assert 0.4 < metrics["roc_auc"] < 0.6


class TestCalibration:
    """Tests for calibration metrics."""
    
    def test_calibration_metrics_returns_expected_keys(self, sample_labels, sample_predictions):
        """Test calibration metrics keys."""
        metrics = compute_calibration_metrics(sample_labels, sample_predictions)
        
        assert "ece" in metrics
        assert "mce" in metrics
    
    def test_perfect_calibration(self):
        """Test calibration with perfectly calibrated model."""
        # Create perfectly calibrated predictions
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.7, 0.7])
        
        metrics = compute_calibration_metrics(y_true, y_pred)
        
        # Should have low calibration error
        assert metrics["ece"] < 0.3


class TestThresholdAnalysis:
    """Tests for threshold analysis."""
    
    def test_threshold_analysis_returns_list(self, sample_labels, sample_predictions):
        """Test threshold analysis returns list of dicts."""
        results = compute_threshold_analysis(sample_labels, sample_predictions)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert "threshold" in results[0]
        assert "precision" in results[0]
        assert "recall" in results[0]
    
    def test_threshold_ordering(self, sample_labels, sample_predictions):
        """Test thresholds are in order."""
        results = compute_threshold_analysis(sample_labels, sample_predictions)
        
        thresholds = [r["threshold"] for r in results]
        assert thresholds == sorted(thresholds)


class TestPrecisionAtK:
    """Tests for precision@k."""
    
    def test_precision_at_k_values(self):
        """Test precision@k calculation."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        
        # Top 2 predictions are both fraud, so precision@2 = 1.0
        p_at_2 = compute_precision_at_k(y_true, y_pred, k=2)
        assert p_at_2 == 1.0
        
        # Top 5 has 2 fraud out of 5, so precision@5 = 0.4
        p_at_5 = compute_precision_at_k(y_true, y_pred, k=5)
        assert p_at_5 == 0.4
