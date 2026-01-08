"""
Stress tests for model robustness.

The brutal truth: your model will face conditions worse than your test set.
- Noisy labels
- Missing features
- Distribution shift
- Adversarial inputs

Test for these before production does it for you.
"""

from typing import Callable, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger

from .metrics import compute_metrics


@dataclass
class StressTestResult:
    """Result of a single stress test."""
    name: str
    description: str
    passed: bool
    baseline_metric: float
    stressed_metric: float
    degradation: float
    threshold: float
    details: Optional[dict] = None


def label_noise_test(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    predict_fn: Callable,
    X: pd.DataFrame,
    noise_rate: float = 0.1,
    metric: str = "pr_auc",
    max_degradation: float = 0.1,
) -> StressTestResult:
    """
    Test robustness to label noise.
    
    Simulates training with noisy labels by flipping a fraction of test labels.
    A robust model should degrade gracefully.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        predict_fn: Function to get predictions (not used here, for interface consistency)
        X: Features (not used here)
        noise_rate: Fraction of labels to flip
        metric: Metric to evaluate
        max_degradation: Maximum allowed degradation
        
    Returns:
        StressTestResult
    """
    # Baseline
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    baseline_metrics = compute_metrics(y_true, y_pred, y_pred_proba, threshold)
    baseline_value = baseline_metrics.get(metric, 0)
    
    # Flip labels
    n_flip = int(len(y_true) * noise_rate)
    flip_indices = np.random.choice(len(y_true), n_flip, replace=False)
    y_noisy = y_true.copy()
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    
    # Evaluate with noisy labels
    noisy_metrics = compute_metrics(y_noisy, y_pred, y_pred_proba, threshold)
    noisy_value = noisy_metrics.get(metric, 0)
    
    degradation = (baseline_value - noisy_value) / baseline_value if baseline_value > 0 else 0
    passed = degradation <= max_degradation
    
    return StressTestResult(
        name="label_noise",
        description=f"Test with {noise_rate:.0%} label noise",
        passed=passed,
        baseline_metric=baseline_value,
        stressed_metric=noisy_value,
        degradation=degradation,
        threshold=max_degradation,
        details={"noise_rate": noise_rate, "n_flipped": n_flip},
    )


def feature_perturbation_test(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    predict_fn: Callable,
    X: pd.DataFrame,
    perturb_rate: float = 0.1,
    perturb_columns: Optional[list[str]] = None,
    metric: str = "pr_auc",
    max_degradation: float = 0.15,
) -> StressTestResult:
    """
    Test robustness to feature perturbation.
    
    Simulates noisy or corrupted features.
    """
    # Baseline
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    baseline_metrics = compute_metrics(y_true, y_pred, y_pred_proba, threshold)
    baseline_value = baseline_metrics.get(metric, 0)
    
    # Perturb features
    X_perturbed = X.copy()
    
    if perturb_columns is None:
        # Perturb numerical columns
        perturb_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in perturb_columns:
        if col in X_perturbed.columns:
            n_perturb = int(len(X_perturbed) * perturb_rate)
            perturb_indices = np.random.choice(len(X_perturbed), n_perturb, replace=False)
            
            # Add noise proportional to std
            std = X_perturbed[col].std()
            noise = np.random.normal(0, std * 0.5, n_perturb)
            X_perturbed.iloc[perturb_indices, X_perturbed.columns.get_loc(col)] += noise
    
    # Re-predict
    perturbed_proba = predict_fn(X_perturbed)
    perturbed_pred = (perturbed_proba >= threshold).astype(int)
    
    perturbed_metrics = compute_metrics(y_true, perturbed_pred, perturbed_proba, threshold)
    perturbed_value = perturbed_metrics.get(metric, 0)
    
    degradation = (baseline_value - perturbed_value) / baseline_value if baseline_value > 0 else 0
    passed = degradation <= max_degradation
    
    return StressTestResult(
        name="feature_perturbation",
        description=f"Test with {perturb_rate:.0%} feature noise",
        passed=passed,
        baseline_metric=baseline_value,
        stressed_metric=perturbed_value,
        degradation=degradation,
        threshold=max_degradation,
        details={
            "perturb_rate": perturb_rate,
            "columns_perturbed": perturb_columns,
        },
    )


def missing_feature_test(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    predict_fn: Callable,
    X: pd.DataFrame,
    missing_rate: float = 0.2,
    metric: str = "pr_auc",
    max_degradation: float = 0.2,
) -> StressTestResult:
    """
    Test robustness to missing features.
    
    Simulates data quality issues where features are missing.
    """
    # Baseline
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    baseline_metrics = compute_metrics(y_true, y_pred, y_pred_proba, threshold)
    baseline_value = baseline_metrics.get(metric, 0)
    
    # Create missing values
    X_missing = X.copy()
    
    for col in X_missing.columns:
        n_missing = int(len(X_missing) * missing_rate)
        missing_indices = np.random.choice(len(X_missing), n_missing, replace=False)
        X_missing.iloc[missing_indices, X_missing.columns.get_loc(col)] = 0  # Fill with 0
    
    # Re-predict
    missing_proba = predict_fn(X_missing)
    missing_pred = (missing_proba >= threshold).astype(int)
    
    missing_metrics = compute_metrics(y_true, missing_pred, missing_proba, threshold)
    missing_value = missing_metrics.get(metric, 0)
    
    degradation = (baseline_value - missing_value) / baseline_value if baseline_value > 0 else 0
    passed = degradation <= max_degradation
    
    return StressTestResult(
        name="missing_features",
        description=f"Test with {missing_rate:.0%} missing features",
        passed=passed,
        baseline_metric=baseline_value,
        stressed_metric=missing_value,
        degradation=degradation,
        threshold=max_degradation,
        details={"missing_rate": missing_rate},
    )


def class_imbalance_test(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    predict_fn: Callable,
    X: pd.DataFrame,
    target_positive_rate: float = 0.01,
    metric: str = "recall",
    min_performance: float = 0.5,
) -> StressTestResult:
    """
    Test performance under extreme class imbalance.
    
    Subsample to create more imbalanced test set.
    """
    # Baseline
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)
    baseline_metrics = compute_metrics(y_true, y_pred, y_pred_proba, threshold)
    baseline_value = baseline_metrics.get(metric, 0)
    
    # Current positive rate
    current_rate = y_true.mean()
    
    if current_rate <= target_positive_rate:
        # Already more imbalanced, skip
        return StressTestResult(
            name="class_imbalance",
            description=f"Test skipped - already imbalanced ({current_rate:.2%})",
            passed=True,
            baseline_metric=baseline_value,
            stressed_metric=baseline_value,
            degradation=0,
            threshold=min_performance,
        )
    
    # Subsample negatives to achieve target rate
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    n_pos = pos_mask.sum()
    target_n_neg = int(n_pos / target_positive_rate - n_pos)
    target_n_neg = min(target_n_neg, neg_mask.sum())
    
    neg_indices = np.where(neg_mask)[0]
    sampled_neg = np.random.choice(neg_indices, target_n_neg, replace=False)
    
    all_indices = np.concatenate([np.where(pos_mask)[0], sampled_neg])
    
    y_imb = y_true[all_indices]
    proba_imb = y_pred_proba[all_indices]
    pred_imb = (proba_imb >= threshold).astype(int)
    
    imb_metrics = compute_metrics(y_imb, pred_imb, proba_imb, threshold)
    imb_value = imb_metrics.get(metric, 0)
    
    passed = imb_value >= min_performance
    degradation = (baseline_value - imb_value) / baseline_value if baseline_value > 0 else 0
    
    return StressTestResult(
        name="class_imbalance",
        description=f"Test with {target_positive_rate:.2%} positive rate",
        passed=passed,
        baseline_metric=baseline_value,
        stressed_metric=imb_value,
        degradation=degradation,
        threshold=min_performance,
        details={
            "original_positive_rate": float(current_rate),
            "tested_positive_rate": float(y_imb.mean()),
        },
    )


class StressTest:
    """Run a suite of stress tests on a model."""
    
    def __init__(self):
        self.results: list[StressTestResult] = []
    
    def run_all(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        predict_fn: Callable,
        X: pd.DataFrame,
    ) -> list[StressTestResult]:
        """
        Run all stress tests.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            predict_fn: Function to get new predictions
            X: Feature matrix
            
        Returns:
            List of StressTestResults
        """
        logger.info("Running stress tests...")
        
        np.random.seed(42)  # Reproducibility
        
        self.results = []
        
        # Label noise
        result = label_noise_test(y_true, y_pred_proba, predict_fn, X)
        self.results.append(result)
        logger.info(f"  {result.name}: {'PASS' if result.passed else 'FAIL'}")
        
        # Feature perturbation
        result = feature_perturbation_test(y_true, y_pred_proba, predict_fn, X)
        self.results.append(result)
        logger.info(f"  {result.name}: {'PASS' if result.passed else 'FAIL'}")
        
        # Missing features
        result = missing_feature_test(y_true, y_pred_proba, predict_fn, X)
        self.results.append(result)
        logger.info(f"  {result.name}: {'PASS' if result.passed else 'FAIL'}")
        
        # Class imbalance
        result = class_imbalance_test(y_true, y_pred_proba, predict_fn, X)
        self.results.append(result)
        logger.info(f"  {result.name}: {'PASS' if result.passed else 'FAIL'}")
        
        n_passed = sum(1 for r in self.results if r.passed)
        logger.info(f"Stress tests: {n_passed}/{len(self.results)} passed")
        
        return self.results
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([
            {
                "name": r.name,
                "description": r.description,
                "passed": r.passed,
                "baseline": r.baseline_metric,
                "stressed": r.stressed_metric,
                "degradation": f"{r.degradation:.1%}",
            }
            for r in self.results
        ])


def run_stress_tests(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    predict_fn: Callable,
    X: pd.DataFrame,
) -> list[StressTestResult]:
    """Convenience function to run all stress tests."""
    tester = StressTest()
    return tester.run_all(y_true, y_pred_proba, predict_fn, X)
