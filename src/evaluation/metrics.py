"""
Evaluation metrics.

The brutal truth: a single accuracy number is meaningless.
You need:
- Multiple metrics (precision, recall, PR-AUC for imbalanced)
- Slice metrics (how does it perform on subgroups?)
- Calibration (are the probabilities honest?)
- Operating point analysis (precision at different thresholds)
"""

from typing import Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from loguru import logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_pred_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (or will be derived from proba + threshold)
        y_pred_proba: Predicted probabilities (optional but recommended)
        threshold: Classification threshold
        
    Returns:
        Dict of metric name -> value
    """
    # If only probas given, derive predictions
    if y_pred_proba is not None and y_pred is None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {}
    
    # Basic metrics
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix derived
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Probability-based metrics (if available)
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba)
        metrics["brier_score"] = brier_score_loss(y_true, y_pred_proba)
        
        # Log loss
        eps = 1e-15
        y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1 - eps)
        metrics["log_loss"] = -np.mean(
            y_true * np.log(y_pred_proba_clipped) +
            (1 - y_true) * np.log(1 - y_pred_proba_clipped)
        )
    
    # Class distribution
    metrics["positive_rate"] = y_true.mean()
    metrics["predicted_positive_rate"] = y_pred.mean()
    metrics["n_samples"] = len(y_true)
    
    return metrics


def compute_slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    slice_column: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute metrics for each slice/subgroup.
    
    This is critical for fairness and debugging.
    A model that performs well overall might fail on specific segments.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        slice_column: Series defining slices (e.g., merchant_category)
        threshold: Classification threshold
        
    Returns:
        DataFrame with metrics per slice
    """
    results = []
    
    for slice_value in slice_column.unique():
        mask = slice_column == slice_value
        
        if mask.sum() < 10:  # Skip tiny slices
            continue
        
        slice_y_true = y_true[mask]
        slice_y_pred = y_pred[mask]
        slice_proba = y_pred_proba[mask] if y_pred_proba is not None else None
        
        metrics = compute_metrics(
            slice_y_true, slice_y_pred, slice_proba, threshold
        )
        metrics["slice"] = slice_value
        metrics["n_samples"] = int(mask.sum())
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics.
    
    A well-calibrated model: when it says 80% fraud, it should be right 80% of the time.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dict with calibration metrics and curve data
    """
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
    )
    
    # Expected Calibration Error (ECE)
    # Average absolute difference between confidence and accuracy
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_pred_proba[mask].mean()
            bin_weight = mask.sum() / total_samples
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    # Maximum Calibration Error (MCE)
    mce = max(abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0
    
    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier_score_loss(y_true, y_pred_proba),
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
    }


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Analyze model performance at different thresholds.
    
    Essential for choosing the right operating point.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        thresholds: Thresholds to evaluate (default: 0.1 to 0.9)
        
    Returns:
        DataFrame with metrics at each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "predicted_positive_rate": (tp + fp) / len(y_true),
        })
    
    return pd.DataFrame(results)


def compute_precision_at_k(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k_values: Optional[list[int]] = None,
) -> dict[str, float]:
    """
    Compute precision at top-k predictions.
    
    Useful when you can only review N cases.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        k_values: K values to evaluate
        
    Returns:
        Dict of k -> precision@k
    """
    if k_values is None:
        n = len(y_true)
        k_values = [int(n * p) for p in [0.01, 0.05, 0.1, 0.2]]
        k_values = [k for k in k_values if k > 0]
    
    # Sort by probability
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    results = {}
    for k in k_values:
        if k <= len(sorted_labels):
            precision_at_k = sorted_labels[:k].mean()
            results[f"precision@{k}"] = precision_at_k
    
    return results
