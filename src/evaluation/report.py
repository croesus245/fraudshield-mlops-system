"""
Evaluation report generation.

A proper evaluation report includes:
1. Global metrics
2. Slice metrics (by category, amount bucket, time)
3. Calibration analysis
4. Threshold analysis
5. Failure mode analysis
"""

from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from .metrics import (
    compute_metrics,
    compute_slice_metrics,
    compute_calibration_metrics,
    compute_threshold_analysis,
    compute_precision_at_k,
)


@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation report.
    
    This is what you show in interviews.
    Not just accuracy — disaggregated, honest evaluation.
    """
    
    # Metadata
    model_version: str
    eval_timestamp: str
    n_samples: int
    positive_rate: float
    
    # Global metrics
    global_metrics: dict[str, float]
    
    # Slice metrics
    slice_metrics: dict[str, pd.DataFrame]  # slice_name -> DataFrame
    
    # Calibration
    calibration: dict
    
    # Threshold analysis
    threshold_analysis: pd.DataFrame
    
    # Precision at k
    precision_at_k: dict[str, float]
    
    # Warnings and notes
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "model_version": self.model_version,
            "eval_timestamp": self.eval_timestamp,
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
            "global_metrics": self.global_metrics,
            "slice_metrics": {
                k: v.to_dict("records") for k, v in self.slice_metrics.items()
            },
            "calibration": self.calibration,
            "threshold_analysis": self.threshold_analysis.to_dict("records"),
            "precision_at_k": self.precision_at_k,
            "warnings": self.warnings,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Report saved to {path}")
    
    def print_summary(self) -> None:
        """Print a human-readable summary."""
        print("=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"Model: {self.model_version}")
        print(f"Timestamp: {self.eval_timestamp}")
        print(f"Samples: {self.n_samples:,} (positive rate: {self.positive_rate:.2%})")
        print()
        
        print("GLOBAL METRICS")
        print("-" * 40)
        for metric, value in self.global_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        print()
        
        print("CALIBRATION")
        print("-" * 40)
        print(f"  ECE: {self.calibration['ece']:.4f}")
        print(f"  MCE: {self.calibration['mce']:.4f}")
        print(f"  Brier Score: {self.calibration['brier_score']:.4f}")
        print()
        
        print("PRECISION @ K")
        print("-" * 40)
        for k, value in self.precision_at_k.items():
            print(f"  {k}: {value:.4f}")
        print()
        
        if self.warnings:
            print("⚠️ WARNINGS")
            print("-" * 40)
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print("=" * 60)


def generate_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_version: str = "v1",
    threshold: float = 0.5,
    slice_columns: Optional[dict[str, pd.Series]] = None,
) -> EvaluationReport:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        model_version: Model version string
        threshold: Classification threshold
        slice_columns: Dict of slice_name -> Series for disaggregated eval
        
    Returns:
        EvaluationReport
    """
    logger.info(f"Generating evaluation report for {len(y_true)} samples")
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    warnings = []
    
    # Global metrics
    global_metrics = compute_metrics(y_true, y_pred, y_pred_proba, threshold)
    
    # Check for issues
    if global_metrics["precision"] < 0.5:
        warnings.append(f"Low precision ({global_metrics['precision']:.2f}) - high false positive rate")
    if global_metrics["recall"] < 0.5:
        warnings.append(f"Low recall ({global_metrics['recall']:.2f}) - missing many fraud cases")
    
    # Slice metrics
    slice_metrics = {}
    if slice_columns:
        for slice_name, slice_col in slice_columns.items():
            slice_df = compute_slice_metrics(
                y_true, y_pred, y_pred_proba, slice_col, threshold
            )
            slice_metrics[slice_name] = slice_df
            
            # Check for slice issues
            for _, row in slice_df.iterrows():
                if row["recall"] < 0.3 and row["n_samples"] > 100:
                    warnings.append(
                        f"Very low recall ({row['recall']:.2f}) for {slice_name}={row['slice']}"
                    )
    
    # Calibration
    calibration = compute_calibration_metrics(y_true, y_pred_proba)
    
    if calibration["ece"] > 0.1:
        warnings.append(f"Poor calibration (ECE={calibration['ece']:.3f})")
    
    # Threshold analysis
    threshold_analysis = compute_threshold_analysis(y_true, y_pred_proba)
    
    # Precision at k
    precision_at_k = compute_precision_at_k(y_true, y_pred_proba)
    
    report = EvaluationReport(
        model_version=model_version,
        eval_timestamp=datetime.now().isoformat(),
        n_samples=len(y_true),
        positive_rate=float(y_true.mean()),
        global_metrics=global_metrics,
        slice_metrics=slice_metrics,
        calibration=calibration,
        threshold_analysis=threshold_analysis,
        precision_at_k=precision_at_k,
        warnings=warnings,
    )
    
    logger.info(f"Report generated with {len(warnings)} warnings")
    return report


def plot_evaluation_report(
    report: EvaluationReport,
    output_dir: Union[str, Path],
) -> None:
    """
    Generate visualization plots for the report.
    
    Creates:
    - Calibration curve
    - Precision-recall curve
    - Threshold analysis
    - Slice comparison
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Calibration curve
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cal = report.calibration["calibration_curve"]
    ax.plot(cal["prob_pred"], cal["prob_true"], "o-", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual probability")
    ax.set_title(f"Calibration Curve (ECE={report.calibration['ece']:.3f})")
    ax.legend()
    
    fig.savefig(output_dir / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 2. Threshold analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ta = report.threshold_analysis
    
    # Precision-Recall tradeoff
    axes[0].plot(ta["threshold"], ta["precision"], "b-", label="Precision")
    axes[0].plot(ta["threshold"], ta["recall"], "r-", label="Recall")
    axes[0].plot(ta["threshold"], ta["f1"], "g--", label="F1")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision/Recall vs Threshold")
    axes[0].legend()
    axes[0].grid(True)
    
    # Confusion matrix counts
    axes[1].plot(ta["threshold"], ta["true_positives"], label="True Positives")
    axes[1].plot(ta["threshold"], ta["false_positives"], label="False Positives")
    axes[1].plot(ta["threshold"], ta["false_negatives"], label="False Negatives")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confusion Matrix vs Threshold")
    axes[1].legend()
    axes[1].grid(True)
    
    fig.savefig(output_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 3. Slice comparison (if available)
    for slice_name, slice_df in report.slice_metrics.items():
        if len(slice_df) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(slice_df))
            width = 0.35
            
            ax.bar(x - width/2, slice_df["precision"], width, label="Precision")
            ax.bar(x + width/2, slice_df["recall"], width, label="Recall")
            
            ax.set_xlabel(slice_name)
            ax.set_ylabel("Score")
            ax.set_title(f"Metrics by {slice_name}")
            ax.set_xticks(x)
            ax.set_xticklabels(slice_df["slice"], rotation=45, ha="right")
            ax.legend()
            
            fig.savefig(
                output_dir / f"slice_{slice_name}.png",
                dpi=150,
                bbox_inches="tight"
            )
            plt.close(fig)
    
    logger.info(f"Plots saved to {output_dir}")
