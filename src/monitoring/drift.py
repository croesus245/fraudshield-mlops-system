"""
Drift detection.

The brutal truth: your model is wrong the moment you deploy it.
Data changes. User behavior changes. The world changes.

This module detects when your model's assumptions break.
"""

from typing import Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


class DriftType(Enum):
    """Types of drift we detect."""
    FEATURE = "feature"          # Input distribution changed
    PREDICTION = "prediction"    # Output distribution changed
    PERFORMANCE = "performance"  # Metrics degraded (requires labels)


class DriftSeverity(Enum):
    """How severe is the drift?"""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_type: DriftType
    feature: str
    severity: DriftSeverity
    statistic: float
    threshold: float
    p_value: Optional[float] = None
    details: dict = field(default_factory=dict)
    
    @property
    def is_drifted(self) -> bool:
        return self.severity != DriftSeverity.NONE


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI).
    
    PSI measures how much a distribution has shifted.
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.2: Moderate shift
    - PSI >= 0.2: Significant shift
    
    Args:
        reference: Reference distribution (training data)
        current: Current distribution (production data)
        n_bins: Number of bins
        
    Returns:
        PSI value
    """
    # Create bins from reference
    eps = 1e-10
    
    # Handle edge cases
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    
    # Compute percentile bins from reference
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(reference, percentiles)
    bins = np.unique(bins)  # Remove duplicates
    
    if len(bins) < 2:
        return 0.0
    
    # Compute histograms
    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)
    
    # Normalize to percentages
    ref_pct = (ref_counts + eps) / (len(reference) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(current) + eps * len(cur_counts))
    
    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    
    return float(psi)


def compute_ks_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov statistic.
    
    Tests if two samples come from the same distribution.
    
    Returns:
        Tuple of (ks_statistic, p_value)
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0, 1.0
    
    statistic, p_value = stats.ks_2samp(reference, current)
    return float(statistic), float(p_value)


def compute_kl_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute KL divergence between distributions.
    
    Note: This is asymmetric. We compute KL(current || reference).
    """
    eps = 1e-10
    
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    
    # Create bins
    all_data = np.concatenate([reference, current])
    bins = np.histogram_bin_edges(all_data, bins=n_bins)
    
    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    cur_hist, _ = np.histogram(current, bins=bins, density=True)
    
    # Add smoothing
    ref_hist = ref_hist + eps
    cur_hist = cur_hist + eps
    
    # Normalize
    ref_hist = ref_hist / ref_hist.sum()
    cur_hist = cur_hist / cur_hist.sum()
    
    # KL divergence
    kl = np.sum(cur_hist * np.log(cur_hist / ref_hist))
    
    return float(kl)


class DriftDetector:
    """
    Detect drift in features and predictions.
    
    Usage:
        detector = DriftDetector()
        detector.set_reference(training_data)
        
        # Later, in production:
        results = detector.detect(production_data)
        for r in results:
            if r.is_drifted:
                alert(f"Drift detected in {r.feature}")
    """
    
    def __init__(
        self,
        psi_warning: float = 0.1,
        psi_critical: float = 0.2,
        ks_alpha: float = 0.05,
        kl_threshold: float = 0.1,
    ):
        """
        Args:
            psi_warning: PSI threshold for warning
            psi_critical: PSI threshold for critical alert
            ks_alpha: Significance level for KS test
            kl_threshold: KL divergence threshold
        """
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.ks_alpha = ks_alpha
        self.kl_threshold = kl_threshold
        
        # Reference distributions
        self._reference_stats: dict[str, dict] = {}
        self._reference_data: Optional[pd.DataFrame] = None
    
    def set_reference(self, df: pd.DataFrame) -> None:
        """
        Set reference distribution from training data.
        
        Call this once with your training data.
        """
        self._reference_data = df.copy()
        self._reference_stats = {}
        
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                self._reference_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "percentiles": {
                        p: float(df[col].quantile(p / 100))
                        for p in [1, 5, 25, 50, 75, 95, 99]
                    }
                }
        
        logger.info(f"Reference set with {len(df)} samples, {len(self._reference_stats)} features")
    
    def detect(self, df: pd.DataFrame) -> list[DriftResult]:
        """
        Detect drift in current data compared to reference.
        
        Args:
            df: Current production data
            
        Returns:
            List of DriftResults for each feature
        """
        if self._reference_data is None:
            raise RuntimeError("Reference not set. Call set_reference() first.")
        
        results = []
        
        for col in df.columns:
            if col not in self._reference_data.columns:
                continue
            
            if df[col].dtype not in [np.float64, np.int64]:
                continue
            
            # Get data
            ref_data = self._reference_data[col].dropna().values
            cur_data = df[col].dropna().values
            
            if len(cur_data) < 10:
                continue
            
            # Compute PSI
            psi = compute_psi(ref_data, cur_data)
            
            # Compute KS
            ks_stat, ks_pvalue = compute_ks_statistic(ref_data, cur_data)
            
            # Determine severity
            if psi >= self.psi_critical:
                severity = DriftSeverity.CRITICAL
            elif psi >= self.psi_warning or ks_pvalue < self.ks_alpha:
                severity = DriftSeverity.WARNING
            else:
                severity = DriftSeverity.NONE
            
            result = DriftResult(
                drift_type=DriftType.FEATURE,
                feature=col,
                severity=severity,
                statistic=psi,
                threshold=self.psi_warning,
                p_value=ks_pvalue,
                details={
                    "psi": psi,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "ref_mean": self._reference_stats.get(col, {}).get("mean"),
                    "cur_mean": float(df[col].mean()),
                }
            )
            results.append(result)
        
        # Log summary
        n_warning = sum(1 for r in results if r.severity == DriftSeverity.WARNING)
        n_critical = sum(1 for r in results if r.severity == DriftSeverity.CRITICAL)
        
        if n_critical > 0:
            logger.warning(f"Drift detection: {n_critical} critical, {n_warning} warnings")
        elif n_warning > 0:
            logger.info(f"Drift detection: {n_warning} warnings")
        else:
            logger.debug("Drift detection: no significant drift")
        
        return results
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> DriftResult:
        """
        Detect drift in model predictions.
        
        This catches cases where features look similar but model
        behavior has changed.
        """
        psi = compute_psi(reference_predictions, current_predictions)
        kl = compute_kl_divergence(reference_predictions, current_predictions)
        
        if psi >= self.psi_critical or kl >= self.kl_threshold * 2:
            severity = DriftSeverity.CRITICAL
        elif psi >= self.psi_warning or kl >= self.kl_threshold:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NONE
        
        return DriftResult(
            drift_type=DriftType.PREDICTION,
            feature="predictions",
            severity=severity,
            statistic=psi,
            threshold=self.psi_warning,
            details={
                "psi": psi,
                "kl_divergence": kl,
                "ref_mean": float(reference_predictions.mean()),
                "cur_mean": float(current_predictions.mean()),
            }
        )
    
    def to_dataframe(self, results: list[DriftResult]) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([
            {
                "feature": r.feature,
                "drift_type": r.drift_type.value,
                "severity": r.severity.value,
                "statistic": r.statistic,
                "threshold": r.threshold,
                "p_value": r.p_value,
                "is_drifted": r.is_drifted,
            }
            for r in results
        ])


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.2,
) -> list[DriftResult]:
    """
    Convenience function to detect drift.
    
    Args:
        reference: Reference data (training)
        current: Current data (production)
        psi_threshold: PSI threshold for critical drift
        
    Returns:
        List of DriftResults
    """
    detector = DriftDetector(psi_critical=psi_threshold)
    detector.set_reference(reference)
    return detector.detect(current)
