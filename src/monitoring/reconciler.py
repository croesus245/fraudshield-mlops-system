"""
Label Reconciliation Service.

THE DELAYED LABEL PROBLEM:
- Transaction happens NOW
- Fraud label arrives 30-90 DAYS LATER (via chargeback)
- We need to match labels back to predictions

THIS SERVICE:
1. Stores predictions with transaction IDs
2. Receives delayed labels
3. Joins them together
4. Computes "true" performance metrics
5. Triggers retraining when performance degrades
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable
import json
from pathlib import Path
from loguru import logger

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class PredictionRecord:
    """A prediction waiting for its label."""
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    fraud_probability: float
    decision: str
    model_version: str
    prediction_timestamp: datetime
    features_snapshot: dict = field(default_factory=dict)


@dataclass
class LabelRecord:
    """A ground truth label (from chargeback, investigation, etc.)."""
    transaction_id: str
    is_fraud: bool
    label_source: str  # "chargeback", "investigation", "customer_report"
    label_timestamp: datetime
    fraud_type: Optional[str] = None  # "card_not_present", "account_takeover", etc.


@dataclass
class ReconciledRecord:
    """A prediction joined with its label."""
    transaction_id: str
    fraud_probability: float
    predicted_fraud: bool
    actual_fraud: bool
    correct: bool
    model_version: str
    prediction_timestamp: datetime
    label_timestamp: datetime
    label_delay_days: float
    amount: float


class LabelReconciler:
    """
    Reconcile predictions with delayed labels.
    
    WHY THIS MATTERS:
    - Can't evaluate model on recent predictions (no labels yet)
    - Need to track "stale" predictions waiting for labels
    - Performance metrics are always ~30 days behind reality
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_age_days: int = 120,  # Drop predictions older than this
        reconciliation_window_days: int = 90,  # Labels can arrive this late
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_age_days = max_age_days
        self.reconciliation_window_days = reconciliation_window_days
        
        # In-memory storage (production would use a database)
        self._pending_predictions: dict[str, PredictionRecord] = {}
        self._reconciled: list[ReconciledRecord] = []
        self._labels_received: int = 0
        self._labels_matched: int = 0
        
        # Callbacks
        self._on_reconcile: Optional[Callable[[ReconciledRecord], None]] = None
        
        # Load from disk if exists
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def record_prediction(self, prediction: PredictionRecord) -> None:
        """Store a prediction awaiting its label."""
        self._pending_predictions[prediction.transaction_id] = prediction
        self._cleanup_old_predictions()
    
    def record_label(self, label: LabelRecord) -> Optional[ReconciledRecord]:
        """
        Record a label and try to match it with a prediction.
        
        Returns ReconciledRecord if matched, None otherwise.
        """
        self._labels_received += 1
        
        if label.transaction_id not in self._pending_predictions:
            logger.debug(f"Label for unknown transaction: {label.transaction_id}")
            return None
        
        prediction = self._pending_predictions.pop(label.transaction_id)
        self._labels_matched += 1
        
        # Compute label delay
        label_delay = (label.label_timestamp - prediction.prediction_timestamp).total_seconds() / 86400
        
        # Create reconciled record
        reconciled = ReconciledRecord(
            transaction_id=label.transaction_id,
            fraud_probability=prediction.fraud_probability,
            predicted_fraud=prediction.decision in ["block", "review"],
            actual_fraud=label.is_fraud,
            correct=(prediction.decision == "block") == label.is_fraud,
            model_version=prediction.model_version,
            prediction_timestamp=prediction.prediction_timestamp,
            label_timestamp=label.label_timestamp,
            label_delay_days=label_delay,
            amount=prediction.amount,
        )
        
        self._reconciled.append(reconciled)
        
        # Trigger callback if set
        if self._on_reconcile:
            self._on_reconcile(reconciled)
        
        return reconciled
    
    def get_performance_metrics(
        self,
        model_version: Optional[str] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
    ) -> dict:
        """
        Compute performance metrics on reconciled predictions.
        
        These are the TRUE metrics (not on held-out test set).
        """
        records = self._reconciled
        
        if model_version:
            records = [r for r in records if r.model_version == model_version]
        
        if min_date:
            records = [r for r in records if r.prediction_timestamp >= min_date]
        
        if max_date:
            records = [r for r in records if r.prediction_timestamp <= max_date]
        
        if not records:
            return {"error": "No reconciled records found"}
        
        # Compute metrics
        n = len(records)
        actual_frauds = sum(1 for r in records if r.actual_fraud)
        predicted_frauds = sum(1 for r in records if r.predicted_fraud)
        
        tp = sum(1 for r in records if r.actual_fraud and r.predicted_fraud)
        fp = sum(1 for r in records if not r.actual_fraud and r.predicted_fraud)
        fn = sum(1 for r in records if r.actual_fraud and not r.predicted_fraud)
        tn = sum(1 for r in records if not r.actual_fraud and not r.predicted_fraud)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute AUC if we have both classes
        if actual_frauds > 0 and actual_frauds < n:
            from sklearn.metrics import roc_auc_score, average_precision_score
            y_true = [r.actual_fraud for r in records]
            y_score = [r.fraud_probability for r in records]
            roc_auc = roc_auc_score(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
        else:
            roc_auc = None
            pr_auc = None
        
        # Dollar metrics
        total_fraud_amount = sum(r.amount for r in records if r.actual_fraud)
        caught_fraud_amount = sum(r.amount for r in records if r.actual_fraud and r.predicted_fraud)
        false_positive_amount = sum(r.amount for r in records if not r.actual_fraud and r.predicted_fraud)
        
        return {
            "n_samples": n,
            "actual_fraud_rate": actual_frauds / n,
            "predicted_fraud_rate": predicted_frauds / n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_fraud_amount": total_fraud_amount,
            "caught_fraud_amount": caught_fraud_amount,
            "false_positive_amount": false_positive_amount,
            "fraud_catch_rate_dollars": caught_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0,
            "avg_label_delay_days": sum(r.label_delay_days for r in records) / n,
        }
    
    def get_performance_by_model_version(self) -> dict[str, dict]:
        """Get performance metrics broken down by model version."""
        versions = set(r.model_version for r in self._reconciled)
        return {v: self.get_performance_metrics(model_version=v) for v in versions}
    
    def get_rolling_metrics(self, window_days: int = 7) -> list[dict]:
        """Compute rolling performance metrics over time."""
        if not self._reconciled:
            return []
        
        # Sort by prediction timestamp
        sorted_records = sorted(self._reconciled, key=lambda r: r.prediction_timestamp)
        
        results = []
        window = timedelta(days=window_days)
        
        # Compute for each day
        min_date = sorted_records[0].prediction_timestamp.date()
        max_date = sorted_records[-1].prediction_timestamp.date()
        
        current_date = min_date
        while current_date <= max_date:
            window_start = datetime.combine(current_date, datetime.min.time())
            window_end = window_start + window
            
            window_records = [
                r for r in sorted_records
                if window_start <= r.prediction_timestamp < window_end
            ]
            
            if window_records:
                metrics = {
                    "date": current_date.isoformat(),
                    "n_samples": len(window_records),
                    "fraud_rate": sum(1 for r in window_records if r.actual_fraud) / len(window_records),
                    "precision": sum(1 for r in window_records if r.actual_fraud and r.predicted_fraud) / max(1, sum(1 for r in window_records if r.predicted_fraud)),
                    "recall": sum(1 for r in window_records if r.actual_fraud and r.predicted_fraud) / max(1, sum(1 for r in window_records if r.actual_fraud)),
                }
                results.append(metrics)
            
            current_date += timedelta(days=1)
        
        return results
    
    def stats(self) -> dict:
        """Get reconciliation statistics."""
        return {
            "pending_predictions": len(self._pending_predictions),
            "reconciled_records": len(self._reconciled),
            "labels_received": self._labels_received,
            "labels_matched": self._labels_matched,
            "match_rate": self._labels_matched / self._labels_received if self._labels_received > 0 else 0,
        }
    
    def on_reconcile(self, callback: Callable[[ReconciledRecord], None]) -> None:
        """Register a callback for when a prediction is reconciled."""
        self._on_reconcile = callback
    
    def _cleanup_old_predictions(self) -> None:
        """Remove predictions that are too old to ever get labels."""
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        
        old_ids = [
            tid for tid, pred in self._pending_predictions.items()
            if pred.prediction_timestamp < cutoff
        ]
        
        for tid in old_ids:
            del self._pending_predictions[tid]
        
        if old_ids:
            logger.info(f"Cleaned up {len(old_ids)} old predictions")
    
    def save(self) -> None:
        """Save state to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save pending predictions
        pending_path = self.storage_path / "pending.json"
        with open(pending_path, "w") as f:
            json.dump(
                {tid: {
                    "transaction_id": p.transaction_id,
                    "user_id": p.user_id,
                    "merchant_id": p.merchant_id,
                    "amount": p.amount,
                    "fraud_probability": p.fraud_probability,
                    "decision": p.decision,
                    "model_version": p.model_version,
                    "prediction_timestamp": p.prediction_timestamp.isoformat(),
                } for tid, p in self._pending_predictions.items()},
                f,
            )
        
        # Save reconciled records
        reconciled_path = self.storage_path / "reconciled.json"
        with open(reconciled_path, "w") as f:
            json.dump(
                [{
                    "transaction_id": r.transaction_id,
                    "fraud_probability": r.fraud_probability,
                    "predicted_fraud": r.predicted_fraud,
                    "actual_fraud": r.actual_fraud,
                    "correct": r.correct,
                    "model_version": r.model_version,
                    "prediction_timestamp": r.prediction_timestamp.isoformat(),
                    "label_timestamp": r.label_timestamp.isoformat(),
                    "label_delay_days": r.label_delay_days,
                    "amount": r.amount,
                } for r in self._reconciled],
                f,
            )
        
        logger.info(f"Saved reconciler state to {self.storage_path}")
    
    def _load(self) -> None:
        """Load state from disk."""
        pending_path = self.storage_path / "pending.json"
        if pending_path.exists():
            with open(pending_path) as f:
                data = json.load(f)
            self._pending_predictions = {
                tid: PredictionRecord(
                    transaction_id=p["transaction_id"],
                    user_id=p["user_id"],
                    merchant_id=p["merchant_id"],
                    amount=p["amount"],
                    fraud_probability=p["fraud_probability"],
                    decision=p["decision"],
                    model_version=p["model_version"],
                    prediction_timestamp=datetime.fromisoformat(p["prediction_timestamp"]),
                )
                for tid, p in data.items()
            }
        
        reconciled_path = self.storage_path / "reconciled.json"
        if reconciled_path.exists():
            with open(reconciled_path) as f:
                data = json.load(f)
            self._reconciled = [
                ReconciledRecord(
                    transaction_id=r["transaction_id"],
                    fraud_probability=r["fraud_probability"],
                    predicted_fraud=r["predicted_fraud"],
                    actual_fraud=r["actual_fraud"],
                    correct=r["correct"],
                    model_version=r["model_version"],
                    prediction_timestamp=datetime.fromisoformat(r["prediction_timestamp"]),
                    label_timestamp=datetime.fromisoformat(r["label_timestamp"]),
                    label_delay_days=r["label_delay_days"],
                    amount=r["amount"],
                )
                for r in data
            ]
        
        logger.info(f"Loaded reconciler state from {self.storage_path}")
