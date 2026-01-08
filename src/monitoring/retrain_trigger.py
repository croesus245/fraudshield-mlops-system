"""
Automated Retraining Trigger.

WHEN TO RETRAIN?
- Performance degradation (recall drops below threshold)
- Data drift detected (feature distributions shift)
- Scheduled (weekly/monthly refresh)
- Manual trigger (new fraud pattern discovered)

THIS MODULE:
1. Monitors metrics from label reconciler
2. Monitors drift from drift detector
3. Triggers retraining pipeline when thresholds exceeded
4. Manages model versioning and rollback
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable
from enum import Enum
from pathlib import Path
import json
from loguru import logger


class TriggerReason(Enum):
    """Why retraining was triggered."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    FRAUD_RATE_SPIKE = "fraud_rate_spike"


@dataclass
class RetrainTrigger:
    """A retraining trigger event."""
    trigger_id: str
    reason: TriggerReason
    timestamp: datetime
    metrics_snapshot: dict
    threshold_violated: Optional[str] = None
    model_version_before: Optional[str] = None
    model_version_after: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    notes: str = ""


@dataclass
class RetrainConfig:
    """Configuration for retraining triggers."""
    # Performance thresholds
    min_precision: float = 0.5
    min_recall: float = 0.7
    min_pr_auc: float = 0.75
    
    # Drift thresholds
    max_psi: float = 0.1  # Prediction distribution shift
    max_feature_drift_pct: float = 10.0  # % of features drifting
    
    # Fraud rate thresholds
    fraud_rate_change_threshold: float = 0.2  # 20% change from baseline
    
    # Scheduling
    max_days_without_retrain: int = 30
    min_samples_for_eval: int = 1000
    
    # Cooldown (prevent rapid retraining)
    min_hours_between_retrains: int = 24


class RetrainingTriggerService:
    """
    Service that monitors and triggers retraining.
    
    Watches:
    1. Label reconciler (true performance)
    2. Drift detector (distribution shifts)
    3. Scheduled timer
    """
    
    def __init__(
        self,
        config: Optional[RetrainConfig] = None,
        storage_path: Optional[str] = None,
    ):
        self.config = config or RetrainConfig()
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._triggers: list[RetrainTrigger] = []
        self._last_retrain: Optional[datetime] = None
        self._baseline_fraud_rate: Optional[float] = None
        self._current_model_version: str = "v1"
        
        # Callbacks
        self._on_trigger: Optional[Callable[[RetrainTrigger], None]] = None
        self._retrain_fn: Optional[Callable[[], str]] = None  # Returns new model version
        
        # Load history
        if self.storage_path and self.storage_path.exists():
            self._load()
    
    def check_performance(self, metrics: dict) -> Optional[RetrainTrigger]:
        """
        Check if performance has degraded enough to trigger retraining.
        
        Args:
            metrics: Dict with precision, recall, pr_auc, etc.
        
        Returns:
            RetrainTrigger if triggered, None otherwise.
        """
        if not self._can_retrain():
            return None
        
        violations = []
        
        if metrics.get("precision", 1.0) < self.config.min_precision:
            violations.append(f"precision={metrics['precision']:.3f} < {self.config.min_precision}")
        
        if metrics.get("recall", 1.0) < self.config.min_recall:
            violations.append(f"recall={metrics['recall']:.3f} < {self.config.min_recall}")
        
        if metrics.get("pr_auc") and metrics["pr_auc"] < self.config.min_pr_auc:
            violations.append(f"pr_auc={metrics['pr_auc']:.3f} < {self.config.min_pr_auc}")
        
        if not violations:
            return None
        
        trigger = RetrainTrigger(
            trigger_id=f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reason=TriggerReason.PERFORMANCE_DEGRADATION,
            timestamp=datetime.now(),
            metrics_snapshot=metrics,
            threshold_violated="; ".join(violations),
            model_version_before=self._current_model_version,
        )
        
        return self._process_trigger(trigger)
    
    def check_drift(self, drift_metrics: dict) -> Optional[RetrainTrigger]:
        """
        Check if data drift is significant enough to trigger retraining.
        
        Args:
            drift_metrics: Dict with psi, drifting_features, etc.
        
        Returns:
            RetrainTrigger if triggered, None otherwise.
        """
        if not self._can_retrain():
            return None
        
        violations = []
        
        if drift_metrics.get("psi", 0) > self.config.max_psi:
            violations.append(f"psi={drift_metrics['psi']:.3f} > {self.config.max_psi}")
        
        drift_pct = drift_metrics.get("drifting_features_pct", 0)
        if drift_pct > self.config.max_feature_drift_pct:
            violations.append(f"feature_drift={drift_pct:.1f}% > {self.config.max_feature_drift_pct}%")
        
        if not violations:
            return None
        
        trigger = RetrainTrigger(
            trigger_id=f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reason=TriggerReason.DATA_DRIFT,
            timestamp=datetime.now(),
            metrics_snapshot=drift_metrics,
            threshold_violated="; ".join(violations),
            model_version_before=self._current_model_version,
        )
        
        return self._process_trigger(trigger)
    
    def check_fraud_rate(self, current_fraud_rate: float) -> Optional[RetrainTrigger]:
        """
        Check if fraud rate has changed significantly.
        
        Args:
            current_fraud_rate: Current observed fraud rate
        
        Returns:
            RetrainTrigger if triggered, None otherwise.
        """
        if self._baseline_fraud_rate is None:
            self._baseline_fraud_rate = current_fraud_rate
            return None
        
        if not self._can_retrain():
            return None
        
        change = abs(current_fraud_rate - self._baseline_fraud_rate) / self._baseline_fraud_rate
        
        if change <= self.config.fraud_rate_change_threshold:
            return None
        
        trigger = RetrainTrigger(
            trigger_id=f"fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reason=TriggerReason.FRAUD_RATE_SPIKE,
            timestamp=datetime.now(),
            metrics_snapshot={
                "baseline_fraud_rate": self._baseline_fraud_rate,
                "current_fraud_rate": current_fraud_rate,
                "change_pct": change * 100,
            },
            threshold_violated=f"fraud_rate_change={change*100:.1f}% > {self.config.fraud_rate_change_threshold*100:.1f}%",
            model_version_before=self._current_model_version,
        )
        
        return self._process_trigger(trigger)
    
    def check_scheduled(self) -> Optional[RetrainTrigger]:
        """
        Check if scheduled retraining is due.
        
        Returns:
            RetrainTrigger if triggered, None otherwise.
        """
        if self._last_retrain is None:
            return None
        
        days_since_retrain = (datetime.now() - self._last_retrain).days
        
        if days_since_retrain < self.config.max_days_without_retrain:
            return None
        
        trigger = RetrainTrigger(
            trigger_id=f"sched_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reason=TriggerReason.SCHEDULED,
            timestamp=datetime.now(),
            metrics_snapshot={"days_since_retrain": days_since_retrain},
            threshold_violated=f"days_since_retrain={days_since_retrain} > {self.config.max_days_without_retrain}",
            model_version_before=self._current_model_version,
        )
        
        return self._process_trigger(trigger)
    
    def trigger_manual(self, reason: str = "") -> RetrainTrigger:
        """
        Manually trigger retraining.
        
        Args:
            reason: Why manual retraining was triggered
        
        Returns:
            RetrainTrigger
        """
        trigger = RetrainTrigger(
            trigger_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reason=TriggerReason.MANUAL,
            timestamp=datetime.now(),
            metrics_snapshot={},
            notes=reason,
            model_version_before=self._current_model_version,
        )
        
        return self._process_trigger(trigger)
    
    def _can_retrain(self) -> bool:
        """Check if we're allowed to retrain (cooldown period)."""
        if self._last_retrain is None:
            return True
        
        hours_since = (datetime.now() - self._last_retrain).total_seconds() / 3600
        return hours_since >= self.config.min_hours_between_retrains
    
    def _process_trigger(self, trigger: RetrainTrigger) -> RetrainTrigger:
        """Process a retraining trigger."""
        logger.warning(f"Retraining triggered: {trigger.reason.value}")
        logger.warning(f"  Threshold violated: {trigger.threshold_violated}")
        
        self._triggers.append(trigger)
        
        # Call callback
        if self._on_trigger:
            self._on_trigger(trigger)
        
        # Execute retraining if function is set
        if self._retrain_fn:
            trigger.status = "in_progress"
            try:
                new_version = self._retrain_fn()
                trigger.model_version_after = new_version
                trigger.status = "completed"
                self._current_model_version = new_version
                self._last_retrain = datetime.now()
                logger.info(f"Retraining completed. New model version: {new_version}")
            except Exception as e:
                trigger.status = "failed"
                trigger.notes += f" Error: {str(e)}"
                logger.error(f"Retraining failed: {e}")
        
        self.save()
        return trigger
    
    def set_retrain_function(self, fn: Callable[[], str]) -> None:
        """
        Set the function to call for retraining.
        
        Args:
            fn: Function that executes retraining and returns new model version
        """
        self._retrain_fn = fn
    
    def on_trigger(self, callback: Callable[[RetrainTrigger], None]) -> None:
        """Register a callback for when retraining is triggered."""
        self._on_trigger = callback
    
    def set_baseline_fraud_rate(self, rate: float) -> None:
        """Set the baseline fraud rate for comparison."""
        self._baseline_fraud_rate = rate
    
    def get_trigger_history(self, limit: int = 10) -> list[RetrainTrigger]:
        """Get recent trigger history."""
        return self._triggers[-limit:]
    
    def stats(self) -> dict:
        """Get trigger statistics."""
        by_reason = {}
        for t in self._triggers:
            reason = t.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        return {
            "total_triggers": len(self._triggers),
            "triggers_by_reason": by_reason,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "current_model_version": self._current_model_version,
            "baseline_fraud_rate": self._baseline_fraud_rate,
        }
    
    def save(self) -> None:
        """Save state to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        state = {
            "triggers": [
                {
                    "trigger_id": t.trigger_id,
                    "reason": t.reason.value,
                    "timestamp": t.timestamp.isoformat(),
                    "metrics_snapshot": t.metrics_snapshot,
                    "threshold_violated": t.threshold_violated,
                    "model_version_before": t.model_version_before,
                    "model_version_after": t.model_version_after,
                    "status": t.status,
                    "notes": t.notes,
                }
                for t in self._triggers
            ],
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "baseline_fraud_rate": self._baseline_fraud_rate,
            "current_model_version": self._current_model_version,
        }
        
        with open(self.storage_path / "retrain_state.json", "w") as f:
            json.dump(state, f, indent=2)
    
    def _load(self) -> None:
        """Load state from disk."""
        state_path = self.storage_path / "retrain_state.json"
        if not state_path.exists():
            return
        
        with open(state_path) as f:
            state = json.load(f)
        
        self._triggers = [
            RetrainTrigger(
                trigger_id=t["trigger_id"],
                reason=TriggerReason(t["reason"]),
                timestamp=datetime.fromisoformat(t["timestamp"]),
                metrics_snapshot=t["metrics_snapshot"],
                threshold_violated=t["threshold_violated"],
                model_version_before=t["model_version_before"],
                model_version_after=t["model_version_after"],
                status=t["status"],
                notes=t["notes"],
            )
            for t in state["triggers"]
        ]
        
        if state["last_retrain"]:
            self._last_retrain = datetime.fromisoformat(state["last_retrain"])
        
        self._baseline_fraud_rate = state["baseline_fraud_rate"]
        self._current_model_version = state["current_model_version"]
