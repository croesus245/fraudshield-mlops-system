"""
Alert management.

When drift or issues are detected, alerts need to fire.
This module handles alert routing and logging.
"""

from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
from loguru import logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert to be sent."""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str = "fraudshield"
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "details": self.details,
        }


class AlertManager:
    """
    Manage and route alerts.
    
    In production, this would integrate with:
    - PagerDuty
    - Slack
    - Email
    - etc.
    
    For this project, we log to file and console.
    """
    
    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ):
        """
        Args:
            log_path: Path to write alert logs
            min_severity: Minimum severity to send
        """
        self.log_path = Path(log_path) if log_path else None
        self.min_severity = min_severity
        self._alert_history: list[Alert] = []
    
    def send(self, alert: Alert) -> bool:
        """
        Send an alert.
        
        Returns True if sent, False if filtered.
        """
        # Filter by severity
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
        }
        
        if severity_order[alert.severity] < severity_order[self.min_severity]:
            return False
        
        # Log to console
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(f"🚨 ALERT: {alert.name} - {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(f"⚠️ ALERT: {alert.name} - {alert.message}")
        else:
            logger.info(f"ℹ️ ALERT: {alert.name} - {alert.message}")
        
        # Log to file
        if self.log_path:
            self._write_to_file(alert)
        
        # Track history
        self._alert_history.append(alert)
        
        return True
    
    def _write_to_file(self, alert: Alert) -> None:
        """Write alert to log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")
    
    def create_drift_alert(
        self,
        feature: str,
        drift_value: float,
        threshold: float,
        is_critical: bool = False,
    ) -> Alert:
        """Create an alert for drift detection."""
        severity = AlertSeverity.CRITICAL if is_critical else AlertSeverity.WARNING
        
        return Alert(
            name="drift_detected",
            severity=severity,
            message=f"Drift detected in '{feature}': {drift_value:.3f} (threshold: {threshold:.3f})",
            timestamp=datetime.now(),
            details={
                "feature": feature,
                "drift_value": drift_value,
                "threshold": threshold,
            }
        )
    
    def create_performance_alert(
        self,
        metric: str,
        current_value: float,
        threshold: float,
    ) -> Alert:
        """Create an alert for performance degradation."""
        return Alert(
            name="performance_degradation",
            severity=AlertSeverity.CRITICAL,
            message=f"Performance degraded: {metric}={current_value:.3f} (min: {threshold:.3f})",
            timestamp=datetime.now(),
            details={
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
            }
        )
    
    def create_data_quality_alert(
        self,
        issue: str,
        details: Optional[dict] = None,
    ) -> Alert:
        """Create an alert for data quality issues."""
        return Alert(
            name="data_quality_issue",
            severity=AlertSeverity.WARNING,
            message=f"Data quality issue: {issue}",
            timestamp=datetime.now(),
            details=details,
        )
    
    def create_latency_alert(
        self,
        current_p99: float,
        threshold: float,
    ) -> Alert:
        """Create an alert for latency issues."""
        return Alert(
            name="latency_degradation",
            severity=AlertSeverity.WARNING,
            message=f"Latency degraded: p99={current_p99:.1f}ms (target: {threshold:.1f}ms)",
            timestamp=datetime.now(),
            details={
                "current_p99_ms": current_p99,
                "threshold_ms": threshold,
            }
        )
    
    def get_recent_alerts(
        self,
        n: int = 10,
        severity: Optional[AlertSeverity] = None,
    ) -> list[Alert]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = self._alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts[-n:]
    
    def clear_history(self) -> None:
        """Clear alert history."""
        self._alert_history = []


# Singleton instance for convenience
_default_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the default alert manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager()
    return _default_manager


def send_alert(alert: Alert) -> bool:
    """Send an alert using the default manager."""
    return get_alert_manager().send(alert)
