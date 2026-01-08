"""
Monitoring and drift detection.

This is where most ML systems fail silently.
This module catches drift before your users do.
"""

from .drift import DriftDetector, detect_drift
from .alerts import AlertManager, Alert

__all__ = [
    "DriftDetector",
    "detect_drift",
    "AlertManager", 
    "Alert",
]
