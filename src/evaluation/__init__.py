"""
Evaluation module.

This is the heart of production ML.
Not just "accuracy" but disaggregated, stress-tested evaluation.
"""

from .metrics import (
    compute_metrics,
    compute_slice_metrics,
    compute_calibration_metrics,
)
from .report import EvaluationReport, generate_report
from .stress_tests import StressTest, run_stress_tests

__all__ = [
    "compute_metrics",
    "compute_slice_metrics",
    "compute_calibration_metrics",
    "EvaluationReport",
    "generate_report",
    "StressTest",
    "run_stress_tests",
]
