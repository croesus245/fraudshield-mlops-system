"""
Data contracts and validation.

The brutal truth: most ML pipelines fail silently because data changes without warning.
This module enforces contracts that make failures loud and fast.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger


class ValidationSeverity(Enum):
    """How bad is the violation?"""
    WARNING = "warning"  # Log and continue
    ERROR = "error"      # Fail the pipeline


class Constraint:
    """Base class for column constraints."""
    name: str = ""
    message: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    
    def check(self, series: pd.Series) -> bool:
        raise NotImplementedError


class Range(Constraint):
    """Value must be within range."""
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val
        self.name = f"range_{min_val}_{max_val}"
        self.message = f"Values must be in range [{min_val}, {max_val}]"
    
    def check(self, series: pd.Series) -> bool:
        if self.min_val is not None and series.min() < self.min_val:
            return False
        if self.max_val is not None and series.max() > self.max_val:
            return False
        return True


class NotNull(Constraint):
    """Column must not have null values."""
    def __init__(self, max_null_rate: float = 0.0):
        self.max_null_rate = max_null_rate
        self.name = "not_null"
        self.message = f"Null rate must be <= {max_null_rate}"
    
    def check(self, series: pd.Series) -> bool:
        null_rate = series.isnull().mean()
        return null_rate <= self.max_null_rate


class InSet(Constraint):
    """Value must be in allowed set."""
    def __init__(self, allowed_values: set = None):
        self.allowed_values = allowed_values or set()
        self.name = "in_set"
        self.message = f"Values must be in {self.allowed_values}"
    
    def check(self, series: pd.Series) -> bool:
        unique_values = set(series.dropna().unique())
        return unique_values.issubset(self.allowed_values)


class Unique(Constraint):
    """Column must have unique values."""
    def __init__(self):
        self.name = "unique"
        self.message = "Values must be unique"
    
    def check(self, series: pd.Series) -> bool:
        return series.nunique() == len(series)


@dataclass
class AnomalyCheck:
    """Statistical anomaly detection on a column."""
    name: str
    compute: Callable[[pd.Series], float]
    threshold: float
    comparison: str = "max"  # "max" or "min"
    severity: ValidationSeverity = ValidationSeverity.WARNING
    
    def check(self, series: pd.Series) -> tuple[bool, float]:
        """Returns (passed, actual_value)."""
        value = self.compute(series)
        if self.comparison == "max":
            passed = value <= self.threshold
        else:
            passed = value >= self.threshold
        return passed, value


@dataclass
class ValidationResult:
    """Result of validating a dataframe against a contract."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "✓ VALID" if self.valid else "✗ INVALID"
        lines = [f"Validation: {status}"]
        
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")
        
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        
        return "\n".join(lines)


@dataclass
class DataContract:
    """
    A contract defining expectations for a dataframe.
    
    If data violates the contract, the pipeline fails fast.
    No silent corruption.
    
    Example:
        contract = DataContract(
            name="transactions",
            required_columns=["amount", "merchant_id", "timestamp"],
            column_types={"amount": float, "merchant_id": str},
            constraints={
                "amount": [Range(min_val=0, max_val=100000)],
                "timestamp": [NotNull()],
            },
            anomaly_checks=[
                AnomalyCheck(
                    name="null_rate",
                    compute=lambda s: s.isnull().mean(),
                    threshold=0.05,
                ),
            ]
        )
    """
    name: str
    required_columns: list[str] = field(default_factory=list)
    column_types: dict[str, type] = field(default_factory=dict)
    constraints: dict[str, list[Constraint]] = field(default_factory=dict)
    anomaly_checks: list[AnomalyCheck] = field(default_factory=list)
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a dataframe against this contract.
        
        Returns ValidationResult with errors/warnings.
        """
        errors = []
        warnings = []
        stats = {"rows": len(df), "columns": len(df.columns)}
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check column types
        for col, expected_type in self.column_types.items():
            if col not in df.columns:
                continue
            
            actual_dtype = df[col].dtype
            if expected_type == float and not np.issubdtype(actual_dtype, np.floating):
                # Allow int as float
                if not np.issubdtype(actual_dtype, np.integer):
                    errors.append(f"Column '{col}' expected {expected_type}, got {actual_dtype}")
            elif expected_type == int and not np.issubdtype(actual_dtype, np.integer):
                errors.append(f"Column '{col}' expected {expected_type}, got {actual_dtype}")
            elif expected_type == str and actual_dtype != object:
                errors.append(f"Column '{col}' expected {expected_type}, got {actual_dtype}")
        
        # Check constraints
        for col, col_constraints in self.constraints.items():
            if col not in df.columns:
                continue
            
            for constraint in col_constraints:
                try:
                    passed = constraint.check(df[col])
                    if not passed:
                        msg = f"Column '{col}' failed constraint '{constraint.name}': {constraint.message}"
                        if constraint.severity == ValidationSeverity.ERROR:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                except Exception as e:
                    errors.append(f"Column '{col}' constraint '{constraint.name}' raised error: {e}")
        
        # Run anomaly checks
        for check in self.anomaly_checks:
            try:
                # Apply to entire dataframe or specific columns
                for col in df.columns:
                    if df[col].dtype in [np.float64, np.int64, object]:
                        passed, value = check.check(df[col])
                        stats[f"{col}_{check.name}"] = value
                        if not passed:
                            msg = f"Anomaly detected in '{col}': {check.name}={value:.4f} (threshold={check.threshold})"
                            if check.severity == ValidationSeverity.ERROR:
                                errors.append(msg)
                            else:
                                warnings.append(msg)
            except Exception as e:
                warnings.append(f"Anomaly check '{check.name}' raised error: {e}")
        
        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, stats=stats)


def validate_dataframe(
    df: pd.DataFrame,
    contract: DataContract,
    fail_on_error: bool = True
) -> ValidationResult:
    """
    Validate a dataframe against a contract.
    
    Args:
        df: DataFrame to validate
        contract: Contract to validate against
        fail_on_error: If True, raise exception on validation errors
        
    Returns:
        ValidationResult
        
    Raises:
        ValueError: If validation fails and fail_on_error=True
    """
    result = contract.validate(df)
    
    if result.warnings:
        for warn in result.warnings:
            logger.warning(warn)
    
    if not result.valid:
        logger.error(f"Data validation failed for contract '{contract.name}'")
        for err in result.errors:
            logger.error(err)
        
        if fail_on_error:
            raise ValueError(f"Data contract '{contract.name}' violated: {result.errors}")
    else:
        logger.info(f"Data validation passed for contract '{contract.name}' ({result.stats['rows']} rows)")
    
    return result


# Pre-built contracts for common data types
TRANSACTION_CONTRACT = DataContract(
    name="transactions",
    required_columns=[
        "transaction_id",
        "timestamp",
        "amount",
        "merchant_id",
        "merchant_category",
        "user_id",
    ],
    column_types={
        "amount": float,
        "merchant_id": str,
        "user_id": str,
    },
    constraints={
        "transaction_id": [Unique()],
        "amount": [Range(min_val=0, max_val=100000)],
        "timestamp": [NotNull()],
    },
    anomaly_checks=[
        AnomalyCheck(
            name="null_rate",
            compute=lambda s: s.isnull().mean(),
            threshold=0.05,
            severity=ValidationSeverity.WARNING,
        ),
    ]
)

LABEL_CONTRACT = DataContract(
    name="labels",
    required_columns=["transaction_id", "is_fraud", "label_timestamp"],
    column_types={
        "is_fraud": int,
    },
    constraints={
        "transaction_id": [NotNull()],
        "is_fraud": [InSet(allowed_values={0, 1})],
    },
)
