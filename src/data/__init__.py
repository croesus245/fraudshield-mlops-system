"""
Data loading, validation, and contracts.

This module handles:
- Loading raw transaction data
- Validating against data contracts
- Schema enforcement
- Anomaly detection on data quality
"""

from .contracts import DataContract, ValidationResult, validate_dataframe
from .loader import DataLoader, load_transactions, load_labels
from .schemas import (
    TransactionSchema, 
    LabelSchema,
    TransactionType,
    VerificationLevel,
    RiskTier,
    ActionType,
    ChallengeType,
    UserSignals,
    DeviceSignals,
    NetworkSignals,
    VelocitySignals,
    FraudCheckRequest,
    FraudCheckResponse,
    RiskReason,
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
)

__all__ = [
    # Contracts & Validation
    "DataContract",
    "ValidationResult", 
    "validate_dataframe",
    # Data Loading
    "DataLoader",
    "load_transactions",
    "load_labels",
    # Schemas
    "TransactionSchema",
    "LabelSchema",
    # Enums
    "TransactionType",
    "VerificationLevel",
    "RiskTier",
    "ActionType",
    "ChallengeType",
    # Signal Types
    "UserSignals",
    "DeviceSignals",
    "NetworkSignals",
    "VelocitySignals",
    # API Schemas
    "FraudCheckRequest",
    "FraudCheckResponse",
    "RiskReason",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
]
