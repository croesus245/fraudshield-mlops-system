"""
Unit tests for data module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.contracts import (
    DataContract,
    Constraint,
    Range,
    NotNull,
    InSet,
    ValidationResult,
    TRANSACTION_CONTRACT,
)
from src.data.schemas import (
    TransactionSchema,
    PredictionRequest,
    PredictionResponse,
)


class TestDataContracts:
    """Tests for data contracts."""
    
    def test_range_constraint_valid(self):
        """Test Range constraint with valid data."""
        constraint = Range(min_value=0, max_value=100)
        data = pd.Series([10, 50, 99])
        assert constraint.validate(data) == []
    
    def test_range_constraint_invalid(self):
        """Test Range constraint with invalid data."""
        constraint = Range(min_value=0, max_value=100)
        data = pd.Series([10, 150, -5])
        errors = constraint.validate(data)
        assert len(errors) > 0
    
    def test_not_null_constraint_valid(self):
        """Test NotNull constraint with valid data."""
        constraint = NotNull()
        data = pd.Series([1, 2, 3])
        assert constraint.validate(data) == []
    
    def test_not_null_constraint_invalid(self):
        """Test NotNull constraint with nulls."""
        constraint = NotNull()
        data = pd.Series([1, None, 3])
        errors = constraint.validate(data)
        assert len(errors) > 0
    
    def test_in_set_constraint_valid(self):
        """Test InSet constraint with valid data."""
        constraint = InSet(values={"a", "b", "c"})
        data = pd.Series(["a", "b", "a"])
        assert constraint.validate(data) == []
    
    def test_in_set_constraint_invalid(self):
        """Test InSet constraint with invalid data."""
        constraint = InSet(values={"a", "b", "c"})
        data = pd.Series(["a", "d", "b"])
        errors = constraint.validate(data)
        assert len(errors) > 0
    
    def test_data_contract_validation(self, sample_transactions):
        """Test full data contract validation."""
        # Create a simple contract
        contract = DataContract(
            name="test_contract",
            constraints={
                "amount": [Range(min_value=0)],
            }
        )
        
        result = contract.validate(sample_transactions)
        assert isinstance(result, ValidationResult)


class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_transaction_schema_valid(self):
        """Test TransactionSchema with valid data."""
        txn = TransactionSchema(
            transaction_id="txn_001",
            user_id="user_001",
            merchant_id="merchant_001",
            merchant_category="retail",
            amount=100.50,
            timestamp=datetime.now(),
            device_type="mobile",
            is_foreign=False,
        )
        assert txn.transaction_id == "txn_001"
        assert txn.amount == 100.50
    
    def test_transaction_schema_invalid_amount(self):
        """Test TransactionSchema rejects negative amount."""
        with pytest.raises(ValueError):
            TransactionSchema(
                transaction_id="txn_001",
                user_id="user_001",
                merchant_id="merchant_001",
                merchant_category="retail",
                amount=-100,  # Invalid
                timestamp=datetime.now(),
            )
    
    def test_prediction_request_valid(self):
        """Test PredictionRequest schema."""
        req = PredictionRequest(
            transaction_id="txn_001",
            user_id="user_001",
            merchant_id="merchant_001",
            merchant_category="retail",
            amount=100.50,
            timestamp=datetime.now(),
        )
        assert req.transaction_id == "txn_001"
    
    def test_prediction_response_valid(self):
        """Test PredictionResponse schema."""
        resp = PredictionResponse(
            transaction_id="txn_001",
            fraud_probability=0.85,
            risk_tier="high",
            model_version="1.0.0",
            latency_ms=15.5,
        )
        assert resp.fraud_probability == 0.85
        assert resp.risk_tier == "high"
