"""
Pytest configuration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_transactions():
    """Generate sample transaction data for testing."""
    n = 100
    return pd.DataFrame({
        "transaction_id": [f"txn_{i:05d}" for i in range(n)],
        "user_id": [f"user_{i % 10:03d}" for i in range(n)],
        "merchant_id": [f"merchant_{i % 5:02d}" for i in range(n)],
        "merchant_category": np.random.choice(["retail", "food", "travel"], n),
        "amount": np.random.lognormal(4, 1, n).round(2),
        "timestamp": [datetime.now() - timedelta(hours=i) for i in range(n)],
        "device_type": np.random.choice(["mobile", "desktop"], n),
        "is_foreign": np.random.choice([True, False], n, p=[0.1, 0.9]),
        "is_fraud": np.random.choice([True, False], n, p=[0.05, 0.95]),
    })


@pytest.fixture
def sample_features():
    """Generate sample feature matrix."""
    n = 100
    return pd.DataFrame({
        "amount": np.random.lognormal(4, 1, n),
        "hour": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "is_weekend": np.random.choice([0, 1], n),
        "amount_log": np.random.normal(4, 1, n),
        "amount_zscore": np.random.normal(0, 1, n),
    })


@pytest.fixture
def sample_labels():
    """Generate sample labels."""
    n = 100
    return np.random.choice([0, 1], n, p=[0.95, 0.05])


@pytest.fixture
def sample_predictions():
    """Generate sample prediction probabilities."""
    n = 100
    return np.random.beta(1, 10, n)  # Skewed towards low (few frauds)
