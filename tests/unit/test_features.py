"""
Unit tests for feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.transformers import (
    TimeFeatures,
    AmountFeatures,
    CategoricalEncoder,
)
from src.features.pipeline import FeaturePipeline


class TestTimeFeatures:
    """Tests for time feature extraction."""
    
    def test_extracts_hour(self, sample_transactions):
        """Test hour extraction."""
        transformer = TimeFeatures()
        result = transformer.fit_transform(sample_transactions)
        
        assert "hour" in result.columns
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23
    
    def test_extracts_day_of_week(self, sample_transactions):
        """Test day of week extraction."""
        transformer = TimeFeatures()
        result = transformer.fit_transform(sample_transactions)
        
        assert "day_of_week" in result.columns
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6
    
    def test_extracts_is_weekend(self, sample_transactions):
        """Test weekend flag."""
        transformer = TimeFeatures()
        result = transformer.fit_transform(sample_transactions)
        
        assert "is_weekend" in result.columns
        assert set(result["is_weekend"].unique()).issubset({0, 1})


class TestAmountFeatures:
    """Tests for amount feature extraction."""
    
    def test_log_transform(self, sample_transactions):
        """Test log transformation."""
        transformer = AmountFeatures()
        result = transformer.fit_transform(sample_transactions)
        
        assert "amount_log" in result.columns
        # Log of positive numbers should be finite
        assert np.isfinite(result["amount_log"]).all()
    
    def test_zscore_transform(self, sample_transactions):
        """Test z-score normalization."""
        transformer = AmountFeatures()
        result = transformer.fit_transform(sample_transactions)
        
        assert "amount_zscore" in result.columns
        # Z-scores should have roughly mean=0, std=1
        assert abs(result["amount_zscore"].mean()) < 0.1
    
    def test_handles_zero_amount(self):
        """Test handling of zero amounts."""
        df = pd.DataFrame({"amount": [0, 10, 100]})
        transformer = AmountFeatures()
        result = transformer.fit_transform(df)
        
        # Should not have -inf or nan
        assert np.isfinite(result["amount_log"]).all()


class TestCategoricalEncoder:
    """Tests for categorical encoding."""
    
    def test_one_hot_encoding(self, sample_transactions):
        """Test one-hot encoding."""
        encoder = CategoricalEncoder(columns=["merchant_category"])
        result = encoder.fit_transform(sample_transactions)
        
        # Should have one-hot columns
        category_cols = [c for c in result.columns if c.startswith("merchant_category_")]
        assert len(category_cols) > 0
    
    def test_handles_unseen_categories(self, sample_transactions):
        """Test handling of unseen categories."""
        encoder = CategoricalEncoder(columns=["merchant_category"])
        encoder.fit(sample_transactions)
        
        # Create data with new category
        new_df = sample_transactions.copy()
        new_df.loc[0, "merchant_category"] = "new_category"
        
        # Should not raise
        result = encoder.transform(new_df)
        assert result is not None


class TestFeaturePipeline:
    """Tests for the full feature pipeline."""
    
    def test_fit_transform(self, sample_transactions):
        """Test full pipeline fit_transform."""
        pipeline = FeaturePipeline()
        result = pipeline.fit_transform(sample_transactions)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_transactions)
    
    def test_transform_without_fit_raises(self, sample_transactions):
        """Test transform without fit raises error."""
        pipeline = FeaturePipeline()
        
        with pytest.raises(RuntimeError):
            pipeline.transform(sample_transactions)
    
    def test_save_and_load(self, sample_transactions, tmp_path):
        """Test pipeline serialization."""
        pipeline = FeaturePipeline()
        pipeline.fit(sample_transactions)
        
        # Save
        save_path = tmp_path / "pipeline.pkl"
        pipeline.save(save_path)
        
        # Load
        loaded = FeaturePipeline.load(save_path)
        
        # Transform should work
        result = loaded.transform(sample_transactions)
        assert result is not None
