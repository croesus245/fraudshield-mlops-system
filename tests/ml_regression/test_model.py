"""
ML regression tests.

These tests catch model performance regressions.
Run them in CI to prevent shipping broken models.
"""

import pytest
import pandas as pd
import numpy as np


class TestModelRegression:
    """Tests to prevent model performance regression."""
    
    @pytest.fixture
    def trained_model(self, sample_transactions):
        """Train a model for testing."""
        from src.features.pipeline import FeaturePipeline
        from src.models.trainer import ModelTrainer
        
        df = sample_transactions.copy()
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(df)
        y = df["is_fraud"].values
        
        trainer = ModelTrainer()
        trainer.train(X, y)
        
        return trainer, pipeline, X, y
    
    def test_roc_auc_above_random(self, trained_model):
        """Test ROC-AUC is better than random."""
        from src.evaluation.metrics import compute_metrics
        
        trainer, pipeline, X, y = trained_model
        y_pred = trainer.predict_proba(X)
        
        metrics = compute_metrics(y, y_pred)
        
        # Should be better than random (0.5)
        assert metrics["roc_auc"] > 0.5
    
    def test_predictions_in_valid_range(self, trained_model):
        """Test all predictions are valid probabilities."""
        trainer, pipeline, X, y = trained_model
        predictions = trainer.predict_proba(X)
        
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_model_not_predicting_constant(self, trained_model):
        """Test model doesn't predict constant value."""
        trainer, pipeline, X, y = trained_model
        predictions = trainer.predict_proba(X)
        
        # Should have some variance
        assert np.std(predictions) > 0.01
    
    def test_feature_importance_exists(self, trained_model):
        """Test feature importance is computed."""
        trainer, pipeline, X, y = trained_model
        
        importance = trainer.feature_importance()
        
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())


class TestDataRegression:
    """Tests to catch data quality issues."""
    
    def test_features_no_nan_after_transform(self, sample_transactions):
        """Test no NaN in features after transformation."""
        from src.features.pipeline import FeaturePipeline
        
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(sample_transactions)
        
        # Check for NaN
        nan_cols = X.columns[X.isna().any()].tolist()
        assert len(nan_cols) == 0, f"NaN in columns: {nan_cols}"
    
    def test_features_no_inf_after_transform(self, sample_transactions):
        """Test no infinity in features after transformation."""
        from src.features.pipeline import FeaturePipeline
        
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(sample_transactions)
        
        # Check for inf
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert np.isfinite(X[col]).all(), f"Inf in column: {col}"


class TestStressRegression:
    """Stress tests for robustness."""
    
    def test_handles_extreme_amounts(self, sample_transactions):
        """Test handling of extreme transaction amounts."""
        from src.features.pipeline import FeaturePipeline
        from src.models.trainer import ModelTrainer
        
        # Train on normal data
        df = sample_transactions.copy()
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(df)
        y = df["is_fraud"].values
        
        trainer = ModelTrainer()
        trainer.train(X, y)
        
        # Create extreme data
        extreme_df = df.copy()
        extreme_df["amount"] = 1000000  # Very high amount
        
        X_extreme = pipeline.transform(extreme_df)
        
        # Should still produce valid predictions
        predictions = trainer.predict_proba(X_extreme)
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_handles_single_sample(self, sample_transactions):
        """Test prediction on single sample."""
        from src.features.pipeline import FeaturePipeline
        from src.models.trainer import ModelTrainer
        
        # Train
        df = sample_transactions.copy()
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(df)
        y = df["is_fraud"].values
        
        trainer = ModelTrainer()
        trainer.train(X, y)
        
        # Predict on single sample
        single_df = df.iloc[[0]]
        X_single = pipeline.transform(single_df)
        
        predictions = trainer.predict_proba(X_single)
        assert len(predictions) == 1
        assert 0 <= predictions[0] <= 1
