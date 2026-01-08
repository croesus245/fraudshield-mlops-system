"""
Integration tests.

Test that components work together correctly.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestTrainingPipeline:
    """Test the full training pipeline."""
    
    def test_end_to_end_training(self, sample_transactions):
        """Test training from data to predictions."""
        from src.features.pipeline import FeaturePipeline
        from src.models.trainer import ModelTrainer
        
        # Prepare data
        df = sample_transactions.copy()
        
        # Split
        train_df = df.iloc[:80]
        test_df = df.iloc[80:]
        
        # Feature engineering
        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(train_df)
        y_train = train_df["is_fraud"].values
        
        X_test = pipeline.transform(test_df)
        y_test = test_df["is_fraud"].values
        
        # Train
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        
        # Predict
        predictions = trainer.predict_proba(X_test)
        
        # Verify
        assert len(predictions) == len(y_test)
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_model_save_and_load(self, sample_transactions, tmp_path):
        """Test model persistence."""
        from src.features.pipeline import FeaturePipeline
        from src.models.trainer import ModelTrainer
        
        # Train
        df = sample_transactions.copy()
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(df)
        y = df["is_fraud"].values
        
        trainer = ModelTrainer()
        trainer.train(X, y)
        
        # Save
        model_path = tmp_path / "model.json"
        pipeline_path = tmp_path / "pipeline.pkl"
        
        trainer.save(model_path)
        pipeline.save(pipeline_path)
        
        # Load
        new_trainer = ModelTrainer()
        new_trainer.load(model_path)
        
        new_pipeline = FeaturePipeline.load(pipeline_path)
        
        # Predict with loaded model
        X_new = new_pipeline.transform(df)
        predictions = new_trainer.predict_proba(X_new)
        
        assert len(predictions) == len(y)


class TestEvaluationPipeline:
    """Test evaluation components together."""
    
    def test_full_evaluation(self, sample_transactions):
        """Test full evaluation flow."""
        from src.features.pipeline import FeaturePipeline
        from src.models.trainer import ModelTrainer
        from src.evaluation.report import generate_report
        
        # Prepare
        df = sample_transactions.copy()
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(df)
        y = df["is_fraud"].values
        
        trainer = ModelTrainer()
        trainer.train(X, y)
        
        y_pred = trainer.predict_proba(X)
        
        # Generate report
        report = generate_report(y, y_pred, df, slice_columns=["merchant_category"])
        
        # Verify report
        assert report.overall_metrics is not None
        assert "roc_auc" in report.overall_metrics
        assert report.slice_metrics is not None


class TestMonitoringIntegration:
    """Test monitoring with real data flow."""
    
    def test_drift_detection_in_pipeline(self, sample_transactions):
        """Test drift detection on features."""
        from src.features.pipeline import FeaturePipeline
        from src.monitoring.drift import DriftDetector
        
        # Create reference data
        ref_df = sample_transactions.copy()
        
        # Create drifted data (shift amount)
        drifted_df = sample_transactions.copy()
        drifted_df["amount"] = drifted_df["amount"] * 10  # Big shift
        
        # Transform both
        pipeline = FeaturePipeline()
        X_ref = pipeline.fit_transform(ref_df)
        X_drifted = pipeline.transform(drifted_df)
        
        # Detect drift
        detector = DriftDetector(psi_warning=0.1, psi_critical=0.2)
        detector.set_reference(X_ref)
        results = detector.detect(X_drifted)
        
        # Should detect drift in amount-related features
        amount_drifts = [r for r in results if "amount" in r.feature.lower() and r.is_drifted]
        assert len(amount_drifts) > 0
