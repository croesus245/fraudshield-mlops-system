"""
Model training module.

The brutal truth about model training:
1. The model itself is the easy part
2. The hard parts are: data quality, evaluation, monitoring
3. A simpler model you understand beats a complex one you don't
"""

from typing import Any, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import xgboost as xgb
import joblib
from loguru import logger


class ModelTrainer:
    """
    Train fraud detection models.
    
    This trainer emphasizes:
    - Reproducibility (random seeds, logged params)
    - Validation rigor (stratified splits, proper metrics)
    - Transparency (no black boxes)
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        model_params: Optional[dict] = None,
        threshold: float = 0.5,
        random_state: int = 42,
    ):
        """
        Args:
            model_type: Type of model ("xgboost", "lightgbm", "logistic")
            model_params: Parameters for the model
            threshold: Classification threshold
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.threshold = threshold
        self.random_state = random_state
        self.model_params = model_params or self._default_params()
        
        self.model = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._training_metrics: dict[str, float] = {}
        self._training_timestamp: Optional[datetime] = None
    
    def _default_params(self) -> dict:
        """Default XGBoost parameters tuned for fraud detection."""
        return {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 10,  # Handle class imbalance
            "eval_metric": "aucpr",
            "early_stopping_rounds": 20,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ModelTrainer":
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Labels (0=legitimate, 1=fraud)
            eval_set: Optional validation set for early stopping
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        logger.info(f"Training {self.model_type} model on {len(X)} samples")
        
        # Handle both Series and ndarray
        if hasattr(y, 'value_counts'):
            logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        else:
            unique, counts = np.unique(y, return_counts=True)
            logger.info(f"Class distribution: {dict(zip(unique, counts))}")
        
        self._feature_names = list(X.columns)
        self._training_timestamp = datetime.now()
        
        # Create model
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Prepare eval set
        fit_params = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        
        # Train
        self.model.fit(X, y, **fit_params)
        
        self._is_fitted = True
        
        # Compute training metrics
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        self._training_metrics = {
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_pred_proba),
            "pr_auc": average_precision_score(y, y_pred_proba),
            "brier_score": brier_score_loss(y, y_pred_proba),
        }
        
        logger.info(f"Training complete. Metrics: {self._training_metrics}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud labels (0/1)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path.with_suffix(".model")
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "threshold": self.threshold,
            "random_state": self.random_state,
            "feature_names": self._feature_names,
            "training_metrics": self._training_metrics,
            "training_timestamp": self._training_timestamp.isoformat() if self._training_timestamp else None,
        }
        
        metadata_path = path.with_suffix(".meta")
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelTrainer":
        """Load model and metadata from disk."""
        path = Path(path)
        
        # Load metadata
        metadata_path = path.with_suffix(".meta")
        metadata = joblib.load(metadata_path)
        
        # Create trainer
        trainer = cls(
            model_type=metadata["model_type"],
            model_params=metadata["model_params"],
            threshold=metadata["threshold"],
            random_state=metadata["random_state"],
        )
        
        # Load model
        model_path = path.with_suffix(".model")
        trainer.model = joblib.load(model_path)
        trainer._is_fitted = True
        trainer._feature_names = metadata["feature_names"]
        trainer._training_metrics = metadata["training_metrics"]
        
        if metadata["training_timestamp"]:
            trainer._training_timestamp = datetime.fromisoformat(metadata["training_timestamp"])
        
        logger.info(f"Model loaded from {path}")
        return trainer


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    model_params: Optional[dict] = None,
    threshold: float = 0.5,
    random_state: int = 42,
) -> tuple[ModelTrainer, dict[str, float]]:
    """
    Convenience function to train a model with train/val split.
    
    Args:
        X: Features
        y: Labels
        test_size: Fraction for validation
        model_params: Model parameters
        threshold: Classification threshold
        random_state: Random seed
        
    Returns:
        Tuple of (trained model, validation metrics)
    """
    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Train
    trainer = ModelTrainer(
        model_params=model_params,
        threshold=threshold,
        random_state=random_state,
    )
    trainer.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    # Evaluate on validation
    y_pred = trainer.predict(X_val)
    y_pred_proba = trainer.predict_proba(X_val)
    
    val_metrics = {
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "pr_auc": average_precision_score(y_val, y_pred_proba),
        "brier_score": brier_score_loss(y_val, y_pred_proba),
    }
    
    logger.info(f"Validation metrics: {val_metrics}")
    return trainer, val_metrics
