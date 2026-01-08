"""
Feature Pipeline - Transform raw transactions into ML features.

HOW IT WORKS:
1. fit() on training data → learns statistics (mean, percentiles, categories)
2. transform() on any data → applies same transformations
3. fit_transform() → does both in one call

WHY A PIPELINE?
- Consistent transformations train→production
- Saves learned statistics (what's the 95th percentile?)
- Avoids data leakage (don't learn from test data)
"""

from pathlib import Path
import pandas as pd
import joblib
from loguru import logger

from .transformers import (
    TimeFeatures, AmountFeatures, CategoricalEncoder,
    DeviceFeatures, UserFeatures, VelocityFeatures, NetworkFeatures
)


class FeaturePipeline:
    """
    End-to-end feature engineering.
    
    Turns raw transaction data into model-ready features.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Transformers
        self.time = TimeFeatures()
        self.amount = AmountFeatures()
        self.categorical = CategoricalEncoder()
        self.device = DeviceFeatures()
        self.user = UserFeatures()
        self.velocity = VelocityFeatures()
        self.network = NetworkFeatures()
        
        self._fitted = False
        self._feature_names = []
    
    def fit(self, df: pd.DataFrame):
        """Learn statistics from training data."""
        logger.info(f"Fitting on {len(df)} samples")
        
        # Fit transformers that need it
        if "amount" in df.columns:
            self.amount.fit(df["amount"])
        
        cat_cols = [c for c in ["merchant_category", "device_type"] if c in df.columns]
        if cat_cols:
            self.categorical.fit(df[cat_cols])
        
        self._fitted = True
        
        # Learn feature names
        sample = self._transform(df.head(10))
        self._feature_names = list(sample.columns)
        logger.info(f"Fitted. {len(self._feature_names)} features.")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data into features."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return self._transform(df)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(df)
        return self.transform(df)
    
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal transform - runs all transformers."""
        result = pd.DataFrame(index=df.index)
        
        # Keep transaction ID for joining later
        if "transaction_id" in df.columns:
            result["transaction_id"] = df["transaction_id"]
        
        # Time features (hour, weekend, night)
        if "timestamp" in df.columns:
            result = pd.concat([result, self.time.transform(df["timestamp"])], axis=1)
        
        # Amount features (log, zscore)
        if "amount" in df.columns:
            result = pd.concat([result, self.amount.transform(df["amount"])], axis=1)
        
        # Categorical features (one-hot)
        cat_cols = [c for c in ["merchant_category", "device_type"] if c in df.columns]
        if cat_cols:
            result = pd.concat([result, self.categorical.transform(df[cat_cols])], axis=1)
        
        # Device risk features
        result = pd.concat([result, self.device.transform(df)], axis=1)
        
        # User risk features
        result = pd.concat([result, self.user.transform(df)], axis=1)
        
        # Velocity features
        result = pd.concat([result, self.velocity.transform(df)], axis=1)
        
        # Network features
        result = pd.concat([result, self.network.transform(df)], axis=1)
        
        # Fill NaN (new users, missing data)
        return result.fillna(0)
    
    def save(self, path: str):
        """Save fitted pipeline."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "amount": self.amount,
            "categorical": self.categorical,
            "fitted": self._fitted,
            "feature_names": self._feature_names,
        }, path)
        logger.info(f"Saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Load fitted pipeline."""
        state = joblib.load(path)
        pipeline = cls()
        pipeline.amount = state["amount"]
        pipeline.categorical = state["categorical"]
        pipeline._fitted = state["fitted"]
        pipeline._feature_names = state["feature_names"]
        logger.info(f"Loaded from {path}")
        return pipeline
    
    @property
    def feature_names(self):
        return self._feature_names
