"""
Data loading utilities.

Handles loading transactions, labels, and joining them
with proper handling of delayed labels.
"""

from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from .contracts import (
    DataContract,
    TRANSACTION_CONTRACT,
    LABEL_CONTRACT,
    validate_dataframe,
)


class DataLoader:
    """
    Loads and manages transaction and label data.
    
    Key responsibility: Handle the reality that labels arrive
    30-90 days after predictions.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        transaction_contract: Optional[DataContract] = None,
        label_contract: Optional[DataContract] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transaction_contract = transaction_contract or TRANSACTION_CONTRACT
        self.label_contract = label_contract or LABEL_CONTRACT
        
        # Create directories if needed
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    def load_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load transaction data, optionally filtered by date range.
        
        Args:
            start_date: Include transactions on or after this date
            end_date: Include transactions before this date
            validate: Whether to validate against contract
            
        Returns:
            DataFrame with transactions
        """
        # Load from parquet files (one per day/partition)
        raw_dir = self.data_dir / "raw"
        parquet_files = list(raw_dir.glob("transactions_*.parquet"))
        
        if not parquet_files:
            # Check for CSV fallback
            csv_files = list(raw_dir.glob("transactions*.csv"))
            if csv_files:
                dfs = [pd.read_csv(f, parse_dates=["timestamp"]) for f in csv_files]
            else:
                logger.warning(f"No transaction files found in {raw_dir}")
                return pd.DataFrame()
        else:
            dfs = [pd.read_parquet(f) for f in parquet_files]
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by date range
        if start_date and "timestamp" in df.columns:
            df = df[df["timestamp"] >= start_date]
        if end_date and "timestamp" in df.columns:
            df = df[df["timestamp"] < end_date]
        
        # Validate
        if validate and len(df) > 0:
            validate_dataframe(df, self.transaction_contract, fail_on_error=True)
        
        logger.info(f"Loaded {len(df)} transactions")
        return df
    
    def load_labels(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load label data (fraud outcomes).
        
        Labels arrive with delay, so this might have fewer records
        than transactions for recent dates.
        """
        labels_dir = self.data_dir / "labels"
        parquet_files = list(labels_dir.glob("labels_*.parquet"))
        
        if not parquet_files:
            csv_files = list(labels_dir.glob("labels*.csv"))
            if csv_files:
                dfs = [pd.read_csv(f, parse_dates=["label_timestamp"]) for f in csv_files]
            else:
                logger.warning(f"No label files found in {labels_dir}")
                return pd.DataFrame()
        else:
            dfs = [pd.read_parquet(f) for f in parquet_files]
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure timestamp is datetime
        if "label_timestamp" in df.columns:
            df["label_timestamp"] = pd.to_datetime(df["label_timestamp"])
        
        # Filter by date range (using label_timestamp)
        if start_date and "label_timestamp" in df.columns:
            df = df[df["label_timestamp"] >= start_date]
        if end_date and "label_timestamp" in df.columns:
            df = df[df["label_timestamp"] < end_date]
        
        # Validate
        if validate and len(df) > 0:
            validate_dataframe(df, self.label_contract, fail_on_error=True)
        
        logger.info(f"Loaded {len(df)} labels")
        return df
    
    def load_predictions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load logged predictions."""
        pred_dir = self.data_dir / "predictions"
        parquet_files = list(pred_dir.glob("predictions_*.parquet"))
        
        if not parquet_files:
            csv_files = list(pred_dir.glob("predictions*.csv"))
            if csv_files:
                dfs = [pd.read_csv(f, parse_dates=["prediction_timestamp"]) for f in csv_files]
            else:
                logger.warning(f"No prediction files found in {pred_dir}")
                return pd.DataFrame()
        else:
            dfs = [pd.read_parquet(f) for f in parquet_files]
        
        df = pd.concat(dfs, ignore_index=True)
        
        if "prediction_timestamp" in df.columns:
            df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
        
        # Filter by date range
        if start_date and "prediction_timestamp" in df.columns:
            df = df[df["prediction_timestamp"] >= start_date]
        if end_date and "prediction_timestamp" in df.columns:
            df = df[df["prediction_timestamp"] < end_date]
        
        logger.info(f"Loaded {len(df)} predictions")
        return df
    
    def join_predictions_to_labels(
        self,
        predictions: pd.DataFrame,
        labels: pd.DataFrame,
        max_label_delay_days: int = 120,
    ) -> pd.DataFrame:
        """
        Join predictions to their (delayed) labels.
        
        This is the core of operating with delayed feedback:
        - Predictions happen at time T
        - Labels arrive at time T + delay (30-90 days typically)
        - We need to join them to evaluate model performance
        
        Args:
            predictions: DataFrame with prediction_timestamp, transaction_id, predicted_score
            labels: DataFrame with transaction_id, is_fraud, label_timestamp
            max_label_delay_days: Ignore labels that arrive after this many days
            
        Returns:
            Joined DataFrame with both predictions and labels
        """
        # Validate inputs
        if predictions.empty or labels.empty:
            logger.warning("Empty predictions or labels, returning empty DataFrame")
            return pd.DataFrame()
        
        # Merge on transaction_id
        merged = predictions.merge(
            labels[["transaction_id", "is_fraud", "label_timestamp"]],
            on="transaction_id",
            how="left",
        )
        
        # Calculate label delay
        if "prediction_timestamp" in merged.columns and "label_timestamp" in merged.columns:
            merged["label_delay_days"] = (
                merged["label_timestamp"] - merged["prediction_timestamp"]
            ).dt.days
            
            # Flag predictions that are still waiting for labels
            merged["has_label"] = merged["is_fraud"].notna()
            
            # Flag labels that took too long (might be stale)
            merged["label_too_late"] = merged["label_delay_days"] > max_label_delay_days
        
        # Stats
        n_with_labels = merged["has_label"].sum() if "has_label" in merged.columns else 0
        n_total = len(merged)
        label_rate = n_with_labels / n_total if n_total > 0 else 0
        
        logger.info(f"Joined {n_with_labels}/{n_total} predictions to labels ({label_rate:.1%})")
        
        return merged
    
    def get_evaluation_window(
        self,
        window_end: datetime,
        window_days: int = 30,
        min_label_rate: float = 0.8,
    ) -> pd.DataFrame:
        """
        Get data for evaluation over a time window.
        
        Only includes predictions that have had enough time for labels to arrive.
        
        Args:
            window_end: End of evaluation window
            window_days: Number of days in window
            min_label_rate: Minimum fraction of predictions with labels required
            
        Returns:
            DataFrame ready for evaluation
        """
        window_start = window_end - timedelta(days=window_days)
        
        # Load predictions from the window
        predictions = self.load_predictions(
            start_date=window_start,
            end_date=window_end,
        )
        
        # Load all labels (they might be from after the window)
        labels = self.load_labels()
        
        # Join
        joined = self.join_predictions_to_labels(predictions, labels)
        
        if joined.empty:
            logger.warning("No data available for evaluation window")
            return pd.DataFrame()
        
        # Check label rate
        label_rate = joined["has_label"].mean() if "has_label" in joined.columns else 0
        if label_rate < min_label_rate:
            logger.warning(
                f"Label rate {label_rate:.1%} below minimum {min_label_rate:.1%}. "
                f"Evaluation may be unreliable."
            )
        
        return joined


# Convenience functions
def load_transactions(
    data_dir: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """Load transactions from a data directory."""
    loader = DataLoader(data_dir)
    return loader.load_transactions(**kwargs)


def load_labels(
    data_dir: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """Load labels from a data directory."""
    loader = DataLoader(data_dir)
    return loader.load_labels(**kwargs)
